"""Train reinforcement-learning policies for selective context expansion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import tyro
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from refrag.compress.encoder import ChunkEncoder
from refrag.data.hotpotqa import HotpotQADataset, load_hotpotqa_samples
from refrag.llm.inference_real import (
    RealLLMInference,
    compute_em_f1,
    compute_evidence_metrics,
)
from refrag.rl.bandit import LinUCBPolicy, ThompsonSamplingPolicy
from refrag.rl.features import compute_chunk_features
from refrag.utils.io import (
    get_device,
    load_yaml as load_config,
    save_checkpoint,
)


@dataclass
class Args:
    config: str = "configs/rl_bandit.yaml"
    algo: str = "linucb"  # linucb, thompson


def _build_policy(config: dict, algo: str, feature_dim: int):
    if algo == "linucb":
        return LinUCBPolicy(
            feature_dim=feature_dim,
            alpha=config["rl"]["bandit"]["alpha"],
            lambda_=config["rl"]["bandit"]["lambda_"],
        )
    if algo == "thompson":
        return ThompsonSamplingPolicy(
            feature_dim=feature_dim,
            prior_mean=config["rl"]["thompson"]["prior_mean"],
            prior_var=config["rl"]["thompson"]["prior_variance"],
        )
    raise ValueError(f"Unknown algorithm: {algo}")


def _maybe_load_retriever(config: dict):
    from refrag.retrieval.real_retrieval import (
        RealBM25Retriever,
        RealDenseRetriever,
        RealHybridRetriever,
    )

    try:
        bm25 = RealBM25Retriever(f"{config['data']['path']}/../indexes/bm25_index.pkl")
        dense = RealDenseRetriever(index_path=f"{config['data']['path']}/../indexes/dense_index.pkl")
        print("[RL] Loaded retrieval indexes")
        return RealHybridRetriever(bm25, dense)
    except Exception as exc:
        print(f"[RL] Retrieval indexes unavailable ({exc}); fallback to dataset contexts.")
        return None


def _create_inference_clients(config: dict, config_path: str | Path):
    groq_client = None
    try:
        from refrag.llm.groq_client import create_groq_client

        groq_client = create_groq_client(config_path)
        print("[RL] Using Groq API for rollouts.")
    except Exception as exc:
        print(f"[RL] Groq client unavailable ({exc}); using local inference.")

    local_inference = None
    if groq_client is None:
        local_inference = RealLLMInference(
            model_name=config["llm"]["model_name"],
            device=str(get_device(config.get("device", "auto"))),
            torch_dtype=config["llm"].get("torch_dtype", "float16"),
        )
    return groq_client, local_inference


def _ensure_selection(
    selected: List[int],
    teacher_scores: np.ndarray,
    token_costs: List[int],
    budget: int,
) -> List[int]:
    if selected:
        return selected

    ordered = np.argsort(-teacher_scores)
    tokens = 0
    fallback: List[int] = []
    for idx in ordered:
        chunk_tokens = token_costs[idx]
        if tokens + chunk_tokens <= budget or not fallback:
            fallback.append(int(idx))
            tokens += chunk_tokens
        if tokens >= budget:
            break
    return fallback


def train_policy(config_path: str, algo: str = "linucb") -> None:
    """Train RL policy for chunk selection using offline teacher signals."""
    config = load_config(config_path)

    budget = config["rl"]["token_budget"]
    feature_dim = 8  # compute_chunk_features returns 8-dimensional vectors

    policy = _build_policy(config, algo, feature_dim)

    device = get_device(config.get("device", "auto"))
    print(f"[RL] Device for feature extraction: {device}")

    encoder = ChunkEncoder(
        model_name=config["encoder"]["model_name"],
        device=str(device),
        normalize=config["encoder"].get("normalize", True),
    )
    cross_encoder = CrossEncoder(
        config["retrieval"]["reranker_model"],
        device=str(device),
    )

    groq_client, local_inference = _create_inference_clients(config, config_path)
    hybrid_retriever = _maybe_load_retriever(config)

    num_rollouts = config["rl"]["bandit"]["num_rollouts"]
    update_freq = config["rl"]["bandit"]["update_freq"]
    top_k = config["rl"]["top_k_candidates"]

    print(f"[RL] Training {algo} policy for {num_rollouts} rollouts with budget {budget} tokens")

    eval_samples = load_hotpotqa_samples(
        config["data"]["path"],
        split="dev",
        num_samples=min(200, num_rollouts),
    )

    reward_history: List[float] = []
    total_reward = 0.0

    for rollout_idx in tqdm(range(num_rollouts), desc="RL rollouts"):
        sample = eval_samples[rollout_idx % len(eval_samples)]
        question = sample["question"]
        answer = sample["answer"]
        supporting_facts = sample.get("supporting_facts", [])

        if hybrid_retriever:
            try:
                retrieved = hybrid_retriever.search(question, top_k=top_k)
                context_chunks = [text for text, _, _ in retrieved]
            except Exception:
                context_chunks = HotpotQADataset.get_context_chunks(sample, chunk_size=512)[:top_k]
        else:
            context_chunks = HotpotQADataset.get_context_chunks(sample, chunk_size=512)[:top_k]

        if not context_chunks:
            context_chunks = [f"Context {i+1}: {question}"] * top_k

        features, token_costs, extras = compute_chunk_features(
            question=question,
            chunks=context_chunks,
            answer=answer,
            encoder=encoder,
            cross_encoder=cross_encoder,
            token_budget=budget,
        )

        if features.shape[0] == 0:
            reward_history.append(0.0)
            continue

        selected_indices = policy.select(
            features=features,
            budget=budget,
            token_costs=token_costs,
        )
        selected_indices = _ensure_selection(
            selected_indices,
            teacher_scores=extras["teacher_q"],
            token_costs=token_costs,
            budget=budget,
        )

        expanded_contexts = [context_chunks[i] for i in selected_indices]

        try:
            if groq_client:
                result = groq_client.generate_answer(question, expanded_contexts)
                generated_answer = result["answer"]
                tokens_used = sum(token_costs[i] for i in selected_indices)
                perplexity = groq_client.compute_perplexity(question, expanded_contexts, answer)
            else:
                result = local_inference.generate_answer(question, expanded_contexts)
                generated_answer = result["answer"]
                tokens_used = sum(token_costs[i] for i in selected_indices)
                perplexity = local_inference.compute_perplexity(question, expanded_contexts, answer)
        except Exception as exc:
            print(f"[RL] Inference error at rollout {rollout_idx}: {exc}")
            generated_answer = ""
            tokens_used = sum(token_costs[i] for i in selected_indices)
            perplexity = max(15.0 - extras["teacher_q"][selected_indices].mean() * 5.0, 1.0)

        em, f1 = compute_em_f1(generated_answer, answer)
        prec, recall = compute_evidence_metrics(expanded_contexts, supporting_facts)

        reward = (
            config["rl"]["reward"]["correctness_bonus"] * (em + f1) * 0.5
            + config["rl"]["reward"]["perplexity_weight"] * perplexity
            - config["rl"]["reward"]["token_penalty"] * tokens_used
            + 0.1 * float(extras["teacher_q"][selected_indices].mean())
        )

        policy.update(features, selected_indices, reward)

        total_reward += reward
        reward_history.append(reward)

        if (rollout_idx + 1) % update_freq == 0:
            avg_reward = total_reward / (rollout_idx + 1)
            print(
                f"[RL] Rollout {rollout_idx+1}/{num_rollouts} | "
                f"selected {len(selected_indices)}/{features.shape[0]} chunks | "
                f"reward {reward:.3f} | avg {avg_reward:.3f} | EM {em:.2f} | F1 {f1:.2f}"
            )

    avg_reward = total_reward / max(len(reward_history), 1)
    tail = reward_history[-10:] if len(reward_history) >= 10 else reward_history
    tail_avg = sum(tail) / max(len(tail), 1)

    print(f"[RL] Training complete. Avg reward: {avg_reward:.3f}, last-10 avg: {tail_avg:.3f}")

    policy_state = {
        "A": getattr(policy, "A", None),
        "b": getattr(policy, "b", None),
        "mean": getattr(policy, "mean", None),
        "cov": getattr(policy, "cov", None),
        "feature_dim": feature_dim,
    }
    checkpoint = {
        "policy_state": policy_state,
        "config": config,
        "rollouts": num_rollouts,
        "avg_reward": avg_reward,
        "tail_reward": tail_avg,
    }
    save_checkpoint(checkpoint, f"{config['checkpoint_dir']}/rl_policy_{algo}.pt")
    print(f"[RL] Saved policy checkpoint to {config['checkpoint_dir']}/rl_policy_{algo}.pt")


def main() -> None:
    args = tyro.cli(Args)
    train_policy(args.config, args.algo)


if __name__ == "__main__":
    main()
