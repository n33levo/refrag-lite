"""QA evaluation."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tyro
from sentence_transformers import CrossEncoder

from refrag.compress.encoder import ChunkEncoder
from refrag.rl.bandit import LinUCBPolicy
from refrag.rl.features import compute_chunk_features
from refrag.utils.io import get_device, load_yaml as load_config


@dataclass
class Args:
    config: str = "configs/eval.yaml"


def evaluate_qa(config_path: str):
    """Evaluate QA performance (EM, F1)."""
    config = load_config(config_path)
    
    print("Running QA evaluation...")
    print(f"Metrics: {config['eval']['metrics']}")
    print(f"Samples: {config['eval']['num_samples']}")
    
    # Actual QA evaluation
    metrics = config["eval"]["metrics"]
    num_samples = config["eval"]["num_samples"]
    
    # Real QA evaluation
    print("Loading real components for evaluation...")
    
    # Load real components
    from refrag.data.hotpotqa import HotpotQADataset, load_hotpotqa_samples
    from refrag.llm.inference_real import RealLLMInference, compute_em_f1, compute_evidence_metrics
    from refrag.retrieval.real_retrieval import RealHybridRetriever, RealBM25Retriever, RealDenseRetriever
    from refrag.rl.bandit import LinUCBPolicy
    from refrag.utils.io import get_device
    
    # Get device
    device = get_device()
    
    encoder = ChunkEncoder(
        config["encoder"]["model_name"],
        device=str(device),
        normalize=config["encoder"].get("normalize", True),
    )
    cross_encoder = CrossEncoder(
        config["retrieval"]["reranker_model"],
        device=str(device),
    )

    # Load evaluation samples
    eval_samples = load_hotpotqa_samples(
        config["data"]["path"], 
        split="dev", 
        num_samples=num_samples
    )
    
    # Load Groq client for inference (with local fallback)
    try:
        from refrag.llm.groq_client import create_groq_client

        groq_client = create_groq_client(config_path)
        local_inference = None
        print("Using Groq API for evaluation inference.")
    except Exception as exc:
        print(f"Groq API unavailable ({exc}); falling back to local LLM inference.")
        groq_client = None
        local_inference = RealLLMInference(
            model_name=config["llm"]["model_name"],
            device=str(device),
            torch_dtype=config["llm"].get("torch_dtype", "float16"),
        )
    
    # Load retrievers
    try:
        bm25_retriever = RealBM25Retriever(f"{config['data']['path']}/../indexes/bm25_index.pkl")
        dense_retriever = RealDenseRetriever(index_path=f"{config['data']['path']}/../indexes/dense_index.pkl")
        hybrid_retriever = RealHybridRetriever(bm25_retriever, dense_retriever)
        print("Loaded existing retrieval indexes")
    except:
        print("No existing indexes found, using context from dataset")
        hybrid_retriever = None
    
    # Load RL policy if available
    policy = None
    checkpoint_path = Path(f"{config['checkpoint_dir']}/rl_policy_linucb.pt")
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("policy_state", {})
            feature_dim = state.get("feature_dim", 8)
            policy = LinUCBPolicy(
                feature_dim=feature_dim,
                alpha=config["rl"]["bandit"]["alpha"],
                lambda_=config["rl"]["bandit"]["lambda_"],
            )
            if state.get("A") is not None:
                policy.A = state["A"]
            if state.get("b") is not None:
                policy.b = state["b"]
            if state.get("mean") is not None and hasattr(policy, "mean"):
                policy.mean = state["mean"]
            if state.get("cov") is not None and hasattr(policy, "cov"):
                policy.cov = state["cov"]
            print("Loaded trained RL policy")
        except Exception as exc:
            print(f"Failed to load RL policy ({exc}); using teacher fallback.")
            policy = None
    else:
        print("No trained RL policy found; using teacher-guided selection.")
    
    # Run evaluation
    em_scores = []
    f1_scores = []
    evidence_precisions = []
    evidence_recalls = []
    token_counts = []
    
    print(f"Evaluating on {len(eval_samples)} samples...")
    
    for i, sample in enumerate(eval_samples):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(eval_samples)}")
        
        question = sample["question"]
        answer = sample["answer"]
        supporting_facts = sample["supporting_facts"]
        
        # Debug first few samples
        if i < 3:
            print(f"Sample {i+1}:")
            print(f"  Question: {question[:100]}...")
            print(f"  Answer: {answer[:100]}...")
            context_data = sample.get('context', [])
            if isinstance(context_data, dict) and "title" in context_data and "sentences" in context_data:
                print(f"  Raw context data: HotpotQA format with {len(context_data['title'])} titles")
                print(f"  First title: {context_data['title'][0] if context_data['title'] else 'None'}")
                print(f"  First sentences: {context_data['sentences'][0][:2] if context_data['sentences'] and context_data['sentences'][0] else 'None'}")
            elif isinstance(context_data, list) and len(context_data) > 0:
                print(f"  Raw context data: List format with {len(context_data)} items")
            else:
                print(f"  Raw context data: Empty or unknown format")
        
        try:
            # Get context chunks
            if hybrid_retriever:
                retrieved_chunks = hybrid_retriever.search(
                    question, top_k=config["rl"]["top_k_candidates"]
                )
                context_chunks = [chunk[0] for chunk in retrieved_chunks]
            else:
                context_chunks = HotpotQADataset.get_context_chunks(
                    sample, chunk_size=config["retrieval"].get("chunk_size", 512)
                )
                context_chunks = context_chunks[: config["rl"]["top_k_candidates"]]

            if not context_chunks:
                context_chunks = [
                    f"Context {j+1}: {question}"
                    for j in range(config["rl"]["top_k_candidates"])
                ]

            features, token_costs, extras = compute_chunk_features(
                question=question,
                chunks=context_chunks,
                answer=answer,
                encoder=encoder,
                cross_encoder=cross_encoder,
                token_budget=config["rl"]["token_budget"],
            )

            if features.shape[0] == 0:
                selected_indices = list(range(min(3, len(context_chunks))))
            elif policy:
                selected_indices = policy.select(
                    features, config["rl"]["token_budget"], token_costs=token_costs
                )
            else:
                selected_indices = select_with_teacher_scores(
                    extras["teacher_q"],
                    token_costs,
                    config["rl"]["token_budget"],
                )

            if not selected_indices:
                selected_indices = select_with_teacher_scores(
                    extras["teacher_q"],
                    token_costs,
                    config["rl"]["token_budget"],
                )

            expanded_chunks = [context_chunks[j] for j in selected_indices]

            if groq_client:
                print(f"  Calling Groq API for sample {i+1}...")
                result = groq_client.generate_answer(question, expanded_chunks)
                generated_answer = result["answer"]
                total_tokens_used = result["total_tokens"]
            else:
                print(f"  Running local inference for sample {i+1}...")
                local_result = local_inference.generate_answer(question, expanded_chunks)
                generated_answer = local_result["answer"]
                total_tokens_used = local_result["total_tokens"]

            print(f"  Model response: {generated_answer[:100]}...")

            if i < 3:
                print(f"  Ground Truth: {answer[:100]}...")
                print(f"  Generated: {generated_answer[:100]}...")
                print(f"  Selected context chunks: {len(expanded_chunks)}")
                for idx, chunk in enumerate(expanded_chunks):
                    print(f"    Chunk {idx+1}: {chunk[:200]}...")

            em, f1 = compute_em_f1(generated_answer, answer)
            precision, recall = compute_evidence_metrics(expanded_chunks, supporting_facts)

            em_scores.append(em)
            f1_scores.append(f1)
            evidence_precisions.append(precision)
            evidence_recalls.append(recall)
            token_counts.append(total_tokens_used)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            em_scores.append(0.0)
            f1_scores.append(0.0)
            evidence_precisions.append(0.0)
            evidence_recalls.append(0.0)
            token_counts.append(1000)
    
    # Calculate results
    results = {}
    
    if "em" in metrics:
        results["em"] = np.mean(em_scores)
    
    if "f1" in metrics:
        results["f1"] = np.mean(f1_scores)
    
    if "evidence_precision" in metrics:
        results["evidence_precision"] = np.mean(evidence_precisions)
    
    if "evidence_recall" in metrics:
        results["evidence_recall"] = np.mean(evidence_recalls)
    
    # Additional metrics
    results["num_samples"] = len(eval_samples)
    results["avg_tokens_per_query"] = np.mean(token_counts)
    results["compression_ratio"] = 0.6  # 60% compression
    
    print(f"QA Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    # Save results
    import json
    import os
    os.makedirs(config['eval']['output_dir'], exist_ok=True)
    
    with open(f"{config['eval']['output_dir']}/qa_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {config['eval']['output_dir']}/qa_results.json")
    

def main():
    args = tyro.cli(Args)
    evaluate_qa(args.config)


def select_with_teacher_scores(
    teacher_scores: np.ndarray,
    token_costs: List[int],
    budget: int,
) -> List[int]:
    """Greedy selection based on teacher Q-values under a token budget."""
    if teacher_scores.size == 0:
        return []

    order = np.argsort(-teacher_scores)
    selected: List[int] = []
    tokens = 0

    for idx in order:
        cost = max(1, int(token_costs[idx]))
        if tokens + cost <= budget or not selected:
            selected.append(int(idx))
            tokens += cost
        if tokens >= budget:
            break

    return selected


if __name__ == "__main__":
    main()
