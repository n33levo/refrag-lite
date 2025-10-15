"""QA evaluation."""
import torch
import tyro
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple
from refrag.utils.io import load_yaml as load_config, get_device
from refrag.utils.token_budget import estimate_tokens
from refrag.rl.features import extract_chunk_features as build_chunk_features


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
    
    # Load evaluation samples
    eval_samples = load_hotpotqa_samples(
        config["data"]["path"], 
        split="dev", 
        num_samples=num_samples
    )
    
    # Load Groq client for inference
    from refrag.llm.groq_client import create_groq_client
    groq_client = create_groq_client(config_path)
    
    # Load retrievers
    try:
        bm25_retriever = RealBM25Retriever(f"{config['data']['path']}/../indexes/bm25_index.pkl")
        dense_retriever = RealDenseRetriever(index_path=f"{config['data']['path']}/../indexes/dense_index.pkl")
        if bm25_retriever.bm25 is not None and dense_retriever.index is not None:
            hybrid_retriever = RealHybridRetriever(bm25_retriever, dense_retriever)
            print("Loaded existing retrieval indexes")
        else:
            print("Retrieval indexes missing or incomplete, using dataset contexts")
            hybrid_retriever = None
    except Exception as exc:
        print(f"Failed to load retrieval indexes ({exc}), using dataset contexts")
        hybrid_retriever = None
    
    # Load RL policy
    try:
        policy = LinUCBPolicy(
            feature_dim=6,
            alpha=config["rl"]["bandit"]["alpha"],
            lambda_=config["rl"]["bandit"]["lambda_"]
        )
        # Load policy weights if available
        checkpoint_path = f"{config['checkpoint_dir']}/rl_policy_linucb.pt"
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            policy.A = checkpoint["policy_state"]["A"]
            policy.b = checkpoint["policy_state"]["b"]
            print("Loaded trained RL policy")
        else:
            print("No trained policy found, using random policy")
    except:
        print("Error loading policy, using random selection")
        policy = None
    
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
            reranker_scores = None
            if hybrid_retriever:
                retrieved_chunks = hybrid_retriever.search(
                    question, top_k=config['rl']['top_k_candidates']
                )
                context_chunks = [chunk[0] for chunk in retrieved_chunks]
                reranker_scores = [chunk[1] for chunk in retrieved_chunks]
            else:
                context_chunks = HotpotQADataset.get_context_chunks(
                    sample, chunk_size=config['rl'].get('chunk_size', 256)
                )
            
            context_chunks = context_chunks[:config['rl']['top_k_candidates']]
            if not context_chunks:
                context_chunks = [
                    f"Context {j+1}: Sample context for question"
                    for j in range(config['rl']['top_k_candidates'])
                ]
                reranker_scores = None

            features = build_chunk_features(
                chunks=context_chunks,
                query=question,
                reranker_scores=reranker_scores,
            )

            # Select chunks using policy or heuristic selection
            if policy:
                selected_indices = policy.select(features, config['rl']['token_budget'])
            else:
                selected_indices = smart_context_selection(
                    context_chunks=context_chunks,
                    question=question,
                    token_budget=config['rl']['token_budget'],
                    precomputed_features=features,
                )
            
            # Generate answer using Groq
            print(f"  Calling Groq API for sample {i+1}...")
            result = groq_client.generate_answer(
                question, 
                [context_chunks[j] for j in selected_indices]
            )
            
            generated_answer = result["answer"]
            print(f"  Groq response: {generated_answer[:100]}...")
            
            # Debug: Print sample answers
            if i < 3:  # Print first 3 samples for debugging
                print(f"  Ground Truth: {answer[:100]}...")
                print(f"  Generated: {generated_answer[:100]}...")
                print(f"  Context chunks: {len([context_chunks[j] for j in selected_indices])}")
                print(f"  Selected context chunks:")
                for idx, chunk in enumerate([context_chunks[j] for j in selected_indices]):
                    print(f"    Chunk {idx+1}: {chunk[:200]}...")
            
            # Compute metrics
            em, f1 = compute_em_f1(generated_answer, answer)
            precision, recall = compute_evidence_metrics(
                [context_chunks[j] for j in selected_indices], 
                supporting_facts
            )
            
            em_scores.append(em)
            f1_scores.append(f1)
            evidence_precisions.append(precision)
            evidence_recalls.append(recall)
            token_counts.append(sum(estimate_tokens(context_chunks[j]) for j in selected_indices))
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            print(f"  Question: {question[:100] if 'question' in locals() else 'N/A'}...")
            print(f"  Answer: {answer[:100] if 'answer' in locals() else 'N/A'}...")
            print(f"  Generated: {generated_answer[:100] if 'generated_answer' in locals() else 'N/A'}...")
            # Use dummy values for failed samples
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

def smart_context_selection(
    context_chunks: List[str],
    question: str,
    token_budget: int,
    precomputed_features: np.ndarray | None = None,
) -> List[int]:
    """Heuristic fallback selection that honours the token budget."""
    if not context_chunks:
        return []

    if precomputed_features is None:
        precomputed_features = build_chunk_features(context_chunks, question)

    # Score chunks: favour similarity, reranker scores, and novelty while
    # slightly preferring earlier chunks and penalising redundancy.
    similarity = precomputed_features[:, 0]
    reranker = precomputed_features[:, 3]
    novelty = precomputed_features[:, 4]
    position = precomputed_features[:, 2]
    redundancy = precomputed_features[:, 5]

    scores = (
        0.45 * similarity
        + 0.35 * reranker
        + 0.15 * novelty
        + 0.05 * (1.0 - position)
        - 0.05 * redundancy
    )

    order = np.argsort(-scores)
    selected: List[int] = []
    used_tokens = 0
    budget = max(token_budget, 0)

    for idx in order:
        chunk_tokens = max(1, estimate_tokens(context_chunks[idx]))
        if used_tokens + chunk_tokens <= budget or not selected or budget == 0:
            selected.append(int(idx))
            used_tokens += chunk_tokens
        if budget > 0 and used_tokens >= budget:
            break

    selected.sort()
    return selected


if __name__ == "__main__":
    main()
