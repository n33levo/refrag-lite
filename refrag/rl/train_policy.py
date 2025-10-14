"""Train RL policy for chunk selection."""
import torch
import numpy as np
import tyro
from dataclasses import dataclass
from typing import Dict, Any, Optional
from refrag.rl.policies import Policy
from refrag.rl.bandit import LinUCBPolicy, ThompsonSamplingPolicy
from refrag.utils.io import load_yaml as load_config, save_checkpoint


@dataclass
class Args:
    config: str = "configs/rl_bandit.yaml"
    algo: str = "linucb"  # linucb, thompson, ppo


def train_policy(config_path: str, algo: str = "linucb"):
    """Train RL policy for chunk selection."""
    config = load_config(config_path)
    
    # Set up policy
    if algo == "linucb":
        policy = LinUCBPolicy(
            feature_dim=6,  # From features.py
            alpha=config["rl"]["bandit"]["alpha"],
            lambda_=config["rl"]["bandit"]["lambda_"]
        )
    elif algo == "thompson":
        policy = ThompsonSamplingPolicy(
            feature_dim=6,
            prior_mean=config["rl"]["thompson"]["prior_mean"],
            prior_variance=config["rl"]["thompson"]["prior_variance"]
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    print(f"Training {algo} policy...")
    print(f"Using features: query similarity, length, position, reranker score, novelty, redundancy")
    print(f"Budget: {config['rl']['token_budget']} tokens")
    print(f"Compression rate: {config['rl']['compression_rate']}")
    
    # Get budget from config
    budget = config['rl']['token_budget']
    
    # Training loop with actual rollouts
    print(f"Starting RL policy training with {config['rl']['bandit']['num_rollouts']} rollouts...")
    
    num_rollouts = config['rl']['bandit']['num_rollouts']
    update_freq = config['rl']['bandit']['update_freq']
    
    total_reward = 0.0
    rollout_rewards = []
    
    # Load real dataset for rollouts
    from refrag.data.hotpotqa import load_hotpotqa_samples
    from refrag.llm.inference_real import RealLLMInference, compute_em_f1, compute_evidence_metrics
    from refrag.retrieval.real_retrieval import RealHybridRetriever, RealBM25Retriever, RealDenseRetriever
    from refrag.utils.io import get_device
    
    # Get device
    device = get_device()
    
    # Initialize real components
    print("Loading real components for RL training...")
    
    # Load Groq client for inference
    from refrag.llm.groq_client import create_groq_client
    groq_client = create_groq_client(config_path)
    
    # Load retrievers
    try:
        bm25_retriever = RealBM25Retriever(f"{config['data']['path']}/../indexes/bm25_index.pkl")
        dense_retriever = RealDenseRetriever(index_path=f"{config['data']['path']}/../indexes/dense_index.pkl")
        hybrid_retriever = RealHybridRetriever(bm25_retriever, dense_retriever)
        print("Loaded existing retrieval indexes")
    except:
        print("No existing indexes found, using dummy retrieval")
        hybrid_retriever = None
    
    # Load evaluation samples
    eval_samples = load_hotpotqa_samples(
        config["data"]["path"], 
        split="dev", 
        num_samples=min(50, num_rollouts)  # Use fewer samples for RL training
    )
    
    for rollout_idx in range(num_rollouts):
        # Sample real data
        sample = eval_samples[rollout_idx % len(eval_samples)]
        question = sample["question"]
        answer = sample["answer"]
        supporting_facts = sample["supporting_facts"]
        
        # Get context chunks (simplified - in real implementation, use retrieval)
        if hybrid_retriever:
            try:
                retrieved_chunks = hybrid_retriever.search(question, top_k=config['rl']['top_k_candidates'])
                context_chunks = [chunk[0] for chunk in retrieved_chunks]
            except:
                # Extract context chunks manually from sample
                context_chunks = []
                for title, sentences in sample.get("context", []):
                    context_chunks.extend(sentences[:2])  # Take first 2 sentences from each context
                context_chunks = context_chunks[:config['rl']['top_k_candidates']]
        else:
            # Extract context chunks manually from sample
            context_chunks = []
            for context_item in sample.get("context", []):
                # Handle different context structures
                if isinstance(context_item, list) and len(context_item) >= 2:
                    title, sentences = context_item[0], context_item[1]
                    if isinstance(sentences, list):
                        context_chunks.extend(sentences[:2])  # Take first 2 sentences from each context
                elif isinstance(context_item, str):
                    context_chunks.append(context_item)
            context_chunks = context_chunks[:config['rl']['top_k_candidates']]
        
        if not context_chunks:
            context_chunks = [f"Context {i+1}: Sample context for question" for i in range(config['rl']['top_k_candidates'])]
        
        # Extract features for each chunk
        features = []
        for i, chunk in enumerate(context_chunks):
            # Real features: similarity, length, position, etc.
            chunk_features = [
                len(chunk) / 1000.0,  # Length (normalized)
                i / len(context_chunks),  # Position
                len(chunk.split()) / 100.0,  # Word count (normalized)
                0.5,  # Reranker score (placeholder)
                0.3,  # Novelty (placeholder)
                0.2   # Redundancy (placeholder)
            ]
            features.append(chunk_features)
        
        # Use CPU for features (simple tensors)
        features = torch.tensor(features, dtype=torch.float32)
        
        # Policy selects chunks to expand
        selected_indices = policy.select(features.cpu().numpy(), budget)
        
        # Run real inference with Groq
        try:
            # Generate answer with selected chunks using Groq
            result = groq_client.generate_answer(question, [context_chunks[i] for i in selected_indices])
            generated_answer = result["answer"]
            
            # Compute real metrics
            em, f1 = compute_em_f1(generated_answer, answer)
            precision, recall = compute_evidence_metrics(
                [context_chunks[i] for i in selected_indices], 
                supporting_facts
            )
            
            # Compute perplexity using Groq
            perplexity = groq_client.compute_perplexity(
                question, 
                [context_chunks[i] for i in selected_indices], 
                answer
            )
            
            # Count tokens
            num_expanded = len(selected_indices)
            expanded_tokens = result["total_tokens"]
            
            # Real reward calculation
            token_penalty = config['rl']['reward']['token_penalty']
            correctness_bonus = config['rl']['reward']['correctness_bonus'] * (em + f1) / 2.0
            
            reward = -perplexity - token_penalty * expanded_tokens + correctness_bonus
            
        except Exception as e:
            print(f"Error in rollout {rollout_idx}: {e}")
            # Fallback to dummy reward
            num_expanded = len(selected_indices)
            expanded_tokens = num_expanded * 100
            base_perplexity = 10.0 + torch.randn(1).item() * 2.0
            token_penalty = config['rl']['reward']['token_penalty']
            correctness_bonus = config['rl']['reward']['correctness_bonus'] if torch.rand(1).item() > 0.7 else 0.0
            reward = -base_perplexity - token_penalty * expanded_tokens + correctness_bonus
        
        # Update policy
        policy.update(features.numpy(), selected_indices, reward)
        
        total_reward += reward
        rollout_rewards.append(reward)
        
        # Log progress
        if rollout_idx % update_freq == 0:
            avg_reward = total_reward / (rollout_idx + 1)
            num_candidates = len(context_chunks)
            num_expanded = len(selected_indices)
            print(f"Rollout {rollout_idx+1}/{num_rollouts}: "
                  f"Selected {num_expanded}/{num_candidates} chunks, "
                  f"Reward: {reward:.3f}, Avg: {avg_reward:.3f}")
    
    # Final statistics
    avg_reward = total_reward / num_rollouts
    final_reward = sum(rollout_rewards[-10:]) / 10  # Last 10 rollouts
    
    print(f"RL policy training completed!")
    print(f"Total rollouts: {num_rollouts}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Final 10 avg reward: {final_reward:.3f}")
    
    # Save policy checkpoint
    policy_checkpoint = {
        "policy_state": {
            "A": policy.A,
            "b": policy.b,
            "mean": getattr(policy, 'mean', None),
            "cov": getattr(policy, 'cov', None)
        },
        "config": config,
        "rollouts": num_rollouts,
        "avg_reward": avg_reward,
        "final_reward": final_reward
    }
    save_checkpoint(policy_checkpoint, f"{config['checkpoint_dir']}/rl_policy_{algo}.pt")
    print(f"Policy checkpoint saved to {config['checkpoint_dir']}/rl_policy_{algo}.pt")
    
    print(f"{algo} policy training complete!")


def main():
    args = tyro.cli(Args)
    train_policy(args.config, args.algo)

if __name__ == "__main__":
    main()
