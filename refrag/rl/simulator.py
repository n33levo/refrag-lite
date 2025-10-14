"""Offline RL simulator for training policies."""
import numpy as np
from typing import List, Dict, Any, Tuple
from refrag.rl.features import extract_chunk_features
from refrag.rl.rewarders import compute_reward
from refrag.utils.token_budget import estimate_tokens


class OfflineSimulator:
    """Offline simulator for RL policy training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.token_budget = config.get("token_budget", 2000)
        self.compression_rate = config.get("compression_rate", 0.75)
        
    def simulate_rollout(self, query: str, chunks: List[str], 
                        query_embedding: np.ndarray, chunk_embeddings: np.ndarray,
                        policy, ground_truth: str = None) -> Dict[str, Any]:
        """Simulate a single rollout.
        
        Args:
            query: Input query
            chunks: Retrieved chunks
            query_embedding: Query embedding vector
            chunk_embeddings: Chunk embedding vectors
            policy: RL policy to evaluate
            ground_truth: Ground truth answer (optional)
            
        Returns:
            Dictionary with rollout results
        """
        # Extract features
        features = extract_chunk_features(
            chunks, query, query_embedding, chunk_embeddings
        )
        
        # Policy selects chunks to expand
        selected_indices = policy.select(features, self.token_budget)
        
        # Compute token usage
        expanded_tokens = sum(estimate_tokens(chunks[i]) for i in selected_indices)
        compressed_tokens = len(chunks) - len(selected_indices)  # 1 token per compressed
        
        # Simulate reward (placeholder)
        perplexity = 2.5  # Placeholder
        correctness = 0.8 if ground_truth else 0.0  # Placeholder
        
        reward = compute_reward(
            perplexity=perplexity,
            expanded_tokens=expanded_tokens,
            correctness=correctness,
            perp_weight=self.config.get("reward", {}).get("perplexity_weight", -1.0),
            token_penalty=self.config.get("reward", {}).get("token_penalty", 0.001),
            correct_bonus=self.config.get("reward", {}).get("correctness_bonus", 1.0)
        )
        
        return {
            "query": query,
            "chunks": chunks,
            "selected_indices": selected_indices,
            "features": features,
            "expanded_tokens": expanded_tokens,
            "compressed_tokens": compressed_tokens,
            "total_tokens": expanded_tokens + compressed_tokens,
            "reward": reward,
            "perplexity": perplexity,
            "correctness": correctness
        }
    
    def simulate_batch(self, queries: List[str], chunks_batch: List[List[str]],
                      query_embeddings: np.ndarray, chunk_embeddings_batch: List[np.ndarray],
                      policy, ground_truths: List[str] = None) -> List[Dict[str, Any]]:
        """Simulate a batch of rollouts.
        
        Args:
            queries: List of queries
            chunks_batch: List of chunk lists
            query_embeddings: Query embedding matrix
            chunk_embeddings_batch: List of chunk embedding matrices
            policy: RL policy to evaluate
            ground_truths: List of ground truth answers (optional)
            
        Returns:
            List of rollout results
        """
        results = []
        for i, query in enumerate(queries):
            result = self.simulate_rollout(
                query=query,
                chunks=chunks_batch[i],
                query_embedding=query_embeddings[i],
                chunk_embeddings=chunk_embeddings_batch[i],
                policy=policy,
                ground_truth=ground_truths[i] if ground_truths else None
            )
            results.append(result)
        return results
