"""Feature extraction."""
import numpy as np
from typing import List, Dict, Any
from refrag.utils.token_budget import estimate_tokens

def extract_chunk_features(chunks: List[str], query: str, query_embedding: np.ndarray,
                          chunk_embeddings: np.ndarray, reranker_scores: List[float] = None) -> np.ndarray:
    n = len(chunks)
    features = np.zeros((n, 6))

    # Similarity to query
    if query_embedding is not None and chunk_embeddings is not None:
        features[:, 0] = np.dot(chunk_embeddings, query_embedding.T).flatten()

    # Length
    features[:, 1] = np.array([estimate_tokens(c) for c in chunks]) / 1000.0

    # Position
    features[:, 2] = np.arange(n) / max(n - 1, 1)

    # Reranker score
    if reranker_scores:
        features[:, 3] = np.array(reranker_scores)

    # Novelty (placeholder)
    features[:, 4] = 1.0

    # Redundancy (placeholder)
    features[:, 5] = 0.0

    return features