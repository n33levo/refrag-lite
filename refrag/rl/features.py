"""Feature extraction helpers for RL policies."""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Set

import numpy as np

from refrag.utils.token_budget import estimate_tokens

# The policies expect the length feature to be measured in thousands of tokens.
TOKEN_SCALE = 1000.0


def _tokenize(text: str) -> Set[str]:
    """Lightweight tokenizer shared across feature helpers."""
    if not text:
        return set()
    return set(re.findall(r"\b\w+\b", text.lower()))


def _lexical_overlap(query_words: Set[str], chunk_words: Set[str]) -> float:
    if not query_words or not chunk_words:
        return 0.0
    intersection = len(query_words & chunk_words)
    union = len(query_words | chunk_words)
    if union == 0:
        return 0.0
    return intersection / union


def _novelty_and_redundancy(chunk_word_sets: Sequence[Set[str]]) -> np.ndarray:
    """Compute novelty (1-overlap) and redundancy heuristics for each chunk."""
    novelty = np.zeros(len(chunk_word_sets), dtype=np.float32)
    redundancy = np.zeros(len(chunk_word_sets), dtype=np.float32)

    for i, words in enumerate(chunk_word_sets):
        if not words:
            novelty[i] = 1.0
            redundancy[i] = 0.0
            continue

        overlaps = []
        for j, other_words in enumerate(chunk_word_sets):
            if i == j or not other_words:
                continue
            overlap = len(words & other_words) / max(len(words), 1)
            overlaps.append(overlap)

        if overlaps:
            mean_overlap = float(np.clip(np.mean(overlaps), 0.0, 1.0))
            novelty[i] = 1.0 - mean_overlap
            redundancy[i] = mean_overlap
        else:
            novelty[i] = 1.0
            redundancy[i] = 0.0

    return np.stack([novelty, redundancy], axis=-1)


def extract_chunk_features(
    chunks: List[str],
    query: str,
    query_embedding: Optional[np.ndarray] = None,
    chunk_embeddings: Optional[np.ndarray] = None,
    reranker_scores: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Create feature matrix expected by bandit and PPO policies.

    Feature layout:
        0: Query similarity (embedding cosine or lexical overlap)
        1: Normalised token length (tokens / TOKEN_SCALE)
        2: Position within retrieved list
        3: Reranker score (if available, else fallback to similarity)
        4: Novelty heuristic
        5: Redundancy heuristic
    """
    n = len(chunks)
    features = np.zeros((n, 6), dtype=np.float32)

    if n == 0:
        return features

    token_estimates = np.array([estimate_tokens(c) for c in chunks], dtype=np.float32)
    features[:, 1] = token_estimates / TOKEN_SCALE

    if n == 1:
        features[0, 2] = 0.0
    else:
        features[:, 2] = np.linspace(0.0, 1.0, num=n, endpoint=True)

    if reranker_scores is not None:
        features[:, 3] = np.asarray(reranker_scores, dtype=np.float32)

    chunk_word_sets = [_tokenize(chunk) for chunk in chunks]
    query_words = _tokenize(query)

    if query_embedding is not None and chunk_embeddings is not None:
        similarities = np.dot(chunk_embeddings, query_embedding.T).astype(np.float32).flatten()
        features[:, 0] = similarities
        if reranker_scores is None:
            features[:, 3] = similarities
    else:
        lexical_similarities = np.array(
            [_lexical_overlap(query_words, words) for words in chunk_word_sets],
            dtype=np.float32,
        )
        features[:, 0] = lexical_similarities
        if reranker_scores is None:
            features[:, 3] = lexical_similarities

    novelty_redundancy = _novelty_and_redundancy(chunk_word_sets)
    features[:, 4:] = novelty_redundancy

    return features
