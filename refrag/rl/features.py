"""Feature extraction utilities for chunk selection policies."""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder

from refrag.compress.encoder import ChunkEncoder
from refrag.utils.token_budget import estimate_tokens


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _lexical_overlap(question: str, chunk: str) -> float:
    question_tokens = set(re.findall(r"\b\w+\b", question.lower()))
    chunk_tokens = set(re.findall(r"\b\w+\b", chunk.lower()))
    if not question_tokens or not chunk_tokens:
        return 0.0
    return len(question_tokens & chunk_tokens) / len(question_tokens)


def _answer_overlap(answer: str, chunk: str) -> float:
    normalized_answer = _normalize_text(answer)
    normalized_chunk = _normalize_text(chunk)
    if not normalized_answer:
        return 0.0
    if normalized_answer in normalized_chunk:
        return 1.0
    answer_tokens = set(normalized_answer.split())
    chunk_tokens = set(normalized_chunk.split())
    if not chunk_tokens or not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & chunk_tokens)
    return overlap / len(answer_tokens)


def compute_chunk_features(
    question: str,
    chunks: Sequence[str],
    answer: str,
    encoder: ChunkEncoder,
    cross_encoder: CrossEncoder,
    token_budget: int,
) -> Tuple[np.ndarray, List[int], Dict[str, np.ndarray]]:
    """Compute rich features for each chunk.

    Returns
    -------
    features:
        Array of shape (num_chunks, feature_dim) with
        [semantic_sim, token_ratio, position, cross_score,
         lexical_overlap, novelty, answer_overlap, teacher_q]
    token_counts:
        List of token cost per chunk.
    extras:
        Dictionary containing intermediate signals (cross_probs, semantic_sim, etc.)
    """
    if not chunks:
        return np.zeros((0, 8), dtype=np.float32), [], {}

    with torch.no_grad():
        question_embedding = encoder.encode([question]).to(torch.float32)
        chunk_embeddings = encoder.encode(chunks).to(torch.float32)

    question_embedding = F.normalize(question_embedding, dim=-1)
    chunk_embeddings = F.normalize(chunk_embeddings, dim=-1)

    semantic_scores = (chunk_embeddings @ question_embedding.T).squeeze(-1).cpu().numpy()

    sim_matrix = chunk_embeddings @ chunk_embeddings.T
    sim_matrix.fill_diagonal_(0.0)
    # novelty = 1 - average similarity with other chunks
    novelty_scores = []
    for i in range(sim_matrix.size(0)):
        others = torch.cat([sim_matrix[i, :i], sim_matrix[i, i + 1 :]])
        if others.numel() == 0:
            novelty_scores.append(1.0)
        else:
            novelty_scores.append(1.0 - torch.clamp(others.mean(), min=0.0).item())
    novelty_scores = np.array(novelty_scores, dtype=np.float32)

    cross_inputs = [(question, chunk) for chunk in chunks]
    cross_logits = cross_encoder.predict(cross_inputs)
    cross_logits = np.array(cross_logits, dtype=np.float32).reshape(-1)
    cross_probs = 1.0 / (1.0 + np.exp(-cross_logits))

    token_counts = [max(1, estimate_tokens(chunk)) for chunk in chunks]
    token_ratio = np.array(
        [min(count / max(token_budget, 1), 1.0) for count in token_counts],
        dtype=np.float32,
    )

    lexical = np.array([_lexical_overlap(question, chunk) for chunk in chunks], dtype=np.float32)
    answer_signal = np.array([_answer_overlap(answer, chunk) for chunk in chunks], dtype=np.float32)
    teacher_q = 0.7 * cross_probs + 0.3 * answer_signal

    positions = np.linspace(0.0, 1.0, num=len(chunks), endpoint=True).astype(np.float32)

    feature_matrix = np.stack(
        [
            semantic_scores,
            token_ratio,
            positions,
            cross_probs,
            lexical,
            novelty_scores,
            answer_signal,
            teacher_q,
        ],
        axis=1,
    )

    extras = {
        "semantic": semantic_scores,
        "cross_probs": cross_probs,
        "novelty": novelty_scores,
        "answer_overlap": answer_signal,
        "teacher_q": teacher_q,
    }
    return feature_matrix, token_counts, extras
