"""Evaluation metrics for QA and evidence selection."""

import re
import string
from collections import Counter
from typing import List, Set, Tuple

import numpy as np


def normalize_answer(s: str) -> str:
    """Normalize answer string for evaluation.

    Args:
        s: Answer string

    Returns:
        Normalized string

    Example:
        >>> normalize_answer("The Answer!")
        'answer'
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(prediction: str, ground_truth: str) -> float:
    """Compute exact match score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match, 0.0 otherwise

    Example:
        >>> compute_em("Paris", "paris")
        1.0
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score between 0 and 1

    Example:
        >>> compute_f1("Paris France", "Paris")
        0.6666...
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def compute_em_f1(prediction: str, ground_truths: List[str]) -> Tuple[float, float]:
    """Compute EM and F1 against multiple ground truths (max score).

    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers

    Returns:
        Tuple of (max_em, max_f1)

    Example:
        >>> compute_em_f1("Paris", ["Paris", "paris"])
        (1.0, 1.0)
    """
    em_scores = [compute_em(prediction, gt) for gt in ground_truths]
    f1_scores = [compute_f1(prediction, gt) for gt in ground_truths]

    return max(em_scores), max(f1_scores)


def extract_answer_spans(text: str, answer: str) -> List[Tuple[int, int]]:
    """Extract all spans in text that match answer (normalized).

    Args:
        text: Source text
        answer: Answer to find

    Returns:
        List of (start, end) character spans

    Example:
        >>> spans = extract_answer_spans("Paris is in France", "Paris")
        >>> len(spans) > 0
        True
    """
    normalized_answer = normalize_answer(answer)
    normalized_text = normalize_answer(text)

    spans = []
    start = 0
    while True:
        idx = normalized_text.find(normalized_answer, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(normalized_answer)))
        start = idx + 1

    return spans


def evidence_prec_recall(
    selected_chunks: List[str],
    gold_answers: List[str],
    all_chunks: List[str],
) -> Tuple[float, float]:
    """Compute evidence precision and recall.

    Precision: fraction of selected chunks containing answer
    Recall: fraction of answer-containing chunks that were selected

    Args:
        selected_chunks: Chunks selected for expansion
        gold_answers: Ground truth answers
        all_chunks: All candidate chunks

    Returns:
        Tuple of (precision, recall)

    Example:
        >>> selected = ["Paris is the capital", "Other text"]
        >>> gold = ["Paris"]
        >>> all_chunks = ["Paris is the capital", "Other text", "More text"]
        >>> prec, rec = evidence_prec_recall(selected, gold, all_chunks)
        >>> prec
        0.5
    """
    # Find which chunks contain answer spans
    def contains_answer(chunk: str, answers: List[str]) -> bool:
        for answer in answers:
            if extract_answer_spans(chunk, answer):
                return True
        return False

    # Count answer-containing chunks
    selected_with_answer = sum(
        1 for chunk in selected_chunks if contains_answer(chunk, gold_answers)
    )
    all_with_answer = sum(1 for chunk in all_chunks if contains_answer(chunk, gold_answers))

    # Compute precision and recall
    precision = selected_with_answer / len(selected_chunks) if selected_chunks else 0.0
    recall = selected_with_answer / all_with_answer if all_with_answer > 0 else 0.0

    return precision, recall


def aggregate_metrics(results: List[dict]) -> dict:
    """Aggregate metrics from multiple samples.

    Args:
        results: List of per-sample metric dictionaries

    Returns:
        Dictionary with aggregated metrics (mean, std)

    Example:
        >>> results = [{"em": 1.0, "f1": 1.0}, {"em": 0.0, "f1": 0.5}]
        >>> agg = aggregate_metrics(results)
        >>> agg["em_mean"]
        0.5
    """
    if not results:
        return {}

    # Collect all metric names
    metric_names = set()
    for result in results:
        metric_names.update(result.keys())

    # Compute mean and std for each metric
    aggregated = {}
    for metric in metric_names:
        values = [r.get(metric, 0.0) for r in results if metric in r]
        if values:
            aggregated[f"{metric}_mean"] = float(np.mean(values))
            aggregated[f"{metric}_std"] = float(np.std(values))
            aggregated[f"{metric}_min"] = float(np.min(values))
            aggregated[f"{metric}_max"] = float(np.max(values))

    aggregated["num_samples"] = len(results)

    return aggregated
