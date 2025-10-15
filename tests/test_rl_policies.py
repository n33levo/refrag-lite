"""Tests for RL policies and fallback selection."""
import numpy as np

from refrag.eval.qa_eval import smart_context_selection
from refrag.rl.bandit import LinUCBPolicy
from refrag.rl.features import TOKEN_SCALE
from refrag.utils.token_budget import estimate_tokens


def test_linucb_respects_token_budget():
    policy = LinUCBPolicy(feature_dim=6, alpha=0.0, lambda_=1.0)
    features = np.array(
        [
            [1.0, 0.05, 0.0, 0.8, 1.0, 0.0],   # ~50 tokens
            [0.8, 0.08, 0.5, 0.6, 0.8, 0.1],   # ~80 tokens
            [0.2, 0.40, 1.0, 0.1, 0.2, 0.9],   # ~400 tokens
        ],
        dtype=np.float32,
    )

    selected = policy.select(features, budget=120)
    token_usage = sum(int(round(features[idx, 1] * TOKEN_SCALE)) for idx in selected)

    assert selected, "policy should select at least one chunk"
    assert token_usage <= 120 or len(selected) == 1


def test_smart_selection_respects_token_budget():
    chunks = [
        "Paris is the capital and most populous city of France.",
        "London is the capital of England and the United Kingdom.",
        "Berlin is the capital and largest city of Germany.",
    ]
    question = "What city is the capital of France?"

    selected = smart_context_selection(chunks, question, token_budget=40)
    token_usage = sum(estimate_tokens(chunks[idx]) for idx in selected)

    assert selected, "fallback selection must return at least one chunk"
    assert token_usage <= 40 or len(selected) == 1
