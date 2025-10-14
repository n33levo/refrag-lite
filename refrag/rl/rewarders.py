"""Reward functions."""
from typing import Dict, Any

def compute_reward(perplexity: float, expanded_tokens: int, correctness: float,
                  perp_weight: float = -1.0, token_penalty: float = 0.001,
                  correct_bonus: float = 1.0) -> float:
    reward = perp_weight * perplexity - token_penalty * expanded_tokens + correct_bonus * correctness
    return reward