"""Bandit policies."""
import numpy as np
from typing import List
from refrag.rl.policies import Policy
from refrag.rl.features import TOKEN_SCALE

class LinUCBPolicy(Policy):
    def __init__(self, feature_dim: int, alpha: float = 1.0, lambda_: float = 1.0):
        self.alpha = alpha
        self.A = lambda_ * np.eye(feature_dim)
        self.b = np.zeros(feature_dim)
        self.theta = np.zeros(feature_dim)

    def select(self, features: np.ndarray, budget: int) -> List[int]:
        n = features.shape[0]
        A_inv = np.linalg.inv(self.A)
        self.theta = A_inv @ self.b

        ucb = features @ self.theta + self.alpha * np.sqrt(np.sum(features @ A_inv * features, axis=1))

        selected = []
        tokens = 0
        for idx in np.argsort(-ucb):
            chunk_tokens = max(1, int(round(features[idx, 1] * TOKEN_SCALE)))
            if tokens + chunk_tokens <= budget or not selected:
                selected.append(int(idx))
                tokens += chunk_tokens
                if tokens >= budget:
                    break

        return selected

    def update(self, features: np.ndarray, actions: List[int], reward: float) -> None:
        for idx in actions:
            x = features[idx]
            self.A += np.outer(x, x)
            self.b += reward * x

class ThompsonSamplingPolicy(Policy):
    def __init__(self, feature_dim: int, prior_mean: float = 0.0, prior_var: float = 1.0):
        self.mean = np.zeros(feature_dim)
        self.cov = prior_var * np.eye(feature_dim)

    def select(self, features: np.ndarray, budget: int) -> List[int]:
        theta = np.random.multivariate_normal(self.mean, self.cov)
        scores = features @ theta

        selected = []
        tokens = 0
        for idx in np.argsort(-scores):
            chunk_tokens = max(1, int(round(features[idx, 1] * TOKEN_SCALE)))
            if tokens + chunk_tokens <= budget or not selected:
                selected.append(int(idx))
                tokens += chunk_tokens
                if tokens >= budget:
                    break

        return selected

    def update(self, features: np.ndarray, actions: List[int], reward: float) -> None:
        for idx in actions:
            x = features[idx]
            self.cov = np.linalg.inv(np.linalg.inv(self.cov) + np.outer(x, x))
            self.mean = self.cov @ (np.linalg.inv(self.cov) @ self.mean + reward * x)
