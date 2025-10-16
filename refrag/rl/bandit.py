"""Bandit policies."""
import numpy as np
from typing import List
from refrag.rl.policies import Policy

class LinUCBPolicy(Policy):
    def __init__(self, feature_dim: int, alpha: float = 1.0, lambda_: float = 1.0):
        self.alpha = alpha
        self.A = lambda_ * np.eye(feature_dim)
        self.b = np.zeros(feature_dim)
        self.theta = np.zeros(feature_dim)

    def select(
        self,
        features: np.ndarray,
        budget: int,
        token_costs: List[int] | None = None,
    ) -> List[int]:
        n = features.shape[0]
        if n == 0:
            return []

        A_inv = np.linalg.inv(self.A)
        self.theta = A_inv @ self.b

        ucb = features @ self.theta + self.alpha * np.sqrt(
            np.sum(features @ A_inv * features, axis=1)
        )

        selected: List[int] = []
        tokens = 0

        for idx in np.argsort(-ucb):
            chunk_tokens = (
                int(token_costs[idx])
                if token_costs is not None
                else int(features[idx, 1] * 1000)
            )
            if chunk_tokens <= 0:
                chunk_tokens = 1

            if tokens + chunk_tokens <= budget:
                selected.append(int(idx))
                tokens += chunk_tokens

        if not selected and n > 0:
            # Ensure we always expand at least the highest scoring chunk
            best_idx = int(np.argmax(ucb))
            selected.append(best_idx)

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

    def select(
        self,
        features: np.ndarray,
        budget: int,
        token_costs: List[int] | None = None,
    ) -> List[int]:
        if features.size == 0:
            return []

        theta = np.random.multivariate_normal(self.mean, self.cov)
        scores = features @ theta

        selected: List[int] = []
        tokens = 0
        for idx in np.argsort(-scores):
            chunk_tokens = (
                int(token_costs[idx])
                if token_costs is not None
                else int(features[idx, 1] * 1000)
            )
            if chunk_tokens <= 0:
                chunk_tokens = 1

            if tokens + chunk_tokens <= budget:
                selected.append(int(idx))
                tokens += chunk_tokens

        if not selected and features.shape[0] > 0:
            selected.append(int(np.argmax(scores)))

        return selected

    def update(self, features: np.ndarray, actions: List[int], reward: float) -> None:
        for idx in actions:
            x = features[idx]
            self.cov = np.linalg.inv(np.linalg.inv(self.cov) + np.outer(x, x))
            self.mean = self.cov @ (np.linalg.inv(self.cov) @ self.mean + reward * x)
