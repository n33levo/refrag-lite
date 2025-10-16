"""Policy interface."""
import numpy as np
from typing import List
from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def select(
        self,
        features: np.ndarray,
        budget: int,
        token_costs: List[int] | None = None,
    ) -> List[int]:
        """Return indices of chunks to expand."""

    @abstractmethod
    def update(self, features: np.ndarray, actions: List[int], reward: float) -> None:
        """Update policy parameters from observed reward."""
