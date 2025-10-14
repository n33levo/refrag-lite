"""Policy interface."""
import numpy as np
from typing import List
from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def select(self, features: np.ndarray, budget: int) -> List[int]:
        pass

    @abstractmethod
    def update(self, features: np.ndarray, actions: List[int], reward: float) -> None:
        pass