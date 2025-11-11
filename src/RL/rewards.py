from __future__ import annotations
from typing import Protocol, Dict, Any
import numpy as np


class Reward(Protocol):
    def reset(self) -> None: ...
    def __call__(self, *, t: int, action: int, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float: ...


class SimpleSignReward:
    """
    Reward = +1 if sign(action) == sign(target) (beyond threshold), else -1.
    If |target| < threshold, treat as 'flat': reward +1 if action==0 else -1.
    """
    def __init__(self, threshold: float = 0.0, wrong_penalty: float = -1.0, correct_reward: float = 1.0):
        self.threshold = float(threshold)
        self.wrong_penalty = float(wrong_penalty)
        self.correct_reward = float(correct_reward)

    def reset(self) -> None:
        pass

    def __call__(self, *, t: int, action: int, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float:
        if abs(target) < self.threshold:
            return self.correct_reward if action == 0 else self.wrong_penalty
        sign_t = 1 if target > 0 else -1
        return self.correct_reward if action == sign_t else self.wrong_penalty


class CustomReward:
    """Placeholder to be filled by the user if desired."""
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def reset(self) -> None:
        pass

    def __call__(self, *, t: int, action: int, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float:
        return 0.0
