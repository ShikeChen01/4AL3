from __future__ import annotations
from typing import Protocol, Dict, Any
import numpy as np
from collections import deque
import numpy as np
from math import prod

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

class WindowedSignReward:
    """
    Sign-based reward with:
    - penalty for 'hold' actions
    - extra penalty if performance over a window is not positive.

    Actions: -1, 0, +1
    target: scalar (e.g. forward return)
    """
    def __init__(
        self,
        threshold: float = 0.0,
        wrong_penalty: float = -1.0,
        correct_reward: float = 1.0,
        hold_penalty: float = -0.1,
        window_size: int = 50,
        window_penalty: float = -5.0,
    ) -> None:
        """
        threshold: if |target| < threshold -> 'flat' regime, correct action is 0
        wrong_penalty: reward when prediction is wrong
        correct_reward: reward when prediction is correct
        hold_penalty: extra negative reward whenever action == 0
                      (discourages always-hold behavior)
        window_size: length of rolling window to monitor performance
        window_penalty: extra negative reward if sum(window_rewards) <= 0
                        (model not making meaningful predictions over the window)
        """
        self.threshold = float(threshold)
        self.wrong_penalty = float(wrong_penalty)
        self.correct_reward = float(correct_reward)
        self.hold_penalty = float(hold_penalty)
        self.window_size = int(window_size)
        self.window_penalty = float(window_penalty)

        self._window = deque(maxlen=self.window_size)
        self._action_window = deque(maxlen=self.window_size)
        self.next_base = 0.0

    def reset(self) -> None:
        self._window.clear()
        self._action_window.clear()
        self.next_base = 0.0

    def __call__(self, *, t: int, action: int, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float:     
        base = self.next_base
        base += action*target
        self._action_window.append(action)
        # --- base sign reward (same logic as SimpleSignReward) ---
        if abs(target) < self.threshold:
            base += self.correct_reward if action == 0 else self.wrong_penalty
        else:
            base += self.correct_reward if np.sign(action) == np.sign(target) else self.wrong_penalty

        # --- window logic: penalize if recent performance is not positive ---
        self._window.append(base)
        window_sum = sum(self._window)

        applied_window_penalty = False
        if len(self._window) == self.window_size and window_sum <= 0.0:
            base += self.window_penalty
            applied_window_penalty = True

        # (Optional) write diagnostics back into info
        info["window_sum"] = float(window_sum)
        info["window_len"] = len(self._window)
        info["window_penalty_applied"] = applied_window_penalty

        return float(base)
    
class WindowedSignReward_v2:
    """
    Sign-based reward with:
    - penalty for 'hold' actions
    - extra penalty if performance over a window is not positive.

    Actions: -1, 0, +1
    target: scalar (e.g. forward return)
    """
    def __init__(
        self,
        threshold: float = 0.0,
        wrong_penalty: float = -0.65,
        correct_reward: float = 0.0,
        hold_penalty: float = -0.15,
        window_size: int = 25,
        window_penalty: float = -1.0,
        action_penality_threshold: float = -0.25,
        action_window_size: int = 10,
        action_penalty: float = -0.05,
    ) -> None:
        """
        threshold: if |target| < threshold -> 'flat' regime, correct action is 0
        wrong_penalty: reward when prediction is wrong
        correct_reward: reward when prediction is correct
        hold_penalty: extra negative reward whenever action == 0
                      (discourages always-hold behavior)
        window_size: length of rolling window to monitor performance
        window_penalty: extra negative reward if sum(window_rewards) <= 0
                        (model not making meaningful predictions over the window)
        """

        self.threshold = threshold
        self.wrong_penalty = wrong_penalty
        self.correct_reward = correct_reward
        self.hold_penalty = hold_penalty
        self.window_size = window_size
        self.window_penalty = window_penalty
        self.action_window_size = action_window_size
        self.action_window_penalty = action_penalty
        self.action_penality_threshold = action_penality_threshold

        self._window = deque(maxlen=self.window_size)
        self._action_window = deque(maxlen=self.action_window_size)
        self.cur_window_penalty = float(0.0) #to be fixed
        self.cur_action_window_penalty = float(0.0)

    def reset(self) -> None:
        self._window.clear()
        self._action_window.clear()
        self.window_penalty = float(0.0)
        self.action_window_penalty = float(0.0)

    def __call__(self, *, t: int, action: int, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float:  
        print("Target before scaling:", target)  
        target = target * 100 
        base = 0.0
        base += action*target
        self._window.append(base)
        self._action_window.append(action)

        # --- base sign reward (same logic as SimpleSignReward) ---
        if abs(target) < self.threshold:
            base += self.correct_reward if action == 0 else self.wrong_penalty
        else:
            base += self.correct_reward if np.sign(action) == np.sign(target) else self.wrong_penalty 

        # --- window logic: penalize if recent performance is not positive ---

        window_sum = sum(self._window)

        applied_window_penalty = False
        if len(self._window) == self.window_size and window_sum <= 0.0:
            self.cur_window_penalty += self.window_penalty
            applied_window_penalty = True
        else:
            self.cur_window_penalty = 0.0
        
        actions = abs(sum(self._action_window)/len(self._action_window))
        applied_action_window_penalty = False

        if actions <= 1 and len(self._action_window) == self.action_window_size and self.cur_action_window_penalty > self.action_penality_threshold:
            self.cur_action_window_penalty += self.action_window_penalty
            applied_action_window_penalty = True
        else:
            self.cur_action_window_penalty = 0.0

        base += self.cur_window_penalty + self.cur_action_window_penalty

        # (Optional) write diagnostics back into info
        info["window_sum"] = float(window_sum)
        info["window_len"] = len(self._window)
        info["window_penalty_applied"] = applied_window_penalty and applied_action_window_penalty

        return float(base)
