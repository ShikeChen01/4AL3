from __future__ import annotations
from typing import Protocol, Dict, Any
import numpy as np
from collections import deque
import math

class Reward(Protocol):
    def reset(self) -> None: ...
    def __call__(self, *, t: int, action: float, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float: ...

class CompetitionMetricReward:
    """
    Calculates the volatility-adjusted Sharpe ratio.
    Includes NaN protection for when strategies go bankrupt (return < -100%).
    """
    def __init__(
        self,
        window_size: int = 252,
        trading_days_per_yr: int = 252
    ) -> None:
        self.window_size = window_size
        self.trading_days = trading_days_per_yr
        
        self._strat_returns = deque(maxlen=window_size)
        self._market_returns = deque(maxlen=window_size)
        self._risk_free_rates = deque(maxlen=window_size)

    def reset(self) -> None:
        self._strat_returns.clear()
        self._market_returns.clear()
        self._risk_free_rates.clear()

    def __call__(self, *, t: int, action: float, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float:
        risk_free = info.get("risk_free_rate", 0.0)
        
        # 1. Calculate Strategy Return
        strat_ret = risk_free * (1 - action) + action * target
        
        self._strat_returns.append(strat_ret)
        self._market_returns.append(target)
        self._risk_free_rates.append(risk_free)

        # Warmup phase
        if len(self._strat_returns) < 2:
            return 0.0

        # 2. Convert to Numpy
        s_ret = np.array(self._strat_returns)
        m_ret = np.array(self._market_returns)
        rf = np.array(self._risk_free_rates)

        # 3. Strategy Metrics
        strat_excess = s_ret - rf
        
        # --- FIX: Clip negative bases to avoid NaN ---
        # If (1 + return) < 0, it means bankruptcy. We clip to epsilon (near zero).
        s_term = np.maximum(1 + strat_excess, 1e-9)
        strat_excess_cum = np.prod(s_term)
        
        # Geometric Mean
        strat_mean_excess = (strat_excess_cum) ** (1 / len(s_ret)) - 1
        strat_std = np.std(s_ret) 

        # Avoid div by zero
        if strat_std < 1e-9:
            return 0.0
            
        sharpe = strat_mean_excess / strat_std * np.sqrt(self.trading_days)
        strat_vol = strat_std * np.sqrt(self.trading_days) * 100

        # 4. Market Metrics
        market_excess = m_ret - rf
        
        # --- FIX: Clip market negative bases too ---
        m_term = np.maximum(1 + market_excess, 1e-9)
        market_excess_cum = np.prod(m_term)
        
        market_mean_excess = (market_excess_cum) ** (1 / len(m_ret)) - 1
        market_std = np.std(m_ret)
        market_vol = market_std * np.sqrt(self.trading_days) * 100

        # 5. Penalties
        if market_vol == 0:
            excess_vol = 0
        else:
            excess_vol = max(0, strat_vol / market_vol - 1.2)
        vol_penalty = 1 + excess_vol

        return_gap = max(0, (market_mean_excess - strat_mean_excess) * 100 * self.trading_days)
        return_penalty = 1 + (return_gap**2) / 100

        # 6. Final Calculation
        adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
        
        # Final Safety Check against any rogue NaNs
        if np.isnan(adjusted_sharpe) or np.isinf(adjusted_sharpe):
            return 0.0
            
        return np.clip(adjusted_sharpe, -10, 10)


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

    def __call__(self, *, t: int, action: float, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float:
        # Note: action is a float leverage (e.g. 0.0, 1.0, 2.0). 
        # If using standard leverage, action is always >= 0.
        if abs(target) < self.threshold:
            return self.correct_reward if action == 0 else self.wrong_penalty
        
        sign_t = 1 if target > 0 else -1
        
        # If action is leverage (0 to 2), it will likely never be exactly -1.
        # This logic works best if action is mapped to {-1, 0, 1} or if we just check sign direction.
        # Assuming strict equality based on user request:
        return self.correct_reward if action == sign_t else self.wrong_penalty


class CustomReward:
    """Placeholder to be filled by the user if desired."""
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def reset(self) -> None:
        pass

    def __call__(self, *, t: int, action: float, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float:
        return 0.0


class WindowedSignReward:
    """
    Sign-based reward with:
    - penalty for 'hold' actions
    - extra penalty if performance over a window is not positive.

    Actions: -1, 0, +1 (or leverage amounts)
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

    def __call__(self, *, t: int, action: float, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float:     
        base = self.next_base
        base += action * target
        self._action_window.append(action)
        
        # --- base sign reward ---
        if abs(target) < self.threshold:
            base += self.correct_reward if action == 0 else self.wrong_penalty
        else:
            # np.sign(0.0) is 0. np.sign(0.5) is 1. np.sign(-0.5) is -1.
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
        self.cur_window_penalty = float(0.0)
        self.cur_action_window_penalty = float(0.0)

    def reset(self) -> None:
        self._window.clear()
        self._action_window.clear()
        self.cur_window_penalty = float(0.0)
        self.cur_action_window_penalty = float(0.0)

    def __call__(self, *, t: int, action: float, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float:  
        # Debug print disabled to avoid spamming console during training, 
        # enable if needed: print("Target before scaling:", target)  
        
        target_scaled = target * 100 
        base = 0.0
        base += action * target_scaled
        
        self._window.append(base)
        self._action_window.append(action)

        # --- base sign reward ---
        if abs(target_scaled) < self.threshold:
            base += self.correct_reward if action == 0 else self.wrong_penalty
        else:
            base += self.correct_reward if np.sign(action) == np.sign(target_scaled) else self.wrong_penalty 

        # --- window logic ---
        window_sum = sum(self._window)

        applied_window_penalty = False
        if len(self._window) == self.window_size and window_sum <= 0.0:
            self.cur_window_penalty += self.window_penalty
            applied_window_penalty = True
        else:
            self.cur_window_penalty = 0.0
        
        # Calculate average action over window
        avg_actions = abs(sum(self._action_window) / len(self._action_window)) if self._action_window else 0.0
        applied_action_window_penalty = False

        if avg_actions <= 1 and len(self._action_window) == self.action_window_size and self.cur_action_window_penalty > self.action_penality_threshold:
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