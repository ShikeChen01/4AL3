from __future__ import annotations
from typing import Protocol, Dict, Any
import numpy as np
from collections import deque

class Reward(Protocol):
    def reset(self) -> None: ...
    def __call__(self, *, t: int, action: int, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float: ...

class CompetitionMetricReward:
    """
    Calculates the exact volatility-adjusted Sharpe ratio from the user provided scoring function.
    It uses a rolling window to estimate the current Sharpe and penalties.
    """
    def __init__(
        self,
        window_size: int = 252,  # Annualized window (approx 1 trading year)
        trading_days_per_yr: int = 252
    ) -> None:
        self.window_size = window_size
        self.trading_days = trading_days_per_yr
        
        # Buffers to store history for sharpe calculation
        self._strat_returns = deque(maxlen=window_size)
        self._market_returns = deque(maxlen=window_size)
        self._risk_free_rates = deque(maxlen=window_size)

    def reset(self) -> None:
        self._strat_returns.clear()
        self._market_returns.clear()
        self._risk_free_rates.clear()

    def __call__(self, *, t: int, action: float, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float:
        # Note: 'action' here is the raw position (e.g. 0.0 to 2.0), not the index.
        # 'target' is the forward_return of the asset (Market Return).
        
        risk_free = info.get("risk_free_rate", 0.0)
        
        # 1. Calculate Strategy Return for this step
        # Formula: rf * (1 - pos) + pos * market_ret
        # Simplified: rf + pos * (market_ret - rf)
        strat_ret = risk_free * (1 - action) + action * target
        
        self._strat_returns.append(strat_ret)
        self._market_returns.append(target)
        self._risk_free_rates.append(risk_free)

        # If window is too small, just return raw strategy return to encourage learning early
        if len(self._strat_returns) < 10:
            return strat_ret * 100  # Scale up slightly

        # 2. Convert to Numpy for Vectorized Math
        s_ret = np.array(self._strat_returns)
        m_ret = np.array(self._market_returns)
        rf = np.array(self._risk_free_rates)

        # 3. Calculate Strategy Metrics
        strat_excess = s_ret - rf
        # Geometric Mean approx for stability in RL step: using arithmetic for speed or standard log return
        # The competition code uses Geometric: (1+r).prod()^(1/n) - 1
        strat_excess_cum = np.prod(1 + strat_excess)
        strat_mean_excess = (strat_excess_cum) ** (1 / len(s_ret)) - 1
        strat_std = np.std(s_ret) 

        if strat_std == 0:
            return 0.0 # Avoid div by zero
            
        sharpe = strat_mean_excess / strat_std * np.sqrt(self.trading_days)
        strat_vol = strat_std * np.sqrt(self.trading_days) * 100

        # 4. Calculate Market Metrics
        market_excess = m_ret - rf
        market_excess_cum = np.prod(1 + market_excess)
        market_mean_excess = (market_excess_cum) ** (1 / len(m_ret)) - 1
        market_std = np.std(m_ret)
        market_vol = market_std * np.sqrt(self.trading_days) * 100

        # 5. Volatility Penalty
        if market_vol == 0:
            excess_vol = 0
        else:
            excess_vol = max(0, strat_vol / market_vol - 1.2)
        vol_penalty = 1 + excess_vol

        # 6. Return Penalty
        return_gap = max(0, (market_mean_excess - strat_mean_excess) * 100 * self.trading_days)
        return_penalty = 1 + (return_gap**2) / 100

        # 7. Final Adjusted Score
        adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
        
        # Clip reward to prevent exploding gradients in PPO
        return np.clip(adjusted_sharpe, -10, 10)