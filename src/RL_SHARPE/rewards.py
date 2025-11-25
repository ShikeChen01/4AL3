from __future__ import annotations
from typing import Protocol, Dict, Any
import numpy as np
from collections import deque

class Reward(Protocol):
    def reset(self) -> None: ...
    def __call__(self, *, t: int, action: int, target: float, obs: np.ndarray, info: Dict[str, Any]) -> float: ...

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
        if len(self._strat_returns) < 5:
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
            
        return np.clip(adjusted_sharpe, -100, 100)