from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Sequence, List
import numpy as np
import pandas as pd
from rewards import Reward, CompetitionMetricReward

class Environment:
    """
    Stock environment supporting Leverage (0x to 2x) and No Shorting.
    Action space: Discrete indices mapped to [0.0, 2.0].
    """
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
        start_index: int = 0,
        reward: Optional[Reward] = None,
        include_bias: bool = False,
        n_action_bins: int = 5, # Number of discrete steps between 0 and 2
        dtype: np.dtype = np.float32,
    ) -> None:
        self.df = data.reset_index(drop=True)
        self.target_col = target_col
        self.feature_cols = feature_cols if feature_cols else [c for c in self.df.columns if c != target_col]

        # --- Create Action Mapping (0 to 2) ---
        # e.g. 5 bins -> [0.0, 0.5, 1.0, 1.5, 2.0]
        self.action_map = np.linspace(0.0, 2.0, n_action_bins)
        self.n_action_bins = n_action_bins

        # Data Prep
        X = self.df[self.feature_cols].to_numpy(dtype=dtype, copy=False)
        if include_bias:
            bias = np.ones((X.shape[0], 1), dtype=dtype)
            X = np.concatenate([X, bias], axis=1)
        self.X = np.ascontiguousarray(X)
        self.y = np.ascontiguousarray(self.df[self.target_col].to_numpy(dtype=dtype, copy=False))
        
        # Handle Risk Free Rate if present (needed for Sharpe)
        if "risk_free_rate" in self.df.columns:
            self.rf = self.df["risk_free_rate"].to_numpy(dtype=dtype, copy=False)
        else:
            self.rf = np.zeros(len(self.df), dtype=dtype)

        self.n_steps = len(self.df)
        self.start_index = int(start_index)
        self.t = 0
        self.cum_reward = 0.0
        self.reward = reward if reward is not None else CompetitionMetricReward()

    def reset(self) -> np.ndarray:
        self.t = self.start_index
        self.cum_reward = 0.0
        self.reward.reset()
        return self._obs()

    def step(self, a_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        t = self.t
        n_steps = self.n_steps
        
        # --- Map Discrete Index to Leverage Amount ---
        # Clip index just in case
        a_idx = max(0, min(a_idx, self.n_action_bins - 1))
        action_val = float(self.action_map[a_idx]) # e.g. 0.5 or 2.0

        obs_t = self.X[t]
        target_t = float(self.y[t])
        rf_t = float(self.rf[t])

        info: Dict[str, Any] = {
            "t": int(t), 
            "target": target_t, 
            "action": action_val, 
            "risk_free_rate": rf_t
        }
        
        # Calculate reward using the mapped float action
        r = float(self.reward(t=t, action=action_val, target=target_t, obs=obs_t, info=info))

        cum = self.cum_reward + r
        t_next = t + 1
        done = t_next >= n_steps

        if done:
            next_obs = self.X[n_steps - 1]
        else:
            next_obs = self.X[t_next]

        self.t = t_next
        self.cum_reward = cum
        
        return next_obs, r, done, info

    def _obs(self) -> np.ndarray:
        idx = min(self.t, self.n_steps - 1)
        return self.X[idx]

    @property
    def obs_dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_actions(self) -> int:
        return self.n_action_bins