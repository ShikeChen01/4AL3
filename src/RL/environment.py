from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Sequence, List
import numpy as np
import pandas as pd
from rewards import Reward, SimpleSignReward, WindowedSignReward


class Environment:
    """
    Stock *prediction* environment.
    Action space: {-1, 0, +1} meaning predict "down / flat / up".
    Observation: feature vector at time t.
    Reward: injected via Reward policy (default: SimpleSignReward).
    Episode ends at the end of the dataframe.
    """
    ACTIONS: Tuple[int, int, int] = (-1, 0, 1)

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
        start_index: int = 0,
        reward: Reward = WindowedSignReward(threshold=0.0, wrong_penalty=-1.0, correct_reward=1.0, window_size=60),
        include_bias: bool = False,
        dtype: np.dtype = np.float32,  # new, default faster dtype
    ) -> None:
        assert target_col in data.columns, f"Missing target column: {target_col}"
        # Keep behavior: reset index to 0..N-1
        self.df = data.reset_index(drop=True)
        self.target_col = target_col
        self.feature_cols = (
            feature_cols
            if feature_cols is not None
            else [c for c in self.df.columns if c != target_col]
        )

        # Use float32 by default for speed / memory
        X = self.df[self.feature_cols].to_numpy(dtype=dtype, copy=False)
        if include_bias:
            bias = np.ones((X.shape[0], 1), dtype=dtype)
            X = np.concatenate([X, bias], axis=1)
        self.X = np.ascontiguousarray(X)

        self.y = np.ascontiguousarray(
            self.df[self.target_col].to_numpy(dtype=dtype, copy=False)
        )

        self.n_steps = len(self.df)
        self.start_index = int(start_index)

        # Runtime
        self.t: int = 0
        self.cum_reward: float = 0.0

        self.reward = reward

    # -------- RL-like interface --------
    def reset(self) -> np.ndarray:
        self.t = self.start_index
        self.cum_reward = 0.0
        self.reward.reset()
        return self._obs()

    def actions(self, s: Optional[np.ndarray] = None) -> Sequence[int]:
        # Reuse a shared tuple instead of allocating each call
        return self.ACTIONS

    def step(self, a: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Faster version of the same logic:
        - manual clip for action
        - inline obs/terminal_obs path
        - fewer attribute lookups
        """
        # --- local bindings for speed ---
        t = self.t
        n_steps = self.n_steps
        X = self.X
        y = self.y
        reward_fn = self.reward

        # clip action to {-1, 0, +1}
        a_int = int(a)
        if a_int < -1:
            a_int = -1
        elif a_int > 1:
            a_int = 1

        obs_t = X[t]
        target_t = float(y[t])

        info: Dict[str, Any] = {"t": int(t), "target": target_t, "action": a_int}
        r = float(
            reward_fn(t=t, action=a_int, target=target_t, obs=obs_t, info=info)
        )

        cum = self.cum_reward + r
        t_next = t + 1
        done = t_next >= n_steps

        # Next observation: last row if done, otherwise next row
        if done:
            next_obs = X[n_steps - 1]
        else:
            next_obs = X[t_next]

        # write back state
        self.t = t_next
        self.cum_reward = cum

        info["reward"] = r
        info["cum_reward"] = cum
        info["done"] = done

        return next_obs, r, done, info

    # -------- internals --------
    def _obs(self) -> np.ndarray:
        idx = min(self.t, self.n_steps - 1)
        return self.X[idx]

    def _terminal_obs(self) -> np.ndarray:
        return self.X[self.n_steps - 1]

    @property
    def obs_dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_actions(self) -> int:
        return 3  # -1, 0, +1 mapped to indices (0,1,2)
