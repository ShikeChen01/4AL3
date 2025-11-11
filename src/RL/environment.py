from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Sequence, List
import numpy as np
import pandas as pd
from rewards import Reward, SimpleSignReward


class Environment:
    """
    Stock *prediction* environment.
    Action space: {-1, 0, +1} meaning predict "down / flat / up".
    Observation: feature vector at time t.
    Reward: injected via Reward policy (default: SimpleSignReward).
    Episode ends at the end of the dataframe.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
        start_index: int = 0,
        reward: Reward = SimpleSignReward(threshold=0.0),
        include_bias: bool = False,
    ) -> None:
        assert target_col in data.columns, f"Missing target column: {target_col}"
        self.df = data.reset_index(drop=True)
        self.target_col = target_col
        self.feature_cols = (
            feature_cols
            if feature_cols is not None
            else [c for c in self.df.columns if c != target_col]
        )
        self.X = self.df[self.feature_cols].to_numpy(dtype=float)
        if include_bias:
            self.X = np.concatenate([self.X, np.ones((self.X.shape[0], 1))], axis=1)
        self.y = self.df[self.target_col].to_numpy(dtype=float)

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
        return (-1, 0, 1)

    def step(self, a: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        a = int(np.clip(a, -1, 1))
        obs_t = self._obs()
        target_t = float(self.y[self.t])

        info: Dict[str, Any] = {"t": int(self.t), "target": target_t, "action": a}
        r = float(self.reward(t=self.t, action=a, target=target_t, obs=obs_t, info=info))

        self.cum_reward += r
        self.t += 1
        done = self.t >= self.n_steps
        next_obs = self._terminal_obs() if done else self._obs()
        info.update({"reward": r, "cum_reward": self.cum_reward, "done": done})
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
