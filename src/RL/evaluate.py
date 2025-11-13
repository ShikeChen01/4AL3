# run_trading_from_ppo.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from agent import ActorCritic

# -----------------------------
# Data utils
# -----------------------------
def add_missing_flags_and_zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(float)
            df[col] = df[col].fillna(0.0)
    return df

def load_data(path: str | Path, target: str = "forward_returns",
              feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Load CSV/Parquet, rename target->'target', add missing flags, scale features only (NOT target)."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        try:
            df = pd.read_csv(path, engine="pyarrow")
        except Exception:
            df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Got: {list(df.columns)[:10]} ...")

    df = df.rename(columns={target: "target"})
    # Drop known leakage columns if present
    to_drop = [c for c in df.columns if ("market_forward_excess_returns" in c) or ("risk_free_rate" in c)]
    if to_drop:
        df = df.drop(columns=to_drop)

    df = add_missing_flags_and_zero_fill(df)

    feat_cols = feature_cols if feature_cols is not None else [c for c in df.columns if c != "target"]
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])  # target is NOT scaled
    return df, feat_cols

# -----------------------------
# Load model from state_dict
# -----------------------------
def load_actorcritic_from_checkpoint(state_path: str | Path, obs_dim: int,
                                     n_actions: int = 3, l2_lambda: float = 0.0,
                                     map_location: str = "cpu") -> ActorCritic:
    state_dict = torch.load(state_path, map_location=map_location, weights_only=True)
    if "policy_head.weight" not in state_dict:
        raise RuntimeError("Checkpoint missing 'policy_head.weight' – not an ActorCritic state_dict?")
    hidden_w = state_dict["policy_head.weight"].shape[1]

    model = ActorCritic(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden_w, l2_lambda=l2_lambda)
    saved_in = state_dict.get("shared.0.weight", None)
    if saved_in is None:
        raise RuntimeError("Checkpoint missing 'shared.0.weight'.")
    saved_obs_dim = saved_in.shape[1]
    if saved_obs_dim != obs_dim:
        raise RuntimeError(
            f"Input feature mismatch: checkpoint expects obs_dim={saved_obs_dim}, "
            f"but current data has obs_dim={obs_dim}. Align your feature set & preprocessing."
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

# -----------------------------
# Trading simulation
# -----------------------------
@dataclass
class TradeResult:
    dates: Optional[np.ndarray]
    actions: np.ndarray          # -1 sell, 0 hold, +1 buy
    daily_pnl: np.ndarray        # +target for buy, -target for sell, 0 for hold
    cum_pnl: np.ndarray
    hit_rate: float              # sign(action) == sign(target) on non-hold
    hold_ratio: float
    target: np.ndarray           # raw, not normalized target
    target_cum: np.ndarray       # cumulative sum of raw target

def run_trading(df: pd.DataFrame, feat_cols: List[str], model: ActorCritic,
                eps: float = 0.1, date_col: Optional[str] = None) -> TradeResult:
    """
    Decision rule:
      score = logits[+1] - logits[-1]
      if |score| < eps -> hold (0)
      else buy if score>0 -> +1, sell if score<0 -> -1
    PnL per step uses the RAW target (not normalized):
      daily_pnl = {+y, -y, 0} for buy/sell/hold respectively
    """
    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y = df["target"].to_numpy(dtype=np.float32, copy=False)  # RAW target (e.g., S&P500 fwd returns)
    dates = df[date_col].to_numpy() if date_col and date_col in df.columns else None

    actions = np.zeros(len(df), dtype=np.int8)
    daily_pnl = np.zeros(len(df), dtype=np.float32)

    with torch.no_grad():
        for i in range(len(df)):
            x = torch.from_numpy(X[i:i+1])
            logits, _ = model(x)
            log = logits[0]
            score = float(log[2] - log[0])  # buy lane - sell lane
            if abs(score) < eps:
                a = 0
            else:
                a = 1 if score > 0 else -1
            actions[i] = a
            daily_pnl[i] = y[i] if a == 1 else (-y[i] if a == -1 else 0.0)

    cum_pnl = np.cumsum(daily_pnl)
    target_cum = np.cumsum(y)

    non_hold = actions != 0
    hit_rate = float(np.mean(np.sign(actions[non_hold]) == np.sign(y[non_hold]))) if non_hold.any() else float("nan")
    hold_ratio = float(np.mean(actions == 0))

    return TradeResult(dates, actions, daily_pnl, cum_pnl, hit_rate, hold_ratio, target=y, target_cum=target_cum)

# -----------------------------
# Plotting
# -----------------------------
def plot_cum_target_vs_strategy(result: TradeResult, title: str = "Cumulative Return (Raw Target vs Strategy)"):
    plt.figure(figsize=(11, 4))
    if result.dates is not None:
        plt.plot(result.dates, result.target_cum, label="Raw Target Cum Return")
        plt.plot(result.dates, result.cum_pnl, label="Strategy Cum Return")
        plt.xticks(rotation=45)
    else:
        plt.plot(result.target_cum, label="Raw Target Cum Return")
        plt.plot(result.cum_pnl, label="Strategy Cum Return")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main: edit your paths here
# -----------------------------
if __name__ == "__main__":
    # --- EDIT THESE ---
    CKPT_PATH = r"C:\Year4\4AL3\final-project\4AL3\checkpoints\ppo_fold2.pt"
    DATA_PATH = r"C:\Year4\4AL3\final-project\4AL3\result\RL_1\data\train.csv"
    TARGET_COL = "forward_returns"
    DATE_COL: Optional[str] = None   # e.g., "date" if present
    EPS = 0.01
    # ------------------

    df, feat_cols = load_data(DATA_PATH, target=TARGET_COL)
    obs_dim = len(feat_cols)

    model = load_actorcritic_from_checkpoint(CKPT_PATH, obs_dim=obs_dim, n_actions=3, map_location="cpu")

    res = run_trading(df, feat_cols, model, eps=EPS, date_col=DATE_COL)

    # Metrics (target is RAW; MSE between raw y and strategy's per-step return)
    mse = float(np.mean((res.target - res.daily_pnl) ** 2))

    print(f"Trades: buy={(res.actions==1).sum()}  sell={(res.actions==-1).sum()}  hold={(res.actions==0).sum()}")
    print(f"Hit-rate (non-hold): {res.hit_rate:.4f} | Hold ratio: {res.hold_ratio:.4f}")
    print(f"Final cumulative target:  {res.target_cum[-1]:.6f}")
    print(f"Final cumulative strategy:{res.cum_pnl[-1]:.6f}")
    print(f"MSE(target vs strategy per-step returns): {mse:.8f}")

    plot_cum_target_vs_strategy(res, title=f"Cumulative Return (ε={EPS}, raw target vs strategy)")
