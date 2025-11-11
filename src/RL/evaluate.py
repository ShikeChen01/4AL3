from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib

from environment import Environment
from rewards import SimpleSignReward
from agent import PPOAgent, PPOConfig
from sklearn.preprocessing import StandardScaler

# ------------- argparse ON TOP -------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="CSV/Parquet with features + target")
parser.add_argument("--target_col", type=str, default="forward_returns")
parser.add_argument("--feature_cols", type=str, nargs="*", default=None)
parser.add_argument("--weights", type=str, required=True, help="Path to trained ppo .pt")
parser.add_argument("--include_bias", action="store_true", default=False)
parser.add_argument("--threshold", type=float, default=0.0)
parser.add_argument("--device", type=str, choices=["cuda","cpu"], default="cuda")
parser.add_argument("--out_png", type=str, default="./equity_curve.png")
args = parser.parse_args()

DATA_PATH     = args.data_path
TARGET_COL    = args.target_col
FEATURE_COLS  = args.feature_cols
WEIGHTS_PATH  = args.weights
INCLUDE_BIAS  = args.include_bias
THRESHOLD     = args.threshold
DEVICE        = args.device
OUT_PNG       = args.out_png


def load_df_and_scaler(path: str | Path, target_col: str) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        try:
            df = pd.read_csv(path, engine="pyarrow")
        except Exception:
            df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found.")
    df = df.rename(columns={target_col: "target"})

    # prefer artifacts’ feature order if available

    feat_order = [c for c in df.columns if c != "target"] if FEATURE_COLS is None else FEATURE_COLS

    # scaler

    scaler = StandardScaler()
    df[feat_order] = scaler.fit_transform(df[feat_order])

    return df, scaler, feat_order


def deterministic_signals(env: Environment, agent: PPOAgent) -> np.ndarray:
    """Return actions in {-1, 0, +1} deterministically (argmax)."""
    device = agent.device
    obs = env.reset()
    done = False
    actions = []
    while not done:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = agent.model(obs_t)
            a_idx = int(torch.argmax(logits, dim=-1).item())
        idx_to_action = {0: -1, 1: 0, 2: 1}
        a_env = idx_to_action[a_idx]
        actions.append(a_env)
        obs, _, done, _ = env.step(a_env)
    return np.asarray(actions, dtype=np.int8)


def main():
    df, scaler, feat_cols = load_df_and_scaler(DATA_PATH, TARGET_COL)

    # Build env with same features
    env = Environment(
        data=df,
        target_col="target",
        feature_cols=feat_cols,
        reward=SimpleSignReward(threshold=THRESHOLD),
        include_bias=INCLUDE_BIAS,
    )

    # Dummy PPO config (only architecture matters at inference)
    cfg = PPOConfig()
    device = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    agent = PPOAgent(obs_dim=env.obs_dim, n_actions=3, device=device, cfg=cfg)
    agent.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    agent.model.eval()

    # Signals and returns (shift signals by 1 to avoid lookahead)
    acts = deterministic_signals(env, agent)  # length N
    ret = df["target"].to_numpy(dtype=np.float64)  # S&P 500 period returns (e.g., daily)
    sig = np.sign(acts).astype(np.int8)            # {-1,0,+1}
    sig_shift = np.roll(sig, 1); sig_shift[0] = 0  # enter next period

    strat_ret = sig_shift * ret
    # Equity curves (1 + r) cumulative product
    bh_equity = np.cumprod(1.0 + ret)
    strat_equity = np.cumprod(1.0 + strat_ret)

    # Plot
    plt.figure()
    plt.plot(bh_equity, label="Buy & Hold (S&P 500)")
    plt.plot(strat_equity, label="Strategy (PPO signals)")
    plt.title("Accumulated Return Over Time")
    plt.xlabel("Time")
    plt.ylabel("Equity (Growth of $1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, bbox_inches="tight")
    plt.show()
    print(f"Saved: {OUT_PNG}")

if __name__ == "__main__":
    main()
