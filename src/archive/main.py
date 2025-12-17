# main_predict.py
from __future__ import annotations
import argparse, os
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Your files
from RL.environment import PredictionEnv, SimpleNegMSE   
# reward = -(pred-target)^2  :contentReference[oaicite:2]{index=2}
# RL_framework.Env is used by PredictionEnv’s interface                          :contentReference[oaicite:3]{index=3}

# --------- Simple MLP regressor (scalar output) ----------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,) scalar predictions


def make_env(df: pd.DataFrame,
             target_col: str,
             feature_cols: Optional[List[str]],
             start_index: int,
             clip: Optional[tuple[float, float]]) -> PredictionEnv:
    reward = SimpleNegMSE()  # swap in your custom Reward later
    return PredictionEnv(
        data=df,
        target_col=target_col,
        feature_cols=feature_cols,
        start_index=start_index,
        reward=reward,
        action_clip=clip,
    )  # :contentReference[oaicite:4]{index=4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV")
    p.add_argument("--target", default="target", help="Target column name")
    p.add_argument("--features", default="", help="Comma-separated feature columns (empty = all except target)")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--l2", type=float, default=1e-4, help="L2 weight decay")
    p.add_argument("--clip_lo", type=float, default=None, help="Optional lower bound for predictions")
    p.add_argument("--clip_hi", type=float, default=None, help="Optional upper bound for predictions")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--standardize", action="store_true", help="Z-score features per column")
    args = p.parse_args()

    # --------- Device ----------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # --------- Load data ----------
    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Missing target col: {args.target}")

    if args.features.strip():
        feature_cols = [c.strip() for c in args.features.split(",")]
    else:
        feature_cols = [c for c in df.columns if c != args.target]

    # Optional standardization (fit on full file for simplicity)
    if args.standardize:
        mu = df[feature_cols].mean(axis=0)
        std = df[feature_cols].std(axis=0).replace(0.0, 1.0)
        df[feature_cols] = (df[feature_cols] - mu) / std

    # Optional pred clipping
    clip = None
    if args.clip_lo is not None and args.clip_hi is not None:
        clip = (float(args.clip_lo), float(args.clip_hi))

    # --------- Env & model ----------
    env = make_env(df, args.target, feature_cols, args.start_index, clip)
    obs = env.reset()
    obs_dim = int(np.asarray(obs, dtype=np.float32).shape[-1])

    model = MLP(in_dim=obs_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)  # L2 regularizer

    # --------- Training loop ----------
    # We run sequentially over the episode; loss = -reward (since reward = -MSE by default).
    for ep in range(1, args.epochs + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        total_steps = 0

        # Accumulate step losses, then backprop once per episode (stable & simple)
        step_losses: list[torch.Tensor] = []

        while not done:
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
            pred = model(x)              # scalar prediction \hat{y}_t
            a = float(pred.detach().cpu().item())  # action = prediction to feed env

            # env computes reward via Reward policy (e.g., -(pred-target)^2)
            next_obs, r, done, info = env.step(a)

            # Convert reward to loss: maximize reward <=> minimize (-reward)
            step_losses.append((-torch.tensor([r], dtype=torch.float32, device=device)))

            total_reward += float(r)
            total_steps += 1
            obs = next_obs

        # Backprop once per episode
        opt.zero_grad()
        loss = torch.mean(torch.stack(step_losses))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        print(f"Epoch {ep:03d} | steps={total_steps} | total_reward={total_reward:.6f} | loss={loss.item():.6f}")

    # --------- Save weights ----------
    out_path = os.path.splitext(os.path.basename(args.data))[0] + "_predict_env.pt"
    torch.save({"model_state": model.state_dict(),
                "obs_dim": obs_dim,
                "feature_cols": feature_cols,
                "target_col": args.target}, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
