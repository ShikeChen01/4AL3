from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from environment import Environment
from rewards import CompetitionMetricReward
from A1 import PPOAgent, PPOConfig
from sklearn.preprocessing import StandardScaler
import joblib

# =========================
# Central config
# =========================
@dataclass
class TrainConfig:
    # Data / features
    data_path: str = r"C:\Year4\4AL3\final-project\4AL3\src\RL\data\train.csv"
    target_col: str = "forward_returns"
    feature_cols: Optional[List[str]] = None  # None => all except target

    # Device / env
    device: str = "cuda"          # "cuda" or "cpu"
    include_bias: bool = False
    threshold: float = 0.0        # reward threshold

    # Saving
    save_dir: str = "./checkpoints"
    save_every: int = 250          # save model every N iterations

    # PPO / Training
    iters: int = 1000
    epochs: int = 5
    batch_size: int = 3000
    minibatch_size: int = 300
    lr: float = 3e-4
    clip_eps: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    l2_lambda: float = 1e-4

    # Cross-validation / Eval
    k_folds: int = 1
    eval_every: int = 100         # evaluate MSE/accuracy every N iterations

    # Plot saving (optional)
    plot_dir: str = "./plots"
    save_plots: bool = True


# =========================
# Argparse -> TrainConfig
# =========================
def parse_args() -> TrainConfig:
    default = TrainConfig()
    p = argparse.ArgumentParser(description="PPO stock prediction training")
    
    # (Arg parsing code omitted for brevity - assume standard boilerplate is same as before)
    # ... standard arguments ...
    
    # Re-using your existing argparse logic from previous file
    p.add_argument("--data-path", type=str, default=default.data_path)
    p.add_argument("--target-col", type=str, default=default.target_col)
    p.add_argument("--feature-cols", type=str, nargs="*", default=None)
    p.add_argument("--device", type=str, default=default.device)
    p.add_argument("--save-dir", type=str, default=default.save_dir)
    p.add_argument("--k-folds", type=int, default=default.k_folds)
    p.add_argument("--eval-every", type=int, default=default.eval_every)
    p.add_argument("--save-plots", action="store_true", default=default.save_plots)
    
    args, unknown = p.parse_known_args()
    cfg_dict = vars(args)
    
    if "no_save_plots" in cfg_dict and cfg_dict["no_save_plots"]:
        cfg_dict["save_plots"] = False
        
    return TrainConfig(**cfg_dict)


# =========================
# Data utilities
# =========================
def add_missing_flags_and_zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(float)
            df[col] = df[col].fillna(0.0)
    return df


def load_data(cfg: TrainConfig) -> Tuple[pd.DataFrame,pd.DataFrame, StandardScaler]:
    path = Path(cfg.data_path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found.")

    df = df.rename(columns={cfg.target_col: "target"})

    # IMPORTANT: Do NOT drop risk_free_rate, it is needed for Sharpe reward
    to_drop = [c for c in df.columns if "market_forward_excess_returns" in c]
    if to_drop:
        df = df.drop(columns=to_drop)

    df = add_missing_flags_and_zero_fill(df)

    feature_cols = cfg.feature_cols if cfg.feature_cols is not None else [
        c for c in df.columns if c != "target"
    ]

    df_raw = df.copy(deep=True)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    df = df.copy(deep=True)
    return df_raw, df, scaler


def save_training_artifacts(cfg: TrainConfig, save_dir: Path, agent: PPOAgent, 
                           ppo_cfg: PPOConfig, scaler: StandardScaler, df: pd.DataFrame, extra=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    resolved_features = cfg.feature_cols if cfg.feature_cols else [c for c in df.columns if c != "target"]
    
    # Get correct action dim
    n_actions_out = agent.model.policy_head.out_features

    run_cfg = {
        "ppo_config": asdict(ppo_cfg),
        "n_actions": n_actions_out,
        "device": str(agent.device),
        "feature_cols": resolved_features,
        "env_meta": extra or {},
    }
    with open(save_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)
    joblib.dump(scaler, save_dir / "scaler.joblib")


# =========================
# PPO rollout
# =========================
def rollout(env: Environment, agent: PPOAgent, steps: int, gamma: float, lam: float) -> Dict[str, np.ndarray]:
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    obs = env.reset()
    for _ in range(steps):
        a_idx, logp, v = agent.act(obs)
        
        # DIRECT PASS: Agent gives index 0..4, Env maps to Leverage 0.0..2.0
        next_obs, r, done, info = env.step(a_idx)

        obs_buf.append(obs.copy())
        act_buf.append(a_idx)
        logp_buf.append(logp)
        rew_buf.append(r)
        val_buf.append(v)
        done_buf.append(done)

        obs = next_obs
        if done:
            obs = env.reset()

    obs_arr = np.asarray(obs_buf, dtype=np.float32)
    acts_arr = np.asarray(act_buf, dtype=np.int64)
    logp_arr = np.asarray(logp_buf, dtype=np.float32)
    rew_arr = np.asarray(rew_buf, dtype=np.float32)
    val_arr = np.asarray(val_buf, dtype=np.float32)
    done_arr = np.asarray(done_buf, dtype=np.bool_)

    next_values = np.concatenate([val_arr[1:], np.array([0.0], dtype=np.float32)])
    deltas = rew_arr + (1.0 - done_arr.astype(np.float32)) * gamma * next_values - val_arr
    adv = np.zeros_like(rew_arr, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rew_arr))):
        nonterminal = 1.0 - float(done_arr[t])
        last_gae = deltas[t] + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + val_arr

    return {"obs": obs_arr, "acts": acts_arr, "logp": logp_arr, "adv": adv, "ret": ret}


# =========================
# Evaluation (deterministic)
# =========================
def evaluate_mse(env: Environment, agent: PPOAgent, scaler: StandardScaler) -> Tuple[float, float]:
    device = agent.device
    y_true: List[float] = []
    y_action_val: List[float] = [] # Actual leverage used

    obs = env.reset()
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent.model(obs_t)
            a_idx = int(torch.argmax(logits, dim=-1).item())
        
        # FIX: No dictionary mapping. Pass index directly.
        # Get the float leverage from env for logging
        action_leverage = float(env.action_map[a_idx])
        
        t = env.t
        y_true.append(float(env.y[t]))
        y_action_val.append(action_leverage)

        obs, _, done, _ = env.step(a_idx)

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_act_arr = np.asarray(y_action_val, dtype=np.float32)

    # MSE is not super useful here (leverage vs return), but calculating for code stability
    mse = float(np.mean((y_act_arr - y_true_arr) ** 2))
    
    # Sign accuracy: Did we hold leverage (>0.1) when return was >0? 
    # Or hold cash (<0.1) when return was <0?
    # Simple proxy:
    pos_ret = y_true_arr > 0
    pos_pos = y_act_arr > 0.1 
    acc = float(np.mean(pos_ret == pos_pos))
    
    return mse, acc


# =========================
# Utilities and Main
# =========================
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_curves_csv(save_dir: Path, base_name: str, hist: Dict[str, List[float]]) -> Tuple[Path, Optional[Path]]:
    save_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.DataFrame({
        "iteration": np.arange(1, len(hist["policy_loss"]) + 1, dtype=int),
        "policy_loss": hist["policy_loss"],
        "value_loss": hist["value_loss"],
        "entropy": hist["entropy"],
        "avg_return": hist["avg_return"],
    })
    train_csv = save_dir / f"{base_name}_training_curves.csv"
    train_df.to_csv(train_csv, index=False)
    
    eval_csv = None
    if hist["mse_eval"]:
        eval_df = pd.DataFrame({
            "iteration": hist["iter_eval"],
            "mse": hist["mse_eval"],
            "sign_accuracy": hist["acc_eval"],
        })
        eval_csv = save_dir / f"{base_name}_eval_curves.csv"
        eval_df.to_csv(eval_csv, index=False)
    return train_csv, eval_csv

def train_on_dataframe(cfg, df, df_raw, fold_tag, verbose=1, eval_env=None, scaler=None):
    device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    
    # CONFIG: Define Action Bins (0x to 2x)
    N_ACTION_BINS = 5 # e.g. 0.0, 0.5, 1.0, 1.5, 2.0

    env = Environment(
        data=df,
        target_col="target",
        feature_cols=cfg.feature_cols,
        reward=CompetitionMetricReward(window_size=252), # Use Sharpe Reward
        include_bias=cfg.include_bias,
        n_action_bins=N_ACTION_BINS 
    )

    ppo_cfg = PPOConfig(
        gamma=cfg.gamma, lam=cfg.lam, clip_eps=cfg.clip_eps, lr=cfg.lr,
        epochs=cfg.epochs, batch_size=cfg.batch_size, minibatch_size=cfg.minibatch_size,
        entropy_coef=cfg.entropy_coef, value_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm, l2_lambda=cfg.l2_lambda,
    )
    
    # Pass correct n_actions to Agent
    agent = PPOAgent(obs_dim=env.obs_dim, n_actions=N_ACTION_BINS, device=device, cfg=ppo_cfg)

    hist = {"policy_loss": [], "value_loss": [], "entropy": [], "avg_return": [], "mse_eval": [], "acc_eval": [], "iter_eval": []}

    for it in range(cfg.iters):
        buf = rollout(env, agent, steps=ppo_cfg.batch_size, gamma=ppo_cfg.gamma, lam=ppo_cfg.lam)
        stats = agent.update(buf)
        avg_ret = float(buf["ret"].mean())

        hist["policy_loss"].append(stats["policy_loss"])
        hist["value_loss"].append(stats["value_loss"])
        hist["entropy"].append(stats["entropy"])
        hist["avg_return"].append(avg_ret)

        if (it + 1) % verbose == 0:
            print(f"[{fold_tag}] Iter {it+1}/{cfg.iters} | Ret: {avg_ret:.4f} | P_Loss: {stats['policy_loss']:.4f}")

        if eval_env and scaler and (it + 1) % cfg.eval_every == 0:
            mse, acc = evaluate_mse(eval_env, agent, scaler)
            hist["mse_eval"].append(mse)
            hist["acc_eval"].append(acc)
            hist["iter_eval"].append(it + 1)
            print(f"[{fold_tag}] Eval {it+1}: MSE(lev vs ret)={mse:.5f} | Dir-Acc={acc:.4f}")

        if (it + 1) % cfg.save_every == 0:
            save_dir = ensure_dir(cfg.save_dir)
            torch.save(agent.model.state_dict(), str(Path(save_dir) / f"ppo_{fold_tag.lower()}_iter{it+1}.pt"))

    return agent, stats, hist

def main():
    cfg = parse_args()
    save_dir = ensure_dir(cfg.save_dir)
    device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Config: {cfg}")

    df_raw, full_df, scaler = load_data(cfg)

    if cfg.k_folds <= 1:
        # For evaluation, we use CompetitionMetricReward too to see real performance, or SimpleSign
        env_eval = Environment(
            full_df, "target", cfg.feature_cols, 
            reward=CompetitionMetricReward(), 
            include_bias=cfg.include_bias,
            n_action_bins=5 # Match training
        )
        agent, _, hist = train_on_dataframe(cfg, full_df, df_raw, "FULL", eval_env=env_eval, scaler=scaler)
        save_curves_csv(save_dir, "full", hist)
        torch.save(agent.model.state_dict(), str(save_dir / "ppo_full.pt"))
        return

    # K-Fold
    # ... (Logic is similar, just ensure n_action_bins is consistent) ...
    print("Please run with k_folds=1 for testing basic functionality first.")

if __name__ == "__main__":
    main()