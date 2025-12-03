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
from rewards import WindowedSignReward_v2
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

    # Saving
    save_dir: str = r"C:\Year4\4AL3\final-project\4AL3\src\RL_SHARPE\checkpoints"
    save_every: int = 250          # save model every N iterations

    # PPO / Training
    iters: int = 1000
    epochs: int = 5
    batch_size: int = 3000
    minibatch_size: int = 300
    lr: float = 1e-4
    clip_eps: float = 0.2
    gamma: float = 0.99
    lam: float = 0.9
    entropy_coef: float = 0.05
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    l2_lambda: float = 1e-4

    # Cross-validation / Eval
    eval_every: int = 100         # evaluate every N iterations
    test_size: int = 500          # Number of rows to drop for final test
    ignore_first_n: int = 3000      # Rows to ignore at start for training/eval

    # Plot saving
    plot_dir: str = "./plots"
    save_plots: bool = True

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
    
def parse_args() -> TrainConfig:
    default = TrainConfig()
    p = argparse.ArgumentParser(description="PPO stock prediction training")
    p.add_argument("--data-path", type=str, default=default.data_path)
    p.add_argument("--device", type=str, default=default.device)
    p.add_argument("--save-plots", action="store_true", default=default.save_plots)
    args, _ = p.parse_known_args()
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

def load_data(cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    path = Path(cfg.data_path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    df = df.rename(columns={cfg.target_col: "target"})

    # Drop leakage
    to_drop = [c for c in df.columns if "market_forward_excess_returns" in c]
    if to_drop:
        df = df.drop(columns=to_drop)

    df = add_missing_flags_and_zero_fill(df)

    # 1. Update Config to Exclude risk_free_rate
    if cfg.feature_cols is None:
        cfg.feature_cols = [c for c in df.columns if c != "target" and c != "risk_free_rate"]
    else:
        cfg.feature_cols = [c for c in cfg.feature_cols if c != "risk_free_rate"]

    # 2. Scale
    df_raw = df.copy(deep=True)
    scaler = StandardScaler()
    df[cfg.feature_cols] = scaler.fit_transform(df[cfg.feature_cols])
    df[cfg.feature_cols] = df[cfg.feature_cols].fillna(0.0)
    
    # 3. Split: Train vs Test (Held-out last 500)
    if len(df) > cfg.test_size:
        print(f"Splitting data: Dropping last {cfg.test_size} rows for testing.")
        test_df = df.iloc[-cfg.test_size:].reset_index(drop=True)
        train_df = df.iloc[:-cfg.test_size].reset_index(drop=True)
    else:
        print("Warning: Data too small to split. Using full data for train and test.")
        train_df = df
        test_df = df
    
    # 4. Drop first N rows if specified
    if cfg.ignore_first_n > 0:
        print(f"Dropping first {cfg.ignore_first_n} rows from train and test sets.")
        train_df = train_df.iloc[cfg.ignore_first_n:].reset_index(drop=True)
        test_df = test_df.iloc[cfg.ignore_first_n:].reset_index(drop=True)

    return df_raw, train_df, test_df, scaler


def save_training_artifacts(cfg: TrainConfig, save_dir: Path, agent: PPOAgent, 
                           ppo_cfg: PPOConfig, scaler: StandardScaler):
    save_dir.mkdir(parents=True, exist_ok=True)
    n_actions_out = agent.model.policy_head.out_features
    run_cfg = {
        "ppo_config": asdict(ppo_cfg),
        "n_actions": n_actions_out,
        "device": str(agent.device),
        "feature_cols": cfg.feature_cols,
    }
    with open(save_dir / "run_config.json", "w") as f:
        json.dump(run_cfg, f, indent=2)
    joblib.dump(scaler, save_dir / "scaler.joblib")


# =========================
# PPO Rollout
# =========================
def rollout(env: Environment, agent: PPOAgent, steps: int, gamma: float, lam: float) -> Dict[str, np.ndarray]:
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    h_buf, c_buf = [], [] 

    obs = env.reset()
    # Reset Agent's internal LSTM state for the new episode
    if hasattr(agent, "reset_hidden_state"):
        agent.reset_hidden_state()

    for _ in range(steps):
        # agent.act uses internal state. Returns (h, c) just for storage.
        a_idx, logp, v, (h, c) = agent.act(obs)
        
        next_obs, r, done, info = env.step(a_idx)

        obs_buf.append(obs.copy())
        act_buf.append(a_idx)
        logp_buf.append(logp)
        rew_buf.append(r)
        val_buf.append(v)
        done_buf.append(done)
        
        # Store states for the Update function (required for block shuffling)
        h_buf.append(h)
        c_buf.append(c)

        obs = next_obs
        if done:
            obs = env.reset()
            if hasattr(agent, "reset_hidden_state"):
                agent.reset_hidden_state()

    # Convert to Numpy
    # ... (Standard GAE calculation) ...
    obs_arr = np.asarray(obs_buf, dtype=np.float32)
    acts_arr = np.asarray(act_buf, dtype=np.int64)
    logp_arr = np.asarray(logp_buf, dtype=np.float32)
    rew_arr = np.asarray(rew_buf, dtype=np.float32)
    val_arr = np.asarray(val_buf, dtype=np.float32)
    done_arr = np.asarray(done_buf, dtype=np.bool_)
    h_arr_cpu = torch.stack(h_buf).cpu().numpy()
    c_arr_cpu = torch.stack(c_buf).cpu().numpy()

    next_values = np.concatenate([val_arr[1:], np.array([0.0], dtype=np.float32)])
    deltas = rew_arr + (1.0 - done_arr.astype(np.float32)) * gamma * next_values - val_arr
    adv = np.zeros_like(rew_arr, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rew_arr))):
        nonterminal = 1.0 - float(done_arr[t])
        last_gae = deltas[t] + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    ret = adv + val_arr

    h_arr = torch.stack(h_buf)
    c_arr = torch.stack(c_buf)

    return {
        "obs": obs_arr, "acts": acts_arr, "logp": logp_arr, 
        "adv": adv, "ret": ret, "rew": rew_arr,
        "hidden_h": h_arr, "hidden_c": c_arr
    }


# =========================
# Evaluation
# =========================
def run_evaluation(env: Environment, agent: PPOAgent, desc: str = "Eval") -> Dict[str, float]:
    """
    Runs one full pass over the environment.
    Calculates Sharpe Ratio, MSE, and Direction Accuracy.
    """
    device = agent.device
    obs = env.reset()
    
    # Reset LSTM state for evaluation
    if hasattr(agent, "reset_hidden_state"):
        agent.reset_hidden_state()

    strat_returns = []
    y_true = []
    y_pred = []

    done = False
    while not done:
        # 1. Deterministic Action Selection
        # We use the model directly to get logits. Model handles state internally.
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent.model(obs_t) # Uses & updates internal state
            a_idx = int(torch.argmax(logits, dim=-1).item())
        
        # 2. Step
        lev = float(env.action_map[a_idx])
        next_obs, _, done, info = env.step(a_idx)
        
        # 3. Metrics Calculation
        target = info.get('target', 0.0) # Market Return
        rf = info.get('risk_free_rate', 0.0)
        
        # Strategy Return = RF * (1 - Lev) + Lev * Market
        # Excess Return = Strategy Return - RF
        strat_ret = rf * (1 - lev) + lev * target
        excess_ret = strat_ret - rf
        
        strat_returns.append(excess_ret)
        y_true.append(target)
        y_pred.append(lev)
        
        obs = next_obs

    # Summary Metrics
    excess_arr = np.array(strat_returns)
    mean_excess = np.mean(excess_arr)
    std_excess = np.std(excess_arr)
    
    if std_excess < 1e-9:
        sharpe = 0.0
    else:
        sharpe = (mean_excess / std_excess) * np.sqrt(252)

    # MSE (Leverage vs Target - purely for debug)
    mse = np.mean((np.array(y_pred) - np.array(y_true))**2)
    
    # Accuracy (Direction)
    pos_mkt = np.array(y_true) > 0
    pos_lev = np.array(y_pred) > 0.1
    acc = np.mean(pos_mkt == pos_lev)

    return {"sharpe": sharpe, "mse": mse, "acc": acc}


# =========================
# Plotting
# =========================
def plot_series(cfg: TrainConfig, y: List[float], title: str, x: Optional[List[int]] = None, save_name: Optional[str] = None):
    if not y: return
    if x is None: x = list(range(1, len(y) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(title)
    plt.grid(True, alpha=0.3)
    
    if cfg.save_plots and save_name is not None:
        save_path = Path(cfg.plot_dir) / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close()


# =========================
# Main Training Loop
# =========================
def train_on_dataframe(cfg, train_df, test_df):
    device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    N_ACTION_BINS = 5 

    # 1. Training Environment
    env_train = Environment(
        data=train_df,
        target_col="target",
        feature_cols=cfg.feature_cols, 
        reward=WindowedSignReward_v2(window_size=60), 
        include_bias=cfg.include_bias,
        n_action_bins=N_ACTION_BINS 
    )

    # 2. Evaluation Environment 1: "Train Eval" (Last 500 rows of training data)
    # We create a specific env for this to measure how well we fit the recent training data
    train_eval_df = train_df.iloc[-cfg.test_size:].reset_index(drop=True)
    env_train_eval = Environment(
        data=train_eval_df,
        target_col="target",
        feature_cols=cfg.feature_cols,
        reward=WindowedSignReward_v2(window_size=60), 
        include_bias=cfg.include_bias,
        n_action_bins=N_ACTION_BINS
    )

    # 3. Evaluation Environment 2: "Test Eval" (Held-out data)
    env_test_eval = Environment(
        data=test_df,
        target_col="target",
        feature_cols=cfg.feature_cols,
        reward=WindowedSignReward_v2(window_size=60),
        include_bias=cfg.include_bias,
        n_action_bins=N_ACTION_BINS
    )

    ppo_cfg = PPOConfig(
        gamma=cfg.gamma, lam=cfg.lam, clip_eps=cfg.clip_eps, lr=cfg.lr,
        epochs=cfg.epochs, batch_size=cfg.batch_size, minibatch_size=cfg.minibatch_size,
        entropy_coef=cfg.entropy_coef, value_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm, l2_lambda=cfg.l2_lambda,
    )
    
    agent = PPOAgent(obs_dim=env_train.obs_dim, n_actions=N_ACTION_BINS, device=device, cfg=ppo_cfg)

    # Metrics history
    hist = {
        "policy_loss": [], "value_loss": [], "entropy": [], "avg_return": [], "raw_rew": [],
        "train_sharpe": [], "test_sharpe": [], 
        "train_acc": [], "test_acc": [],
        "eval_iters": []
    }

    print(f"Starting training on {len(train_df)} rows. Evaluating every {cfg.eval_every} iters.")

    for it in range(cfg.iters):
        # 1. Rollout & Update
        buf = rollout(env_train, agent, steps=ppo_cfg.batch_size, gamma=ppo_cfg.gamma, lam=ppo_cfg.lam)
        stats = agent.update(buf)
        
        # Logging
        avg_ret = float(buf["ret"].mean())
        raw_rew = float(buf["rew"].mean())
        hist["policy_loss"].append(stats["policy_loss"])
        hist["value_loss"].append(stats["value_loss"])
        hist["entropy"].append(stats["entropy"])
        hist["avg_return"].append(avg_ret)
        hist["raw_rew"].append(raw_rew)

        if (it + 1) % 10 == 0:
            print(f"Iter {it+1}/{cfg.iters} | Ret: {avg_ret:.4f} | Raw: {raw_rew:.4f} | P_Loss: {stats['policy_loss']:.4f}")

        # 2. Evaluation
        if (it + 1) % cfg.eval_every == 0:
            print(f"--- Evaluating at Iter {it+1} ---")
            
            # Evaluate on Training Subset
            m_train = run_evaluation(env_train_eval, agent, desc="Train")
            hist["train_sharpe"].append(m_train["sharpe"])
            hist["train_acc"].append(m_train["acc"])
            
            # Evaluate on Test Subset
            m_test = run_evaluation(env_test_eval, agent, desc="Test")
            hist["test_sharpe"].append(m_test["sharpe"])
            hist["test_acc"].append(m_test["acc"])
            
            hist["eval_iters"].append(it + 1)
            
            print(f"   [Train (Last 500)] Sharpe: {m_train['sharpe']:.4f} | Acc: {m_train['acc']:.4f}")
            print(f"   [Test (Held-out)]  Sharpe: {m_test['sharpe']:.4f}  | Acc: {m_test['acc']:.4f}")

        # 3. Save Checkpoint
        if (it + 1) % cfg.save_every == 0:
            save_dir = ensure_dir(cfg.save_dir)
            torch.save(agent.model.state_dict(), str(Path(save_dir) / f"ppo_iter{it+1}.pt"))

    return agent, hist


def main():
    cfg = parse_args()
    save_dir = ensure_dir(cfg.save_dir)
    print(f"Config: {cfg}")

    # Load Data (Returns Train and Test separately)
    df_raw, train_df, test_df, scaler = load_data(cfg)
    
    # Train
    agent, hist = train_on_dataframe(cfg, train_df, test_df)
    
    # Save Final Model & Artifacts
    torch.save(agent.model.state_dict(), str(save_dir / "ppo_final.pt"))
    save_training_artifacts(cfg, save_dir, agent, agent.cfg, scaler)
    
    # Save CSVs
    # (Truncate lists to match lengths if needed)
    min_len = len(hist["policy_loss"])
    train_curves = pd.DataFrame({
        "iteration": np.arange(1, min_len+1),
        "policy_loss": hist["policy_loss"],
        "avg_return": hist["avg_return"],
        "raw_rew": hist["raw_rew"]
    })
    train_curves.to_csv(save_dir / "train_curves.csv", index=False)
    
    if hist["eval_iters"]:
        eval_curves = pd.DataFrame({
            "iteration": hist["eval_iters"],
            "train_sharpe": hist["train_sharpe"],
            "test_sharpe": hist["test_sharpe"],
            "train_acc": hist["train_acc"],
            "test_acc": hist["test_acc"]
        })
        eval_curves.to_csv(save_dir / "eval_curves.csv", index=False)

    # Plotting
    if cfg.save_plots:
        print("Saving plots...")
        # Training Metrics
        plot_series(cfg, hist["avg_return"], "Avg Return (RL)", save_name="train_return.png")
        plot_series(cfg, hist["raw_rew"], "Raw Reward", save_name="train_raw_reward.png")
        
        # Evaluation Metrics (Separate Graphs)
        if hist["eval_iters"]:
            # Train Sharpe
            plot_series(cfg, hist["train_sharpe"], "Train Sharpe (Last 500)", 
                       x=hist["eval_iters"], save_name="eval_train_sharpe.png")
            # Test Sharpe
            plot_series(cfg, hist["test_sharpe"], "Test Sharpe (Held-out)", 
                       x=hist["eval_iters"], save_name="eval_test_sharpe.png")
            
    print("Done.")

if __name__ == "__main__":
    main()