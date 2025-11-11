from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt  # <-- NEW

from environment import Environment
from rewards import SimpleSignReward
from agent import PPOAgent, PPOConfig
from sklearn.preprocessing import StandardScaler
import json
from dataclasses import asdict
import joblib


# =========================
# CONFIG (edit here)
# =========================
DATA_PATH: str = r"C:\Year4\4AL3\final-project\4AL3\src\RL\data\train.csv"        # set me
TARGET_COL: str = "forward_returns"       # will be renamed to 'target'
FEATURE_COLS: Optional[List[str]] = None  # None => all except target
DEVICE: str = "cuda"                      # "cuda" or "cpu"
INCLUDE_BIAS: bool = False
THRESHOLD: float = 0.0

# Saving
SAVE_DIR: str = r"C:\Year4\4AL3\final-project\4AL3\src\RL\checkpoints"
SAVE_EVERY: int = 50

# PPO / Training
ITERS: int = 300
EPOCHS: int = 5
BATCH_SIZE: int = 128
MINIBATCH: int = 32
LR: float = 3e-4
CLIP: float = 0.2
GAMMA: float = 0.99
LAMBDA_GAE: float = 0.95
ENTROPY_COEF: float = 0.01
VALUE_COEF: float = 0.5
MAX_GRAD_NORM: float = 0.5
L2_LAMBDA: float = 1e-4

# Cross-validation / Eval
K_FOLDS: int = 1
EVAL_EVERY: int = 100     # Evaluate MSE every this many iterations

# Plot saving (optional)
PLOT_DIR: str = "./plots"
SAVE_PLOTS: bool = True

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

def load_data(path: str | Path, target: str = TARGET_COL) -> Tuple[pd.DataFrame, StandardScaler]:
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
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)[:10]} ...")

    df = df.rename(columns={target: "target"})

    to_drop = [c for c in df.columns if ("market_forward_excess_returns" in c) or ("risk_free_rate" in c)]
    if to_drop:
        df = df.drop(columns=to_drop)

    df = add_missing_flags_and_zero_fill(df)

    # Normalize features
    feature_cols = FEATURE_COLS if FEATURE_COLS is not None else [c for c in df.columns if c != "target"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler

def save_training_artifacts(
    save_dir: Path,
    agent: PPOAgent,
    cfg: PPOConfig,
    scaler: StandardScaler,
    df: pd.DataFrame,
    feature_cols: Optional[List[str]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    # Resolve the exact feature set in the order used to build obs
    resolved_features = feature_cols if feature_cols is not None else [c for c in df.columns if c != "target"]

    run_cfg = {
        "ppo_config": asdict(cfg),
        "obs_dim": int(agent.obs_dim),
        "n_actions": int(agent.n_actions) if hasattr(agent, "n_actions") else 3,
        "device": str(agent.device),
        "include_bias": bool(INCLUDE_BIAS),
        "threshold": float(THRESHOLD),
        "target_col": "target",
        "feature_cols": resolved_features,
        "columns_in_dataframe": list(df.columns),
        "data_path": str(DATA_PATH),
        "env_meta": extra or {},
    }

    # JSON config
    with open(save_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    # Scaler (to guarantee identical transform on reload)
    joblib.dump(scaler, save_dir / "scaler.joblib")

    # Also persist feature order as CSV for eyeballing
    pd.Series(resolved_features, name="feature_cols").to_csv(save_dir / "feature_cols.csv", index=False)


# =========================
# PPO rollout
# =========================
def rollout(env: Environment, agent: PPOAgent, steps: int, gamma: float, lam: float) -> Dict[str, np.ndarray]:
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    obs = env.reset()
    for _ in range(steps):
        a_idx, logp, v = agent.act(obs)
        action_map = {0: -1, 1: 0, 2: 1}
        a_env = action_map[a_idx]
        next_obs, r, done, info = env.step(a_env)

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
    y_pred: List[int] = []

    obs = env.reset()
    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = agent.model(obs_t)
            a_idx = int(torch.argmax(logits, dim=-1).item())
        idx_to_action = {0: -1, 1: 0, 2: 1}
        a_env = idx_to_action[a_idx]

        t = env.t
        y_true.append(float(env.y[t]))
        y_pred.append(int(a_env))

        obs, _, done, _ = env.step(a_env)

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)

    # Inverse-transform hack to put predictions back on same scale if needed
    dummy = np.zeros((len(y_pred_arr), scaler.n_features_in_))
    target_col_idx = env.df.columns.get_loc("target")
    dummy[:, target_col_idx] = y_pred_arr
    y_pred_arr_inv = scaler.inverse_transform(dummy)[:, target_col_idx]

    mse = float(np.mean((y_pred_arr_inv - y_true_arr) ** 2))
    acc = float(np.mean(np.sign(y_pred_arr) == np.sign(y_true_arr)))
    return mse, acc

# =========================
# Utilities
# =========================
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def train_on_dataframe(
    df: pd.DataFrame,
    fold_tag: str,
    verbose: int = 150,
    eval_env: Optional[Environment] = None,
    scaler: Optional[StandardScaler] = None
) -> Tuple[PPOAgent, Dict[str, float], Dict[str, List[float]]]:

    device = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    env = Environment(
        data=df,
        target_col="target",
        feature_cols=FEATURE_COLS,
        reward=SimpleSignReward(threshold=THRESHOLD),
        include_bias=INCLUDE_BIAS,
    )
    cfg = PPOConfig(
        gamma=GAMMA, lam=LAMBDA_GAE, clip_eps=CLIP, lr=LR, epochs=EPOCHS,
        batch_size=BATCH_SIZE, minibatch_size=MINIBATCH, entropy_coef=ENTROPY_COEF,
        value_coef=VALUE_COEF, max_grad_norm=MAX_GRAD_NORM, l2_lambda=L2_LAMBDA,
    )
    agent = PPOAgent(obs_dim=env.obs_dim, n_actions=3, device=device, cfg=cfg)

    # ---- NEW: history trackers ----
    hist = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "avg_return": [],
        "mse_eval": [],     # recorded every EVAL_EVERY (if eval_env provided)
        "iter_eval": []     # x-axis for MSE points
    }

    for it in range(ITERS):
        buf = rollout(env, agent, steps=cfg.batch_size, gamma=cfg.gamma, lam=cfg.lam)
        stats = agent.update(buf)
        avg_ret = float(buf["ret"].mean())

        # save histories
        hist["policy_loss"].append(stats["policy_loss"])
        hist["value_loss"].append(stats["value_loss"])
        hist["entropy"].append(stats["entropy"])
        hist["avg_return"].append(avg_ret)

        if (it + 1) % verbose == 0:
            print(f"[{fold_tag}] Iter {it+1}/{ITERS} | AvgReturn: {avg_ret:.4f} | "
                  f"PolicyLoss: {stats['policy_loss']:.4f} | ValueLoss: {stats['value_loss']:.4f} | "
                  f"Entropy: {stats['entropy']:.4f}")

        # periodic eval MSE
        if eval_env is not None and scaler is not None and (it + 1) % EVAL_EVERY == 0:
            mse, acc = evaluate_mse(eval_env, agent, scaler)
            hist["mse_eval"].append(mse)
            hist["iter_eval"].append(it + 1)
            print(f"[{fold_tag}] Eval at Iter {it+1}: MSE={mse:.6f} | sign-acc={acc:.4f}")

        # periodic save
        if (it + 1) % SAVE_EVERY == 0:
            ensure_dir(SAVE_DIR)
            torch.save(agent.model.state_dict(), str(Path(SAVE_DIR) / f"ppo_{fold_tag.replace(' ', '_').lower()}_iter{it+1}.pt"))

    return agent, stats, hist

def forward_chaining_folds(n: int, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    block = n // k if k > 0 else n
    for i in range(k):
        val_start = i * block
        val_end = n if i == k - 1 else (i + 1) * block
        if val_start == 0:
            continue
        train_idx = np.arange(0, val_start, dtype=np.int64)
        val_idx = np.arange(val_start, val_end, dtype=np.int64)
        folds.append((train_idx, val_idx))
    if not folds:
        cut = int(0.8 * n)
        folds.append((np.arange(0, cut), np.arange(cut, n)))
    return folds

# =========================
# Plot helpers (NEW)
# =========================
def plot_series(y: List[float], title: str, x: Optional[List[int]] = None, save_path: Optional[Path] = None):
    if x is None:
        x = list(range(1, len(y) + 1))
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(title)
    plt.grid(True, alpha=0.3)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# =========================
# Main
# =========================
def main() -> None:
    save_dir = ensure_dir(SAVE_DIR)
    plot_dir = ensure_dir(PLOT_DIR) if SAVE_PLOTS else Path(PLOT_DIR)
    device = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device} | Saving to: {save_dir}")

    full_df, scaler = load_data(DATA_PATH, target=TARGET_COL)

    if K_FOLDS <= 1:
        env_eval = Environment(full_df, "target", FEATURE_COLS, reward=SimpleSignReward(THRESHOLD))
        agent, _, hist = train_on_dataframe(full_df, fold_tag="FULL", eval_env=env_eval, scaler=scaler)

        # ---- NEW: plots ----
        plot_series(hist["policy_loss"], "Policy Loss", save_path=(plot_dir / "policy_loss.png" if SAVE_PLOTS else None))
        plot_series(hist["value_loss"], "Value Loss", save_path=(plot_dir / "value_loss.png" if SAVE_PLOTS else None))
        plot_series(hist["avg_return"], "Average Return", save_path=(plot_dir / "avg_return.png" if SAVE_PLOTS else None))
        if hist["mse_eval"]:
            plot_series(hist["mse_eval"], "Eval MSE", x=hist["iter_eval"],
                        save_path=(plot_dir / "mse_eval.png" if SAVE_PLOTS else None))

        mse, acc = evaluate_mse(env_eval, agent, scaler)
        print(f"[FULL] Final MSE={mse:.6f}  sign-acc={acc:.4f}")
        torch.save(agent.model.state_dict(), str(save_dir / "ppo_full.pt"))
        print(f"Saved weights: {save_dir / 'ppo_full.pt'}")
        return

    # K-fold, training histories per fold (optional: add aggregate plotting)
    n = len(full_df)
    folds = forward_chaining_folds(n, K_FOLDS)
    mses: List[float] = []
    accs: List[float] = []

    for fold_id, (tr, va) in enumerate(folds, start=1):
        df_tr = full_df.iloc[tr].reset_index(drop=True)
        df_va = full_df.iloc[va].reset_index(drop=True)

        env_eval = Environment(df_va, "target", FEATURE_COLS, reward=SimpleSignReward(THRESHOLD))
        agent, _, hist = train_on_dataframe(df_tr, fold_tag=f"FOLD_{fold_id}", eval_env=env_eval, scaler=scaler)

        # Per-fold plots
        plot_series(hist["policy_loss"], f"Policy Loss (Fold {fold_id})",
                    save_path=(plot_dir / f"policy_loss_fold{fold_id}.png" if SAVE_PLOTS else None))
        plot_series(hist["value_loss"], f"Value Loss (Fold {fold_id})",
                    save_path=(plot_dir / f"value_loss_fold{fold_id}.png" if SAVE_PLOTS else None))
        plot_series(hist["avg_return"], f"Average Return (Fold {fold_id})",
                    save_path=(plot_dir / f"avg_return_fold{fold_id}.png" if SAVE_PLOTS else None))
        if hist["mse_eval"]:
            plot_series(hist["mse_eval"], f"Eval MSE (Fold {fold_id})", x=hist["iter_eval"],
                        save_path=(plot_dir / f"mse_eval_fold{fold_id}.png" if SAVE_PLOTS else None))

        mse, acc = evaluate_mse(env_eval, agent, scaler)
        mses.append(mse); accs.append(acc)
        print(f"[FOLD {fold_id}] Final val MSE={mse:.6f}  sign-acc={acc:.4f}  (n={len(df_va)})")

        out_path = save_dir / f"ppo_fold{fold_id}.pt"
        torch.save(agent.model.state_dict(), str(out_path))
        print(f"Saved weights: {out_path}")

    print(f"[CV] mean MSE={np.mean(mses):.6f}  std={np.std(mses):.6f} | mean sign-acc={np.mean(accs):.4f}")
    # after:
    torch.save(agent.model.state_dict(), str(save_dir / "ppo_full.pt"))
    print(f"Saved weights: {save_dir / 'ppo_full.pt'}")

    # NEW: save config + scaler + feature order
    save_training_artifacts(
        save_dir=save_dir,
        agent=agent,
        cfg=cfg,                # <-- pass the PPOConfig you used to build the agent
        scaler=scaler,
        df=full_df,
        feature_cols=FEATURE_COLS,
        extra={"k_folds": K_FOLDS, "iters": ITERS, "epochs": EPOCHS}
    )
    print(f"Saved training artifacts: {save_dir / 'run_config.json'}, {save_dir / 'scaler.joblib'}")

if __name__ == "__main__":
    main()
