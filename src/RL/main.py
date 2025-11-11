
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
import torch

from environment import Environment
from rewards import SimpleSignReward
from agent import PPOAgent, PPOConfig
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG (edit here)
# =========================
DATA_PATH: str = r"data/train.csv"        # set me
TARGET_COL: str = "forward_returns"           # will be renamed to 'target'
FEATURE_COLS: Optional[List[str]] = None      # or list of names; None => all except target
DEVICE: str = "cuda"                          # "cuda" or "cpu"
INCLUDE_BIAS: bool = False
THRESHOLD: float = 0.0                        # abs(target) below this => treat as flat

# Saving
SAVE_DIR: str = r"./checkpoints"               # model weights will be saved here

# PPO / Training
ITERS: int = 3000
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
L2_LAMBDA: float = 1e-4   # L2 regularizer strength
SAVE_EVERY : int = 50     # iterations
# Cross-validation
K_FOLDS: int = 1          # set >1 to enable forward-chaining CV
EVAL_EVERY: int = 100     # Evaluate MSE every this many iterations


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
    """
    Run once over env.df in order. Deterministic policy (argmax).
    Returns (mse, accuracy) where predictions are in {-1,0,1} vs raw 'target'.
    """
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

    # Create a dummy array with the same number of features as the training data
    # and populate the target column with the predictions
    dummy_array = np.zeros((len(y_pred_arr), scaler.n_features_in_))
    # Find the index of the target column in the original DataFrame
    target_col_idx = env.df.columns.get_loc("target")
    dummy_array[:, target_col_idx] = y_pred_arr

    # Inverse transform the dummy array
    y_pred_arr_transformed = scaler.inverse_transform(dummy_array)[:, target_col_idx]


    mse = float(np.mean((y_pred_arr_transformed - y_true_arr) ** 2))
    acc = float(np.mean(np.sign(y_pred_arr) == np.sign(y_true_arr)))
    return mse, acc


# =========================
# Utilities
# =========================
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def train_on_dataframe(df: pd.DataFrame, fold_tag: str, verbose : int = 150, eval_env: Optional[Environment] = None, scaler: Optional[StandardScaler] = None) -> Tuple[PPOAgent, Dict[str, float]]:
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

    for it in range(ITERS):
        buf = rollout(env, agent, steps=cfg.batch_size, gamma=cfg.gamma, lam=cfg.lam)
        stats = agent.update(buf)
        avg_ret = float(buf["ret"].mean())
        print(f"[{fold_tag}] Iter {it+1}/{ITERS} | AvgReturn: {avg_ret:.4f} | "
                f"PolicyLoss: {stats['policy_loss']:.4f} | ValueLoss: {stats['value_loss']:.4f} | "
                f"Entropy: {stats['entropy']:.4f}")

        if eval_env is not None and scaler is not None and (it + 1) % EVAL_EVERY == 0:
            mse, acc = evaluate_mse(eval_env, agent, scaler)
            print(f"[{fold_tag}] Eval at Iter {it+1}: MSE={mse:.6f} | sign-acc={acc:.4f}")


    return agent, stats


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
# Main
# =========================
def main() -> None:
    save_dir = ensure_dir(SAVE_DIR)
    device = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device} | Saving to: {save_dir}")

    full_df, scaler = load_data(DATA_PATH, target=TARGET_COL)

    if K_FOLDS <= 1:
        env_eval = Environment(full_df, "target", FEATURE_COLS, reward=SimpleSignReward(THRESHOLD))
        agent, _ = train_on_dataframe(full_df, fold_tag="FULL", eval_env=env_eval, scaler=scaler)
        mse, acc = evaluate_mse(env_eval, agent, scaler)
        print(f"[FULL] Final MSE={mse:.6f}  sign-acc={acc:.4f}")
        torch.save(agent.model.state_dict(), str(save_dir / "ppo_full.pt"))
        print(f"Saved weights: {save_dir / 'ppo_full.pt'}")
        return

    n = len(full_df)
    folds = forward_chaining_folds(n, K_FOLDS)
    mses: List[float] = []
    accs: List[float] = []

    for fold_id, (tr, va) in enumerate(folds, start=1):
        df_tr = full_df.iloc[tr].reset_index(drop=True)
        df_va = full_df.iloc[va].reset_index(drop=True)

        env_eval = Environment(df_va, "target", FEATURE_COLS, reward=SimpleSignReward(THRESHOLD))
        agent, _ = train_on_dataframe(df_tr, fold_tag=f"FOLD {fold_id} (train {len(df_tr)})", eval_env=env_eval, scaler=scaler)

        mse, acc = evaluate_mse(env_eval, agent, scaler)
        mses.append(mse); accs.append(acc)
        print(f"[FOLD {fold_id}] Final val MSE={mse:.6f}  sign-acc={acc:.4f}  (n={len(df_va)})")

        out_path = save_dir / f"ppo_fold{fold_id}.pt"
        torch.save(agent.model.state_dict(), str(out_path))
        print(f"Saved weights: {out_path}")

    print(f"[CV] mean MSE={np.mean(mses):.6f}  std={np.std(mses):.6f} | mean sign-acc={np.mean(accs):.4f}")
    (save_dir / "cv_metrics.txt").write_text(
        f"mses={mses}\naccs={accs}\nmean_mse={float(np.mean(mses))}\nstd_mse={float(np.std(mses))}\nmean_acc={float(np.mean(accs))}\n",
        encoding="utf-8",
    )
    print(f"Wrote CV metrics to {save_dir / 'cv_metrics.txt'}")


if __name__ == "__main__":
    main()