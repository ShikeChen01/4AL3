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
import matplotlib.pyplot as plt  # plotting optional

from environment import Environment
from rewards import SimpleSignReward
from agent import PPOAgent, PPOConfig
from sklearn.preprocessing import StandardScaler
import joblib
from rewards import WindowedSignReward_v2

# =========================
# Central config
# =========================
@dataclass
class TrainConfig:
    # Data / features
    data_path: str = r"C:\Year4\4AL3\final-project\4AL3\src\RL\data\train.csv"
    target_col: str = "forward_returns"
    feature_cols: Optional[List[str]] = None  # None => all except target

    # Data Splitting
    ignore_first_n: int = 3500       # Drop first N rows (warmup/burn-in)
    test_size_m: int = 500          # Reserve last M rows for final testing

    # Device / env
    device: str = "cuda"          # "cuda" or "cpu"
    include_bias: bool = False
    threshold: float = 0.0        # reward threshold

    # Saving
    save_dir: str = "checkpoints"
    save_every: int = 250          # save model every N iterations

    # PPO / Training
    iters: int = 5000
    epochs: int = 5
    batch_size: int = 3000
    minibatch_size: int = 300
    lr: float = 1e-4
    clip_eps: float = 0.15
    gamma: float = 0.99
    lam: float = 0.9
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
# Argparse → TrainConfig
# =========================
def parse_args() -> TrainConfig:
    default = TrainConfig()

    p = argparse.ArgumentParser(description="PPO stock prediction training")

    # Data
    p.add_argument("--data-path", type=str, default=default.data_path)
    p.add_argument("--target-col", type=str, default=default.target_col)
    p.add_argument("--feature-cols", type=str, nargs="*", default=None)
    
    # Split Params
    p.add_argument("--ignore-first-n", type=int, default=default.ignore_first_n, help="Ignore first N rows")
    p.add_argument("--test-size-m", type=int, default=default.test_size_m, help="Save last M rows for testing")

    # Device / env
    p.add_argument("--device", type=str, default=default.device, choices=["cuda", "cpu"])
    p.add_argument("--include-bias", action="store_true", default=default.include_bias)
    p.add_argument("--threshold", type=float, default=default.threshold)

    # Saving
    p.add_argument("--save-dir", type=str, default=default.save_dir)
    p.add_argument("--save-every", type=int, default=default.save_every)

    # PPO / Training
    p.add_argument("--iters", type=int, default=default.iters)
    p.add_argument("--epochs", type=int, default=default.epochs)
    p.add_argument("--batch-size", type=int, default=default.batch_size)
    p.add_argument("--minibatch-size", type=int, default=default.minibatch_size)
    p.add_argument("--lr", type=float, default=default.lr)
    p.add_argument("--clip-eps", type=float, default=default.clip_eps)
    p.add_argument("--gamma", type=float, default=default.gamma)
    p.add_argument("--lam", type=float, default=default.lam)
    p.add_argument("--entropy-coef", type=float, default=default.entropy_coef)
    p.add_argument("--value-coef", type=float, default=default.value_coef)
    p.add_argument("--max-grad-norm", type=float, default=default.max_grad_norm)
    p.add_argument("--l2-lambda", type=float, default=default.l2_lambda)

    # CV / Eval
    p.add_argument("--k-folds", type=int, default=default.k_folds)
    p.add_argument("--eval-every", type=int, default=default.eval_every)

    # Plots
    p.add_argument("--plot-dir", type=str, default=default.plot_dir)
    p.add_argument("--no-save-plots", action="store_true", help="Disable saving plots")

    args = p.parse_args()
    cfg_dict = vars(args)

    # feature-cols: None or list
    cfg_dict["feature_cols"] = cfg_dict.pop("feature_cols")

    # save_plots is inverse of --no-save-plots
    cfg_dict["save_plots"] = not cfg_dict.pop("no_save_plots")

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
        try:
            df = pd.read_csv(path, engine="pyarrow")
        except Exception:
            df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    if cfg.target_col not in df.columns:
        raise ValueError(
            f"Target column '{cfg.target_col}' not found. "
            f"Available: {list(df.columns)[:10]} ..."
        )

    df = df.rename(columns={cfg.target_col: "target"})

    # drop leakage/irrelevant columns if present
    to_drop = [c for c in df.columns
               if ("market_forward_excess_returns" in c) or ("risk_free_rate" in c)]
    if to_drop:
        df = df.drop(columns=to_drop)

    df = add_missing_flags_and_zero_fill(df)

    feature_cols = cfg.feature_cols if cfg.feature_cols is not None else [
        c for c in df.columns if c != "target"
    ]

    df_raw = df.copy(deep=True)

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df = df.copy(deep=True) #fix fragmented data
    return df_raw, df, scaler


def save_training_artifacts(
    cfg: TrainConfig,
    save_dir: Path,
    agent: PPOAgent,
    ppo_cfg: PPOConfig,
    scaler: StandardScaler,
    df: pd.DataFrame,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    resolved_features = (
        cfg.feature_cols
        if cfg.feature_cols is not None
        else [c for c in df.columns if c != "target"]
    )

    run_cfg = {
        "ppo_config": asdict(ppo_cfg),
        "obs_dim": int(agent.model.shared[0].in_features),  # derived
        "n_actions": 5,
        "device": str(agent.device),
        "include_bias": bool(cfg.include_bias),
        "threshold": float(cfg.threshold),
        "target_col": "target",
        "feature_cols": resolved_features,
        "columns_in_dataframe": list(df.columns),
        "data_path": str(cfg.data_path),
        "env_meta": extra or {},
        "split_config": {
            "ignore_first_n": cfg.ignore_first_n,
            "test_size_m": cfg.test_size_m
        }
    }

    with open(save_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    joblib.dump(scaler, save_dir / "scaler.joblib")
    pd.Series(resolved_features, name="feature_cols").to_csv(
        save_dir / "feature_cols.csv", index=False
    )


# =========================
# PPO rollout
# =========================
def rollout(
    env: Environment,
    agent: PPOAgent,
    steps: int,
    gamma: float,
    lam: float,
) -> Dict[str, np.ndarray]:
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


def save_curves_csv(
    save_dir: Path,
    base_name: str,
    hist: Dict[str, List[float]],
) -> Tuple[Path, Optional[Path]]:
    """
    Writes two CSVs:
      1) '{base_name}_training_curves.csv' with per-iteration policy_loss, value_loss, entropy, avg_return
      2) '{base_name}_eval_curves.csv' with rows at eval points: iter, mse, acc (if any evals were recorded)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Per-iteration training stats
    train_df = pd.DataFrame({
        "iteration": np.arange(1, len(hist["policy_loss"]) + 1, dtype=int),
        "policy_loss": hist["policy_loss"],
        "value_loss": hist["value_loss"],
        "entropy": hist["entropy"],
        "avg_return": hist["avg_return"],
    })
    train_csv = save_dir / f"{base_name}_training_curves.csv"
    train_df.to_csv(train_csv, index=False)

    eval_csv_path = None
    if hist["mse_eval"]:
        eval_df = pd.DataFrame({
            "iteration": hist["iter_eval"],
            "mse": hist["mse_eval"],
            "sign_accuracy": hist["acc_eval"],
        })
        eval_csv_path = save_dir / f"{base_name}_eval_curves.csv"
        eval_df.to_csv(eval_csv_path, index=False)

    return train_csv, eval_csv_path


def train_on_dataframe(
    cfg: TrainConfig,
    df: pd.DataFrame,
    df_raw: pd.DataFrame,
    fold_tag: str,
    verbose: int = 1,
    eval_env: Optional[Environment] = None,
    scaler: Optional[StandardScaler] = None
) -> Tuple[PPOAgent, Dict[str, float], Dict[str, List[float]]]:
    


    device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    env = Environment(
        data=df,
        target_col="target",
        feature_cols=cfg.feature_cols,
        reward=SimpleSignReward(threshold=cfg.threshold),
        include_bias=cfg.include_bias,
    )
    ppo_cfg = PPOConfig(
        gamma=cfg.gamma, lam=cfg.lam, clip_eps=cfg.clip_eps, lr=cfg.lr,
        epochs=cfg.epochs, batch_size=cfg.batch_size, minibatch_size=cfg.minibatch_size,
        entropy_coef=cfg.entropy_coef, value_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm, l2_lambda=cfg.l2_lambda,
    )
    agent = PPOAgent(obs_dim=env.obs_dim, n_actions=3, device=device, cfg=ppo_cfg)

    # ---- history trackers ----
    hist = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "avg_return": [],
        "mse_eval": [],     # recorded every cfg.eval_every
        "acc_eval": [],
        "iter_eval": []
    }

    for it in range(cfg.iters):
        buf = rollout(env, agent, steps=ppo_cfg.batch_size, gamma=ppo_cfg.gamma, lam=ppo_cfg.lam)
        stats = agent.update(buf)
        avg_ret = float(buf["ret"].mean())

        # save histories
        hist["policy_loss"].append(stats["policy_loss"])
        hist["value_loss"].append(stats["value_loss"])
        hist["entropy"].append(stats["entropy"])
        hist["avg_return"].append(avg_ret)

        if (it + 1) % verbose == 0:
            print(
                f"[{fold_tag}] Iter {it+1}/{cfg.iters} | AvgReturn: {avg_ret:.4f} | "
                f"PolicyLoss: {stats['policy_loss']:.4f} | ValueLoss: {stats['value_loss']:.4f} | "
                f"Entropy: {stats['entropy']:.4f}"
            )

        # periodic eval MSE & sign-accuracy
        if eval_env is not None and scaler is not None and (it + 1) % cfg.eval_every == 0:
            mse, acc = evaluate_mse(eval_env, agent, scaler)
            hist["mse_eval"].append(mse)
            hist["acc_eval"].append(acc)
            hist["iter_eval"].append(it + 1)
            print(f"[{fold_tag}] Eval at Iter {it+1}: MSE={mse:.6f} | sign-acc={acc:.4f}")

        # periodic save
        if (it + 1) % cfg.save_every == 0:
            save_dir = ensure_dir(cfg.save_dir)
            torch.save(
                agent.model.state_dict(),
                str(Path(save_dir) / f"ppo_{fold_tag.replace(' ', '_').lower()}_iter{it+1}.pt")
            )

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
# Plot helpers (optional)
# =========================
def plot_series(
    cfg: TrainConfig,
    y: List[float],
    title: str,
    x: Optional[List[int]] = None,
    save_name: Optional[str] = None,
):
    if x is None:
        x = list(range(1, len(y) + 1))
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(title)
    plt.grid(True, alpha=0.3)
    if cfg.save_plots and save_name is not None:
        save_path = Path(cfg.plot_dir) / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# =========================
# Main
# =========================
def main() -> None:
    cfg = parse_args()

    save_dir = ensure_dir(cfg.save_dir)
    if cfg.save_plots:
        ensure_dir(cfg.plot_dir)

    device = "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device} | Saving to: {save_dir}")
    print(f"Config: {cfg}")

    df_raw, df_loaded, scaler = load_data(cfg)

    # --- Manual Train/Test Split logic ---
    total_len = len(df_loaded)
    n = cfg.ignore_first_n
    m = cfg.test_size_m
    
    if n < 0 or m < 0:
        raise ValueError(f"ignore_first_n ({n}) and test_size_m ({m}) must be non-negative.")
    if n + m >= total_len:
        raise ValueError(f"Split invalid: n={n} + m={m} >= total_len={total_len}. No training data left.")

    # Slice the active training/val portion
    train_val_df = df_loaded.iloc[n : total_len - m].reset_index(drop=True)
    df_raw_train = df_raw.iloc[n : total_len - m].reset_index(drop=True)

    # Slice the holdout test portion
    test_df = df_loaded.iloc[total_len - m :].reset_index(drop=True)

    print(f"--- Data Split ---")
    print(f"Total Rows:     {total_len}")
    print(f"Ignored First:  {n}")
    print(f"Train/Val Set:  {len(train_val_df)}")
    print(f"Holdout Test:   {len(test_df)}")
    print(f"------------------")

    # Use train_val_df as the 'full_df' for the existing logic
    full_df = train_val_df

    # ----- single run -----
    if cfg.k_folds <= 1:
        env_eval = Environment(
            full_df,
            "target",
            cfg.feature_cols,
            reward=SimpleSignReward(cfg.threshold),
            include_bias=cfg.include_bias,
        )
        agent, _, hist = train_on_dataframe(
            cfg,
            full_df,
            df_raw_train,
            fold_tag="FULL",
            eval_env=env_eval,
            scaler=scaler,
        )

        # ---- CSV curves (saved alongside checkpoints) ----
        train_csv, eval_csv = save_curves_csv(save_dir, base_name="full", hist=hist)
        print(f"Saved training curves: {train_csv}")
        if eval_csv is not None:
            print(f"Saved eval curves: {eval_csv}")

        # ---- optional plots ----
        if cfg.save_plots:
            plot_series(cfg, hist["policy_loss"], "Policy Loss", save_name="policy_loss.png")
            plot_series(cfg, hist["value_loss"], "Value Loss", save_name="value_loss.png")
            plot_series(cfg, hist["avg_return"], "Average Return", save_name="avg_return.png")
            if hist["mse_eval"]:
                plot_series(
                    cfg,
                    hist["mse_eval"],
                    "Eval MSE",
                    x=hist["iter_eval"],
                    save_name="mse_eval.png",
                )

        mse, acc = evaluate_mse(env_eval, agent, scaler)
        print(f"[FULL] Final MSE={mse:.6f}  sign-acc={acc:.4f}")
        torch.save(agent.model.state_dict(), str(save_dir / "ppo_full.pt"))
        print(f"Saved weights: {save_dir / 'ppo_full.pt'}")
        
        # Determine which agent to use for final testing
        final_test_agent = agent

        # Save run artifacts (config, scaler, feature order)
        save_training_artifacts(
            cfg=cfg,
            save_dir=save_dir,
            agent=agent,
            ppo_cfg=agent.cfg,
            scaler=scaler,
            df=full_df,
            extra={"k_folds": cfg.k_folds, "iters": cfg.iters, "epochs": cfg.epochs},
        )
        print(f"Saved training artifacts: {save_dir / 'run_config.json'}, {save_dir / 'scaler.joblib'}")

    else:
        # ---------- K-fold ----------
        n_cv = len(full_df)
        folds = forward_chaining_folds(n_cv, cfg.k_folds)
        mses: List[float] = []
        accs: List[float] = []
        last_agent: Optional[PPOAgent] = None

        for fold_id, (tr, va) in enumerate(folds, start=1):
            df_tr = full_df.iloc[tr].reset_index(drop=True)
            df_va = full_df.iloc[va].reset_index(drop=True)

            env_eval = Environment(
                df_va,
                "target",
                cfg.feature_cols,
                reward=WindowedSignReward_v2(threshold=cfg.threshold,),
                include_bias=cfg.include_bias,
            )
            agent, _, hist = train_on_dataframe(
                cfg,
                df_tr,
                df_raw_train,
                fold_tag=f"FOLD_{fold_id}",
                eval_env=env_eval,
                scaler=scaler,
            )

            # Save CSV curves per fold
            train_csv, eval_csv = save_curves_csv(save_dir, base_name=f"fold{fold_id}", hist=hist)
            print(f"[FOLD {fold_id}] Saved training curves: {train_csv}")
            if eval_csv is not None:
                print(f"[FOLD {fold_id}] Saved eval curves: {eval_csv}")

            # Optional plots
            if cfg.save_plots:
                plot_series(
                    cfg,
                    hist["policy_loss"],
                    f"Policy Loss (Fold {fold_id})",
                    save_name=f"policy_loss_fold{fold_id}.png",
                )
                plot_series(
                    cfg,
                    hist["value_loss"],
                    f"Value Loss (Fold {fold_id})",
                    save_name=f"value_loss_fold{fold_id}.png",
                )
                plot_series(
                    cfg,
                    hist["avg_return"],
                    f"Average Return (Fold {fold_id})",
                    save_name=f"avg_return_fold{fold_id}.png",
                )
                if hist["mse_eval"]:
                    plot_series(
                        cfg,
                        hist["mse_eval"],
                        f"Eval MSE (Fold {fold_id})",
                        x=hist["iter_eval"],
                        save_name=f"mse_eval_fold{fold_id}.png",
                    )

            mse, acc = evaluate_mse(env_eval, agent, scaler)
            mses.append(mse)
            accs.append(acc)
            print(f"[FOLD {fold_id}] Final val MSE={mse:.6f}  sign-acc={acc:.4f}  (n={len(df_va)})")

            out_path = Path(cfg.save_dir) / f"ppo_fold{fold_id}.pt"
            torch.save(agent.model.state_dict(), str(out_path))
            print(f"Saved weights: {out_path}")
            last_agent = agent

        print(f"[CV] mean MSE={np.mean(mses):.6f}  std={np.std(mses):.6f} | mean sign-acc={np.mean(accs):.4f}")
        torch.save(last_agent.model.state_dict(), str(Path(cfg.save_dir) / "ppo_full.pt"))
        print(f"Saved weights: {Path(cfg.save_dir) / 'ppo_full.pt'}")
        
        final_test_agent = last_agent

        # Save run artifacts
        save_training_artifacts(
            cfg=cfg,
            save_dir=Path(cfg.save_dir),
            agent=last_agent,
            ppo_cfg=last_agent.cfg,
            scaler=scaler,
            df=full_df,
            extra={"k_folds": cfg.k_folds, "iters": cfg.iters, "epochs": cfg.epochs},
        )
        print(f"Saved training artifacts: {Path(cfg.save_dir) / 'run_config.json'}, {Path(cfg.save_dir) / 'scaler.joblib'}")

    # --- Final Holdout Test ---
    if len(test_df) > 0 and final_test_agent is not None:
        print("\n===========================================")
        print(" RUNNING FINAL EVALUATION ON HOLDOUT TEST ")
        print("===========================================")
        env_test = Environment(
            test_df, 
            "target", 
            cfg.feature_cols, 
            reward=SimpleSignReward(cfg.threshold), 
            include_bias=cfg.include_bias
        )
        mse_test, acc_test = evaluate_mse(env_test, final_test_agent, scaler)
        print(f"[TEST SET result] MSE={mse_test:.6f} | sign-acc={acc_test:.4f}")
        print("===========================================\n")

if __name__ == "__main__":
    main()