# run_trading_from_ppo.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- FIX: Import properly from your file 'A1.py' ---
from A1 import ActorCritic

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
    # Note: We keep risk_free_rate if it exists, but don't scale it if it's not in feature_cols
    to_drop = [c for c in df.columns if ("market_forward_excess_returns" in c)]
    if to_drop:
        df = df.drop(columns=to_drop)

    df = add_missing_flags_and_zero_fill(df)

    feat_cols = feature_cols if feature_cols is not None else [c for c in df.columns if c != "target"]
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])

    # --- FIX: Safety fill for NaNs created by scaling constant columns ---
    df[feat_cols] = df[feat_cols].fillna(0.0)

    return df, feat_cols

# -----------------------------
# Load model from state_dict
# -----------------------------
def load_actorcritic_from_checkpoint(state_path: str | Path, obs_dim: int,
                                     n_actions: int = 5, l2_lambda: float = 0.0,
                                     map_location: str = "cpu") -> ActorCritic:
    
    state_dict = torch.load(state_path, map_location=map_location, weights_only=True)
    
    # Auto-detect hidden size from weights
    if "policy_head.weight" not in state_dict:
        raise RuntimeError("Checkpoint missing 'policy_head.weight' – not an ActorCritic state_dict?")
    
    # weight shape is [n_actions, hidden]
    hidden_w = state_dict["policy_head.weight"].shape[1] 
    saved_n_actions = state_dict["policy_head.weight"].shape[0]

    if saved_n_actions != n_actions:
        print(f"WARNING: Checkpoint has {saved_n_actions} actions, but code requested {n_actions}. Using checkpoint value.")
        n_actions = saved_n_actions

    model = ActorCritic(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden_w, l2_lambda=l2_lambda)
    
    # Validation of input dimensions
    saved_in = state_dict.get("shared.0.weight", None)
    if saved_in is not None:
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
    actions: np.ndarray          # Stores the leverage (0.0 to 2.0)
    daily_pnl: np.ndarray        # Leverage * Target
    cum_pnl: np.ndarray
    hit_rate: float              
    hold_ratio: float            # Ratio of time we are in cash (leverage < 0.1)
    target: np.ndarray           
    target_cum: np.ndarray       

def run_trading(
    df: pd.DataFrame,
    feat_cols: List[str],
    model: ActorCritic,
    date_col: Optional[str] = None,
    csv_path: Optional[str | Path] = None,
) -> TradeResult:
    """
    New Decision rule for Leverage Bot:
      1. Get logits for 5 actions.
      2. Argmax -> index 0..4
      3. Map index -> Leverage [0.0, 0.5, 1.0, 1.5, 2.0]
    """
    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y = df["target"].to_numpy(dtype=np.float32, copy=False)
    dates = df[date_col].to_numpy() if date_col and date_col in df.columns else None

    # Prepare Action Map (Matches environment.py)
    # If model has 5 actions, we assume linspace(0, 2, 5)
    n_actions = model.policy_head.out_features
    action_map = np.linspace(0.0, 2.0, n_actions)

    actions = np.zeros(len(df), dtype=np.float32) # Now stores float leverage
    daily_pnl = np.zeros(len(df), dtype=np.float32)

    print(f"Running inference on {len(df)} rows with {n_actions} discrete leverage bins...")

    with torch.no_grad():
        for i in range(len(df)):
            x = torch.from_numpy(X[i:i+1])
            logits, _ = model(x)
            
            # --- FIX: Use Argmax for discrete leverage bin ---
            a_idx = torch.argmax(logits, dim=-1).item()
            leverage = float(action_map[a_idx])
            
            actions[i] = leverage
            
            # Simple PnL: Leverage * Market Return
            # (Ignoring risk-free rate for pure directional visual)
            daily_pnl[i] = leverage * y[i]

    cum_pnl = np.cumsum(daily_pnl)
    target_cum = np.cumsum(y)

    # Metrics
    # "Hit" = We held leverage > 0.1 and market went up, OR we held cash < 0.1 and market went down
    # (This is a simplification for 'No Shorting' environments)
    market_up = y > 0
    we_are_invested = actions > 0.1
    hits = (market_up == we_are_invested)
    hit_rate = float(np.mean(hits))
    
    hold_ratio = float(np.mean(actions < 0.1))

    # ---- build per-day trading results DataFrame ----
    out = {}
    if dates is not None:
        out["date"] = dates

    out.update({
        "leverage": actions,
        "target": y,
        "daily_pnl": daily_pnl,
        "cum_pnl": cum_pnl,
        "target_cum": target_cum,
        "correct_direction": hits.astype(int),
    })

    results_df = pd.DataFrame(out)

    if csv_path is not None:
        csv_path = Path(csv_path)
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

    return TradeResult(
        dates=dates,
        actions=actions,
        daily_pnl=daily_pnl,
        cum_pnl=cum_pnl,
        hit_rate=hit_rate,
        hold_ratio=hold_ratio,
        target=y,
        target_cum=target_cum,
    )

# -----------------------------
# Plotting
# -----------------------------
def plot_cum_target_vs_strategy(result: TradeResult, title: str = "Cumulative Return"):
    plt.figure(figsize=(12, 6))
    
    # If we have many points, plotting index is cleaner than dates usually
    x_axis = result.dates if result.dates is not None else np.arange(len(result.cum_pnl))
    
    plt.plot(x_axis, result.target_cum, label="Market (Buy & Hold 1x)", color='gray', alpha=0.6)
    plt.plot(x_axis, result.cum_pnl, label="RL Strategy (Leverage 0-2x)", color='blue', linewidth=1.5)
    
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # --- EDIT THESE ---
    CKPT_PATH = r"C:\Year4\4AL3\final-project\4AL3\checkpoints\temp5\checkpoints\ppo_full_iter1250.pt" 
    DATA_PATH = r"C:\Year4\4AL3\final-project\4AL3\result\RL_1\data\train.csv"
    TARGET_COL = "forward_returns"
    DATE_COL: Optional[str] = None 
    # ------------------

    # 1. Load Data
    print("Loading data...")
    df, feat_cols = load_data(DATA_PATH, target=TARGET_COL)
    obs_dim = len(feat_cols)
    print(f"Data loaded. Obs Dim: {obs_dim}")

    # 2. Load Model
    # FIX: n_actions=5 (matches your 0.0, 0.5, 1.0, 1.5, 2.0 setup)
    print(f"Loading model from {CKPT_PATH}...")
    try:
        model = load_actorcritic_from_checkpoint(CKPT_PATH, obs_dim=obs_dim, n_actions=5, map_location="cpu")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {CKPT_PATH}")
        exit()

    # 3. Run Inference
    res = run_trading(df, feat_cols, model, date_col=DATE_COL, csv_path="trading_results.csv")

    # 4. Print Metrics
    mse = float(np.mean((res.target - res.daily_pnl) ** 2))

    print("-" * 40)
    print(f"Average Leverage Used: {np.mean(res.actions):.2f}x")
    print(f"Hit-rate (Direction):  {res.hit_rate:.4f}")
    print(f"Time in Cash (<0.1x):  {res.hold_ratio:.4f}")
    print("-" * 40)
    print(f"Market Total Return:   {res.target_cum[-1]:.6f}")
    print(f"Strategy Total Return: {res.cum_pnl[-1]:.6f}")
    print(f"MSE (Strategy vs Mkt): {mse:.6f}")
    print("-" * 40)

    plot_cum_target_vs_strategy(res, title="RL Agent Performance vs Market")