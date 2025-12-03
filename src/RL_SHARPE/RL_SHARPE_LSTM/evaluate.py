# evaluate.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import os
import argparse

import numpy as np
import pandas as pd
import pandas.api.types # Required for score function
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import the LSTM ActorCritic from A1
from A1 import ActorCritic

# -----------------------------
# Competition Scoring Function
# -----------------------------
MIN_INVESTMENT = 0
MAX_INVESTMENT = 2

class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates a custom evaluation metric (volatility-adjusted Sharpe ratio).
    """
    # Create a copy to avoid modifying the original dataframe outside this scope
    solution = solution.copy()
    submission = submission.copy()

    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
        raise ParticipantVisibleError('Predictions must be numeric')

    solution['position'] = submission['prediction']

    if solution['position'].max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].max()} exceeds maximum of {MAX_INVESTMENT}')
    if solution['position'].min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f'Position of {solution["position"].min()} below minimum of {MIN_INVESTMENT}')

    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    # Geometric mean requires (1+r).prod()
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        # If strategy is flat (e.g. all cash), std is 0. 
        # In competition this raises error, here we might return 0.0 or raise.
        # Let's return 0.0 for safety in eval script.
        return 0.0 
        # raise ParticipantVisibleError('Division by zero, strategy std is zero')

    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')

    # Calculate the volatility penalty
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol

    # Calculate the return penalty
    return_gap = max(
        0,
        (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr,
    )
    return_penalty = 1 + (return_gap**2) / 100

    # Adjust the Sharpe ratio by the volatility and return penalty
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)

# -----------------------------
# Data Loading
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
        raise ValueError(f"Target column '{target}' not found.")

    df = df.rename(columns={target: "target"})
    
    # Drop leakage
    to_drop = [c for c in df.columns if "market_forward_excess_returns" in c]
    if to_drop:
        df = df.drop(columns=to_drop)

    df = add_missing_flags_and_zero_fill(df)

    # Determine Feature Cols (Exclude Target & Risk Free Rate)
    if feature_cols is None:
        feat_cols = [c for c in df.columns if c != "target" and c != "risk_free_rate"]
    else:
        feat_cols = [c for c in feature_cols if c != "risk_free_rate"]

    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])
    df[feat_cols] = df[feat_cols].fillna(0.0)

    return df, feat_cols

# -----------------------------
# Model Loading
# -----------------------------
def load_actorcritic_from_checkpoint(state_path: str | Path, obs_dim: int,
                                     n_actions: int = 5, l2_lambda: float = 0.0,
                                     map_location: str = "cpu") -> ActorCritic:
    
    state_dict = torch.load(state_path, map_location=map_location, weights_only=True)
    
    if "policy_head.weight" not in state_dict:
        raise RuntimeError("Checkpoint missing 'policy_head.weight'")
    
    hidden_w = state_dict["policy_head.weight"].shape[1] 
    saved_n_actions = state_dict["policy_head.weight"].shape[0]

    if saved_n_actions != n_actions:
        print(f"WARNING: Checkpoint has {saved_n_actions} actions, but code requested {n_actions}. Using checkpoint value.")
        n_actions = saved_n_actions

    model = ActorCritic(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden_w, l2_lambda=l2_lambda)
    
    # Validation logic to handle potential key mismatches if class names changed
    if "feature_extractor.0.weight" in state_dict:
        saved_in = state_dict["feature_extractor.0.weight"]
    elif "shared.0.weight" in state_dict:
        saved_in = state_dict["shared.0.weight"]
    else:
        saved_in = None

    if saved_in is not None:
        saved_obs_dim = saved_in.shape[1]
        if saved_obs_dim != obs_dim:
            raise RuntimeError(f"Feature mismatch: Checkpoint={saved_obs_dim}, Data={obs_dim}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# -----------------------------
# Trading Simulation
# -----------------------------
@dataclass
class TradeResult:
    dates: Optional[np.ndarray]
    actions: np.ndarray          
    daily_pnl: np.ndarray        
    cum_pnl: np.ndarray
    hit_rate: float              
    hold_ratio: float            
    target: np.ndarray           
    target_cum: np.ndarray
    comp_score: float            # NEW: Adjusted Sharpe Score

def run_trading(
    df: pd.DataFrame,
    feat_cols: List[str],
    model: ActorCritic,
    date_col: Optional[str] = None,
    csv_path: Optional[str | Path] = None,
    device: str = "cpu"
) -> TradeResult:
    
    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y = df["target"].to_numpy(dtype=np.float32, copy=False)
    
    # Retrieve Risk Free Rate if present, else 0
    if "risk_free_rate" in df.columns:
        rf = df["risk_free_rate"].to_numpy(dtype=np.float32, copy=False)
    else:
        print("WARNING: 'risk_free_rate' column not found. Assuming 0.0 for Sharpe calculation.")
        rf = np.zeros(len(df), dtype=np.float32)

    dates = df[date_col].to_numpy() if date_col and date_col in df.columns else None

    # Map indices to Leverage
    n_actions = model.policy_head.out_features
    action_map = np.linspace(0.0, 2.0, n_actions)

    actions = np.zeros(len(df), dtype=np.float32)
    daily_pnl = np.zeros(len(df), dtype=np.float32)

    # Reset LSTM State
    model.to(device)
    model.reset_hidden_state(batch_size=1, device=device)

    print(f"Running inference on {len(df)} rows...")

    with torch.no_grad():
        for i in range(len(df)):
            x = torch.from_numpy(X[i:i+1]).to(device)
            logits, _ = model(x)
            
            a_idx = torch.argmax(logits, dim=-1).item()
            leverage = float(action_map[a_idx])
            
            actions[i] = leverage
            # Simple PnL for visual check
            daily_pnl[i] = leverage * y[i]

    cum_pnl = np.cumsum(daily_pnl)
    target_cum = np.cumsum(y)

    # --- CALCULATE COMPETITION SCORE ---
    # Construct solution DF
    solution_df = pd.DataFrame({
        "risk_free_rate": rf,
        "forward_returns": y
    })
    # Construct submission DF
    submission_df = pd.DataFrame({
        "prediction": actions
    })
    
    print("Calculating Adjusted Sharpe Score...")
    try:
        comp_score = score(solution_df, submission_df, row_id_column_name="id")
    except Exception as e:
        print(f"Error calculating score: {e}")
        comp_score = 0.0

    # Basic Metrics
    market_up = y > 0
    we_are_invested = actions > 0.1
    hits = (market_up == we_are_invested)
    hit_rate = float(np.mean(hits))
    hold_ratio = float(np.mean(actions < 0.1))

    # Save to CSV
    if csv_path:
        out = {
            "leverage": actions,
            "target": y,
            "risk_free": rf,
            "daily_pnl": daily_pnl,
            "cum_pnl": cum_pnl
        }
        if dates is not None: out["date"] = dates
        pd.DataFrame(out).to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

    return TradeResult(
        dates=dates, actions=actions, daily_pnl=daily_pnl, cum_pnl=cum_pnl,
        hit_rate=hit_rate, hold_ratio=hold_ratio, target=y, target_cum=target_cum,
        comp_score=comp_score
    )

# -----------------------------
# Plotting
# -----------------------------
def plot_cum_target_vs_strategy(result: TradeResult, title: str = "Cumulative Return"):
    plt.figure(figsize=(12, 6))
    x_axis = result.dates if result.dates is not None else np.arange(len(result.cum_pnl))
    
    plt.plot(x_axis, result.target_cum, label="Market (Buy & Hold 1x)", color='gray', alpha=0.6)
    plt.plot(x_axis, result.cum_pnl, label=f"RL Strategy (Score: {result.comp_score:.4f})", color='blue', linewidth=1.5)
    
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
    # --- CONFIG ---
    CKPT_PATH = r"C:\Year4\4AL3\final-project\4AL3\checkpoints\LSTM2_150\checkpoints\ppo_iter1500.pt" 
    DATA_PATH = r"C:\Year4\4AL3\final-project\4AL3\src\RL\data\train.csv"
    TARGET_COL = "forward_returns"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # --------------

    print(f"Loading data from {DATA_PATH}...")
    df, feat_cols = load_data(DATA_PATH, target=TARGET_COL)
    obs_dim = len(feat_cols)
    print(f"Data loaded. Obs Dim: {obs_dim}")

    print(f"Loading model from {CKPT_PATH}...")
    try:
        model = load_actorcritic_from_checkpoint(CKPT_PATH, obs_dim=obs_dim, n_actions=5, map_location=DEVICE)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found.")
        exit()

    res = run_trading(df, feat_cols, model, csv_path="eval_results.csv", device=DEVICE)

    mse = float(np.mean((res.target - res.daily_pnl) ** 2))

    print("-" * 40)
    print(f"Avg Leverage:       {np.mean(res.actions):.2f}x")
    print(f"Hit-rate:           {res.hit_rate:.4f}")
    print(f"Cash Ratio:         {res.hold_ratio:.4f}")
    print("-" * 40)
    print(f"Market Return:      {res.target_cum[-1]:.6f}")
    print(f"Strategy Ret:       {res.cum_pnl[-1]:.6f}")
    print(f"Adj Sharpe Score:   {res.comp_score:.6f}") # The official metric
    print(f"MSE:                {mse:.6f}")
    print("-" * 40)

    plot_cum_target_vs_strategy(res, title="Final Evaluation: RL vs Market")