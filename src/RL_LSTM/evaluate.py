# evaluate.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import os
import argparse

import numpy as np
import pandas as pd
import pandas.api.types 
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from A1 import ActorCritic

# -----------------------------
# Competition Scoring Function
# -----------------------------
MIN_INVESTMENT = 0
MAX_INVESTMENT = 2

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    # ... (Same as before) ...
    solution = solution.copy()
    submission = submission.copy()
    if not pandas.api.types.is_numeric_dtype(submission['prediction']):
         try: submission['prediction'] = submission['prediction'].astype(float)
         except: raise Exception('Predictions must be numeric')
    solution['position'] = submission['prediction']
    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()
    trading_days_per_yr = 252
    if strategy_std == 0: return 0.0 
    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)
    if market_volatility == 0: return 0.0
    excess_vol = max(0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0
    vol_penalty = 1 + excess_vol
    return_gap = max(0, (market_mean_excess_return - strategy_mean_excess_return) * 100 * trading_days_per_yr)
    return_penalty = 1 + (return_gap**2) / 100
    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000)

# -----------------------------
# Data Loading (Updated)
# -----------------------------
def add_missing_flags_and_zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(float)
            df[col] = df[col].fillna(0.0)
    return df

def load_data(
    data_path: str, 
    target_col: str = "forward_returns", 
    feature_cols: Optional[List[str]] = None,
    ignore_first_n: int = 0, # Drop from start
    ignore_last_n: int = 0   # <--- NEW: Drop from end
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    
    path = Path(data_path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        try:
            df = pd.read_csv(path, engine="pyarrow")
        except Exception:
            df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    df = df.rename(columns={target_col: "target"})

    # Extract Risk Free Rate
    if "risk_free_rate" in df.columns:
        risk_free_series = df["risk_free_rate"].copy()
    else:
        print("Warning: 'risk_free_rate' not found. Creating 0-filled column.")
        risk_free_series = pd.Series(0.0, index=df.index)

    # Drop leakage
    to_drop = [c for c in df.columns if ("market_forward_excess_returns" in c) or ("risk_free_rate" in c)]
    if to_drop:
        df = df.drop(columns=to_drop)

    df = add_missing_flags_and_zero_fill(df)

    if feature_cols is None:
        feat_cols = [c for c in df.columns if c != "target"]
    else:
        feat_cols = feature_cols

    # Save raw
    df_raw = df.copy(deep=True)
    df_raw["risk_free_rate"] = risk_free_series

    # Scale features
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])

    # Re-attach Risk Free Rate for Evaluation
    df["risk_free_rate"] = risk_free_series 
    
    # --- SLICING LOGIC ---
    # 1. Drop from Start
    if ignore_first_n > 0:
        print(f"Load Data: Dropping first {ignore_first_n} rows.")
        df = df.iloc[ignore_first_n:]
        df_raw = df_raw.iloc[ignore_first_n:]

    # 2. Drop from End
    if ignore_last_n > 0:
        print(f"Load Data: Dropping last {ignore_last_n} rows.")
        df = df.iloc[:-ignore_last_n]
        df_raw = df_raw.iloc[:-ignore_last_n]
    
    # 3. Reset Index
    df = df.reset_index(drop=True)
    df_raw = df_raw.reset_index(drop=True)
    
    return df_raw, df, scaler


# -----------------------------
# Model Loading (Unchanged)
# -----------------------------
def load_actorcritic_from_checkpoint(state_path: str | Path, obs_dim: int,
                                     n_actions: int = 5, l2_lambda: float = 0.0,
                                     map_location: str = "cpu") -> ActorCritic:
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Checkpoint not found: {state_path}")
    state_dict = torch.load(state_path, map_location=map_location)
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    key = "policy_head.weight"
    if key in state_dict:
        hidden_w = state_dict[key].shape[1] 
        saved_n_actions = state_dict[key].shape[0]
        if saved_n_actions != n_actions:
            print(f"WARNING: Checkpoint has {saved_n_actions} actions, but code requested {n_actions}. Using checkpoint value.")
            n_actions = saved_n_actions
    else:
        hidden_w = 256 
    model = ActorCritic(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden_w, l2_lambda=l2_lambda)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# -----------------------------
# Trading Simulation (Unchanged)
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
    comp_score: float            
    std_sharpe: float            

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
    rf = df["risk_free_rate"].to_numpy(dtype=np.float32, copy=False)
    dates = df[date_col].to_numpy() if date_col and date_col in df.columns else None

    n_actions = model.policy_head.out_features
    action_map = np.linspace(0.0, 2, n_actions)

    actions = np.zeros(len(df), dtype=np.float32)
    daily_pnl = np.zeros(len(df), dtype=np.float32)

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
            daily_pnl[i] = leverage * y[i]

    cum_pnl = np.cumsum(daily_pnl)
    target_cum = np.cumsum(y)

    solution_df = pd.DataFrame({"risk_free_rate": rf, "forward_returns": y})
    submission_df = pd.DataFrame({"prediction": actions})
    
    print("Calculating Adjusted Sharpe Score...")
    try: comp_score = score(solution_df, submission_df, row_id_column_name="id")
    except Exception as e: print(f"Error calculating score: {e}"); comp_score = 0.0

    strat_ret = rf * (1 - actions) + actions * y
    excess_ret = strat_ret - rf
    avg_excess = np.mean(excess_ret)
    std_excess = np.std(excess_ret)
    std_sharpe = (avg_excess / std_excess) * np.sqrt(252) if std_excess > 1e-9 else 0.0

    hits = ((y > 0) == (actions > 0.1))
    hit_rate = float(np.mean(hits))
    hold_ratio = float(np.mean(actions < 0.1))

    if csv_path:
        out = {"leverage": actions, "target": y, "risk_free": rf, "daily_pnl": daily_pnl, "cum_pnl": cum_pnl}
        if dates is not None: out["date"] = dates
        pd.DataFrame(out).to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

    return TradeResult(dates, actions, daily_pnl, cum_pnl, hit_rate, hold_ratio, y, target_cum, comp_score, std_sharpe)

def plot_cum_target_vs_strategy(result: TradeResult, title: str = "Cumulative Return"):
    plt.figure(figsize=(12, 6))
    x_axis = result.dates if result.dates is not None else np.arange(len(result.cum_pnl))
    plt.plot(x_axis, result.target_cum, label="Market (Buy & Hold 1x)", color='gray', alpha=0.6)
    plt.plot(x_axis, result.cum_pnl, label=f"RL Strategy (Adj Sharpe: {result.comp_score:.2f})", color='blue', linewidth=1.5)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main (Updated)
# -----------------------------
if __name__ == "__main__":
    # --- CONFIG ---
    CKPT_PATH = r"C:\Year4\4AL3\final-project\4AL3\checkpoints\LSTM_Wv2_1\RL_LSTM\checkpoints\ppo_iter1000.pt" 
    DATA_PATH = r"C:\Year4\4AL3\final-project\4AL3\src\RL\data\train.csv"
    TARGET_COL = "forward_returns"
    
    # Configurable Defaults
    DEFAULT_IGNORE_START = 4000 
    DEFAULT_IGNORE_LAST = 0
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_PATH)
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # New Arguments
    parser.add_argument("--ignore-first-n", type=int, default=DEFAULT_IGNORE_START, help="Ignore first N rows (Burn-in)")
    parser.add_argument("--ignore-last-n", type=int, default=DEFAULT_IGNORE_LAST, help="Ignore last N rows (Hide recent data)")
    parser.add_argument("--test-size-m", type=int, default=0, help="Reserve last M rows for evaluation (0 = use all remaining)")
    
    args = parser.parse_args()
    # --------------

    print(f"Loading data from {args.data}...")
    
    # Pass both parameters to load_data
    _, df_full, scaler = load_data(
        args.data, 
        target_col=TARGET_COL, 
        ignore_first_n=args.ignore_first_n,
        ignore_last_n=args.ignore_last_n
    )
    
    feat_cols = [c for c in df_full.columns if c not in ["target", "risk_free_rate"]]
    
    # --- EVAL SLICING LOGIC ---
    # At this point, df_full has already been trimmed of first N and last N.
    # Now we decide how much of the *remaining* data to evaluate.
    
    m = args.test_size_m
    
    if m > 0:
        print(f"Slicing: Taking last {m} rows of the remaining set for evaluation.")
        if m > len(df_full):
             print(f"Warning: Requested test size {m} is larger than available data {len(df_full)}. Using all data.")
             df_eval = df_full
        else:
             df_eval = df_full.iloc[-m:].reset_index(drop=True)
    else:
        print(f"Using all loaded data ({len(df_full)} rows) for evaluation.")
        df_eval = df_full
        
    print(f"Eval Data Shape: {df_eval.shape}")
    obs_dim = len(feat_cols)

    print(f"Loading model from {args.ckpt}...")
    try:
        model = load_actorcritic_from_checkpoint(args.ckpt, obs_dim=obs_dim, n_actions=5, map_location=args.device)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.ckpt}")
        exit()

    res = run_trading(df_eval, feat_cols, model, csv_path="eval_results.csv", device=args.device)

    mse = float(np.mean((res.target - res.daily_pnl) ** 2))

    print("-" * 40)
    print(" EVALUATION RESULTS")
    print("-" * 40)
    print(f"Avg Leverage:       {np.mean(res.actions):.2f}x")
    print(f"Hit-rate:           {res.hit_rate:.4f}")
    print(f"Cash Ratio:         {res.hold_ratio:.4f}")
    print("-" * 40)
    print(f"Market Return:      {res.target_cum[-1]:.4f}")
    print(f"Strategy Return:    {res.cum_pnl[-1]:.4f}")
    print(f"Standard Sharpe:    {res.std_sharpe:.4f}")
    print(f"Adj Sharpe Score:   {res.comp_score:.6f}") 
    print(f"MSE:                {mse:.6f}")
    print("-" * 40)

    plot_cum_target_vs_strategy(res, title="Final Evaluation: RL vs Market")