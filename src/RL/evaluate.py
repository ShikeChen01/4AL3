from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import argparse
import os

import numpy as np
import pandas as pd
import pandas.api.types
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Model Definition (MLP)
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128, l2_lambda: float = 0.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)  # logits
        self.value_head = nn.Linear(hidden, 1)
        self.l2_lambda = l2_lambda

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def l2_reg(self) -> torch.Tensor:
        if self.l2_lambda <= 0:
            return torch.zeros((), device=next(self.parameters()).device)
        l2 = torch.tensor(0.0, device=next(self.parameters()).device)
        for p in self.parameters():
            l2 = l2 + torch.sum(p * p)
        return self.l2_lambda * l2

# -----------------------------
# 2. Competition Scoring Function
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

    # Validation checks
    if solution['position'].max() > MAX_INVESTMENT + 1e-5:
        print(f"WARNING: Position max {solution['position'].max()} exceeds limit.")
    if solution['position'].min() < MIN_INVESTMENT - 1e-5:
         print(f"WARNING: Position min {solution['position'].min()} below limit.")

    solution['strategy_returns'] = solution['risk_free_rate'] * (1 - solution['position']) + solution['position'] * solution['forward_returns']

    # Calculate strategy's Sharpe ratio
    strategy_excess_returns = solution['strategy_returns'] - solution['risk_free_rate']
    strategy_excess_cumulative = (1 + strategy_excess_returns).prod()
    strategy_mean_excess_return = (strategy_excess_cumulative) ** (1 / len(solution)) - 1
    strategy_std = solution['strategy_returns'].std()

    trading_days_per_yr = 252
    if strategy_std == 0:
        return 0.0 # Handle division by zero gracefully for evaluation

    sharpe = strategy_mean_excess_return / strategy_std * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100)

    # Calculate market return and volatility
    market_excess_returns = solution['forward_returns'] - solution['risk_free_rate']
    market_excess_cumulative = (1 + market_excess_returns).prod()
    market_mean_excess_return = (market_excess_cumulative) ** (1 / len(solution)) - 1
    market_std = solution['forward_returns'].std()

    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100)

    if market_volatility == 0:
        return 0.0

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
# 3. Data Loading
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
    drop_first_n: int = 4000, 
    drop_last_n: int = 3500
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

    # Extract Risk Free Rate before dropping
    if "risk_free_rate" in df.columns:
        risk_free_series = df["risk_free_rate"].copy()
    else:
        risk_free_series = pd.Series(0.0, index=df.index)

    # Drop leakage/irrelevant columns
    to_drop = [c for c in df.columns if ("market_forward_excess_returns" in c) or ("risk_free_rate" in c)]
    if to_drop:
        df = df.drop(columns=to_drop)

    df = add_missing_flags_and_zero_fill(df)

    if feature_cols is None:
        feat_cols = [c for c in df.columns if c != "target"]
    else:
        feat_cols = feature_cols

    # Save raw for returning (attach RF back)
    df_raw = df.copy(deep=True)
    df_raw["risk_free_rate"] = risk_free_series

    # Scale features
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])

    # Fix fragmentation
    df = df.copy(deep=True)
    df["risk_free_rate"] = risk_free_series # Re-attach for evaluation access if needed
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    if drop_first_n > 0:
        df = df.iloc[drop_first_n:]
    if drop_last_n > 0:
        df = df.iloc[:-drop_last_n]
    print(f"Data after dropping: {df.shape[0]} rows, {df.shape[1]} columns.")

    return df_raw, df, scaler

# -----------------------------
# 4. Evaluation Loop
# -----------------------------
def evaluate(df: pd.DataFrame, model: ActorCritic, feature_cols: List[str], device: str):
    print(f"Evaluating on {len(df)} rows...")
    
    X = df[feature_cols].values.astype(np.float32)
    y_target = df["target"].values.astype(np.float32)
    risk_free = df["risk_free_rate"].values.astype(np.float32)
    
    # 1. Get Model Predictions
    with torch.no_grad():
        obs_tensor = torch.tensor(X, device=device)
        logits, _ = model(obs_tensor)
        actions_idx = torch.argmax(logits, dim=-1).cpu().numpy()

    # 2. Apply Custom "Long-Only" Mapping
    # Original: 0 -> -1 (Short), 1 -> 0 (Flat), 2 -> 1 (Long)
    # Requested: -1 -> 0 (Switch Short to Flat)
    # New Map: 0 -> 0, 1 -> 0, 2 -> 1
    
    # Create the map
    # Index 0 was Short (-1) -> Become 0
    # Index 1 was Flat (0)  -> Remain 0
    # Index 2 was Long (1)  -> Remain 1
    mapping = {0: 0.0, 1: 0.0, 2: 1.0}
    
    predictions = np.vectorize(mapping.get)(actions_idx)
    
    # 3. Calculate Sharpe Ratio
    solution_df = pd.DataFrame({
        "forward_returns": y_target,
        "risk_free_rate": risk_free
    })
    submission_df = pd.DataFrame({
        "prediction": predictions
    })
    
    sharpe_score = score(solution_df, submission_df, row_id_column_name="id")

    # 4. Calculate PnL Curves for Plotting
    # Strategy PnL = RF * (1 - pos) + pos * return
    strategy_returns = risk_free * (1 - predictions) + predictions * y_target
    
    # Native Buy = Sum of Target (Cumulative sum of raw returns)
    native_buy_cum = np.cumsum(y_target)
    strategy_cum = np.cumsum(strategy_returns)

    # 5. Print Results
    unique, counts = np.unique(predictions, return_counts=True)
    dist = dict(zip(unique, counts))
    
    print("-" * 40)
    print(" EVALUATION RESULTS (Long-Only Modified)")
    print("-" * 40)
    print(f"Sharpe Ratio:       {sharpe_score:.6f}")
    print(f"Total Native Ret:   {native_buy_cum[-1]:.4f}")
    print(f"Total Strategy Ret: {strategy_cum[-1]:.4f}")
    print(f"Action Distribution: {dist}")
    print("-" * 40)

    # 6. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(native_buy_cum, label="Native Buy (Sum of Target)", color='gray', alpha=0.6)
    plt.plot(strategy_cum, label=f"RL Strategy (Long-Only, Sharpe={sharpe_score:.2f})", color='blue', linewidth=1.5)
    plt.title("Cumulative Returns: Strategy vs Native Buy")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # DEFAULTS - Update these as needed
    DEFAULT_DATA = r"C:\Year4\4AL3\final-project\4AL3\src\RL\data\train.csv"
    DEFAULT_CKPT = r"C:\Year4\4AL3\final-project\4AL3\checkpoints\temp0\ppo_full.pt"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DEFAULT_DATA)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-col", type=str, default="forward_returns")
    args = parser.parse_args()

    # Load Data
    print(f"Loading data from {args.data}...")
    _, df_proc, scaler = load_data(args.data, target_col=args.target_col)
    
    feature_cols = [c for c in df_proc.columns if c not in ["target", "risk_free_rate"]]
    obs_dim = len(feature_cols)
    print(f"Features loaded: {obs_dim}")

    # Load Model (MLP)
    model = ActorCritic(obs_dim=obs_dim, n_actions=3, hidden=128).to(args.device)
    if os.path.exists(args.ckpt):
        print(f"Loading weights from {args.ckpt}...")
        state_dict = torch.load(args.ckpt, map_location=args.device)
        if list(state_dict.keys())[0].startswith("model."):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print(f"ERROR: Checkpoint {args.ckpt} not found.")
        exit()

    # Evaluate
    evaluate(df_proc, model, feature_cols, args.device)