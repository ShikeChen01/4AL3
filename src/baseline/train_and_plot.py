# run_trading_baselines.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

# -----------------------------
# Data utils (features scaled, target raw)
# -----------------------------
def add_missing_flags_and_zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(float)
            df[col] = df[col].fillna(0.0)
    return df

def load_data(path: str | Path, target: str = "forward_returns",
              feature_cols: Optional[List[str]] = None
             ) -> Tuple[pd.DataFrame, List[str]]:
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
    # Drop known leakage if present
    to_drop = [c for c in df.columns if ("market_forward_excess_returns" in c) or ("risk_free_rate" in c)]
    if to_drop:
        df = df.drop(columns=to_drop)

    df = add_missing_flags_and_zero_fill(df)

    feat_cols = feature_cols if feature_cols is not None else [c for c in df.columns if c != "target"]
    scaler = StandardScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])  # only features scaled
    return df, feat_cols

# -----------------------------
# Trading sim (same rule as PPO version)
# -----------------------------
@dataclass
class TradingEval:
    name: str
    actions: np.ndarray
    daily_pnl: np.ndarray
    cum_pnl: np.ndarray
    mse_vs_target: float

def simulate_trading_from_preds(pred: np.ndarray, y: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decision rule on prediction (regresses next return):
      if pred > +eps -> buy (+1); if pred < -eps -> sell (-1); else hold (0)
    Per-step PnL uses RAW target y:
      buy -> +y, sell -> -y, hold -> 0
    """
    actions = np.zeros_like(pred, dtype=np.int8)
    actions[pred >  eps] =  1
    actions[pred < -eps] = -1

    daily_pnl = np.where(actions == 1, y, np.where(actions == -1, -y, 0.0))
    cum_pnl = np.cumsum(daily_pnl)
    return actions, daily_pnl, cum_pnl

# -----------------------------
# Plotting
# -----------------------------
def plot_cumulative(df_dates: Optional[np.ndarray],
                    target_cum: np.ndarray,
                    model_curves: Dict[str, np.ndarray],
                    title: str):
    plt.figure(figsize=(12, 4))
    x = df_dates if df_dates is not None else np.arange(len(target_cum))
    plt.plot(x, target_cum, label="Raw Target Cum Return", linewidth=2)
    for name, series in model_curves.items():
        plt.plot(x, series, label=f"{name} Cum Return")
    if df_dates is not None:
        plt.xticks(rotation=45)
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
    # ---- Edit these paths/settings ----
    DATA_PATH = r"C:\Year4\4AL3\final-project\4AL3\result\RL_1\data\train.csv"
    TARGET_COL = "forward_returns"
    DATE_COL: Optional[str] = None    # e.g., "date" if present
    EPS = 0.1                         # hold band
    # -----------------------------------

    # Load
    df, feat_cols = load_data(DATA_PATH, target=TARGET_COL)
    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y = df["target"].to_numpy(dtype=np.float32, copy=False)  # RAW target (e.g., S&P 500 forward return)
    dates = df[DATE_COL].to_numpy() if DATE_COL and DATE_COL in df.columns else None
    target_cum = np.cumsum(y)

    # Simple models
    models: Dict[str, object] = {
        "MLP_small": MLPRegressor(hidden_layer_sizes=(64,),
                                  activation="relu",
                                  solver="adam",
                                  alpha=1e-4,
                                  learning_rate_init=1e-3,
                                  max_iter=300,
                                  random_state=42,
                                  verbose=False),
        "MLP_big":   MLPRegressor(hidden_layer_sizes=(128, 64),
                                  activation="relu",
                                  solver="adam",
                                  alpha=1e-4,
                                  learning_rate_init=1e-3,
                                  max_iter=400,
                                  random_state=43,
                                  verbose=False),
        "GBT":       GradientBoostingRegressor(n_estimators=300,
                                              learning_rate=0.05,
                                              max_depth=5,
                                              subsample=1.0,
                                              random_state=44),
    }

    evals: Dict[str, TradingEval] = {}
    curves: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        model.fit(X, y)  # train on all data (simple baseline)
        pred = model.predict(X).astype(np.float32)

        actions, daily_pnl, cum_pnl = simulate_trading_from_preds(pred, y, EPS)
        mse = float(np.mean((y - daily_pnl) ** 2))

        evals[name] = TradingEval(
            name=name,
            actions=actions,
            daily_pnl=daily_pnl,
            cum_pnl=cum_pnl,
            mse_vs_target=mse
        )
        curves[name] = cum_pnl

    # Print metrics
    print("=== Baseline Trading (per-step MSE vs RAW target) ===")
    print(f"Final cumulative RAW target: {target_cum[-1]:.6f}")
    for name, ev in evals.items():
        buys = int((ev.actions == 1).sum())
        sells = int((ev.actions == -1).sum())
        holds = int((ev.actions == 0).sum())
        non_hold = ev.actions != 0
        hit = float(np.mean(np.sign(ev.actions[non_hold]) == np.sign(y[non_hold]))) if non_hold.any() else float("nan")
        print(f"[{name:9s}]  MSE: {ev.mse_vs_target:.8f} | "
              f"Final cum: {ev.cum_pnl[-1]:.6f} | "
              f"Trades  buy/sell/hold: {buys}/{sells}/{holds} | "
              f"Hit-rate: {hit:.4f}")

    # Plot
    plot_cumulative(dates, target_cum, curves, title=f"Cumulative Return (Raw Target vs Baselines, ε={EPS})")
