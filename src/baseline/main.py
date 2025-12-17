# baseline_main.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

########################################
# CONFIG — EDIT THESE
########################################
DATA_PATH = r"C:\Year4\4AL3\final-project\4AL3\src\baseline\data\train.csv"     # set me
TARGET_COL = "forward_returns"     # will be renamed to 'target'
FEATURE_COLS = None                # None => use all non-target columns

# MLP settings
MLP_HIDDEN = 128
MLP_LR = 1e-3
MLP_L2 = 1e-4
MLP_EPOCHS = 300
MLP_BATCH = 256

# Polynomial Ridge settings
POLY_DEGREE = 3
RIDGE_ALPHA = 1e-2

# Gradient Boosted Tree settings
GBRT_EST = 300
GBRT_LR = 0.05
GBRT_DEPTH = 30
########################################

def add_missing_flags_and_zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(float)
            df[col] = df[col].fillna(0.0)
    return df

def load_data(path: str | Path, target: str = "forward_returns") -> pd.DataFrame:
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
    return df

def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].isna().any():
            df[f"{c}_missing"] = df[c].isna().astype(float)
            df[c] = df[c].fillna(0.0)
    return df


# =========================================================
#  1) POLYNOMIAL RIDGE
# =========================================================
def run_poly(df):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import Ridge

    X = (df.drop(columns=["target"]) if FEATURE_COLS is None else df[FEATURE_COLS]).to_numpy(np.float32)
    y = df["target"].to_numpy(np.float32)

    cut = int(0.8 * len(df))
    Xtr, Xva = X[:cut], X[cut:]
    ytr, yva = y[:cut], y[cut:]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("ridge", Ridge(alpha=RIDGE_ALPHA))
    ])
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xva)

    mse = np.mean((yhat - yva) ** 2)
    acc = np.mean(np.sign(yhat) == np.sign(yva))
    print(f"[POLY] MSE={mse:.6f}, sign-acc={acc:.4f}")
    return pipe


# =========================================================
#  2) GRADIENT BOOSTED TREES
# =========================================================
def run_gbr(df):
    from sklearn.ensemble import GradientBoostingRegressor

    X = (df.drop(columns=["target"]) if FEATURE_COLS is None else df[FEATURE_COLS]).to_numpy(np.float32)
    y = df["target"].to_numpy(np.float32)

    cut = int(0.8 * len(df))
    Xtr, Xva = X[:cut], X[cut:]
    ytr, yva = y[:cut], y[cut:]

    model = GradientBoostingRegressor(
        n_estimators=GBRT_EST,
        learning_rate=GBRT_LR,
        max_depth=GBRT_DEPTH,
        subsample=0.9,
        random_state=42
    )
    model.fit(Xtr, ytr)
    yhat = model.predict(Xva)

    mse = np.mean((yhat - yva) ** 2)
    acc = np.mean(np.sign(yhat) == np.sign(yva))
    print(f"[GBRT] MSE={mse:.6f}, sign-acc={acc:.4f}")
    return model


# =========================================================
#  3) MLP (PyTorch)
# =========================================================
def run_mlp(df):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    def standardize_fit(X):
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-8
        return (X - mu) / sd, mu, sd

    def standardize_apply(X, mu, sd):
        return (X - mu) / sd

    class MLP(nn.Module):
        def __init__(self, d_in, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    X = (df.drop(columns=["target"]) if FEATURE_COLS is None else df[FEATURE_COLS]).to_numpy(np.float32)
    y = df["target"].to_numpy(np.float32)

    cut = int(0.8 * len(df))
    Xtr, Xva = X[:cut], X[cut:]
    ytr, yva = y[:cut], y[cut:]

    Xtr, mu, sd = standardize_fit(Xtr)
    Xva = standardize_apply(Xva, mu, sd)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(d_in=Xtr.shape[1], hidden=MLP_HIDDEN).to(device)
    opt = optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_L2)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)

    for _ in range(MLP_EPOCHS):
        idx = torch.randperm(len(Xtr_t), device=device)
        for start in range(0, len(Xtr_t), MLP_BATCH):
            batch_idx = idx[start:start+MLP_BATCH]
            pred = model(Xtr_t[batch_idx])
            loss = loss_fn(pred, ytr_t[batch_idx])
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        yhat = model(torch.tensor(Xva, device=device)).cpu().numpy()

    mse = np.mean((yhat - yva) ** 2)
    acc = np.mean(np.sign(yhat) == np.sign(yva))
    print(f"[MLP] MSE={mse:.6f}, sign-acc={acc:.4f}")
    return model


# =========================================================
# MAIN
# =========================================================
def main():
    df = load_data(path=DATA_PATH)
    print("Running baselines on:", DATA_PATH)
    run_gbr(df)
    run_mlp(df)

if __name__ == "__main__":
    main()
