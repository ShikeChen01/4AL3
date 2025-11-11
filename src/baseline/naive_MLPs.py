# ===== MLP baseline (PyTorch) =====
from typing import Optional, List, Tuple
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim

def _add_missing_flags_and_zero_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].isna().any():
            df[f"{c}_missing"] = df[c].isna().astype(float)
            df[c] = df[c].fillna(0.0)
    return df

def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd

def _standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd

class _MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def mlp_baseline(df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
                 epochs: int = 20, batch: int = 256, lr: float = 1e-3, l2: float = 1e-4) -> _MLP:
    df = _add_missing_flags_and_zero_fill(df)
    assert "target" in df.columns, "DataFrame must contain 'target'."

    X = (df.drop(columns=["target"]) if feature_cols is None else df[feature_cols]).to_numpy(np.float32)
    y = df["target"].to_numpy(np.float32)

    # time-based 80/20 split
    cut = int(0.8 * len(df))
    Xtr, Xva = X[:cut], X[cut:];  ytr, yva = y[:cut], y[cut:]

    # standardize on train only
    Xtr, mu, sd = _standardize_fit(Xtr)
    Xva = _standardize_apply(Xva, mu, sd)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _MLP(d_in=Xtr.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)  # L2 regularizer
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    n = len(Xtr_t)

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s+batch]
            pred = model(Xtr_t[idx])
            loss = loss_fn(pred, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    # eval
    model.eval()
    with torch.no_grad():
        yhat = model(torch.tensor(Xva, dtype=torch.float32, device=device)).cpu().numpy()
    mse = float(np.mean((yhat - yva) ** 2))
    acc = float(np.mean(np.sign(yhat) == np.sign(yva)))
    print(f"[MLP] MSE={mse:.6f}  sign-acc={acc:.4f}  (n_val={len(yva)})")
    return model
