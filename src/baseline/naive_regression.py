# ===== Polynomial Ridge regression (L2) =====
from typing import Optional, List
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

def poly_ridge_baseline(df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
                        degree: int = 3, alpha: float = 1e-2):
    df = df.copy()
    for c in df.columns:
        if df[c].isna().any():
            df[f"{c}_missing"] = df[c].isna().astype(float)
            df[c] = df[c].fillna(0.0)
    assert "target" in df.columns, "DataFrame must contain 'target'."

    X = (df.drop(columns=["target"]) if feature_cols is None else df[feature_cols]).to_numpy(np.float32)
    y = df["target"].to_numpy(np.float32)

    cut = int(0.8 * len(df))
    Xtr, Xva = X[:cut], X[cut:];  ytr, yva = y[:cut], y[cut:]

    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=alpha))  # L2 regularizer strength
    ])
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xva)
    mse = float(np.mean((yhat - yva) ** 2))
    acc = float(np.mean(np.sign(yhat) == np.sign(yva)))
    print(f"[Poly+Ridge d={degree}, alpha={alpha}] MSE={mse:.6f}  sign-acc={acc:.4f}  (n_val={len(yva)})")
    return pipe
