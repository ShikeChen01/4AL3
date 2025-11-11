# ===== Gradient Boosted Trees (sklearn) =====
from typing import Optional, List
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

def gbr_baseline(df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
                 n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 3):
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

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        random_state=42
    )
    model.fit(Xtr, ytr)
    yhat = model.predict(Xva)
    mse = float(np.mean((yhat - yva) ** 2))
    acc = float(np.mean(np.sign(yhat) == np.sign(yva)))
    print(f"[GBR] MSE={mse:.6f}  sign-acc={acc:.4f}  (n_val={len(yva)})")
    return model
