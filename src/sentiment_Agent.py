import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt
from util import load_data, preprocess_data, run_tree_models, discretize_target
import math

'''
hypothesis:

Sentiments are accumulative over time, weak correlation between sentiments and next day return
with the exception for extreme sentiments
'''

def plot_S_features_vs_target(df: pd.DataFrame, target: str = "target",
                              pattern: str = r"^S\d+$", sample: int | None = None,
                              kind: str = "scatter"):
    """
    Plot all columns matching `pattern` (e.g., S1,S2,...) on X vs `target` on Y.
    kind: 'scatter' or 'hexbin' (use hexbin for big data).
    """
    s_cols = [c for c in df.columns if re.match(pattern, c)]
    if not s_cols:
        raise ValueError("No columns matching pattern (e.g., S1, S2, ...).")

    d = df if sample is None or len(df) <= sample else df.sample(sample, random_state=0)
    n = len(s_cols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
    axes = axes.ravel()

    for i, col in enumerate(s_cols):
        ax = axes[i]
        g = d[[col, target]].dropna()
        if kind == "hexbin":
            ax.hexbin(g[col], g[target], gridsize=35)
        else:
            ax.plot(g[col], g[target], linestyle="none", marker="o", markersize=2, alpha=0.6)
        ax.set_xlabel(col); ax.set_ylabel(target); ax.set_title(f"{col} vs {target}")

    for j in range(i+1, len(axes)):  # hide unused axes
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def find_sentiment_features(df: pd.DataFrame, pattern: str = r"^S\d+$") -> list[str]:
    """
    Return list of columns matching `pattern` (e.g., S1,S2,...).
    """
    s_cols = [c for c in df.columns if re.match(pattern, c)]
    if not s_cols:
        raise ValueError("No columns matching pattern (e.g., S1, S2, ...).")
    return s_cols

def normalize_target(df: pd.DataFrame, target: str = "target") -> pd.DataFrame:
    """
    Normalize target column to [0,1].
    """
    df = df.copy()
    min_val = df[target].min()
    max_val = df[target].max()
    if min_val == max_val:
        raise ValueError("Target column has constant value.")
    df[target] = (df[target] - min_val) / (max_val - min_val)
    return df


def main():
    # Example usage
    df = load_data(r"C:\Year4\4AL3\final-project\4AL3\data\Hull Market\train.csv")
    df = preprocess_data(df, mode="y")

    s_features = find_sentiment_features(df)
    print("Sentiment features found:", s_features)
    run_tree_models(df, s_features, target="target", test_size=0.3, tree_type="gbr")
if __name__ == "__main__":
    main()
    