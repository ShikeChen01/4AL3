#!/usr/bin/env python3
import argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dataset CSV")
    ap.add_argument("--target", required=True, help="Target column name")
    ap.add_argument("--n", type=int, default=5, help="Top-N features by |Pearson r|")
    ap.add_argument("--sample", type=int, default=0, help="Optional row cap for speed (0 = all)")
    ap.add_argument("--out", default="top_corr.png", help="Output figure path")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.drop(columns=["market_forward_excess_returns"], inplace=True)
    if args.sample and args.sample > 0 and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)

    # keep numeric columns only
    num_df = df.select_dtypes(include=[np.number]).copy()
    if args.target not in num_df.columns:
        raise ValueError(f"Target '{args.target}' must be numeric and present in the data.")

    # compute Pearson correlations with target (drop rows with NaNs per pair)
    corrs = {}
    y = num_df[args.target]
    for col in num_df.columns:
        if col == args.target: 
            continue
        s = pd.concat([num_df[col], y], axis=1).dropna()
        if len(s) >= 2 and s[col].std(ddof=0) > 0 and s[args.target].std(ddof=0) > 0:
            corrs[col] = s[col].corr(s[args.target])
    if not corrs:
        raise ValueError("No valid numeric features found to correlate with the target.")

    top = pd.Series(corrs).reindex(pd.Series(corrs).abs().sort_values(ascending=False).index)[:args.n]

    # plot
    k = len(top)
    rows = 2 if k > 3 else 1
    cols = math.ceil(k / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
    axes = axes.flatten()

    for ax, (col, r) in zip(axes, top.items()):
        pair = df[[col, args.target]].dropna()
        x = pair[col].to_numpy()
        yv = pair[args.target].to_numpy()
        ax.scatter(x, yv, s=10, alpha=0.5)
        # best-fit line
        if len(x) >= 2:
            m, b = np.polyfit(x, yv, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, m*xs + b, linewidth=2)
        ax.set_title(f"{col}  (r = {r:.3f})")
        ax.set_xlabel(col)
        ax.set_ylabel(args.target)
        ax.grid(True, alpha=0.3)

    # clean empty axes
    for j in range(k, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    # plt.show()  # uncomment if running interactively

if __name__ == "__main__":
    main()
