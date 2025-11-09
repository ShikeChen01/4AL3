#!/usr/bin/env python3
"""
TDA from CSV (Persistent Homology with Ripser)

Usage (examples):
  python tda_from_csv.py --csv /path/to/data.csv
  python tda_from_csv.py --csv data.csv --columns x,y,z --standardize
  python tda_from_csv.py --csv data.csv --exclude id,label --pca-dim 10 --maxdim 2 --sample 2000

Requires:
  pip install pandas numpy matplotlib scikit-learn ripser persim

Outputs (saved next to the CSV unless --outdir is given):
  - persistence_diagrams.npy          (NumPy array of diagrams per homology dim)
  - persistence_diagrams.png          (Plot of persistence diagrams)
  - barcode_Hk.png                    (Barcodes for each homology dimension k = 0..maxdim)
  - X_processed.npy                   (Processed point cloud used as input)

Notes:
  - Use --sample to randomly downsample large datasets to speed up (Ripser scales ~ O(n^2)).
  - --maxdim controls the maximum homology dimension (0=components, 1=loops, 2=voids).
"""
import argparse
import os
import sys
import numpy as np

# graceful import with helpful error message
def _safe_import():
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from ripser import ripser
        from persim import plot_diagrams
        return pd, plt, StandardScaler, PCA, ripser, plot_diagrams
    except Exception as e:
        print("[ERROR] Missing a dependency. Please install:")
        print("  pip install pandas numpy matplotlib scikit-learn ripser persim")
        print("Detailed import error:", repr(e))
        sys.exit(1)

pd, plt, StandardScaler, PCA, ripser, plot_diagrams = _safe_import()

def parse_args():
    p = argparse.ArgumentParser(description="Compute persistent homology (Ripser) from a CSV file.")
    p.add_argument("--csv", required=True, help="Path to input CSV file.")
    p.add_argument("--columns", type=str, default=None,
                   help="Comma-separated column names to use (numeric expected). If omitted, all numeric columns are used.")
    p.add_argument("--exclude", type=str, default=None,
                   help="Comma-separated column names to exclude (applied after --columns).")
    p.add_argument("--standardize", action="store_true",
                   help="Z-score standardize features before TDA.")
    p.add_argument("--pca-dim", type=int, default=None,
                   help="Optional PCA dimension reduction (e.g., 10).")
    p.add_argument("--maxdim", type=int, default=1,
                   help="Maximum homology dimension to compute (0, 1, or 2). Default: 1")
    p.add_argument("--metric", type=str, default="euclidean",
                   help="Distance metric passed to ripser (default: euclidean).")
    p.add_argument("--sample", type=int, default=None,
                   help="Optional random subsample size to speed up on large CSVs.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for subsampling.")
    p.add_argument("--outdir", type=str, default=None,
                   help="Directory to write outputs. Default: directory of the CSV.")
    return p.parse_args()

def select_columns(df, columns, exclude):
    if columns:
        keep = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        if not keep:
            raise ValueError("None of the specified --columns were found in the CSV.")
        df = df[keep]
    else:
        # auto-pick numeric columns
        df = df.select_dtypes(include=[np.number])
        if df.shape[1] == 0:
            raise ValueError("No numeric columns found. Use --columns to pick features.")
    if exclude:
        drops = [c.strip() for c in exclude.split(",")]
        df = df.drop(columns=[c for c in drops if c in df.columns], errors="ignore")
    return df

def preprocess_point_cloud(df, standardize=False, pca_dim=None):
    X = df.to_numpy(dtype=float)
    # drop rows with NaNs
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    if X.shape[0] == 0:
        raise ValueError("No rows left after dropping NaNs.")
    if standardize:
        X = StandardScaler().fit_transform(X)
    if pca_dim is not None and pca_dim > 0 and pca_dim < X.shape[1]:
        X = PCA(n_components=pca_dim, random_state=0).fit_transform(X)
    return X

def maybe_subsample(X, sample, seed):
    if sample is None or sample >= X.shape[0]:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=sample, replace=False)
    return X[idx]

def run_ripser(X, maxdim=1, metric="euclidean"):
    res = ripser(X, maxdim=maxdim, metric=metric)
    # res["dgms"] is a list of diagrams by homology dimension
    return res["dgms"]

def save_outputs(dgms, X, outdir):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "persistence_diagrams.npy"), dgms, allow_pickle=True)
    np.save(os.path.join(outdir, "X_processed.npy"), X)

    # Plot persistence diagrams
    plt.figure()
    plot_diagrams(dgms, show=False)
    plt.title("Persistence Diagrams")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "persistence_diagrams.png"), dpi=180)
    plt.close()

    # Plot barcodes per Hk
    for k, dgm in enumerate(dgms):
        # skip empty diagrams
        if dgm is None or len(dgm) == 0:
            continue
        # sort intervals by length (death-birth) descending for readability
        intervals = np.array(dgm)
        lengths = intervals[:,1] - intervals[:,0]
        order = np.argsort(-lengths)
        intervals = intervals[order]

        plt.figure(figsize=(6, max(2, 0.2*len(intervals))))
        for i, (b, d) in enumerate(intervals):
            y = len(intervals) - i
            plt.hlines(y, b, d if np.isfinite(d) else b + (lengths[np.isfinite(lengths)].max() if np.isfinite(lengths).any() else 1.0))
        plt.xlabel("Scale")
        plt.ylabel("Intervals")
        plt.title(f"Barcode H{k}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"barcode_H{k}.png"), dpi=180)
        plt.close()

def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv)) or "."
    print(f"[INFO] Reading CSV: {args.csv}")
    df_raw = pd.read_csv(args.csv)

    df_sel = select_columns(df_raw, args.columns, args.exclude)
    print(f"[INFO] Using columns: {list(df_sel.columns)} (rows={df_sel.shape[0]})")

    X = preprocess_point_cloud(df_sel, standardize=args.standardize, pca_dim=args.pca_dim)
    print(f"[INFO] Point cloud shape after preprocess: {X.shape}")

    X = maybe_subsample(X, args.sample, args.seed)
    if args.sample is not None:
        print(f"[INFO] After subsample: {X.shape}")

    print(f"[INFO] Running Ripser with maxdim={args.maxdim}, metric={args.metric} ...")
    dgms = run_ripser(X, maxdim=args.maxdim, metric=args.metric)

    print(f"[INFO] Saving outputs to: {outdir}")
    save_outputs(dgms, X, outdir)
    print("[DONE] Files saved:")
    print(" - persistence_diagrams.npy")
    print(" - persistence_diagrams.png")
    for k in range(len(dgms)):
        print(f" - barcode_H{k}.png")
    print(" - X_processed.npy")

if __name__ == "__main__":
    main()
