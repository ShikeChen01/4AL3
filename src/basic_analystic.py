import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import load_data, analyze_linear_relationships, analyze_dummy_features
from matplotlib import pyplot as plot

def preprocess_data(df, remove_mode='middle', percentile=47.5, normalize=True):
    """
    Preprocess a pandas DataFrame by normalizing and removing data based on percentiles.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to preprocess
    remove_mode : str, default='extremes'
        Mode for removing data:
        - 'extremes': Remove top and bottom percentiles
        - 'middle': Remove middle percentiles
    percentile : float, default=5
        Percentile threshold for removal (0-50)
        - For 'extremes': removes bottom `percentile`% and top `percentile`%
        - For 'middle': removes middle (50 - percentile)% to (50 + percentile)%
    normalize : bool, default=True
        Whether to normalize using standard deviation (z-score normalization)
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe
    """
    df_processed = df.copy()
    
    # Normalize using standard deviation (z-score)
    if normalize:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = (df_processed[numeric_cols] - df_processed[numeric_cols].mean()) / df_processed[numeric_cols].std()
    
    # Remove data based on percentiles
    # Calculate overall score (mean of all numeric columns) for filtering
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        overall_score = df_processed[numeric_cols].mean(axis=1)
        
        if remove_mode == 'extremes':
            # Remove top and bottom percentiles
            lower_bound = np.percentile(overall_score, percentile)
            upper_bound = np.percentile(overall_score, 100 - percentile)
            mask = (overall_score >= lower_bound) & (overall_score <= upper_bound)
        elif remove_mode == 'middle':
            # Remove middle percentiles
            lower_bound = np.percentile(overall_score, 50 - percentile)
            upper_bound = np.percentile(overall_score, 50 + percentile)
            mask = (overall_score < lower_bound) | (overall_score > upper_bound)
        else:
            raise ValueError("remove_mode must be either 'extremes' or 'middle'")
        
        df_processed = df_processed[mask].reset_index(drop=True)
    
    return df_processed

def _fd_bins(x: np.ndarray):
    x = x[~np.isnan(x)]
    if x.size == 0: return "auto"
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr == 0: return "auto"
    bw = 2 * iqr * (x.size ** (-1/3))
    if bw <= 0: return "auto"
    return max(10, int(np.ceil((x.max() - x.min()) / bw)))

def plot_hist_ecdf(df: pd.DataFrame, col: str, dropna=True, logy=False):
    x = df[col].to_numpy()
    if dropna: x = x[~np.isnan(x)]
    bins = _fd_bins(x)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(x, bins=bins, density=False)
    ax.set_xlabel(col); ax.set_ylabel("Count"); ax.set_title(f"Histogram ({col})")
    if logy: ax.set_yscale('log')

    # ECDF on twin axis
    ax2 = ax.twinx()
    xs = np.sort(x); ys = np.arange(1, xs.size+1) / xs.size
    ax2.plot(xs, ys)
    ax2.set_ylabel("ECDF")
    fig.tight_layout()
    return fig, (ax, ax2)

def plot_box(df: pd.DataFrame, col: str, dropna=True):
    x = df[col].dropna().to_numpy() if dropna else df[col].to_numpy()
    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.boxplot(x, vert=False, showfliers=True)
    ax.set_xlabel(col); ax.set_yticks([]); ax.set_title(f"Boxplot ({col})")
    fig.tight_layout()
    return fig, ax


df = load_data(r'data\Hull Market\train.csv')

for col in ["market_forward_excess_returns", "market_forward_excess_return"]:
    df_cp = df.copy()
    if col in df_cp.columns:
        df.drop(columns=[col], inplace=True)

for col in df.columns:
    print(pd.isna(df[col]).sum())

eps_movement = 0.02

analyze_dummy_features(df, 'target', eps_movement=eps_movement)

df_cp = df.copy()
df_cp['target'] = np.where(np.abs(df_cp['target']) <= eps_movement, 0, df_cp['target'])
df_cp = df_cp[df_cp['target'] != 0]

targets = df_cp[df_cp['D6'] == 0]

plot_hist_ecdf(targets, 'target', dropna=True, logy=True)
plot.show()
plot_box(targets, 'target', dropna=True)
plot.show()








