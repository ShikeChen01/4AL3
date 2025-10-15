import pandas as pd
from pathlib import Path
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, pointbiserialr
import seaborn as sns

def load_data(path: str | Path, target='forward_returns') -> pd.DataFrame:
    """
    Load the dataset and return a DataFrame with:
      - 'forward_returns' renamed to 'target'
      - 'market_forward_excess_returns' dropped (if present)
    Supports CSV or Parquet.
    """
    path = Path(path)

    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        # Fast, reliable CSV read; falls back if pyarrow isn't installed
        try:
            df = pd.read_csv(path, engine="pyarrow")
        except Exception:
            df = pd.read_csv(path)

    # Clean/standardize column names
    df.columns = df.columns.str.strip()


    df.rename(columns={target: "target"}, inplace=True)
    print(df.columns)

    return df

import numpy as np
import pandas as pd

def _mask_pct_rmv(s: pd.Series, remove_mode: str, percentile: float) -> pd.Series:
    """
    Returns a boolean mask: True for values to KEEP.
    
    Args:
        s: Series to evaluate
        remove_mode: 'extremes' or 'middle'
        percentile: 0-50, defines the band size
    """
    s = s.astype(float)
    if remove_mode == "extremes":
        # Keep middle values (between percentiles)
        lb = np.nanpercentile(s, percentile)
        ub = np.nanpercentile(s, 100 - percentile)
        return (s >= lb) & (s <= ub)
    else:  # 'middle'
        # Keep extreme values (outside middle band)
        lb = np.nanpercentile(s, 50 - percentile)
        ub = np.nanpercentile(s, 50 + percentile)
        return (s < lb) | (s > ub)


def preprocess_data(
    df: pd.DataFrame,
    remove_mode: str = "middle",   # 'extremes' or 'middle' (defines the band)
    percentile: float = 47.5,      # 0–50
    normalize: bool = True,
    y_col: str = "target",
    mode: str = "xy",              # 'x', 'y', or 'xy'
):
    """
    Remove rows based on percentile bands.
      remove_mode:
        - 'extremes': drop bottom p% and top p%
        - 'middle'  : drop middle band [50-p, 50+p]%
      mode:
        - 'x'  : filter using X columns only
        - 'y'  : filter using y_col only
        - 'xy' : filter if y_col OR ANY X is in the band
    """
    if y_col not in df.columns:
        raise ValueError(f"y_col '{y_col}' not found")
    if remove_mode not in ("extremes", "middle"):
        raise ValueError("remove_mode must be 'extremes' or 'middle'")
    if mode not in ("x", "y", "xy"):
        raise ValueError("mode must be 'x', 'y', or 'xy'")
    if not (0 <= percentile <= 50):
        raise ValueError("percentile must be in [0, 50]")
    
    dfp = df.copy()
    numeric = dfp.select_dtypes(include=[np.number]).columns.tolist()
    
    # Auto-pick X (prefer S*)
    x_cols = [c for c in numeric if c != y_col and c.startswith("S")]
    if not x_cols:
        x_cols = [c for c in numeric if c != y_col]
    if mode in ("x", "xy") and not x_cols:
        raise ValueError("No numeric X columns found (excluding y_col).")
    
    # Normalize so percentiles are comparable
    if normalize and numeric:
        z = dfp[numeric].astype(float)
        dfp[numeric] = (z - z.mean()) / z.std(ddof=0)
    
    # Build keep mask directly
    if mode == "y":
        keep = _mask_pct_rmv(dfp[y_col], remove_mode, percentile)
    elif mode == "x":
        keep = _mask_pct_rmv(dfp[x_cols[0]], remove_mode, percentile)
        for c in x_cols[1:]:
            keep &= _mask_pct_rmv(dfp[c], remove_mode, percentile)
    else:  # mode == "xy"
        keep = _mask_pct_rmv(dfp[y_col], remove_mode, percentile)
        for c in x_cols:
            keep &= _mask_pct_rmv(dfp[c], remove_mode, percentile)
    
    return dfp[keep].reset_index(drop=True)



def linear_regression_plot(df_x, df_y, xlabel='X', ylabel='Y', title='Linear Regression'):
    """
    Perform linear regression and plot the results.
    
    Parameters:
    -----------
    df_x : pandas DataFrame or Series
        Independent variable(s)
    df_y : pandas DataFrame or Series
        Dependent variable
    xlabel : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    title : str
        Plot title
    
    Returns:
    --------
    model : LinearRegression
        Fitted sklearn model
    """
    # Convert to numpy arrays and reshape if needed
    X = np.array(df_x).reshape(-1, 1) if df_x.ndim == 1 else np.array(df_x)
    y = np.array(df_y).ravel()
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Data points')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
    
    # Add equation and R-squared to plot
    if X.shape[1] == 1:
        equation = f'y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}'
    else:
        equation = f'y = {model.intercept_:.3f} + ...'
    
    plt.text(0.05, 0.95, f'{equation}\nR² = {r_squared:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return model


def analyze_correlations(df, target_col, feature_cols=None, plot_type='bar'):
    """
    Analyze correlations between binary features and a target variable.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing features and target
    target_col : str
        Name of the target column
    feature_cols : list, optional
        List of binary feature column names. If None, uses all columns except target
    plot_type : str
        'bar' for bar plot, 'heatmap' for correlation heatmap
    
    Returns:
    --------
    results_df : pandas DataFrame
        DataFrame with correlation statistics for each feature
    """
    # Select features
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
    
    # Store results
    results = []
    
    # Get target values
    target = df[target_col]
    is_binary_target = df[target_col].nunique() == 2
    
    for feature in feature_cols:
        feature_data = df[feature]
        
        # Calculate Pearson correlation
        pearson_corr = feature_data.corr(target)
        
        # Calculate point-biserial correlation if target is binary
        if is_binary_target:
            pb_corr, pb_pval = pointbiserialr(target, feature_data)
        else:
            pb_corr, pb_pval = pearson_corr, np.nan
        
        # Chi-square test for independence (if both are binary)
        if is_binary_target:
            contingency = pd.crosstab(feature_data, target)
            chi2, p_value, dof, expected = chi2_contingency(contingency)
        else:
            chi2, p_value = np.nan, np.nan
        
        results.append({
            'Feature': feature,
            'Pearson_Correlation': pearson_corr,
            'Point_Biserial_Corr': pb_corr,
            'Point_Biserial_PValue': pb_pval,
            'Chi2_Statistic': chi2,
            'Chi2_PValue': p_value
        })
    
    results_df = pd.DataFrame(results).sort_values('Pearson_Correlation', 
                                                    key=abs, 
                                                    ascending=False)
    
    # Create visualization
    if plot_type == 'bar':
        plt.figure(figsize=(12, 6))
        colors = ['green' if x > 0 else 'red' for x in results_df['Pearson_Correlation']]
        plt.barh(results_df['Feature'], results_df['Pearson_Correlation'], color=colors, alpha=0.7)
        plt.xlabel('Correlation with Target', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Binary Feature Correlations with {target_col}', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    elif plot_type == 'heatmap':
        # Create correlation matrix
        corr_matrix = df[feature_cols + [target_col]].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Heatmap (including {target_col})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return results_df
