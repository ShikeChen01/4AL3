import pandas as pd
from pathlib import Path
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings

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

def analyze_linear_relationships(df, target_col, k=5):
    """
    Analyze linear relationships between features and target variable.
    Shows top k plots one by one.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing features and target
    target_col : str
        Name of the target column
    k : int
        Number of top correlations to plot (default: 5)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing R values and other metrics for each feature
    """
    
    # Check if target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Separate features from target
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Initialize results dictionary
    results = {
        'Feature': [],
        'R_value': [],
        'R_squared': [],
        'P_value': [],
        'RMSE': [],
        'Slope': [],
        'Intercept': []
    }
    
    # Store plot data
    plot_data = {}
    
    # Analyze each feature
    for feature in feature_cols:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[feature]):
            print(f"Skipping non-numeric column: {feature}")
            continue
            
        # Remove NaN values
        valid_mask = df[[feature, target_col]].notna().all(axis=1)
        if valid_mask.sum() < 2:
            print(f"Skipping {feature}: insufficient valid data")
            continue
            
        X = df.loc[valid_mask, feature].values.reshape(-1, 1)
        y = df.loc[valid_mask, target_col].values
        
        # Check for constant features (zero variance)
        if np.var(X) == 0 or np.var(y) == 0:
            print(f"Skipping {feature}: constant values (zero variance)")
            continue
        
        # Check if feature has very low variance
        if np.std(X) < 1e-10 or np.std(y) < 1e-10:
            print(f"Skipping {feature}: near-constant values (extremely low variance)")
            continue
        
        # Fit linear regression
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        
        # Calculate metrics with error handling
        try:
            # Suppress the specific warning and handle it
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='An input array is constant')
                r_value, p_value = pearsonr(X.flatten(), y)
                
            # Check if correlation is undefined (NaN)
            if np.isnan(r_value):
                print(f"Skipping {feature}: correlation undefined")
                continue
                
        except Exception as e:
            print(f"Skipping {feature}: error calculating correlation - {e}")
            continue
        
        # Calculate other metrics
        r_squared = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Store results
        results['Feature'].append(feature)
        results['R_value'].append(r_value)
        results['R_squared'].append(r_squared)
        results['P_value'].append(p_value)
        results['RMSE'].append(rmse)
        results['Slope'].append(lr.coef_[0])
        results['Intercept'].append(lr.intercept_)
        
        # Store data for plotting
        plot_data[feature] = {
            'X': X.flatten(),
            'y': y,
            'lr': lr
        }
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No valid numeric features found for analysis")
        return results_df
    
    # Sort by absolute R value
    results_df = results_df.sort_values('R_value', key=abs, ascending=False)
    
    # Select top k features for plotting
    top_k_features = results_df.head(k)['Feature'].tolist()
    
    # Plot top k features one by one
    for i, feature in enumerate(top_k_features, 1):
        # Get data
        data = plot_data[feature]
        X = data['X']
        y = data['y']
        lr = data['lr']
        
        # Get statistics
        feature_stats = results_df[results_df['Feature'] == feature].iloc[0]
        
        # Create individual plot
        plt.figure(figsize=(10, 6))
        
        # Calculate axis limits with padding
        x_range = X.max() - X.min()
        y_range = y.max() - y.min()
        
        # Handle case where range might be zero
        x_padding = max(x_range * 0.1, 0.1)
        y_padding = max(y_range * 0.1, 0.1)
        
        x_min, x_max = X.min() - x_padding, X.max() + x_padding
        y_min, y_max = y.min() - y_padding, y.max() + y_padding
        
        # Scatter plot
        plt.scatter(X, y, alpha=0.6, s=30, color='blue', edgecolors='darkblue', 
                   linewidth=0.5, label='Data points')
        
        # Regression line
        X_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_line = lr.predict(X_line)
        plt.plot(X_line, y_line, 'r-', linewidth=2, alpha=0.8,
                label=f'y = {feature_stats["Slope"]:.3f}x + {feature_stats["Intercept"]:.3f}')
        
        # Set limits
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Labels and title
        plt.xlabel(feature, fontsize=12, fontweight='bold')
        plt.ylabel(target_col, fontsize=12, fontweight='bold')
        
        title = f'Linear Regression: {feature} vs {target_col}\n'
        title += f'R = {feature_stats["R_value"]:.3f}, '
        title += f'R² = {feature_stats["R_squared"]:.3f}, '
        title += f'RMSE = {feature_stats["RMSE"]:.3f}, '
        title += f'p-value = {feature_stats["P_value"]:.3e}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add data range info
        range_text = f'X range: [{X.min():.2f}, {X.max():.2f}]\n'
        range_text += f'Y range: [{y.min():.2f}, {y.max():.2f}]\n'
        range_text += f'N samples: {len(X)}'
        plt.text(0.02, 0.98, range_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Legend
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # Show plot number
        plt.text(0.98, 0.02, f'Plot {i} of {len(top_k_features)}', 
                transform=plt.gca().transAxes, fontsize=10, 
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    return results_df

def analyze_dummy_features(df_original, target_col, eps_movement=0.005):
    """
    Analyze ternary dummy features (0, 1, -1) against a ternary target variable.
    Shows percentage of exact agreement and partial patterns.
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        The dataframe containing dummy features and target
    target_col : str
        The name of the target column (should be continuous, will be converted to -1, 0, 1)
    """

    df = df_original.copy()

    # Convert target to -1, 0, 1
    df['target'] = np.where(abs(df[target_col]) <= eps_movement, 0, df[target_col])

    df['target'] = np.where(df['target'] > eps_movement, 1, df['target'])
    df['target'] = np.where(df['target'] < -eps_movement, -1, df['target'])

    df['target'] = df['target'].astype(int)
    print(df['target'])

    # Find all columns that start with 'D' followed by digits
    dummy_cols = [col for col in df.columns if col.startswith('D') and col[1:].isdigit()]
    
    if not dummy_cols:
        print("No dummy features found (format: D1, D2, D3, etc.)")
        return
    
    # Sort dummy columns numerically (D1, D2, D3, ...)
    dummy_cols = sorted(dummy_cols, key=lambda x: int(x[1:]))
    
    print(f"Found {len(dummy_cols)} dummy features\n")
    print(f"Target: {target_col} (converted to -1, 0, 1)")
    print("=" * 80)
    
    results = []
    
    for col in dummy_cols:
        # Calculate exact matches for each value
        both_positive = ((df[col] == 1) & (df['target'] == 1)).sum()
        both_zero = ((df[col] == 0) & (df['target'] == 0)).sum()
        both_negative = ((df[col] == -1) & (df['target'] == -1)).sum()
        
        total_agreement = both_positive + both_zero + both_negative
        
        # Calculate percentages
        total_records = len(df)
        agreement_pct = (total_agreement / total_records) * 100
        both_positive_pct = (both_positive / total_records) * 100
        both_zero_pct = (both_zero / total_records) * 100
        both_negative_pct = (both_negative / total_records) * 100
        
        # Calculate disagreement
        disagreement = total_records - total_agreement
        disagreement_pct = (disagreement / total_records) * 100
        
        # Additional insights: same sign (both positive or both negative)
        same_sign = both_positive + both_negative
        same_sign_pct = (same_sign / total_records) * 100
        
        results.append({
            'Feature': col,
            'Agreement %': agreement_pct,
            'Both +1 %': both_positive_pct,
            'Both 0 %': both_zero_pct,
            'Both -1 %': both_negative_pct,
            'Same Sign %': same_sign_pct,
            'Disagreement %': disagreement_pct
        })
        
        print(f"\n{col}:")
        print(f"  Total Agreement:     {agreement_pct:6.2f}% ({total_agreement}/{total_records})")
        print(f"    - Both +1:         {both_positive_pct:6.2f}% ({both_positive}/{total_records})")
        print(f"    - Both  0:         {both_zero_pct:6.2f}% ({both_zero}/{total_records})")
        print(f"    - Both -1:         {both_negative_pct:6.2f}% ({both_negative}/{total_records})")
        print(f"  Same Sign (+1/-1):   {same_sign_pct:6.2f}% ({same_sign}/{total_records})")
        print(f"  Disagreement:        {disagreement_pct:6.2f}% ({disagreement}/{total_records})")
    
    print("\n" + "=" * 80)
    print("\nSUMMARY - Sorted by Agreement %:")
    print("-" * 80)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Agreement %', ascending=False)
    
    print(results_df.to_string(index=False))
    
    return results_df

# Example usage:
# results = analyze_dummy_features(df, 'target')