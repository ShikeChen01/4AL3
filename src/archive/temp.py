import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr

def analyze_binary_features(df, target_col, feature_cols=None, plot_type='bar'):
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


# Example usage:
if __name__ == "__main__":
    # Create sample data with binary features
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'feature_1': np.random.binomial(1, 0.6, n_samples),
        'feature_2': np.random.binomial(1, 0.4, n_samples),
        'feature_3': np.random.binomial(1, 0.5, n_samples),
        'feature_4': np.random.binomial(1, 0.7, n_samples),
        'feature_5': np.random.binomial(1, 0.3, n_samples),
    }
    
    # Create target that's correlated with some features
    data['target'] = (data['feature_1'] * 0.6 + 
                     data['feature_2'] * 0.3 + 
                     np.random.binomial(1, 0.2, n_samples))
    data['target'] = (data['target'] > 0.5).astype(int)
    
    df = pd.DataFrame(data)
    
    # Analyze correlations
    print("=== Bar Plot ===")
    results = analyze_binary_features(df, 'target', plot_type='bar')
    print("\nCorrelation Results:")
    print(results.to_string(index=False))
    
    print("\n=== Heatmap ===")
    analyze_binary_features(df, 'target', plot_type='heatmap')