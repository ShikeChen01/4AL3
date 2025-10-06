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
        print(np.isnan(df_processed).sum())
    
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

df = load_data(r'data\Hull Market\train.csv')

for col in ["market_forward_excess_returns", "market_forward_excess_return"]:
    df_cp = df.copy()
    if col in df_cp.columns:
        df.drop(columns=[col], inplace=True)
    

for col in df.columns:
    print(pd.isna(df[col]).sum())



analyze_dummy_features(df, 'target', eps_movement=0.005)
df_cp = df.copy()
df_cp['target'] = np.where(abs(df_cp['target']) <= 0.005, 0, df_cp['target'])

targets = df_cp['target'].where(df_cp['D1'] == 1)
targets = targets.to_frame()
targets = targets[targets['target'] != 0]
print(targets.shape)












