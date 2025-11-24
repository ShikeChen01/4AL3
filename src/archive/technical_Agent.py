from util import load_data, preprocess_data, linear_regression_plot, analyze_correlations
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt



def main():
    train_path = r"C:\Year4\4AL3\final-project\4AL3\data\Hull Market\train.csv"
    eps = 0.005

    df = load_data(train_path)
    cols = []

    for col in df.columns:
        if re.match(r"^D\d+$", col):
            cols.append(col)
    df = preprocess_data(df=df, remove_mode='middle', percentile=47.5, normalize=True, mode='y')
    masked_df = df.copy()
    masked_df['target'] = df['target'].apply(lambda x: 1 if x > eps else x)
    masked_df['target'] = masked_df['target'].apply(lambda x: -1 if x < -eps else 0)
    results_df = analyze_correlations(df, target_col='target', feature_cols=cols, plot_type='heatmap')
    print(results_df)

if __name__ == "__main__":
    main()