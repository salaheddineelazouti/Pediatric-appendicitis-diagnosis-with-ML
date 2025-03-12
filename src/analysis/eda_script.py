"""
Exploratory Data Analysis for Pediatric Appendicitis Diagnosis.

This script analyzes the pediatric appendicitis dataset to address:
1. Missing values
2. Outliers
3. Class imbalance
4. Correlation analysis
5. Memory optimization

The results can be copied into the Jupyter notebook for visualization.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from src.data_processing.preprocess import optimize_memory

# Set plot styling
plt.style.use('ggplot')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14

def main():
    """Run the EDA analysis"""
    # Load the actual data file
    data_path = os.path.join(project_root, 'DATA', 'pediatric_appendicitis_data.xlsx')
    print(f"Loading dataset from {data_path}")
    
    # Load Excel file
    df = pd.read_excel(data_path)
    
    # Display basic information
    print(f"\nDataset shape: {df.shape}")
    print("\nSample of the dataset:")
    print(df.head())
    
    # Check data types and missing values
    print("\nData information:")
    df.info()
    
    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe().T)
    
    # ========== Target Variable Analysis ==========
    print("\n" + "="*50)
    print("TARGET VARIABLE ANALYSIS")
    print("="*50)
    
    # Check target variable name
    print("\nColumns in dataset:")
    print(df.columns.tolist())
    
    # Try to identify the target variable (appendicitis diagnosis)
    target_candidates = [col for col in df.columns if 'appendicitis' in col.lower() or 'diagnosis' in col.lower()]
    
    if target_candidates:
        print(f"\nPotential target variables: {target_candidates}")
        target_variable = target_candidates[0]
    else:
        binary_cols = df.columns[df.nunique() == 2].tolist()
        print(f"\nBinary columns that could be target variables: {binary_cols}")
        # For now, let's assume the last binary column might be the target
        if binary_cols:
            target_variable = binary_cols[-1]
        else:
            # If we can't find a good candidate, let's just print value counts for all columns
            print("\nCannot identify target variable. Checking value counts for all columns:")
            for col in df.columns:
                print(f"\n{col}: {df[col].value_counts()}")
            return
    
    print(f"\nUsing '{target_variable}' as the target variable.")
    
    # Analyze target distribution
    target_counts = df[target_variable].value_counts()
    target_percentages = df[target_variable].value_counts(normalize=True) * 100
    
    print(f"\nTarget variable distribution:")
    for value, count in target_counts.items():
        print(f"  Value {value}: {count} instances ({target_percentages[value]:.1f}%)")
    
    # ========== Missing Values Analysis ==========
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentages
    })
    missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values('Percentage', ascending=False)
    
    print("\nMissing values by column:")
    if len(missing_data) > 0:
        print(missing_data)
        print("\nRecommendation for handling missing values:")
        
        for col, percentage in zip(missing_data.index, missing_data['Percentage']):
            if percentage > 50:
                print(f"  - {col}: {percentage:.1f}% missing - Consider dropping this column")
            elif percentage > 20:
                print(f"  - {col}: {percentage:.1f}% missing - Impute using advanced methods (KNN, MICE)")
            else:
                print(f"  - {col}: {percentage:.1f}% missing - Impute using median/mode")
    else:
        print("No missing values found in the dataset.")
    
    # ========== Outlier Analysis ==========
    print("\n" + "="*50)
    print("OUTLIER ANALYSIS")
    print("="*50)
    
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Calculate Z-scores for numerical columns
    z_scores = pd.DataFrame()
    for col in numerical_cols:
        if col != target_variable:  # Skip target variable
            z_scores[col] = np.abs(scipy.stats.zscore(df[col], nan_policy='omit'))
    
    # Identify outliers using Z-score > 3
    outliers = (z_scores > 3).sum()
    outlier_percentages = (outliers / len(df)) * 100
    
    print("\nOutliers by column (Z-score > 3):")
    outlier_data = pd.DataFrame({
        'Outlier Count': outliers,
        'Percentage': outlier_percentages
    }).sort_values('Outlier Count', ascending=False)
    
    print(outlier_data[outlier_data['Outlier Count'] > 0])
    
    print("\nRecommendation for handling outliers:")
    for col, percentage in zip(outlier_data.index, outlier_data['Percentage']):
        if outlier_data.loc[col, 'Outlier Count'] > 0:
            if percentage > 5:
                print(f"  - {col}: {percentage:.1f}% outliers - Investigate specific domain relationship with target")
            else:
                print(f"  - {col}: {percentage:.1f}% outliers - Consider capping/flooring at 3 std devs")
    
    # ========== Correlation Analysis ==========
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Calculate correlation matrix
    correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    
    # Find highly correlated feature pairs (excluding self-correlations)
    high_correlations = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:  # Threshold for high correlation
                high_correlations.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
    
    # Print high correlations
    if high_correlations:
        print("\nHighly correlated features (|r| > 0.7):")
        for col1, col2, corr in sorted(high_correlations, key=lambda x: abs(x[2]), reverse=True):
            print(f"  - {col1} and {col2}: {corr:.2f}")
        
        print("\nRecommendation for managing correlations:")
        print("  - Consider removing one feature from each highly correlated pair")
        print("  - Alternatively, apply dimensionality reduction methods like PCA")
    else:
        print("\nNo highly correlated features found (|r| > 0.7).")
    
    # ========== Memory Optimization Analysis ==========
    print("\n" + "="*50)
    print("MEMORY OPTIMIZATION ANALYSIS")
    print("="*50)
    
    # Analyze memory usage before optimization
    df_memory = df.memory_usage(deep=True)
    total_memory_mb = df_memory.sum() / (1024 * 1024)
    
    print(f"\nMemory usage before optimization: {total_memory_mb:.2f} MB")
    
    # Apply memory optimization
    optimized_df, optimization_stats = optimize_memory(df)
    
    # Analyze memory usage after optimization
    optimized_memory = optimized_df.memory_usage(deep=True)
    optimized_total_mb = optimized_memory.sum() / (1024 * 1024)
    
    print(f"\nMemory usage after optimization: {optimized_total_mb:.2f} MB")
    print(f"Memory reduction: {(1 - optimized_total_mb/total_memory_mb) * 100:.2f}%")
    
    print("\nOptimization statistics:")
    for col, stats in optimization_stats.items():
        old_type = stats.get('old_type', 'unknown')
        new_type = stats.get('new_type', 'unknown')
        reduction = stats.get('memory_reduction_percentage', 0)
        
        if reduction > 0:
            print(f"  - {col}: {old_type} → {new_type} ({reduction:.1f}% reduction)")

if __name__ == "__main__":
    main()
