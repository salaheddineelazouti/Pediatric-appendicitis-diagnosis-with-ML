"""
Prepare the pediatric appendicitis dataset for modeling.

This script:
1. Loads the dataset
2. Handles missing values
3. Handles outliers
4. Encodes categorical variables
5. Optimizes memory usage
6. Creates train/test splits
7. Saves the processed dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from src.data_processing.preprocess import optimize_memory

def load_dataset():
    """
    Load the pediatric appendicitis dataset.
    """
    data_path = os.path.join(project_root, 'DATA', 'pediatric_appendicitis_data.xlsx')
    print(f"Loading dataset from {data_path}")
    df = pd.read_excel(data_path)
    print(f"Dataset shape: {df.shape}")
    return df

def identify_target_variable(df):
    """
    Identify the target variable (appendicitis diagnosis).
    """
    print("Identifying target variable...")
    
    # Fill NaN values in Diagnosis column before processing..
    df['Diagnosis'] = df['Diagnosis'].fillna('Unknown')
    
    # Create binary target where 1 = appendicitis and 0 = no appendicitis..
    df['is_appendicitis'] = (df['Diagnosis'].str.lower() == 'appendicitis').astype(int)
    
    # Display class distribution
    appendicitis_counts = df['is_appendicitis'].value_counts()
    appendicitis_percentages = df['is_appendicitis'].value_counts(normalize=True) * 100
    
    print(f"Target variable distribution:")
    for value, count in appendicitis_counts.items():
        label = "Appendicitis" if value == 1 else "No Appendicitis"
        print(f"  {label}: {count} instances ({appendicitis_percentages[value]:.1f}%)")
    
    return df

def drop_high_missing_columns(df, threshold=50):
    """
    Drop columns with high percentage of missing values.
    
    Args:
        df: DataFrame
        threshold: Maximum percentage of missing values allowed (default: 50%)
    """
    print(f"\nDropping columns with more than {threshold}% missing values...")
    
    # Calculate missing percentages..
    missing_percentages = df.isnull().mean() * 100
    high_missing_cols = missing_percentages[missing_percentages > threshold].index.tolist()
    
    print(f"Dropping {len(high_missing_cols)} columns: {', '.join(high_missing_cols)}")
    df_reduced = df.drop(columns=high_missing_cols)
    
    print(f"Dataset shape after dropping high-missing columns: {df_reduced.shape}")
    return df_reduced

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    """
    print("\nHandling remaining missing values...")
    
    # Separate numerical and categorical columns..
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Impute numerical columns with median
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"  - Imputed {col} missing values with median: {median_value:.2f}")
    
    # Impute categorical columns with mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            print(f"  - Imputed {col} missing values with mode: {mode_value}")
    
    # Verify no missing values remain
    assert df.isnull().sum().sum() == 0, "Missing values still exist after imputation"
    print("All missing values have been handled")
    
    return df

def handle_outliers(df, z_threshold=3):
    """
    Handle outliers in numerical columns using capping/flooring.
    """
    print(f"\nHandling outliers with Z-score threshold of {z_threshold}...")
    
    # Only process numeric columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Skip the target variable
    if 'is_appendicitis' in numerical_cols:
        numerical_cols = numerical_cols.drop('is_appendicitis')
    
    outliers_handled = 0
    
    for col in numerical_cols:
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(df[col]))
        
        # Identify outliers
        outliers = (z_scores > z_threshold)
        num_outliers = outliers.sum()
        
        if num_outliers > 0:
            # Get the upper and lower bounds
            upper_bound = df[col].mean() + z_threshold * df[col].std()
            lower_bound = df[col].mean() - z_threshold * df[col].std()
            
            # Cap/floor the outliers
            df.loc[df[col] > upper_bound, col] = upper_bound
            df.loc[df[col] < lower_bound, col] = lower_bound
            
            outliers_handled += num_outliers
            print(f"  - Capped {num_outliers} outliers in {col}")
    
    print(f"Total outliers handled: {outliers_handled}")
    return df

def encode_categorical_features(df):
    """
    Encode categorical features using one-hot encoding.
    """
    print("\nEncoding categorical features...")
    
    # Identify categorical columns (excluding the target if it's binary)
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Get the number of categorical columns
    num_categorical = len(categorical_cols)
    print(f"Found {num_categorical} categorical columns to encode")
    
    # One-hot encode all categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Calculate how many new columns were created
    num_new_cols = df_encoded.shape[1] - df.shape[1] + num_categorical
    print(f"One-hot encoding created {num_new_cols} new columns")
    print(f"Dataset shape after encoding: {df_encoded.shape}")
    
    return df_encoded

def handle_correlated_features(df, threshold=0.7):
    """
    Identify and handle highly correlated features.
    """
    print(f"\nHandling highly correlated features (threshold: {threshold})...")
    
    # Calculate correlation matrix for numeric features
    corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr().abs()
    
    # Create a mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features: {', '.join(to_drop)}")
        df_reduced = df.drop(columns=to_drop)
        print(f"Dataset shape after removing correlated features: {df_reduced.shape}")
    else:
        print("No highly correlated features found")
        df_reduced = df
    
    return df_reduced

def create_train_test_split(df, target_column, test_size=0.2, random_state=42):
    """
    Create train/test splits.
    """
    print(f"\nCreating train/test split (test size: {test_size})...")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def normalize_features(X_train, X_test):
    """
    Normalize numerical features using StandardScaler.
    """
    print("\nNormalizing numerical features...")
    
    # Get numerical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numerical_cols) == 0:
        print("No numerical features found to normalize")
        return X_train, X_test, None
    
    print(f"Found {len(numerical_cols)} numerical features to normalize")
    
    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"Normalized {len(numerical_cols)} numerical features")
    
    return X_train_scaled, X_test_scaled, scaler

def main():
    """Main function to prepare the dataset."""
    print("="*80)
    print("PEDIATRIC APPENDICITIS DATASET PREPARATION")
    print("="*80)
    
    # Load dataset
    df = load_dataset()
    
    # Create binary target variable
    df = identify_target_variable(df)
    
    # Drop columns with high missing values
    df = drop_high_missing_columns(df, threshold=50)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df, z_threshold=3)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Handle highly correlated features
    df = handle_correlated_features(df, threshold=0.7)
    
    # Optimize memory usage
    df_optimized, optimization_stats = optimize_memory(df)
    
    # Create train/test splits
    X_train, X_test, y_train, y_test = create_train_test_split(
        df_optimized, target_column='is_appendicitis', test_size=0.2
    )
    
    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Save processed datasets
    output_dir = os.path.join(project_root, 'DATA', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    X_train_scaled.to_pickle(os.path.join(output_dir, 'X_train.pkl'))
    X_test_scaled.to_pickle(os.path.join(output_dir, 'X_test.pkl'))
    y_train.to_pickle(os.path.join(output_dir, 'y_train.pkl'))
    y_test.to_pickle(os.path.join(output_dir, 'y_test.pkl'))
    
    # Save scaler
    import pickle
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nDataset preparation complete!")
    print(f"Processed datasets saved to {output_dir}")
    print(f"Final dataset shapes:")
    print(f"  - X_train: {X_train_scaled.shape}")
    print(f"  - X_test: {X_test_scaled.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test: {y_test.shape}")

if __name__ == "__main__":
    main()
