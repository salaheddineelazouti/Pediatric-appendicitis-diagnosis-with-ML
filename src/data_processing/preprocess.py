"""
Data preprocessing module for the Pediatric Appendicitis Diagnosis project.
This module contains functions for:
- Loading and cleaning data
- Feature transformation and engineering
- Memory optimization
- Data splitting
"""

import os
import logging
import logging.config
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, Union

# Setup logging data
logging.config.fileConfig(os.path.join(os.path.dirname(__file__), '../config/logging.conf'))
logger = logging.getLogger('dataProcessing')

# Load configuration of data
def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from specified filepath.
    
    Args:
        filepath: Path to the raw data file
        
    Returns:
        DataFrame containing the raw data
    """
    logger.info(f"Loading data from {filepath}")
    
    file_extension = Path(filepath).suffix
    
    if file_extension == '.csv':
        df = pd.read_csv(filepath)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    logger.info(f"Loaded data with shape {df.shape}")
    return df

def optimize_memory(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Union[str, int]]]]:
    """
    Optimize memory usage of a DataFrame by converting data types.
    
    This function reduces memory usage by:
    - Converting float64 to float32 where appropriate
    - Converting int64 to smaller integer types where appropriate
    - Optimizing categorical and object columns
    
    Args:
        df: Input DataFrame to optimize
        
    Returns:
        Tuple of (optimized DataFrame, dictionary of transformations applied)
    """
    logger.info("Starting memory optimization")
    
    # Track memory usage before optimization
    start_memory = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory usage before optimization: {start_memory:.2f} MB")
    
    
    transformations = {}
    
    # Create a copy of the DataFrame to avoid modifying original
    result = df.copy()
    
    # Process each column
    for col in df.columns:
        col_type = df[col].dtype
        transformations[col] = {'original_type': str(col_type), 'new_type': str(col_type), 'memory_saved_kb': 0}
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_type):
            # Integer columns
            if pd.api.types.is_integer_dtype(col_type):
                # Get min and max values
                c_min = df[col].min()
                c_max = df[col].max()
                
                # Integer type based on range
                if c_min >= 0:  # Unsigned integers
                    if c_max < 2**8:
                        new_type = np.uint8
                    elif c_max < 2**16:
                        new_type = np.uint16
                    elif c_max < 2**32:
                        new_type = np.uint32
                    else:
                        new_type = np.uint64
                else:  # Signed integers
                    if c_min > -2**7 and c_max < 2**7:
                        new_type = np.int8
                    elif c_min > -2**15 and c_max < 2**15:
                        new_type = np.int16
                    elif c_min > -2**31 and c_max < 2**31:
                        new_type = np.int32
                    else:
                        new_type = np.int64
                
                # Convert to the new type
                result[col] = df[col].astype(new_type)
                
                # Update transformation tracking
                transformations[col]['new_type'] = str(new_type)
            
            # Float columns - convert float64 to float32
            elif col_type == np.float64:
                result[col] = df[col].astype(np.float32)
                transformations[col]['new_type'] = str(np.float32)
        
        # Categorical and object columns
        elif col_type == 'object':
            # Check if the column could be categorical
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% of values are unique
                result[col] = df[col].astype('category')
                transformations[col]['new_type'] = 'category'
    
    # Calculate memory usage after optimization
    end_memory = result.memory_usage(deep=True).sum() / 1024**2
    
    # Update memory savings for each column
    for col in transformations:
        before = df[col].memory_usage(deep=True) / 1024  # KB
        after = result[col].memory_usage(deep=True) / 1024  # KB
        transformations[col]['memory_saved_kb'] = before - after
    
    # Log the results of all
    memory_saved = start_memory - end_memory
    memory_ratio = start_memory / end_memory if end_memory > 0 else float('inf')
    
    logger.info(f"Memory usage after optimization: {end_memory:.2f} MB")
    logger.info(f"Memory saved: {memory_saved:.2f} MB ({memory_ratio:.1f}x reduction)")
    
    return result, transformations

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Strategy to handle missing values ('median', 'mean', 'mode', 'drop')
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info(f"Handling missing values using strategy: {strategy}")
    
    result = df.copy()
    
    # Count missing values in data
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) == 0:
        logger.info("No missing values found")
        return result
    
    logger.info(f"Found {len(missing_cols)} columns with missing values")
    
    for col in missing_cols.index:
        if strategy == 'drop':
            result = result.dropna(subset=[col])
            logger.info(f"Dropped {missing_cols[col]} rows with missing values in column {col}")
        else:
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                if strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                else:  # mode
                    fill_value = df[col].mode()[0]
                
                result[col] = result[col].fillna(fill_value)
                logger.info(f"Filled {missing_cols[col]} missing values in column {col} with {strategy}: {fill_value}")
            else:
                # For non-numeric columns, use mode
                fill_value = df[col].mode()[0]
                result[col] = result[col].fillna(fill_value)
                logger.info(f"Filled {missing_cols[col]} missing values in non-numeric column {col} with mode: {fill_value}")
    
    return result

def handle_outliers(df: pd.DataFrame, method: str = 'iqr', columns: Optional[list] = None) -> pd.DataFrame:
    """
    Handle outliers in numeric columns using specified method.
    
    Args:
        df: Input DataFrame
        method: Method to identify outliers ('iqr', 'zscore')
        columns: List of columns to check for outliers. If None, checks all numeric columns.
        
    Returns:
        DataFrame with outliers handled
    """
    logger.info(f"Handling outliers using method: {method}")
    
    result = df.copy()
    
    # Identify numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            continue
            
        if method == 'iqr':
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            # Cap outliers at the boundaries
            result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info(f"Column {col}: Capped {outliers} outliers using IQR method")
            
        elif method == 'zscore':
            # Z-score method
            mean = df[col].mean()
            std = df[col].std()
            
            z_scores = (df[col] - mean) / std
            abs_z_scores = np.abs(z_scores)
            
            # Identify outliers (z-score > 3)
            outlier_mask = abs_z_scores > 3
            outliers = outlier_mask.sum()
            
            # Replace outliers with mean
            result.loc[outlier_mask, col] = mean
            
            logger.info(f"Column {col}: Replaced {outliers} outliers using Z-score method")
    
    return result

def split_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Splitting data into training and testing sets")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Load split parameters from config
    test_size = config['data_preprocessing']['test_size']
    random_seed = config['general']['random_seed']
    shuffle = config['data_preprocessing']['shuffle']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, shuffle=shuffle
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series, y_test: pd.Series, 
                        output_dir: str) -> None:
    """
    Save processed data to output directory.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        output_dir: Output directory
    """
    logger.info(f"Saving processed data to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    logger.info("Processed data saved successfully")

def preprocess_pipeline(input_file: str, output_dir: str, target_column: str,
                       missing_strategy: str = 'median', outlier_method: str = 'iqr') -> dict:
    """
    Complete preprocessing pipeline that:
    1. Loads data
    2. Handles missing values
    3. Optimizes memory usage
    4. Handles outliers
    5. Splits data
    6. Saves processed data
    
    Args:
        input_file: Path to input file
        output_dir: Output directory for processed data
        target_column: Name of target column
        missing_strategy: Strategy for handling missing values
        outlier_method: Method for handling outliers
    
    Returns:
        Dictionary containing preprocessing statistics and metadata
    """
    logger.info("Starting preprocessing pipeline")
    
    # Load data
    df = load_data(input_file)
    
    # Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)
    
    # Optimize memory usage
    df, memory_transformations = optimize_memory(df)
    
    # Handle outliers
    df = handle_outliers(df, method=outlier_method)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, output_dir)
    
    # Compile preprocessing metadata
    metadata = {
        'original_shape': df.shape,
        'memory_transformations': memory_transformations,
        'train_shape': (X_train.shape[0], X_train.shape[1]),
        'test_shape': (X_test.shape[0], X_test.shape[1]),
        'class_distribution': {
            'train': y_train.value_counts().to_dict(),
            'test': y_test.value_counts().to_dict()
        }
    }
    
    logger.info("Preprocessing pipeline completed successfully")
    
    return metadata