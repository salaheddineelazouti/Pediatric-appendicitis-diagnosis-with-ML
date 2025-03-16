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

# Setup logging
logging.config.fileConfig(os.path.join(os.path.dirname(__file__), '../config/logging.conf'))
logger = logging.getLogger('dataProcessing')

# Load configuration
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
    
    # Track transformations applied to each column
    transformations = {}
    
    # Create a copy of the DataFrame to avoid modifying the original
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
    
    # Log the results
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
    
    # Count missing values
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

def handle_outliers(df: pd.DataFrame, method: str = 'iqr', treatment: str = 'cap', 
                   columns: Optional[list] = None, visualize: bool = False,
                   column_specific_params: Optional[Dict[str, Dict]] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle outliers in numeric columns using specified method and treatment strategy.
    
    Args:
        df: Input DataFrame
        method: Method to identify outliers. Options:
               - 'iqr': Interquartile Range method
               - 'zscore': Standard deviation based method
               - 'isolation_forest': Use Isolation Forest algorithm
               - 'dbscan': Use DBSCAN clustering algorithm
               - 'percentile': Use fixed percentile thresholds
        treatment: Treatment strategy for outliers. Options:
                  - 'cap': Cap outliers at the boundaries
                  - 'remove': Remove the rows containing outliers
                  - 'mean': Replace outliers with mean
                  - 'median': Replace outliers with median
                  - 'winsorize': Apply winsorization
                  - 'none': Only identify but don't modify outliers
        columns: List of columns to check for outliers. If None, checks all numeric columns.
        visualize: Whether to create visualization of outliers (saved to figures directory)
        column_specific_params: Dictionary with column-specific parameters that override the general method/treatment
                               Format: {'column_name': {'method': 'method_name', 'treatment': 'treatment_name'}}
        
    Returns:
        Tuple of (DataFrame with outliers handled, dictionary of outlier statistics)
    """
    logger.info(f"Handling outliers using method: {method}, treatment: {treatment}")
    
    result = df.copy()
    outlier_stats = {}
    
    # Import visualization packages if needed
    if visualize:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from pathlib import Path
            
            # Create figures directory if it doesn't exist
            figures_dir = Path('figures/outliers')
            figures_dir.mkdir(parents=True, exist_ok=True)
        except ImportError:
            logger.warning("Visualization packages not available. Setting visualize=False")
            visualize = False
    
    # Import model-based outlier detection if needed
    if method in ['isolation_forest', 'dbscan']:
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error(f"scikit-learn is required for {method} method. Falling back to 'iqr'")
            method = 'iqr'
    
    # Identify numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Keep track of all outlier indices for possible row removal
    all_outlier_indices = set()
    
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            continue
        
        # Check if we have column-specific parameters
        col_method = method
        col_treatment = treatment
        if column_specific_params and col in column_specific_params:
            col_method = column_specific_params[col].get('method', method)
            col_treatment = column_specific_params[col].get('treatment', treatment)
            
        logger.info(f"Processing column {col} with method {col_method} and treatment {col_treatment}")
        
        # Initialize outlier mask
        outlier_mask = pd.Series(False, index=df.index)
        
        # Detect outliers based on specified method
        if col_method == 'iqr':
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            logger.info(f"Column {col}: IQR bounds [{lower_bound:.3f}, {upper_bound:.3f}]")
            
        elif col_method == 'zscore':
            # Z-score method
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:  # Handle constant columns
                logger.warning(f"Column {col} has zero standard deviation. Skipping.")
                continue
                
            z_scores = (df[col] - mean) / std
            outlier_mask = np.abs(z_scores) > 3
            
        elif col_method == 'percentile':
            # Percentile method
            lower_percentile = 0.01  # Bottom 1%
            upper_percentile = 0.99  # Top 1%
            
            lower_bound = df[col].quantile(lower_percentile)
            upper_bound = df[col].quantile(upper_percentile)
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            logger.info(f"Column {col}: Percentile bounds [{lower_bound:.3f}, {upper_bound:.3f}]")
            
        elif col_method == 'isolation_forest':
            # Isolation Forest method
            data_reshaped = df[col].values.reshape(-1, 1)
            isolation_forest = IsolationForest(contamination=0.05, random_state=42)
            outlier_predictions = isolation_forest.fit_predict(data_reshaped)
            outlier_mask = outlier_predictions == -1  # -1 for outliers, 1 for inliers
            
        elif col_method == 'dbscan':
            # DBSCAN method
            scaler = StandardScaler()
            data_reshaped = df[col].values.reshape(-1, 1)
            data_scaled = scaler.fit_transform(data_reshaped)
            
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(data_scaled)
            outlier_mask = clusters == -1  # -1 for noise points (outliers)
        
        # Count outliers
        outlier_count = outlier_mask.sum()
        outlier_percentage = 100 * outlier_count / len(df)
        
        logger.info(f"Column {col}: Identified {outlier_count} outliers ({outlier_percentage:.2f}%)")
        
        # Save outlier statistics
        outlier_stats[col] = {
            'method': col_method,
            'treatment': col_treatment,
            'count': outlier_count,
            'percentage': outlier_percentage
        }
        
        # Store outlier indices for possible row removal
        column_outlier_indices = df.index[outlier_mask].tolist()
        all_outlier_indices.update(column_outlier_indices)
        
        # Visualize outliers if requested
        if visualize and outlier_count > 0:
            plt.figure(figsize=(12, 6))
            
            # Before treatment
            plt.subplot(1, 2, 1)
            sns.boxplot(y=df[col])
            plt.title(f"Before outlier treatment\n{outlier_count} outliers detected")
            
            # Create "after treatment" version based on treatment strategy
            temp_series = df[col].copy()
            
            if col_treatment == 'cap':
                if col_method == 'iqr':
                    temp_series = temp_series.clip(lower=lower_bound, upper=upper_bound)
                elif col_method == 'percentile':
                    temp_series = temp_series.clip(lower=lower_bound, upper=upper_bound)
                elif col_method == 'zscore':
                    lower_z = mean - 3 * std
                    upper_z = mean + 3 * std
                    temp_series = temp_series.clip(lower=lower_z, upper=upper_z)
            elif col_treatment in ['mean', 'median']:
                if col_treatment == 'mean':
                    replace_value = df[col].mean()
                else:
                    replace_value = df[col].median()
                temp_series[outlier_mask] = replace_value
            
            # After treatment (if applicable)
            plt.subplot(1, 2, 2)
            sns.boxplot(y=temp_series)
            plt.title(f"After {col_treatment} treatment")
            
            plt.tight_layout()
            plt.savefig(f"figures/outliers/{col}_outliers.png")
            plt.close()
        
        # Apply treatment strategy
        if col_treatment == 'none':
            continue
        elif treatment == 'remove':
            # Don't remove rows yet, collect all outliers from all columns first
            continue
        else:
            # Apply treatment strategy for this column
            if col_treatment == 'cap':
                if col_method == 'iqr':
                    result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
                elif col_method == 'percentile':
                    result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
                elif col_method in ['zscore', 'isolation_forest', 'dbscan']:
                    # Calculate bounds for zscore or model-based methods
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    lower_z = mean_val - 3 * std_val
                    upper_z = mean_val + 3 * std_val
                    result[col] = result[col].clip(lower=lower_z, upper=upper_z)
            
            elif col_treatment == 'mean':
                replace_value = df[col].mean()
                result.loc[outlier_mask, col] = replace_value
                
            elif col_treatment == 'median':
                replace_value = df[col].median()
                result.loc[outlier_mask, col] = replace_value
                
            elif col_treatment == 'winsorize':
                # Winsorization - replace extreme values with percentiles
                lower_percentile = df[col].quantile(0.05)  # 5th percentile
                upper_percentile = df[col].quantile(0.95)  # 95th percentile
                result[col] = result[col].clip(lower=lower_percentile, upper=upper_percentile)
    
    # Apply row removal if that's the treatment strategy
    if treatment == 'remove' and all_outlier_indices:
        # Remove rows containing any outliers
        original_row_count = len(result)
        result = result.drop(index=list(all_outlier_indices))
        removed_count = original_row_count - len(result)
        
        logger.info(f"Removed {removed_count} rows containing outliers")
        outlier_stats['rows_removed'] = removed_count
    
    # Create summary visualization of all numeric columns if requested
    if visualize and len(columns) > 1:
        try:
            plt.figure(figsize=(15, 10))
            
            # Create a subplot grid based on number of numeric columns
            num_cols = len(columns)
            rows = (num_cols + 2) // 3  # Ceiling division to determine number of rows
            
            for i, col in enumerate(columns, 1):
                if not pd.api.types.is_numeric_dtype(df[col].dtype):
                    continue
                
                plt.subplot(rows, 3, i)
                
                # Plot both original and cleaned data
                sns.kdeplot(df[col], color='red', label='Original')
                sns.kdeplot(result[col], color='blue', label='Cleaned')
                
                plt.title(f"{col}")
                plt.legend()
            
            plt.tight_layout()
            plt.savefig("figures/outliers/all_columns_comparison.png")
            plt.close()
            
            logger.info("Saved outlier visualization to figures/outliers/all_columns_comparison.png")
        except Exception as e:
            logger.error(f"Error creating summary visualization: {e}")
    
    # Add overall statistics
    outlier_stats['total_outlier_columns'] = sum(1 for col_stats in outlier_stats.values() 
                                               if isinstance(col_stats, dict) and col_stats.get('count', 0) > 0)
    
    return result, outlier_stats

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

def advanced_outlier_detection(df: pd.DataFrame, columns: Optional[list] = None, 
                             contamination: float = 0.05, 
                             n_neighbors: int = 20,
                             visualize: bool = False,
                             robust_pca_enabled: bool = True,
                             lof_enabled: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Advanced outlier detection using ensemble methods combining multiple techniques.
    
    This function uses more sophisticated detection methods:
    1. Robust PCA (Principal Component Analysis) to detect multidimensional outliers
    2. Local Outlier Factor to detect local density-based outliers
    
    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers. If None, uses all numeric columns.
        contamination: Expected proportion of outliers in the dataset (0.0 to 0.5)
        n_neighbors: Number of neighbors to consider for LOF
        visualize: Whether to create visualizations of outliers
        robust_pca_enabled: Whether to use Robust PCA for detection
        lof_enabled: Whether to use Local Outlier Factor for detection
        
    Returns:
        Tuple of (DataFrame with outliers identified, dictionary of outlier statistics)
    """
    logger.info("Starting advanced outlier detection")
    
    result = df.copy()
    outlier_stats = {}
    
    try:
        # Import required libraries
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.decomposition import PCA
        from sklearn.covariance import EllipticEnvelope
        import numpy as np
        
        if visualize:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from pathlib import Path
            
            # Create figures directory
            figures_dir = Path('figures/advanced_outliers')
            figures_dir.mkdir(parents=True, exist_ok=True)
    
    except ImportError as e:
        logger.error(f"Required libraries not available: {e}. Falling back to standard outlier detection.")
        # Fall back to standard outlier detection
        return handle_outliers(df, method='iqr', treatment='none', columns=columns, visualize=visualize)
    
    # Identify numeric columns if not specified
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Filter out columns with too many missing values or zero variance
    filtered_columns = []
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            continue
        if df[col].isna().sum() / len(df) > 0.3:  # More than 30% missing
            logger.warning(f"Column {col} has >30% missing values. Excluding from advanced outlier detection.")
            continue
        if df[col].std() == 0:  # Zero variance
            logger.warning(f"Column {col} has zero variance. Excluding from advanced outlier detection.")
            continue
        filtered_columns.append(col)
    
    if not filtered_columns:
        logger.warning("No suitable numeric columns for advanced outlier detection.")
        return result, {'error': 'No suitable numeric columns found'}
    
    # Prepare the data - standardize features
    X = df[filtered_columns].copy()
    
    # Handle any remaining NaN values by filling with column median
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a mask to track outliers from each method
    outlier_masks = {}
    
    # Method 1: Robust PCA with Mahalanobis distance
    if robust_pca_enabled and len(filtered_columns) > 1:
        try:
            # Compute robust covariance and Mahalanobis distances
            ee = EllipticEnvelope(contamination=contamination, random_state=42)
            ee.fit(X_scaled)
            scores_pca = ee.decision_function(X_scaled)
            
            # Lower scores (more negative) = more likely to be outliers
            pca_mask = ee.predict(X_scaled) == -1  # -1 for outliers
            outlier_masks['robust_pca'] = pca_mask
            
            logger.info(f"Robust PCA identified {pca_mask.sum()} outliers ({100 * pca_mask.sum() / len(df):.2f}%)")
            
            if visualize:
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(scores_pca)), np.sort(scores_pca), c='blue')
                plt.axhline(y=ee.threshold_, color='red', linestyle='--', 
                           label=f'Threshold: {ee.threshold_:.3f}')
                plt.xlabel('Samples (sorted by score)')
                plt.ylabel('Outlier Score')
                plt.title('Robust PCA Outlier Scores')
                plt.legend()
                plt.savefig('figures/advanced_outliers/robust_pca_scores.png')
                plt.close()
        except Exception as e:
            logger.warning(f"Robust PCA failed: {e}")
    
    # Method 2: Local Outlier Factor
    if lof_enabled:
        try:
            # Adjust n_neighbors if it's larger than sample size
            adjusted_n_neighbors = min(n_neighbors, len(X) - 1)
            if adjusted_n_neighbors < n_neighbors:
                logger.warning(f"Reduced n_neighbors from {n_neighbors} to {adjusted_n_neighbors} due to sample size")
            
            # Apply LOF
            lof = LocalOutlierFactor(n_neighbors=adjusted_n_neighbors, contamination=contamination)
            lof_pred = lof.fit_predict(X_scaled)
            lof_mask = lof_pred == -1  # -1 for outliers
            outlier_masks['lof'] = lof_mask
            
            # Get the negative outlier factor (higher = more outlier-ish)
            lof_scores = -lof.negative_outlier_factor_
            
            logger.info(f"LOF identified {lof_mask.sum()} outliers ({100 * lof_mask.sum() / len(df):.2f}%)")
            
            if visualize:
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(lof_scores)), np.sort(lof_scores), c='blue')
                threshold = np.sort(lof_scores)[int((1 - contamination) * len(lof_scores))]
                plt.axhline(y=threshold, color='red', linestyle='--', 
                           label=f'Threshold: {threshold:.3f}')
                plt.xlabel('Samples (sorted by score)')
                plt.ylabel('Outlier Score')
                plt.title('Local Outlier Factor Scores')
                plt.legend()
                plt.savefig('figures/advanced_outliers/lof_scores.png')
                plt.close()
        except Exception as e:
            logger.warning(f"LOF detection failed: {e}")
    
    # Combine outlier detections using ensemble voting
    if len(outlier_masks) == 0:
        logger.warning("All advanced outlier detection methods failed")
        return result, {'error': 'All advanced outlier detection methods failed'}
    
    # Final outlier mask - sample is outlier if detected by any method
    final_outlier_mask = pd.Series(False, index=df.index)
    
    for method_name, mask in outlier_masks.items():
        final_outlier_mask = final_outlier_mask | pd.Series(mask, index=df.index)
        outlier_stats[f'{method_name}_count'] = mask.sum()
        outlier_stats[f'{method_name}_percentage'] = 100 * mask.sum() / len(df)
    
    outlier_count = final_outlier_mask.sum()
    outlier_percentage = 100 * outlier_count / len(df)
    
    logger.info(f"Combined methods identified {outlier_count} outliers ({outlier_percentage:.2f}%)")
    
    # Store statistics
    outlier_stats['total_count'] = outlier_count
    outlier_stats['total_percentage'] = outlier_percentage
    outlier_stats['outlier_indices'] = df.index[final_outlier_mask].tolist()
    
    # Create a visualization comparing clean data to data with outliers
    if visualize and outlier_count > 0 and len(filtered_columns) > 1:
        try:
            # Use PCA to visualize in 2D
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            plt.figure(figsize=(12, 10))
            
            # Plot all points with outliers highlighted
            plt.scatter(X_pca[~final_outlier_mask, 0], X_pca[~final_outlier_mask, 1], 
                      c='blue', label='Normal', alpha=0.5)
            plt.scatter(X_pca[final_outlier_mask, 0], X_pca[final_outlier_mask, 1], 
                      c='red', label='Outlier', alpha=0.7)
            
            plt.title(f'PCA Visualization of Advanced Outlier Detection\n{outlier_count} outliers detected ({outlier_percentage:.2f}%)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.savefig('figures/advanced_outliers/pca_visualization.png')
            plt.close()
            
            logger.info("Saved PCA visualization of outliers to figures/advanced_outliers/pca_visualization.png")
        except Exception as e:
            logger.error(f"Error creating PCA visualization: {e}")
    
    # Return results with outlier stats
    result['is_outlier'] = final_outlier_mask
    
    return result, outlier_stats

def enhanced_memory_optimization(df: pd.DataFrame, aggressive: bool = False, 
                              convert_sparse: bool = True, 
                              deduplicate: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Enhanced memory optimization with more aggressive techniques.
    
    This function extends the basic memory optimization with:
    1. Column deduplication - Detect and remove duplicate columns
    2. Sparse matrix conversion - Convert sparse dataframes to sparse format
    3. More aggressive downcasting of numeric types
    4. Category optimization with hash tables for high-cardinality columns
    
    Args:
        df: Input DataFrame to optimize
        aggressive: Whether to use aggressive downcasting (may sacrifice some precision)
        convert_sparse: Whether to convert sparse dataframes to sparse format
        deduplicate: Whether to detect and remove duplicate columns
        
    Returns:
        Tuple of (optimized DataFrame, dictionary of transformations applied)
    """
    logger.info("Starting enhanced memory optimization")
    
    # Track memory usage before optimization
    start_memory = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory usage before optimization: {start_memory:.2f} MB")
    
    # Track transformations applied to each column
    transformations = {}
    
    # Create a copy of the DataFrame to avoid modifying the original
    result = df.copy()
    
    # Step 1: Check for duplicate columns
    if deduplicate:
        try:
            # Calculate column hash values for quick comparison
            logger.info("Checking for duplicate columns")
            column_hashes = {}
            duplicate_cols = []
            
            for col in df.columns:
                # Create a hash from column values
                col_hash = hash(tuple(pd.util.hash_array(df[col].values)))
                
                if col_hash in column_hashes:
                    # This is a duplicate of another column
                    original_col = column_hashes[col_hash]
                    duplicate_cols.append(col)
                    transformations[col] = {
                        'action': 'removed_duplicate', 
                        'duplicate_of': original_col
                    }
                    logger.info(f"Column '{col}' is a duplicate of '{original_col}'")
                else:
                    column_hashes[col_hash] = col
            
            # Remove duplicate columns
            if duplicate_cols:
                logger.info(f"Removing {len(duplicate_cols)} duplicate columns")
                result = result.drop(columns=duplicate_cols)
        except Exception as e:
            logger.warning(f"Error in duplicate column detection: {e}")
    
    # Step 2: Process each column for type optimization
    for col in result.columns:
        col_type = result[col].dtype
        
        # Initialize transformation tracking
        transformations[col] = {'original_type': str(col_type), 'new_type': str(col_type), 'memory_saved_kb': 0}
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_type):
            # Integer columns
            if pd.api.types.is_integer_dtype(col_type):
                # Get min and max values
                c_min = result[col].min()
                c_max = result[col].max()
                
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
                result[col] = result[col].astype(new_type)
                transformations[col]['new_type'] = str(new_type)
                logger.info(f"Optimized column '{col}' from {col_type} to {new_type}")
            
            # Float columns - more aggressive downcasting based on parameter
            elif col_type in (np.float64, np.float32):
                if aggressive:
                    # Check the decimal precision actually needed
                    # Extract the part after the decimal point
                    decimals = result[col].apply(lambda x: 
                                               str(x).split('.')[-1] if '.' in str(x) else '')
                    
                    # Calculate the max number of decimal places needed
                    max_precision = decimals.apply(len).max()
                    
                    if max_precision <= 7:  # float32 has ~7 decimal precision
                        new_type = np.float32
                    else:
                        new_type = np.float64
                        
                    # For very limited precision, consider using float16
                    if max_precision <= 3:  # float16 has ~3 decimal precision
                        try:
                            # Test if float16 will cause overflow
                            c_min = result[col].min()
                            c_max = result[col].max()
                            
                            if c_min > -65500 and c_max < 65500:  # Approximate float16 range
                                new_type = np.float16
                        except:
                            pass  # If error, stick with higher precision
                else:
                    # Standard downcast
                    new_type = np.float32 if col_type == np.float64 else col_type
                
                result[col] = result[col].astype(new_type)
                transformations[col]['new_type'] = str(new_type)
                logger.info(f"Optimized column '{col}' from {col_type} to {new_type}")
        
        # Categorical and object columns - optimize based on cardinality
        elif col_type == 'object':
            n_unique = result[col].nunique()
            
            # Calculate the ratio of unique values
            unique_ratio = n_unique / len(result)
            
            if unique_ratio < 0.5:  # If less than 50% of values are unique
                if n_unique > 100 and convert_sparse:  # High cardinality but still worth categorizing
                    try:
                        # Use category dtype with a more efficient hash table representation
                        result[col] = result[col].astype('category')
                        transformations[col]['new_type'] = 'category'
                    except Exception as e:
                        logger.warning(f"Error converting column '{col}' to category: {e}")
                else:
                    # Regular category conversion for low-medium cardinality
                    result[col] = result[col].astype('category')
                    transformations[col]['new_type'] = 'category'
    
    # Step 3: Convert sparse numeric data to sparse format if enabled
    if convert_sparse:
        logger.info("Checking for sparsity in numeric columns")
        for col in result.columns:
            try:
                # Only process numeric columns
                if not pd.api.types.is_numeric_dtype(result[col].dtype):
                    continue
                    
                # Check sparsity (percentage of zeros)
                zero_percentage = (result[col] == 0).mean() * 100
                
                # If column is sparse (more than 50% zeros), convert to sparse format
                if zero_percentage > 50:
                    # Get the fill value (usually 0)
                    fill_value = 0
                    
                    # Convert to sparse
                    result[col] = result[col].astype(pd.SparseDtype(result[col].dtype, fill_value))
                    
                    transformations[col]['new_type'] = f'sparse[{result[col].dtype}]'
                    transformations[col]['sparse_density'] = f"{100 - zero_percentage:.2f}%"
                    
                    logger.info(f"Converted column '{col}' to sparse format (density: {100 - zero_percentage:.2f}%)")
            except Exception as e:
                logger.warning(f"Error in sparse conversion for column '{col}': {e}")
    
    # Calculate memory usage after optimization
    end_memory = result.memory_usage(deep=True).sum() / 1024**2
    
    # Update memory savings for each column
    for col in result.columns:
        try:
            before = df[col].memory_usage(deep=True) / 1024  # KB
            after = result[col].memory_usage(deep=True) / 1024  # KB
            transformations[col]['memory_saved_kb'] = before - after
        except:
            transformations[col]['memory_saved_kb'] = 0
    
    # Log the results
    memory_saved = start_memory - end_memory
    memory_ratio = start_memory / end_memory if end_memory > 0 else float('inf')
    
    logger.info(f"Memory usage after enhanced optimization: {end_memory:.2f} MB")
    logger.info(f"Memory saved: {memory_saved:.2f} MB ({memory_ratio:.1f}x reduction)")
    
    # Add summary stats to transformations
    transformations['__summary__'] = {
        'original_memory_mb': start_memory,
        'optimized_memory_mb': end_memory,
        'memory_saved_mb': memory_saved,
        'reduction_ratio': memory_ratio,
        'duplicate_columns_removed': len(df.columns) - len(result.columns) if deduplicate else 0
    }
    
    return result, transformations

def preprocess_pipeline(input_file: str, output_dir: str, target_column: str,
                       missing_strategy: str = 'median', outlier_method: str = 'iqr',
                       outlier_treatment: str = 'cap', visualize_outliers: bool = False,
                       column_specific_outlier_params: Optional[Dict[str, Dict]] = None,
                       advanced_outlier_detection_enabled: bool = False,
                       enhanced_memory_optimization_enabled: bool = False,
                       aggressive_memory_optimization: bool = False) -> dict:
    """
    Complete preprocessing pipeline that:
    1. Loads data
    2. Handles missing values
    3. Optimizes memory usage (standard or enhanced)
    4. Handles outliers (standard or advanced)
    5. Splits data
    6. Saves processed data
    
    Args:
        input_file: Path to input file
        output_dir: Output directory for processed data
        target_column: Name of target column
        missing_strategy: Strategy for handling missing values
        outlier_method: Method for handling outliers
        outlier_treatment: Treatment strategy for outliers
        visualize_outliers: Whether to create visualizations of outliers
        column_specific_outlier_params: Dictionary with column-specific parameters for outlier handling
        advanced_outlier_detection_enabled: Whether to use advanced outlier detection
        enhanced_memory_optimization_enabled: Whether to use enhanced memory optimization
        aggressive_memory_optimization: Whether to use aggressive memory optimization strategies
    
    Returns:
        Dictionary containing preprocessing statistics and metadata
    """
    logger.info("Starting preprocessing pipeline")
    
    # Load data
    df = load_data(input_file)
    
    # Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)
    
    # Optimize memory usage
    if enhanced_memory_optimization_enabled:
        logger.info("Using enhanced memory optimization")
        df, memory_transformations = enhanced_memory_optimization(
            df, 
            aggressive=aggressive_memory_optimization,
            convert_sparse=True,
            deduplicate=True
        )
    else:
        logger.info("Using standard memory optimization")
        df, memory_transformations = optimize_memory(df)
    
    # Handle outliers
    if advanced_outlier_detection_enabled:
        logger.info("Using advanced outlier detection")
        df, outlier_stats = advanced_outlier_detection(
            df,
            visualize=visualize_outliers,
            robust_pca_enabled=True, 
            lof_enabled=True
        )
        
        # Apply treatment to detected outliers if needed
        if outlier_treatment != 'none' and 'is_outlier' in df.columns:
            logger.info(f"Applying {outlier_treatment} treatment to detected outliers")
            
            # Get outlier mask
            outlier_mask = df['is_outlier']
            
            # Apply treatment
            if outlier_treatment == 'remove':
                original_row_count = len(df)
                df = df[~outlier_mask].drop(columns=['is_outlier'])
                removed_count = original_row_count - len(df)
                logger.info(f"Removed {removed_count} rows containing outliers")
                outlier_stats['rows_removed'] = removed_count
            else:
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col != 'is_outlier']
                
                for col in numeric_cols:
                    if outlier_treatment == 'mean':
                        fill_value = df[col].mean()
                        df.loc[outlier_mask, col] = fill_value
                    elif outlier_treatment == 'median':
                        fill_value = df[col].median()
                        df.loc[outlier_mask, col] = fill_value
                    elif outlier_treatment == 'cap':
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        df.loc[outlier_mask & (df[col] < lower_bound), col] = lower_bound
                        df.loc[outlier_mask & (df[col] > upper_bound), col] = upper_bound
                
                # Remove the outlier marker column after treatment
                df = df.drop(columns=['is_outlier'])
    else:
        logger.info("Using standard outlier detection")
        df, outlier_stats = handle_outliers(
            df, 
            method=outlier_method,
            treatment=outlier_treatment,
            visualize=visualize_outliers,
            column_specific_params=column_specific_outlier_params
        )
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, output_dir)
    
    # Compile preprocessing metadata
    metadata = {
        'original_shape': df.shape,
        'memory_transformations': memory_transformations,
        'outlier_statistics': outlier_stats,
        'train_shape': (X_train.shape[0], X_train.shape[1]),
        'test_shape': (X_test.shape[0], X_test.shape[1]),
        'class_distribution': {
            'train': y_train.value_counts().to_dict(),
            'test': y_test.value_counts().to_dict()
        }
    }
    
    logger.info("Preprocessing pipeline completed successfully")
    
    return metadata

def generate_optimization_report(memory_transformations: Dict, 
                              output_dir: str = 'reports') -> str:
    """
    Generate a detailed report on memory optimization results.
    
    Args:
        memory_transformations: Dictionary of memory transformations from optimization
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report file
    """
    logger.info("Generating memory optimization report")
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from datetime import datetime
    from pathlib import Path
    
    # Create reports directory
    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Format date for filename
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(output_dir, f'memory_optimization_report_{date_str}.html')
    
    # Extract summary statistics
    summary = memory_transformations.get('__summary__', {})
    original_memory = summary.get('original_memory_mb', 0)
    optimized_memory = summary.get('optimized_memory_mb', 0)
    memory_saved = summary.get('memory_saved_mb', 0)
    reduction_ratio = summary.get('reduction_ratio', 1)
    duplicate_cols_removed = summary.get('duplicate_columns_removed', 0)
    
    # Create a DataFrame for column transformations
    columns_data = []
    
    for col_name, col_info in memory_transformations.items():
        if col_name == '__summary__':
            continue
            
        if isinstance(col_info, dict) and 'original_type' in col_info:
            columns_data.append({
                'Column': col_name,
                'Original Type': col_info.get('original_type', ''),
                'New Type': col_info.get('new_type', ''),
                'Memory Saved (KB)': col_info.get('memory_saved_kb', 0),
                'Sparse Density': col_info.get('sparse_density', '')
            })
        elif isinstance(col_info, dict) and 'action' in col_info:
            columns_data.append({
                'Column': col_name,
                'Action': col_info.get('action', ''),
                'Duplicate Of': col_info.get('duplicate_of', '')
            })
    
    # Create DataFrame and sort by memory saved
    columns_df = pd.DataFrame(columns_data)
    if not columns_df.empty and 'Memory Saved (KB)' in columns_df.columns:
        columns_df = columns_df.sort_values('Memory Saved (KB)', ascending=False)
    
    # Generate visualization - Memory savings by column
    plt.figure(figsize=(12, 8))
    
    if not columns_df.empty and 'Memory Saved (KB)' in columns_df.columns:
        top_columns = columns_df.nlargest(15, 'Memory Saved (KB)')
        plt.barh(top_columns['Column'], top_columns['Memory Saved (KB)'])
        plt.xlabel('Memory Saved (KB)')
        plt.ylabel('Column')
        plt.title('Top 15 Columns by Memory Savings')
        plt.tight_layout()
        
        # Save visualization
        chart_file = os.path.join(output_dir, f'memory_savings_chart_{date_str}.png')
        plt.savefig(chart_file)
        plt.close()
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Memory Optimization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .highlight {{ color: #e74c3c; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .chart {{ margin: 20px 0; max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>Memory Optimization Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Original Memory Usage: <span class="highlight">{original_memory:.2f} MB</span></p>
            <p>Optimized Memory Usage: <span class="highlight">{optimized_memory:.2f} MB</span></p>
            <p>Memory Saved: <span class="highlight">{memory_saved:.2f} MB ({reduction_ratio:.1f}x reduction)</span></p>
            <p>Duplicate Columns Removed: <span class="highlight">{duplicate_cols_removed}</span></p>
        </div>
    """
    
    if not columns_df.empty:
        # Add chart to HTML if it was created
        if 'Memory Saved (KB)' in columns_df.columns:
            html_content += f"""
            <h2>Memory Savings Visualization</h2>
            <img src="{os.path.basename(chart_file)}" class="chart" alt="Memory Savings Chart">
            """
        
        # Add table of column transformations
        html_content += """
        <h2>Column Transformations</h2>
        <table>
            <tr>
        """
        
        # Add appropriate headers based on what's in the dataframe
        for col in columns_df.columns:
            html_content += f"<th>{col}</th>"
        
        html_content += "</tr>"
        
        # Add rows
        for _, row in columns_df.iterrows():
            html_content += "<tr>"
            for col in columns_df.columns:
                value = row.get(col, "")
                html_content += f"<td>{value}</td>"
            html_content += "</tr>"
        
        html_content += """
        </table>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Memory optimization report saved to {report_file}")
    return report_file

def visualize_outlier_impact(df_original: pd.DataFrame, 
                           df_processed: pd.DataFrame,
                           outlier_stats: Dict,
                           target_column: str,
                           output_dir: str = 'reports') -> str:
    """
    Visualize the impact of outlier handling on model features.
    
    Args:
        df_original: Original DataFrame before outlier handling
        df_processed: Processed DataFrame after outlier handling
        outlier_stats: Statistics from outlier detection
        target_column: Target variable column name
        output_dir: Directory to save visualizations
        
    Returns:
        Path to the generated report file
    """
    logger.info("Generating outlier impact visualizations")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os
        from datetime import datetime
        from pathlib import Path
        
        # Create reports directory
        reports_dir = Path(output_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Format date for filename
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f'outlier_impact_report_{date_str}.html')
        
        # Get numeric columns excluding the target
        numeric_cols = df_original.select_dtypes(include=np.number).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        # Limit to top columns if there are too many
        if len(numeric_cols) > 10:
            # If outlier_stats has info about which columns had most outliers, use that
            if any(isinstance(val, dict) and 'count' in val for key, val in outlier_stats.items() if isinstance(key, str)):
                cols_with_outliers = {k: v['count'] for k, v in outlier_stats.items() 
                                   if isinstance(k, str) and isinstance(v, dict) and 'count' in v}
                top_cols = sorted(cols_with_outliers.items(), key=lambda x: x[1], reverse=True)
                top_cols = [col for col, _ in top_cols[:10] if col in numeric_cols]
                if len(top_cols) > 0:
                    numeric_cols = top_cols
                else:
                    numeric_cols = numeric_cols[:10]
            else:
                numeric_cols = numeric_cols[:10]
        
        # Create visualizations comparing original vs processed data
        plt.figure(figsize=(15, len(numeric_cols) * 4))
        
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(len(numeric_cols), 2, i*2-1)
            sns.boxplot(y=df_original[col])
            plt.title(f"{col} - Original")
            
            plt.subplot(len(numeric_cols), 2, i*2)
            sns.boxplot(y=df_processed[col])
            plt.title(f"{col} - After Outlier Treatment")
        
        plt.tight_layout()
        boxplot_file = os.path.join(output_dir, f'outlier_boxplots_{date_str}.png')
        plt.savefig(boxplot_file)
        plt.close()
        
        # Create density plot comparisons
        plt.figure(figsize=(15, len(numeric_cols) * 3))
        
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(len(numeric_cols), 1, i)
            sns.kdeplot(df_original[col], label='Original', color='red')
            sns.kdeplot(df_processed[col], label='Processed', color='blue')
            plt.title(f"Distribution of {col}")
            plt.legend()
        
        plt.tight_layout()
        density_file = os.path.join(output_dir, f'outlier_densities_{date_str}.png')
        plt.savefig(density_file)
        plt.close()
        
        # If PCA visualization was generated by advanced outlier detection, include it
        pca_file = 'figures/advanced_outliers/pca_visualization.png'
        has_pca_viz = os.path.exists(pca_file)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Outlier Treatment Impact Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .highlight {{ color: #e74c3c; font-weight: bold; }}
                .img-container {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Outlier Treatment Impact Report</h1>
            <div class="summary">
                <h2>Summary</h2>
        """
        
        # Add outlier statistics
        if 'total_count' in outlier_stats:
            html_content += f"""
                <p>Total Outliers Detected: <span class="highlight">{outlier_stats['total_count']}</span>
                   ({outlier_stats.get('total_percentage', 0):.2f}% of the data)</p>
            """
        
        # Add method-specific outlier counts
        for key, value in outlier_stats.items():
            if key.endswith('_count') and not key == 'total_count':
                method = key.replace('_count', '')
                percentage = outlier_stats.get(f"{method}_percentage", 0)
                html_content += f"""
                <p>{method.replace('_', ' ').title()}: <span class="highlight">{value}</span> 
                   outliers ({percentage:.2f}%)</p>
                """
        
        html_content += """
            </div>
        """
        
        # Add PCA visualization if it exists
        if has_pca_viz:
            html_content += f"""
            <h2>Multivariate Outlier Visualization (PCA)</h2>
            <div class="img-container">
                <img src="../{pca_file}" alt="PCA Outlier Visualization">
            </div>
            """
        
        # Add boxplot comparisons
        html_content += f"""
            <h2>Box Plot Comparisons</h2>
            <p>Comparison of feature distributions before and after outlier treatment</p>
            <div class="img-container">
                <img src="{os.path.basename(boxplot_file)}" alt="Outlier Box Plots">
            </div>
        """
        
        # Add density plot comparisons
        html_content += f"""
            <h2>Density Plot Comparisons</h2>
            <p>Comparison of feature density distributions before and after outlier treatment</p>
            <div class="img-container">
                <img src="{os.path.basename(density_file)}" alt="Outlier Density Plots">
            </div>
        """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Outlier impact report saved to {report_file}")
        return report_file
        
    except Exception as e:
        logger.error(f"Error generating outlier impact visualizations: {e}")
        return ""

def optimize_pipeline_for_production(input_data: Union[pd.DataFrame, dict],
                                   model_path: str = 'models/best_model_retrained.pkl',
                                   memory_optimize: bool = True,
                                   outlier_detection: bool = True,
                                   contamination: float = 0.05) -> Tuple[pd.DataFrame, bool, Dict]:
    """
    Optimized pipeline for production use in the Flask API.
    
    This function:
    1. Takes input data (either a DataFrame or a single patient dict)
    2. Applies appropriate memory optimization 
    3. Performs outlier detection and provides a warning if outliers are detected
    4. Returns processed data ready for model prediction
    
    Args:
        input_data: Either a DataFrame or a dictionary with patient data
        model_path: Path to the trained model
        memory_optimize: Whether to apply memory optimization
        outlier_detection: Whether to perform outlier detection
        contamination: Expected proportion of outliers in the dataset
        
    Returns:
        Tuple of (processed_data, outlier_warning_flag, stats_dict)
    """
    import joblib
    import numpy as np
    import pandas as pd
    from pathlib import Path
    
    logger.info("Running optimized pipeline for production")
    
    # Convert dict to DataFrame if needed
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
        single_patient = True
    else:
        df = input_data.copy()
        single_patient = False
    
    stats = {}
    outlier_warning = False
    
    # Load the model to get feature names and expected types
    try:
        model = joblib.load(model_path)
        
        # Try to extract feature names from model if available
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            logger.info(f"Model expects {len(expected_features)} features")
            
            # Ensure all expected features are present
            missing_features = [feat for feat in expected_features if feat not in df.columns]
            if missing_features:
                logger.warning(f"Missing expected features: {missing_features}")
                
                # Fill missing features with zeros (or appropriate default values)
                for feat in missing_features:
                    df[feat] = 0
                    
            # Keep only features expected by the model
            df = df[expected_features]
    except Exception as e:
        logger.warning(f"Could not load model for feature validation: {e}")
    
    # Fill missing values with median for numeric columns, mode for categorical
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    # Optimize memory usage if requested
    if memory_optimize:
        original_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Apply memory optimization
        if df.shape[0] > 1:
            # For multiple patients, use the full optimization
            df, memory_stats = enhanced_memory_optimization(df, aggressive=True)
        else:
            # For a single patient, just do basic type conversion
            for col in df.select_dtypes(include=np.number).columns:
                if df[col].dtype == np.float64:
                    df[col] = df[col].astype(np.float32)
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_stats = {
                "__summary__": {
                    "original_memory_mb": original_memory,
                    "optimized_memory_mb": optimized_memory,
                    "memory_saved_mb": original_memory - optimized_memory
                }
            }
        
        stats['memory_optimization'] = memory_stats.get('__summary__', {})
    
    # Perform outlier detection if requested
    if outlier_detection and df.shape[0] > 0:
        # Skip advanced methods for single patient and use simpler approach
        if single_patient:
            # Use Z-scores to check for outliers in each feature
            outlier_cols = []
            
            for col in df.select_dtypes(include=np.number).columns:
                try:
                    # Load historical mean and std from a stats file (would need to be generated)
                    stats_file = Path('models/feature_statistics.pkl')
                    if stats_file.exists():
                        col_stats = joblib.load(stats_file)
                        if col in col_stats:
                            mean = col_stats[col]['mean']
                            std = col_stats[col]['std']
                            z_score = abs((df[col].values[0] - mean) / std)
                            if z_score > 3:
                                outlier_cols.append((col, z_score))
                    else:
                        # If no stats file, we can't detect outliers for single patient
                        logger.warning("No feature statistics file found for outlier detection")
                except Exception as e:
                    logger.warning(f"Error in single-patient outlier detection for {col}: {e}")
            
            if outlier_cols:
                outlier_warning = True
                stats['outliers'] = {
                    'outlier_features': outlier_cols,
                    'warning': 'Patient has unusual values in some features'
                }
        else:
            # For datasets, use the advanced outlier detection
            outlier_df, outlier_stats = advanced_outlier_detection(
                df, contamination=contamination, visualize=False
            )
            
            # Check if outliers were detected
            if 'is_outlier' in outlier_df.columns:
                outlier_count = outlier_df['is_outlier'].sum()
                if outlier_count > 0:
                    outlier_warning = True
                    stats['outliers'] = {
                        'count': outlier_count,
                        'percentage': 100 * outlier_count / len(df)
                    }
                
                # Remove the outlier marker column
                df = outlier_df.drop(columns=['is_outlier'])
            else:
                stats['outliers'] = outlier_stats
    
    return df, outlier_warning, stats

def setup_optimization_for_flask_app(app):
    """
    Setup memory and outlier optimization for a Flask app.
    
    This function adds routes and utilities to a Flask application for:
    1. Memory usage monitoring
    2. Outlier detection in incoming patient data
    3. Generating optimization reports
    
    Args:
        app: Flask application instance
    """
    from flask import jsonify, request, render_template
    import psutil
    import os
    
    # Add a route to monitor memory usage
    @app.route('/api/memory-usage')
    def memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return jsonify({
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent()
        })
    
    # Add a route to generate optimization reports
    @app.route('/api/generate-optimization-report')
    def generate_report():
        try:
            # In a real app, you would pass actual data here
            # This is just a placeholder
            import pandas as pd
            df = pd.read_csv('path/to/your/data.csv')
            
            # Optimize memory
            df_optimized, memory_stats = enhanced_memory_optimization(df)
            
            # Generate report
            report_path = generate_optimization_report(memory_stats)
            
            return jsonify({'success': True, 'report_path': report_path})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # Add memory optimization middleware to the Flask app
    @app.before_request
    def optimize_memory_before_request():
        # This runs before each request
        pass  # Add specific optimizations if needed
    
    @app.after_request
    def add_outlier_warnings(response):
        # For endpoints that return patient data
        if request.endpoint == 'predict':  # Assuming you have a 'predict' endpoint
            # You can modify the response to include outlier warnings
            pass
        return response
    
    # Add Jinja2 template filter for formatting memory sizes
    @app.template_filter('format_bytes')
    def format_bytes(num_bytes):
        """Convert bytes to human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if num_bytes < 1024:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024
        return f"{num_bytes:.2f} PB"
    
    logger.info("Memory optimization and outlier detection setup complete for Flask app")