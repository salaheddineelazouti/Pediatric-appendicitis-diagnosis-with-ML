"""
Outlier detection utility functions for the pediatric appendicitis diagnosis application.
This module handles different outlier detection methods with robust error handling.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.covariance import MinCovDet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

def detect_outliers(data, method='all', n_neighbors=20, contamination=0.1):
    """
    Detect outliers in the data using multiple methods with robust error handling.
    
    Args:
        data (pd.DataFrame or np.ndarray): The data to analyze for outliers
        method (str): The outlier detection method to use ('robust_pca', 'lof', 'isolation_forest', or 'all')
        n_neighbors (int): Number of neighbors for LocalOutlierFactor
        contamination (float): Expected proportion of outliers in the data
        
    Returns:
        dict: Dictionary containing outlier information for each method that succeeded
    """
    results = {}
    
    # Convert to numpy array if DataFrame
    if isinstance(data, pd.DataFrame):
        data_array = data.to_numpy()
    else:
        data_array = data
    
    # Check if we have enough samples
    if len(data_array) < 2:
        logger.warning("Insufficient samples for outlier detection (minimum 2 required)")
        return {"error": "Insufficient samples for outlier detection"}
    
    # Robust PCA (MinCovDet) for outlier detection
    if method in ['robust_pca', 'all']:
        try:
            # Robust PCA requires more samples than features
            min_samples = min(len(data_array), data_array.shape[1] + 1)
            if min_samples < data_array.shape[1] + 1:
                logger.warning(f"Insufficient samples for robust PCA: {len(data_array)} samples with {data_array.shape[1]} features")
            else:
                robust_pca = MinCovDet(support_fraction=max(0.5, min_samples / len(data_array)))
                robust_pca.fit(data_array)
                mahalanobis_dist = robust_pca.mahalanobis(data_array)
                # Flag as outlier if distance exceeds chi-square critical value
                threshold = np.percentile(mahalanobis_dist, 97.5)  # 97.5th percentile as threshold
                outliers_robust = mahalanobis_dist > threshold
                results['robust_pca'] = {
                    'outliers': outliers_robust,
                    'scores': mahalanobis_dist
                }
        except Exception as e:
            logger.warning(f"Robust PCA outlier detection failed: {str(e)}")
    
    # Local Outlier Factor
    if method in ['lof', 'all']:
        try:
            # Set n_neighbors to min(20, n_samples - 1) to avoid error
            effective_n_neighbors = min(n_neighbors, len(data_array) - 1)
            if effective_n_neighbors < 1:
                logger.warning("Insufficient samples for LOF (need at least 2 samples)")
            else:
                lof = LocalOutlierFactor(n_neighbors=effective_n_neighbors, contamination=contamination)
                outliers_lof = lof.fit_predict(data_array) == -1  # -1 for outliers, 1 for inliers
                # Get the negative outlier factor (higher means more likely to be outlier)
                outlier_scores = -lof.negative_outlier_factor_
                results['lof'] = {
                    'outliers': outliers_lof,
                    'scores': outlier_scores
                }
        except Exception as e:
            logger.warning(f"LOF outlier detection failed: {str(e)}")
    
    # Isolation Forest
    if method in ['isolation_forest', 'all']:
        try:
            # Isolation Forest works even with small sample sizes
            isof = IsolationForest(contamination=contamination, random_state=42)
            outliers_isof = isof.fit_predict(data_array) == -1  # -1 for outliers, 1 for inliers
            # Get anomaly score
            outlier_scores = isof.score_samples(data_array)
            # Invert scores so higher means more likely to be outlier (consistent with other methods)
            outlier_scores = -outlier_scores
            results['isolation_forest'] = {
                'outliers': outliers_isof,
                'scores': outlier_scores
            }
        except Exception as e:
            logger.warning(f"Isolation Forest outlier detection failed: {str(e)}")
    
    # If no methods succeeded, return error
    if not results:
        return {"error": "All outlier detection methods failed"}
    
    # Add combined results if multiple methods were used
    if len(results) > 1:
        try:
            # Combine outliers (a point is an outlier if flagged by any method)
            combined_outliers = np.zeros(len(data_array), dtype=bool)
            for method_results in results.values():
                if 'outliers' in method_results:
                    combined_outliers = combined_outliers | method_results['outliers']
            
            results['combined'] = {
                'outliers': combined_outliers,
                'count': np.sum(combined_outliers)
            }
        except Exception as e:
            logger.warning(f"Error combining outlier results: {str(e)}")
    
    return results


def filter_outliers(data, outlier_info=None, method='all', n_neighbors=20, contamination=0.1):
    """
    Filter outliers from the data.
    
    Args:
        data (pd.DataFrame): The data to filter
        outlier_info (dict, optional): Pre-computed outlier information from detect_outliers
        method (str): The outlier detection method to use if outlier_info not provided
        n_neighbors (int): Number of neighbors for LocalOutlierFactor
        contamination (float): Expected proportion of outliers in the data
        
    Returns:
        tuple: (filtered_data, outlier_indices)
    """
    if outlier_info is None:
        outlier_info = detect_outliers(data, method, n_neighbors, contamination)
    
    if "error" in outlier_info:
        logger.warning(f"Outlier detection returned error: {outlier_info['error']}")
        return data, []
    
    # Use combined results if available, otherwise use the first method's results
    if 'combined' in outlier_info:
        outlier_mask = outlier_info['combined']['outliers']
    else:
        # Get the first method's results
        first_method = list(outlier_info.keys())[0]
        outlier_mask = outlier_info[first_method]['outliers']
    
    # Get indices of outliers
    outlier_indices = np.where(outlier_mask)[0]
    
    # Filter the data
    filtered_data = data[~outlier_mask] if isinstance(data, pd.DataFrame) else data[~outlier_mask]
    
    return filtered_data, outlier_indices
