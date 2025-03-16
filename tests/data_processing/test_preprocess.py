"""
Unit tests for the data preprocessing module.
Tests the data processing functionality, memory optimization, and outlier handling.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_processing.preprocess import (
    load_data,
    optimize_memory,
    handle_missing_values,
    handle_outliers,
    split_data,
    save_processed_data,
    advanced_outlier_detection,
    enhanced_memory_optimization
)

class TestPreprocessing(unittest.TestCase):
    """Test case for the data preprocessing functionality."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a test DataFrame
        self.test_data = pd.DataFrame({
            'age': [5, 10, 15, 8, 12, np.nan],
            'gender': [0, 1, 0, 1, 0, 1],
            'duration': [24.0, 48.0, 12.0, 36.0, 72.0, 8.0],
            'fever': [1, 0, 1, 1, 0, 0],
            'white_blood_cell_count': [15.0, 8.5, 22.0, 10.5, np.nan, 9.0],
            'neutrophil_percentage': [80.0, 65.0, 85.0, 75.0, 60.0, np.nan],
            'c_reactive_protein': [50.0, 15.0, 100.0, 25.0, 10.0, 5.0],
            'appendicitis': [1, 0, 1, 0, 0, 0]  # Target column
        })
        
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.test_dir.name
        
        # Create a test CSV file
        self.test_csv_path = os.path.join(self.output_dir, 'test_data.csv')
        self.test_data.to_csv(self.test_csv_path, index=False)
    
    def tearDown(self):
        """Clean up after each test method."""
        self.test_dir.cleanup()
    
    def test_load_data(self):
        """Test that data can be loaded from a CSV file."""
        df = load_data(self.test_csv_path)
        
        # Check that the data was loaded
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, self.test_data.shape)
        self.assertTrue(all(col in df.columns for col in self.test_data.columns))
    
    def test_optimize_memory(self):
        """Test memory optimization functionality."""
        optimized_df, transformations = optimize_memory(self.test_data)
        
        # Check that the optimized DataFrame has the same data
        self.assertEqual(optimized_df.shape, self.test_data.shape)
        self.assertTrue(all(col in optimized_df.columns for col in self.test_data.columns))
        
        # Check that transformations were recorded
        self.assertIsInstance(transformations, dict)
        self.assertTrue(all(col in transformations for col in self.test_data.columns))
        
        # Check that memory was optimized
        original_memory = self.test_data.memory_usage(deep=True).sum()
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        self.assertLessEqual(optimized_memory, original_memory)
    
    def test_handle_missing_values_median(self):
        """Test handling missing values with median strategy."""
        df_fixed = handle_missing_values(self.test_data, strategy='median')
        
        # Check that no NaN values remain
        self.assertEqual(df_fixed.isnull().sum().sum(), 0)
        
        # Check that the correct values were imputed
        # For age, the median of [5, 10, 15, 8, 12] is 10
        self.assertAlmostEqual(df_fixed.loc[5, 'age'], 10.0)
        
        # For white_blood_cell_count, the median of [15.0, 8.5, 22.0, 10.5, 9.0] is 10.5
        self.assertAlmostEqual(df_fixed.loc[4, 'white_blood_cell_count'], 10.5)
        
        # For neutrophil_percentage, the median of [80.0, 65.0, 85.0, 75.0, 60.0] is 75.0
        self.assertAlmostEqual(df_fixed.loc[5, 'neutrophil_percentage'], 75.0)
    
    def test_handle_missing_values_mean(self):
        """Test handling missing values with mean strategy."""
        df_fixed = handle_missing_values(self.test_data, strategy='mean')
        
        # Check that no NaN values remain
        self.assertEqual(df_fixed.isnull().sum().sum(), 0)
        
        # Calculate the expected mean values
        age_mean = self.test_data['age'].dropna().mean()
        wbc_mean = self.test_data['white_blood_cell_count'].dropna().mean()
        neutro_mean = self.test_data['neutrophil_percentage'].dropna().mean()
        
        # Check that the correct values were imputed
        self.assertAlmostEqual(df_fixed.loc[5, 'age'], age_mean)
        self.assertAlmostEqual(df_fixed.loc[4, 'white_blood_cell_count'], wbc_mean)
        self.assertAlmostEqual(df_fixed.loc[5, 'neutrophil_percentage'], neutro_mean)
    
    def test_handle_outliers_iqr(self):
        """Test IQR-based outlier detection and handling."""
        # Add some outliers to test data
        test_data_with_outliers = self.test_data.copy()
        test_data_with_outliers.loc[6] = [10, 1, 500.0, 1, 50.0, 99.0, 300.0, 1]  # Outliers in duration and c_reactive_protein
        
        # Process outliers
        df_processed, outlier_stats = handle_outliers(
            test_data_with_outliers, 
            method='iqr', 
            treatment='cap'
        )
        
        # Check that outliers were detected
        self.assertGreater(outlier_stats['total_outlier_columns'], 0)
        
        # Check that columns with outliers have stats
        columns_with_outliers = [col for col in test_data_with_outliers.columns 
                               if pd.api.types.is_numeric_dtype(test_data_with_outliers[col].dtype) 
                               and col in outlier_stats
                               and isinstance(outlier_stats[col], dict)
                               and outlier_stats[col].get('count', 0) > 0]
        self.assertGreater(len(columns_with_outliers), 0)
        
        # Check that outliers were capped
        self.assertLess(df_processed['duration'].max(), test_data_with_outliers['duration'].max())
        self.assertLess(df_processed['c_reactive_protein'].max(), test_data_with_outliers['c_reactive_protein'].max())
    
    def test_split_data(self):
        """Test data splitting functionality."""
        # Handle missing values first
        df_clean = handle_missing_values(self.test_data, strategy='median')
        
        # Split the data
        X_train, X_test, y_train, y_test = split_data(df_clean, 'appendicitis')
        
        # Check that the splits have the correct shapes
        self.assertEqual(len(X_train.columns) + 1, len(df_clean.columns))  # +1 for the target column removed
        self.assertEqual(len(X_test.columns) + 1, len(df_clean.columns))
        self.assertEqual(len(y_train) + len(y_test), len(df_clean))
        
        # Check that the target column was removed from X
        self.assertNotIn('appendicitis', X_train.columns)
        self.assertNotIn('appendicitis', X_test.columns)
    
    def test_save_processed_data(self):
        """Test saving processed data."""
        # Handle missing values first
        df_clean = handle_missing_values(self.test_data, strategy='median')
        
        # Split the data
        X_train, X_test, y_train, y_test = split_data(df_clean, 'appendicitis')
        
        # Save the data
        save_processed_data(X_train, X_test, y_train, y_test, self.output_dir)
        
        # Check that the files were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'X_train.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'X_test.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'y_train.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'y_test.csv')))
    
    def test_advanced_outlier_detection(self):
        """Test advanced outlier detection."""
        # Create test data with some outliers
        data = pd.DataFrame({
            'feature1': [1, 2, 100, 3, 4, 5, 200],  # 100 and 200 are outliers
            'feature2': [5, 6, 7, 8, 9, 10, 100],   # 100 is an outlier
            'feature3': [10, 20, 30, 40, 50, 60, 70]
        })
        
        # Run outlier detection - function returns (DataFrame, dict)
        df_with_outliers, outlier_stats = advanced_outlier_detection(data)
        
        # Check the output structure
        self.assertIsInstance(outlier_stats, dict)
        self.assertIn('robust_pca_count', outlier_stats)
        self.assertIn('lof_count', outlier_stats)
        self.assertIn('total_count', outlier_stats)
        self.assertIn('outlier_indices', outlier_stats)
        
        # Check that outliers were found
        self.assertTrue(outlier_stats['total_count'] > 0)
        self.assertGreater(len(outlier_stats['outlier_indices']), 0)
        
        # Test with parameters
        df_with_outliers_param, outlier_stats_param = advanced_outlier_detection(
            data, contamination=0.2, n_neighbors=3
        )
        self.assertIsInstance(outlier_stats_param, dict)
    
    def test_enhanced_memory_optimization(self):
        """Test enhanced memory optimization."""
        # Create test data with various data types
        data = pd.DataFrame({
            'age': np.array([10, 20, 30, 40, 50], dtype=np.int64),
            'age_copy': np.array([10, 20, 30, 40, 50], dtype=np.int64),  # Duplicate column
            'sparse_col': np.array([0, 0, 0, 10, 0], dtype=np.int64),    # Sparse column
            'large_int': np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64),
            'category_col': ['A', 'B', 'A', 'C', 'B']  # Categorical column
        })
        
        # Run memory optimization
        result_df, transformations = enhanced_memory_optimization(data)
        
        # Check the output
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIsInstance(transformations, dict)
        
        # Check transformations dictionary structure
        self.assertIn('__summary__', transformations)
        self.assertIn('duplicate_columns_removed', transformations['__summary__'])
        
        # Check that a duplicate column was identified
        self.assertEqual(transformations['__summary__']['duplicate_columns_removed'], 1)
        
        # Verify memory was saved
        self.assertGreater(transformations['__summary__']['memory_saved_mb'], 0)
        
        # Verify optimized dataframe has fewer columns
        self.assertEqual(len(result_df.columns), len(data.columns) - 1)

if __name__ == '__main__':
    unittest.main()
