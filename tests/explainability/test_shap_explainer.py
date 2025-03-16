"""
Unit tests for the SHAP explainer module.
Tests the SHAP explainability functionality for model predictions.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import pickle

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.explainability.shap_explainer import ShapExplainer, get_explainer_for_model

class TestShapExplainer(unittest.TestCase):
    """Test case for the SHAP explainer functionality."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a test DataFrame
        self.X_data = pd.DataFrame({
            'age': [5, 10, 15, 8, 12],
            'gender': [0, 1, 0, 1, 0],
            'duration': [24.0, 48.0, 12.0, 36.0, 72.0],
            'fever': [1, 0, 1, 1, 0],
            'white_blood_cell_count': [15.0, 8.5, 22.0, 10.5, 9.0],
            'neutrophil_percentage': [80.0, 65.0, 85.0, 75.0, 60.0],
            'c_reactive_protein': [50.0, 15.0, 100.0, 25.0, 10.0]
        })
        
        self.y_data = pd.Series([1, 0, 1, 0, 0])  # Target variable
        
        # Train a simple model for testing
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_data, self.y_data)
        
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.test_dir.name
    
    def tearDown(self):
        """Clean up after each test method."""
        self.test_dir.cleanup()
    
    def test_initialize_explainer(self):
        """Test that the SHAP explainer can be initialized."""
        explainer = ShapExplainer(self.model, self.X_data)
        
        # Check that the explainer was initialized
        self.assertIsNotNone(explainer)
        self.assertIsNotNone(explainer.explainer)
        self.assertEqual(explainer.model, self.model)
    
    def test_compute_shap_values(self):
        """Test computing SHAP values for predictions."""
        explainer = ShapExplainer(self.model, self.X_data)
        
        # Compute SHAP values for a single instance
        single_instance = self.X_data.iloc[:1].copy()
        shap_values = explainer.compute_shap_values(single_instance)
        
        # Check that SHAP values were computed
        self.assertIsNotNone(shap_values)
        
        # Check shapes for binary classification (depends on the explainer type)
        # For TreeExplainer with binary classification, it will output a list of 2 arrays
        # or a 3D array with shape (samples, features, 2)
        if isinstance(shap_values, list):
            self.assertEqual(len(shap_values), 2)  # One for each class
            self.assertEqual(shap_values[0].shape[0], 1)  # One instance
            self.assertEqual(shap_values[0].shape[1], len(self.X_data.columns))  # Number of features
        else:
            # Check that the shape is (samples, features) or (samples, features, classes)
            self.assertIn(len(shap_values.shape), [2, 3])
            if len(shap_values.shape) == 2:
                self.assertEqual(shap_values.shape[0], 1)  # One instance
                self.assertEqual(shap_values.shape[1], len(self.X_data.columns))  # Number of features
            else:  # 3D case
                self.assertEqual(shap_values.shape[0], 1)  # One instance
                self.assertEqual(shap_values.shape[1], len(self.X_data.columns))  # Number of features
                self.assertIn(shap_values.shape[2], [1, 2])  # Number of classes
    
    def test_get_transformed_data(self):
        """Test getting transformed data from the explainer."""
        explainer = ShapExplainer(self.model, self.X_data)
        
        # Get transformed data
        transformed_data = explainer.get_transformed_data(self.X_data)
        
        # Check that the data was returned
        self.assertIsNotNone(transformed_data)
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertEqual(transformed_data.shape, self.X_data.shape)
    
    def test_plot_summary(self):
        """Test plotting SHAP summary visualization."""
        explainer = ShapExplainer(self.model, self.X_data)
        
        # Plot summary
        output_path = os.path.join(self.output_dir, 'shap_summary.png')
        fig = explainer.plot_summary(self.X_data, output_path=output_path)
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the output file was created
        self.assertTrue(os.path.exists(output_path))
    
    def test_plot_feature_importance(self):
        """Test plotting feature importance visualization."""
        explainer = ShapExplainer(self.model, self.X_data)
        
        # Compute SHAP values first
        shap_values = explainer.compute_shap_values(self.X_data)
        
        # Plot feature importance
        output_path = os.path.join(self.output_dir, 'feature_importance.png')
        fig = explainer.plot_feature_importance(
            shap_values, 
            self.X_data.columns, 
            output_path=output_path
        )
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that the output file was created
        self.assertTrue(os.path.exists(output_path))
    
    def test_factory_function(self):
        """Test the factory function that creates explainers."""
        explainer = get_explainer_for_model(self.model, self.X_data)
        
        # Check that the explainer was created
        self.assertIsNotNone(explainer)
        self.assertIsInstance(explainer, ShapExplainer)
        self.assertEqual(explainer.model, self.model)

if __name__ == '__main__':
    unittest.main()
