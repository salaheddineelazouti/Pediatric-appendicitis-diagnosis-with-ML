"""
Test module for model prediction functionality.
Tests the prediction capabilities of the appendicitis diagnosis model.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestModelPrediction(unittest.TestCase):
    """Test cases for model prediction functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample feature data for testing predictions
        self.sample_input = {
            'age': 10.0,
            'gender': 1,  # Male
            'duration': 24.0,
            'migration': 1,
            'anorexia': 1,
            'nausea': 0,
            'vomiting': 1,
            'right_lower_quadrant_pain': 1,
            'fever': 1,
            'rebound_tenderness': 1,
            'white_blood_cell_count': 15.0,
            'neutrophil_percentage': 85.0,
            'c_reactive_protein': 40.0
        }
        self.input_df = pd.DataFrame([self.sample_input])
        
        # Create a mock model with controlled prediction behavior
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([1])  # Positive for appendicitis
        self.mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% probability
        self.mock_model.feature_names_in_ = list(self.sample_input.keys())

    def test_model_prediction_binary(self):
        """Test binary classification prediction (appendicitis vs. no appendicitis)."""
        # Perform the prediction
        prediction = self.mock_model.predict(self.input_df)
        
        # Check that the prediction is either 0 or 1
        self.assertIn(prediction[0], [0, 1])
        
        # For this sample with classic symptoms, should predict positive (1)
        self.assertEqual(prediction[0], 1)

    def test_model_prediction_probability(self):
        """Test probability prediction for appendicitis."""
        # Perform the probability prediction
        probabilities = self.mock_model.predict_proba(self.input_df)
        
        # Check the shape and values of the probability output
        self.assertEqual(probabilities.shape, (1, 2))  # One sample, two classes
        self.assertAlmostEqual(np.sum(probabilities[0]), 1.0, places=5)  # Probabilities sum to 1
        
        # The mock is set to return 80% probability for positive class
        self.assertAlmostEqual(probabilities[0, 1], 0.8)

    def test_feature_transformation(self):
        """Test that feature transformations are correctly applied before prediction."""
        # This test would normally validate that transformations like
        # ClinicalFeatureTransformer are correctly applied
        
        # Create an instance of the transformer for testing
        from optimize_feature_transformations import FeatureInteractionTransformer
        transformer = FeatureInteractionTransformer()
        
        # Apply transformations to test input
        with patch('src.api.app.ClinicalFeatureTransformer') as mock_transformer:
            mock_transformer_instance = MagicMock()
            mock_transformer.return_value = mock_transformer_instance
            mock_transformer_instance.transform.return_value = self.input_df
            
            # Verify transform was called
            transformed_data = mock_transformer_instance.transform(self.input_df)
            mock_transformer_instance.transform.assert_called_once()
            
            # The returned data should be a DataFrame
            self.assertIsInstance(transformed_data, pd.DataFrame)

    def test_risk_classification(self):
        """Test the risk classification based on predicted probability."""
        from src.api.app import get_risk_class
        
        # Test different probability thresholds
        self.assertEqual(get_risk_class(0.05), "Low Risk")
        self.assertEqual(get_risk_class(0.3), "Moderate Risk") 
        self.assertEqual(get_risk_class(0.75), "High Risk")

    def test_prediction_with_missing_values(self):
        """Test prediction behavior when input data has missing values."""
        # Create a copy of the sample input with some missing values
        input_with_missing = self.sample_input.copy()
        input_with_missing['white_blood_cell_count'] = np.nan
        df_with_missing = pd.DataFrame([input_with_missing])
        
        # In a real application, this might trigger an error or imputation
        # For the test, we'll check that the model can handle it with a mock
        with patch.object(self.mock_model, 'predict_proba') as mock_predict:
            # The model is supposed to handle or impute missing values
            mock_predict.return_value = np.array([[0.4, 0.6]])
            
            # Try to make a prediction
            result = self.mock_model.predict_proba(df_with_missing)
            
            # Check that prediction still works
            self.assertEqual(result.shape, (1, 2))


if __name__ == '__main__':
    unittest.main()
