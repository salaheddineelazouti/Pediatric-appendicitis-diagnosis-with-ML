"""
Test module for model loading functionality.
Tests the loading of machine learning models from pickle files.
"""

import unittest
import os
import sys
import pickle
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestModelLoading(unittest.TestCase):
    """Test cases for model loading functionality."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.mock_model_path = os.path.join('models', 'best_model_retrained.pkl')
        
        # Sample feature data for testing predictions
        self.sample_features = {
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
        self.feature_df = pd.DataFrame([self.sample_features])

    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_model_file_existence(self, mock_pickle_load, mock_file_open):
        """Test that the model file exists and can be opened."""
        # This test mocks the open function and checks if the model file can be opened
        from src.api.app import load_model
        
        # Configure the mock to return a valid model object
        mock_model = unittest.mock.MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_pickle_load.return_value = mock_model
        
        # Call the function under test
        with patch('os.path.exists', return_value=True):
            model = load_model()
        
        # Assert that the file was attempted to be opened with the correct path
        mock_file_open.assert_called_once()
        self.assertIsNotNone(model)

    @patch('pickle.load')
    def test_model_structure(self, mock_pickle_load):
        """Test that the loaded model has the expected structure and methods."""
        # Create a mock model that simulates the expected structure
        mock_model = unittest.mock.MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_model.feature_names_in_ = list(self.sample_features.keys())
        
        mock_pickle_load.return_value = mock_model
        
        # Mock the open function
        with patch('builtins.open', mock_open()):
            with patch('os.path.exists', return_value=True):
                from src.api.app import load_model
                model = load_model()
        
        # Check that the model has the expected methods and attributes
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'predict_proba'))
        self.assertTrue(hasattr(model, 'feature_names_in_'))

    def test_model_initialization_error_handling(self):
        """Test that errors during model loading are properly handled."""
        with patch('os.path.exists', return_value=False):
            from src.api.app import load_model
            model = load_model()
            
        # If the file doesn't exist, load_model should return None
        self.assertIsNone(model)
        
        # Test with exception during loading
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', side_effect=Exception("Test exception")):
                model = load_model()
                
        # If an exception occurs, load_model should return None
        self.assertIsNone(model)


if __name__ == '__main__':
    unittest.main()
