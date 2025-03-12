"""
Unit tests for the machine learning model functionality.
Tests loading and prediction capabilities of the model.
"""

import os
import sys
import unittest
import pickle
import pandas as pd
import numpy as np

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class TestModel(unittest.TestCase):
    """Test case for the ML model functionality."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        self.model_path = os.path.join(project_root, 'models', 'best_model.pkl')
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            self.model = None
            print(f"Failed to load model: {e}")
    
    def test_model_exists(self):
        """Test that the model file exists."""
        self.assertTrue(os.path.exists(self.model_path), 
                       f"Model file does not exist at {self.model_path}")
    
    def test_model_loaded(self):
        """Test that the model can be loaded."""
        self.assertIsNotNone(self.model, "Model failed to load")
    
    def test_model_features(self):
        """Test that the model has the expected features."""
        expected_features = [
            'age', 'gender', 'duration', 'migration', 'anorexia', 'nausea', 'vomiting',
            'right_lower_quadrant_pain', 'fever', 'rebound_tenderness',
            'white_blood_cell_count', 'neutrophil_percentage', 'c_reactive_protein',
            'pediatric_appendicitis_score', 'alvarado_score'
        ]
        
        self.assertTrue(hasattr(self.model, 'feature_names_in_'), 
                       "Model does not have feature_names_in_ attribute")
        
        actual_features = self.model.feature_names_in_.tolist()
        for feature in expected_features:
            self.assertIn(feature, actual_features, 
                         f"Expected feature '{feature}' not found in model features")
    
    def test_model_prediction(self):
        """Test that the model can make predictions with valid input."""
        # Create sample data
        sample_data = {
            'age': 10.0,
            'gender': 1,  # Male
            'duration': 24.0,
            'migration': 1,
            'anorexia': 1,
            'nausea': 1,
            'vomiting': 0,
            'right_lower_quadrant_pain': 1,
            'fever': 1,
            'rebound_tenderness': 1,
            'white_blood_cell_count': 15.0,
            'neutrophil_percentage': 80.0,
            'c_reactive_protein': 50.0,
            'pediatric_appendicitis_score': 8.0,
            'alvarado_score': 7.0
        }
        
        # Convert to DataFrame
        sample_df = pd.DataFrame([sample_data])
        
        # Ensure all required columns are present
        for feature in self.model.feature_names_in_:
            if feature not in sample_df.columns:
                sample_df[feature] = 0
        
        # Reorder columns to match training data
        sample_df = sample_df[list(self.model.feature_names_in_)]
        
        # Make prediction
        prediction = self.model.predict(sample_df)
        probabilities = self.model.predict_proba(sample_df)
        
        # Check that predictions are the expected shape
        self.assertEqual(len(prediction), 1, "Prediction should be length 1")
        self.assertEqual(len(probabilities), 1, "Probabilities should be length 1")
        self.assertEqual(len(probabilities[0]), 2, "Should have probabilities for 2 classes")
        
        # Check that predictions are valid values
        self.assertIn(prediction[0], [0, 1], "Prediction should be 0 or 1")
        self.assertTrue(0 <= probabilities[0][0] <= 1, "Probability should be between 0 and 1")
        self.assertTrue(0 <= probabilities[0][1] <= 1, "Probability should be between 0 and 1")
        self.assertAlmostEqual(sum(probabilities[0]), 1.0, places=6, 
                              msg="Probabilities should sum to 1")

if __name__ == '__main__':
    unittest.main()
