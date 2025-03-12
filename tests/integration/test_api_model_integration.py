"""
Integration tests for testing the interaction between the API and the model.
This ensures that the API can correctly use the model for predictions.
"""

import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the Flask app and create a test client
from src.api.app import app, model

class TestApiModelIntegration(unittest.TestCase):
    """Test case for the integration between API and model."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_diagnose_get_page(self):
        """Test that the diagnose page can be loaded."""
        response = self.app.get('/diagnose')
        self.assertEqual(response.status_code, 200, "Should return status code 200")
        # Check that the page contains the form elements
        self.assertIn(b'Duration of Pain', response.data)
        self.assertIn(b'White Blood Cell Count', response.data)
    
    @patch('src.api.app.model')
    def test_diagnose_post_with_model(self, mock_model):
        """Test POST to diagnose endpoint with a mocked model."""
        # Setup the mock model
        mock_prediction = MagicMock()
        mock_prediction.return_value = [[0.2, 0.8]]  # 80% probability of appendicitis
        mock_model.predict_proba = mock_prediction
        
        # Create test data
        data = {
            'age': '10',
            'gender': 'male',
            'duration': '24',
            'migration': '1',
            'anorexia': '1',
            'nausea': '1',
            'vomiting': '0',
            'right_lower_quadrant_pain': '1',
            'fever': '1',
            'rebound_tenderness': '1',
            'white_blood_cell_count': '15',
            'neutrophil_percentage': '80',
            'c_reactive_protein': '50',
            'pediatric_appendicitis_score': '8',
            'alvarado_score': '7'
        }
        
        # Test with redirection disabled to get the first response
        response = self.app.post('/diagnose', data=data)
        
        # Check for redirect status code
        self.assertIn(response.status_code, [200, 302], "Should return status code 200 or 302 (redirect)")
        
        # Check that the mock was called
        mock_model.predict_proba.assert_called()
    
    def test_diagnose_post_without_model(self):
        """Test POST to diagnose endpoint when the model is not available."""
        # Use patching instead of direct attribute access
        with patch('src.api.app.model', None):
            # Create test data with missing some required fields
            data = {
                'age': '10',
                'gender': 'male',
                'duration': '24',
                'migration': '1',
                'anorexia': '1',
                'white_blood_cell_count': '15',
                'neutrophil_percentage': '80',
                'c_reactive_protein': '50'
            }
            
            # Submit the form
            response = self.app.post('/diagnose', data=data)
            
            # Check that the response is successful even without a model (demo mode)
            self.assertIn(response.status_code, [200, 302], "Should return status code 200 or 302 (redirect)")
    
    def test_error_handling(self):
        """Test that API properly handles errors."""
        # Test with invalid data (string in numeric field)
        data = {
            'age': 'invalid',
            'gender': 'male',
            # Other fields omitted
        }
        
        # Submit the form
        response = self.app.post('/diagnose', data=data)
        
        # Check that the error is handled
        self.assertEqual(response.status_code, 200, "Should return status code 200 even on error")

if __name__ == '__main__':
    unittest.main()
