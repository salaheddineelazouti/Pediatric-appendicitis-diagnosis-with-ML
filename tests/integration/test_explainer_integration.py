"""
Integration tests for the SHAP explainer component of the pediatric appendicitis diagnosis system.
Tests the integration between the model predictions and the SHAP explainability component.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestExplainerIntegration(unittest.TestCase):
    """Integration test case for the SHAP explainer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample input data for testing
        self.sample_input = pd.DataFrame({
            'age': [10.0],
            'gender': [1],  # Male
            'duration': [24.0],
            'migration': [1],
            'anorexia': [1],
            'nausea': [0],
            'vomiting': [1],
            'right_lower_quadrant_pain': [1],
            'fever': [1],
            'rebound_tenderness': [1],
            'white_blood_cell_count': [15.0],
            'neutrophil_percentage': [85.0],
            'c_reactive_protein': [40.0]
        })
        
        # Create a mock for the processed form data
        self.form_data = {
            'age': 10.0,
            'gender': 'male',
            'duration': 24.0,
            'migration': True,
            'anorexia': True,
            'nausea': False,
            'vomiting': True,
            'right_lower_quadrant_pain': True,
            'fever': True,
            'rebound_tenderness': True,
            'white_blood_cell_count': 15.0,
            'neutrophil_percentage': 85.0,
            'c_reactive_protein': 40.0
        }
    
    @patch('src.api.app.model')
    @patch('src.api.app.explainer')
    def test_diagnose_endpoint_with_explainer(self, mock_explainer, mock_model):
        """Test that the diagnose endpoint correctly uses the explainer."""
        # Configure the mock model
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% probability of appendicitis
        
        # Configure the mock explainer
        mock_explainer.explain.return_value = {
            'shap_values': np.array([[
                0.05, 0.02, 0.1, 0.15, 0.08, 0.0, 0.03, 0.2, 0.12, 0.18, 0.3, 0.25, 0.4
            ]]),
            'base_value': 0.2,
            'feature_names': list(self.sample_input.columns)
        }
        
        # Import app for testing (with patched dependencies)
        from src.api.app import app
        
        # Create a test client
        client = app.test_client()
        
        # Prepare form data for POST request
        form_data = {
            'age': '10.0',
            'gender': 'male',
            'duration': '24.0',
            'migration': 'on',
            'anorexia': 'on',
            'vomiting': 'on',
            'right_lower_quadrant_pain': 'on',
            'fever': 'on',
            'rebound_tenderness': 'on',
            'white_blood_cell_count': '15.0',
            'neutrophil_percentage': '85.0',
            'c_reactive_protein': '40.0'
        }
        
        # Patch create_waterfall_chart to avoid actual image creation during test
        with patch('src.api.app.create_waterfall_chart', return_value='mock_image_data'):
            # Send POST request to diagnose endpoint
            response = client.post('/diagnose', data=form_data)
            
            # Check that the response is OK
            self.assertEqual(response.status_code, 200)
            
            # Check that explainer.explain was called
            mock_explainer.explain.assert_called_once()
            
            # Check response content for SHAP visualization reference
            self.assertIn(b'Prediction Explanation', response.data)
    
    @patch('src.api.app.ShapExplainer')
    def test_explainer_initialization(self, mock_explainer_class):
        """Test that the explainer is properly initialized with sample data."""
        # Configure mock
        mock_explainer_instance = MagicMock()
        mock_explainer_class.return_value = mock_explainer_instance
        
        # Import initialize_explainer for testing
        from src.api.app import initialize_explainer, model
        
        # Patch model to return predictable values
        model_mock = MagicMock()
        model_mock.feature_names_in_ = list(self.sample_input.columns)
        
        # Patch os.makedirs to avoid directory creation
        with patch('os.makedirs'):
            with patch('src.api.app.model', model_mock):
                # Call the function being tested
                explainer = initialize_explainer()
                
                # Check that the explainer was created
                mock_explainer_class.assert_called_once()
                self.assertEqual(explainer, mock_explainer_instance)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('io.BytesIO')
    @patch('base64.b64encode')
    def test_waterfall_chart_creation(self, mock_b64encode, mock_bytesio, mock_close, mock_savefig):
        """Test the creation of the waterfall chart visualization."""
        # Configure mocks
        mock_bytesio_instance = MagicMock()
        mock_bytesio.return_value = mock_bytesio_instance
        mock_b64encode.return_value = b'test_base64_image_data'
        
        from src.api.app import create_waterfall_chart
        
        # Test data
        base_value = 0.2
        shap_values = np.array([0.05, 0.02, 0.1, 0.15, 0.08, 0.0, 0.03, 0.2, 0.12, 0.18, 0.3, 0.25, 0.4])
        feature_names = list(self.sample_input.columns)
        final_prediction = 0.8
        
        # Call the function being tested
        result = create_waterfall_chart(base_value, shap_values, feature_names, final_prediction)
        
        # Check that the chart was saved as an image
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        # Check that the image was encoded to base64
        mock_b64encode.assert_called_once()
        
        # Check the return value
        self.assertEqual(result, 'test_base64_image_data')
    
    @patch('src.ai_assistant.gemini_integration.explain_prediction_features')
    def test_ai_explanation_integration(self, mock_explain_features):
        """Test the integration of AI explanations with the SHAP visualizations."""
        # Configure mock
        mock_explain_features.return_value = "This is an AI explanation of the features."
        
        # Import app for testing
        from src.api.app import app
        
        # Create a test client
        client = app.test_client()
        
        # Prepare test data
        test_data = {
            'features': [
                {'name': 'WBC Count', 'value': 0.35},
                {'name': 'Rebound Tenderness', 'value': 0.25},
                {'name': 'Right Lower Quadrant Pain', 'value': 0.2}
            ]
        }
        
        # Send POST request to the explain features API endpoint
        with app.test_request_context():
            from flask import session
            with client.session_transaction() as sess:
                sess['diagnosis_result'] = {'prediction': 0.8}
            
            response = client.post('/api/explain-features', 
                                   json=test_data,
                                   content_type='application/json')
            
            # Check that the response is OK
            self.assertEqual(response.status_code, 200)
            
            # Check response content
            response_data = response.get_json()
            self.assertIn('explanation', response_data)
            self.assertEqual(response_data['explanation'], "This is an AI explanation of the features.")
            
            # Check that the explain_prediction_features function was called
            mock_explain_features.assert_called_once()


if __name__ == '__main__':
    unittest.main()
