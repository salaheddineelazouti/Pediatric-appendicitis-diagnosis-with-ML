"""
End-to-end integration tests for the Pediatric Appendicitis Diagnosis application.
Tests the complete workflow from data input through model prediction to result display.
"""

import os
import sys
import unittest
from unittest.mock import patch
import pandas as pd
import pickle
import tempfile
import numpy as np

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.api.app import app as flask_app
from src.data_processing.preprocess import handle_missing_values, optimize_pipeline_for_production
from src.explainability.shap_explainer import ShapExplainer

class TestEndToEnd(unittest.TestCase):
    """Test case for end-to-end functionality."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Set up Flask test client
        self.app = flask_app
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF protection for testing
        self.app.config['SECRET_KEY'] = 'test-key'
        
        # Create a test client
        self.client = self.app.test_client()
        
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.test_dir.name
        
        # Create a simple test model
        from sklearn.ensemble import RandomForestClassifier
        self.test_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create test data
        self.X_train = pd.DataFrame({
            'age': [5, 10, 15, 8, 12],
            'gender': [0, 1, 0, 1, 0],
            'duration': [24.0, 48.0, 12.0, 36.0, 72.0],
            'migration': [1, 0, 1, 0, 1],
            'anorexia': [1, 0, 1, 1, 0],
            'nausea': [1, 1, 0, 1, 0],
            'vomiting': [0, 1, 0, 1, 0],
            'right_lower_quadrant_pain': [1, 0, 1, 1, 0],
            'fever': [1, 0, 1, 0, 0],
            'rebound_tenderness': [1, 0, 1, 0, 0],
            'white_blood_cell_count': [15.0, 8.5, 22.0, 10.5, 9.0],
            'neutrophil_percentage': [80.0, 65.0, 85.0, 75.0, 60.0],
            'c_reactive_protein': [50.0, 15.0, 100.0, 25.0, 10.0],
            'pediatric_appendicitis_score': [8.0, 3.0, 9.0, 5.0, 2.0],
            'alvarado_score': [8.0, 4.0, 9.0, 6.0, 3.0]
        })
        
        self.y_train = pd.Series([1, 0, 1, 0, 0])  # Target variable
        
        # Train the test model
        self.test_model.fit(self.X_train, self.y_train)
        
        # Save the test model to a temporary file
        self.model_path = os.path.join(self.output_dir, 'test_model.pkl')
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.test_model, f)
        
        # Create a test patient for prediction
        self.test_patient = {
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
    
    def tearDown(self):
        """Clean up after each test method."""
        self.test_dir.cleanup()
    
    def test_data_preprocessing_to_model_prediction(self):
        """Test the complete pipeline from data preprocessing to model prediction."""
        # Create test input data
        input_df = pd.DataFrame([self.test_patient])
        
        # 1. Apply preprocessing
        processed_data, outlier_warning, stats_dict = optimize_pipeline_for_production(
            input_df,
            model_path=self.model_path,
            memory_optimize=True,
            outlier_detection=True
        )
        
        # Check that preprocessing was successful
        self.assertIsNotNone(processed_data)
        self.assertIsInstance(processed_data, pd.DataFrame)
        
        # 2. Make prediction with the model
        prediction = self.test_model.predict(processed_data)[0]
        probabilities = self.test_model.predict_proba(processed_data)[0]
        
        # Check that the prediction is valid
        self.assertIn(prediction, [0, 1])
        self.assertEqual(len(probabilities), 2)
        self.assertTrue(0 <= probabilities[0] <= 1)
        self.assertTrue(0 <= probabilities[1] <= 1)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
    
    def test_api_with_form_submission_simulation(self):
        """Test the complete API workflow with form submission."""
        # Create a patch for the model loading function
        with patch('src.api.app.load_model', return_value=self.test_model):
            # Initialize the explainer
            explainer = ShapExplainer(self.test_model, self.X_train)
            
            # Create a patch for the explainer initialization
            with patch('src.api.app.initialize_explainer', return_value=explainer):
                # Prepare form data from test patient
                form_data = {
                    'age': str(self.test_patient['age']),
                    'gender': str(self.test_patient['gender']),
                    'duration': str(self.test_patient['duration']),
                    'migration': 'on' if self.test_patient['migration'] == 1 else '',
                    'anorexia': 'on' if self.test_patient['anorexia'] == 1 else '',
                    'nausea': 'on' if self.test_patient['nausea'] == 1 else '',
                    'vomiting': 'on' if self.test_patient['vomiting'] == 1 else '',
                    'right_lower_quadrant_pain': 'on' if self.test_patient['right_lower_quadrant_pain'] == 1 else '',
                    'fever': 'on' if self.test_patient['fever'] == 1 else '',
                    'rebound_tenderness': 'on' if self.test_patient['rebound_tenderness'] == 1 else '',
                    'white_blood_cell_count': str(self.test_patient['white_blood_cell_count']),
                    'neutrophil_percentage': str(self.test_patient['neutrophil_percentage']),
                    'c_reactive_protein': str(self.test_patient['c_reactive_protein']),
                    'pediatric_appendicitis_score': str(self.test_patient['pediatric_appendicitis_score']),
                    'alvarado_score': str(self.test_patient['alvarado_score'])
                }
                
                # Create a session
                with self.client.session_transaction() as session:
                    session['csrf_token'] = 'test-token'
                
                # Submit the form
                response = self.client.post('/diagnose', data=form_data, follow_redirects=True)
                
                # Check that the response is successful
                self.assertEqual(response.status_code, 200)

    def test_model_api_integration(self):
        """Test the integration between the model and API endpoints."""
        # Create patches for model loading and explainer initialization
        with patch('src.api.app.load_model', return_value=self.test_model):
            with patch('src.api.app.initialize_explainer') as mock_initialize:
                mock_initialize.return_value = ShapExplainer(self.test_model, self.X_train)
                
                # Prepare patient data
                patient_data = pd.DataFrame([self.test_patient])
                
                # Test the model's prediction directly
                model_prediction = self.test_model.predict(patient_data)[0]
                model_probabilities = self.test_model.predict_proba(patient_data)[0]
                
                # Check direct model prediction
                self.assertIn(model_prediction, [0, 1])
                
                # Now test through API
                form_data = {
                    'age': str(self.test_patient['age']),
                    'gender': str(self.test_patient['gender']),
                    'duration': str(self.test_patient['duration']),
                    'migration': 'on' if self.test_patient['migration'] == 1 else '',
                    'white_blood_cell_count': str(self.test_patient['white_blood_cell_count']),
                    'neutrophil_percentage': str(self.test_patient['neutrophil_percentage']),
                    'c_reactive_protein': str(self.test_patient['c_reactive_protein'])
                }
                
                # Create a session with the minimum required data
                with self.client.session_transaction() as session:
                    session['csrf_token'] = 'test-token'
                
                # Submit the form with minimal data
                response = self.client.post('/diagnose', data=form_data, follow_redirects=True)
                
                # Check that the API response is successful
                self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
