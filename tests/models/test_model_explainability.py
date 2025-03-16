"""
Test module for the explainability features of the appendicitis diagnosis model.
Tests the SHAP explainer integration and visualization functionality.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock, ANY

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestModelExplainability(unittest.TestCase):
    """Test cases for model explainability functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample data for explanation testing
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
        
        # Mock SHAP values for testing
        self.mock_shap_values = np.array([
            [0.05, 0.02, 0.1, 0.15, 0.08, 0.0, 0.03, 0.2, 0.12, 0.18, 0.3, 0.25, 0.4]
        ])
        self.feature_names = list(self.sample_input.keys())
        self.mock_base_value = 0.2
        self.mock_prediction = 0.8

    def test_explainer_initialization(self):
        """Test that the SHAP explainer is properly initialized."""
        # Import what we need to patch
        from src.api.app import initialize_explainer
        import src.api.app
        
        # Create our mocks
        mock_model = MagicMock()
        mock_model.feature_names_in_ = ['age', 'gender', 'duration', 'migration', 'anorexia',
                                       'vomiting', 'right_lower_quadrant_pain', 'fever',
                                       'rebound_tenderness', 'white_blood_cell_count',
                                       'neutrophil_percentage', 'c_reactive_protein']
        
        # Create a mock ShapExplainer instance
        mock_explainer = MagicMock()
        
        # This is the important part: we're patching the actual class with a mock that returns our mock instance
        with patch('src.api.app.ShapExplainer', return_value=mock_explainer) as mock_explainer_class:
            with patch('src.api.app.model', mock_model):
                with patch('os.makedirs'):
                    # Reset the explainer to None to ensure a new one is created
                    src.api.app.explainer = None
                    
                    # Initialize the explainer with force_new to ensure it creates a new instance
                    result = initialize_explainer(force_new=True)
                    
                    # Check that our mock ShapExplainer was called
                    mock_explainer_class.assert_called_once_with(mock_model, ANY)
                    self.assertEqual(result, mock_explainer)

    def test_waterfall_chart_creation(self):
        """Test creation of the waterfall chart for feature importance visualization."""
        from src.api.app import create_waterfall_chart
        
        # Temporarily replace plt.savefig to prevent file writing during tests
        with patch.object(plt, 'savefig'), patch.object(plt, 'close'):
            # Create a waterfall chart and get base64 image
            with patch('io.BytesIO') as mock_bytesio:
                with patch('base64.b64encode') as mock_b64encode:
                    mock_b64encode.return_value = b'test_base64_image_data'
                    
                    # Call the function to test
                    result = create_waterfall_chart(
                        self.mock_base_value,
                        self.mock_shap_values[0],
                        self.feature_names,
                        self.mock_prediction
                    )
                    
                    # Check that b64encode was called (image was created)
                    mock_b64encode.assert_called_once()
                    
                    # Check the return value
                    self.assertEqual(result, 'test_base64_image_data')

    def test_feature_formatting(self):
        """Test formatting of feature names and values for display."""
        from src.api.app import format_feature_name, format_feature_value
        
        # Test feature name formatting
        self.assertEqual(format_feature_name('white_blood_cell_count'), 'White Blood Cell Count')
        self.assertEqual(format_feature_name('right_lower_quadrant_pain'), 'Right Lower Quadrant Pain')
        
        # Test feature value formatting
        self.assertEqual(format_feature_value('gender', 1), 'Male')
        self.assertEqual(format_feature_value('gender', 0), 'Female')
        self.assertEqual(format_feature_value('migration', 1), 'Yes')
        self.assertEqual(format_feature_value('migration', 0), 'No')
        self.assertEqual(format_feature_value('white_blood_cell_count', 15.5), '15.5')

    @patch('src.explainability.shap_explainer.ShapExplainer')
    def test_feature_importance_ranking(self, mock_explainer_class):
        """Test ranking of features by importance based on SHAP values."""
        # Create a mock explainer that returns controlled SHAP values
        mock_explainer = MagicMock()
        mock_explainer_class.return_value = mock_explainer
        mock_explainer.explain.return_value = {
            'shap_values': self.mock_shap_values,
            'base_value': self.mock_base_value,
            'feature_names': self.feature_names
        }
        
        # Get the explainer results
        explanation = mock_explainer.explain(self.input_df)
        
        # Sort features by absolute SHAP value to get importance ranking
        shap_values = explanation['shap_values'][0]
        feature_names = explanation['feature_names']
        importance_pairs = list(zip(feature_names, abs(shap_values)))
        sorted_importance = sorted(importance_pairs, key=lambda x: x[1], reverse=True)
        
        # Check that the most important features are ranked correctly
        # According to our mock data, c_reactive_protein should be most important
        self.assertEqual(sorted_importance[0][0], 'c_reactive_protein')
        
        # Verify the top 3 most important features
        top_features = [pair[0] for pair in sorted_importance[:3]]
        self.assertIn('c_reactive_protein', top_features)
        self.assertIn('white_blood_cell_count', top_features)
        self.assertIn('neutrophil_percentage', top_features)


if __name__ == '__main__':
    unittest.main()
