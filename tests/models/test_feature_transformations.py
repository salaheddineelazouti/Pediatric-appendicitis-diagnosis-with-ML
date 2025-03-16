"""
Test module for feature transformation functionality.
Tests the custom feature transformations used in the appendicitis diagnosis model.
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

# Import the custom transformers
from src.api.app import ClinicalFeatureTransformer


class TestFeatureTransformations(unittest.TestCase):
    """Test cases for feature transformation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample data for transformation testing
        self.sample_data = pd.DataFrame({
            'age': [5.0, 10.0, 15.0],
            'gender': [0, 1, 0],  # 0=Female, 1=Male
            'duration_of_pain': [6.0, 24.0, 48.0],
            'migration': [0, 1, 1],
            'anorexia': [1, 1, 0],
            'nausea': [1, 0, 1],
            'vomiting': [0, 1, 1],
            'right_lower_quadrant_pain': [1, 1, 1],
            'fever': [0, 1, 1],
            'rebound_tenderness': [0, 1, 1],
            'white_blood_cell_count': [8.0, 15.0, 20.0],
            'neutrophil_percentage': [60.0, 85.0, 90.0],
            'c_reactive_protein': [10.0, 40.0, 100.0],
            'alvarado_score': [5, 8, 9]
        })
        
        # Define clinical and lab features for transformer
        self.clinical_features = [
            'duration_of_pain', 'migration', 'anorexia', 'nausea', 'vomiting',
            'right_lower_quadrant_pain', 'fever', 'rebound_tenderness'
        ]
        self.lab_features = [
            'white_blood_cell_count', 'neutrophil_percentage', 'c_reactive_protein'
        ]
    
    def test_clinical_feature_transformer_initialization(self):
        """Test initialization of the ClinicalFeatureTransformer."""
        transformer = ClinicalFeatureTransformer(
            clinical_features=self.clinical_features,
            lab_features=self.lab_features
        )
        
        # Check that the transformer was initialized with the correct features
        self.assertEqual(transformer.clinical_features, self.clinical_features)
        self.assertEqual(transformer.lab_features, self.lab_features)
    
    def test_symptom_count_calculation(self):
        """Test calculation of symptom count."""
        # Initialize transformer
        transformer = ClinicalFeatureTransformer(
            clinical_features=self.clinical_features,
            lab_features=self.lab_features
        )
        
        # Transform the data
        transformed_data = transformer.transform(self.sample_data)
        
        # Check that symptom_count was added
        self.assertIn('symptom_count', transformed_data.columns)
        
        # Check symptom count calculation for each sample
        # Sample 1: migration(0), anorexia(1), nausea(1), vomiting(0), rlq_pain(1), fever(0), rebound(0) = 3
        # Sample 2: migration(1), anorexia(1), nausea(0), vomiting(1), rlq_pain(1), fever(1), rebound(1) = 6
        # Sample 3: migration(1), anorexia(0), nausea(1), vomiting(1), rlq_pain(1), fever(1), rebound(1) = 6
        expected_counts = [3, 6, 6]
        self.assertListEqual(transformed_data['symptom_count'].tolist(), expected_counts)
    
    def test_absolute_neutrophils_calculation(self):
        """Test calculation of absolute neutrophil count."""
        # Initialize transformer
        transformer = ClinicalFeatureTransformer(
            clinical_features=self.clinical_features,
            lab_features=self.lab_features
        )
        
        # Transform the data
        transformed_data = transformer.transform(self.sample_data)
        
        # Check that absolute_neutrophils was added
        self.assertIn('absolute_neutrophils', transformed_data.columns)
        
        # Check absolute neutrophil calculation for each sample
        # Sample 1: 8.0 * (60.0 / 100) = 4.8
        # Sample 2: 15.0 * (85.0 / 100) = 12.75
        # Sample 3: 20.0 * (90.0 / 100) = 18.0
        expected_values = [4.8, 12.75, 18.0]
        np.testing.assert_almost_equal(
            transformed_data['absolute_neutrophils'].tolist(),
            expected_values
        )
    
    def test_classic_triad_feature(self):
        """Test creation of classic appendicitis triad feature."""
        # Initialize transformer
        transformer = ClinicalFeatureTransformer(
            clinical_features=self.clinical_features,
            lab_features=self.lab_features
        )
        
        # Transform the data
        transformed_data = transformer.transform(self.sample_data)
        
        # Check that classic_triad was added
        self.assertIn('classic_triad', transformed_data.columns)
        
        # Check classic triad calculation for each sample
        # Sample 1: migration(0), rlq_pain(1), rebound(0) = 0 (not all present)
        # Sample 2: migration(1), rlq_pain(1), rebound(1) = 1 (all present)
        # Sample 3: migration(1), rlq_pain(1), rebound(1) = 1 (all present)
        expected_values = [0, 1, 1]
        self.assertListEqual(transformed_data['classic_triad'].tolist(), expected_values)
    
    def test_lab_composite_score(self):
        """Test creation of laboratory composite score."""
        # Initialize transformer
        transformer = ClinicalFeatureTransformer(
            clinical_features=self.clinical_features,
            lab_features=self.lab_features
        )
        
        # Transform the data
        transformed_data = transformer.transform(self.sample_data)
        
        # Check that lab_composite_score was added
        self.assertIn('lab_composite_score', transformed_data.columns)
        
        # Sample 1: wbc_norm = 8.0/20.0 = 0.4, neutro_norm = 60.0/100.0 = 0.6, crp_norm = 10.0/150.0 = 0.067
        #           lab_score = 0.4*0.4 + 0.6*0.3 + 0.067*0.3 = 0.16 + 0.18 + 0.02 = 0.36
        # Sample 2: wbc_norm = 15.0/20.0 = 0.75, neutro_norm = 85.0/100.0 = 0.85, crp_norm = 40.0/150.0 = 0.267
        #           lab_score = 0.75*0.4 + 0.85*0.3 + 0.267*0.3 = 0.3 + 0.255 + 0.08 = 0.635
        # Sample 3: wbc_norm = 1.0 (capped at 1.0), neutro_norm = 0.9, crp_norm = 0.667
        #           lab_score = 1.0*0.4 + 0.9*0.3 + 0.667*0.3 = 0.4 + 0.27 + 0.2 = 0.87
        expected_scores = [0.36, 0.635, 0.87]
        
        # Use almost equal for floating point comparison
        for i, expected in enumerate(expected_scores):
            self.assertAlmostEqual(
                transformed_data['lab_composite_score'].iloc[i],
                expected,
                places=2
            )
    
    def test_alvarado_risk_categorization(self):
        """Test categorization of Alvarado score into risk categories."""
        # Initialize transformer
        transformer = ClinicalFeatureTransformer(
            clinical_features=self.clinical_features,
            lab_features=self.lab_features
        )
        
        # Transform the data
        transformed_data = transformer.transform(self.sample_data)
        
        # Check that alvarado_risk was added
        self.assertIn('alvarado_risk', transformed_data.columns)
        
        # Check alvarado risk categorization for each sample
        # Sample 1: score 5 = Medium (1)
        # Sample 2: score 8 = High (2)
        # Sample 3: score 9 = High (2)
        expected_categories = [1, 2, 2]
        self.assertListEqual(transformed_data['alvarado_risk'].tolist(), expected_categories)
    
    def test_feature_preservation(self):
        """Test that original features are preserved after transformation."""
        # Initialize transformer
        transformer = ClinicalFeatureTransformer(
            clinical_features=self.clinical_features,
            lab_features=self.lab_features
        )
        
        # Transform the data
        transformed_data = transformer.transform(self.sample_data)
        
        # Check that all original columns are still present
        for column in self.sample_data.columns:
            self.assertIn(column, transformed_data.columns)
        
        # Check that values in original columns are unchanged
        for column in self.sample_data.columns:
            pd.testing.assert_series_equal(
                transformed_data[column],
                self.sample_data[column]
            )


if __name__ == '__main__':
    unittest.main()
