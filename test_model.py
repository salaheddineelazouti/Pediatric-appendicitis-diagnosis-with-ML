"""
Test script to verify that the model can be loaded and used with the expected features.
This script attempts to create a sample input and make a prediction to validate the model integration.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_test')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_model():
    """Load the trained model from the models directory."""
    model_path = os.path.join(project_root, 'models', 'best_model.pkl')
    try:
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully: {type(model).__name__}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def create_sample_input():
    """Create a sample input with the expected features."""
    # Create a dictionary with all features used during training
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
    return sample_data

def test_model_prediction():
    """Test that the model can make predictions with the sample input."""
    model = load_model()
    if not model:
        logger.error("Failed to load model")
        return False
    
    # Get feature names used during training
    logger.info(f"Feature names used during training: {model.feature_names_in_}")
    
    # Create sample input
    sample_data = create_sample_input()
    logger.info(f"Sample input: {sample_data}")
    
    # Convert to DataFrame
    sample_df = pd.DataFrame([sample_data])
    
    # Ensure all columns from training are present
    for feature in model.feature_names_in_:
        if feature not in sample_df.columns:
            logger.warning(f"Missing feature: {feature}, adding default value 0")
            sample_df[feature] = 0
    
    # Ensure columns are in the same order as during training
    sample_df = sample_df[list(model.feature_names_in_)]
    logger.info(f"Features for prediction: {sample_df.columns.tolist()}")
    
    try:
        # Make prediction
        prediction_proba = model.predict_proba(sample_df)[0][1]
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        logger.info(f"Prediction successful: {'Appendicitis' if prediction == 1 else 'No Appendicitis'}")
        logger.info(f"Probability: {prediction_proba:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting model test")
    success = test_model_prediction()
    if success:
        logger.info("Test completed successfully! The model is working as expected.")
    else:
        logger.error("Test failed! There are issues with the model integration.")
