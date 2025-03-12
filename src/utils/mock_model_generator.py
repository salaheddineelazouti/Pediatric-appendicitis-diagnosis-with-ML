"""
Utility script to generate mock model and explainer files for development purposes.
This enables testing of the web application without requiring a fully trained model.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap

def generate_mock_model(output_path, random_state=42):
    """
    Generate a simple mock model for development/testing purposes.
    
    Args:
        output_path: Path to save the model
        random_state: Random state for reproducibility
    """
    print(f"Generating mock model at: {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a simple random forest model
    model = RandomForestClassifier(n_estimators=10, random_state=random_state)
    
    # Generate some random data to fit the model
    X = np.random.rand(100, 15)  # 15 features
    y = np.random.randint(0, 2, 100)  # Binary classification
    
    # Add feature names that match expected inputs
    feature_names = [
        'age', 'gender', 'duration', 'migration', 'anorexia', 
        'nausea', 'vomiting', 'right_lower_quadrant_pain', 'fever',
        'rebound_tenderness', 'white_blood_cell_count', 'neutrophil_percentage',
        'c_reactive_protein', 'pediatric_appendicitis_score', 'alvarado_score'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Fit the model
    model.fit(X_df, y)
    
    # Save the model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Mock model saved to {output_path}")
    return model, X_df

def generate_mock_explainer(model, X, output_path):
    """
    Generate a mock SHAP explainer for the model.
    
    Args:
        model: Trained model
        X: Sample data
        output_path: Path to save the explainer
    """
    print(f"Generating mock explainer at: {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Save the explainer
    with open(output_path, 'wb') as f:
        pickle.dump(explainer, f)
    
    print(f"Mock explainer saved to {output_path}")
    return explainer

if __name__ == "__main__":
    # Set paths
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    model_path = os.path.join(models_dir, 'best_model.pkl')
    explainer_path = os.path.join(models_dir, 'explainer.pkl')
    
    # Generate mock model and explainer
    model, X = generate_mock_model(model_path)
    generate_mock_explainer(model, X, explainer_path)
    
    print("Mock model and explainer generated successfully for development testing.")
