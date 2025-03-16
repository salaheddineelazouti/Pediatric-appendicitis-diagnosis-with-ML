"""
Create Simple Model for Pediatric Appendicitis Diagnosis

This script creates a simple model that can be loaded by the Flask app
without pickle serialization issues.
"""

import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load training data"""
    try:
        data_path = os.path.join('DATA', 'processed', 'training_data.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        else:
            logger.error(f"Training data not found at {data_path}")
            return create_synthetic_data()
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic data if real data not available"""
    logger.info("Creating synthetic data for model training")
    np.random.seed(42)
    n_samples = 1000
    
    # Core features from the form
    features = {
        'age': np.random.uniform(1, 18, n_samples),
        'gender': np.random.choice([0, 1], n_samples),
        'duration': np.random.uniform(0, 120, n_samples),
        'migration': np.random.choice([0, 1], n_samples),
        'anorexia': np.random.choice([0, 1], n_samples),
        'nausea': np.random.choice([0, 1], n_samples),
        'vomiting': np.random.choice([0, 1], n_samples),
        'right_lower_quadrant_pain': np.random.choice([0, 1], n_samples),
        'fever': np.random.choice([0, 1], n_samples),
        'rebound_tenderness': np.random.choice([0, 1], n_samples),
        'white_blood_cell_count': np.random.uniform(4, 25, n_samples),
        'neutrophil_percentage': np.random.uniform(40, 95, n_samples),
        'c_reactive_protein': np.random.uniform(0, 200, n_samples)
    }
    
    # Generate synthetic target based on some rules
    X = pd.DataFrame(features)
    
    # Synthetic appendicitis probability based on clinical criteria:
    # - Higher neutrophil percentage and WBC indicates inflammation
    # - Rebound tenderness is a strong indicator
    # - Migration of pain is highly specific
    
    prob_appendicitis = (
        0.3 * X['rebound_tenderness'] + 
        0.2 * X['migration'] + 
        0.1 * X['right_lower_quadrant_pain'] +
        0.1 * X['fever'] +
        0.1 * (X['white_blood_cell_count'] > 12) +  
        0.1 * (X['neutrophil_percentage'] > 75) +
        0.1 * (X['c_reactive_protein'] > 50)
    )
    
    # Normalize to 0-1 range
    prob_appendicitis = prob_appendicitis / prob_appendicitis.max()
    
    # Convert to binary outcome
    y = (prob_appendicitis > 0.5).astype(int)
    
    # Add target to dataframe
    X['appendicitis'] = y
    
    return X

def create_simple_model():
    """Create a simple model without custom transformers"""
    logger.info("Creating simple model for appendicitis diagnosis")
    
    # Load or create data
    df = load_data()
    
    # Split features and target
    X = df.drop('appendicitis', axis=1) if 'appendicitis' in df.columns else df.iloc[:, :-1]
    y = df['appendicitis'] if 'appendicitis' in df.columns else df.iloc[:, -1]
    
    # Split training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define a simple pipeline with scaling and random forest
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=42
        ))
    ])
    
    # Train the model
    logger.info("Training the model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = 0.5  # Default if calculation fails
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"Model performance metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    
    # Save the model
    output_path = os.path.join('models', 'best_model_simple.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Simple model saved to {output_path}")
    
    return model

def main():
    """Main function to create the simple model"""
    logger.info("Starting simple model creation...")
    model = create_simple_model()
    
    # Copy model to the app model path for immediate use
    original_model_path = os.path.join('models', 'best_model_retrained.pkl')
    if model is not None:
        # Create a backup of the original model
        if os.path.exists(original_model_path):
            backup_path = f"{original_model_path}.backup.{int(time.time())}"
            try:
                with open(original_model_path, 'rb') as src:
                    with open(backup_path, 'wb') as dst:
                        dst.write(src.read())
                logger.info(f"Created backup of original model at {backup_path}")
            except Exception as e:
                logger.error(f"Error creating backup: {str(e)}")
        
        # Copy the simple model to the app model path
        try:
            with open(os.path.join('models', 'best_model_simple.pkl'), 'rb') as src:
                with open(original_model_path, 'wb') as dst:
                    dst.write(src.read())
            logger.info(f"Updated app model with simple model")
        except Exception as e:
            logger.error(f"Error updating app model: {str(e)}")
    
    logger.info("Simple model creation completed")
    return 0

if __name__ == "__main__":
    import time
    sys.exit(main())
