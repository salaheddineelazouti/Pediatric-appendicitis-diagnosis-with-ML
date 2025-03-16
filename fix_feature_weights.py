"""
Script to fix the feature weights in the pediatric appendicitis diagnosis model,
specifically addressing:
1. Alvarado Score negative transformation issue
2. Underweighted clinical and laboratory features
3. Increasing discrimination power between appendicitis and non-appendicitis cases
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
MODEL_PATH = os.path.join('models', 'best_model_retrained.pkl')
FIXED_MODEL_PATH = os.path.join('models', 'best_model_fixed.pkl')
TRAINING_DATA_PATH = os.path.join('DATA', 'processed', 'training_data.csv')

def load_model_and_data():
    """Load the existing model and training data"""
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded: {type(model).__name__}")
        
        # Load training data
        if os.path.exists(TRAINING_DATA_PATH):
            df = pd.read_csv(TRAINING_DATA_PATH)
            logger.info(f"Training data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Split features and target
            X = df.drop('appendicitis', axis=1) if 'appendicitis' in df.columns else df
            y = df['appendicitis'] if 'appendicitis' in df.columns else None
            
            return model, X, y
        else:
            logger.error(f"Training data file not found: {TRAINING_DATA_PATH}")
            return model, None, None
    except Exception as e:
        logger.error(f"Error loading model or data: {str(e)}")
        return None, None, None

def create_feature_weighted_pipeline(X, y):
    """Create a new pipeline with proper feature weighting"""
    logger.info("Creating feature-weighted pipeline...")
    
    # Group features for different treatment
    clinical_features = ['migration', 'anorexia', 'nausea', 'vomiting', 
                         'right_lower_quadrant_pain', 'fever', 'rebound_tenderness']
    
    lab_features = ['white_blood_cell_count', 'neutrophil_percentage', 'c_reactive_protein']
    
    score_features = ['pediatric_appendicitis_score', 'alvarado_score']
    
    demographic_features = ['age', 'gender']
    
    # First, check which features actually exist in the data
    clinical_features = [f for f in clinical_features if f in X.columns]
    lab_features = [f for f in lab_features if f in X.columns]
    score_features = [f for f in score_features if f in X.columns]
    demographic_features = [f for f in demographic_features if f in X.columns]
    
    logger.info(f"Using clinical features: {clinical_features}")
    logger.info(f"Using lab features: {lab_features}")
    logger.info(f"Using score features: {score_features}")
    logger.info(f"Using demographic features: {demographic_features}")
    
    # Create preprocessing for each feature group
    
    # 1. For clinical features - Use MinMaxScaler to give them more weight
    clinical_transformer = Pipeline([
        ('clinical_scaler', MinMaxScaler(feature_range=(0, 2)))  # Increase weight by scaling to (0,2)
    ])
    
    # 2. For lab features - Use standardization with specific features
    lab_transformer = Pipeline([
        ('lab_scaler', StandardScaler())
    ])
    
    # 3. For score features - Use MinMaxScaler with higher range to increase importance
    score_transformer = Pipeline([
        ('score_scaler', MinMaxScaler(feature_range=(0, 3)))  # Give even more weight to scores
    ])
    
    # 4. Demographic features - Regular standardization
    demographic_transformer = Pipeline([
        ('demographic_scaler', StandardScaler())
    ])
    
    # Create a column transformer that applies the appropriate preprocessing to each feature group
    preprocessor = ColumnTransformer(
        transformers=[
            ('clinical', clinical_transformer, clinical_features),
            ('lab', lab_transformer, lab_features),
            ('score', score_transformer, score_features),
            ('demographic', demographic_transformer, demographic_features)
        ],
        remainder='passthrough'
    )
    
    # Create a new classifier - Random Forest with optimized hyperparameters
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    # Create the new pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Train and evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    
    logger.info("Model performance:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    # Check feature importances
    if hasattr(classifier, 'feature_importances_'):
        # Get feature names after preprocessing
        feature_names = X.columns.tolist()
        
        # Get feature importances
        importances = classifier.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        logger.info("Feature importances:")
        for i in range(min(10, len(feature_names))):
            logger.info(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return pipeline

def test_discrimination_power(model, n_samples=10):
    """Test the discrimination power of the model with multiple cases"""
    logger.info("Testing model discrimination power...")
    
    # Generate typical appendicitis cases
    appendicitis_cases = pd.DataFrame({
        "age": np.random.uniform(5, 15, n_samples),
        "gender": np.random.choice([0, 1], n_samples),
        "duration": np.random.uniform(20, 48, n_samples),
        "migration": np.ones(n_samples),
        "anorexia": np.ones(n_samples),
        "nausea": np.ones(n_samples),
        "vomiting": np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        "right_lower_quadrant_pain": np.ones(n_samples),
        "fever": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        "rebound_tenderness": np.ones(n_samples),
        "white_blood_cell_count": np.random.uniform(12, 20, n_samples),
        "neutrophil_percentage": np.random.uniform(75, 95, n_samples),
        "c_reactive_protein": np.random.uniform(50, 150, n_samples),
        "pediatric_appendicitis_score": np.random.uniform(7, 10, n_samples),
        "alvarado_score": np.random.uniform(7, 10, n_samples)
    })
    
    # Generate typical non-appendicitis cases
    non_appendicitis_cases = pd.DataFrame({
        "age": np.random.uniform(5, 15, n_samples),
        "gender": np.random.choice([0, 1], n_samples),
        "duration": np.random.uniform(4, 12, n_samples),
        "migration": np.zeros(n_samples),
        "anorexia": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        "nausea": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        "vomiting": np.zeros(n_samples),
        "right_lower_quadrant_pain": np.zeros(n_samples),
        "fever": np.zeros(n_samples),
        "rebound_tenderness": np.zeros(n_samples),
        "white_blood_cell_count": np.random.uniform(5, 10, n_samples),
        "neutrophil_percentage": np.random.uniform(40, 65, n_samples),
        "c_reactive_protein": np.random.uniform(5, 30, n_samples),
        "pediatric_appendicitis_score": np.random.uniform(1, 4, n_samples),
        "alvarado_score": np.random.uniform(1, 4, n_samples)
    })
    
    # Make predictions
    appendicitis_probs = model.predict_proba(appendicitis_cases)[:, 1]
    non_appendicitis_probs = model.predict_proba(non_appendicitis_cases)[:, 1]
    
    # Calculate statistics
    mean_app_prob = np.mean(appendicitis_probs)
    mean_non_app_prob = np.mean(non_appendicitis_probs)
    discrimination_power = mean_app_prob - mean_non_app_prob
    
    logger.info(f"Mean probability for appendicitis cases: {mean_app_prob:.4f}")
    logger.info(f"Mean probability for non-appendicitis cases: {mean_non_app_prob:.4f}")
    logger.info(f"Discrimination power: {discrimination_power:.4f}")
    
    # Test with extreme cases
    extreme_appendicitis = pd.DataFrame({
        "age": [10],
        "gender": [1],
        "duration": [36],
        "migration": [1],
        "anorexia": [1],
        "nausea": [1],
        "vomiting": [1],
        "right_lower_quadrant_pain": [1],
        "fever": [1],
        "rebound_tenderness": [1],
        "white_blood_cell_count": [18],
        "neutrophil_percentage": [90],
        "c_reactive_protein": [120],
        "pediatric_appendicitis_score": [10],
        "alvarado_score": [10]
    })
    
    extreme_non_appendicitis = pd.DataFrame({
        "age": [8],
        "gender": [0],
        "duration": [6],
        "migration": [0],
        "anorexia": [0],
        "nausea": [0],
        "vomiting": [0],
        "right_lower_quadrant_pain": [0],
        "fever": [0],
        "rebound_tenderness": [0],
        "white_blood_cell_count": [7],
        "neutrophil_percentage": [55],
        "c_reactive_protein": [10],
        "pediatric_appendicitis_score": [2],
        "alvarado_score": [2]
    })
    
    extreme_app_prob = model.predict_proba(extreme_appendicitis)[0, 1]
    extreme_non_app_prob = model.predict_proba(extreme_non_appendicitis)[0, 1]
    extreme_diff = extreme_app_prob - extreme_non_app_prob
    
    logger.info(f"Extreme appendicitis case probability: {extreme_app_prob:.4f}")
    logger.info(f"Extreme non-appendicitis case probability: {extreme_non_app_prob:.4f}")
    logger.info(f"Extreme case discrimination: {extreme_diff:.4f}")
    
    return discrimination_power, extreme_diff

def save_fixed_model(model):
    """Save the fixed model"""
    try:
        with open(FIXED_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Fixed model saved to {FIXED_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving fixed model: {str(e)}")
        return False

def update_app_model():
    """Update the model used by the Flask app"""
    try:
        app_model_path = os.path.join('models', 'best_model_retrained.pkl')
        
        # Create a backup of the current model
        import shutil
        import time
        backup_path = os.path.join('models', f'best_model_retrained_backup_{int(time.time())}.pkl')
        if os.path.exists(app_model_path):
            shutil.copy2(app_model_path, backup_path)
            logger.info(f"Created backup of current model: {backup_path}")
        
        # Copy the improved model to the app model path
        if os.path.exists(FIXED_MODEL_PATH):
            shutil.copy2(FIXED_MODEL_PATH, app_model_path)
            logger.info(f"Updated app model with fixed model")
            return True
        else:
            logger.error(f"Fixed model file not found: {FIXED_MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error updating app model: {str(e)}")
        return False

def main():
    """Main function to fix feature weights"""
    # Load model and data
    logger.info("Starting feature weight correction process...")
    _, X, y = load_model_and_data()
    
    if X is None or y is None:
        logger.error("Could not load data. Exiting.")
        return
    
    # Create a new model with properly weighted features
    fixed_model = create_feature_weighted_pipeline(X, y)
    
    # Test discrimination power
    test_discrimination_power(fixed_model)
    
    # Save the fixed model
    if save_fixed_model(fixed_model):
        logger.info("Feature weight correction completed successfully.")
        
        # Update app model automatically
        update_app_model()
        logger.info("App model updated with the fixed model")
    else:
        logger.error("Failed to save fixed model.")

if __name__ == "__main__":
    main()
