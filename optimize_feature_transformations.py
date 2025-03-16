"""
Optimize Feature Transformations for Pediatric Appendicitis Diagnosis
---------------------------------------------------------------------
This script implements advanced feature transformations to optimize model performance,
particularly focusing on:
1. Clinical feature engineering based on medical knowledge
2. Feature interaction modeling
3. Non-linear transformations
4. Recalibrating the Alvarado and PAS scores based on their components
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join('models', 'best_model_retrained.pkl')
OPTIMIZED_MODEL_PATH = os.path.join('models', 'best_model_optimized.pkl')
TRAINING_DATA_PATH = os.path.join('DATA', 'processed', 'training_data.csv')

class FeatureInteractionTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to create medically-informed feature interactions"""
    
    def __init__(self):
        self.feature_names = None
        
    def fit(self, X, y=None):
        """Fit the transformer to the data"""
        self.feature_names = X.columns
        return self
    
    def transform(self, X):
        """Apply the transformation"""
        X_df = X.copy()
        
        # Clinical features
        clinical_features = ['migration', 'anorexia', 'nausea', 'vomiting', 
                            'right_lower_quadrant_pain', 'fever', 'rebound_tenderness']
        available_clinical = [f for f in clinical_features if f in X_df.columns]
        
        # Lab features
        lab_features = ['white_blood_cell_count', 'neutrophil_percentage', 'c_reactive_protein']
        available_lab = [f for f in lab_features if f in X_df.columns]
        
        # Create new features
        
        # 1. Symptom intensity - number of symptoms present (non-zero)
        if len(available_clinical) >= 3:
            X_df['symptom_count'] = X_df[available_clinical].sum(axis=1)
            
        # 2. Inflammatory response - composite lab value feature
        if 'white_blood_cell_count' in X_df.columns and 'neutrophil_percentage' in X_df.columns:
            # WBC * Neutrophil % gives absolute neutrophil count, a strong indicator
            X_df['absolute_neutrophils'] = X_df['white_blood_cell_count'] * X_df['neutrophil_percentage'] / 100.0
        
        # 3. Key symptom combinations
        # Classic appendicitis triad: migration + RLQ pain + rebound tenderness
        if 'migration' in X_df.columns and 'right_lower_quadrant_pain' in X_df.columns and 'rebound_tenderness' in X_df.columns:
            X_df['classic_triad'] = ((X_df['migration'] >= 0.5) & 
                                     (X_df['right_lower_quadrant_pain'] >= 0.5) & 
                                     (X_df['rebound_tenderness'] >= 0.5)).astype(int)
        
        # 4. Pain and systemic response
        if 'right_lower_quadrant_pain' in X_df.columns and 'fever' in X_df.columns:
            X_df['localized_pain_with_fever'] = ((X_df['right_lower_quadrant_pain'] >= 0.5) & 
                                                (X_df['fever'] >= 0.5)).astype(int)
        
        # 5. Lab values combined score
        if len(available_lab) >= 2:
            # Normalize and combine lab values
            scaler = StandardScaler()
            lab_normalized = scaler.fit_transform(X_df[available_lab])
            X_df['lab_composite_score'] = np.mean(lab_normalized, axis=1)
            
        # 6. Alvarado components weights - if using the alvarado score
        if 'alvarado_score' in X_df.columns:
            # Emphasize score ranges based on clinical significance
            X_df['alvarado_risk'] = pd.cut(X_df['alvarado_score'], 
                                         bins=[0, 4, 6, 10], 
                                         labels=[0, 1, 2]).astype(int)
        
        return X_df
    
    def get_feature_names_out(self, input_features=None):
        """Return the feature names after transformation"""
        transformed_features = list(self.transform(pd.DataFrame(np.zeros((1, len(self.feature_names))), 
                                                 columns=self.feature_names)).columns)
        return transformed_features

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

def create_optimized_pipeline(X, y):
    """Create a new pipeline with optimized feature transformations"""
    logger.info("Creating optimized feature transformation pipeline...")
    
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
    
    # Create feature interaction transformer
    interaction_transformer = FeatureInteractionTransformer()
    
    # First apply the interactions
    feature_pipeline = Pipeline([
        ('interactions', interaction_transformer)
    ])
    
    # Apply the interaction pipeline to get the enhanced dataset
    X_enhanced = feature_pipeline.fit_transform(X)
    
    # Now determine which new features were added
    new_features = [col for col in X_enhanced.columns if col not in X.columns]
    logger.info(f"Created new derived features: {new_features}")
    
    # Split features into categories again, now including new features
    derived_lab_features = [f for f in new_features if 
                           'neutrophil' in f or 'lab' in f or 'composite' in f]
    
    derived_clinical_features = [f for f in new_features if 
                                'symptom' in f or 'triad' in f or 'pain' in f or f not in derived_lab_features]
    
    derived_score_features = [f for f in new_features if 'risk' in f]
    
    # Create preprocessing for each feature group
    
    # For clinical features - Use PowerTransformer for non-linear relations
    clinical_transformer = Pipeline([
        ('clinical_power', PowerTransformer(method='yeo-johnson', standardize=True))
    ])
    
    # For lab features - Use PowerTransformer with standardization
    lab_transformer = Pipeline([
        ('lab_power', PowerTransformer(method='yeo-johnson', standardize=True))
    ])
    
    # For score features - Use specialized scaling
    score_transformer = Pipeline([
        ('score_poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
        ('score_scaler', StandardScaler())
    ])
    
    # For demographic features - Use standard scaling
    demographic_transformer = Pipeline([
        ('demographic_scaler', StandardScaler())
    ])
    
    # Build column transformer to process all feature types
    preprocessor = ColumnTransformer(
        transformers=[
            ('clinical', clinical_transformer, clinical_features + derived_clinical_features),
            ('lab', lab_transformer, lab_features + derived_lab_features),
            ('score', score_transformer, score_features + derived_score_features),
            ('demographic', demographic_transformer, demographic_features)
        ],
        remainder='passthrough'
    )
    
    # Create complete pipeline with feature engineering and model
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42
    )
    
    # Create calibrated pipeline
    pipeline = Pipeline([
        ('feature_engineering', feature_pipeline),
        ('preprocessor', preprocessor),
        ('classifier', CalibratedClassifierCV(
            estimator=classifier, 
            method='sigmoid',  # Platt scaling
            cv=5
        ))
    ])
    
    # Evaluate model with cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    logger.info(f"Cross-validation ROC AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Train on full dataset
    logger.info("Training optimized model on full dataset...")
    pipeline.fit(X, y)
    
    # Evaluate on a test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info("Model performance metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
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

def save_optimized_model(model):
    """Save the optimized model"""
    try:
        with open(OPTIMIZED_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Optimized model saved to {OPTIMIZED_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving optimized model: {str(e)}")
        return False

def compare_with_previous_models():
    """Compare the optimized model with previously saved models"""
    logger.info("Comparing model versions...")
    
    model_paths = [
        ('Original', MODEL_PATH),
        ('Fixed', os.path.join('models', 'best_model_fixed.pkl')),
        ('Optimized', OPTIMIZED_MODEL_PATH)
    ]
    
    models = {}
    
    # Load all available models
    for name, path in model_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
                logger.info(f"Loaded {name} model from {path}")
            except Exception as e:
                logger.error(f"Error loading {name} model: {str(e)}")
    
    if len(models) <= 1:
        logger.info("Not enough models available for comparison")
        return
    
    # Generate test cases
    n_test = 20
    
    # Generate typical appendicitis cases
    appendicitis_cases = pd.DataFrame({
        "age": np.random.uniform(5, 15, n_test),
        "gender": np.random.choice([0, 1], n_test),
        "duration": np.random.uniform(20, 48, n_test),
        "migration": np.ones(n_test),
        "anorexia": np.ones(n_test),
        "nausea": np.ones(n_test),
        "vomiting": np.random.choice([0, 1], n_test, p=[0.2, 0.8]),
        "right_lower_quadrant_pain": np.ones(n_test),
        "fever": np.random.choice([0, 1], n_test, p=[0.3, 0.7]),
        "rebound_tenderness": np.ones(n_test),
        "white_blood_cell_count": np.random.uniform(12, 20, n_test),
        "neutrophil_percentage": np.random.uniform(75, 95, n_test),
        "c_reactive_protein": np.random.uniform(50, 150, n_test),
        "pediatric_appendicitis_score": np.random.uniform(7, 10, n_test),
        "alvarado_score": np.random.uniform(7, 10, n_test)
    })
    
    # Generate typical non-appendicitis cases
    non_appendicitis_cases = pd.DataFrame({
        "age": np.random.uniform(5, 15, n_test),
        "gender": np.random.choice([0, 1], n_test),
        "duration": np.random.uniform(4, 12, n_test),
        "migration": np.zeros(n_test),
        "anorexia": np.random.choice([0, 1], n_test, p=[0.8, 0.2]),
        "nausea": np.random.choice([0, 1], n_test, p=[0.5, 0.5]),
        "vomiting": np.zeros(n_test),
        "right_lower_quadrant_pain": np.zeros(n_test),
        "fever": np.zeros(n_test),
        "rebound_tenderness": np.zeros(n_test),
        "white_blood_cell_count": np.random.uniform(5, 10, n_test),
        "neutrophil_percentage": np.random.uniform(40, 65, n_test),
        "c_reactive_protein": np.random.uniform(5, 30, n_test),
        "pediatric_appendicitis_score": np.random.uniform(1, 4, n_test),
        "alvarado_score": np.random.uniform(1, 4, n_test)
    })
    
    # Make predictions and compare
    logger.info("\nAppendicitis case predictions:")
    for name, model in models.items():
        try:
            probas = model.predict_proba(appendicitis_cases)[:, 1]
            logger.info(f"  {name} model: mean={np.mean(probas):.4f}, std={np.std(probas):.4f}")
        except Exception as e:
            logger.error(f"  Error with {name} model: {str(e)}")
    
    logger.info("\nNon-appendicitis case predictions:")
    for name, model in models.items():
        try:
            probas = model.predict_proba(non_appendicitis_cases)[:, 1]
            logger.info(f"  {name} model: mean={np.mean(probas):.4f}, std={np.std(probas):.4f}")
        except Exception as e:
            logger.error(f"  Error with {name} model: {str(e)}")
    
    # Calculate discrimination power
    logger.info("\nDiscrimination power:")
    for name, model in models.items():
        try:
            app_probs = model.predict_proba(appendicitis_cases)[:, 1]
            non_app_probs = model.predict_proba(non_appendicitis_cases)[:, 1]
            discrim = np.mean(app_probs) - np.mean(non_app_probs)
            logger.info(f"  {name} model: {discrim:.4f}")
        except Exception as e:
            logger.error(f"  Error with {name} model: {str(e)}")

def update_app_model():
    """Update the model used by the Flask app with optimized model"""
    try:
        app_model_path = os.path.join('models', 'best_model_retrained.pkl')
        
        # Create a backup of the current model
        import shutil
        import time
        backup_path = os.path.join('models', f'best_model_retrained_backup_{int(time.time())}.pkl')
        if os.path.exists(app_model_path):
            shutil.copy2(app_model_path, backup_path)
            logger.info(f"Created backup of current model: {backup_path}")
        
        # Copy the optimized model to the app model path
        if os.path.exists(OPTIMIZED_MODEL_PATH):
            shutil.copy2(OPTIMIZED_MODEL_PATH, app_model_path)
            logger.info(f"Updated app model with optimized model")
            return True
        else:
            logger.error(f"Optimized model file not found: {OPTIMIZED_MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error updating app model: {str(e)}")
        return False

def main():
    """Main function to optimize feature transformations"""
    logger.info("Starting feature transformation optimization process...")
    
    # Load model and data
    _, X, y = load_model_and_data()
    
    if X is None or y is None:
        logger.error("Could not load data. Exiting.")
        return
    
    # Create optimized pipeline
    optimized_model = create_optimized_pipeline(X, y)
    
    # Test discrimination power
    test_discrimination_power(optimized_model)
    
    # Save the optimized model
    if save_optimized_model(optimized_model):
        logger.info("Feature transformation optimization completed successfully.")
        
        # Compare with previous models
        compare_with_previous_models()
        
        # Prompt for model update
        response = input("Do you want to update the app model with the optimized model? (y/n): ")
        if response.lower() == 'y':
            if update_app_model():
                logger.info("App model updated with the optimized model")
            else:
                logger.error("Failed to update app model")
    else:
        logger.error("Failed to save optimized model.")

if __name__ == "__main__":
    main()
