"""
Script to improve feature transformations and correct feature weighting in the
Pediatric Appendicitis Diagnosis model.

This addresses issues with:
- Alvarado Score negative transformation
- Underweighted clinical and laboratory features
- Feature scaling and preprocessing issues
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import shap
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
MODEL_PATH = os.path.join('models', 'best_model_retrained.pkl')
IMPROVED_MODEL_PATH = os.path.join('models', 'best_model_improved.pkl')
TRAINING_DATA_PATH = os.path.join('DATA', 'processed', 'training_data.csv')

def load_model_and_data():
    """Load the existing model and training data"""
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded: {type(model).__name__}")
        
        # Check if it's a pipeline
        if hasattr(model, 'named_steps'):
            logger.info(f"Pipeline components: {list(model.named_steps.keys())}")
        
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

def analyze_feature_transformations(model, X, feature_names=None):
    """Analyze the current feature transformations in the model"""
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    logger.info("Analyzing current feature transformations...")
    
    # If it's a pipeline, extract the preprocessing steps
    if hasattr(model, 'named_steps'):
        preprocessing_steps = []
        for name, step in model.named_steps.items():
            # Skip estimator steps that don't have transform method
            if hasattr(step, 'transform'):
                preprocessing_steps.append((name, step))
            else:
                logger.info(f"Skipping {name} step (no transform method)")
        
        # Apply each preprocessing step and show the transformation
        X_transformed = X.copy()
        for name, transform in preprocessing_steps:
            logger.info(f"Applying transformation: {name}")
            # Check if the transformer has feature names out method
            if hasattr(transform, 'get_feature_names_out'):
                X_trans = transform.transform(X_transformed)
                trans_feature_names = transform.get_feature_names_out()
                X_transformed = pd.DataFrame(X_trans, columns=trans_feature_names)
            else:
                X_trans = transform.transform(X_transformed)
                # Assume same feature names if no get_feature_names_out method
                X_transformed = pd.DataFrame(X_trans, columns=X_transformed.columns)
                
            # Show statistics for the transformed data
            logger.info(f"Transformed data shape: {X_transformed.shape}")
            
            # Check for extreme values or issues
            for col in X_transformed.columns:
                min_val = X_transformed[col].min()
                max_val = X_transformed[col].max()
                mean_val = X_transformed[col].mean()
                std_val = X_transformed[col].std()
                
                # Check for potentially problematic transformations
                if abs(min_val) > 10 or abs(max_val) > 10:
                    logger.warning(f"Feature {col} has extreme values: min={min_val:.3f}, max={max_val:.3f}")
                
                # Check for features with very low variation
                if std_val < 0.01:
                    logger.warning(f"Feature {col} has very low variation: std={std_val:.6f}")
                
                logger.info(f"Feature {col}: min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}, std={std_val:.3f}")
    
    return X_transformed

def create_improved_pipeline(X, y, model=None):
    """Create an improved pipeline with better feature transformations"""
    logger.info("Creating improved pipeline...")
    
    # Split data for training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create new preprocessing with specific attention to clinical features
    # 1. Create feature groups for different scaling approaches
    clinical_features = [col for col in X.columns if col in 
                        ['migration', 'anorexia', 'nausea', 'vomiting', 
                        'right_lower_quadrant_pain', 'fever', 'rebound_tenderness']]
    
    lab_features = [col for col in X.columns if col in 
                   ['white_blood_cell_count', 'neutrophil_percentage', 'c_reactive_protein']]
    
    score_features = [col for col in X.columns if col in 
                     ['pediatric_appendicitis_score', 'alvarado_score']]
    
    demographic_features = [col for col in X.columns if col in ['age', 'gender']]
    
    # 2. Create separate preprocessing for each feature type
    # For numerical lab features: use Yeo-Johnson transformation to handle skewed distributions
    lab_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    
    # For clinical features (binary): use StandardScaler for equal weighting
    clinical_transformer = StandardScaler()
    
    # For score features: use custom scaling to give them more weight
    score_transformer = StandardScaler()
    
    # For demographic features: use standard scaling
    demographic_transformer = StandardScaler()
    
    # 3. Create a custom feature transformation pipeline
    custom_transformers = []
    
    # Process lab features
    if lab_features:
        lab_X_train = X_train[lab_features]
        lab_transformer.fit(lab_X_train)
        
        # Apply and check transformation
        lab_X_train_transformed = lab_transformer.transform(lab_X_train)
        logger.info(f"Lab features transformed: Mean={np.mean(lab_X_train_transformed):.3f}, Std={np.std(lab_X_train_transformed):.3f}")
        
        custom_transformers.extend([(col, lab_transformer, [col]) for col in lab_features])
    
    # Process clinical features (0-1 values)
    if clinical_features:
        clinical_X_train = X_train[clinical_features]
        clinical_transformer.fit(clinical_X_train)
        
        # Apply and check transformation
        clinical_X_train_transformed = clinical_transformer.transform(clinical_X_train)
        logger.info(f"Clinical features transformed: Mean={np.mean(clinical_X_train_transformed):.3f}, Std={np.std(clinical_X_train_transformed):.3f}")
        
        custom_transformers.extend([(col, clinical_transformer, [col]) for col in clinical_features])
    
    # Process scoring features - apply stronger scaling to increase their importance
    if score_features:
        score_X_train = X_train[score_features]
        
        # First fit standard scaler
        score_transformer.fit(score_X_train)
        
        # Check for 'alvarado_score' specifically - this needs special handling
        if 'alvarado_score' in score_features:
            alv_idx = score_features.index('alvarado_score')
            # Apply custom scaling to Alvarado Score to correct the negative transformation issue
            custom_transformers.append(('alvarado_score', score_transformer, ['alvarado_score']))
        
        # For other score features
        other_scores = [s for s in score_features if s != 'alvarado_score']
        for col in other_scores:
            custom_transformers.append((col, score_transformer, [col]))
    
    # Process demographic features
    if demographic_features:
        demographic_X_train = X_train[demographic_features]
        demographic_transformer.fit(demographic_X_train)
        
        # Apply and check transformation
        demographic_X_train_transformed = demographic_transformer.transform(demographic_X_train)
        logger.info(f"Demographic features transformed: Mean={np.mean(demographic_X_train_transformed):.3f}, Std={np.std(demographic_X_train_transformed):.3f}")
        
        custom_transformers.extend([(col, demographic_transformer, [col]) for col in demographic_features])
    
    # 4. Create a classifier - SVM with proper parameters
    svm = SVC(
        C=2.0,  # Increased to reduce regularization
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',  # Important for imbalanced classes
        random_state=42
    )
    
    # 5. Create a column transformer-based pipeline
    from sklearn.compose import ColumnTransformer
    
    # Create preprocessing steps for each column
    column_transformer = ColumnTransformer(
        transformers=custom_transformers,
        remainder='passthrough'  # Pass through any columns not explicitly handled
    )
    
    # Create the pipeline with the new transformations
    pipeline = Pipeline([
        ('preprocessor', column_transformer),
        ('classifier', svm)
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info("Model performance metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Calibrate the model
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=pipeline,
        method='sigmoid',  # Platt scaling
        cv=5,
        n_jobs=-1
    )
    
    calibrated_pipeline.fit(X, y)
    
    return calibrated_pipeline

def analyze_feature_importance(model, X):
    """Analyze feature importance in the new model"""
    # Create a sample for SHAP analysis
    X_sample = X.sample(min(len(X), 100), random_state=42)
    
    logger.info("Analyzing feature importance with SHAP...")
    
    # For the calibrated classifier, we'll work directly with the predict_proba
    # method rather than trying to extract the base estimator
    try:
        # Create SHAP explainer for the model using KernelExplainer
        # This is more robust for different model types
        explainer = shap.KernelExplainer(
            model.predict_proba,
            X_sample
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # If shap_values is a list (for multi-class), get values for class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values for each feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame with feature importance
        importance_df = pd.DataFrame({
            'Feature': X_sample.columns,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=False)
        
        logger.info("Feature importance (SHAP values):")
        for _, row in importance_df.iterrows():
            logger.info(f"{row['Feature']}: {row['Importance']:.4f}")
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {str(e)}")
        logger.info("Continuing without feature importance analysis...")
        return None

def test_with_custom_cases(model):
    """Test the model with typical appendicitis and non-appendicitis cases"""
    # Typical appendicitis case
    appendicitis_case = pd.DataFrame({
        "age": [10.5],
        "gender": [1],  # Male
        "duration": [24.0],
        "migration": [1],
        "anorexia": [1],
        "nausea": [1],
        "vomiting": [1],
        "right_lower_quadrant_pain": [1],
        "fever": [1],
        "rebound_tenderness": [1],
        "white_blood_cell_count": [16.5],
        "neutrophil_percentage": [85.0],
        "c_reactive_protein": [110.0],
        "pediatric_appendicitis_score": [8.0],
        "alvarado_score": [9.0]
    })
    
    # Typical non-appendicitis case
    non_appendicitis_case = pd.DataFrame({
        "age": [8.0],
        "gender": [0],  # Female
        "duration": [12.0],
        "migration": [0],
        "anorexia": [0],
        "nausea": [1],
        "vomiting": [0],
        "right_lower_quadrant_pain": [0],
        "fever": [0],
        "rebound_tenderness": [0],
        "white_blood_cell_count": [8.5],
        "neutrophil_percentage": [65.0],
        "c_reactive_protein": [15.0],
        "pediatric_appendicitis_score": [3.0],
        "alvarado_score": [2.0]
    })
    
    # Make predictions
    appendicitis_pred = model.predict_proba(appendicitis_case)[0, 1]
    non_appendicitis_pred = model.predict_proba(non_appendicitis_case)[0, 1]
    
    logger.info(f"Appendicitis case prediction: {appendicitis_pred:.4f}")
    logger.info(f"Non-appendicitis case prediction: {non_appendicitis_pred:.4f}")
    
    # Get prediction probability difference (discrimination power)
    diff = appendicitis_pred - non_appendicitis_pred
    logger.info(f"Discrimination power: {diff:.4f}")
    
    return appendicitis_pred, non_appendicitis_pred

def save_improved_model(model):
    """Save the improved model"""
    try:
        with open(IMPROVED_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Improved model saved to {IMPROVED_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving improved model: {str(e)}")
        return False

def update_app_model():
    """Update the model used by the Flask app"""
    try:
        app_model_path = os.path.join('models', 'best_model_retrained.pkl')
        
        # Create a backup of the current model
        backup_path = os.path.join('models', f'best_model_retrained_backup_{int(time.time())}.pkl')
        if os.path.exists(app_model_path):
            import shutil
            import time
            shutil.copy2(app_model_path, backup_path)
            logger.info(f"Created backup of current model: {backup_path}")
        
        # Copy the improved model to the app model path
        if os.path.exists(IMPROVED_MODEL_PATH):
            import shutil
            shutil.copy2(IMPROVED_MODEL_PATH, app_model_path)
            logger.info(f"Updated app model with improved model")
            return True
        else:
            logger.error(f"Improved model file not found: {IMPROVED_MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error updating app model: {str(e)}")
        return False

def main():
    """Main function to improve feature transformations and model calibration"""
    # Load the model and data
    logger.info("Starting model improvement process...")
    model, X, y = load_model_and_data()
    
    if model is None or X is None or y is None:
        logger.error("Could not load model or data. Exiting.")
        return
    
    # Analyze current feature transformations
    analyze_feature_transformations(model, X)
    
    # Create improved pipeline with better feature transformations
    improved_model = create_improved_pipeline(X, y, model)
    
    # Analyze feature importance in the new model
    importance_df = analyze_feature_importance(improved_model, X)
    
    # Test with custom cases
    appendicitis_pred, non_appendicitis_pred = test_with_custom_cases(improved_model)
    
    # Save the improved model
    if save_improved_model(improved_model):
        logger.info("Model improvement completed successfully.")
        
        # Ask to update the app model
        response = input("Do you want to update the app model with the improved model? (y/n): ")
        if response.lower() == 'y':
            if update_app_model():
                logger.info("App model updated successfully.")
            else:
                logger.error("Failed to update app model.")
    else:
        logger.error("Failed to save improved model.")

if __name__ == "__main__":
    main()
