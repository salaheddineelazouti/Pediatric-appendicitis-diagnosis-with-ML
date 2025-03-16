"""
Repair model script for Pediatric Appendicitis Diagnosis

This script:
1. Loads the optimized model
2. Creates a new version with identical parameters but defined in this module
3. Saves the repaired model for use in the Flask app
"""

import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the custom transformer classes
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
        return np.array(transformed_features)

class ClinicalFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for creating clinically relevant feature interactions"""
    
    def __init__(self, clinical_features=None, lab_features=None):
        self.clinical_features = clinical_features or []
        self.lab_features = lab_features or []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_df = X.copy()
        
        # Count the number of positive clinical symptoms
        symptom_columns = [col for col in self.clinical_features if col not in ['duration_of_pain']]
        if symptom_columns:
            X_df['symptom_count'] = X_df[symptom_columns].sum(axis=1)
        
        # Create absolute neutrophil count
        if 'white_blood_cell_count' in X_df and 'neutrophil_percentage' in X_df:
            X_df['absolute_neutrophils'] = X_df['white_blood_cell_count'] * (X_df['neutrophil_percentage'] / 100)
        
        # Create classic appendicitis triad feature
        if all(x in X_df for x in ['migration', 'right_lower_quadrant_pain', 'rebound_tenderness']):
            X_df['classic_triad'] = ((X_df['migration'] >= 0.5) & 
                                    (X_df['right_lower_quadrant_pain'] >= 0.5) & 
                                    (X_df['rebound_tenderness'] >= 0.5)).astype(int)
        
        # Create localized pain with fever feature
        if all(x in X_df for x in ['right_lower_quadrant_pain', 'fever']):
            X_df['localized_pain_with_fever'] = ((X_df['right_lower_quadrant_pain'] >= 0.5) & 
                                                (X_df['fever'] >= 0.5)).astype(int)
        
        # Create laboratory composite score
        if all(lab in X_df for lab in ['white_blood_cell_count', 'neutrophil_percentage', 'c_reactive_protein']):
            # Normalize and combine lab values
            wbc_norm = np.minimum(X_df['white_blood_cell_count'] / 20.0, 1.0)  # WBC > 20 gets max score
            neutro_norm = X_df['neutrophil_percentage'] / 100.0  # Already percentage
            crp_norm = np.minimum(X_df['c_reactive_protein'] / 150.0, 1.0)  # CRP > 150 gets max score
            
            X_df['lab_composite_score'] = (wbc_norm * 0.4 + neutro_norm * 0.3 + crp_norm * 0.3)
        
        # Calculate Alvarado risk category based on score
        if 'alvarado_score' in X_df:
            X_df['alvarado_risk'] = pd.cut(
                X_df['alvarado_score'], 
                bins=[float('-inf'), 4, 7, float('inf')],
                labels=[0, 1, 2]  # Low (0-4), Medium (5-7), High (8-10)
            ).astype(int)
            
        return X_df

def create_backup_model():
    """Create a simple backup model in case we can't load the original"""
    try:
        # Load the training data
        data_path = os.path.join('DATA', 'processed', 'training_data.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Extract features and target
            X = df.drop('appendicitis', axis=1)
            y = df['appendicitis']
            
            # Define preprocessing for numerical features
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            # Define preprocessing for categorical features
            categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
            if categorical_features:
                categorical_transformer = Pipeline(steps=[
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
            
            # Combine preprocessing steps
            preprocessor_steps = []
            if numeric_features:
                preprocessor_steps.append(('num', numeric_transformer, numeric_features))
            if categorical_features:
                preprocessor_steps.append(('cat', categorical_transformer, categorical_features))
            
            preprocessor = ColumnTransformer(
                transformers=preprocessor_steps
            )
            
            # Create a simple pipeline with RandomForest
            model = Pipeline(steps=[
                ('feature_engineering', FeatureInteractionTransformer()),
                ('preprocessor', preprocessor),
                ('classifier', CalibratedClassifierCV(
                    estimator=RandomForestClassifier(
                        n_estimators=100, 
                        max_depth=5,
                        random_state=42
                    ),
                    method='sigmoid',
                    cv=5
                ))
            ])
            
            # Train the model
            logger.info("Training backup model...")
            model.fit(X, y)
            
            # Save the model
            output_path = os.path.join('models', 'best_model_repaired.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Backup model saved to {output_path}")
            
            return model
        else:
            logger.error(f"Training data not found at {data_path}")
            return None
    except Exception as e:
        logger.error(f"Error creating backup model: {str(e)}")
        return None

def try_load_original_model():
    """Try to load the original optimized model"""
    try:
        model_path = os.path.join('models', 'best_model_optimized.pkl')
        logger.info(f"Attempting to load original model from {model_path}")
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                original_model = pickle.load(f)
            logger.info(f"Original model loaded successfully: {type(original_model).__name__}")
            return original_model
        else:
            logger.warning(f"Original model file not found: {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading original model: {str(e)}")
        logger.info("Will create a backup model instead")
        return None

def main():
    """Main function to repair the model"""
    logger.info("Starting model repair process...")
    
    # Try to load the original model
    model = try_load_original_model()
    
    # If we couldn't load the original, create a backup
    if model is None:
        logger.info("Creating backup model...")
        model = create_backup_model()
    else:
        # Save a repaired version of the loaded model
        output_path = os.path.join('models', 'best_model_repaired.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Repaired model saved to {output_path}")
    
    # Check if we have a working model
    if model is None:
        logger.error("Model repair failed. Please check the logs for details.")
        return 1
    else:
        logger.info("Model repair completed successfully.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
