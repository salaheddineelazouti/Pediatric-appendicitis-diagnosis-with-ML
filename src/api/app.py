"""
Main application file for the Pediatric Appendicitis Diagnosis web application.
This module initializes the Flask application and sets up the web server.
"""

import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, session
import traceback
import io
import uuid
import base64
from datetime import datetime
import time
from sklearn.base import BaseEstimator, TransformerMixin
import json

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the AI assistant module
from src.ai_assistant.gemini_integration import get_assistant_response, explain_prediction_features, get_clinical_recommendations, reset_assistant

# Import custom transformer for model loading
sys.path.append(project_root)
from optimize_feature_transformations import FeatureInteractionTransformer

# Import calibration components
from src.calibration.calibration_utils import integrate_calibration_with_shap

# Import specialized utilities
from ..utils.outlier_detection import detect_outliers, filter_outliers

# Define ClinicalFeatureTransformer class here for model loading
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

# Import SHAP explainability components
from src.explainability.shap_explainer import ShapExplainer

# Setup logging
# Make sure logs directory exists
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'app.log'), mode='a')
    ]
)
logger = logging.getLogger('api')

# Initialize Flask app
app = Flask(__name__, 
           static_folder=os.path.join(os.path.dirname(__file__), 'static'),
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Configure app
app.config['SECRET_KEY'] = 'pediatric-appendicitis-diagnosis-key'
app.config['MODEL_PATH'] = os.path.join(project_root, 'models', 'best_model_simple.pkl')
app.config['CALIBRATED_MODEL_PATH'] = os.path.join(project_root, 'models', 'best_model_calibrated.pkl')
app.config['USE_CALIBRATED_MODEL'] = True  # Flag pour utiliser le modèle calibré

# Feature lists for the form (matched exactly to model training features)
DEMOGRAPHIC_FEATURES = [
    {"name": "age", "label": "Age (years)", "type": "number", "min": 1, "max": 18, "step": 0.1, "required": True},
    {"name": "gender", "label": "Gender", "type": "select", "options": [{"value": "male", "label": "Male"}, {"value": "female", "label": "Female"}], "required": True}
]

CLINICAL_FEATURES = [
    {"name": "duration", "label": "Duration of Pain (hours)", "type": "number", "min": 0, "max": 120, "step": 0.5, "required": True},
    {"name": "migration", "label": "Migration of Pain", "type": "checkbox", "description": "Pain began periumbilically and migrated to RLQ"},
    {"name": "anorexia", "label": "Anorexia (Loss of appetite)", "type": "checkbox"},
    {"name": "nausea", "label": "Nausea", "type": "checkbox"},
    {"name": "vomiting", "label": "Vomiting", "type": "checkbox"},
    {"name": "right_lower_quadrant_pain", "label": "Right Lower Quadrant Pain", "type": "checkbox", "description": "Pain on palpation in the right lower quadrant"},
    {"name": "fever", "label": "Fever (>38°C/100.4°F)", "type": "checkbox"},
    {"name": "rebound_tenderness", "label": "Rebound Tenderness", "type": "checkbox", "description": "Pain when quickly releasing pressure from abdomen"}
]

LABORATORY_FEATURES = [
    {"name": "white_blood_cell_count", "label": "White Blood Cell Count (×10³/μL)", "type": "number", "min": 0, "max": 40, "step": 0.1, "required": True},
    {"name": "neutrophil_percentage", "label": "Neutrophil Percentage (%)", "type": "number", "min": 0, "max": 100, "step": 0.1, "required": True},
    {"name": "c_reactive_protein", "label": "C-Reactive Protein (mg/L)", "type": "number", "min": 0, "max": 300, "step": 0.1, "required": True}
]

SCORING_FEATURES = [
    {"name": "pediatric_appendicitis_score", "label": "Pediatric Appendicitis Score (PAS)", "type": "number", "min": 0, "max": 10, "step": 1, "required": False, "description": "If known"},
    {"name": "alvarado_score", "label": "Alvarado Score", "type": "number", "min": 0, "max": 10, "step": 1, "required": False, "description": "If known"}
]

# Load model
def load_model():
    try:
        model_path = app.config['MODEL_PATH']
        calibrated_model_path = app.config['CALIBRATED_MODEL_PATH']
        use_calibrated = app.config['USE_CALIBRATED_MODEL']
        
        # Check if model files exist before trying to load them
        calibrated_exists = os.path.exists(calibrated_model_path)
        standard_exists = os.path.exists(model_path)
        
        # Log model file status
        logger.info(f"Calibrated model file exists: {calibrated_exists}")
        logger.info(f"Standard model file exists: {standard_exists}")
        
        # Si on utilise le modèle calibré et qu'il existe
        if use_calibrated and calibrated_exists:
            logger.info(f"Tentative de chargement du modèle calibré depuis {calibrated_model_path}")
            with open(calibrated_model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Modèle calibré chargé avec succès: {type(model).__name__}")
            return model
        # Sinon, charger le modèle standard s'il existe
        elif standard_exists:
            logger.info(f"Chargement du modèle standard depuis {model_path}")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Modèle standard chargé avec succès: {type(model).__name__}")
            return model
        # Fallback to other available models if neither primary model exists
        else:
            # Try to find any available model file
            models_dir = os.path.dirname(model_path)
            if os.path.exists(models_dir):
                available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and f.startswith('best_model')]
                if available_models:
                    fallback_model_path = os.path.join(models_dir, available_models[0])
                    logger.warning(f"Primary models not found. Falling back to: {fallback_model_path}")
                    with open(fallback_model_path, 'rb') as f:
                        model = pickle.load(f)
                    logger.info(f"Fallback model loaded successfully: {type(model).__name__}")
                    return model
            
            logger.error(f"No model files found in {models_dir}")
            return None
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        logger.error(traceback.format_exc())
        return None

model = load_model()

# Create global explainer singleton to avoid recreating for each request
explainer = None
X_sample = None

def initialize_explainer(force_new=False):
    """Initialize SHAP explainer with sample data for background distribution"""
    global explainer, X_sample, model
    
    # Return the existing explainer if already initialized and not forcing new
    if explainer is not None and not force_new:
        return explainer
        
    if model is None:
        logger.warning("Cannot initialize explainer: model not loaded")
        return None
    
    try:
        # Ensure images directory exists
        os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'images'), exist_ok=True)
        
        # Load a sample of data for the explainer background
        # Typically this would be your training data
        # Here we'll use synthetic data with the right feature names
        feature_names = model.feature_names_in_
        sample_size = 100
        
        # Create sample data with reasonable ranges for each feature
        sample_data = {}
        
        # Generate sensible values for each feature
        for feature in feature_names:
            if feature == 'age':
                sample_data[feature] = np.random.uniform(1, 18, sample_size)
            elif feature == 'gender':
                sample_data[feature] = np.random.choice([0, 1], sample_size)
            elif feature == 'duration':
                sample_data[feature] = np.random.uniform(0, 120, sample_size)
            elif feature in ['migration', 'anorexia', 'nausea', 'vomiting', 
                           'right_lower_quadrant_pain', 'fever', 'rebound_tenderness']:
                sample_data[feature] = np.random.choice([0, 1], sample_size)
            elif feature == 'white_blood_cell_count':
                sample_data[feature] = np.random.uniform(4, 25, sample_size)
            elif feature == 'neutrophil_percentage':
                sample_data[feature] = np.random.uniform(40, 95, sample_size)
            elif feature == 'c_reactive_protein':
                sample_data[feature] = np.random.uniform(0, 200, sample_size)
            else:
                # Default for any other features
                sample_data[feature] = np.random.uniform(0, 1, sample_size)
        
        X_sample = pd.DataFrame(sample_data)
        
        # Initialize explainer
        explainer = ShapExplainer(model, X_sample)
        logger.info("SHAP explainer initialized successfully")
        return explainer
        
    except Exception as e:
        logger.error(f"Error initializing SHAP explainer: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Try to initialize explainer at startup
if model is not None:
    explainer = initialize_explainer()

def extract_form_data(form):
    """
    Extract form data into a dictionary.
    """
    form_data = {}
    
    # Process each feature group
    for feature_group in [DEMOGRAPHIC_FEATURES, CLINICAL_FEATURES, LABORATORY_FEATURES, SCORING_FEATURES]:
        for feature in feature_group:
            name = feature['name']
            # Handle checkboxes - these..should be 1 if checked, 0 if not
            if feature['type'] == 'checkbox':
                form_data[name] = 1 if form.get(name) else 0
            # Handle gender specifically
            elif name == 'gender':
                form_data[name] = 1 if form.get(name) == 'male' else 0
            # Handle all other fields
            else:
                value = form.get(name, '')
                if value:
                    try:
                        form_data[name] = float(value)
                    except ValueError:
                        logger.warning(f"Invalid value for {name}: {value}, setting to None")
                        form_data[name] = None
                else:
                    form_data[name] = None
                    
    return form_data

def get_risk_class(probability):
    """Get risk class based on probability"""
    if probability >= 0.7:
        return 'High Risk'
    elif probability >= 0.3:
        return 'Moderate Risk'
    else:
        return 'Low Risk'

def format_feature_name(feature_name):
    """Format feature name for display"""
    display_names = {
        'age': 'Age (years)',
        'gender': 'Gender (Male)',
        'duration': 'Pain Duration (hours)',
        'migration': 'Pain Migration to RLQ',
        'anorexia': 'Anorexia',
        'nausea': 'Nausea',
        'vomiting': 'Vomiting',
        'right_lower_quadrant_pain': 'Right Lower Quadrant Pain',
        'fever': 'Fever',
        'rebound_tenderness': 'Rebound Tenderness',
        'white_blood_cell_count': 'White Blood Cell Count',
        'neutrophil_percentage': 'Neutrophil Percentage',
        'c_reactive_protein': 'C-Reactive Protein'
    }
    return display_names.get(feature_name, ' '.join(word.capitalize() for word in feature_name.split('_')))

def format_feature_value(feature_name, value):
    """Format feature value for display"""
    if value is None:
        return "N/A"
        
    # Handle gender specifically
    if feature_name == 'gender':
        return "Male" if int(value) == 1 else "Female"
        
    # Handle boolean features
    boolean_features = ['migration', 'anorexia', 'nausea', 'vomiting', 
                        'right_lower_quadrant_pain', 'fever', 'rebound_tenderness']
    if feature_name in boolean_features:
        return "Yes" if int(value) == 1 else "No"
        
    # For numeric values in tests, just return the string representation
    if isinstance(value, (int, float)):
        return str(value)
    else:
        return str(value)

def create_waterfall_chart(base_value, shap_values, feature_names, final_prediction, output_path=None):
    """
    Create a waterfall chart showing how each feature contributes to the final prediction.
    
    Args:
        base_value: Base prediction value (expected value)
        shap_values: SHAP values for each feature
        feature_names: Names of features
        final_prediction: Final prediction value
        output_path: Optional path to save the figure
        
    Returns:
        Base64 encoded string of the figure
    """
    try:
        # Ensure base_value is a float
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[0])
        else:
            base_value = float(base_value)
        
        # Ensure shap_values is a numpy array
        shap_values = np.array(shap_values)
        
        # Sort features by absolute SHAP value
        indices = np.argsort(np.abs(shap_values))[::-1]
        sorted_values = shap_values[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        # Limit to top 10 features for clarity
        if len(sorted_values) > 10:
            sorted_values = sorted_values[:10]
            sorted_names = sorted_names[:10]
        
        # Set up the figure
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        # Create bottom values for each bar (cumulative sum)
        cumulative = np.zeros(len(sorted_values) + 1)
        cumulative[0] = base_value
        for i in range(len(sorted_values)):
            cumulative[i+1] = cumulative[i] + sorted_values[i]
        
        # Position of bars on y-axis
        pos = range(len(sorted_values) + 1)
        
        # Create custom colormap for positive and negative values
        pos_cmap = LinearSegmentedColormap.from_list('green_gradient', ['#c8e6c9', '#2e7d32'])
        neg_cmap = LinearSegmentedColormap.from_list('red_gradient', ['#ffcdd2', '#c62828'])
        
        # Plot base value
        ax.barh(pos[0], base_value, color='lightgray', alpha=0.8, label='Base value')
        
        # Plot feature contributions
        for i in range(len(sorted_values)):
            value = sorted_values[i]
            if value >= 0:
                # Positive contribution - green
                ax.barh(pos[i+1], value, left=cumulative[i], color='#4CAF50', alpha=0.8)
            else:
                # Negative contribution - red
                ax.barh(pos[i+1], value, left=cumulative[i], color='#F44336', alpha=0.8)
        
        # Add expected value connector lines
        for i in range(len(sorted_values)):
            plt.plot([cumulative[i], cumulative[i]], [pos[i], pos[i+1]], 'k--', alpha=0.3)
        
        # Add final prediction point
        final_prediction = float(final_prediction) if isinstance(final_prediction, (list, np.ndarray)) else float(final_prediction)
        ax.scatter(final_prediction, len(sorted_values), color='navy', s=100, zorder=10, 
                 label='Final prediction')
        
        # Add feature names as y-tick labels
        labels = ['Base value'] + sorted_names
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        
        # Set labels and title
        ax.set_xlabel('Contribution to prediction probability')
        ax.set_title('Feature Contributions to Prediction (Waterfall Chart)')
        
        # Add a grid
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Create legend
        pos_patch = mpatches.Patch(color='#4CAF50', alpha=0.8, label='Increases probability')
        neg_patch = mpatches.Patch(color='#F44336', alpha=0.8, label='Decreases probability')
        base_patch = mpatches.Patch(color='lightgray', alpha=0.8, label='Base value')
        
        ax.legend(handles=[base_patch, pos_patch, neg_patch], 
                 loc='best', frameon=True, framealpha=1)
        
        # Format the plot
        plt.tight_layout()
        
        # Save the figure if output_path is provided and not in a test environment
        if output_path and os.path.isdir(os.path.dirname(output_path)):
            try:
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
            except Exception as e:
                logger.warning(f"Failed to save figure to {output_path}: {e}")
            
        # Convert figure to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close the figure
        plt.close(fig)
        
        return img_str
        
    except Exception as e:
        logger.error(f"Error creating waterfall chart: {str(e)}")
        logger.error(traceback.format_exc())
        return ""

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    """
    Diagnose route to process the form data and generate a prediction.
    """
    if request.method == 'POST':
        try:
            start_time = time.time()
            
            # Ensure the model is loaded
            global model
            if model is None:
                model = load_model()
                if model is None:
                    flash("Le modèle de diagnostic n'a pas pu être chargé. Veuillez contacter l'administrateur.", "danger")
                    logger.error("Model could not be loaded during diagnosis request")
                    return render_template('error.html', 
                                          error_message="Model loading failed", 
                                          error_details="The diagnostic model could not be loaded. Please check server logs for details.")
            
            # Extract form data
            form_data = extract_form_data(request.form)
            logger.info(f"Processed form data: {form_data}")

            # Create feature dataframe for prediction
            features_df = pd.DataFrame([form_data])
            
            # Check for outliers in input data
            try:
                outlier_results = detect_outliers(features_df, method='isolation_forest')
                if outlier_results.get('isolation_forest', {}).get('outliers', [np.array([False])])[0]:
                    logger.warning("Input data contains potential outliers")
                    # We continue processing but record this information
                    session['contains_outliers'] = True
                else:
                    session['contains_outliers'] = False
            except Exception as e:
                logger.warning(f"Outlier detection failed: {str(e)}")
                session['contains_outliers'] = False
            
            # Fill NA values and properly handle data types to avoid errors and warnings
            # First convert all columns to appropriate types to avoid string conversion issues
            for col in features_df.columns:
                # Try to convert to numeric, coerce errors to NaN
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            
            # Now handle NaN values - fill with zeros for all numeric columns
            features_df = features_df.fillna(0)
            
            # Make prediction
            if model is not None:
                try:
                    # Obtenir la prédiction brute du modèle
                    prediction_proba = model.predict_proba(features_df)[0][1]
                    prediction_class = get_risk_class(prediction_proba)
                    probability = round(prediction_proba * 100, 1)
                    
                    # Format probability for display
                    probability = f"{probability:.1f}"
                    
                    # Enregistrement des prédictions pour debugging
                    logger.info(f"Prédiction raw: {prediction_proba}, classe: {prediction_class}, probabilité: {probability}")
                except Exception as e:
                    logger.error(f"Error during model prediction: {str(e)}")
                    logger.error(traceback.format_exc())
                    flash("Erreur lors de la prédiction. Veuillez réessayer.", "danger")
                    return render_template('error.html', 
                                          error_message="Prediction error", 
                                          error_details=f"Error during model prediction: {str(e)}")
            else:
                raise ValueError("Model is not loaded correctly")
            
            # Store patient data in session for AI assistant
            patient_info = {
                'age': form_data.get('age'),
                'gender': 'Masculin' if form_data.get('gender') == 1 else 'Féminin',
                'prediction': prediction_proba,
                'prediction_class': prediction_class,
                'probability': probability,
                'contains_outliers': session.get('contains_outliers', False)
            }
            session['patient_data'] = patient_info
            logger.info(f"Stored patient data in session: {patient_info}")

            # Generate explanation
            try:
                # Ensure explainer is initialized
                if explainer is None:
                    logger.warning("SHAP explainer not initialized, attempting to initialize now")
                    initialize_explainer()
                    if explainer is None:
                        raise ValueError("Failed to initialize SHAP explainer")
                
                # Use the explainer.explain method
                explanation_data = explainer.explain(features_df)
                logger.info(f"Explanation data generated successfully")
                
                # Get SHAP values from explanation
                shap_values_raw = explanation_data['shap_values']
                base_value = explanation_data['base_value']
                
                logger.info(f"SHAP values shape: {shap_values_raw.shape}")
                
                # Get transformed data (the actual values used by the model)
                transformed_df = explainer.get_transformed_data(features_df)
                logger.info(f"Transformed data shape: {transformed_df.shape}")
                
                # Pour la compréhension des probabilités calibrées
                if app.config['USE_CALIBRATED_MODEL']:
                    logger.info("Utilisation des explications pour modèle calibré")
                    # Si c'est déjà un modèle calibré, les valeurs SHAP reflètent déjà la calibration
                    logger.info(f"Base value (expected value) for calibrated model: {base_value}")
                else:
                    logger.info("Prédiction avec modèle standard (non calibré)")
                
                # Get SHAP values for positive class (index 1)
                # shap_values_raw shape is (samples, features, classes)
                positive_class_values = shap_values_raw[0, :, 1] if len(shap_values_raw.shape) == 3 else shap_values_raw[0]
                
                # Create feature contributions
                feature_contributions = []
                
                # Get max absolute value for scaling
                if len(positive_class_values) > 0:
                    max_abs_value = max(abs(value) for value in positive_class_values)
                else:
                    logger.warning("Empty positive_class_values sequence, using default max_abs_value")
                    max_abs_value = 1.0  # Default value when there are no SHAP values
                
                # Get base value (expected value) from the explainer
                logger.info(f"Base value (expected value): {base_value}")
                
                # Create feature contribution data
                for i, feature_name in enumerate(features_df.columns):
                    try:
                        # Safely access SHAP value - if out of bounds during testing, use 0
                        if i < len(positive_class_values):
                            shap_value = positive_class_values[i]
                        else:
                            logger.warning(f"SHAP value index {i} out of bounds, using 0")
                            shap_value = 0
                            
                        original_feature_value = features_df.iloc[0, i]
                        
                        # Get transformed value if available
                        transformed_value = None
                        try:
                            # Safe access to transformed value that works with mocks during testing
                            if hasattr(transformed_df, 'shape') and hasattr(transformed_df.shape, '__getitem__'):
                                if i < transformed_df.shape[1]:
                                    transformed_value = transformed_df.iloc[0, i]
                        except (TypeError, AttributeError, IndexError) as e:
                            logger.debug(f"Could not access transformed value for feature {i}: {str(e)}")
                        
                        # Calculate percentage for display
                        value_percent = min(int(abs(shap_value) / max_abs_value * 100), 100) if max_abs_value > 0 else 0
                        value_percent = max(value_percent, 5) if abs(shap_value) > 0.001 else 0
                        
                        # Format feature name for display
                        display_name = format_feature_name(feature_name)
                        
                        # Format feature value based on type
                        formatted_value = format_feature_value(feature_name, original_feature_value)
                        
                        # Format transformed value if available
                        if transformed_value is not None:
                            try:
                                formatted_transformed = f"{float(transformed_value):.4f}"
                            except:
                                formatted_transformed = str(transformed_value)
                        else:
                            formatted_transformed = "N/A"
                        
                        feature_contributions.append({
                            'name': display_name,
                            'value': float(shap_value),
                            'value_percent': value_percent,
                            'is_positive': bool(shap_value >= 0),
                            'feature_value': original_feature_value,
                            'display_value': formatted_value,
                            'transformed_value': formatted_transformed,
                            'contribution_to_probability': float(shap_value)
                        })
                    except Exception as e:
                        logger.error(f"Error processing feature {feature_name}: {str(e)}")
                        # Skip this feature and continue with others
                        continue
                
                # Sort by absolute value contribution
                feature_contributions.sort(key=lambda x: abs(x['value']), reverse=True)
                
                # Add base value to the results
                base_value_results = {}
                try:
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value_float = float(base_value[0])
                    else:
                        base_value_float = float(base_value)
                    
                    base_value_results = {
                        'base_value': base_value_float,
                        'formatted_base_value': f"{base_value_float:.3f}"
                    }
                except Exception as e:
                    logger.error(f"Error formatting base value: {str(e)}")
                    base_value_results = {
                        'base_value': 0.0,
                        'formatted_base_value': "0.000"
                    }
                
                # Generate SHAP summary plot
                images_dir = os.path.join(os.path.dirname(__file__), 'static', 'images')
                os.makedirs(images_dir, exist_ok=True)
                summary_path = os.path.join(images_dir, 'shap_summary.png')
                beeswarm_path = os.path.join(images_dir, 'shap_beeswarm.png')
                bar_path = os.path.join(images_dir, 'shap_bar.png')
                decision_path = os.path.join(images_dir, 'shap_decision.png')
                heatmap_path = os.path.join(images_dir, 'shap_heatmap.png')
                
                shap_image = None
                beeswarm_image = None
                bar_image = None
                decision_image = None
                heatmap_image = None
                
                try:
                    fig = explainer.plot_summary(features_df, output_path=summary_path)
                    
                    explainer.plot_beeswarm(features_df, output_path=beeswarm_path)
                    
                    explainer.plot_bar(features_df, output_path=bar_path)
                    
                    explainer.plot_decision(features_df, sample_index=0, output_path=decision_path)
                    
                    # For now, just use the current instance repeated a few times for the heatmap
                    repeated_df = pd.concat([features_df] * 5, ignore_index=True)
                    explainer.plot_heatmap(repeated_df, output_path=heatmap_path)
                    
                    if os.path.exists(summary_path):
                        with open(summary_path, 'rb') as img_file:
                            shap_image = base64.b64encode(img_file.read()).decode('utf-8')
                            
                    if os.path.exists(beeswarm_path):
                        with open(beeswarm_path, 'rb') as img_file:
                            beeswarm_image = base64.b64encode(img_file.read()).decode('utf-8')
                            
                    if os.path.exists(bar_path):
                        with open(bar_path, 'rb') as img_file:
                            bar_image = base64.b64encode(img_file.read()).decode('utf-8')
                            
                    if os.path.exists(decision_path):
                        with open(decision_path, 'rb') as img_file:
                            decision_image = base64.b64encode(img_file.read()).decode('utf-8')
                            
                    if os.path.exists(heatmap_path):
                        with open(heatmap_path, 'rb') as img_file:
                            heatmap_image = base64.b64encode(img_file.read()).decode('utf-8')
                            
                except Exception as plot_error:
                    logger.error(f"Error generating SHAP plots: {str(plot_error)}")
                    logger.error(traceback.format_exc())
                    shap_image = None
                    beeswarm_image = None
                    bar_image = None
                    decision_image = None
                    heatmap_image = None
                
                # Generate waterfall chart
                waterfall_image = create_waterfall_chart(
                    base_value=base_value[0] if isinstance(base_value, (list, np.ndarray)) else base_value, 
                    shap_values=positive_class_values, 
                    feature_names=features_df.columns, 
                    final_prediction=prediction_proba
                )
            
            except Exception as e:
                logger.error(f"Error generating SHAP explanation: {str(e)}")
                logger.error(traceback.format_exc())
                feature_contributions = []
                shap_image = None
                beeswarm_image = None
                bar_image = None
                decision_image = None
                heatmap_image = None
                base_value_results = {
                    'base_value': 0.0,
                    'formatted_base_value': "0.000"
                }
                waterfall_image = None
                error_message = f"Could not generate explanations: {str(e)}"
            else:
                error_message = None
            
            # Generate timestamp and report ID
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_id = f"APX-{int(time.time())}"
            
            # Render results template
            return render_template(
                'results.html',
                results={
                    'prediction_probability': prediction_proba,
                    'prediction_class': prediction_class,
                    'probability': probability
                },
                form_data=request.form,
                shap_values=feature_contributions,
                shap_image=shap_image,
                beeswarm_image=beeswarm_image,
                bar_image=bar_image,
                decision_image=decision_image,
                heatmap_image=heatmap_image,
                waterfall_image=waterfall_image,
                timestamp=timestamp,
                report_id=report_id,
                base_value_results=base_value_results,
                error_message=error_message
            )
        except Exception as e:
            logger.error(f"Error processing diagnosis: {str(e)}")
            logger.error(traceback.format_exc())
            return render_template('error.html', error=str(e))
    
    # GET request - show diagnosis form
    return render_template('diagnose.html')

@app.route('/response')
def response():
    """
    Redirect to diagnose route when /response is accessed.
    This handles any incorrect links or submissions to the /response endpoint.
    """
    return redirect(url_for('diagnose'))

@app.route('/shap-image/<path:filename>')
def serve_shap_image(filename):
    """Serve SHAP visualization images from the static directory"""
    directory = os.path.join(app.static_folder, 'shap_images')
    return send_file(os.path.join(directory, filename))

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html', current_year=datetime.now().year)

@app.route('/ai-assistant')
def ai_assistant():
    """Render the AI assistant page."""
    # Check if there's patient data in the session to pass to the template
    patient_data = session.get('patient_data', None)
    return render_template('ai_assistant.html', current_time=datetime.now().strftime('%H:%M'), patient_data=patient_data)

@app.route('/api/ai-assistant', methods=['POST'])
def ai_assistant_api():
    """API endpoint for the AI assistant to process queries."""
    data = request.json
    query = data.get('query', '')
    include_patient_data = data.get('include_patient_data', False)
    
    # Get patient data from session if available and requested
    patient_data = session.get('patient_data', None) if include_patient_data else None
    
    try:
        # Get response from Gemini AI
        response = get_assistant_response(query, patient_data)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in AI assistant: {str(e)}")
        return jsonify({'response': "Je suis désolé, une erreur s'est produite lors du traitement de votre demande. Veuillez réessayer."})

@app.route('/api/explain-features', methods=['POST'])
def explain_features_api():
    """API endpoint to get explanations for prediction features."""
    data = request.json
    features = data.get('features', [])
    
    try:
        explanation = explain_prediction_features(features)
        return jsonify({'explanation': explanation})
    except Exception as e:
        logger.error(f"Error explaining features: {str(e)}")
        return jsonify({'explanation': "Je n'ai pas pu générer une explication pour ces caractéristiques."})

@app.route('/api/clinical-recommendations', methods=['POST'])
def clinical_recommendations_api():
    """API endpoint to get clinical recommendations based on prediction."""
    data = request.json
    prediction = data.get('prediction', 0.0)
    features = data.get('features', [])
    
    try:
        recommendations = get_clinical_recommendations(prediction, features)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({'recommendations': "Je n'ai pas pu générer des recommandations cliniques."})

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """
    Page des paramètres permettant de configurer la clé API Gemini.
    """
    try:
        # Import ici pour éviter les erreurs d'importation circulaire
        import importlib
        import src.ai_assistant.gemini_integration
        importlib.reload(src.ai_assistant.gemini_integration)
        
        # Récupérer les variables après le rechargement du module
        GEMINI_AVAILABLE = src.ai_assistant.gemini_integration.GEMINI_AVAILABLE
        API_KEY = src.ai_assistant.gemini_integration.API_KEY
        MODEL_NAME = getattr(src.ai_assistant.gemini_integration, 'MODEL_NAME', 'Non disponible')
        PROJECT_ROOT = src.ai_assistant.gemini_integration.PROJECT_ROOT
    except Exception as e:
        logger.error(f"Error importing Gemini integration: {str(e)}")
        GEMINI_AVAILABLE = False
        API_KEY = ""
        MODEL_NAME = "Erreur lors du chargement"
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Message de retour pour l'utilisateur
    message = request.args.get('message', '')
    message_type = request.args.get('type', 'info')
    
    # Si méthode POST, traiter les données du formulaire
    if request.method == 'POST':
        api_key = request.form.get('api_key', '').strip()
        storage_method = request.form.get('storage_method', 'env_file')
        
        if not api_key:
            message = "La clé API ne peut pas être vide."
            message_type = "danger"
        else:
            try:
                # Selon la méthode de stockage choisie
                if storage_method == 'env_file':
                    # Créer/modifier le fichier .env à la racine du projet
                    env_path = os.path.join(PROJECT_ROOT, '.env')
                    
                    # Lire le contenu existant s'il existe
                    env_content = {}
                    if os.path.exists(env_path):
                        with open(env_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#') and '=' in line:
                                    key, value = line.split('=', 1)
                                    env_content[key.strip()] = value.strip()
                    
                    # Mettre à jour la clé API
                    env_content['GEMINI_API_KEY'] = api_key
                    
                    # Écrire le fichier .env
                    with open(env_path, 'w') as f:
                        for key, value in env_content.items():
                            f.write(f"{key}={value}\n")
                    
                    message = "Clé API enregistrée avec succès dans le fichier .env"
                    message_type = "success"
                    
                    # Rediriger pour recharger la page et réinitialiser le module d'intégration
                    return redirect(url_for('reload_gemini_integration', message=message, type=message_type))
                
                elif storage_method == 'session':
                    # Stocker la clé dans la session Flask
                    session['gemini_api_key'] = api_key
                    
                    # Mettre à jour directement l'API_KEY dans le module
                    import src.ai_assistant.gemini_integration as gemini_integration
                    os.environ['GEMINI_API_KEY'] = api_key
                    gemini_integration.API_KEY = api_key
                    
                    # Réinitialiser genai avec la nouvelle clé
                    if gemini_integration.GEMINI_AVAILABLE:
                        import google.generativeai as genai
                        genai.configure(api_key=api_key)
                        gemini_integration.GEMINI_AVAILABLE = True
                    
                    message = "Clé API enregistrée pour la session courante"
                    message_type = "success"
            
            except Exception as e:
                message = f"Erreur lors de l'enregistrement de la clé API: {str(e)}"
                message_type = "danger"
                logger.error(f"Error saving API key: {str(e)}")
    
    # Afficher la page avec les informations actuelles
    return render_template(
        'settings.html',
        current_api_key=API_KEY,
        gemini_available=GEMINI_AVAILABLE,
        model_name=MODEL_NAME,
        message=message,
        message_type=message_type
    )

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    """
    Traite le retour utilisateur soumis par le formulaire.
    Enregistre les commentaires et retours pour amélioration continue.
    """
    if request.method == 'POST':
        try:
            # Récupération des données du formulaire
            report_id = request.form.get('report_id')
            prediction_class = request.form.get('prediction_class')
            prediction_probability = request.form.get('prediction_probability')
            diagnostic_accuracy = request.form.get('diagnosticAccuracy')
            actual_diagnosis = request.form.get('actualDiagnosis')
            other_diagnosis = request.form.get('otherDiagnosis')
            usefulness_rating = request.form.get('usefulnessRating')
            comments = request.form.get('comments')
            
            # Création du dictionnaire de feedback
            feedback_data = {
                'report_id': report_id,
                'prediction_class': prediction_class,
                'prediction_probability': prediction_probability,
                'diagnostic_accuracy': diagnostic_accuracy,
                'actual_diagnosis': actual_diagnosis,
                'other_diagnosis': other_diagnosis if actual_diagnosis == 'autre' else '',
                'usefulness_rating': usefulness_rating,
                'comments': comments,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'patient_data': session.get('patient_data', {})
            }
            
            # Logging du feedback pour analyse
            logger.info(f"User feedback received: {feedback_data}")
            
            # Création du répertoire feedback s'il n'existe pas
            feedback_dir = os.path.join(project_root, 'feedback')
            if not os.path.exists(feedback_dir):
                os.makedirs(feedback_dir)
                
            # Enregistrement du feedback dans un fichier JSON
            feedback_file = os.path.join(feedback_dir, f"feedback_{report_id}.json")
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=4)
            
            # Remerciement à l'utilisateur
            flash("Merci pour votre retour ! Vos commentaires nous aideront à améliorer notre système de diagnostic.", "success")
            
            # Redirection vers la page d'accueil
            return redirect(url_for('index'))
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            flash("Une erreur s'est produite lors du traitement de votre retour.", "danger")
            return redirect(url_for('index'))
    
    # Redirect to index if not POST
    return redirect(url_for('index'))

@app.route('/reload_gemini_integration')
def reload_gemini_integration():
    """
    Recharge le module d'intégration Gemini pour prendre en compte les nouvelles configurations.
    """
    try:
        # Réimporter le module pour qu'il prenne en compte les changements dans le fichier .env
        import importlib
        import src.ai_assistant.gemini_integration
        importlib.reload(src.ai_assistant.gemini_integration)
        
        # Récupérer le message de la redirection
        message = request.args.get('message', 'Configuration mise à jour avec succès')
        message_type = request.args.get('type', 'success')
        
        return redirect(url_for('settings', message=message, type=message_type))
    
    except Exception as e:
        logger.error(f"Error reloading Gemini integration: {str(e)}")
        return redirect(url_for('settings', message=f"Erreur lors du rechargement: {str(e)}", type="danger"))

@app.route('/api/test-gemini', methods=['POST'])
def test_gemini_api():
    """
    Endpoint API pour tester une clé API Gemini.
    """
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        prompt = data.get('prompt', 'Bonjour, es-tu fonctionnel?')
        
        if not api_key:
            return jsonify({'success': False, 'error': 'Clé API manquante'})
        
        # Tester la clé API
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Vérifier les modèles disponibles
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            
            # Choisir le modèle approprié
            if "models/gemini-1.5-pro" in available_models:
                model_name = "models/gemini-1.5-pro"
            elif "models/gemini-pro" in available_models:
                model_name = "models/gemini-pro"
            else:
                # Utiliser le premier modèle disponible compatible
                model_name = available_models[0] if available_models else None
                
            if not model_name:
                return jsonify({
                    'success': False,
                    'error': 'Aucun modèle compatible disponible avec cette clé API'
                })
                
            # Créer le modèle et générer la réponse
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            return jsonify({
                'success': True,
                'response': response.text,
                'model_used': model_name
            })
        except Exception as gen_error:
            logger.error(f"Error with Gemini API: {str(gen_error)}")
            return jsonify({
                'success': False,
                'error': f"Erreur d'API Gemini: {str(gen_error)}"
            })
            
    except Exception as e:
        logger.error(f"Error testing Gemini API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', error_code="404", error_message="Page Not Found", error_description="The page you requested could not be found."), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('error.html', error_code=500, error_message="Erreur serveur"), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    logger.info(f"Starting Flask application on port {port} with debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)