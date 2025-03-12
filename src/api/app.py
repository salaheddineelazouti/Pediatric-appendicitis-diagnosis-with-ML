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
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import traceback

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'app.log'))
    ]
)
logger = logging.getLogger('api')

# Initialize Flask app
app = Flask(__name__, 
           static_folder=os.path.join(os.path.dirname(__file__), 'static'),
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Configure app
app.config['SECRET_KEY'] = 'pediatric-appendicitis-diagnosis-key'
app.config['MODEL_PATH'] = os.path.join(project_root, 'models', 'best_model_retrained.pkl')

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
        logger.info(f"Loading model from {model_path}")
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded successfully: {type(model).__name__}")
            return model
        else:
            logger.warning(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    """Render the diagnosis form or process the submitted form."""
    if request.method == 'POST':
        try:
            # Extract form data
            form_data = {}
            for feature_group in [DEMOGRAPHIC_FEATURES, CLINICAL_FEATURES, LABORATORY_FEATURES, SCORING_FEATURES]:
                for feature in feature_group:
                    name = feature['name']
                    # Handle checkboxes - these should be 1 if checked, 0 if not
                    if feature['type'] == 'checkbox':
                        form_data[name] = 1 if request.form.get(name) else 0
                    # Handle gender specifically
                    elif name == 'gender':
                        gender_value = request.form.get(name)
                        # Convert text to integer if needed
                        if gender_value == 'male':
                            form_data[name] = 1
                        elif gender_value == 'female':
                            form_data[name] = 0
                        else:
                            form_data[name] = int(gender_value) if gender_value else None
                    # Handle numeric fields
                    else:
                        value = request.form.get(name)
                        if value and value.strip():
                            form_data[name] = float(value)
                        else:
                            # For optional fields, use None
                            form_data[name] = None
            
            # Log the processed form data for debugging
            logger.info(f"Processed form data: {form_data}")
            
            # Filter out None values for prediction
            prediction_data = {k: v for k, v in form_data.items() if v is not None}
            
            # Check that all required features are present
            required_features = [
                'age', 'gender', 'duration', 'migration', 'anorexia', 'nausea', 'vomiting',
                'right_lower_quadrant_pain', 'fever', 'rebound_tenderness',
                'white_blood_cell_count', 'neutrophil_percentage', 'c_reactive_protein'
            ]
            
            missing_features = [f for f in required_features if f not in prediction_data]
            if missing_features:
                flash(f"Missing required features: {', '.join(missing_features)}", 'danger')
                return render_template('diagnose.html', 
                                      demographic_features=DEMOGRAPHIC_FEATURES,
                                      clinical_features=CLINICAL_FEATURES,
                                      laboratory_features=LABORATORY_FEATURES,
                                      scoring_features=SCORING_FEATURES,
                                      form_data=form_data)
            
            # Perform prediction if model is available
            if model:
                # Convert form data to model input format
                features_df = pd.DataFrame([prediction_data])
                
                # Make sure all required columns are present
                for feature in model.feature_names_in_:
                    if feature not in features_df.columns:
                        features_df[feature] = 0  # Default value for missing features
                
                # Reorder columns to match the training data
                features_df = features_df[list(model.feature_names_in_)]
                
                # Log the final features for debugging
                logger.info(f"Features for prediction: {features_df.columns.tolist()}")
                
                # Make prediction
                prediction_proba = model.predict_proba(features_df)[0][1]
                prediction = 1 if prediction_proba >= 0.5 else 0
                
                # Prepare results
                results = {
                    'probability': round(prediction_proba * 100, 1),
                    'prediction': 'Appendicitis' if prediction == 1 else 'No Appendicitis',
                    'risk_level': 'High' if prediction_proba >= 0.7 else ('Medium' if prediction_proba >= 0.3 else 'Low'),
                    'prediction_probability': round(prediction_proba * 100, 1),  # Pour compatibilité avec le template
                    'prediction_class': 'High' if prediction_proba >= 0.7 else ('Medium' if prediction_proba >= 0.3 else 'Low')  # Pour compatibilité avec le template
                }
                
                return render_template('results.html', results=results, form_data=form_data)
            else:
                # Demo mode - return simulated results
                flash('Model is not available. Running in demo mode with simulated results.', 'warning')
                results = {
                    'probability': 75.5,
                    'prediction': 'Appendicitis',
                    'risk_level': 'High',
                    'prediction_probability': 75.5,  # Pour compatibilité avec le template
                    'prediction_class': 'High'  # Pour compatibilité avec le template
                }
                return render_template('results.html', results=results, form_data=form_data)
        
        except Exception as e:
            logger.error(f"Error processing diagnosis: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f"An error occurred: {str(e)}", 'danger')
            return render_template('error.html', error=str(e))
    
    # GET request - render the form
    return render_template('diagnose.html', 
                          demographic_features=DEMOGRAPHIC_FEATURES,
                          clinical_features=CLINICAL_FEATURES,
                          laboratory_features=LABORATORY_FEATURES,
                          scoring_features=SCORING_FEATURES)

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('error.html', error='Server error'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    logger.info(f"Starting Flask application on port {port} with debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)