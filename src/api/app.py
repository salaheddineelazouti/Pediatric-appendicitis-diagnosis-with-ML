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
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
import traceback
import io
import uuid
import base64
from datetime import datetime
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import SHAP explainability components
from src.explainability.shap_explainer import ShapExplainer

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

# Create global explainer singleton to avoid recreating for each request
explainer = None
X_sample = None

def initialize_explainer():
    """Initialize SHAP explainer with sample data for background distribution"""
    global explainer, X_sample, model
    
    if model is None:
        logger.warning("Cannot initialize explainer: model not loaded")
        return None
    
    try:
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
    """Extract form data into a dictionary"""
    form_data = {}
    for feature_group in [DEMOGRAPHIC_FEATURES, CLINICAL_FEATURES, LABORATORY_FEATURES, SCORING_FEATURES]:
        for feature in feature_group:
            name = feature['name']
            # Handle checkboxes - these should be 1 if checked, 0 if not
            if feature['type'] == 'checkbox':
                form_data[name] = 1 if form.get(name) else 0
            # Handle gender specifically
            elif name == 'gender':
                gender_value = form.get(name)
                # Convert text to integer if needed
                if gender_value == 'male':
                    form_data[name] = 1
                elif gender_value == 'female':
                    form_data[name] = 0
                else:
                    form_data[name] = int(gender_value) if gender_value else None
            # Handle numeric fields
            else:
                value = form.get(name)
                if value and value.strip():
                    form_data[name] = float(value)
                else:
                    # For optional fields, use None
                    form_data[name] = None
    return form_data

def get_risk_class(probability):
    """Get risk class based on probability"""
    if probability >= 0.7:
        return 'High'
    elif probability >= 0.3:
        return 'Medium'
    else:
        return 'Low'

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
        'right_lower_quadrant_pain': 'RLQ Pain',
        'fever': 'Fever',
        'rebound_tenderness': 'Rebound Tenderness',
        'white_blood_cell_count': 'WBC Count',
        'neutrophil_percentage': 'Neutrophil %',
        'c_reactive_protein': 'CRP Level'
    }
    return display_names.get(feature_name, feature_name)

def format_feature_value(feature_name, value):
    """Format feature value for display"""
    if feature_name in ['white_blood_cell_count']:
        return f"{value:.1f} ×10³/μL"
    elif feature_name in ['neutrophil_percentage']:
        return f"{value:.1f}%"
    elif feature_name in ['c_reactive_protein']:
        return f"{value:.1f} mg/L"
    elif feature_name in ['age']:
        return f"{value:.1f}"
    elif feature_name in ['duration']:
        return f"{value:.1f} hours"
    elif feature_name in ['alvarado_score', 'pediatric_appendicitis_score']:
        return f"{value:.1f}"
    elif feature_name in ['gender', 'migration', 'anorexia', 'nausea', 'vomiting', 
                         'right_lower_quadrant_pain', 'fever', 'rebound_tenderness']:
        return "Yes" if value == 1 else "No"
    else:
        return str(value)

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
            # Extract form data
            form_data = extract_form_data(request.form)
            logger.info(f"Processed form data: {form_data}")

            # Create feature dataframe for prediction
            features_df = pd.DataFrame([form_data])
            
            # Drop scores if they're None
            features_df = features_df.fillna(0)
            
            # Get feature names
            feature_names = list(features_df.columns)
            logger.info(f"Features for prediction: {feature_names}")
            
            # Make prediction
            prediction_proba = model.predict_proba(features_df)[0][1]
            prediction_class = get_risk_class(prediction_proba)
            
            # Format probability for display
            probability = f"{prediction_proba * 100:.1f}"
            
            # Generate explanation
            try:
                # Compute SHAP values
                shap_values_raw = explainer.compute_shap_values(features_df)
                logger.info(f"SHAP values shape: {shap_values_raw.shape}")
                
                # For binary classification, get values for positive class (index 1)
                # shap_values_raw shape is (samples, features, classes)
                positive_class_values = shap_values_raw[0, :, 1]
                
                # Create feature contributions
                feature_contributions = []
                
                # Get max absolute value for scaling
                max_abs_value = max(abs(value) for value in positive_class_values)
                
                # Create feature contribution data
                for i, feature_name in enumerate(feature_names):
                    shap_value = positive_class_values[i]
                    feature_value = features_df.iloc[0, i]
                    
                    # Calculate percentage for display
                    value_percent = min(int(abs(shap_value) / max_abs_value * 100), 100)
                    value_percent = max(value_percent, 5) if abs(shap_value) > 0.001 else 0
                    
                    # Format feature name for display
                    display_name = format_feature_name(feature_name)
                    
                    # Format feature value based on type
                    formatted_value = format_feature_value(feature_name, feature_value)
                    
                    feature_contributions.append({
                        'name': display_name,
                        'value': float(shap_value),
                        'value_percent': value_percent,
                        'is_positive': bool(shap_value >= 0),
                        'feature_value': feature_value,
                        'display_value': formatted_value
                    })
                
                # Sort by absolute value contribution
                feature_contributions.sort(key=lambda x: abs(x['value']), reverse=True)
                
                # Generate SHAP summary plot
                fig = explainer.plot_summary(features_df, output_path='./static/images/shap_summary.png')
                with open('./static/images/shap_summary.png', 'rb') as img_file:
                    shap_image = base64.b64encode(img_file.read()).decode('utf-8')
                
            except Exception as e:
                logger.error(f"Error generating SHAP explanation: {str(e)}")
                logger.error(traceback.format_exc())
                feature_contributions = []
                shap_image = None
            
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
                timestamp=timestamp,
                report_id=report_id
            )
        except Exception as e:
            logger.error(f"Error processing diagnosis: {str(e)}")
            logger.error(traceback.format_exc())
            return render_template('error.html', error=str(e))
    
    # GET request - show diagnosis form
    return render_template('diagnose.html')

@app.route('/shap-image/<path:filename>')
def serve_shap_image(filename):
    """Serve SHAP visualization images from the static directory"""
    directory = os.path.join(app.static_folder, 'shap_images')
    return send_file(os.path.join(directory, filename))

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