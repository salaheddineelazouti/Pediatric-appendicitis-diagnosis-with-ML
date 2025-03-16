"""
Routes and API endpoints for the Pediatric Appendicitis Diagnosis web application.
This module defines all HTTP endpoints and views for the application.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, flash, session
import json
import pickle
import traceback

#Setup logging
logger = logging.getLogger('api')

#Feature lists for the form
DEMOGRAPHIC_FEATURES = [
    {"name": "age", "label": "Age (years)", "type": "number", "min": 1, "max": 18, "step": 0.1, "required": True},
    {"name": "gender", "label": "Gender", "type": "select", "options": [{"value": 1, "label": "Male"}, {"value": 0, "label": "Female"}], "required": True}
]

CLINICAL_FEATURES = [
    {"name": "duration_of_pain_hrs", "label": "Duration of Pain (hours)", "type": "number", "min": 0, "max": 120, "step": 0.5, "required": True},
    {"name": "migration_of_pain", "label": "Migration of Pain", "type": "checkbox", "description": "Pain began periumbilically and migrated to RLQ"},
    {"name": "anorexia", "label": "Anorexia (Loss of appetite)", "type": "checkbox"},
    {"name": "nausea_vomiting", "label": "Nausea or Vomiting", "type": "checkbox"},
    {"name": "right_lower_quadrant_tenderness", "label": "Right Lower Quadrant Tenderness", "type": "checkbox", "description": "Pain on palpation in the right lower quadrant"},
    {"name": "rebound_pain", "label": "Rebound Pain", "type": "checkbox", "description": "Pain when quickly releasing pressure from abdomen"},
    {"name": "fever", "label": "Fever (>38°C/100.4°F)", "type": "checkbox"}
]

LABORATORY_FEATURES = [
    {"name": "wbc_count", "label": "White Blood Cell Count (×10³/μL)", "type": "number", "min": 0, "max": 40, "step": 0.1, "required": True},
    {"name": "neutrophil_percent", "label": "Neutrophil Percentage (%)", "type": "number", "min": 0, "max": 100, "step": 0.1, "required": True},
    {"name": "crp", "label": "C-Reactive Protein (mg/L)", "type": "number", "min": 0, "max": 300, "step": 0.1, "required": True, "description": "Leave empty if not available"}
]

IMAGING_FEATURES = [
    {"name": "us_appendix_diameter_mm", "label": "Appendix Diameter on Ultrasound (mm)", "type": "number", "min": 0, "max": 20, "step": 0.1, "required": False, "description": "Leave empty if not visualized"},
    {"name": "us_appendix_non_compressibility", "label": "Appendix Non-Compressibility", "type": "checkbox", "description": "Non-compressible appendix on ultrasound"},
    {"name": "us_appendix_fecolith", "label": "Fecolith Visualization", "type": "checkbox", "description": "Appendicolith/fecolith seen on imaging"},
    {"name": "us_periappendiceal_fluid", "label": "Periappendiceal Fluid", "type": "checkbox", "description": "Fluid collection around the appendix"}
]

#All features grouped by category
FEATURE_GROUPS = [
    {"name": "demographic", "label": "Demographic Information", "features": DEMOGRAPHIC_FEATURES},
    {"name": "clinical", "label": "Clinical Symptoms & Signs", "features": CLINICAL_FEATURES},
    {"name": "laboratory", "label": "Laboratory Values", "features": LABORATORY_FEATURES},
    {"name": "imaging", "label": "Imaging Findings", "features": IMAGING_FEATURES}
]

#All features flattened
ALL_FEATURES = DEMOGRAPHIC_FEATURES + CLINICAL_FEATURES + LABORATORY_FEATURES + IMAGING_FEATURES

def setup_routes(app: Flask, model: Any, explainer: Any) -> None:
    """
    Setup all routes for the application.
    
    Args:
        app: Flask application
        model: Trained machine learning model
        explainer: SHAP explainer object
    """
    logger.info("Setting up routes for the application")
    
    @app.route('/')
    def index():
        """Render the home page."""
        from datetime import datetime
        return render_template('index.html', current_year=datetime.now().year)
    
    @app.route('/diagnose', methods=['GET', 'POST'])
    def diagnose():
        """Render the diagnosis form and handle form submissions."""
        from datetime import datetime
        
        if request.method == 'POST':
            try:
                # Extract data from form
                form_data = {}
                for feature in ALL_FEATURES:
                    name = feature["name"]
                    
                    if feature["type"] == "checkbox":
                        form_data[name] = 1 if name in request.form else 0
                    elif feature["type"] == "select":
                        form_data[name] = int(request.form.get(name, feature["options"][0]["value"]))
                    else:  # number inputs
                        value = request.form.get(name, '')
                        if value == '':
                            if feature.get("required", False):
                                # If required but empty, set a default value or show an error
                                flash(f"Please provide a value for {feature['label']}", "danger")
                                return render_template('diagnose.html', feature_groups=FEATURE_GROUPS, current_year=datetime.now().year)
                            else:
                                # If not required and empty, set to NaN
                                form_data[name] = np.nan
                        else:
                            form_data[name] = float(value)
                
                # Create a DataFrame with a single row for prediction
                input_data = pd.DataFrame([form_data])
                
                # Store input data in session for result page
                session['input_data'] = form_data
                
                # Redirect to results page
                return redirect(url_for('results'))
                
            except Exception as e:
                logger.error(f"Error processing form data: {str(e)}")
                logger.error(traceback.format_exc())
                flash(f"Error processing form: {str(e)}", "danger")
                return render_template('diagnose.html', feature_groups=FEATURE_GROUPS, current_year=datetime.now().year)
        
        # GET request - display the form
        return render_template('diagnose.html', feature_groups=FEATURE_GROUPS, current_year=datetime.now().year)
    
    @app.route('/results')
    def results():
        """Display the diagnostic results."""
        # Get input data from session
        input_data = session.get('input_data', None)
        
        if not input_data:
            flash("No diagnostic data found. Please fill out the diagnosis form.", "warning")
            return redirect(url_for('diagnose'))
        
        try:
            #Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # If model is available, make a prediction
            if model is not None:
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Get probability for positive class if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_df)[0]
                    probability = probabilities[1]  # Probability of positive class
                else:
                    probability = None
                
                # Get explanation if explainer is available
                if explainer is not None:
                    explanation = explainer.explain_prediction(input_df)
                else:
                    explanation = None
                
                result = {
                    'prediction': int(prediction),
                    'probability': float(probability) if probability is not None else None,
                    'explanation': explanation
                }
            else:
                # Demo mode - generate a mock prediction
                logger.info("Using demo mode to generate results")
                
                # Calculate a mock score based on clinical features
                clinical_score = (
                    (input_data.get('migration_of_pain', 0) * 1) + 
                    (input_data.get('anorexia', 0) * 1) + 
                    (input_data.get('nausea_vomiting', 0) * 1) + 
                    (input_data.get('right_lower_quadrant_tenderness', 0) * 2) + 
                    (input_data.get('rebound_pain', 0) * 1) + 
                    (input_data.get('fever', 0) * 1) + 
                    ((input_data.get('wbc_count', 0) > 10) * 2) + 
                    ((input_data.get('neutrophil_percent', 0) > 75) * 1)
                )
                
                # Mock probability based on clinical score
                probability = min(0.95, max(0.05, clinical_score / 10))
                prediction = 1 if probability > 0.5 else 0
                
                # Mock SHAP values for explanation visualization
                shap_values = []
                
                # Add RLQ tenderness
                rlq_value = input_data.get('right_lower_quadrant_tenderness', 0)
                shap_values.append({
                    'name': 'Right Lower Quadrant Tenderness',
                    'value': 0.3 if rlq_value == 1 else -0.3,
                    'value_percent': 30 if rlq_value == 1 else 30  # Percent for visualization
                })
                
                # Add WBC count
                wbc_value = input_data.get('wbc_count', 0)
                shap_values.append({
                    'name': 'WBC Count',
                    'value': 0.25 if wbc_value > 10 else -0.25,
                    'value_percent': 25 if wbc_value > 10 else 25
                })
                
                # Add neutrophil percentage
                neutrophil_value = input_data.get('neutrophil_percent', 0)
                shap_values.append({
                    'name': 'Neutrophil Percentage',
                    'value': 0.2 if neutrophil_value > 75 else -0.2,
                    'value_percent': 20 if neutrophil_value > 75 else 20
                })
                
                # Add fever
                fever_value = input_data.get('fever', 0)
                shap_values.append({
                    'name': 'Fever',
                    'value': 0.15 if fever_value == 1 else -0.15,
                    'value_percent': 15 if fever_value == 1 else 15
                })
                
                # Add duration of pain
                pain_duration = input_data.get('duration_of_pain_hrs', 0)
                shap_values.append({
                    'name': 'Duration of Pain',
                    'value': 0.1 if pain_duration > 24 else -0.1,
                    'value_percent': 10 if pain_duration > 24 else 10
                })
                
                # Create prediction class based on probability
                if probability >= 0.7:
                    prediction_class = 'High'
                elif probability >= 0.4:
                    prediction_class = 'Medium'
                else:
                    prediction_class = 'Low'
                
                # Format timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Generate a report ID
                import uuid
                report_id = str(uuid.uuid4())[:8].upper()
                
                # Return render template with all required data
                return render_template('results.html',
                                      prediction_probability=probability,
                                      prediction_class=prediction_class,
                                      shap_values=shap_values,
                                      input_data=input_data,
                                      timestamp=timestamp,
                                      report_id=report_id,
                                      current_year=datetime.now().year)
            
            #Render the results page with the prediction and explanation
            return render_template('results.html', 
                                  result=result, 
                                  input_data=input_data, 
                                  feature_groups=FEATURE_GROUPS)
            
        except Exception as e:
            logger.error(f"Error generating results: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f"Error generating diagnostic results: {str(e)}", "danger")
            return redirect(url_for('diagnose'))
    
    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        """API endpoint for making predictions."""
        try:
            # Get data from request
            data = request.json
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Convert to DataFrame
            input_df = pd.DataFrame([data])
            
            # Make prediction if model is available
            if model is not None:
                prediction = int(model.predict(input_df)[0])
                
                if hasattr(model, 'predict_proba'):
                    probability = float(model.predict_proba(input_df)[0][1])
                else:
                    probability = None
                
                #Get explanation if explainer is available
                if explainer is not None:
                    explanation = explainer.explain_prediction(input_df)
                else:
                    explanation = None
                
                result = {
                    'prediction': prediction,
                    'probability': probability,
                    'explanation': explanation
                }
            else:
                # Demo mode -- provide a mock prediction
                #(similar to the results route)
                clinical_score = (
                    (data.get('migration_of_pain', 0) * 1) + 
                    (data.get('anorexia', 0) * 1) + 
                    (data.get('nausea_vomiting', 0) * 1) + 
                    (data.get('right_lower_quadrant_tenderness', 0) * 2) + 
                    (data.get('rebound_pain', 0) * 1) + 
                    (data.get('fever', 0) * 1) + 
                    ((data.get('wbc_count', 0) > 10) * 2) + 
                    ((data.get('neutrophil_percent', 0) > 75) * 1)
                )
                
                probability = min(0.95, max(0.05, clinical_score / 10))
                prediction = 1 if probability > 0.5 else 0
                
                result = {
                    'prediction': prediction,
                    'probability': probability,
                    'demo_mode': True
                }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/static/<path:filename>')
    def static_files(filename):
        """Serve static files."""
        return send_from_directory(app.static_folder, filename)
    
    @app.route('/about')
    def about():
        """Render the about page."""
        from datetime import datetime
        return render_template('about.html', current_year=datetime.now().year)
    
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        from datetime import datetime
        return render_template('error.html', 
                              error_code=404, 
                              error_message="Page Not Found", 
                              error_description="The page you're looking for does not exist.",
                              current_year=datetime.now().year), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        from datetime import datetime
        return render_template('error.html', 
                              error_code=500, 
                              error_message="Server Error", 
                              error_description="Something went wrong on our end. Please try again later.",
                              current_year=datetime.now().year), 500
    
    logger.info("All routes set up successfully")r API response
                shap_data = None
                if explainer:
                    try:
                        shap_values = get_shap_values_for_display(explainer, X)
                        shap_data = [{"name": item["name"], "value": item["value"]} for item in shap_values]
                    except Exception as e:
                        logger.error(f"Error generating SHAP values for API: {str(e)}")
                
                return jsonify({
                    "probability": float(prediction_probability),
                    "risk_class": prediction_class,
                    "shap_values": shap_data
                })
            else:
                # Demo mode
                return jsonify({
                    "probability": 0.75,
                    "risk_class": "High",
                    "note": "Demo mode - no model available"
                })
                
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return jsonify({"error": "Error processing diagnosis"}), 500
    
    @app.route('/static/<path:filename>')
    def static_files(filename):
        """Serve static files."""
        return send_from_directory(app.static_folder, filename)
    
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        from datetime import datetime
        return render_template('error.html', 
                              error_code=404, 
                              error_message="Page Not Found", 
                              error_description="The page you're looking for does not exist.",
                              current_year=datetime.now().year), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        from datetime import datetime
        return render_template('error.html', 
                              error_code=500, 
                              error_message="Server Error", 
                              error_description="Something went wrong on our end. Please try again later.",
                              current_year=datetime.now().year), 500
    
    logger.info("All routes set up successfully")