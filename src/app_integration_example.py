"""
Flask application that demonstrates the integration of advanced outlier detection
and memory optimization with the pediatric appendicitis diagnosis web application.

This example shows how to:
1. Set up Flask routes for advanced data processing
2. Integrate outlier detection and memory optimization with prediction
3. Add admin monitoring endpoints for optimization statistics
4. Display warnings to users when outliers are detected
"""

from flask import Flask, request, jsonify, render_template, Response, make_response
import pandas as pd
import numpy as np
import json
import joblib
import os
import logging
import psutil
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

# Import the preprocessing functions
from data_processing.preprocess import (
    optimize_pipeline_for_production,
    advanced_outlier_detection,
    enhanced_memory_optimization,
    generate_optimization_report,
    visualize_outlier_impact
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AppIntegration')

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join('models', 'best_model_retrained.pkl')

# Configuration
class Config:
    MEMORY_MONITORING_ENABLED = True
    OUTLIER_DETECTION_ENABLED = True
    MEMORY_OPTIMIZATION_ENABLED = True
    ADMIN_PASSWORD = "admin123"  # For demonstration, use a proper auth system in production
    MONITOR_INTERVAL_SECONDS = 60
    
app.config.from_object(Config)

# Track memory stats
memory_stats_history = []

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'rss_mb': memory_info.rss / (1024 * 1024),
        'vms_mb': memory_info.vms / (1024 * 1024)
    }

@app.before_request
def track_memory_usage():
    """Track memory usage before each request"""
    if app.config['MEMORY_MONITORING_ENABLED']:
        memory_stats = get_memory_usage()
        memory_stats_history.append(memory_stats)
        # Keep only the last 1000 readings
        if len(memory_stats_history) > 1000:
            memory_stats_history.pop(0)

def load_model():
    """Load the ML model"""
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            return model
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict appendicitis risk with advanced preprocessing
    
    This endpoint:
    1. Receives patient data
    2. Applies memory optimization (if enabled)
    3. Performs outlier detection (if enabled)
    4. Makes a prediction using the ML model
    5. Returns the prediction with outlier warnings if applicable
    """
    try:
        # Get patient data from request
        if request.is_json:
            patient_data = request.json
        else:
            patient_data = request.form.to_dict()
            
            # Convert string values to appropriate numeric types
            for key in patient_data:
                try:
                    # Try to convert to float or int
                    if '.' in patient_data[key]:
                        patient_data[key] = float(patient_data[key])
                    else:
                        patient_data[key] = int(patient_data[key])
                except ValueError:
                    # Keep as string if conversion fails
                    pass
        
        logger.info(f"Received patient data: {patient_data}")
        
        # Process data with optimization and outlier detection
        processed_data, is_outlier, stats = optimize_pipeline_for_production(
            patient_data,
            memory_optimize=app.config['MEMORY_OPTIMIZATION_ENABLED'],
            outlier_detection=app.config['OUTLIER_DETECTION_ENABLED']
        )
        
        # Load model and make prediction
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model could not be loaded'}), 500
            
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'outlier_warning': is_outlier,
            'risk_level': get_risk_level(probability)
        }
        
        # Add outlier details if applicable
        if is_outlier and 'outliers' in stats:
            response['outlier_details'] = {}
            if 'outlier_features' in stats['outliers']:
                response['outlier_details']['features'] = [
                    {'feature': feat, 'score': float(score)} 
                    for feat, score in stats['outliers']['outlier_features']
                ]
                response['outlier_details']['message'] = (
                    "Some patient values are unusual. Please verify these values "
                    "as they may affect the reliability of the prediction."
                )
            elif 'warning' in stats['outliers']:
                response['outlier_details']['message'] = stats['outliers']['warning']
        
        # Add memory optimization stats if enabled
        if app.config['MEMORY_OPTIMIZATION_ENABLED'] and 'memory_optimization' in stats:
            mem_stats = stats['memory_optimization']
            response['optimization'] = {
                'memory_saved_mb': float(mem_stats.get('memory_saved_mb', 0)),
                'reduction_factor': float(mem_stats.get('reduction_factor', 1))
            }
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.3f}, Outlier: {is_outlier}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.2:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"

@app.route('/api/process-data', methods=['POST'])
def process_data():
    """
    Process data without making a prediction
    
    This endpoint:
    1. Receives patient data
    2. Applies memory optimization and outlier detection
    3. Returns processed data with statistics
    """
    try:
        data = request.json
        
        if 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
            
        input_data = data['data']
        
        # Process with optimization and outlier detection
        processed_data, is_outlier, stats = optimize_pipeline_for_production(
            input_data,
            memory_optimize=True,
            outlier_detection=True
        )
        
        # Convert DataFrame to list for JSON serialization
        processed_data_list = processed_data.to_dict(orient='records')
        
        response = {
            'processed_data': processed_data_list,
            'outlier_detected': is_outlier,
            'statistics': stats
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/admin/memory-stats')
def memory_stats_view():
    """View memory usage statistics"""
    # Simple auth check (replace with proper auth in production)
    if request.args.get('password') != app.config['ADMIN_PASSWORD']:
        return "Unauthorized", 401
        
    current_memory = get_memory_usage()
    
    # Generate memory usage chart
    plt.figure(figsize=(10, 6))
    timestamps = [m['timestamp'] for m in memory_stats_history[-50:]]  # Last 50 records
    rss_values = [m['rss_mb'] for m in memory_stats_history[-50:]]
    
    plt.plot(range(len(timestamps)), rss_values, marker='o')
    plt.title('Memory Usage Over Time')
    plt.ylabel('RSS Memory (MB)')
    plt.xlabel('Request Number')
    plt.grid(True)
    plt.tight_layout()
    
    # Convert plot to base64 for embedding in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    memory_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Memory Usage Statistics</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            .stats-container {{ 
                display: flex; 
                flex-wrap: wrap; 
                gap: 20px; 
                margin-bottom: 30px; 
            }}
            .stat-card {{ 
                background-color: #f7f9fc; 
                border-radius: 8px; 
                padding: 20px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                flex: 1;
                min-width: 200px;
            }}
            .stat-card h2 {{ margin-top: 0; color: #3498db; }}
            .chart-container {{ margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Memory Usage Statistics</h1>
        
        <div class="stats-container">
            <div class="stat-card">
                <h2>Current Memory Usage</h2>
                <p>RSS: <strong>{current_memory['rss_mb']:.2f} MB</strong></p>
                <p>VMS: <strong>{current_memory['vms_mb']:.2f} MB</strong></p>
                <p>Time: {current_memory['timestamp']}</p>
            </div>
            
            <div class="stat-card">
                <h2>Configuration</h2>
                <p>Memory Monitoring: <strong>{'Enabled' if app.config['MEMORY_MONITORING_ENABLED'] else 'Disabled'}</strong></p>
                <p>Memory Optimization: <strong>{'Enabled' if app.config['MEMORY_OPTIMIZATION_ENABLED'] else 'Disabled'}</strong></p>
                <p>Outlier Detection: <strong>{'Enabled' if app.config['OUTLIER_DETECTION_ENABLED'] else 'Disabled'}</strong></p>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>Memory Usage History</h2>
            <img src="data:image/png;base64,{image_base64}" alt="Memory Usage Chart" style="width:100%; max-width:800px;">
        </div>
        
        <h2>Last 10 Memory Readings</h2>
        <table>
            <tr>
                <th>Time</th>
                <th>RSS (MB)</th>
                <th>VMS (MB)</th>
            </tr>
            {''.join([f"<tr><td>{m['timestamp']}</td><td>{m['rss_mb']:.2f}</td><td>{m['vms_mb']:.2f}</td></tr>" for m in memory_stats_history[-10:]])}
        </table>
    </body>
    </html>
    """
    
    return memory_html

@app.route('/admin/optimization-report')
def optimization_report():
    """Generate on-demand optimization report"""
    # Simple auth check (replace with proper auth in production)
    if request.args.get('password') != app.config['ADMIN_PASSWORD']:
        return "Unauthorized", 401
        
    try:
        # Create sample data (in production, you'd use your actual data)
        df = pd.read_csv('data/synthetic_appendicitis_data.csv')
        
        # Apply memory optimization
        df_optimized, memory_stats = enhanced_memory_optimization(
            df,
            aggressive=True,
            convert_sparse=True,
            deduplicate=True
        )
        
        # Generate report
        report_path = generate_optimization_report(memory_stats, output_dir='reports')
        
        # Redirect to static file (or you could embed the HTML directly)
        return f"""
        <html>
        <head>
            <meta http-equiv="refresh" content="0;url=/{report_path}">
        </head>
        <body>
            <p>Redirecting to report...</p>
        </body>
        </html>
        """
    except Exception as e:
        logger.error(f"Error generating optimization report: {str(e)}", exc_info=True)
        return f"Error generating report: {str(e)}", 500

@app.route('/admin/outlier-report')
def outlier_report():
    """Generate on-demand outlier impact report"""
    # Simple auth check (replace with proper auth in production)
    if request.args.get('password') != app.config['ADMIN_PASSWORD']:
        return "Unauthorized", 401
        
    try:
        # Load or create sample data (in production, you'd use your actual data)
        df = pd.read_csv('data/synthetic_appendicitis_data.csv')
        
        # Apply outlier detection
        df_original = df.copy()
        df_outliers, outlier_stats = advanced_outlier_detection(
            df,
            contamination=0.05,
            n_neighbors=20,
            visualize=True,
            robust_pca_enabled=True,
            lof_enabled=True
        )
        
        # Clean the data by removing outliers
        if 'is_outlier' in df_outliers.columns:
            df_cleaned = df_outliers[~df_outliers['is_outlier']].drop(columns=['is_outlier'])
        else:
            df_cleaned = df_outliers
        
        # Generate visualization report
        target_column = 'Appendicitis' if 'Appendicitis' in df.columns else None
        if target_column:
            report_path = visualize_outlier_impact(
                df_original, 
                df_cleaned, 
                outlier_stats,
                target_column,
                output_dir='reports'
            )
            
            # Redirect to static file
            return f"""
            <html>
            <head>
                <meta http-equiv="refresh" content="0;url=/{report_path}">
            </head>
            <body>
                <p>Redirecting to outlier report...</p>
            </body>
            </html>
            """
        else:
            return "No target column found for outlier impact visualization", 400
    except Exception as e:
        logger.error(f"Error generating outlier report: {str(e)}", exc_info=True)
        return f"Error generating outlier report: {str(e)}", 500

@app.route('/toggle-feature', methods=['POST'])
def toggle_feature():
    """Toggle a feature on or off"""
    # Simple auth check (replace with proper auth in production)
    if request.args.get('password') != app.config['ADMIN_PASSWORD']:
        return "Unauthorized", 401
        
    try:
        data = request.json
        feature = data.get('feature')
        enabled = data.get('enabled', False)
        
        if feature == 'memory_monitoring':
            app.config['MEMORY_MONITORING_ENABLED'] = enabled
        elif feature == 'memory_optimization':
            app.config['MEMORY_OPTIMIZATION_ENABLED'] = enabled
        elif feature == 'outlier_detection':
            app.config['OUTLIER_DETECTION_ENABLED'] = enabled
        else:
            return jsonify({'error': f'Unknown feature: {feature}'}), 400
            
        return jsonify({'status': 'success', 'feature': feature, 'enabled': enabled})
    except Exception as e:
        logger.error(f"Error toggling feature: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint that includes memory usage"""
    memory_usage = get_memory_usage()
    
    health_info = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'memory': {
            'rss_mb': memory_usage['rss_mb'],
            'vms_mb': memory_usage['vms_mb']
        },
        'features': {
            'memory_monitoring': app.config['MEMORY_MONITORING_ENABLED'],
            'memory_optimization': app.config['MEMORY_OPTIMIZATION_ENABLED'],
            'outlier_detection': app.config['OUTLIER_DETECTION_ENABLED']
        }
    }
    
    return jsonify(health_info)

if __name__ == "__main__":
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
