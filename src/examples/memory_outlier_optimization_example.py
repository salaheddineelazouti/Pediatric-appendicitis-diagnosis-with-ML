"""
Example script demonstrating the use of advanced outlier detection and memory optimization.
This script shows how to:
1. Load and preprocess data
2. Apply advanced outlier detection
3. Apply enhanced memory optimization
4. Generate visualization reports
5. Integrate with Flask API
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the preprocessing functions
from data_processing.preprocess import (
    load_data, 
    advanced_outlier_detection,
    enhanced_memory_optimization, 
    generate_optimization_report,
    visualize_outlier_impact,
    optimize_pipeline_for_production
)

# Create necessary directories
Path('reports').mkdir(exist_ok=True)
Path('figures/advanced_outliers').mkdir(parents=True, exist_ok=True)

def run_outlier_detection_example():
    """Run example of advanced outlier detection"""
    print("\n=== RUNNING ADVANCED OUTLIER DETECTION EXAMPLE ===")
    
    # Load sample data (update path to your actual data file)
    try:
        # Try to find the data file in common locations
        data_paths = [
            '../data/raw/appendicitis_data.csv',
            'data/raw/appendicitis_data.csv',
            'data/appendicitis_data.csv',
            '../data/appendicitis_data.csv',
        ]
        
        data_file = None
        for path in data_paths:
            if os.path.exists(path):
                data_file = path
                break
        
        if data_file is None:
            # Create synthetic data if no file found
            print("No data file found, creating synthetic data")
            n_samples = 500
            np.random.seed(42)
            
            # Create synthetic data with deliberate outliers
            data = {
                'Age': np.random.normal(10, 3, n_samples),  # Age in years
                'Temperature': np.random.normal(37.5, 0.5, n_samples),  # Temperature in Celsius
                'WBC': np.random.normal(12000, 3000, n_samples),  # White blood cell count
                'CRP': np.random.lognormal(2, 1, n_samples),  # C-reactive protein
                'Pain_Duration': np.random.gamma(2, 5, n_samples),  # Hours of pain
                'Neutrophil_Percent': np.random.normal(70, 10, n_samples)  # Neutrophil percentage
            }
            
            # Add some outliers (5% of the data)
            outlier_idx = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
            data['Age'][outlier_idx] = np.random.uniform(25, 30, len(outlier_idx))
            data['WBC'][outlier_idx] = np.random.uniform(30000, 40000, len(outlier_idx))
            data['CRP'][outlier_idx] = np.random.uniform(200, 300, len(outlier_idx))
            
            # Create target variable (appendicitis diagnosis) - more likely for outliers
            data['Appendicitis'] = np.random.binomial(1, 0.2, n_samples)
            data['Appendicitis'][outlier_idx] = np.random.binomial(1, 0.8, len(outlier_idx))
            
            df = pd.DataFrame(data)
            df.to_csv('data/synthetic_appendicitis_data.csv', index=False)
            data_file = 'data/synthetic_appendicitis_data.csv'
        
        df = pd.read_csv(data_file)
        print(f"Loaded data from {data_file} with shape {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Store original data for comparison
    df_original = df.copy()
    
    # Apply advanced outlier detection
    print("\nApplying advanced outlier detection...")
    df_outliers, outlier_stats = advanced_outlier_detection(
        df,
        contamination=0.05,  # Expected proportion of outliers
        n_neighbors=20,      # For LOF algorithm
        visualize=True,      # Generate visualizations
        robust_pca_enabled=True,
        lof_enabled=True
    )
    
    # Print outlier statistics
    print("\nOutlier Detection Results:")
    if 'total_count' in outlier_stats:
        print(f"Total outliers detected: {outlier_stats['total_count']} "
              f"({outlier_stats['total_percentage']:.2f}% of the data)")
    
    for key, value in outlier_stats.items():
        if key.endswith('_count') and key != 'total_count':
            method = key.replace('_count', '')
            print(f"- {method}: {value} outliers")
    
    # Handle the outliers (e.g., remove them)
    if 'is_outlier' in df_outliers.columns:
        outlier_mask = df_outliers['is_outlier']
        df_cleaned = df_outliers[~outlier_mask].drop(columns=['is_outlier'])
        print(f"\nRemoved {outlier_mask.sum()} outliers. New shape: {df_cleaned.shape}")
    else:
        df_cleaned = df_outliers
        print("\nNo outlier column found in result. Check for errors in outlier detection.")
    
    # Generate visualization report
    print("\nGenerating visualization report...")
    target_column = 'Appendicitis' if 'Appendicitis' in df.columns else None
    if target_column:
        report_path = visualize_outlier_impact(
            df_original, 
            df_cleaned, 
            outlier_stats,
            target_column,
            output_dir='reports'
        )
        print(f"Outlier impact report saved to: {report_path}")
    
    return df_original, df_cleaned, outlier_stats

def run_memory_optimization_example():
    """Run example of memory optimization"""
    print("\n=== RUNNING ENHANCED MEMORY OPTIMIZATION EXAMPLE ===")
    
    # Load or create sample data
    try:
        if os.path.exists('data/synthetic_appendicitis_data.csv'):
            df = pd.read_csv('data/synthetic_appendicitis_data.csv')
        else:
            # Create a larger synthetic dataset to better demonstrate memory optimization
            print("Creating synthetic data for memory optimization")
            n_samples = 10000
            n_features = 50
            
            np.random.seed(42)
            
            # Create core features
            data = {
                'Age': np.random.normal(10, 3, n_samples),
                'Temperature': np.random.normal(37.5, 0.5, n_samples),
                'WBC': np.random.normal(12000, 3000, n_samples),
                'CRP': np.random.lognormal(2, 1, n_samples),
                'Pain_Duration': np.random.gamma(2, 5, n_samples),
                'Neutrophil_Percent': np.random.normal(70, 10, n_samples),
                'Appendicitis': np.random.binomial(1, 0.2, n_samples)
            }
            
            # Add additional features to make the dataset larger
            for i in range(n_features - len(data)):
                feature_type = np.random.choice(['float', 'int', 'categorical', 'sparse'])
                
                if feature_type == 'float':
                    data[f'feature_float_{i}'] = np.random.normal(0, 1, n_samples)
                elif feature_type == 'int':
                    data[f'feature_int_{i}'] = np.random.randint(0, 100, n_samples)
                elif feature_type == 'categorical':
                    categories = ['A', 'B', 'C', 'D', 'E']
                    data[f'feature_cat_{i}'] = np.random.choice(categories, n_samples)
                elif feature_type == 'sparse':
                    # Create a sparse feature (mostly zeros)
                    sparse_data = np.zeros(n_samples)
                    idx = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
                    sparse_data[idx] = np.random.normal(0, 1, len(idx))
                    data[f'feature_sparse_{i}'] = sparse_data
            
            # Add duplicate columns to demonstrate deduplication
            data['Age_copy'] = data['Age'].copy()
            data['WBC_copy'] = data['WBC'].copy()
            
            df = pd.DataFrame(data)
            df.to_csv('data/synthetic_large_appendicitis_data.csv', index=False)
            print(f"Created synthetic dataset with shape {df.shape}")
        
        # Print initial memory usage
        initial_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"\nInitial DataFrame Memory Usage: {initial_memory:.2f} MB")
        print(f"DataFrame shape: {df.shape}")
        print(f"Column types before optimization:")
        for dtype, count in df.dtypes.value_counts().items():
            print(f"  {dtype}: {count} columns")
    
    except Exception as e:
        print(f"Error setting up data for memory optimization: {e}")
        return
    
    # Run enhanced memory optimization
    print("\nRunning enhanced memory optimization...")
    df_optimized, memory_stats = enhanced_memory_optimization(
        df,
        aggressive=True,      # Use aggressive downcasting
        convert_sparse=True,  # Convert sparse columns to sparse format
        deduplicate=True      # Detect and remove duplicate columns
    )
    
    # Print results
    final_memory = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\nOptimized DataFrame Memory Usage: {final_memory:.2f} MB")
    print(f"Memory reduction: {initial_memory - final_memory:.2f} MB "
          f"({(initial_memory / final_memory):.1f}x smaller)")
    
    print(f"\nColumn types after optimization:")
    for dtype, count in df_optimized.dtypes.value_counts().items():
        print(f"  {dtype}: {count} columns")
    
    # Check for removed duplicate columns
    removed_cols = set(df.columns) - set(df_optimized.columns)
    if removed_cols:
        print(f"\nRemoved {len(removed_cols)} duplicate columns: {', '.join(removed_cols)}")
    
    # Generate report
    print("\nGenerating memory optimization report...")
    report_path = generate_optimization_report(memory_stats, output_dir='reports')
    print(f"Memory optimization report saved to: {report_path}")
    
    return df, df_optimized, memory_stats

def run_integrated_pipeline_example():
    """Run example of the integrated pipeline for production"""
    print("\n=== RUNNING INTEGRATED PIPELINE EXAMPLE ===")
    
    # Create a sample patient data point
    patient_data = {
        'Age': 12,
        'Sex': 'M',
        'Temperature': 38.5,
        'WBC': 15000,
        'Neutrophil_Percent': 85,
        'CRP': 30,
        'Pain_Duration': 24,
        'Pain_Migration': 1,
        'RLQ_Tenderness': 1,
        'Rebound_Tenderness': 1,
        'Anorexia': 1
    }
    
    print(f"Sample patient data: {patient_data}")
    
    # Run the production pipeline
    print("\nProcessing using production pipeline...")
    processed_data, is_outlier, stats = optimize_pipeline_for_production(
        patient_data,
        memory_optimize=True,
        outlier_detection=True
    )
    
    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"Outlier warning: {'Yes' if is_outlier else 'No'}")
    
    if is_outlier and 'outliers' in stats:
        print("\nOutlier Details:")
        if 'outlier_features' in stats['outliers']:
            for feature, score in stats['outliers']['outlier_features']:
                print(f"  - {feature}: z-score = {score:.2f}")
        elif 'warning' in stats['outliers']:
            print(f"  {stats['outliers']['warning']}")
    
    if 'memory_optimization' in stats:
        mem_stats = stats['memory_optimization']
        print(f"\nMemory optimization: {mem_stats.get('memory_saved_mb', 0):.4f} MB saved")
    
    print("\nPipeline execution completed")
    
    return processed_data, is_outlier, stats

def run_flask_integration_example():
    """Example of how to integrate with Flask app"""
    print("\n=== FLASK INTEGRATION EXAMPLE (CODE ONLY) ===")
    
    # This is just an example - not actually running the Flask app here
    example_code = """
# In your Flask app.py file:

from data_processing.preprocess import (
    optimize_pipeline_for_production, 
    setup_optimization_for_flask_app
)

# Initialize Flask app
app = Flask(__name__)

# Setup optimization utilities
setup_optimization_for_flask_app(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Get patient data from request
    patient_data = request.json
    
    # Process data with optimization and outlier detection
    processed_data, is_outlier, stats = optimize_pipeline_for_production(
        patient_data,
        memory_optimize=True,
        outlier_detection=True
    )
    
    # Load model and make prediction
    model = joblib.load('models/best_model_retrained.pkl')
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]
    
    # Prepare response
    response = {
        'prediction': int(prediction),
        'probability': float(probability),
        'outlier_warning': is_outlier
    }
    
    # Add outlier details if applicable
    if is_outlier and 'outliers' in stats:
        if 'outlier_features' in stats['outliers']:
            response['outlier_features'] = [
                {'feature': feat, 'score': float(score)} 
                for feat, score in stats['outliers']['outlier_features']
            ]
    
    return jsonify(response)

# Access memory usage statistics
@app.route('/admin/memory-stats')
def memory_stats():
    return render_template('memory_stats.html')
"""
    
    print(example_code)
    
    # Return a dummy Flask app for demonstration
    class DummyFlaskApp:
        def route(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
            
    dummy_app = DummyFlaskApp()
    return dummy_app

def main():
    """Run all examples"""
    print("PEDIATRIC APPENDICITIS DIAGNOSIS - OPTIMIZATION EXAMPLES")
    print("=" * 60)
    
    # Run outlier detection example
    df_original, df_cleaned, outlier_stats = run_outlier_detection_example()
    
    # Run memory optimization example
    df, df_optimized, memory_stats = run_memory_optimization_example()
    
    # Run integrated pipeline example
    processed_data, is_outlier, stats = run_integrated_pipeline_example()
    
    # Show Flask integration example
    dummy_app = run_flask_integration_example()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully.")
    print("Check the 'reports' directory for generated HTML reports.")

if __name__ == "__main__":
    main()
