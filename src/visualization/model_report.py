"""
Model visualization and reporting module.

This module provides functions to:
1. Generate beautiful visualizations for model comparison
2. Create comprehensive model evaluation reports
3. Visualize feature importance
4. Plot ROC curves, precision-recall curves and confusion matrices
5. Export results to HTML reports
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report,
    precision_recall_curve, average_precision_score
)
import jinja2
import base64
from io import BytesIO

# Set style for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
colors = plt.cm.tab10.colors

def save_plot_to_base64(fig):
    """Save a matplotlib figure to a base64 string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def plot_model_performance_comparison(model_results, save_path=None):
    """
    Plot a comparison of model performance metrics.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and performance metrics as values
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = list(model_results.keys())
    
    # Create dataframe for plotting
    df = pd.DataFrame(columns=['Model', 'Metric', 'Value'])
    
    for model_name in model_names:
        for metric in metrics:
            if metric in model_results[model_name]:
                df = pd.concat([df, pd.DataFrame({
                    'Model': [model_name],
                    'Metric': [metric.upper()],
                    'Value': [model_results[model_name][metric]]
                })])
    
    # Create figure with plotly
    fig = px.bar(
        df, 
        x='Model', 
        y='Value', 
        color='Metric',
        barmode='group',
        title='Model Performance Comparison',
        labels={'Value': 'Score', 'Model': 'Model', 'Metric': 'Metric'},
        height=600,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_layout(
        font=dict(family="Arial", size=14),
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
    
    return fig

def plot_roc_curves(model_results, save_path=None):
    """
    Plot ROC curves for all models.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and results as values
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The figure object
    """
    fig = go.Figure()
    
    for i, model_name in enumerate(model_results.keys()):
        if 'fpr' in model_results[model_name] and 'tpr' in model_results[model_name]:
            fig.add_trace(
                go.Scatter(
                    x=model_results[model_name]['fpr'],
                    y=model_results[model_name]['tpr'],
                    mode='lines',
                    name=f"{model_name} (AUC={model_results[model_name]['auc']:.3f})",
                    line=dict(width=2)
                )
            )
    
    # Add diagonal line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', width=2, color='gray')
        )
    )
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        font=dict(family="Arial", size=14),
        plot_bgcolor='white',
        height=600,
        width=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(range=[0, 1], constrain='domain')
    fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
    
    # Save if path provided
    if save_path:
        fig.write_html(save_path)
    
    return fig

def plot_feature_importance(model_results, top_n=15, save_path=None):
    """
    Plot feature importance for all models.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and feature importance as values
    top_n : int, optional
        Number of top features to display
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    figs : list
        List of figure objects
    """
    figs = []
    
    for model_name, results in model_results.items():
        if 'feature_importance' in results and 'feature_names' in results:
            # Get feature importances and names
            importances = results['feature_importance']
            feature_names = results['feature_names']
            
            # Create dataframe for sorting
            df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            df = df.sort_values('Importance', ascending=False).head(top_n)
            
            # Create figure with plotly
            fig = px.bar(
                df, 
                y='Feature', 
                x='Importance',
                orientation='h',
                title=f'Top {top_n} Feature Importance - {model_name}',
                labels={'Importance': 'Importance', 'Feature': 'Feature'},
                height=600,
                color='Importance',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                font=dict(family="Arial", size=14),
                plot_bgcolor='white',
                yaxis={'categoryorder':'total ascending'},
            )
            
            # Save if path provided
            if save_path:
                # Create model-specific path
                model_specific_path = save_path.replace('.html', f'_{model_name}_features.html')
                fig.write_html(model_specific_path)
            
            figs.append(fig)
    
    return figs

def plot_confusion_matrices(model_results, save_path=None):
    """
    Plot confusion matrices for all models.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and confusion matrices as values
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    figs : list
        List of figure objects
    """
    figs = []
    
    for model_name, results in model_results.items():
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            
            # Create figure with plotly
            fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=['Negative', 'Positive'],
                y=['Negative', 'Positive'],
                title=f'Confusion Matrix - {model_name}',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                font=dict(family="Arial", size=14),
                height=500,
                width=500
            )
            
            # Calculate and display metrics
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            fig.add_annotation(
                x=0.5, y=-0.15,
                text=f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}",
                showarrow=False,
                font=dict(size=14)
            )
            
            # Save if path provided
            if save_path:
                # Create model-specific path
                model_specific_path = save_path.replace('.html', f'_{model_name}_cm.html')
                fig.write_html(model_specific_path)
            
            figs.append(fig)
    
    return figs

def create_model_report(model_results, output_dir):
    """
    Create a comprehensive HTML report of model results.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and results as values
    output_dir : str
        Directory to save the report
    
    Returns:
    --------
    report_path : str
        Path to the generated report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    performance_fig = plot_model_performance_comparison(model_results)
    roc_fig = plot_roc_curves(model_results)
    feature_figs = plot_feature_importance(model_results)
    cm_figs = plot_confusion_matrices(model_results)
    
    # Convert plotly figures to HTML
    performance_html = performance_fig.to_html(full_html=False, include_plotlyjs='cdn')
    roc_html = roc_fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    feature_html = ""
    for fig in feature_figs:
        feature_html += fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    cm_html = ""
    for fig in cm_figs:
        cm_html += fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Create table of model metrics
    metrics_table = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Training Time (s)'])
    
    for model_name, results in model_results.items():
        row = {'Model': model_name}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            if metric in results:
                row[metric.capitalize()] = f"{results[metric]:.4f}"
        
        if 'training_time' in results:
            row['Training Time (s)'] = f"{results['training_time']:.2f}"
        
        metrics_table = pd.concat([metrics_table, pd.DataFrame([row])], ignore_index=True)
    
    metrics_html = metrics_table.to_html(index=False, classes="table table-striped table-hover")
    
    # Create report using template
    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pediatric Appendicitis Model Evaluation Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0;
                padding: 0;
                color: #333;
                background-color: #f8f9fa;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
            }
            .header {
                text-align: center;
                padding: 40px 20px;
                background-color: #343a40;
                color: white;
                margin-bottom: 30px;
            }
            .section {
                background-color: white;
                padding: 30px;
                margin-bottom: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            h2 {
                color: #343a40;
                border-bottom: 2px solid #dee2e6;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            .plot-container {
                margin-top: 20px;
                margin-bottom: 30px;
            }
            .table {
                margin-top: 20px;
            }
            footer {
                text-align: center;
                padding: 20px;
                margin-top: 30px;
                color: #6c757d;
                font-size: 0.9rem;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Pediatric Appendicitis Model Evaluation Report</h1>
            <p>Comprehensive analysis of machine learning models for appendicitis diagnosis</p>
        </div>
        
        <div class="container">
            <div class="section">
                <h2>Model Performance Comparison</h2>
                <p>This section compares the performance of different machine learning models evaluated for the task of appendicitis diagnosis.</p>
                
                <h3>Key Performance Metrics</h3>
                {{ metrics_table }}
                
                <div class="plot-container">
                    <h3>Visual Comparison</h3>
                    {{ performance_plot }}
                </div>
            </div>
            
            <div class="section">
                <h2>ROC Curve Analysis</h2>
                <p>This section shows the Receiver Operating Characteristic (ROC) curves for all models, illustrating the trade-off between sensitivity and specificity.</p>
                
                <div class="plot-container">
                    {{ roc_plot }}
                </div>
            </div>
            
            <div class="section">
                <h2>Feature Importance Analysis</h2>
                <p>This section displays the most important features for each model, helping to understand which clinical indicators have the greatest impact on appendicitis diagnosis.</p>
                
                <div class="plot-container">
                    {{ feature_plots }}
                </div>
            </div>
            
            <div class="section">
                <h2>Confusion Matrices</h2>
                <p>This section presents confusion matrices for each model, showing true positives, true negatives, false positives, and false negatives.</p>
                
                <div class="plot-container">
                    {{ cm_plots }}
                </div>
            </div>
            
            <div class="section">
                <h2>Conclusion</h2>
                <p>Based on the evaluation metrics, the models demonstrate strong performance in diagnosing pediatric appendicitis. 
                The {{ best_model }} model achieves the best overall performance with an F1 score of {{ best_f1 }} and AUC of {{ best_auc }}.
                For clinical deployment, we recommend using the {{ recommended_model }} model due to its balance of accuracy, interpretability, and reliability.</p>
            </div>
        </div>
        
        <footer>
            <p>Pediatric Appendicitis Diagnosis Project | Generated on {{ date }}</p>
        </footer>
    </body>
    </html>
    """
    
    # Find best model
    best_model = max(model_results.items(), key=lambda x: x[1].get('f1', 0))[0]
    best_f1 = f"{model_results[best_model].get('f1', 0):.4f}"
    best_auc = f"{model_results[best_model].get('auc', 0):.4f}"
    
    # Generate context for template
    from datetime import datetime
    context = {
        'metrics_table': metrics_html,
        'performance_plot': performance_html,
        'roc_plot': roc_html,
        'feature_plots': feature_html,
        'cm_plots': cm_html,
        'best_model': best_model,
        'best_f1': best_f1,
        'best_auc': best_auc,
        'recommended_model': best_model,  # Could be different based on other criteria
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Render template
    template = jinja2.Template(template_str)
    html_content = template.render(**context)
    
    # Save report
    report_path = os.path.join(output_dir, 'model_evaluation_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report saved to {report_path}")
    return report_path

if __name__ == '__main__':
    # Example usage
    model_results = {
        'Random Forest': {
            'accuracy': 0.92,
            'precision': 0.90,
            'recall': 0.88,
            'f1': 0.89,
            'auc': 0.95,
            'training_time': 10.5,
            'confusion_matrix': np.array([[45, 5], [6, 44]]),
            'feature_importance': np.array([0.2, 0.15, 0.1, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]),
            'feature_names': ['WBC', 'Neutrophil %', 'CRP', 'Duration', 'Age', 'Migration', 'Rebound', 'Fever', 'Anorexia', 'Nausea', 'Vomiting', 'Gender', 'Pain'],
            'fpr': np.linspace(0, 1, 100),
            'tpr': np.power(np.linspace(0, 1, 100), 0.5)
        },
        'SVM': {
            'accuracy': 0.90,
            'precision': 0.88,
            'recall': 0.85,
            'f1': 0.86,
            'auc': 0.93,
            'training_time': 5.2,
            'confusion_matrix': np.array([[44, 6], [8, 42]]),
            'feature_importance': np.array([0.22, 0.18, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.01, 0.01]),
            'feature_names': ['WBC', 'Neutrophil %', 'CRP', 'Duration', 'Age', 'Migration', 'Rebound', 'Fever', 'Anorexia', 'Nausea', 'Vomiting', 'Gender', 'Pain'],
            'fpr': np.linspace(0, 1, 100),
            'tpr': np.power(np.linspace(0, 1, 100), 0.7)
        },
        'LightGBM': {
            'accuracy': 0.93,
            'precision': 0.91,
            'recall': 0.90,
            'f1': 0.91,
            'auc': 0.96,
            'training_time': 3.1,
            'confusion_matrix': np.array([[46, 4], [5, 45]]),
            'feature_importance': np.array([0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0, 0.0]),
            'feature_names': ['WBC', 'Neutrophil %', 'CRP', 'Duration', 'Age', 'Migration', 'Rebound', 'Fever', 'Anorexia', 'Nausea', 'Vomiting', 'Gender', 'Pain'],
            'fpr': np.linspace(0, 1, 100),
            'tpr': np.power(np.linspace(0, 1, 100), 0.3)
        },
        'CatBoost': {
            'accuracy': 0.94,
            'precision': 0.92,
            'recall': 0.91,
            'f1': 0.92,
            'auc': 0.97,
            'training_time': 8.7,
            'confusion_matrix': np.array([[47, 3], [4, 46]]),
            'feature_importance': np.array([0.23, 0.19, 0.14, 0.11, 0.09, 0.08, 0.06, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0]),
            'feature_names': ['WBC', 'Neutrophil %', 'CRP', 'Duration', 'Age', 'Migration', 'Rebound', 'Fever', 'Anorexia', 'Nausea', 'Vomiting', 'Gender', 'Pain'],
            'fpr': np.linspace(0, 1, 100),
            'tpr': np.power(np.linspace(0, 1, 100), 0.2)
        }
    }
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reports')
    create_model_report(model_results, output_dir)
