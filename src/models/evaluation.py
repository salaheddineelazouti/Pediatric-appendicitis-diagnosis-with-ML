"""
Model evaluation module for the Pediatric Appendicitis Diagnosis project.
This module contains functions for:
- Evaluating model performance with various metrics
- Comparing different models
- Generating evaluation reports
"""

import os
import logging
import logging.config
import yaml
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Setup logging
logging.config.fileConfig(os.path.join(os.path.dirname(__file__), '../config/logging.conf'))
logger = logging.getLogger('modelTraining')

# Load configuration
def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained classifier model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # For ROC-AUC, we need probability predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # If the model doesn't support predict_proba, use decision_function if available
        if hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        else:
            y_pred_proba = y_pred  # Fallback, but ROC-AUC will be less meaningful
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info(f"Model evaluation completed with results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics

def plot_confusion_matrix(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                         output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix for model evaluation.
    
    Args:
        model: Trained classifier model
        X_test: Test features
        y_test: Test labels
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    logger.info("Generating confusion matrix")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save if output_path is provided..
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {output_path}")
    
    return plt.gcf()

def plot_roc_curve(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                  output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curve for model evaluation.
    
    Args:
        model: Trained classifier model
        X_test: Test features
        y_test: Test labels
        output_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    logger.info("Generating ROC curve")
    
    # Get probability predictions..
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # If the model doesn't support predict_proba, use decision_function if available..
        if hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)
        else:
            logger.warning("Model doesn't support probability predictions, ROC curve may be unreliable")
            y_pred_proba = model.predict(X_test)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # Save if output_path is provided..
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {output_path}")
    
    return plt.gcf()

def compare_models(models_dict: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, 
                  output_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models using various metrics.
    
    Args:
        models_dict: Dictionary mapping model names to model objects
        X_test: Test features
        y_test: Test labels
        output_path: Path to save comparison plot
        
    Returns:
        Dictionary mapping model names to performance metrics
    """
    logger.info(f"Comparing {len(models_dict)} models")
    
    results = {}
    
    # Evaluate each model
    for model_name, model in models_dict.items():
        logger.info(f"Evaluating {model_name}")
        results[model_name] = evaluate_model(model, X_test, y_test)
    
    # Create a comparison dataframe
    comparison_df = pd.DataFrame(results).T
    
    # Sort by ROC-AUC (descending)
    comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
    
    # Log results
    logger.info("\nModel comparison results:")
    logger.info(f"\n{comparison_df}")
    
    # Plot comparison
    if output_path:
        plt.figure(figsize=(10, 6))
        
        # Plot ROC-AUC
        comparison_df['roc_auc'].plot(kind='bar', color='skyblue')
        plt.title('Model Comparison: ROC-AUC')
        plt.ylabel('ROC-AUC')
        plt.xlabel('Model')
        plt.ylim(0.5, 1.0)  # ROC-AUC range is typically 0.5-1.0
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {output_path}")
    
    return results

def generate_classification_report(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    """
    Generate a detailed classification report.
    
    Args:
        model: Trained classifier model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        String containing the classification report
    """
    logger.info("Generating classification report")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Generate report
    report = classification_report(y_test, y_pred, output_dict=False)
    
    logger.info(f"\nClassification Report:\n{report}")
    
    return report

def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info("Model loaded successfully")
    
    return model

def perform_full_evaluation(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                           output_dir: str, model_name: str) -> Dict[str, Any]:
    """
    Perform a comprehensive evaluation of a model and save results.
    
    Args:
        model: Trained classifier model
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save evaluation results
        model_name: Name of the model
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"Performing full evaluation for {model_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Generate and save confusion matrix
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(model, X_test, y_test, cm_path)
    
    # Generate and save ROC curve
    roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    plot_roc_curve(model, X_test, y_test, roc_path)
    
    # Generate classification report
    report = generate_classification_report(model, X_test, y_test)
    
    # Save report to file
    report_path = os.path.join(output_dir, f"{model_name}_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Compile all results
    results = {
        'metrics': metrics,
        'confusion_matrix_path': cm_path,
        'roc_curve_path': roc_path,
        'classification_report': report,
        'classification_report_path': report_path
    }
    
    logger.info(f"Full evaluation completed for {model_name}")
    
    return results