"""
Model Visualizer for Pediatric Appendicitis Diagnosis
----------------------------------------------------
This script provides comprehensive visualization tools to understand the model's decision-making process,
including SHAP explanations, feature importance plots, decision boundaries, and calibration curves.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the project's SHAP explainer module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explainability.shap_explainer import ShapExplainer

# Constants
MODEL_PATH = os.path.join('models', 'best_model_retrained.pkl')  # Current app model
TRAINING_DATA_PATH = os.path.join('DATA', 'processed', 'training_data.csv')
OUTPUT_DIR = os.path.join('visualizations')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model_and_data():
    """Load the current model and training data"""
    try:
        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded: {type(model).__name__}")
        
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

def plot_feature_importance(model, X, y, feature_names=None, n_top=15):
    """Plot feature importance using permutation importance"""
    logger.info("Generating feature importance plot...")
    
    # Split data for feature importance calculation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Use permutation importance as it works for all model types
    r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    if feature_names is None:
        feature_names = X.columns
        
    # Sort features by importance
    indices = np.argsort(r.importances_mean)[::-1]
    
    # Select top N features
    indices = indices[:n_top]
    
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importance (Permutation)")
    plt.barh(range(len(indices)), r.importances_mean[indices], color='#1E88E5', 
             align='center', alpha=0.8, yerr=r.importances_std[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.ylim([-1, len(indices)])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved to {os.path.join(OUTPUT_DIR, 'feature_importance.png')}")
    
    # Return the importance values for reporting
    return {feature_names[i]: r.importances_mean[i] for i in indices}

def plot_shap_explanations(model, X, n_samples=100):
    """Generate and save SHAP plots for understanding feature contributions"""
    logger.info("Generating SHAP explanation plots...")
    
    # Sample data for SHAP analysis
    if len(X) > n_samples:
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # Use the project's existing SHAP explainer
    try:
        explainer = ShapExplainer(model)
        shap_values = explainer.generate_shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Dependence plots for top features
        feature_importance = np.abs(shap_values).mean(0)
        top_indices = np.argsort(-feature_importance)[:5]  # Top 5 features
        
        for i in top_indices:
            plt.figure(figsize=(10, 6))
            feature_name = X_sample.columns[i]
            shap.dependence_plot(i, shap_values, X_sample, show=False)
            plt.title(f"SHAP Dependence Plot for {feature_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'shap_dependence_{feature_name}.png'), 
                      dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save a bar plot of SHAP values for a few example cases
        example_cases = X_sample.sample(3, random_state=42)
        for i, (idx, case) in enumerate(example_cases.iterrows()):
            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(explainer.explainer.shap_values(case)[0], max_display=15, show=False)
            plt.title(f"SHAP Explanation for Case {i+1}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'shap_example_case_{i+1}.png'), 
                      dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"SHAP plots saved to {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Error generating SHAP plots: {str(e)}")
        # Fall back to a simpler KernelExplainer
        try:
            logger.info("Falling back to KernelExplainer...")
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            shap_values = explainer.shap_values(X_sample)
            
            # For classification models, shap_values is a list with one element per class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get values for class 1 (appendicitis)
            
            # Summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_kernel.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP summary plot saved to {os.path.join(OUTPUT_DIR, 'shap_summary_kernel.png')}")
        except Exception as inner_e:
            logger.error(f"Error with fallback SHAP analysis: {str(inner_e)}")

def plot_model_performance(model, X, y):
    """Generate various model performance plots"""
    logger.info("Generating model performance plots...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance plots saved to {OUTPUT_DIR}")
    
    # Return metrics dictionary
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist()
    }

def plot_decision_distributions(model, X, y):
    """Plot the distribution of prediction probabilities by actual class"""
    logger.info("Generating prediction distribution plots...")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Create a DataFrame for easier plotting
    results_df = pd.DataFrame({
        'actual': y,
        'predicted_prob': y_pred_proba
    })
    
    # Plot distribution of predictions by actual class
    plt.figure(figsize=(12, 6))
    
    # Plot for negative class (no appendicitis)
    sns.kdeplot(results_df[results_df['actual'] == 0]['predicted_prob'], 
               fill=True, color='green', alpha=0.5, label='No Appendicitis (Actual)')
    
    # Plot for positive class (appendicitis)
    sns.kdeplot(results_df[results_df['actual'] == 1]['predicted_prob'], 
               fill=True, color='red', alpha=0.5, label='Appendicitis (Actual)')
    
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
    plt.xlabel('Predicted Probability of Appendicitis')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities by Actual Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Prediction distribution plot saved to {os.path.join(OUTPUT_DIR, 'prediction_distributions.png')}")

def analyze_feature_correlations(X):
    """Analyze and visualize correlations between features"""
    logger.info("Analyzing feature correlations...")
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", annot_kws={"size": 8})
    
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature correlation plot saved to {os.path.join(OUTPUT_DIR, 'feature_correlations.png')}")
    
    # Identify highly correlated features
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(i, j, corr_matrix.loc[i, j]) 
                 for i in corr_matrix.index 
                 for j in corr_matrix.columns 
                 if abs(corr_matrix.loc[i, j]) > 0.7 and i != j]
    
    if high_corr:
        logger.info("Highly correlated features (|r| > 0.7):")
        for i, j, v in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            logger.info(f"  {i} and {j}: {v:.3f}")
    else:
        logger.info("No highly correlated features found (|r| > 0.7)")

def main():
    """Main function to generate all visualizations"""
    logger.info("Starting model visualization process...")
    
    # Load model and data
    model, X, y = load_model_and_data()
    
    if model is None or X is None or y is None:
        logger.error("Could not load model or data. Exiting.")
        return
    
    # Generate all visualizations
    feature_importances = plot_feature_importance(model, X, y)
    plot_shap_explanations(model, X)
    performance_metrics = plot_model_performance(model, X, y)
    plot_decision_distributions(model, X, y)
    analyze_feature_correlations(X)
    
    # Summarize findings
    logger.info("\n" + "="*50)
    logger.info("Model Visualization Summary")
    logger.info("="*50)
    
    logger.info("\nTop Features by Importance:")
    for feature, importance in list(feature_importances.items())[:5]:
        logger.info(f"  {feature}: {importance:.4f}")
    
    logger.info("\nPerformance Metrics:")
    logger.info(f"  ROC AUC: {performance_metrics['roc_auc']:.4f}")
    logger.info(f"  PR AUC: {performance_metrics['pr_auc']:.4f}")
    
    logger.info("\nConfusion Matrix:")
    cm = performance_metrics['confusion_matrix']
    logger.info(f"  True Negatives: {cm[0][0]}")
    logger.info(f"  False Positives: {cm[0][1]}")
    logger.info(f"  False Negatives: {cm[1][0]}")
    logger.info(f"  True Positives: {cm[1][1]}")
    
    logger.info("\nAll visualizations have been saved to the 'visualizations' directory.")
    logger.info("="*50)

if __name__ == "__main__":
    main()
