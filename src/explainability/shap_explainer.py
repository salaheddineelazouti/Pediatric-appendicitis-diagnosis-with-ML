"""
SHAP-based model explainability module for Pediatric Appendicitis Diagnosis.
This module provides functionality to:
- Generate SHAP values for model predictions
- Create various explanation visualizations
- Extract and interpret feature importance
"""

import os
import logging
import logging.config
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List, Tuple, Optional, Union
import pickle

# Setup logging
logging.config.fileConfig(os.path.join(os.path.dirname(__file__), '../config/logging.conf'))
logger = logging.getLogger('explainability')

# Load configuration
def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

class ShapExplainer:
    """Class for providing SHAP-based explanations for model predictions."""
    
    def __init__(self, model: Any, X_train: pd.DataFrame):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: Trained model to explain
            X_train: Training features used to train the model
        """
        self.model = model
        self.X_train = X_train
        self.explainer = self._initialize_explainer(model, X_train)
        logger.info(f"Initialized SHAP explainer for {type(model).__name__} model")
    
    def _initialize_explainer(self, model: Any, X_train: pd.DataFrame) -> Any:
        """
        Initialize the appropriate SHAP explainer based on model type.
        
        Args:
            model: Trained model
            X_train: Training data
            
        Returns:
            Initialized SHAP explainer
        """
        # Get model type to determine appropriate explainer
        model_type = type(model).__name__
        
        logger.info(f"Initializing SHAP explainer for model type: {model_type}")
        
        # Select explainer based on model type
        if hasattr(model, 'predict_proba'):
            # For tree models (Random Forest, LightGBM, CatBoost)
            if hasattr(model, 'estimators_') or 'LGBMClassifier' in model_type or 'CatBoostClassifier' in model_type:
                try:
                    return shap.TreeExplainer(model)
                except Exception as e:
                    logger.warning(f"Failed to use TreeExplainer, falling back to KernelExplainer. Error: {str(e)}")
            
            # Sample data for kernel explainer (can be memory intensive for large datasets)
            if X_train.shape[0] > 100:
                sample = shap.sample(X_train, 100)
            else:
                sample = X_train
                
            # Use Kernel explainer as fallback
            return shap.KernelExplainer(model.predict_proba, sample)
        
        # For SVM or other models without predict_proba
        if hasattr(model, 'decision_function'):
            # Sample data for kernel explainer
            if X_train.shape[0] > 100:
                sample = shap.sample(X_train, 100)
            else:
                sample = X_train
                
            return shap.KernelExplainer(model.decision_function, sample)
        
        # Fallback for other model types
        if X_train.shape[0] > 100:
            sample = shap.sample(X_train, 100)
        else:
            sample = X_train
            
        return shap.KernelExplainer(model.predict, sample)
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for a set of data points.
        
        Args:
            X: Feature dataset to explain
            
        Returns:
            SHAP values for each feature and each data point
        """
        logger.info(f"Computing SHAP values for {X.shape[0]} instances")
        
        # Check if we should limit the samples for large datasets
        if X.shape[0] > 100 and config.get('explainability', {}).get('limit_samples', True):
            logger.info(f"Limiting to 100 samples for SHAP computation due to computational constraints")
            X_sample = X.sample(100, random_state=42)
        else:
            X_sample = X
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_sample)
        
        logger.info(f"SHAP values computed successfully with shape: {np.array(shap_values).shape}")
        
        return shap_values
    
    def plot_summary(self, X: pd.DataFrame, class_index: int = 1, 
                    max_display: int = 20, output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a summary plot of SHAP values.
        
        Args:
            X: Feature dataset to explain
            class_index: For classification, which class to show (default: 1 for positive class)
            max_display: Maximum number of features to display
            output_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info("Generating SHAP summary plot")
        
        # Sample data if needed
        if X.shape[0] > 100 and config.get('explainability', {}).get('limit_samples', True):
            X_sample = X.sample(100, random_state=42)
        else:
            X_sample = X
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X_sample)
        
        # Handle different output shapes from different explainers
        if isinstance(shap_values, list):
            # For multi-class models, select class_index
            if len(shap_values) > 1:
                plot_values = shap_values[class_index]
            else:
                plot_values = shap_values[0]
        else:
            plot_values = shap_values
        
        # Create plot
        plt.figure(figsize=(10, 12))
        shap.summary_plot(plot_values, X_sample, max_display=max_display, show=False)
        plt.tight_layout()
        
        # Save if output_path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {output_path}")
        
        return plt.gcf()
    
    def plot_dependence(self, X: pd.DataFrame, feature: str, interaction_feature: Optional[str] = None,
                       class_index: int = 1, output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a SHAP dependence plot to show how a feature affects predictions.
        
        Args:
            X: Feature dataset to explain
            feature: Name of the feature to explain
            interaction_feature: Optional feature to use for coloring
            class_index: For classification, which class to show
            output_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Generating SHAP dependence plot for feature: {feature}")
        
        # Sample data if needed
        if X.shape[0] > 100 and config.get('explainability', {}).get('limit_samples', True):
            X_sample = X.sample(100, random_state=42)
        else:
            X_sample = X
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X_sample)
        
        # Handle different output shapes from different explainers
        if isinstance(shap_values, list):
            # For multi-class models, select class_index
            if len(shap_values) > 1:
                plot_values = shap_values[class_index]
            else:
                plot_values = shap_values[0]
        else:
            plot_values = shap_values
        
        # Create plot
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, 
            plot_values, 
            X_sample,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        
        # Save if output_path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP dependence plot saved to {output_path}")
        
        return plt.gcf()
    
    def plot_force(self, X: pd.DataFrame, sample_index: int = 0, 
                  class_index: int = 1, output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a force plot for a single prediction.
        
        Args:
            X: Feature dataset
            sample_index: Index of the sample to explain
            class_index: For classification, which class to show
            output_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Generating SHAP force plot for sample index: {sample_index}")
        
        # Get the sample to explain
        X_sample = X.iloc[[sample_index]]
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X_sample)
        
        # Handle different output shapes from different explainers
        if isinstance(shap_values, list):
            # For multi-class models, select class_index
            if len(shap_values) > 1:
                plot_values = shap_values[class_index][0]
            else:
                plot_values = shap_values[0][0]
        else:
            plot_values = shap_values[0]
        
        # Create figure
        plt.figure(figsize=(12, 3))
        
        # Create force plot
        force_plot = shap.force_plot(
            self.explainer.expected_value[class_index] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
            plot_values, 
            X_sample,
            matplotlib=True,
            show=False
        )
        
        # Save if output_path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP force plot saved to {output_path}")
        
        return plt.gcf()
    
    def get_feature_importance(self, X: pd.DataFrame, class_index: int = 1) -> pd.DataFrame:
        """
        Calculate feature importance based on SHAP values.
        
        Args:
            X: Feature dataset to explain
            class_index: For classification, which class to show
            
        Returns:
            DataFrame with features sorted by importance
        """
        logger.info("Calculating feature importance based on SHAP values")
        
        # Sample data if needed
        if X.shape[0] > 100 and config.get('explainability', {}).get('limit_samples', True):
            X_sample = X.sample(100, random_state=42)
        else:
            X_sample = X
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X_sample)
        
        # Handle different output shapes from different explainers
        if isinstance(shap_values, list):
            # For multi-class models, select class_index
            if len(shap_values) > 1:
                values = shap_values[class_index]
            else:
                values = shap_values[0]
        else:
            values = shap_values
        
        # Calculate mean absolute SHAP value for each feature
        feature_importance = np.abs(values).mean(axis=0)
        
        # Create dataframe with feature names and importance scores
        importance_df = pd.DataFrame({
            'Feature': X_sample.columns,
            'Importance': feature_importance
        })
        
        # Sort by importance (descending)
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        logger.info(f"Feature importance calculated, most important: {importance_df.iloc[0]['Feature']}")
        
        return importance_df
    
    def explain_prediction(self, X: pd.DataFrame, features_to_show: int = 5) -> Dict[str, Any]:
        """
        Provide a detailed explanation for a single prediction.
        
        Args:
            X: Single row DataFrame with features to explain
            features_to_show: Number of top contributing features to include
            
        Returns:
            Dictionary with explanation details
        """
        logger.info("Generating detailed explanation for a prediction")
        
        if X.shape[0] != 1:
            logger.warning(f"Expected single instance, got {X.shape[0]}. Using first instance.")
            X = X.iloc[[0]]
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(X)[0]
            prediction = self.model.predict(X)[0]
            predicted_class = prediction
            confidence = prediction_proba[int(prediction)]
        else:
            prediction = self.model.predict(X)[0]
            predicted_class = prediction
            confidence = None
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X)
        
        # Handle different output shapes from different explainers
        if isinstance(shap_values, list):
            # For multi-class models, select predicted class
            if len(shap_values) > 1:
                values = shap_values[int(predicted_class)][0]
            else:
                values = shap_values[0][0]
        else:
            values = shap_values[0]
        
        # Calculate absolute contributions
        abs_values = np.abs(values)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Value': X.values[0],
            'Contribution': values,
            'AbsContribution': abs_values
        })
        
        # Sort by absolute contribution (descending)
        feature_importance = feature_importance.sort_values('AbsContribution', ascending=False)
        
        # Select top features
        top_features = feature_importance.head(features_to_show)
        
        # Determine if each feature contributes positively or negatively
        top_features['Direction'] = top_features['Contribution'].apply(
            lambda x: 'Increases risk' if x > 0 else 'Decreases risk'
        )
        
        # Format the explanation
        explanation = {
            'prediction': {
                'class': int(predicted_class),
                'label': 'Positive (Appendicitis)' if predicted_class == 1 else 'Negative (No Appendicitis)',
                'confidence': confidence
            },
            'top_features': top_features.to_dict('records'),
            'feature_importance': feature_importance.to_dict('records')
        }
        
        logger.info("Explanation generated successfully")
        
        return explanation
    
    def save_explainer(self, output_path: str) -> None:
        """
        Save the explainer to disk.
        
        Args:
            output_path: Path to save the explainer
        """
        logger.info(f"Saving SHAP explainer to {output_path}")
        
        # Save explainer
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info("SHAP explainer saved successfully")
    
    @staticmethod
    def load_explainer(explainer_path: str) -> 'ShapExplainer':
        """
        Load a saved explainer from disk.
        
        Args:
            explainer_path: Path to the saved explainer
            
        Returns:
            Loaded ShapExplainer object
        """
        logger.info(f"Loading SHAP explainer from {explainer_path}")
        
        with open(explainer_path, 'rb') as f:
            explainer = pickle.load(f)
        
        logger.info("SHAP explainer loaded successfully")
        
        return explainer

def get_explainer_for_model(model: Any, X_train: pd.DataFrame) -> ShapExplainer:
    """
    Factory function to create the appropriate explainer for a model.
    
    Args:
        model: Trained model to explain
        X_train: Training features used to train the model
        
    Returns:
        Initialized ShapExplainer
    """
    return ShapExplainer(model, X_train)

def explain_model(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 output_dir: str, model_name: str) -> Dict[str, Any]:
    """
    Generate a comprehensive set of explanations for a model.
    
    Args:
        model: Trained model to explain
        X_train: Training features
        X_test: Test features for explaining predictions
        output_dir: Directory to save explanations
        model_name: Name of the model for labeling outputs
        
    Returns:
        Dictionary with paths to all generated explanation artifacts
    """
    logger.info(f"Generating comprehensive explanations for {model_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize explainer
    explainer = get_explainer_for_model(model, X_train)
    
    # Generate and save summary plot
    summary_path = os.path.join(output_dir, f"{model_name}_shap_summary.png")
    explainer.plot_summary(X_test, output_path=summary_path)
    
    # Calculate feature importance
    importance_df = explainer.get_feature_importance(X_test)
    importance_path = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    
    # Generate dependence plots for top 3 features
    top_features = importance_df['Feature'].head(3).tolist()
    dependence_paths = []
    
    for feature in top_features:
        dep_path = os.path.join(output_dir, f"{model_name}_dependence_{feature}.png")
        explainer.plot_dependence(X_test, feature, output_path=dep_path)
        dependence_paths.append(dep_path)
    
    # Generate force plot for a sample prediction
    force_path = os.path.join(output_dir, f"{model_name}_force_plot.png")
    explainer.plot_force(X_test, sample_index=0, output_path=force_path)
    
    # Save the explainer
    explainer_path = os.path.join(output_dir, f"{model_name}_explainer.pkl")
    explainer.save_explainer(explainer_path)
    
    # Compile results
    results = {
        'summary_plot_path': summary_path,
        'feature_importance_path': importance_path,
        'dependence_plots': dependence_paths,
        'force_plot_path': force_path,
        'explainer_path': explainer_path
    }
    
    logger.info(f"Comprehensive explanations generated for {model_name}")
    
    return results
