"""
Model training module for the Pediatric Appendicitis Diagnosis project.
This module contains functions for training different machine learning models:
- SVM Classifier
- Random Forest Classifier
- LightGBM Classifier
- CatBoost Classifier
"""

import os
import logging
import logging.config
import yaml
import pickle
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import catboost as cb

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

def train_svm(X_train: pd.DataFrame, y_train: pd.Series, 
             param_grid: Optional[Dict[str, Any]] = None) -> Tuple[SVC, Dict[str, float]]:
    """
    Train a Support Vector Machine classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Dictionary of parameters for grid search
        
    Returns:
        Tuple of (trained model, performance metrics)
    """
    logger.info("Training SVM classifier")
    
    start_time = time.time()
    
    # Get default parameters from config
    if param_grid is None:
        param_grid = {
            'C': [config['models']['svm']['C']],
            'kernel': [config['models']['svm']['kernel']],
            'gamma': [config['models']['svm']['gamma']],
            'probability': [config['models']['svm']['probability']],
            'class_weight': [config['models']['svm']['class_weight']]
        }
    
    # Create base model
    svm_model = SVC()
    
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        svm_model, 
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Calculate training metrics
    training_time = time.time() - start_time
    
    # Get cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
    
    metrics = {
        'best_params': grid_search.best_params_,
        'cv_roc_auc': cv_scores.mean(),
        'cv_roc_auc_std': cv_scores.std(),
        'training_time': training_time
    }
    
    logger.info(f"SVM training completed in {training_time:.2f} seconds")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return best_model, metrics

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                       param_grid: Optional[Dict[str, Any]] = None) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Dictionary of parameters for grid search
        
    Returns:
        Tuple of (trained model, performance metrics)
    """
    logger.info("Training Random Forest classifier")
    
    start_time = time.time()
    
    # Get default parameters from config
    if param_grid is None:
        param_grid = {
            'n_estimators': [config['models']['random_forest']['n_estimators']],
            'max_depth': [config['models']['random_forest']['max_depth']],
            'min_samples_split': [config['models']['random_forest']['min_samples_split']],
            'min_samples_leaf': [config['models']['random_forest']['min_samples_leaf']],
            'class_weight': [config['models']['random_forest']['class_weight']],
            'random_state': [config['models']['random_forest']['random_state']]
        }
    
    # Create base model
    rf_model = RandomForestClassifier()
    
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        rf_model, 
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Calculate training metrics
    training_time = time.time() - start_time
    
    # Get cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
    
    metrics = {
        'best_params': grid_search.best_params_,
        'cv_roc_auc': cv_scores.mean(),
        'cv_roc_auc_std': cv_scores.std(),
        'training_time': training_time,
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importances_))
    }
    
    # Sort feature importance
    sorted_features = sorted(
        metrics['feature_importance'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    logger.info(f"Random Forest training completed in {training_time:.2f} seconds")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"Top features: {sorted_features[:5]}")
    
    return best_model, metrics

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, 
                  param_grid: Optional[Dict[str, Any]] = None) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    """
    Train a LightGBM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Dictionary of parameters for grid search
        
    Returns:
        Tuple of (trained model, performance metrics)
    """
    logger.info("Training LightGBM classifier")
    
    start_time = time.time()
    
    # Get default parameters from config
    if param_grid is None:
        param_grid = {
            'n_estimators': [config['models']['lightgbm']['n_estimators']],
            'learning_rate': [config['models']['lightgbm']['learning_rate']],
            'max_depth': [config['models']['lightgbm']['max_depth']],
            'num_leaves': [config['models']['lightgbm']['num_leaves']],
            'class_weight': [config['models']['lightgbm']['class_weight']],
            'random_state': [config['models']['lightgbm']['random_state']]
        }
    
    # Create base model
    lgb_model = lgb.LGBMClassifier()
    
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        lgb_model, 
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Calculate training metrics
    training_time = time.time() - start_time
    
    # Get cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
    
    metrics = {
        'best_params': grid_search.best_params_,
        'cv_roc_auc': cv_scores.mean(),
        'cv_roc_auc_std': cv_scores.std(),
        'training_time': training_time,
        'feature_importance': dict(zip(X_train.columns, best_model.feature_importances_))
    }
    
    # Sort feature importance
    sorted_features = sorted(
        metrics['feature_importance'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    logger.info(f"LightGBM training completed in {training_time:.2f} seconds")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"Top features: {sorted_features[:5]}")
    
    return best_model, metrics

def train_catboost(X_train: pd.DataFrame, y_train: pd.Series, 
                  param_grid: Optional[Dict[str, Any]] = None) -> Tuple[cb.CatBoostClassifier, Dict[str, float]]:
    """
    Train a CatBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Dictionary of parameters for grid search
        
    Returns:
        Tuple of (trained model, performance metrics)
    """
    logger.info("Training CatBoost classifier")
    
    start_time = time.time()
    
    # Get default parameters from config
    if param_grid is None:
        param_grid = {
            'iterations': [config['models']['catboost']['iterations']],
            'learning_rate': [config['models']['catboost']['learning_rate']],
            'depth': [config['models']['catboost']['depth']],
            'loss_function': [config['models']['catboost']['loss_function']],
            'random_seed': [config['models']['catboost']['random_seed']]
        }
    
    # Adjust class weights if needed
    if y_train.value_counts().nunique() > 1:
        # Calculate class weights based on class frequencies
        class_counts = y_train.value_counts()
        class_weights = {
            0: class_counts.sum() / (2 * class_counts[0]),
            1: class_counts.sum() / (2 * class_counts[1])
        }
        param_grid['class_weights'] = [[class_weights[0], class_weights[1]]]
    
    # Create base model
    # Use verbose=False to suppress CatBoost's verbose output
    cat_model = cb.CatBoostClassifier(verbose=False)
    
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        cat_model, 
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Calculate training metrics
    training_time = time.time() - start_time
    
    # Get cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Get feature importance
    feature_importance = best_model.get_feature_importance()
    
    metrics = {
        'best_params': grid_search.best_params_,
        'cv_roc_auc': cv_scores.mean(),
        'cv_roc_auc_std': cv_scores.std(),
        'training_time': training_time,
        'feature_importance': dict(zip(X_train.columns, feature_importance))
    }
    
    # Sort feature importance
    sorted_features = sorted(
        metrics['feature_importance'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    logger.info(f"CatBoost training completed in {training_time:.2f} seconds")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"Top features: {sorted_features[:5]}")
    
    return best_model, metrics

def save_model(model: Any, model_name: str, metrics: Dict[str, Any], output_dir: str) -> str:
    """
    Save a trained model and its metrics to disk.
    
    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Model performance metrics
        output_dir: Directory to save the model
        
    Returns:
        Path to the saved model
    """
    logger.info(f"Saving {model_name} model")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for versioning
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save model..
    model_path = os.path.join(output_dir, f"{model_name}_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics//
    metrics_path = os.path.join(output_dir, f"{model_name}_{timestamp}_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    return model_path

def train_all_models(X_train: pd.DataFrame, y_train: pd.Series, 
                    output_dir: str) -> Dict[str, Tuple[Any, Dict[str, Any], str]]:
    """
    Train all available models and save them.
    
    Args:
        X_train: Training features
        y_train: Training labels
        output_dir: Directory to save models
        
    Returns:
        Dictionary mapping model names to tuples of (model, metrics, saved_path)
    """
    logger.info("Training all models")
    
    models = {}
    
    # Train SVM..
    svm_model, svm_metrics = train_svm(X_train, y_train)
    svm_path = save_model(svm_model, 'svm', svm_metrics, output_dir)
    models['svm'] = (svm_model, svm_metrics, svm_path)
    
    # Train Random Forest..
    rf_model, rf_metrics = train_random_forest(X_train, y_train)
    rf_path = save_model(rf_model, 'random_forest', rf_metrics, output_dir)
    models['random_forest'] = (rf_model, rf_metrics, rf_path)
    
    # Train LightGBM..
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train)
    lgb_path = save_model(lgb_model, 'lightgbm', lgb_metrics, output_dir)
    models['lightgbm'] = (lgb_model, lgb_metrics, lgb_path)
    
    # Train CatBoost
    cat_model, cat_metrics = train_catboost(X_train, y_train)
    cat_path = save_model(cat_model, 'catboost', cat_metrics, output_dir)
    models['catboost'] = (cat_model, cat_metrics, cat_path)
    
    # Find best model based on cross-validation ROC-AUC
    best_model_name = max(
        models.keys(),
        key=lambda x: models[x][1]['cv_roc_auc']
    )
    
    best_model, best_metrics, best_path = models[best_model_name]
    
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best model ROC-AUC: {best_metrics['cv_roc_auc']:.4f}")
    
    # Save a copy of the best model as "best_model.pkl"
    best_model_path = os.path.join(output_dir, "best_model.pkl")
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    logger.info(f"Best model saved to {best_model_path}")
    
    return models

def find_best_model(trained_models: Dict[str, Tuple[Any, Dict[str, Any], str]]) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Find the best model from a dictionary of trained models.
    
    Args:
        trained_models: Dictionary mapping model names to tuples of (model, metrics, saved_path)
        
    Returns:
        Tuple of (best_model_name, best_model, best_metrics)
    """
    logger.info("Finding best model")
    
    best_model_name = max(
        trained_models.keys(),
        key=lambda x: trained_models[x][1]['cv_roc_auc']
    )
    
    best_model, best_metrics, _ = trained_models[best_model_name]
    
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best model ROC-AUC: {best_metrics['cv_roc_auc']:.4f}")
    
    return best_model_name, best_model, best_metrics