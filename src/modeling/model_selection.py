"""
Model selection for pediatric appendicitis diagnosis.

This script:
1. Loads the processed dataset
2. Trains and evaluates multiple models:
   - Support Vector Machine (SVM)
   - Random Forest
   - LightGBM
   - CatBoost
3. Compares models using cross-validation
4. Reports performance metrics
5. Analyzes feature importance
6. Saves the best model
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report,
    precision_recall_curve, average_precision_score
)
import lightgbm as lgb
import catboost as cb
import warnings

from src.visualization.model_report import create_model_report

# Ignore warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_processed_data():
    """
    Load the processed datasets.
    """
    print(f"Loading preprocessed data...")
    
    data_dir = os.path.join(project_root, 'DATA', 'processed')
    
    X_train = pd.read_pickle(os.path.join(data_dir, 'X_train.pkl'))
    X_test = pd.read_pickle(os.path.join(data_dir, 'X_test.pkl'))
    y_train = pd.read_pickle(os.path.join(data_dir, 'y_train.pkl'))
    y_test = pd.read_pickle(os.path.join(data_dir, 'y_test.pkl'))
    
    print(f"Loaded data with shapes:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_svm_model(X_train, y_train):
    """
    Train a Support Vector Machine model.
    """
    print("\n" + "="*80)
    print("Training SVM model...")
    print("="*80)
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    
    # Initialize model
    svm = SVC(probability=True, random_state=42)
    
    # Use grid search with cross-validation
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_svm = grid_search.best_estimator_
    
    # Time taken
    time_taken = time.time() - start_time
    print(f"SVM training completed in {time_taken:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (F1): {grid_search.best_score_:.4f}")
    
    return best_svm

def train_random_forest_model(X_train, y_train):
    """
    Train a Random Forest model.
    """
    print("\n" + "="*80)
    print("Training Random Forest model...")
    print("="*80)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # Use grid search with cross-validation
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    # Time taken
    time_taken = time.time() - start_time
    print(f"Random Forest training completed in {time_taken:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (F1): {grid_search.best_score_:.4f}")
    
    return best_rf

def train_lightgbm_model(X_train, y_train):
    """
    Train a LightGBM model.
    """
    print("\n" + "="*80)
    print("Training LightGBM model...")
    print("="*80)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 9],
        'num_leaves': [31, 63, 127],
    }
    
    # Initialize model
    lgbm = lgb.LGBMClassifier(random_state=42)
    
    # Use grid search with cross-validation
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_lgbm = grid_search.best_estimator_
    
    # Time taken
    time_taken = time.time() - start_time
    print(f"LightGBM training completed in {time_taken:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (F1): {grid_search.best_score_:.4f}")
    
    return best_lgbm

def train_catboost_model(X_train, y_train):
    """
    Train a CatBoost model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training labels
        
    Returns:
    --------
    best_catboost : CatBoostClassifier
        Trained CatBoost model with best parameters
    """
    print("\n" + "="*80)
    print("Training CatBoost model...")
    print("="*80)
    
    try:
        # Define parameter grid - Optimisé pour réduire le temps d'entraînement
        param_grid = {
            'iterations': [100, 200],
            'learning_rate': [0.05, 0.1],
            'depth': [6, 8],
            'l2_leaf_reg': [3, 5],
            'border_count': [32],
        }
        
        # Initialize model with silent mode to reduce verbosity
        catboost = cb.CatBoostClassifier(
            random_state=42,
            verbose=0,
            thread_count=-1,  # Utiliser tous les coeurs disponibles
            loss_function='Logloss',
            eval_metric='F1'
        )
        
        # Use grid search with cross-validation
        start_time = time.time()
        grid_search = GridSearchCV(
            estimator=catboost,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_catboost = grid_search.best_estimator_
        
        # Time taken
        time_taken = time.time() - start_time
        print(f"CatBoost training completed in {time_taken:.2f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score (F1): {grid_search.best_score_:.4f}")
        
        return best_catboost
        
    except Exception as e:
        print(f"Error training CatBoost model: {e}")
        print("Falling back to a basic CatBoost model with default parameters")
        
        # Create a simple model with minimal parameters if grid search fails
        try:
            basic_model = cb.CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=0
            )
            basic_model.fit(X_train, y_train)
            return basic_model
        except Exception as e:
            print(f"Could not train even a basic CatBoost model: {e}")
            print("CatBoost might not be properly installed or incompatible with your system.")
            print("Please check if CatBoost is installed: pip install catboost")
            return None

def evaluate_model(model, X_test, y_test, model_name, feature_names=None):
    """
    Evaluate a trained model on the test set.
    """
    print(f"\nEvaluating {model_name} model on test set...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    print(f"Test Metrics for {model_name}:")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall/Sensitivity: {recall:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - ROC AUC: {auc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate and display specificity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print(f"  - Specificity: {specificity:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Appendicitis', 'Appendicitis'],
                yticklabels=['No Appendicitis', 'Appendicitis'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Ensure the directory exists
    figures_dir = os.path.join(project_root, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    plt.plot(recall_curve, precision_curve, lw=2, 
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, f'pr_curve_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    # Plot feature importance if available
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        feature_importance = None
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif model_name == 'SVM' and hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_[0])
        
        if feature_importance is not None:
            # Get feature importances
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            })
            feature_importance_df = feature_importance_df.sort_values(
                by='Importance', ascending=False
            )
            
            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', 
                       data=feature_importance_df.head(20))
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png'))
            plt.close()
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity
    }

def compare_models(model_results):
    """
    Compare the performance of different models.
    """
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    
    # Create DataFrame with results
    results_df = pd.DataFrame(model_results)
    results_df = results_df.set_index('model_name')
    
    # Print results
    print(results_df)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    results_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save comparison plot
    figures_dir = os.path.join(project_root, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'model_comparison.png'))
    plt.close()
    
    # Find the best model
    best_model_idx = results_df['f1'].idxmax()
    print(f"\nBest performing model based on F1 score: {best_model_idx}")
    print(f"F1 Score: {results_df.loc[best_model_idx, 'f1']:.4f}")
    
    return best_model_idx

def save_model(model, model_name):
    """
    Save the trained model to disk.
    """
    print(f"\nSaving {model_name} model...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    return model_path

def main():
    """
    Main function for model selection.
    """
    print("="*80)"
    print("PEDIATRIC APPENDICITIS MODEL SELECTION")
    print("="*80)
    
    # Create directories if they don't exist
    figures_dir = os.path.join(project_root, 'figures')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Get feature names
    feature_names = X_train.columns.tolist()
    
    # Train models
    models = {
        'Support Vector Machine': train_svm_model(X_train, y_train),
        'Random Forest': train_random_forest_model(X_train, y_train),
        'LightGBM': train_lightgbm_model(X_train, y_train),
        'CatBoost': train_catboost_model(X_train, y_train)
    }
    
    # Evaluate models
    model_results = []
    for model_name, model in models.items():
        results = evaluate_model(model, X_test, y_test, model_name, feature_names)
        model_results.append(results)
    
    # Compare models
    best_model_name = compare_models(model_results)
    
    # Save the best model
    best_model = models[best_model_name]
    model_path = save_model(best_model, best_model_name)
    
    print("\nModel selection completed!")
    print(f"The best model is {best_model_name} with the path: {model_path}")

if __name__ == "__main__":
    main()
