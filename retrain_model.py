"""
Script pour ré-entraîner le modèle de diagnostic d'appendicite pédiatrique.

Ce script effectue les étapes suivantes :
1. Charge les données d'entraînement
2. Évalue le modèle actuel
3. Entraîne un nouveau modèle avec des hyperparamètres optimisés
4. Compare les performances et sauvegarde le meilleur modèle
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chemins des fichiers
OLD_MODEL_PATH = os.path.join('models', 'best_model.pkl')
NEW_MODEL_PATH = os.path.join('models', 'best_model_retrained.pkl')
TRAINING_DATA_PATH = os.path.join('DATA', 'processed', 'training_data.csv')

def load_training_data():
    """Charge les données d'entraînement"""
    try:
        if os.path.exists(TRAINING_DATA_PATH):
            df = pd.read_csv(TRAINING_DATA_PATH)
            logger.info(f"Données d'entraînement chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            logger.info(f"Distribution de la variable cible: {df['appendicitis'].value_counts().to_dict()}")
            
            # Afficher les corrélations avec la variable cible
            corr_with_target = df.corr()['appendicitis'].sort_values(ascending=False)
            logger.info(f"Corrélations avec la cible (top 5): {corr_with_target.head().to_dict()}")
            
            # Séparer les features et la cible
            X = df.drop('appendicitis', axis=1)
            y = df['appendicitis']
            
            return X, y
        else:
            logger.error(f"Fichier de données non trouvé: {TRAINING_DATA_PATH}")
            return None, None
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        return None, None

def load_old_model():
    """Charge le modèle existant"""
    try:
        with open(OLD_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Type de modèle chargé: {type(model).__name__}")
        
        # Vérifier si c'est un pipeline
        if hasattr(model, 'named_steps'):
            logger.info(f"Composants du pipeline: {list(model.named_steps.keys())}")
            
            # Si SVM, récupérer les hyperparamètres
            if 'svm' in model.named_steps:
                svm = model.named_steps['svm']
                logger.info(f"Hyperparamètres du SVM actuel: C={svm.C}, gamma={svm.gamma if hasattr(svm, 'gamma') else 'auto'}, kernel={svm.kernel}")
        
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def evaluate_model(model, X, y, model_name="Modèle"):
    """Évalue les performances d'un modèle sur les données fournies"""
    try:
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Si le modèle peut prédire des probabilités
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = 0.0
        
        # Calculer les métriques
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Performance de {model_name}:")
        logger.info(f"  - Accuracy: {acc:.4f}")
        logger.info(f"  - Precision: {precision:.4f}")
        logger.info(f"  - Recall: {recall:.4f}")
        logger.info(f"  - F1-Score: {f1:.4f}")
        logger.info(f"  - AUC: {auc:.4f}")
        logger.info(f"  - Matrice de confusion: \n{cm}")
        
        # Tester sur des cas typiques
        test_typical_cases(model)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du modèle: {e}")
        return None

def train_new_models(X, y):
    """Entraîne de nouveaux modèles optimisés et retourne le meilleur"""
    try:
        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Modèle SVM avec optimisation des hyperparamètres
        logger.info("Entraînement d'un nouveau modèle SVM avec optimisation des hyperparamètres...")
        
        # Définir la grille de recherche pour SVM
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42))
        ])
        
        svm_param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1],
            'svm__kernel': ['rbf', 'linear', 'poly'],
            'svm__class_weight': ['balanced', None]
        }
        
        svm_grid = GridSearchCV(
            svm_pipeline, 
            svm_param_grid, 
            cv=5, 
            scoring='roc_auc', 
            n_jobs=-1, 
            verbose=1
        )
        
        svm_grid.fit(X_train, y_train)
        
        # Récupérer le meilleur modèle SVM
        best_svm = svm_grid.best_estimator_
        logger.info(f"Meilleurs hyperparamètres SVM: {svm_grid.best_params_}")
        logger.info(f"Meilleur score SVM (CV): {svm_grid.best_score_:.4f}")
        
        # Évaluer le meilleur SVM
        logger.info("Évaluation du meilleur modèle SVM...")
        evaluate_model(best_svm, X, y, model_name="Meilleur SVM")
        
        # Également essayer un RandomForest pour comparaison
        logger.info("Entraînement d'un modèle RandomForest pour comparaison...")
        
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42))
        ])
        
        rf_param_grid = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [None, 10, 20],
            'rf__min_samples_split': [2, 5],
            'rf__min_samples_leaf': [1, 2],
            'rf__class_weight': ['balanced', None]
        }
        
        rf_grid = GridSearchCV(
            rf_pipeline, 
            rf_param_grid, 
            cv=5, 
            scoring='roc_auc', 
            n_jobs=-1, 
            verbose=1
        )
        
        rf_grid.fit(X_train, y_train)
        
        # Récupérer le meilleur modèle RF
        best_rf = rf_grid.best_estimator_
        logger.info(f"Meilleurs hyperparamètres RF: {rf_grid.best_params_}")
        logger.info(f"Meilleur score RF (CV): {rf_grid.best_score_:.4f}")
        
        # Évaluer le meilleur RF
        logger.info("Évaluation du meilleur modèle RandomForest...")
        evaluate_model(best_rf, X, y, model_name="Meilleur RandomForest")
        
        # Comparer les performances et retourner le meilleur modèle
        if rf_grid.best_score_ > svm_grid.best_score_:
            logger.info("Le modèle RandomForest est meilleur que le SVM selon le score CV.")
            return best_rf
        else:
            logger.info("Le modèle SVM est meilleur que le RandomForest selon le score CV.")
            return best_svm
    
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement des nouveaux modèles: {e}")
        return None

def test_typical_cases(model):
    """Teste le modèle avec des cas typiques"""
    try:
        # Cas typique d'appendicite
        appendicitis_case = {
            "age": 10.5,
            "gender": 1,  # garçon
            "duration": 24.0,
            "migration": 1,
            "anorexia": 1,
            "nausea": 1,
            "vomiting": 1,
            "right_lower_quadrant_pain": 1,
            "fever": 1,
            "rebound_tenderness": 1,
            "white_blood_cell_count": 16.5,
            "neutrophil_percentage": 85.0,
            "c_reactive_protein": 110.0,
            "pediatric_appendicitis_score": 8.0,
            "alvarado_score": 9.0
        }
        
        # Cas typique sans appendicite
        non_appendicitis_case = {
            "age": 8.0,
            "gender": 0,  # fille
            "duration": 12.0,
            "migration": 0,
            "anorexia": 0,
            "nausea": 1,
            "vomiting": 0,
            "right_lower_quadrant_pain": 1,
            "fever": 0,
            "rebound_tenderness": 0,
            "white_blood_cell_count": 8.5,
            "neutrophil_percentage": 45.0,
            "c_reactive_protein": 20.0,
            "pediatric_appendicitis_score": 3.0,
            "alvarado_score": 2.0
        }
        
        # Convertir en DataFrames
        appendicitis_df = pd.DataFrame([appendicitis_case])
        non_appendicitis_df = pd.DataFrame([non_appendicitis_case])
        
        logger.info("=== Test avec cas typiques ===")
        
        # Test cas d'appendicite
        try:
            # S'assurer que les colonnes sont dans le bon ordre
            if hasattr(model, 'feature_names_in_'):
                for feature in model.feature_names_in_:
                    if feature not in appendicitis_df.columns:
                        appendicitis_df[feature] = 0
                appendicitis_df = appendicitis_df[model.feature_names_in_]
            
            y_pred_proba = model.predict_proba(appendicitis_df)[0][1]
            y_pred = 1 if y_pred_proba >= 0.5 else 0
            
            logger.info("Cas typique d'appendicite:")
            logger.info(f"  - Prédiction: {y_pred} ({'Appendicite' if y_pred == 1 else 'Pas appendicite'})")
            logger.info(f"  - Probabilité: {y_pred_proba:.4f}")
            logger.info(f"  - Attendu: Appendicite (1)")
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction du cas d'appendicite: {e}")
        
        # Test cas sans appendicite
        try:
            # S'assurer que les colonnes sont dans le bon ordre
            if hasattr(model, 'feature_names_in_'):
                for feature in model.feature_names_in_:
                    if feature not in non_appendicitis_df.columns:
                        non_appendicitis_df[feature] = 0
                non_appendicitis_df = non_appendicitis_df[model.feature_names_in_]
            
            y_pred_proba = model.predict_proba(non_appendicitis_df)[0][1]
            y_pred = 1 if y_pred_proba >= 0.5 else 0
            
            logger.info("Cas typique sans appendicite:")
            logger.info(f"  - Prédiction: {y_pred} ({'Appendicite' if y_pred == 1 else 'Pas appendicite'})")
            logger.info(f"  - Probabilité: {y_pred_proba:.4f}")
            logger.info(f"  - Attendu: Pas appendicite (0)")
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction du cas sans appendicite: {e}")
    
    except Exception as e:
        logger.error(f"Erreur lors du test avec cas typiques: {e}")

def save_model(model):
    """Sauvegarde le modèle retrained"""
    try:
        with open(NEW_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Nouveau modèle sauvegardé dans {NEW_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du nouveau modèle: {e}")
        return False

def update_model_in_app():
    """Met à jour le modèle dans l'application Flask"""
    try:
        app_file = os.path.join('src', 'api', 'app.py')
        
        # Lire le fichier
        with open(app_file, 'r') as f:
            content = f.readlines()
        
        # Chercher la ligne où le modèle est chargé
        for i, line in enumerate(content):
            if 'MODEL_PATH' in line and 'best_model' in line:
                # Remplacer avec le nouveau chemin
                content[i] = line.replace('best_model.pkl', 'best_model_retrained.pkl').replace('best_model_calibrated.pkl', 'best_model_retrained.pkl')
                break
        
        # Écrire les changements
        with open(app_file, 'w') as f:
            f.writelines(content)
        
        logger.info(f"Mise à jour du fichier app.py pour utiliser le nouveau modèle retrained")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du fichier app.py: {e}")
        return False

def main():
    """Fonction principale pour ré-entraîner le modèle"""
    print("\n" + "="*70)
    print(" RÉ-ENTRAÎNEMENT DU MODÈLE DE DIAGNOSTIC D'APPENDICITE PÉDIATRIQUE ")
    print("="*70 + "\n")
    
    # Charger les données d'entraînement
    logger.info("Chargement des données d'entraînement...")
    X, y = load_training_data()
    
    if X is None or y is None:
        logger.error("Impossible de continuer sans données d'entraînement valides.")
        return
    
    # Charger le modèle existant et évaluer ses performances
    logger.info("Chargement et évaluation du modèle existant...")
    old_model = load_old_model()
    
    if old_model is not None:
        evaluate_model(old_model, X, y, model_name="Modèle existant")
    
    # Entraîner de nouveaux modèles et sélectionner le meilleur
    logger.info("Entraînement de nouveaux modèles optimisés...")
    best_model = train_new_models(X, y)
    
    if best_model is not None:
        # Sauvegarder le meilleur modèle
        logger.info("Sauvegarde du meilleur modèle...")
        if save_model(best_model):
            # Mettre à jour l'application pour utiliser le nouveau modèle
            logger.info("Mise à jour de l'application pour utiliser le nouveau modèle...")
            update_model_in_app()
            
            logger.info("SUCCÈS: Le modèle a été ré-entraîné avec succès et l'application a été mise à jour.")
            print("\nLe modèle a été ré-entraîné avec succès et l'application a été mise à jour.")
            print("Redémarrez l'application Flask pour utiliser le nouveau modèle.")
        else:
            logger.error("Échec de la sauvegarde du nouveau modèle.")
    else:
        logger.error("Échec de l'entraînement des nouveaux modèles.")
    
    print("\n" + "="*70)
    print(" FIN DU RÉ-ENTRAÎNEMENT ")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
