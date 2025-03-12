"""
Script pour calibrer le modèle SVM afin d'améliorer la qualité des probabilités prédites.

Les SVM ne produisent pas naturellement des probabilités bien calibrées et nécessitent
une calibration spéciale (comme Platt scaling ou calibration isotonique).
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chemins des fichiers
MODEL_PATH = os.path.join('models', 'best_model.pkl')
CALIBRATED_MODEL_PATH = os.path.join('models', 'best_model_calibrated.pkl')
TRAINING_DATA_PATH = os.path.join('DATA', 'processed', 'training_data.csv')

def load_model():
    """Charge le modèle depuis le fichier pkl"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Type de modèle chargé: {type(model).__name__}")
        
        # Vérifier si c'est un pipeline
        if hasattr(model, 'named_steps'):
            logger.info(f"Composants du pipeline: {list(model.named_steps.keys())}")
        
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def load_training_data():
    """Charge les données d'entraînement"""
    try:
        if os.path.exists(TRAINING_DATA_PATH):
            df = pd.read_csv(TRAINING_DATA_PATH)
            logger.info(f"Données d'entraînement chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            logger.info(f"Distribution de la variable cible: {df['appendicitis'].value_counts().to_dict()}")
            
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

def create_calibrated_model(model, X, y):
    """Crée un modèle calibré à partir du modèle existant"""
    try:
        # Diviser les données pour l'entraînement et la validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Si c'est un pipeline avec scaler et svm
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps and 'svm' in model.named_steps:
            logger.info("Création d'un nouveau pipeline avec un SVM calibré...")
            
            # Extraire le scaler existant
            scaler = model.named_steps['scaler']
            
            # Extraire le SVM
            svm = model.named_steps['svm']
            
            # Prétraiter les données d'entraînement avec le scaler existant
            X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Créer un modèle calibré avec les données prétraitées
            calibrated_svm = CalibratedClassifierCV(
                estimator=svm,  # Utilisation de 'estimator' au lieu de 'base_estimator'
                method='sigmoid',  # Utilisation de Platt scaling
                cv='prefit',       # Le modèle est déjà entraîné
                n_jobs=-1
            )
            
            # Entraîner le calibrateur sur les données de validation
            calibrated_svm.fit(X_val_scaled, y_val)
            
            # Créer un nouveau pipeline avec le calibrateur
            calibrated_pipeline = Pipeline([
                ('scaler', scaler),
                ('calibrated_svm', calibrated_svm)
            ])
            
            # Évaluer le modèle original
            y_pred_orig = model.predict(X_val)
            y_prob_orig = model.predict_proba(X_val)[:, 1]
            
            acc_orig = accuracy_score(y_val, y_pred_orig)
            auc_orig = roc_auc_score(y_val, y_prob_orig)
            brier_orig = brier_score_loss(y_val, y_prob_orig)
            
            logger.info(f"Performance du modèle original:")
            logger.info(f"  - Accuracy: {acc_orig:.4f}")
            logger.info(f"  - AUC: {auc_orig:.4f}")
            logger.info(f"  - Brier Score: {brier_orig:.4f} (plus bas = meilleur)")
            
            # Évaluer le modèle calibré
            y_pred_cal = calibrated_pipeline.predict(X_val)
            y_prob_cal = calibrated_pipeline.predict_proba(X_val)[:, 1]
            
            acc_cal = accuracy_score(y_val, y_pred_cal)
            auc_cal = roc_auc_score(y_val, y_prob_cal)
            brier_cal = brier_score_loss(y_val, y_prob_cal)
            
            logger.info(f"Performance du modèle calibré:")
            logger.info(f"  - Accuracy: {acc_cal:.4f}")
            logger.info(f"  - AUC: {auc_cal:.4f}")
            logger.info(f"  - Brier Score: {brier_cal:.4f} (plus bas = meilleur)")
            
            # Comparer les distributions des probabilités
            orig_prob_dist = {
                '0.0-0.2': sum((y_prob_orig >= 0.0) & (y_prob_orig < 0.2)),
                '0.2-0.4': sum((y_prob_orig >= 0.2) & (y_prob_orig < 0.4)),
                '0.4-0.6': sum((y_prob_orig >= 0.4) & (y_prob_orig < 0.6)),
                '0.6-0.8': sum((y_prob_orig >= 0.6) & (y_prob_orig < 0.8)),
                '0.8-1.0': sum((y_prob_orig >= 0.8) & (y_prob_orig <= 1.0)),
            }
            
            cal_prob_dist = {
                '0.0-0.2': sum((y_prob_cal >= 0.0) & (y_prob_cal < 0.2)),
                '0.2-0.4': sum((y_prob_cal >= 0.2) & (y_prob_cal < 0.4)),
                '0.4-0.6': sum((y_prob_cal >= 0.4) & (y_prob_cal < 0.6)),
                '0.6-0.8': sum((y_prob_cal >= 0.6) & (y_prob_cal < 0.8)),
                '0.8-1.0': sum((y_prob_cal >= 0.8) & (y_prob_cal <= 1.0)),
            }
            
            logger.info(f"Distribution des probabilités - Modèle original: {orig_prob_dist}")
            logger.info(f"Distribution des probabilités - Modèle calibré: {cal_prob_dist}")
            
            return calibrated_pipeline
        
        else:
            logger.error("Le modèle n'a pas la structure attendue (pipeline avec scaler et svm)")
            return None
    
    except Exception as e:
        logger.error(f"Erreur lors de la création du modèle calibré: {e}")
        return None

def save_calibrated_model(model):
    """Sauvegarde le modèle calibré"""
    try:
        with open(CALIBRATED_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Modèle calibré sauvegardé dans {CALIBRATED_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modèle calibré: {e}")
        return False

def test_with_custom_data(model):
    """Teste le modèle calibré avec des données personnalisées typiques"""
    try:
        if model is None:
            return
            
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
        
        logger.info("=== Test avec cas typiques sur le modèle calibré ===")
        
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
        logger.error(f"Erreur lors du test avec données personnalisées: {e}")

def update_model_in_app():
    """Met à jour le modèle dans l'application Flask"""
    try:
        app_file = os.path.join('src', 'api', 'app.py')
        
        # Lire le fichier
        with open(app_file, 'r') as f:
            content = f.readlines()
        
        # Chercher la ligne où le modèle est chargé
        for i, line in enumerate(content):
            if 'MODEL_PATH' in line and 'best_model.pkl' in line:
                # Remplacer avec le nouveau chemin
                content[i] = line.replace('best_model.pkl', 'best_model_calibrated.pkl')
                break
        
        # Écrire les changements
        with open(app_file, 'w') as f:
            f.writelines(content)
        
        logger.info(f"Mise à jour du fichier app.py pour utiliser le modèle calibré")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du fichier app.py: {e}")
        return False

def main():
    """Fonction principale pour calibrer le modèle SVM"""
    print("\n" + "="*70)
    print(" CALIBRATION DU MODÈLE SVM POUR AMÉLIORER LES PROBABILITÉS ")
    print("="*70 + "\n")
    
    # Charger le modèle existant
    logger.info("Chargement du modèle existant...")
    model = load_model()
    
    # Charger les données d'entraînement
    logger.info("Chargement des données d'entraînement...")
    X, y = load_training_data()
    
    if model is not None and X is not None and y is not None:
        # Créer et évaluer le modèle calibré
        logger.info("Création et évaluation du modèle calibré...")
        calibrated_model = create_calibrated_model(model, X, y)
        
        if calibrated_model is not None:
            # Tester le modèle calibré
            logger.info("Test du modèle calibré avec des cas typiques...")
            test_with_custom_data(calibrated_model)
            
            # Sauvegarder le modèle calibré
            logger.info("Sauvegarde du modèle calibré...")
            if save_calibrated_model(calibrated_model):
                # Mettre à jour l'application pour utiliser le nouveau modèle
                logger.info("Mise à jour de l'application pour utiliser le modèle calibré...")
                update_model_in_app()
                
                logger.info("SUCCÈS: Le modèle a été calibré avec succès et l'application a été mise à jour.")
                print("\nLe modèle a été calibré avec succès et l'application a été mise à jour.")
                print("Redémarrez l'application Flask pour utiliser le nouveau modèle calibré.")
            else:
                logger.error("Échec de la sauvegarde du modèle calibré.")
        else:
            logger.error("Échec de la création du modèle calibré.")
    else:
        logger.error("Impossible de continuer sans modèle ou données valides.")
    
    print("\n" + "="*70)
    print(" FIN DE LA CALIBRATION ")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
