"""
Script pour calibrer le modèle existant et l'intégrer dans l'application avec SHAP.

Ce script permet de:
1. Charger le modèle existant
2. Calibrer le modèle pour obtenir des probabilités précises
3. Évaluer la qualité de la calibration
4. Intégrer le modèle calibré avec SHAP pour des explications précises
5. Mettre à jour le modèle utilisé par l'application Flask
"""

import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Ajouter le répertoire racine au PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importer les utilitaires de calibration
from src.calibration.calibration_utils import ModelCalibrator, recalibrate_model, integrate_calibration_with_shap

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'calibration.log'), mode='a')
    ]
)
logger = logging.getLogger('calibration_script')

# Chemins de fichiers
ORIGINAL_MODEL_PATH = os.path.join(project_root, 'models', 'best_model_retrained.pkl')
CALIBRATED_MODEL_PATH = os.path.join(project_root, 'models', 'best_model_calibrated.pkl')
MODEL_PATH_FOR_APP = os.path.join(project_root, 'models', 'best_model_simple.pkl')
TRAINING_DATA_PATH = os.path.join(project_root, 'DATA', 'processed', 'training_data.csv')
CALIBRATION_CURVE_PATH = os.path.join(project_root, 'visualizations', 'calibration_curve.png')

def load_data():
    """
    Charge les données d'entraînement et les divise pour la calibration.
    
    Returns:
        Tuple contenant (X_train, y_train, X_val, y_val)
    """
    try:
        if os.path.exists(TRAINING_DATA_PATH):
            df = pd.read_csv(TRAINING_DATA_PATH)
            logger.info(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Vérifier la distribution de la variable cible
            if 'appendicitis' in df.columns:
                target_col = 'appendicitis'
            else:
                # Rechercher une colonne qui pourrait être la cible
                potential_targets = [col for col in df.columns if df[col].nunique() <= 2]
                if potential_targets:
                    target_col = potential_targets[0]
                    logger.warning(f"Colonne cible 'appendicitis' non trouvée, utilisation de '{target_col}' à la place")
                else:
                    raise ValueError("Impossible de déterminer la colonne cible")
            
            logger.info(f"Distribution de la variable cible ({target_col}): {df[target_col].value_counts().to_dict()}")
            
            # Diviser les features et la cible
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            # Diviser en ensembles d'entraînement et de validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            logger.info(f"Données divisées: {X_train.shape[0]} échantillons d'entraînement, {X_val.shape[0]} échantillons de validation")
            
            return X_train, y_train, X_val, y_val
        else:
            logger.error(f"Fichier de données non trouvé: {TRAINING_DATA_PATH}")
            return None, None, None, None
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        return None, None, None, None

def load_model(model_path):
    """
    Charge un modèle depuis un fichier pickle.
    
    Args:
        model_path: Chemin vers le fichier du modèle
        
    Returns:
        Modèle chargé ou None en cas d'erreur
    """
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Modèle chargé depuis {model_path}")
            logger.info(f"Type de modèle: {type(model).__name__}")
            
            # Vérifier si c'est un pipeline
            if hasattr(model, 'named_steps'):
                logger.info(f"Pipeline contenant: {list(model.named_steps.keys())}")
            
            return model
        else:
            logger.error(f"Fichier de modèle non trouvé: {model_path}")
            return None
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None

def calibrate_model_and_visualize(model, X_train, y_train, X_val, y_val):
    """
    Calibre le modèle et visualise les résultats.
    
    Args:
        model: Modèle à calibrer
        X_train: Features d'entraînement
        y_train: Cibles d'entraînement
        X_val: Features de validation
        y_val: Cibles de validation
        
    Returns:
        Modèle calibré
    """
    try:
        logger.info("Démarrage de la calibration du modèle")
        
        # Créer un calibrateur
        calibrator = ModelCalibrator(model, method='sigmoid', cv=5)
        
        # Calibrer le modèle en utilisant les données de validation
        calibrator.fit(X_train, y_train, X_val, y_val)
        
        # Évaluer la calibration
        eval_results = calibrator.evaluate_calibration(X_val, y_val)
        
        # Afficher les résultats
        logger.info(f"Amélioration du Brier score: {eval_results['improvement']:.2f}%")
        logger.info(f"Brier score original: {eval_results['brier_score_original']:.4f}")
        logger.info(f"Brier score calibré: {eval_results['brier_score_calibrated']:.4f}")
        logger.info(f"Distribution des probabilités originales: {eval_results['prob_distribution_original']}")
        logger.info(f"Distribution des probabilités calibrées: {eval_results['prob_distribution_calibrated']}")
        
        # Créer et sauvegarder la courbe de calibration
        fig = calibrator.plot_calibration_curve(X_val, y_val, output_path=CALIBRATION_CURVE_PATH)
        logger.info(f"Courbe de calibration sauvegardée dans {CALIBRATION_CURVE_PATH}")
        
        # Sauvegarder le modèle calibré
        calibrator.save(CALIBRATED_MODEL_PATH)
        logger.info(f"Modèle calibré sauvegardé dans {CALIBRATED_MODEL_PATH}")
        
        return calibrator.calibrated_model
    
    except Exception as e:
        logger.error(f"Erreur lors de la calibration du modèle: {str(e)}")
        return None

def test_calibrated_model(model, calibrated_model, X_val, y_val):
    """
    Teste le modèle calibré et compare ses prédictions avec le modèle original.
    
    Args:
        model: Modèle original
        calibrated_model: Modèle calibré
        X_val: Données de validation
        y_val: Cibles de validation
    """
    try:
        logger.info("Test des prédictions du modèle calibré vs. original")
        
        # Prédire avec les deux modèles
        y_prob_orig = model.predict_proba(X_val)[:, 1]
        y_prob_cal = calibrated_model.predict_proba(X_val)[:, 1]
        
        # Comparer quelques prédictions
        comparison = pd.DataFrame({
            'true_label': y_val.values,
            'orig_prob': y_prob_orig,
            'cal_prob': y_prob_cal,
            'diff': np.abs(y_prob_cal - y_prob_orig)
        })
        
        # Calculer les statistiques de différence
        mean_diff = comparison['diff'].mean()
        max_diff = comparison['diff'].max()
        
        logger.info(f"Différence moyenne entre les probabilités: {mean_diff:.4f}")
        logger.info(f"Différence maximale entre les probabilités: {max_diff:.4f}")
        
        # Afficher les exemples avec les plus grandes différences
        top_diff = comparison.sort_values('diff', ascending=False).head(5)
        logger.info("Top 5 des exemples avec les plus grandes différences:")
        for idx, row in top_diff.iterrows():
            logger.info(f"  Vrai: {row['true_label']}, Orig: {row['orig_prob']:.4f}, Cal: {row['cal_prob']:.4f}, Diff: {row['diff']:.4f}")
        
        # Test sur des cas typiques
        test_cases = create_test_cases()
        
        # Préparer les cas de test comme DataFrame
        test_df = pd.DataFrame(test_cases)
        
        # Prédire avec les deux modèles
        test_prob_orig = model.predict_proba(test_df)[:, 1]
        test_prob_cal = calibrated_model.predict_proba(test_df)[:, 1]
        
        # Afficher les résultats
        logger.info("Prédictions sur des cas typiques:")
        for i, case in enumerate(test_cases):
            logger.info(f"Cas {i+1}: Orig: {test_prob_orig[i]:.4f}, Cal: {test_prob_cal[i]:.4f}")
        
    except Exception as e:
        logger.error(f"Erreur lors du test du modèle calibré: {str(e)}")

def create_test_cases():
    """
    Crée des cas de test typiques pour évaluer le modèle.
    
    Returns:
        Liste de dictionnaires contenant des cas de test
    """
    # Cas typique d'appendicite
    high_risk_case = {
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
        "pediatric_appendicitis_score": 8,
        "alvarado_score": 9
    }
    
    # Cas à risque modéré
    moderate_risk_case = {
        "age": 9.0,
        "gender": 0,  # fille
        "duration": 18.0,
        "migration": 1,
        "anorexia": 1,
        "nausea": 1,
        "vomiting": 0,
        "right_lower_quadrant_pain": 1,
        "fever": 0,
        "rebound_tenderness": 1,
        "white_blood_cell_count": 12.5,
        "neutrophil_percentage": 75.0,
        "c_reactive_protein": 60.0,
        "pediatric_appendicitis_score": 6,
        "alvarado_score": 7
    }
    
    # Cas à faible risque
    low_risk_case = {
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
        "white_blood_cell_count": 9.0,
        "neutrophil_percentage": 65.0,
        "c_reactive_protein": 20.0,
        "pediatric_appendicitis_score": 3,
        "alvarado_score": 4
    }
    
    return [high_risk_case, moderate_risk_case, low_risk_case]

def integrate_with_shap(calibrated_model, X_val):
    """
    Intègre le modèle calibré avec SHAP pour une explication précise des prédictions.
    
    Args:
        calibrated_model: Modèle calibré
        X_val: Données de validation
        
    Returns:
        True si l'intégration est réussie, False sinon
    """
    try:
        logger.info("Intégration du modèle calibré avec SHAP")
        
        # Prendre un exemple pour tester
        sample = X_val.iloc[[0]]
        
        # Intégrer avec SHAP
        result = integrate_calibration_with_shap(calibrated_model, sample)
        
        # Vérifier les résultats
        prediction = result['prediction'][0]
        shap_explanation = result['shap_explanation']
        
        logger.info(f"Prédiction calibrée: {prediction:.4f}")
        logger.info(f"Valeur de base SHAP: {shap_explanation['base_value']}")
        logger.info(f"Forme des valeurs SHAP: {shap_explanation['shap_values'].shape}")
        
        # Mise à jour du modèle pour l'application
        # Copier le modèle calibré vers le chemin utilisé par l'application
        with open(MODEL_PATH_FOR_APP, 'wb') as f:
            pickle.dump(calibrated_model, f)
            
        logger.info(f"Modèle calibré copié vers {MODEL_PATH_FOR_APP} pour l'application")
        
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de l'intégration avec SHAP: {str(e)}")
        return False

def main():
    """
    Fonction principale qui exécute l'ensemble du processus de calibration et d'intégration.
    """
    try:
        logger.info("Démarrage du processus de calibration et d'intégration du modèle")
        
        # Créer le répertoire des logs s'il n'existe pas
        os.makedirs('logs', exist_ok=True)
        
        # Charger les données
        X_train, y_train, X_val, y_val = load_data()
        if X_train is None:
            raise ValueError("Impossible de charger les données")
        
        # Charger le modèle original
        original_model = load_model(ORIGINAL_MODEL_PATH)
        if original_model is None:
            raise ValueError("Impossible de charger le modèle original")
        
        # Calibrer le modèle et visualiser les résultats
        calibrated_model = calibrate_model_and_visualize(original_model, X_train, y_train, X_val, y_val)
        if calibrated_model is None:
            raise ValueError("La calibration du modèle a échoué")
        
        # Tester le modèle calibré
        test_calibrated_model(original_model, calibrated_model, X_val, y_val)
        
        # Intégrer avec SHAP
        success = integrate_with_shap(calibrated_model, X_val)
        if not success:
            logger.warning("L'intégration avec SHAP a rencontré des problèmes")
        
        logger.info("Processus de calibration et d'intégration terminé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur dans le processus de calibration: {str(e)}")

if __name__ == "__main__":
    main()
