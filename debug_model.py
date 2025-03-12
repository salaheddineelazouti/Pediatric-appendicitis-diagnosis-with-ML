"""
Script de diagnostic pour vérifier les problèmes avec le modèle de prédiction
"""
import os
import pickle
import pandas as pd
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chemins des fichiers
MODEL_PATH = os.path.join('models', 'best_model.pkl')
TRAINING_DATA_PATH = os.path.join('data', 'processed', 'training_data.csv')

def load_model():
    """Charge le modèle depuis le fichier pkl"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Type de modèle chargé: {type(model).__name__}")
        
        # Vérifier si c'est un pipeline
        if hasattr(model, 'named_steps'):
            logger.info(f"Composants du pipeline: {list(model.named_steps.keys())}")
            
            # Vérifier si SVM est présent dans le pipeline
            if 'svm' in model.named_steps:
                logger.info(f"Type de l'estimateur SVM: {type(model.named_steps['svm']).__name__}")
        
        # Vérifier les features requises
        if hasattr(model, 'feature_names_in_'):
            logger.info(f"Features requises: {model.feature_names_in_.tolist()}")
        
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
            return df
        else:
            logger.error(f"Fichier de données non trouvé: {TRAINING_DATA_PATH}")
            return None
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        return None

def test_prediction(model, n_samples=5):
    """Teste le modèle sur quelques échantillons aléatoires des données d'entraînement"""
    try:
        df = load_training_data()
        if df is None or model is None:
            return
        
        # Sélectionner des échantillons aléatoires
        sample_indices = np.random.choice(df.index, size=min(n_samples, len(df)), replace=False)
        samples = df.loc[sample_indices]
        
        logger.info("=== Test de prédiction sur échantillons aléatoires ===")
        
        for i, (idx, row) in enumerate(samples.iterrows()):
            # Extraire les features et la cible
            X = row.drop('appendicitis').to_frame().T
            y_true = row['appendicitis']
            
            # S'assurer que l'ordre des colonnes est correct
            if hasattr(model, 'feature_names_in_'):
                X = X[model.feature_names_in_]
            
            # Faire la prédiction
            try:
                y_pred_proba = model.predict_proba(X)[0][1]
                y_pred = 1 if y_pred_proba >= 0.5 else 0
                
                logger.info(f"Échantillon {i+1}:")
                logger.info(f"  - Valeur réelle: {y_true} ({'Appendicite' if y_true == 1 else 'Pas appendicite'})")
                logger.info(f"  - Prédiction: {y_pred} ({'Appendicite' if y_pred == 1 else 'Pas appendicite'})")
                logger.info(f"  - Probabilité: {y_pred_proba:.4f}")
                logger.info(f"  - Correct: {'✓' if y_pred == y_true else '✗'}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la prédiction: {e}")
    
    except Exception as e:
        logger.error(f"Erreur lors du test de prédiction: {e}")

def test_with_custom_data():
    """Teste le modèle avec des données personnalisées typiques"""
    try:
        model = load_model()
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
        logger.error(f"Erreur lors du test avec données personnalisées: {e}")

def main():
    """Fonction principale pour le diagnostic du modèle"""
    print("\n" + "="*70)
    print(" DIAGNOSTIC DU MODÈLE DE PRÉDICTION D'APPENDICITE PÉDIATRIQUE ")
    print("="*70 + "\n")
    
    # Charger le modèle
    model = load_model()
    
    # Charger les données d'entraînement
    training_data = load_training_data()
    
    # Tester sur des échantillons aléatoires
    if model is not None and training_data is not None:
        test_prediction(model)
    
    # Tester avec des cas personnalisés
    test_with_custom_data()
    
    print("\n" + "="*70)
    print(" FIN DU DIAGNOSTIC ")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
