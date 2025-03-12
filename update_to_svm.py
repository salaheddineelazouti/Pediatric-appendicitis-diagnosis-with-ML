"""
Script pour créer et déployer directement un modèle SVM
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Chemins
MODEL_DIR = 'models'
DATA_DIR = os.path.join('data', 'processed')
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
BACKUP_MODEL_PATH = os.path.join(MODEL_DIR, 'previous_model_rf.pkl')
TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'training_data.csv')

# Caractéristiques
FEATURES = [
    "age", "gender", "duration", "migration", "anorexia", "nausea", "vomiting",
    "right_lower_quadrant_pain", "fever", "rebound_tenderness", 
    "white_blood_cell_count", "neutrophil_percentage", "c_reactive_protein",
    "pediatric_appendicitis_score", "alvarado_score"
]

def backup_existing_model():
    """Sauvegarde le modèle existant"""
    try:
        if os.path.exists(CURRENT_MODEL_PATH):
            # Charger le modèle pour vérifier son type
            with open(CURRENT_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            model_type = type(model).__name__
            
            # Créer une copie de sauvegarde
            with open(BACKUP_MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Modèle {model_type} sauvegardé dans {BACKUP_MODEL_PATH}")
            return True
        else:
            print(f"Aucun modèle trouvé à {CURRENT_MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle: {e}")
        return False

def load_training_data():
    """Charge les données d'entraînement"""
    try:
        if os.path.exists(TRAINING_DATA_PATH):
            df = pd.read_csv(TRAINING_DATA_PATH)
            print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            return df
        else:
            print(f"Fichier de données non trouvé: {TRAINING_DATA_PATH}")
            return None
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return None

def create_svm_model(df=None):
    """Crée un modèle SVM, avec ou sans données d'entraînement"""
    print("Création du modèle SVM...")
    
    if df is not None:
        # Utiliser les données d'entraînement
        X = df[FEATURES]
        y = df['appendicitis']
        
        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Créer et entraîner le pipeline
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, C=10, gamma='scale', kernel='rbf', random_state=42))
        ])
        
        svm_pipeline.fit(X_train, y_train)
        print("Modèle SVM entraîné avec les données existantes")
        
    else:
        # Créer un modèle SVM non entraîné avec les bons paramètres
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, C=10, gamma='scale', kernel='rbf', random_state=42))
        ])
        print("Modèle SVM créé (non entraîné)")
    
    # Vérifier que le modèle est bien un SVM
    svm_model = svm_pipeline.named_steps.get('svm')
    if isinstance(svm_model, SVC):
        print(f"Confirmation: le modèle est bien un SVM ({type(svm_model).__name__})")
    else:
        print(f"ERREUR: le modèle n'est pas un SVM mais un {type(svm_model).__name__}")
    
    return svm_pipeline

def save_svm_model(model):
    """Sauvegarde le modèle SVM comme modèle principal"""
    try:
        with open(CURRENT_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modèle SVM sauvegardé comme modèle principal: {CURRENT_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle SVM: {e}")
        return False

def main():
    print("\nMISE À JOUR DU MODÈLE VERS SVM")
    print("=" * 40)
    
    # 1. Sauvegarder le modèle existant
    backup_existing_model()
    
    # 2. Charger les données d'entraînement si disponibles
    df = load_training_data()
    
    # 3. Créer un modèle SVM
    svm_model = create_svm_model(df)
    
    # 4. Sauvegarder le modèle SVM comme modèle principal
    save_svm_model(svm_model)
    
    print("\nOpération terminée!")
    print("Le modèle principal est maintenant un SVM.")
    print("Pour vérifier, exécutez 'python verify_model_features.py'")

if __name__ == "__main__":
    main()
