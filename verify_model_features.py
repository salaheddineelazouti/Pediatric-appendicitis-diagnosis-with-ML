"""
Script pour vérifier les données d'entraînement et les caractéristiques du modèle
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from tabulate import tabulate

# Définition des chemins
model_path = os.path.join('models', 'best_model.pkl')
data_folder = os.path.join('data', 'processed')

# Liste des caractéristiques attendues basée sur le rapport médical
EXPECTED_FEATURES = [
    "age", "gender", "duration", "migration", "anorexia", "nausea", "vomiting",
    "right_lower_quadrant_pain", "fever", "rebound_tenderness", 
    "white_blood_cell_count", "neutrophil_percentage", "c_reactive_protein",
    "pediatric_appendicitis_score", "alvarado_score"
]

def load_model():
    """Charge le modèle et retourne ses caractéristiques"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        model_type = type(model).__name__
        print(f"\nModèle chargé : {model_type}")
        
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
            return model, features
        else:
            print("[ATTENTION] Le modèle n'a pas d'attribut 'feature_names_in_'")
            return model, []
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None, []

def find_training_data():
    """Recherche les fichiers de données d'entraînement"""
    try:
        data_files = []
        
        # Chercher dans le dossier de données traitées
        if os.path.exists(data_folder):
            for file in os.listdir(data_folder):
                if file.endswith('.csv') or file.endswith('.xlsx'):
                    data_files.append(os.path.join(data_folder, file))
        
        # Chercher dans le dossier racine si rien n'a été trouvé
        if not data_files:
            for file in os.listdir('.'):
                if (file.endswith('.csv') or file.endswith('.xlsx')) and ('train' in file.lower() or 'data' in file.lower()):
                    data_files.append(file)
        
        return data_files
    except Exception as e:
        print(f"Erreur lors de la recherche des fichiers de données : {e}")
        return []

def load_training_data(data_files):
    """Charge les données d'entraînement"""
    dataframes = []
    
    for file in data_files:
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.endswith('.xlsx'):
                df = pd.read_excel(file)
            
            print(f"\nDonnées chargées depuis : {file}")
            print(f"Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            dataframes.append(df)
        except Exception as e:
            print(f"Erreur lors du chargement de {file} : {e}")
    
    return dataframes

def analyze_features(df_list, model_features):
    """Analyse les caractéristiques des données et du modèle"""
    for i, df in enumerate(df_list):
        print(f"\n=== Analyse du dataset {i+1} ===")
        
        # Vérifier les colonnes
        columns = df.columns.tolist()
        print(f"Nombre total de colonnes : {len(columns)}")
        
        # Vérifier si target (appendicitis) est présent
        target_col = None
        target_candidates = ['appendicitis', 'diagnosis', 'target', 'label', 'appendicite']
        for candidate in target_candidates:
            if candidate in columns:
                target_col = candidate
                break
        
        if target_col:
            print(f"Variable cible trouvée : '{target_col}'")
            target_distribution = df[target_col].value_counts()
            print("Distribution de la variable cible :")
            for value, count in target_distribution.items():
                percent = count / len(df) * 100
                print(f"  - {value}: {count} ({percent:.1f}%)")
        else:
            print("Aucune variable cible clairement identifiée.")
        
        # Vérifier les caractéristiques attendues
        found_expected = [feature for feature in EXPECTED_FEATURES if feature in columns]
        missing_expected = [feature for feature in EXPECTED_FEATURES if feature not in columns]
        
        print(f"\nCaractéristiques attendues trouvées : {len(found_expected)}/{len(EXPECTED_FEATURES)}")
        if missing_expected:
            print("Caractéristiques attendues manquantes :")
            for feature in missing_expected:
                print(f"  - {feature}")
        
        # Comparer avec les caractéristiques du modèle
        if len(model_features) > 0:
            found_in_model = [feature for feature in columns if feature in model_features]
            print(f"\nCaractéristiques présentes dans le modèle : {len(found_in_model)}/{len(model_features)}")
            
            missing_in_model = [feature for feature in model_features if feature not in columns]
            if missing_in_model:
                print("Caractéristiques du modèle non trouvées dans les données :")
                for feature in missing_in_model:
                    print(f"  - {feature}")
        
        # Analyser les types de données
        print("\nTypes de données des caractéristiques :")
        dtypes_summary = df.dtypes.value_counts().to_dict()
        for dtype, count in dtypes_summary.items():
            print(f"  - {dtype}: {count} colonnes")
        
        # Vérifier les valeurs manquantes
        missing_values = df.isnull().sum()
        features_with_missing = missing_values[missing_values > 0]
        if not features_with_missing.empty:
            print("\nCaractéristiques avec valeurs manquantes :")
            for feature, count in features_with_missing.items():
                percent = count / len(df) * 100
                print(f"  - {feature}: {count} ({percent:.1f}%)")
        else:
            print("\nAucune valeur manquante détectée.")
        
        # Résumé des données numériques
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if not numeric_columns.empty:
            print("\nRésumé des caractéristiques numériques :")
            stats = df[numeric_columns].describe().T[['mean', 'std', 'min', 'max']]
            print(tabulate(stats, headers='keys', tablefmt='psql', floatfmt='.2f'))

def verify_model_type(model):
    """Vérifie si le modèle est du type recommandé (SVM)"""
    if model is None:
        return "Inconnu"
    
    # Vérifier si c'est un pipeline contenant un SVM
    if hasattr(model, 'named_steps') and 'svm' in model.named_steps:
        svm_model = model.named_steps['svm']
        if isinstance(svm_model, SVC):
            return "Pipeline avec SVM (recommandé)"
    
    # Vérifier si c'est directement un SVM
    if isinstance(model, SVC):
        return "SVM (recommandé)"
    
    # Vérifier si c'est un ensemble contenant des SVMs
    if hasattr(model, 'estimators_'):
        if any(isinstance(est, SVC) for est in model.estimators_):
            return f"{type(model).__name__} avec composants SVM"
    
    return type(model).__name__

def main():
    print("\n" + "="*70)
    print("VÉRIFICATION DES DONNÉES D'ENTRAÎNEMENT ET DU MODÈLE")
    print("="*70)
    
    # 1. Charger le modèle
    model, model_features = load_model()
    
    print("\nCaractéristiques utilisées par le modèle :")
    if len(model_features) > 0:
        for i, feature in enumerate(model_features):
            print(f"  {i+1}. {feature}")
    else:
        print("  Aucune caractéristique identifiée dans le modèle.")
    
    # 2. Vérifier le type de modèle
    model_type_info = verify_model_type(model)
    print(f"\nType de modèle : {model_type_info}")
    
    # 3. Rechercher les données d'entraînement
    data_files = find_training_data()
    if data_files:
        print("\nFichiers de données trouvés :")
        for file in data_files:
            print(f"  - {file}")
        
        # 4. Charger et analyser les données
        df_list = load_training_data(data_files)
        analyze_features(df_list, model_features)
    else:
        print("\nAucun fichier de données d'entraînement trouvé.")
    
    # 5. Conclusion
    print("\n" + "="*70)
    print("RÉSUMÉ DE L'ANALYSE")
    print("="*70)
    
    # Vérifier la cohérence entre les caractéristiques du modèle et les caractéristiques attendues
    if len(model_features) > 0:
        missing_in_expected = [f for f in model_features if f not in EXPECTED_FEATURES]
        missing_from_model = [f for f in EXPECTED_FEATURES if f not in model_features]
        
        if not missing_in_expected and not missing_from_model:
            print("\n[OK] Les caractéristiques du modèle correspondent exactement à celles attendues.")
        else:
            if missing_in_expected:
                print(f"\n[ATTENTION] Le modèle utilise {len(missing_in_expected)} caractéristiques non listées dans le rapport médical :")
                for feature in missing_in_expected:
                    print(f"  - {feature}")
            
            if missing_from_model:
                print(f"\n[ATTENTION] {len(missing_from_model)} caractéristiques attendues ne sont pas utilisées par le modèle :")
                for feature in missing_from_model:
                    print(f"  - {feature}")
    
    # Vérifier le type de modèle
    if "SVM" in model_type_info:
        print("\n[OK] Le modèle est bien un SVM comme recommandé par les tests.")
    else:
        print("\n[ATTENTION] Le modèle n'est pas un SVM comme recommandé par les tests.")
    
    print("\nConclusion :")
    if model is not None and len(model_features) > 0 and "SVM" in model_type_info:
        print("Le modèle et les données d'entraînement semblent correctement configurés.")
    else:
        print("Des problèmes ont été identifiés avec le modèle ou les données d'entraînement.")
    
    print("\nRecommandations :")
    if "SVM" not in model_type_info:
        print("- Entraîner un modèle SVM comme recommandé par les tests comparatifs")
    
    if len(missing_from_model) > 0:
        print("- S'assurer que toutes les caractéristiques requises sont utilisées par le modèle")
    
    if len(missing_in_expected) > 0:
        print("- Mettre à jour le rapport médical pour inclure toutes les caractéristiques utilisées par le modèle")
    
    print("- Vérifier que les médecins sont informés de toutes les données à collecter")
    print("- S'assurer que l'interface utilisateur demande toutes les caractéristiques nécessaires")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erreur lors de l'exécution du script : {e}")
