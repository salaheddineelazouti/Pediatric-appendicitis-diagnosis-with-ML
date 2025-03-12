"""
Script simplifié pour vérifier l'intégration du modèle avec l'interface utilisateur.
"""

import os
import sys
import pickle
import pandas as pd
from sklearn.svm import SVC

# Chemins des fichiers
model_path = os.path.join('models', 'best_model.pkl')
app_path = os.path.join('src', 'api', 'app.py')

def main():
    print("\n" + "="*70)
    print("VERIFICATION DE L'INTEGRATION DU MODELE A L'INTERFACE UTILISATEUR")
    print("="*70)
    
    # 1. Vérifier l'existence du modèle
    if not os.path.exists(model_path):
        print(f"[ERREUR] Le fichier du modèle n'existe pas: {model_path}")
        return
    
    # 2. Charger le modèle
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"[INFO] Modèle chargé: {type(model).__name__}")
    except Exception as e:
        print(f"[ERREUR] Echec du chargement du modèle: {e}")
        return
    
    # 3. Vérifier si c'est un SVM
    if isinstance(model, SVC):
        print("[OK] Le modèle est bien un SVM comme recommandé.")
    else:
        print(f"[ALERTE] Le modèle n'est pas un SVM mais un {type(model).__name__}.")
        
    # 4. Vérifier les caractéristiques du modèle
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        print(f"\n[INFO] Le modèle utilise {len(model_features)} caractéristiques:")
        for i, feature in enumerate(model_features):
            print(f"  {i+1}. {feature}")
    else:
        print("[ALERTE] Le modèle n'a pas d'attribut 'feature_names_in_'.")
    
    # 5. Extraire les caractéristiques de l'interface
    ui_features = []
    required_features = []
    
    try:
        # Extraire les caractéristiques définies dans l'application
        with open(app_path, 'r') as f:
            content = f.read()
            
        # Extraire la liste des caractéristiques requises
        for line in content.split('\n'):
            if 'required_features' in line and '=' in line:
                features_str = line.split('=')[1].strip()
                if features_str.startswith('['):
                    features_str = features_str.strip('[]')
                    features = [f.strip().strip("'\"") for f in features_str.split(',')]
                    required_features.extend(features)
        
        # Extraire toutes les caractéristiques de l'interface
        feature_sections = [
            "DEMOGRAPHIC_FEATURES", "CLINICAL_FEATURES", 
            "LABORATORY_FEATURES", "SCORING_FEATURES"
        ]
        
        for section in feature_sections:
            section_start = content.find(f"{section} = [")
            if section_start == -1:
                continue
            
            section_end = content.find("]", section_start)
            section_content = content[section_start:section_end]
            
            feature_entries = section_content.split('{"name": "')
            for entry in feature_entries[1:]:
                feature_name = entry.split('"')[0]
                ui_features.append(feature_name)
        
        print(f"\n[INFO] L'interface collecte {len(ui_features)} caractéristiques:")
        for i, feature in enumerate(ui_features):
            print(f"  {i+1}. {feature}")
        
        print(f"\n[INFO] Caractéristiques requises dans l'interface ({len(required_features)}):")
        for i, feature in enumerate(required_features):
            print(f"  {i+1}. {feature}")
    except Exception as e:
        print(f"[ERREUR] Echec de l'extraction des caractéristiques de l'interface: {e}")
    
    # 6. Vérifier la compatibilité des caractéristiques
    if hasattr(model, 'feature_names_in_'):
        # Vérifier si toutes les caractéristiques du modèle sont dans l'interface
        missing_in_ui = []
        for feature in model_features:
            if feature not in ui_features:
                missing_in_ui.append(feature)
        
        if missing_in_ui:
            print(f"\n[ALERTE] Caractéristiques du modèle manquantes dans l'interface ({len(missing_in_ui)}):")
            for feature in missing_in_ui:
                print(f"  - {feature}")
        else:
            print("\n[OK] Toutes les caractéristiques du modèle sont collectées par l'interface.")
        
        # Vérifier si toutes les caractéristiques du modèle sont requises
        not_required = []
        for feature in model_features:
            if feature not in required_features:
                not_required.append(feature)
        
        if not_required:
            print(f"\n[ALERTE] Caractéristiques du modèle non marquées comme requises ({len(not_required)}):")
            for feature in not_required:
                print(f"  - {feature}")
        else:
            print("\n[OK] Toutes les caractéristiques du modèle sont marquées comme requises.")
    
    # 7. Conclusion
    print("\n" + "="*70)
    
    if hasattr(model, 'feature_names_in_') and not missing_in_ui and isinstance(model, SVC):
        print("[OK] Le modèle SVM est correctement intégré à l'interface.")
    else:
        print("[ALERTE] Des problèmes ont été détectés dans l'intégration du modèle.")
    
    # 8. Recommandations
    print("\nRecommandations:")
    if not isinstance(model, SVC):
        print("- Mettre à jour le modèle vers un SVM comme recommandé par l'évaluation")
    
    if missing_in_ui:
        print("- Ajouter les caractéristiques manquantes dans l'interface utilisateur")
    
    if not_required:
        print("- Marquer toutes les caractéristiques du modèle comme requises dans l'interface")
    
    print("- Vérifier que les données saisies sont correctement prétraitées avant la prédiction")
    print("- S'assurer que l'interface affiche clairement les résultats et leurs interprétations")

if __name__ == "__main__":
    main()
