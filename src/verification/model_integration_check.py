"""
Script de vérification de l'intégration du modèle SVM à l'interface utilisateur.

Ce script vérifie:
1. Si le meilleur modèle sauvegardé est bien un SVM
2. Si les caractéristiques attendues par le modèle correspondent aux données collectées via l'interface
3. Si les champs obligatoires de l'interface couvrent toutes les caractéristiques nécessaires au modèle
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Ajouter le répertoire racine au path Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Chemins des fichiers
model_path = os.path.join(project_root, 'models', 'best_model.pkl')
app_path = os.path.join(project_root, 'src', 'api', 'app.py')

# Charger le modèle
def load_model(model_path):
    """Charge le modèle depuis le fichier pickle"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# Extraire les caractéristiques de l'interface depuis app.py
def extract_features_from_app(app_path):
    """Extrait les caractéristiques définies dans l'application Flask"""
    try:
        # Cette méthode est un peu risquée, mais c'est une façon simple pour vérifier
        # Idéalement, nous aurions une fonction d'API pour récupérer ces informations
        
        features = []
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Extraire les définitions des caractéristiques
        feature_sections = [
            "DEMOGRAPHIC_FEATURES", "CLINICAL_FEATURES", 
            "LABORATORY_FEATURES", "SCORING_FEATURES"
        ]
        
        for section in feature_sections:
            # Trouver la section correspondante
            section_start = content.find(f"{section} = [")
            if section_start == -1:
                continue
            
            section_end = content.find("]", section_start)
            section_content = content[section_start:section_end]
            
            # Extraire les noms des caractéristiques
            feature_entries = section_content.split('{"name": "')
            for entry in feature_entries[1:]:  # Ignorer le premier élément
                feature_name = entry.split('"')[0]
                features.append(feature_name)
        
        return features
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des caractéristiques depuis app.py: {e}")
        return []

# Extraire les caractéristiques requises à partir des données dans app.py
def extract_required_features_from_app(app_path):
    """Extrait les caractéristiques requises définies dans l'application Flask"""
    try:
        required_features = []
        with open(app_path, 'r') as f:
            for line in f:
                if 'required_features' in line and '=' in line:
                    # Extraire la liste des caractéristiques requises
                    features_str = line.split('=')[1].strip()
                    if features_str.startswith('['):
                        features_str = features_str.strip('[]')
                        features = [f.strip().strip("'\"") for f in features_str.split(',')]
                        required_features.extend(features)
        return required_features
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des caractéristiques requises: {e}")
        return []

def analyze_model_features(model, ui_features, required_ui_features):
    """Analyse la correspondance entre les caractéristiques du modèle et celles de l'interface"""
    if not hasattr(model, 'feature_names_in_'):
        print("❌ Le modèle n'a pas d'attribut 'feature_names_in_'.")
        return False

    model_features = model.feature_names_in_
    print(f"\n📊 Caractéristiques du modèle: {len(model_features)}")
    print(model_features)
    
    print(f"\n📋 Caractéristiques de l'interface: {len(ui_features)}")
    print(ui_features)
    
    print(f"\n🔍 Caractéristiques requises dans l'interface: {len(required_ui_features)}")
    print(required_ui_features)
    
    # Vérifier si toutes les caractéristiques du modèle sont couvertes par l'interface
    missing_in_ui = [f for f in model_features if f not in ui_features]
    if missing_in_ui:
        print(f"\n❌ Caractéristiques du modèle manquantes dans l'interface: {missing_in_ui}")
        return False
    
    # Vérifier si toutes les caractéristiques requises par le modèle sont requises dans l'interface
    not_required_but_needed = [f for f in model_features if f not in required_ui_features]
    if not_required_but_needed:
        print(f"\n⚠️ Caractéristiques nécessaires au modèle mais non requises dans l'interface: {not_required_but_needed}")
    
    return True

def verify_model_type(model, expected_type=SVC):
    """Vérifie si le modèle est du type attendu"""
    if isinstance(model, expected_type):
        print(f"✅ Le modèle est bien du type attendu: {expected_type.__name__}")
        return True
    else:
        print(f"❌ Le modèle n'est pas du type attendu: {type(model).__name__} au lieu de {expected_type.__name__}")
        return False

def update_model_if_needed(model_path):
    """Met à jour le modèle SVM depuis le script d'évaluation si nécessaire"""
    try:
        # Vérifier si le modèle SVM existe dans le dossier reports
        reports_dir = os.path.join(project_root, 'reports')
        model_selection_path = os.path.join(reports_dir, 'model_selection_reasoning.txt')
        
        if os.path.exists(model_selection_path):
            print(f"\n📄 Fichier de raisonnement de sélection de modèle trouvé: {model_selection_path}")
            
            # Lire le fichier pour confirmer que SVM est recommandé
            with open(model_selection_path, 'r') as f:
                content = f.read()
                
            if 'MODÈLE SÉLECTIONNÉ: SVM' in content or 'Meilleur modèle recommandé: SVM' in content:
                print("✅ Le modèle SVM est bien recommandé selon l'évaluation.")
                
                # Importer les modules nécessaires pour l'entraînement du SVM
                from sklearn.model_selection import train_test_split
                from sklearn.svm import SVC
                
                # Charger les données ou générer des données synthétiques
                try:
                    from src.visualization.compare_models import load_data
                    X, y = load_data()
                    
                    # Diviser les données
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Créer et entraîner le modèle SVM
                    svm_model = SVC(
                        kernel='rbf', C=1, gamma='scale', probability=True, random_state=42
                    )
                    svm_model.fit(X_train, y_train)
                    
                    # Sauvegarder le modèle
                    with open(model_path, 'wb') as f:
                        pickle.dump(svm_model, f)
                    
                    print(f"✅ Modèle SVM mis à jour et sauvegardé dans {model_path}")
                    return svm_model
                except Exception as e:
                    print(f"❌ Erreur lors de la mise à jour du modèle SVM: {e}")
        
        return None
    except Exception as e:
        print(f"❌ Erreur lors de la vérification du modèle SVM: {e}")
        return None

def main():
    """Fonction principale"""
    print("\n" + "="*70)
    print("VÉRIFICATION DE L'INTÉGRATION DU MODÈLE À L'INTERFACE UTILISATEUR")
    print("="*70)
    
    # 1. Charger le modèle actuel
    model = load_model(model_path)
    if model is None:
        print("❌ Le modèle n'a pas pu être chargé. Tentative de mise à jour du modèle...")
        model = update_model_if_needed(model_path)
        if model is None:
            print("❌ Impossible de mettre à jour le modèle. Vérification impossible.")
            return
    
    # 2. Vérifier le type du modèle
    is_svm = verify_model_type(model, SVC)
    
    # 3. Si ce n'est pas un SVM, mise à jour nécessaire
    if not is_svm:
        print("\n🔄 Le modèle actuel n'est pas un SVM. Tentative de mise à jour...")
        model = update_model_if_needed(model_path)
        if model is None:
            print("❌ Impossible de mettre à jour le modèle vers SVM. Vérification terminée.")
            return
    
    # 4. Extraire les caractéristiques de l'interface
    ui_features = extract_features_from_app(app_path)
    required_ui_features = extract_required_features_from_app(app_path)
    
    # 5. Vérifier la correspondance entre les caractéristiques
    features_match = analyze_model_features(model, ui_features, required_ui_features)
    
    # 6. Conclusion
    print("\n" + "="*70)
    if is_svm and features_match:
        print("✅ VÉRIFICATION RÉUSSIE: Le modèle SVM est correctement intégré à l'interface.")
    else:
        print("⚠️ VÉRIFICATION INCOMPLÈTE: Des problèmes ont été détectés dans l'intégration.")
    
    print("\nRecommandations:")
    if not is_svm:
        print("- Mettre à jour le fichier 'best_model.pkl' avec le modèle SVM recommandé")
    
    if not features_match:
        print("- Assurer la correspondance entre les caractéristiques du modèle et de l'interface")
        print("- Mettre à jour la liste des caractéristiques requises dans l'application")

if __name__ == "__main__":
    main()
