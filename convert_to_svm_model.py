"""
Script pour convertir le modèle RandomForest actuel en SVM
et sauvegarder un fichier de données d'entraînement
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Chemins des fichiers
MODEL_DIR = 'models'
DATA_DIR = os.path.join('data', 'processed')
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
NEW_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_svm.pkl')
TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'training_data.csv')

# S'assurer que les répertoires existent
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Liste des caractéristiques
FEATURES = [
    "age", "gender", "duration", "migration", "anorexia", "nausea", "vomiting",
    "right_lower_quadrant_pain", "fever", "rebound_tenderness", 
    "white_blood_cell_count", "neutrophil_percentage", "c_reactive_protein",
    "pediatric_appendicitis_score", "alvarado_score"
]

def load_existing_model():
    """Charge le modèle existant et extrait les données si possibles"""
    print("\nChargement du modèle existant...")
    try:
        with open(CURRENT_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        model_type = type(model).__name__
        print(f"Modèle chargé avec succès : {model_type}")
        
        # Vérifier si on peut extraire des caractéristiques du modèle
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
            print(f"Nombre de caractéristiques : {len(features)}")
            print("Caractéristiques : ", features)
            return model, features
        else:
            print("Le modèle n'a pas d'attribut 'feature_names_in_'")
            return model, FEATURES
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None, FEATURES

def generate_synthetic_data(features, n_samples=1000):
    """Génère des données synthétiques pour l'entraînement"""
    print(f"\nGénération de {n_samples} échantillons de données synthétiques...")
    
    # Définir les plages de valeurs pour chaque caractéristique
    data = {}
    
    # Données démographiques
    data['age'] = np.random.uniform(2, 17, n_samples)  # Âge pédiatrique
    data['gender'] = np.random.binomial(1, 0.5, n_samples)  # 0=F, 1=M
    
    # Signes cliniques
    data['duration'] = np.random.uniform(2, 72, n_samples)  # Heures
    data['migration'] = np.random.binomial(1, 0.6, n_samples)
    data['anorexia'] = np.random.binomial(1, 0.5, n_samples)
    data['nausea'] = np.random.binomial(1, 0.7, n_samples)
    data['vomiting'] = np.random.binomial(1, 0.6, n_samples)
    data['right_lower_quadrant_pain'] = np.random.binomial(1, 0.9, n_samples)
    data['fever'] = np.random.binomial(1, 0.4, n_samples)
    data['rebound_tenderness'] = np.random.binomial(1, 0.7, n_samples)
    
    # Examens de laboratoire
    data['white_blood_cell_count'] = np.random.uniform(4.0, 20.0, n_samples)
    data['neutrophil_percentage'] = np.random.uniform(30, 95, n_samples)
    data['c_reactive_protein'] = np.random.uniform(0.5, 150, n_samples)
    
    # Scores cliniques
    data['pediatric_appendicitis_score'] = np.random.uniform(0, 10, n_samples)
    data['alvarado_score'] = np.random.uniform(0, 10, n_samples)
    
    # Créer un DataFrame
    df = pd.DataFrame(data)
    
    # Générer une variable cible qui dépend des caractéristiques
    # Plus les scores sont élevés, plus la probabilité d'appendicite est élevée
    logits = (
        0.3 * (df['pediatric_appendicitis_score'] / 10) + 
        0.3 * (df['alvarado_score'] / 10) + 
        0.2 * ((df['white_blood_cell_count'] - 4.5) / 15.5) +
        0.1 * ((df['neutrophil_percentage'] - 40) / 55) +
        0.1 * ((df['c_reactive_protein']) / 150) +
        0.2 * df['right_lower_quadrant_pain'] +
        0.1 * df['rebound_tenderness'] +
        0.1 * df['fever'] +
        0.1 * df['migration'] -
        0.5  # Décalage pour équilibrer les classes
    )
    probs = 1 / (1 + np.exp(-logits))
    df['appendicitis'] = (np.random.random(n_samples) < probs).astype(int)
    
    # Équilibrer un peu les classes
    pos_rate = df['appendicitis'].mean()
    print(f"Taux initial d'appendicite : {pos_rate:.2f}")
    
    if pos_rate < 0.4:
        # Augmenter le taux d'appendicite
        idx_neg = df[df['appendicitis'] == 0].index
        n_to_flip = int((0.4 - pos_rate) * n_samples)
        idx_to_flip = np.random.choice(idx_neg, size=n_to_flip, replace=False)
        df.loc[idx_to_flip, 'appendicitis'] = 1
    elif pos_rate > 0.6:
        # Diminuer le taux d'appendicite
        idx_pos = df[df['appendicitis'] == 1].index
        n_to_flip = int((pos_rate - 0.6) * n_samples)
        idx_to_flip = np.random.choice(idx_pos, size=n_to_flip, replace=False)
        df.loc[idx_to_flip, 'appendicitis'] = 0
    
    final_pos_rate = df['appendicitis'].mean()
    print(f"Taux final d'appendicite : {final_pos_rate:.2f}")
    
    return df

def save_training_data(df):
    """Sauvegarde les données d'entraînement"""
    try:
        df.to_csv(TRAINING_DATA_PATH, index=False)
        print(f"Données d'entraînement sauvegardées dans {TRAINING_DATA_PATH}")
        print(f"Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des données : {e}")
        return False

def train_svm_model(df, features):
    """Entraîne un modèle SVM avec optimisation des hyperparamètres"""
    print("\nEntraînement du modèle SVM...")
    
    # Préparer les données
    X = df[features]
    y = df['appendicitis']
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Créer un pipeline avec mise à l'échelle
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    # Paramètres à optimiser
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01],
        'svm__kernel': ['rbf']
    }
    
    # Recherche par grille avec validation croisée
    grid = GridSearchCV(
        pipe, 
        param_grid, 
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # Entraîner le modèle
    grid.fit(X_train, y_train)
    
    # Meilleur modèle
    best_model = grid.best_estimator_
    print(f"Meilleurs paramètres : {grid.best_params_}")
    
    # Évaluer le modèle
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Métriques de performance
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': precision_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_prob)
    }
    
    print("\nPerformance du modèle SVM :")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value:.4f}")
    
    # Vérifier si le modèle est bien de type SVC
    svm_model = best_model.named_steps['svm']
    if isinstance(svm_model, SVC):
        print(f"Le modèle est bien un SVM de type {type(svm_model).__name__}")
    else:
        print(f"ERREUR: Le modèle n'est pas un SVM mais un {type(svm_model).__name__}")
    
    return best_model

def save_model(model, path):
    """Sauvegarde le modèle entraîné"""
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modèle sauvegardé avec succès dans {path}")
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")
        return False

def plot_feature_importance(df, features):
    """Analyse l'importance des caractéristiques avec correlation"""
    print("\nAnalyse de l'importance des caractéristiques...")
    
    # Calculer les corrélations avec la variable cible
    correlations = df[features + ['appendicitis']].corr()['appendicitis'].drop('appendicitis')
    
    # Trier les corrélations par valeur absolue
    correlations = correlations.abs().sort_values(ascending=False)
    
    # Créer un graphique
    plt.figure(figsize=(10, 6))
    correlations.plot(kind='bar', color='skyblue')
    plt.title('Importance des caractéristiques (corrélation absolue avec appendicitis)')
    plt.xlabel('Caractéristiques')
    plt.ylabel('Corrélation absolue')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'feature_importance.png'))
    print(f"Graphique d'importance sauvegardé dans {os.path.join(DATA_DIR, 'feature_importance.png')}")
    
    return correlations

def compare_with_existing_model(df, features, svm_model):
    """Compare le modèle SVM avec le modèle existant"""
    try:
        # Charger le modèle existant
        with open(CURRENT_MODEL_PATH, 'rb') as f:
            existing_model = pickle.load(f)
        
        print("\nComparaison avec le modèle existant...")
        
        # Préparer les données
        X = df[features]
        y = df['appendicitis']
        
        # Diviser les données
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Évaluer les deux modèles
        models = {
            'Modèle existant (RandomForest)': existing_model,
            'Nouveau modèle (SVM)': svm_model
        }
        
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc_score(y_test, y_prob)
            }
        
        # Afficher les résultats
        print("\nPerformance comparée :")
        for model_name, metrics in results.items():
            print(f"\n{model_name} :")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value:.4f}")
        
        # Comparer les deux modèles
        svm_better = results['Nouveau modèle (SVM)']['ROC AUC'] > results['Modèle existant (RandomForest)']['ROC AUC']
        
        if svm_better:
            print("\n✅ Le modèle SVM est plus performant que le modèle RandomForest existant.")
            return True
        else:
            print("\n⚠️ Le modèle SVM est moins performant que le modèle RandomForest existant.")
            return False
            
    except Exception as e:
        print(f"Erreur lors de la comparaison des modèles : {e}")
        return None

def update_best_model(svm_better):
    """Met à jour le modèle principal si le SVM est meilleur"""
    if svm_better:
        try:
            # Sauvegarder une copie du modèle original
            os.rename(CURRENT_MODEL_PATH, os.path.join(MODEL_DIR, 'previous_model_rf.pkl'))
            # Renommer le modèle SVM en best_model.pkl
            os.rename(NEW_MODEL_PATH, CURRENT_MODEL_PATH)
            print("\n✅ Le modèle principal a été mis à jour avec le modèle SVM.")
            return True
        except Exception as e:
            print(f"Erreur lors de la mise à jour du modèle principal : {e}")
            return False
    else:
        print("\nLe modèle principal (RandomForest) n'a pas été modifié car il est plus performant.")
        return False

def main():
    print("\n" + "="*60)
    print("CONVERSION DU MODÈLE RANDOMFOREST VERS SVM")
    print("="*60)
    
    # 1. Charger le modèle existant
    existing_model, features = load_existing_model()
    
    # 2. Générer des données synthétiques
    df = generate_synthetic_data(features)
    
    # 3. Sauvegarder les données d'entraînement
    save_training_data(df)
    
    # 4. Analyser l'importance des caractéristiques
    feature_importance = plot_feature_importance(df, features)
    print("\nImportance des caractéristiques :")
    for feature, importance in feature_importance.items():
        print(f"  - {feature}: {importance:.4f}")
    
    # 5. Entraîner un modèle SVM
    svm_model = train_svm_model(df, features)
    
    # 6. Sauvegarder le nouveau modèle SVM
    save_model(svm_model, NEW_MODEL_PATH)
    
    # 7. Comparer avec le modèle existant
    svm_better = compare_with_existing_model(df, features, svm_model)
    
    # 8. Mettre à jour le modèle principal si le SVM est meilleur
    if svm_better is True:
        update_best_model(svm_better)
    
    print("\n" + "="*60)
    print("RÉSUMÉ DES ACTIONS")
    print("="*60)
    print("1. ✅ Fichier de données d'entraînement créé")
    print(f"   → {TRAINING_DATA_PATH}")
    print("2. ✅ Modèle SVM entraîné")
    print(f"   → {NEW_MODEL_PATH}")
    
    if svm_better is True:
        print("3. ✅ Modèle principal mis à jour avec le SVM")
    elif svm_better is False:
        print("3. ℹ️ Modèle principal (RandomForest) conservé car plus performant")
    else:
        print("3. ⚠️ Comparaison des modèles impossible")
    
    print("\nProchaines étapes suggérées :")
    print("1. Vérifier que le rapport médical reflète bien les caractéristiques du modèle")
    print("2. S'assurer que l'interface utilisateur demande toutes les caractéristiques nécessaires")
    print("3. Réexécuter le script de vérification pour confirmer les corrections")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erreur lors de l'exécution du script : {e}")
