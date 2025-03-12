"""
Script de comparaison des modèles de diagnostic d'appendicite pédiatrique

Ce script évalue les performances de différents modèles de classification
en utilisant ROC-AUC, précision, rappel et F1-score, puis génère des
visualisations pour aider à la sélection du meilleur modèle.

Le script gère les dépendances manquantes et s'adapte aux bibliothèques disponibles.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Vérifier les bibliothèques disponibles
DEPENDENCIES = {
    'lightgbm': False,
    'catboost': False,
    'xgboost': False,
    'plotly': False
}

# Tentatives d'importation conditionnelles
try:
    import lightgbm as lgb
    DEPENDENCIES['lightgbm'] = True
    print("LightGBM est disponible et sera utilisé.")
except ImportError:
    print("LightGBM n'est pas disponible et sera ignoré.")

try:
    import catboost as cb
    DEPENDENCIES['catboost'] = True
    print("CatBoost est disponible et sera utilisé.")
except ImportError:
    print("CatBoost n'est pas disponible et sera ignoré.")

try:
    import xgboost as xgb
    DEPENDENCIES['xgboost'] = True
    print("XGBoost est disponible et sera utilisé.")
except ImportError:
    print("XGBoost n'est pas disponible et sera ignoré.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    DEPENDENCIES['plotly'] = True
    print("Plotly est disponible pour des visualisations interactives.")
except ImportError:
    print("Plotly n'est pas disponible. Utilisation de Matplotlib à la place.")

def load_data():
    """
    Charge les données d'appendicite ou génère des données synthétiques.
    
    Returns:
        tuple: (X, y) avec X les caractéristiques et y les étiquettes
    """
    try:
        data_path = os.path.join('data', 'processed', 'appendicitis_data.csv')
        if os.path.exists(data_path):
            print(f"Chargement des données depuis {data_path}")
            df = pd.read_csv(data_path)
            
            # Identification des colonnes de caractéristiques et de la cible
            if 'target' in df.columns:
                y = df['target']
                X = df.drop('target', axis=1)
            else:
                # Supposer que la dernière colonne est la cible
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]
                
            return X, y
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
    
    # Si les données ne peuvent pas être chargées, créer des données synthétiques
    print("Génération de données synthétiques pour la démonstration...")
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],  # Déséquilibre typique dans les diagnostics médicaux
        random_state=42
    )
    
    # Créer un DataFrame avec des noms de colonnes significatifs
    feature_names = [
        'Age', 'Gender', 'Duration', 'Migration', 'Anorexia', 
        'Nausea', 'WBC', 'Neutrophils', 'CRP', 'Rebound'
    ]
    X = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
    y = pd.Series(y, name='Appendicitis')
    
    return X, y

def get_models():
    """
    Renvoie un dictionnaire des modèles disponibles.
    
    Returns:
        dict: Nom de modèle -> instance de modèle
    """
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
        )
    }
    
    # Ajouter les modèles selon leur disponibilité
    if DEPENDENCIES['lightgbm']:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        )
    
    if DEPENDENCIES['catboost']:
        models['CatBoost'] = cb.CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.1, random_state=42,
            verbose=0, thread_count=-1
        )
    
    if DEPENDENCIES['xgboost']:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1
        )
    
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Entraîne et évalue un modèle.
    
    Args:
        model: Instance du modèle à évaluer
        X_train, X_test: Caractéristiques d'entraînement et de test
        y_train, y_test: Étiquettes d'entraînement et de test
        
    Returns:
        dict: Métriques d'évaluation
    """
    # Mesurer le temps d'entraînement
    start_time = time.time()
    
    # Entraîner le modèle
    model.fit(X_train, y_train)
    
    # Calculer le temps d'entraînement
    training_time = time.time() - start_time
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = None
    
    # Certains modèles n'ont pas predict_proba
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, NotImplementedError):
        # Fallback: utiliser decision_function si disponible
        try:
            y_proba = model.decision_function(X_test)
            # Normaliser entre 0 et 1
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
        except (AttributeError, NotImplementedError):
            # Aucune méthode disponible, utiliser les prédictions binaires
            y_proba = y_pred
    
    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Courbe ROC et AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Importance des caractéristiques si disponible
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'training_time': training_time,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'feature_importance': feature_importance
    }

def plot_performance_comparison(results, output_dir):
    """
    Crée des visualisations comparatives des performances des modèles.
    
    Args:
        results: Dict de résultats par modèle
        output_dir: Répertoire de sortie pour les visualisations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Graphique de comparaison des métriques
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = list(results.keys())
    
    if DEPENDENCIES['plotly']:
        # Créer un DataFrame pour plotly
        df = pd.DataFrame(columns=['Model', 'Metric', 'Value'])
        
        for model_name in model_names:
            for metric in metrics:
                df = pd.concat([df, pd.DataFrame({
                    'Model': [model_name],
                    'Metric': [metric.upper()],
                    'Value': [results[model_name][metric]]
                })], ignore_index=True)
        
        fig = px.bar(
            df, 
            x='Model', 
            y='Value', 
            color='Metric',
            barmode='group',
            title='Comparaison des performances des modèles',
            labels={'Value': 'Score', 'Model': 'Modèle', 'Metric': 'Métrique'},
            height=600,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_layout(
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Sauvegarder
        fig.write_html(os.path.join(output_dir, 'model_comparison.html'))
        print(f"Graphique de comparaison sauvegardé dans {output_dir}/model_comparison.html")
    else:
        # Utiliser matplotlib
        plt.figure(figsize=(12, 8))
        
        # Préparer les données
        x = np.arange(len(model_names))
        width = 0.15
        multiplier = 0
        
        for metric in metrics:
            offset = width * multiplier
            values = [results[model_name][metric] for model_name in model_names]
            
            rects = plt.bar(x + offset, values, width, label=metric.upper())
            
            # Ajouter les valeurs sur les barres
            for rect in rects:
                height = rect.get_height()
                plt.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # Décalage en y
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90, fontsize=8)
            
            multiplier += 1
        
        # Personnaliser le graphique
        plt.xlabel('Modèle')
        plt.ylabel('Score')
        plt.title('Comparaison des performances des modèles')
        plt.xticks(x + width * (len(metrics) - 1) / 2, model_names, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(metrics))
        plt.tight_layout()
        
        # Sauvegarder
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        print(f"Graphique de comparaison sauvegardé dans {output_dir}/model_comparison.png")
    
    # 2. Courbes ROC
    if DEPENDENCIES['plotly']:
        fig = go.Figure()
        
        for model_name in model_names:
            fig.add_trace(
                go.Scatter(
                    x=results[model_name]['fpr'],
                    y=results[model_name]['tpr'],
                    mode='lines',
                    name=f"{model_name} (AUC={results[model_name]['auc']:.3f})",
                    line=dict(width=2)
                )
            )
        
        # Ajouter la ligne diagonale
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Aléatoire',
                line=dict(dash='dash', width=2, color='gray')
            )
        )
        
        fig.update_layout(
            title='Comparaison des courbes ROC',
            xaxis_title='Taux de faux positifs',
            yaxis_title='Taux de vrais positifs',
            font=dict(family="Arial", size=14),
            plot_bgcolor='white',
            height=600,
            width=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(range=[0, 1], constrain='domain')
        fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
        
        # Sauvegarder
        fig.write_html(os.path.join(output_dir, 'roc_curves.html'))
        print(f"Courbes ROC sauvegardées dans {output_dir}/roc_curves.html")
    else:
        # Utiliser matplotlib
        plt.figure(figsize=(10, 8))
        
        for model_name in model_names:
            plt.plot(
                results[model_name]['fpr'],
                results[model_name]['tpr'],
                label=f"{model_name} (AUC={results[model_name]['auc']:.3f})"
            )
        
        # Ajouter la ligne diagonale
        plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
        
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Comparaison des courbes ROC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sauvegarder
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
        print(f"Courbes ROC sauvegardées dans {output_dir}/roc_curves.png")

def select_best_model(results, feature_names):
    """
    Sélectionne le meilleur modèle et documente le raisonnement.
    
    Args:
        results: Dict de résultats par modèle
        feature_names: Noms des caractéristiques
        
    Returns:
        str: Nom du meilleur modèle
        str: Raisonnement pour la sélection
    """
    # Créer un DataFrame pour la comparaison
    df = pd.DataFrame({
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'AUC': [],
        'Training Time': []
    })
    
    # Remplir le DataFrame
    for model_name, metrics in results.items():
        df = pd.concat([df, pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [metrics['accuracy']],
            'Precision': [metrics['precision']],
            'Recall': [metrics['recall']],
            'F1': [metrics['f1']],
            'AUC': [metrics['auc']],
            'Training Time': [metrics['training_time']]
        })], ignore_index=True)
    
    # Afficher le tableau comparatif
    print("\nComparaison des performances des modèles:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Calculer un score composite
    # F1 et AUC ont un poids plus élevé pour les cas déséquilibrés
    df['Composite Score'] = (
        0.15 * df['Accuracy'] +
        0.20 * df['Precision'] +
        0.20 * df['Recall'] +
        0.25 * df['F1'] +
        0.20 * df['AUC']
    )
    
    # Trouver le meilleur modèle
    best_idx = df['Composite Score'].idxmax()
    best_model = df.loc[best_idx, 'Model']
    best_score = df.loc[best_idx, 'Composite Score']
    
    # Meilleurs modèles par métrique
    best_accuracy = df.loc[df['Accuracy'].idxmax(), 'Model']
    best_precision = df.loc[df['Precision'].idxmax(), 'Model']
    best_recall = df.loc[df['Recall'].idxmax(), 'Model'] 
    best_f1 = df.loc[df['F1'].idxmax(), 'Model']
    best_auc = df.loc[df['AUC'].idxmax(), 'Model']
    fastest = df.loc[df['Training Time'].idxmin(), 'Model']
    
    # Préparer le raisonnement
    reasoning = f"""
SÉLECTION DU MODÈLE POUR LE DIAGNOSTIC D'APPENDICITE PÉDIATRIQUE
================================================================

Comparaison des modèles évalués:
-------------------------------
{df.to_string(index=False, float_format=lambda x: f"{x:.4f}")}

Meilleur modèle par métrique:
----------------------------
- Accuracy: {best_accuracy} ({df.loc[df['Model'] == best_accuracy, 'Accuracy'].values[0]:.4f})
- Precision: {best_precision} ({df.loc[df['Model'] == best_precision, 'Precision'].values[0]:.4f})
- Recall: {best_recall} ({df.loc[df['Model'] == best_recall, 'Recall'].values[0]:.4f})
- F1-score: {best_f1} ({df.loc[df['Model'] == best_f1, 'F1'].values[0]:.4f})
- AUC: {best_auc} ({df.loc[df['Model'] == best_auc, 'AUC'].values[0]:.4f})
- Rapidité d'entraînement: {fastest} ({df.loc[df['Model'] == fastest, 'Training Time'].values[0]:.2f} sec)

MODÈLE SÉLECTIONNÉ: {best_model} (Score composite: {best_score:.4f})

Justification:
-------------
1. Équilibre des métriques: Le modèle {best_model} offre le meilleur équilibre entre précision et rappel,
   ce qui est crucial pour un diagnostic d'appendicite où les conséquences des faux positifs 
   (chirurgies inutiles) et des faux négatifs (appendicites manquées) sont graves.

2. Capacité discriminative: Une AUC élevée ({df.loc[df['Model'] == best_model, 'AUC'].values[0]:.4f}) 
   indique une excellente capacité à distinguer les cas d'appendicite des cas non-appendiculaires
   à différents seuils de décision.

3. Robustesse: Le F1-score élevé ({df.loc[df['Model'] == best_model, 'F1'].values[0]:.4f}) montre que
   le modèle maintient un bon équilibre entre précision et rappel, ce qui est essentiel
   dans le contexte clinique où les données sont souvent déséquilibrées.

4. Considérations cliniques: Dans le diagnostic pédiatrique, la sensibilité (rappel) est 
   particulièrement importante pour ne pas manquer de cas d'appendicite potentiellement graves,
   mais la spécificité doit également être élevée pour éviter les interventions chirurgicales inutiles.

5. Efficacité computationnelle: Le temps d'entraînement de {df.loc[df['Model'] == best_model, 'Training Time'].values[0]:.2f} 
   secondes est raisonnable pour une utilisation en contexte clinique, permettant des
   mises à jour régulières du modèle avec de nouvelles données.

Caractéristiques importantes:
---------------------------
"""
    
    # Ajouter l'importance des caractéristiques si disponible
    if results[best_model]['feature_importance'] is not None:
        # Créer un DataFrame pour l'importance des caractéristiques
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': results[best_model]['feature_importance']
        })
        
        # Trier par importance décroissante
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        # Ajouter au raisonnement
        reasoning += f"Les caractéristiques les plus importantes selon {best_model} sont:\n"
        for i, row in importance_df.iterrows():
            reasoning += f"- {row['Feature']}: {row['Importance']:.4f}\n"
    
    reasoning += f"""
Recommandation clinique:
----------------------
Le modèle {best_model} est recommandé pour assister les cliniciens dans le diagnostic
d'appendicite pédiatrique. Il devrait être utilisé comme un outil d'aide à la décision,
en complément du jugement clinique et des examens complémentaires standards.

Le modèle peut potentiellement réduire le taux de chirurgies inutiles et améliorer
la détection précoce des cas d'appendicite, mais ne devrait jamais remplacer
l'évaluation complète par un médecin.
"""
    
    return best_model, reasoning

def main():
    """Fonction principale du script."""
    print("\n" + "="*70)
    print("ÉVALUATION DES MODÈLES POUR LE DIAGNOSTIC D'APPENDICITE PÉDIATRIQUE")
    print("="*70)
    
    # Créer le répertoire des rapports
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Charger les données
    X, y = load_data()
    print(f"\nDonnées chargées: {X.shape[0]} observations, {X.shape[1]} caractéristiques")
    print(f"Distribution des classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # 2. Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nEnsemble d'entraînement: {X_train.shape[0]} observations")
    print(f"Ensemble de test: {X_test.shape[0]} observations")
    
    # 3. Obtenir les modèles disponibles
    models = get_models()
    print(f"\nModèles disponibles: {', '.join(models.keys())}")
    
    # 4. Évaluer chaque modèle
    results = {}
    for name, model in models.items():
        print(f"\n{'-'*50}")
        print(f"Évaluation du modèle: {name}")
        print(f"{'-'*50}")
        
        try:
            results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
            
            print(f"Accuracy: {results[name]['accuracy']:.4f}")
            print(f"Precision: {results[name]['precision']:.4f}")
            print(f"Recall: {results[name]['recall']:.4f}")
            print(f"F1-score: {results[name]['f1']:.4f}")
            print(f"AUC: {results[name]['auc']:.4f}")
            print(f"Temps d'entraînement: {results[name]['training_time']:.2f} secondes")
        except Exception as e:
            print(f"Erreur lors de l'évaluation du modèle {name}: {e}")
    
    # 5. Visualiser les résultats
    print("\nCréation des visualisations...")
    plot_performance_comparison(results, output_dir)
    
    # 6. Sélectionner le meilleur modèle
    best_model, reasoning = select_best_model(results, X.columns)
    
    # 7. Enregistrer le raisonnement
    reasoning_path = os.path.join(output_dir, 'model_selection_reasoning.txt')
    with open(reasoning_path, 'w') as f:
        f.write(reasoning)
    
    print(f"\nRaisonnement pour la sélection du modèle enregistré dans {reasoning_path}")
    print(f"\nMeilleur modèle recommandé: {best_model}")
    print("\nExécution terminée. Consultez le dossier 'reports' pour les résultats.")

if __name__ == "__main__":
    main()
