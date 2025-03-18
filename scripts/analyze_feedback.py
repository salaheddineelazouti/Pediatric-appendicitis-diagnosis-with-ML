"""
Analyse des retours utilisateurs pour améliorer le modèle de diagnostic de l'appendicite pédiatrique.
Ce script analyse les fichiers JSON de feedback et génère des rapports sur la précision du modèle
et l'utilité des explications fournies.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from collections import Counter

# Chemin vers le répertoire des retours utilisateurs
FEEDBACK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feedback')
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')

def load_feedback_data():
    """
    Charge tous les fichiers de retour utilisateur du répertoire feedback.
    """
    feedback_data = []
    
    if not os.path.exists(FEEDBACK_DIR):
        print(f"Le répertoire {FEEDBACK_DIR} n'existe pas.")
        return pd.DataFrame()
    
    for filename in os.listdir(FEEDBACK_DIR):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(FEEDBACK_DIR, filename), 'r') as f:
                    data = json.load(f)
                    feedback_data.append(data)
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier {filename}: {str(e)}")
    
    if not feedback_data:
        print("Aucun retour utilisateur trouvé.")
        return pd.DataFrame()
    
    # Convertir en DataFrame pour faciliter l'analyse
    return pd.DataFrame(feedback_data)

def analyze_diagnostic_accuracy(df):
    """
    Analyse la précision du diagnostic prédictif selon les retours utilisateurs.
    """
    if df.empty:
        return
    
    # Convertir les évaluations en valeurs numériques
    df['diagnostic_accuracy'] = pd.to_numeric(df['diagnostic_accuracy'], errors='coerce')
    
    # Calculer les métriques
    avg_accuracy = df['diagnostic_accuracy'].mean()
    median_accuracy = df['diagnostic_accuracy'].median()
    
    # Créer un histogramme de la distribution des précisions
    plt.figure(figsize=(10, 6))
    sns.histplot(df['diagnostic_accuracy'], bins=5, kde=True)
    plt.title('Distribution des évaluations de précision du diagnostic')
    plt.xlabel('Évaluation (1-5)')
    plt.ylabel('Fréquence')
    
    # Ajouter une ligne verticale pour la moyenne
    plt.axvline(x=avg_accuracy, color='r', linestyle='--', label=f'Moyenne: {avg_accuracy:.2f}')
    plt.legend()
    
    # Enregistrer le graphique
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    
    plt.savefig(os.path.join(REPORTS_DIR, f'diagnostic_accuracy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    plt.close()
    
    # Analyser les cas où le modèle s'est trompé
    incorrect_predictions = df[df['diagnostic_accuracy'] <= 2]
    correct_predictions = df[df['diagnostic_accuracy'] >= 4]
    
    print(f"\nMétrique de précision du diagnostic:")
    print(f"Nombre total de retours: {len(df)}")
    print(f"Moyenne de précision: {avg_accuracy:.2f}/5")
    print(f"Médiane de précision: {median_accuracy}/5")
    print(f"Nombre de prédictions considérées comme incorrectes (≤ 2): {len(incorrect_predictions)}")
    print(f"Nombre de prédictions considérées comme correctes (≥ 4): {len(correct_predictions)}")
    
    # Analyser la concordance entre prédiction et diagnostic réel
    concordance_analysis = []
    
    for _, row in df.iterrows():
        pred_class = row.get('prediction_class', '')
        actual_diag = row.get('actual_diagnosis', '')
        
        if pred_class == 'Élevé' and actual_diag == 'appendicite':
            concordance = 'Vrai Positif'
        elif pred_class == 'Faible' and actual_diag == 'non_appendicite':
            concordance = 'Vrai Négatif'
        elif pred_class == 'Élevé' and actual_diag == 'non_appendicite':
            concordance = 'Faux Positif'
        elif pred_class == 'Faible' and actual_diag == 'appendicite':
            concordance = 'Faux Négatif'
        else:
            concordance = 'Indéterminé'
        
        concordance_analysis.append(concordance)
    
    concordance_counts = Counter(concordance_analysis)
    
    print("\nAnalyse de concordance:")
    for category, count in concordance_counts.items():
        print(f"{category}: {count}")
    
    # Visualiser la concordance
    plt.figure(figsize=(10, 6))
    sns.countplot(x=concordance_analysis)
    plt.title('Concordance entre prédiction et diagnostic réel')
    plt.xlabel('Catégorie')
    plt.ylabel('Nombre')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f'concordance_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    plt.close()

def analyze_explanation_usefulness(df):
    """
    Analyse l'utilité des explications fournies selon les retours utilisateurs.
    """
    if df.empty:
        return
    
    # Convertir les évaluations en valeurs numériques
    df['usefulness_rating'] = pd.to_numeric(df['usefulness_rating'], errors='coerce')
    
    # Calculer les métriques
    avg_usefulness = df['usefulness_rating'].mean()
    median_usefulness = df['usefulness_rating'].median()
    
    # Créer un histogramme de la distribution des utilités
    plt.figure(figsize=(10, 6))
    sns.histplot(df['usefulness_rating'], bins=5, kde=True)
    plt.title('Distribution des évaluations d\'utilité des explications')
    plt.xlabel('Évaluation (1-5)')
    plt.ylabel('Fréquence')
    
    # Ajouter une ligne verticale pour la moyenne
    plt.axvline(x=avg_usefulness, color='r', linestyle='--', label=f'Moyenne: {avg_usefulness:.2f}')
    plt.legend()
    
    # Enregistrer le graphique
    plt.savefig(os.path.join(REPORTS_DIR, f'explanation_usefulness_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    plt.close()
    
    print(f"\nMétrique d'utilité des explications:")
    print(f"Moyenne d'utilité: {avg_usefulness:.2f}/5")
    print(f"Médiane d'utilité: {median_usefulness}/5")
    
    # Relation entre précision du diagnostic et utilité des explications
    if 'diagnostic_accuracy' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='diagnostic_accuracy', y='usefulness_rating')
        plt.title('Relation entre précision du diagnostic et utilité des explications')
        plt.xlabel('Précision du diagnostic (1-5)')
        plt.ylabel('Utilité des explications (1-5)')
        
        # Ajouter une ligne de tendance
        sns.regplot(data=df, x='diagnostic_accuracy', y='usefulness_rating', scatter=False, color='red')
        
        # Calculer la corrélation
        correlation = df['diagnostic_accuracy'].corr(df['usefulness_rating'])
        plt.annotate(f'Corrélation: {correlation:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.savefig(os.path.join(REPORTS_DIR, f'accuracy_usefulness_relation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        plt.close()
        
        print(f"\nCorrélation entre précision et utilité: {correlation:.2f}")

def extract_common_themes_from_comments(df):
    """
    Analyse les commentaires pour identifier les thèmes communs.
    Cette fonction est simplifiée et pourrait être améliorée avec du NLP.
    """
    if df.empty or 'comments' not in df.columns:
        return
    
    # Filtrer les commentaires non vides
    comments = df['comments'].dropna().tolist()
    comments = [c for c in comments if c.strip() != '']
    
    if not comments:
        print("\nAucun commentaire à analyser.")
        return
    
    print(f"\nAnalyse des {len(comments)} commentaires:")
    
    # Afficher quelques commentaires à titre d'exemple
    print("\nExemples de commentaires:")
    for i, comment in enumerate(comments[:5]):
        print(f"{i+1}. {comment[:100]}..." if len(comment) > 100 else f"{i+1}. {comment}")
    
    # Pour une analyse plus poussée, il faudrait intégrer du NLP
    # Ici, nous nous contentons d'afficher les commentaires

def generate_report(df):
    """
    Génère un rapport HTML complet de l'analyse des retours utilisateurs.
    """
    if df.empty:
        return
    
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f'feedback_analysis_report_{report_time}.html')
    
    # Calculer les statistiques de base
    total_feedback = len(df)
    avg_accuracy = df['diagnostic_accuracy'].mean() if 'diagnostic_accuracy' in df.columns else "N/A"
    avg_usefulness = df['usefulness_rating'].mean() if 'usefulness_rating' in df.columns else "N/A"
    
    # Créer le contenu HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport d'analyse des retours utilisateurs</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .metrics {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; }}
            .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; margin: 10px; min-width: 200px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
            .metric-title {{ font-size: 1.2em; color: #7f8c8d; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 0.9em; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; display: block; }}
        </style>
    </head>
    <body>
        <h1>Rapport d'analyse des retours utilisateurs</h1>
        <p>Généré le {datetime.now().strftime("%d/%m/%Y à %H:%M:%S")}</p>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{total_feedback}</div>
                <div class="metric-title">Retours totaux</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_accuracy:.2f}/5</div>
                <div class="metric-title">Précision moyenne</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_usefulness:.2f}/5</div>
                <div class="metric-title">Utilité moyenne</div>
            </div>
        </div>
        
        <h2>Détails des retours</h2>
        <table>
            <thead>
                <tr>
                    <th>ID Rapport</th>
                    <th>Date</th>
                    <th>Précision</th>
                    <th>Diagnostic réel</th>
                    <th>Utilité</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Ajouter les lignes du tableau
    for _, row in df.iterrows():
        html_content += f"""
                <tr>
                    <td>{row.get('report_id', 'N/A')}</td>
                    <td>{row.get('timestamp', 'N/A')}</td>
                    <td>{row.get('diagnostic_accuracy', 'N/A')}/5</td>
                    <td>{row.get('actual_diagnosis', 'N/A')}</td>
                    <td>{row.get('usefulness_rating', 'N/A')}/5</td>
                </tr>
        """
    
    html_content += """
            </tbody>
        </table>
        
        <h2>Images d'analyse</h2>
        <p>Ces graphiques seront disponibles après l'exécution de l'analyse complète.</p>
        
        <div class="footer">
            <p>© 2025 Système de diagnostic d'appendicite pédiatrique</p>
        </div>
    </body>
    </html>
    """
    
    # Enregistrer le rapport HTML
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nRapport HTML généré: {report_path}")
    
    return report_path

def main():
    """
    Fonction principale qui exécute l'analyse complète.
    """
    print("Analyse des retours utilisateurs...")
    
    # Charger les données
    feedback_df = load_feedback_data()
    
    if feedback_df.empty:
        print("Aucune donnée de retour utilisateur à analyser.")
        return
    
    print(f"Nombre de retours utilisateurs chargés: {len(feedback_df)}")
    
    # Analyser la précision du diagnostic
    analyze_diagnostic_accuracy(feedback_df)
    
    # Analyser l'utilité des explications
    analyze_explanation_usefulness(feedback_df)
    
    # Analyser les commentaires
    extract_common_themes_from_comments(feedback_df)
    
    # Générer le rapport
    report_path = generate_report(feedback_df)
    
    print("\nAnalyse terminée !")
    if report_path:
        print(f"Consultez le rapport complet: {report_path}")

if __name__ == "__main__":
    main()
