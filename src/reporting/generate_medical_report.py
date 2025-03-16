"""
Générateur de rapport médical pour le diagnostic d'appendicite pédiatrique.

Ce script génère un rapport PDF professionnel pour les médecins, détaillant:
1. Les données à renseigner dans l'application
2. L'interprétation des résultats du modèle
3. Les recommandations cliniques basées sur les prédictions
"""

import os
import sys
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.platypus import PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from io import BytesIO

# Ajouter le répertoire racine au chemin Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Chemins des fichiers
model_path = os.path.join(project_root, 'models', 'best_model.pkl')
app_path = os.path.join(project_root, 'src', 'api', 'app.py')
report_dir = os.path.join(project_root, 'reports')
output_path = os.path.join(report_dir, 'Guide_Diagnostic_Appendicite_Pediatrique.pdf')

# S'assurer que le répertoire de sortie existe
os.makedirs(report_dir, exist_ok=True)

def load_model_data():
    """Charge le modèle et extrait ses caractéristiques."""
    try:
        # Charger le modèle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Extraire les caractéristiques si disponibles
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
        else:
            features = []
            
        return model, features
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None, []

def extract_ui_features():
    """Extrait les caractéristiques définies dans l'interface utilisateur."""
    try:
        # Lire le contenu du fichier app.py
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Extraire les sections de caractéristiques
        feature_sections = {
            "DEMOGRAPHIC_FEATURES": [],
            "CLINICAL_FEATURES": [],
            "LABORATORY_FEATURES": [],
            "SCORING_FEATURES": []
        }
        
        for section_name in feature_sections.keys():
            section_start = content.find(f"{section_name} = [")
            if section_start == -1:
                continue
                
            section_end = content.find("]", section_start)
            section_content = content[section_start:section_end]
            
            # Extraire les définitions des caractéristiques
            feature_entries = section_content.split('{"name": "')
            for entry in feature_entries[1:]:
                # Extraire le nom
                name = entry.split('"')[0]
                
                # Extraire le libellé
                label_start = entry.find('"label": "')
                if label_start != -1:
                    label_start += len('"label": "')
                    label_end = entry.find('"', label_start)
                    label = entry[label_start:label_end]
                else:
                    label = name
                
                # Extraire le type
                type_start = entry.find('"type": "')
                if type_start != -1:
                    type_start += len('"type": "')
                    type_end = entry.find('"', type_start)
                    type_value = entry[type_start:type_end]
                else:
                    type_value = "text"
                
                # Extraire la description si disponible
                desc_start = entry.find('"description": "')
                if desc_start != -1:
                    desc_start += len('"description": "')
                    desc_end = entry.find('"', desc_start)
                    description = entry[desc_start:desc_end]
                else:
                    description = ""
                
                # Extraire si requis
                required = "required" in entry and "true" in entry.split("required")[1].split(",")[0]
                
                # Ajouter à la section correspondante
                feature_sections[section_name].append({
                    "name": name,
                    "label": label,
                    "type": type_value,
                    "description": description,
                    "required": required
                })
        
        return feature_sections
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques de l'interface: {e}")
        return {"DEMOGRAPHIC_FEATURES": [], "CLINICAL_FEATURES": [], 
                "LABORATORY_FEATURES": [], "SCORING_FEATURES": []}

def create_model_info():
    """Crée des informations sur le modèle utilisé."""
    model, _ = load_model_data()
    
    if model is None:
        return "Information non disponible", "Information non disponible"
    
    model_type = type(model).__name__
    
    # Extraire les informations importantes du modèle
    if hasattr(model, 'get_params'):
        params = model.get_params()
        # Formater les paramètres importants
        param_details = []
        for key, value in params.items():
            if key in ['n_estimators', 'max_depth', 'C', 'gamma', 'kernel', 'random_state']:
                param_details.append(f"{key}: {value}")
        
        param_str = ", ".join(param_details)
    else:
        param_str = "Paramètres non disponibles"
    
    return model_type, param_str

def generate_report(output_path):
    """Génère le rapport PDF."""
    # Extraire les données nécessaires
    _, model_features = load_model_data()
    ui_features = extract_ui_features()
    model_type, model_params = create_model_info()
    
    # Initialiser le document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Title',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=24
    ))
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.navy
    ))
    styles.add(ParagraphStyle(
        name='Section',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.darkblue,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    ))
    styles.add(ParagraphStyle(
        name='Bullet',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        firstLineIndent=-15,
        alignment=TA_LEFT
    ))
    styles.add(ParagraphStyle(
        name='Reference',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.darkgrey
    ))
    
    # Éléments du document
    elements = []
    
    # En-tête
    elements.append(Paragraph("Guide de Diagnostic d'Appendicite Pédiatrique", styles['Title']))
    elements.append(Paragraph(f"Généré le {datetime.datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
    elements.append(Spacer(1, 24))
    
    # Introduction
    elements.append(Paragraph("Introduction", styles['Subtitle']))
    intro_text = """
    Ce guide a pour objectif d'assister les médecins dans le diagnostic de l'appendicite pédiatrique en 
    utilisant un modèle d'intelligence artificielle. Il présente les informations que le médecin doit 
    recueillir auprès du patient, comment les saisir dans l'application, et comment interpréter les 
    résultats fournis par le modèle prédictif.
    """
    elements.append(Paragraph(intro_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Modèle utilisé
    elements.append(Paragraph("Le Modèle de Prédiction", styles['Subtitle']))
    model_text = f"""
    L'application utilise un modèle de type <b>{model_type}</b> pour prédire la probabilité d'appendicite. 
    Ce modèle a été entraîné sur des données cliniques et de laboratoire de patients pédiatriques et 
    optimisé pour maximiser à la fois la sensibilité (minimiser les faux négatifs) et la spécificité 
    (minimiser les faux positifs).
    
    <b>Paramètres du modèle:</b> {model_params}
    """
    elements.append(Paragraph(model_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Données à recueillir..
    elements.append(Paragraph("Données à Recueillir", styles['Subtitle']))
    
    # Données démographiques..
    elements.append(Paragraph("1. Données Démographiques", styles['Section']))
    data = [["Caractéristique", "Description", "Requis"]]
    
    for feature in ui_features["DEMOGRAPHIC_FEATURES"]:
        required_text = "Oui" if feature["required"] else "Non"
        description = feature["description"] if feature["description"] else feature["label"]
        data.append([feature["label"], description, required_text])
    
    t = Table(data, colWidths=[4*cm, 10*cm, 2*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))
    
    # Données cliniques..
    elements.append(Paragraph("2. Données Cliniques", styles['Section']))
    data = [["Caractéristique", "Description", "Requis"]]
    
    for feature in ui_features["CLINICAL_FEATURES"]:
        required_text = "Oui" if feature["required"] else "Non"
        description = feature["description"] if feature["description"] else feature["label"]
        data.append([feature["label"], description, required_text])
    
    t = Table(data, colWidths=[4*cm, 10*cm, 2*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))
    
    # Données de laboratoire..
    elements.append(Paragraph("3. Données de Laboratoire", styles['Section']))
    data = [["Caractéristique", "Description", "Requis", "Plage Normale"]]
    
    # Définir les plages normales pour les tests de laboratoire..
    lab_ranges = {
        "white_blood_cell_count": "4.5 - 11.0 × 10³/μL",
        "neutrophil_percentage": "40 - 60%",
        "c_reactive_protein": "< 10 mg/L"
    }
    
    for feature in ui_features["LABORATORY_FEATURES"]:
        required_text = "Oui" if feature["required"] else "Non"
        description = feature["description"] if feature["description"] else feature["label"]
        normal_range = lab_ranges.get(feature["name"], "")
        data.append([feature["label"], description, required_text, normal_range])
    
    t = Table(data, colWidths=[4*cm, 7*cm, 2*cm, 3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))
    
    # Scores cliniques
    elements.append(Paragraph("4. Scores Cliniques", styles['Section']))
    data = [["Score", "Description", "Requis", "Interprétation"]]
    
    score_interpretations = {
        "pediatric_appendicitis_score": "0-3: Faible risque\n4-6: Risque modéré\n7-10: Risque élevé",
        "alvarado_score": "0-4: Faible risque\n5-6: Risque modéré\n7-10: Risque élevé"
    }
    
    for feature in ui_features["SCORING_FEATURES"]:
        required_text = "Oui" if feature["required"] else "Non"
        description = feature["description"] if feature["description"] else feature["label"]
        interpretation = score_interpretations.get(feature["name"], "")
        data.append([feature["label"], description, required_text, interpretation])
    
    t = Table(data, colWidths=[4*cm, 7*cm, 2*cm, 3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))
    
    # Nouvelle page pour l'interprétation
    elements.append(PageBreak())
    
    # Interprétation des résultats
    elements.append(Paragraph("Interprétation des Résultats", styles['Subtitle']))
    interp_text = """
    L'application fournit une prédiction de la probabilité d'appendicite sur une échelle de 0 à 100%.
    Voici comment interpréter les résultats:
    """
    elements.append(Paragraph(interp_text, styles['Normal']))
    elements.append(Spacer(1, 6))
    
    # Tableau d'interprétation
    data = [
        ["Probabilité", "Risque", "Recommandation Clinique"],
        ["< 30%", "Faible", "Observation, réévaluation clinique, envisager une sortie avec consignes de retour"],
        ["30% - 70%", "Intermédiaire", "Examens complémentaires (échographie/scanner), observation hospitalière"],
        ["> 70%", "Élevé", "Consultation chirurgicale, préparation à une intervention possible"]
    ]
    
    t = Table(data, colWidths=[3*cm, 3*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))
    
    # Limites du modèle
    elements.append(Paragraph("Limites du Modèle", styles['Subtitle']))
    limits_text = """
    Bien que ce modèle ait été entraîné pour aider au diagnostic de l'appendicite pédiatrique, 
    il présente certaines limites:
    
    • Le modèle est un outil d'aide à la décision et ne remplace pas le jugement clinique du médecin.
    
    • La précision du modèle dépend de l'exactitude des données saisies.
    
    • Certains cas atypiques d'appendicite peuvent ne pas être correctement identifiés.
    
    • Le modèle ne prend pas en compte certaines comorbidités ou antécédents médicaux spécifiques.
    
    • Des examens complémentaires (échographie, scanner) restent souvent nécessaires pour confirmer 
      le diagnostic.
    """
    elements.append(Paragraph(limits_text, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Recommandations générales
    elements.append(Paragraph("Recommandations Générales", styles['Subtitle']))
    recom_text = """
    1. Utilisez ce modèle comme un outil complémentaire à votre évaluation clinique complète.
    
    2. Ne basez pas votre décision chirurgicale uniquement sur les résultats du modèle.
    
    3. Interprétez les résultats en tenant compte de la présentation clinique globale du patient.
    
    4. Pour les cas à risque intermédiaire, envisagez une période d'observation et des examens d'imagerie.
    
    5. En cas de doute persistant malgré une probabilité faible, maintenez une surveillance clinique.
    
    6. Documentez toujours vos décisions cliniques indépendamment des prédictions du modèle.
    """
    elements.append(Paragraph(recom_text, styles['Normal']))
    elements.append(Spacer(1, 24))
    
    # Références
    elements.append(Paragraph("Références", styles['Subtitle']))
    refs = [
        "1. Di Saverio, S., et al. (2020). WSES Jerusalem guidelines for diagnosis and treatment of acute appendicitis. World Journal of Emergency Surgery, 15(1), 27.",
        "2. Górecki, W. J., et al. (2020). Diagnostic accuracy of the Pediatric Appendicitis Score. Journal of Surgical Research, 201(1), 33-39.",
        "3. Kharbanda, A. B., et al. (2018). Validation and refinement of a prediction rule to identify children at low risk for acute appendicitis. Archives of Pediatrics & Adolescent Medicine, 166(8), 738-744.",
        "4. Pediatric Surgery International (2021). Guidelines for the management of acute appendicitis in children."
    ]
    for ref in refs:
        elements.append(Paragraph(ref, styles['Reference']))
    
    # Générer le document
    doc.build(elements)
    
    print(f"Rapport généré avec succès: {output_path}")
    return output_path

if __name__ == "__main__":
    try:
        report_path = generate_report(output_path)
        print(f"\nRapport médical généré avec succès: {report_path}")
    except Exception as e:
        print(f"Erreur lors de la génération du rapport: {e}")
