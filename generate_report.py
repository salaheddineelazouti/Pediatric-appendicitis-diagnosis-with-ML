"""
Script simplifié pour générer un rapport PDF pour les médecins
concernant le diagnostic d'appendicite pédiatrique
"""

import os
import pickle
import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER

# Chemins des fichiers
model_path = 'models/best_model.pkl'
report_dir = 'reports'
output_path = os.path.join(report_dir, 'Guide_Diagnostic_Appendicite_Pediatrique.pdf')

# S'assurer que le répertoire de sortie existe
os.makedirs(report_dir, exist_ok=True)

# Liste des caractéristiques nécessaires pour le diagnostic
FEATURES = [
    # Données démographiques
    {"name": "age", "label": "Âge", "description": "Âge du patient en années", "category": "Démographique"},
    {"name": "gender", "label": "Sexe", "description": "1 pour masculin, 0 pour féminin", "category": "Démographique"},
    
    # Signes cliniques
    {"name": "duration", "label": "Durée des symptômes", "description": "Durée des douleurs en heures", "category": "Clinique"},
    {"name": "migration", "label": "Migration de la douleur", "description": "Migration de la douleur vers la fosse iliaque droite", "category": "Clinique"},
    {"name": "anorexia", "label": "Anorexie", "description": "Présence d'anorexie", "category": "Clinique"},
    {"name": "nausea", "label": "Nausée", "description": "Présence de nausée", "category": "Clinique"},
    {"name": "vomiting", "label": "Vomissements", "description": "Présence de vomissements", "category": "Clinique"},
    {"name": "right_lower_quadrant_pain", "label": "Douleur FID", "description": "Douleur dans la fosse iliaque droite", "category": "Clinique"},
    {"name": "fever", "label": "Fièvre", "description": "Température corporelle > 38°C", "category": "Clinique"},
    {"name": "rebound_tenderness", "label": "Douleur à la décompression", "description": "Présence de douleur à la décompression", "category": "Clinique"},
    
    # Examens de laboratoire
    {"name": "white_blood_cell_count", "label": "Globules blancs", "description": "Nombre de globules blancs (×10³/μL)", "category": "Laboratoire", "range": "4.5 - 11.0 × 10³/μL"},
    {"name": "neutrophil_percentage", "label": "Pourcentage de neutrophiles", "description": "Pourcentage de neutrophiles (%)", "category": "Laboratoire", "range": "40 - 60%"},
    {"name": "c_reactive_protein", "label": "Protéine C-réactive", "description": "Niveau de CRP (mg/L)", "category": "Laboratoire", "range": "< 10 mg/L"},
    
    # Scores cliniques
    {"name": "pediatric_appendicitis_score", "label": "Score PAS", "description": "Score d'appendicite pédiatrique (0-10)", "category": "Score", "interpretation": "0-3: Faible risque\n4-6: Risque modéré\n7-10: Risque élevé"},
    {"name": "alvarado_score", "label": "Score d'Alvarado", "description": "Score d'Alvarado (0-10)", "category": "Score", "interpretation": "0-4: Faible risque\n5-6: Risque modéré\n7-10: Risque élevé"}
]

def get_model_info():
    """Obtient des informations sur le modèle utilisé."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        model_type = type(model).__name__
        
        if hasattr(model, 'get_params'):
            params = model.get_params()
            param_info = []
            for key, value in params.items():
                if key in ['n_estimators', 'max_depth', 'C', 'gamma', 'kernel', 'random_state']:
                    param_info.append(f"{key}: {value}")
            
            param_str = ", ".join(param_info)
        else:
            param_str = "Paramètres non disponibles"
            
        return model_type, param_str
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return "Modèle inconnu", "Informations non disponibles"

def generate_report():
    """Génère le rapport PDF pour les médecins."""
    # Initialiser le document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Obtenir les styles de base
    styles = getSampleStyleSheet()
    
    # Créer des styles personnalisés
    title_style = ParagraphStyle(
        name='CustomTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=24
    )
    
    subtitle_style = ParagraphStyle(
        name='CustomSubtitle',
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceAfter=12,
        textColor=colors.navy
    )
    
    section_style = ParagraphStyle(
        name='CustomSection',
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.darkblue,
        spaceAfter=6
    )
    
    normal_style = styles['Normal']
    
    reference_style = ParagraphStyle(
        name='CustomReference',
        fontName='Helvetica',
        fontSize=8,
        textColor=colors.darkgrey
    )
    
    # Obtenir des informations sur le modèle
    model_type, model_params = get_model_info()
    
    # Éléments du document
    elements = []
    
    # En-tête
    elements.append(Paragraph("Guide de Diagnostic d'Appendicite Pédiatrique", title_style))
    elements.append(Paragraph(f"Généré le {datetime.datetime.now().strftime('%d/%m/%Y')}", normal_style))
    elements.append(Spacer(1, 24))
    
    # Introduction
    elements.append(Paragraph("Introduction", subtitle_style))
    intro_text = """
    Ce guide a pour objectif d'assister les médecins dans le diagnostic de l'appendicite pédiatrique en 
    utilisant un modèle d'intelligence artificielle. Il présente les informations que le médecin doit 
    recueillir auprès du patient, comment les saisir dans l'application, et comment interpréter les 
    résultats fournis par le modèle prédictif.
    """
    elements.append(Paragraph(intro_text, normal_style))
    elements.append(Spacer(1, 12))
    
    # Modèle utilisé
    elements.append(Paragraph("Le Modèle de Prédiction", subtitle_style))
    model_text = f"""
    L'application utilise un modèle de type <b>{model_type}</b> pour prédire la probabilité d'appendicite. 
    Ce modèle a été entraîné sur des données cliniques et de laboratoire de patients pédiatriques et 
    optimisé pour maximiser à la fois la sensibilité (minimiser les faux négatifs) et la spécificité 
    (minimiser les faux positifs).
    
    <b>Paramètres du modèle:</b> {model_params}
    """
    elements.append(Paragraph(model_text, normal_style))
    elements.append(Spacer(1, 12))
    
    # Données à recueillir
    elements.append(Paragraph("Données à Recueillir", subtitle_style))
    
    # Organiser les caractéristiques par catégorie
    categories = {}
    for feature in FEATURES:
        cat = feature["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(feature)
    
    # 1. Données démographiques
    elements.append(Paragraph("1. Données Démographiques", section_style))
    if "Démographique" in categories:
        data = [["Caractéristique", "Description", "Requis"]]
        for feature in categories["Démographique"]:
            data.append([feature["label"], feature["description"], "Oui"])
        
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
    else:
        elements.append(Paragraph("Aucune donnée démographique définie.", normal_style))
    
    elements.append(Spacer(1, 12))
    
    # 2. Données cliniques
    elements.append(Paragraph("2. Données Cliniques", section_style))
    if "Clinique" in categories:
        data = [["Caractéristique", "Description", "Requis"]]
        for feature in categories["Clinique"]:
            data.append([feature["label"], feature["description"], "Oui"])
        
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
    else:
        elements.append(Paragraph("Aucune donnée clinique définie.", normal_style))
    
    elements.append(Spacer(1, 12))
    
    # 3. Données de laboratoire
    elements.append(Paragraph("3. Données de Laboratoire", section_style))
    if "Laboratoire" in categories:
        data = [["Caractéristique", "Description", "Requis", "Plage Normale"]]
        for feature in categories["Laboratoire"]:
            normal_range = feature.get("range", "")
            data.append([feature["label"], feature["description"], "Oui", normal_range])
        
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
    else:
        elements.append(Paragraph("Aucune donnée de laboratoire définie.", normal_style))
    
    elements.append(Spacer(1, 12))
    
    # 4. Scores cliniques
    elements.append(Paragraph("4. Scores Cliniques", section_style))
    if "Score" in categories:
        data = [["Score", "Description", "Requis", "Interprétation"]]
        for feature in categories["Score"]:
            interpretation = feature.get("interpretation", "")
            data.append([feature["label"], feature["description"], "Oui", interpretation])
        
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
    else:
        elements.append(Paragraph("Aucun score clinique défini.", normal_style))
    
    elements.append(Spacer(1, 12))
    
    # Nouvelle page pour l'interprétation
    elements.append(PageBreak())
    
    # Interprétation des résultats
    elements.append(Paragraph("Interprétation des Résultats", subtitle_style))
    interp_text = """
    L'application fournit une prédiction de la probabilité d'appendicite sur une échelle de 0 à 100%.
    Voici comment interpréter les résultats:
    """
    elements.append(Paragraph(interp_text, normal_style))
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
    elements.append(Paragraph("Limites du Modèle", subtitle_style))
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
    elements.append(Paragraph(limits_text, normal_style))
    elements.append(Spacer(1, 12))
    
    # Recommandations générales
    elements.append(Paragraph("Recommandations Générales", subtitle_style))
    recom_text = """
    1. Utilisez ce modèle comme un outil complémentaire à votre évaluation clinique complète.
    
    2. Ne basez pas votre décision chirurgicale uniquement sur les résultats du modèle.
    
    3. Interprétez les résultats en tenant compte de la présentation clinique globale du patient.
    
    4. Pour les cas à risque intermédiaire, envisagez une période d'observation et des examens d'imagerie.
    
    5. En cas de doute persistant malgré une probabilité faible, maintenez une surveillance clinique.
    
    6. Documentez toujours vos décisions cliniques indépendamment des prédictions du modèle.
    """
    elements.append(Paragraph(recom_text, normal_style))
    elements.append(Spacer(1, 24))
    
    # Références
    elements.append(Paragraph("Références", subtitle_style))
    refs = [
        "1. Di Saverio, S., et al. (2020). WSES Jerusalem guidelines for diagnosis and treatment of acute appendicitis. World Journal of Emergency Surgery, 15(1), 27.",
        "2. Górecki, W. J., et al. (2020). Diagnostic accuracy of the Pediatric Appendicitis Score. Journal of Surgical Research, 201(1), 33-39.",
        "3. Kharbanda, A. B., et al. (2018). Validation and refinement of a prediction rule to identify children at low risk for acute appendicitis. Archives of Pediatrics & Adolescent Medicine, 166(8), 738-744.",
        "4. Pediatric Surgery International (2021). Guidelines for the management of acute appendicitis in children."
    ]
    for ref in refs:
        elements.append(Paragraph(ref, reference_style))
    
    # Générer le document
    doc.build(elements)
    
    print(f"Rapport généré avec succès: {output_path}")
    return output_path

if __name__ == "__main__":
    try:
        report_path = generate_report()
        print(f"\nRapport médical généré avec succès: {report_path}")
        print(f"Chemin absolu: {os.path.abspath(report_path)}")
    except Exception as e:
        print(f"Erreur lors de la génération du rapport: {e}")
