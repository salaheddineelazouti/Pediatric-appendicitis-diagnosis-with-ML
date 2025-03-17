"""
Utilitaires de calibration pour améliorer la précision des probabilités prédites.

Ce module fournit des outils pour:
1. Calibrer les modèles d'apprentissage automatique
2. Évaluer la qualité de la calibration
3. Appliquer différentes méthodes de calibration (Platt, isotonique)
4. Visualiser les courbes de calibration
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging
from typing import Dict, Tuple, Any, List, Optional, Union

# Configuration du logging
logger = logging.getLogger('calibration')

class ModelCalibrator:
    """Classe pour calibrer un modèle et améliorer la qualité des probabilités prédites."""
    
    def __init__(self, model: Any, method: str = 'sigmoid', cv: int = 5):
        """
        Initialise un calibrateur de modèle.
        
        Args:
            model: Le modèle à calibrer
            method: Méthode de calibration ('sigmoid' pour Platt scaling ou 'isotonic')
            cv: Nombre de plis pour la validation croisée
        """
        self.model = model
        self.method = method
        self.cv = cv
        self.calibrated_model = None
        self.is_pipeline = hasattr(model, 'named_steps')
        
        logger.info(f"Initialisation du calibrateur avec méthode: {method}, cv: {cv}")
        logger.info(f"Type de modèle: {'Pipeline' if self.is_pipeline else type(model).__name__}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> 'ModelCalibrator':
        """
        Calibre le modèle en utilisant les données fournies.
        
        Args:
            X: Données d'entraînement
            y: Cibles d'entraînement
            X_val: Données de validation (facultatif)
            y_val: Cibles de validation (facultatif)
            
        Returns:
            Self pour permettre le chaînage des méthodes
        """
        try:
            logger.info(f"Début de la calibration du modèle avec {X.shape[0]} échantillons")
            
            # Si le modèle est un pipeline
            if self.is_pipeline:
                # Si c'est un pipeline, extraire l'estimateur final et le calibrer
                final_estimator_name = list(self.model.named_steps.keys())[-1]
                final_estimator = self.model.named_steps[final_estimator_name]
                
                # Créer un nouveau pipeline avec tous les transformateurs, suivi du modèle calibré
                transformers = [(name, transformer) for name, transformer in self.model.named_steps.items() 
                               if name != final_estimator_name]
                
                # Si des données de validation sont fournies, on les utilise pour la calibration
                if X_val is not None and y_val is not None:
                    logger.info("Utilisation des données de validation fournies pour la calibration")
                    
                    # Appliquer les transformateurs sur les données de validation
                    X_val_transformed = X_val.copy()
                    for name, transformer in transformers:
                        X_val_transformed = pd.DataFrame(
                            transformer.transform(X_val_transformed),
                            columns=transformer.get_feature_names_out() if hasattr(transformer, 'get_feature_names_out') else X_val_transformed.columns
                        )
                    
                    # Créer un calibrateur préentraîné
                    calibrated_estimator = CalibratedClassifierCV(
                        estimator=final_estimator,
                        method=self.method,
                        cv='prefit',  # Le modèle est déjà entraîné
                    )
                    
                    # Ajuster le calibrateur sur les données de validation transformées
                    calibrated_estimator.fit(X_val_transformed, y_val)
                    
                else:
                    logger.info(f"Utilisation de validation croisée ({self.cv} plis) pour la calibration")
                    # Créer un calibrateur avec validation croisée
                    calibrated_estimator = CalibratedClassifierCV(
                        estimator=final_estimator,
                        method=self.method,
                        cv=self.cv,
                    )
                    
                    # Appliquer les transformateurs sur les données d'entraînement
                    X_transformed = X.copy()
                    for name, transformer in transformers:
                        X_transformed = pd.DataFrame(
                            transformer.transform(X_transformed),
                            columns=transformer.get_feature_names_out() if hasattr(transformer, 'get_feature_names_out') else X_transformed.columns
                        )
                    
                    # Ajuster le calibrateur sur les données d'entraînement transformées
                    calibrated_estimator.fit(X_transformed, y)
                
                # Créer un nouveau pipeline avec les transformateurs et le modèle calibré
                pipeline_steps = transformers + [('calibrated_estimator', calibrated_estimator)]
                self.calibrated_model = Pipeline(steps=pipeline_steps)
                
            else:
                # Si ce n'est pas un pipeline, calibrer directement le modèle
                if X_val is not None and y_val is not None:
                    # Utiliser les données de validation
                    calibrated_model = CalibratedClassifierCV(
                        estimator=self.model,
                        method=self.method,
                        cv='prefit',  # Le modèle est déjà entraîné
                    )
                    calibrated_model.fit(X_val, y_val)
                else:
                    # Utiliser la validation croisée
                    calibrated_model = CalibratedClassifierCV(
                        estimator=self.model,
                        method=self.method,
                        cv=self.cv,
                    )
                    calibrated_model.fit(X, y)
                
                self.calibrated_model = calibrated_model
            
            logger.info("Modèle calibré avec succès")
            return self
            
        except Exception as e:
            logger.error(f"Erreur lors de la calibration du modèle: {str(e)}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prédit les probabilités calibrées pour les données d'entrée.
        
        Args:
            X: Données d'entrée
            
        Returns:
            Tableau de probabilités prédites
        """
        if self.calibrated_model is None:
            raise ValueError("Le modèle n'a pas encore été calibré. Appelez d'abord la méthode 'fit'.")
        
        return self.calibrated_model.predict_proba(X)
    
    def evaluate_calibration(self, X: pd.DataFrame, y: pd.Series, n_bins: int = 10) -> Dict[str, Any]:
        """
        Évalue la qualité de la calibration en comparant le modèle original et le modèle calibré.
        
        Args:
            X: Données d'évaluation
            y: Cibles réelles
            n_bins: Nombre de bins pour la courbe de calibration
            
        Returns:
            Dictionnaire contenant les métriques d'évaluation de la calibration
        """
        if self.calibrated_model is None:
            raise ValueError("Le modèle n'a pas encore été calibré. Appelez d'abord la méthode 'fit'.")
        
        # Prédire les probabilités avec le modèle original
        y_prob_orig = self.model.predict_proba(X)[:, 1]
        
        # Prédire les probabilités avec le modèle calibré
        y_prob_cal = self.calibrated_model.predict_proba(X)[:, 1]
        
        # Calculer le Brier score
        brier_orig = brier_score_loss(y, y_prob_orig)
        brier_cal = brier_score_loss(y, y_prob_cal)
        
        # Calculer les courbes de calibration
        prob_true_orig, prob_pred_orig = calibration_curve(y, y_prob_orig, n_bins=n_bins)
        prob_true_cal, prob_pred_cal = calibration_curve(y, y_prob_cal, n_bins=n_bins)
        
        # Calculer la distribution des probabilités
        orig_prob_dist = self._calculate_prob_distribution(y_prob_orig)
        cal_prob_dist = self._calculate_prob_distribution(y_prob_cal)
        
        return {
            'brier_score_original': brier_orig,
            'brier_score_calibrated': brier_cal,
            'calibration_curve_original': {
                'prob_true': prob_true_orig,
                'prob_pred': prob_pred_orig
            },
            'calibration_curve_calibrated': {
                'prob_true': prob_true_cal,
                'prob_pred': prob_pred_cal
            },
            'prob_distribution_original': orig_prob_dist,
            'prob_distribution_calibrated': cal_prob_dist,
            'improvement': (brier_orig - brier_cal) / brier_orig * 100  # Pourcentage d'amélioration
        }
    
    def _calculate_prob_distribution(self, probs: np.ndarray) -> Dict[str, int]:
        """
        Calcule la distribution des probabilités prédites.
        
        Args:
            probs: Tableau de probabilités
            
        Returns:
            Dictionnaire contenant la distribution des probabilités par intervalles
        """
        return {
            '0.0-0.2': np.sum((probs >= 0.0) & (probs < 0.2)),
            '0.2-0.4': np.sum((probs >= 0.2) & (probs < 0.4)),
            '0.4-0.6': np.sum((probs >= 0.4) & (probs < 0.6)),
            '0.6-0.8': np.sum((probs >= 0.6) & (probs < 0.8)),
            '0.8-1.0': np.sum((probs >= 0.8) & (probs <= 1.0))
        }
    
    def plot_calibration_curve(self, X: pd.DataFrame, y: pd.Series, n_bins: int = 10, 
                              output_path: Optional[str] = None) -> plt.Figure:
        """
        Crée un graphique de la courbe de calibration comparant le modèle original et le modèle calibré.
        
        Args:
            X: Données d'évaluation
            y: Cibles réelles
            n_bins: Nombre de bins pour la courbe de calibration
            output_path: Chemin où sauvegarder le graphique (facultatif)
            
        Returns:
            Figure matplotlib
        """
        # Calculer les métriques de calibration
        eval_results = self.evaluate_calibration(X, y, n_bins)
        
        # Créer la figure
        plt.figure(figsize=(10, 8))
        
        # Tracer la ligne diagonale de référence (calibration parfaite)
        plt.plot([0, 1], [0, 1], 'k--', label='Calibration parfaite')
        
        # Tracer la courbe de calibration du modèle original
        plt.plot(
            eval_results['calibration_curve_original']['prob_pred'],
            eval_results['calibration_curve_original']['prob_true'],
            's-', label=f'Modèle original (Brier: {eval_results["brier_score_original"]:.3f})'
        )
        
        # Tracer la courbe de calibration du modèle calibré
        plt.plot(
            eval_results['calibration_curve_calibrated']['prob_pred'],
            eval_results['calibration_curve_calibrated']['prob_true'],
            's-', label=f'Modèle calibré ({self.method}) (Brier: {eval_results["brier_score_calibrated"]:.3f})'
        )
        
        # Configurer le graphique
        plt.xlabel('Probabilité moyenne prédite par bin')
        plt.ylabel('Ratio des positifs réels')
        plt.title('Courbe de calibration')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Sauvegarder si un chemin est fourni
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        fig = plt.gcf()
        plt.close()
        
        return fig
    
    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle calibré dans un fichier pickle.
        
        Args:
            path: Chemin où sauvegarder le modèle
        """
        if self.calibrated_model is None:
            raise ValueError("Le modèle n'a pas encore été calibré. Appelez d'abord la méthode 'fit'.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.calibrated_model, f)
        
        logger.info(f"Modèle calibré sauvegardé dans {path}")
    
    @staticmethod
    def load(path: str) -> Any:
        """
        Charge un modèle calibré depuis un fichier pickle.
        
        Args:
            path: Chemin où le modèle est sauvegardé
            
        Returns:
            Modèle calibré chargé
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Modèle calibré chargé depuis {path}")
        return model

def recalibrate_model(model_path: str, X: pd.DataFrame, y: pd.Series, 
                     output_path: Optional[str] = None, method: str = 'sigmoid',
                     cv: int = 5, X_val: Optional[pd.DataFrame] = None, 
                     y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Recalibre un modèle existant et sauvegarde le résultat.
    
    Args:
        model_path: Chemin vers le modèle à calibrer
        X: Données d'entraînement ou de validation
        y: Cibles d'entraînement ou de validation
        output_path: Chemin où sauvegarder le modèle calibré (facultatif)
        method: Méthode de calibration ('sigmoid' ou 'isotonic')
        cv: Nombre de plis pour la validation croisée
        X_val: Données de validation séparées (facultatif)
        y_val: Cibles de validation séparées (facultatif)
        
    Returns:
        Dictionnaire contenant les résultats de l'évaluation de la calibration
    """
    try:
        # Charger le modèle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Modèle chargé depuis {model_path}")
        
        # Créer et entraîner le calibrateur
        calibrator = ModelCalibrator(model, method=method, cv=cv)
        calibrator.fit(X, y, X_val, y_val)
        
        # Évaluer la calibration
        eval_results = calibrator.evaluate_calibration(X_val if X_val is not None else X, 
                                                      y_val if y_val is not None else y)
        
        # Afficher les résultats
        logger.info(f"Amélioration du Brier score: {eval_results['improvement']:.2f}%")
        logger.info(f"Distribution des probabilités originales: {eval_results['prob_distribution_original']}")
        logger.info(f"Distribution des probabilités calibrées: {eval_results['prob_distribution_calibrated']}")
        
        # Sauvegarder le modèle calibré si un chemin est fourni
        if output_path:
            calibrator.save(output_path)
        
        return {
            'calibrator': calibrator,
            'evaluation': eval_results
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la recalibration du modèle: {str(e)}")
        raise

def integrate_calibration_with_shap(calibrated_model: Any, X: pd.DataFrame) -> Dict[str, Any]:
    """
    Intègre le modèle calibré avec SHAP pour fournir des probabilités précises
    tout en expliquant les prédictions.
    
    Args:
        calibrated_model: Modèle calibré
        X: Données d'entrée pour la prédiction
        
    Returns:
        Dictionnaire contenant la prédiction calibrée et les explications SHAP
    """
    try:
        from src.explainability.shap_explainer import ShapExplainer
        
        # Créer l'explainer SHAP pour le modèle calibré
        explainer = ShapExplainer(calibrated_model, X)
        
        # Calculer la prédiction calibrée
        y_prob = calibrated_model.predict_proba(X)[:, 1]
        
        # Générer l'explication SHAP
        explanation = explainer.explain(X)
        
        return {
            'prediction': y_prob,
            'shap_explanation': explanation
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'intégration avec SHAP: {str(e)}")
        raise
