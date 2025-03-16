"""
Script pour lancer l'application de diagnostic d'appendicite pédiatrique
sur localhost pour faciliter la prévisualisation.
"""

import os
import sys
import logging

# Créer le dossier logs s'il n'existe pas
os.makedirs('logs', exist_ok=True)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajouter le répertoire racine au chemin Python
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

if __name__ == '__main__':
    try:
        logger.info("Démarrage de l'application de diagnostic d'appendicite pédiatrique sur localhost...")
        
        # Importer l'application Flask du module API
        from src.api.app import app
        
        # Démarrer l'application Flask sur localhost
        app.run(host='localhost', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Erreur lors du démarrage de l'application: {e}")
        sys.exit(1)
