"""
Entry point for the Pediatric Appendicitis Diagnosis application.
This module imports and runs the Flask application.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path to ensure correct imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

if __name__ == '__main__':
    try:
        logger.info("Starting Pediatric Appendicitis Diagnosis application...")
        # Import the Flask app from the API module
        from src.api.app import app
        
        # Start the Flask application
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)
