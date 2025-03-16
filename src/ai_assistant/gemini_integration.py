"""
Gemini AI Integration for Pediatric Appendicitis Diagnosis Assistant.
This module provides a specialized medical AI assistant to help doctors
with appendicitis diagnosis questions and clinical decision support.
"""

import os
import logging
import pathlib
from typing import List, Dict, Any, Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trouver le chemin racine du projet
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()

# Essayer de charger les variables d'environnement depuis différentes sources
def load_api_key():
    """
    Charge la clé API Gemini depuis différentes sources dans cet ordre de priorité:
    1. Variable d'environnement du système
    2. Session Flask (si disponible)
    3. Fichier .env du projet
    4. Fichier .env dans le dossier utilisateur
    """
    # Vérifier d'abord les variables d'environnement du système
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        logger.info("Using Gemini API key from environment variables")
        return api_key
    
    # Vérifier ensuite si la clé est dans la session Flask
    try:
        from flask import session
        if session and 'gemini_api_key' in session:
            logger.info("Using Gemini API key from Flask session")
            return session['gemini_api_key']
    except (ImportError, RuntimeError):
        logger.debug("Flask session not available")
    
    # Essayer les fichiers .env en dernier recours
    try:
        from dotenv import load_dotenv
        
        # Essayer différents emplacements pour le fichier .env
        env_paths = [
            os.path.join(PROJECT_ROOT, '.env'),                      # Racine du projet
            os.path.join(os.path.expanduser('~'), '.pediatric_appendicitis', '.env')  # Dossier utilisateur
        ]
        
        for env_path in env_paths:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                logger.info(f"Loaded environment variables from {env_path}")
                
                # Vérifier si la clé API est maintenant disponible
                api_key = os.environ.get("GEMINI_API_KEY", "")
                if api_key:
                    logger.info(f"Using Gemini API key from {env_path}")
                    return api_key
                
    except ImportError:
        logger.warning("python-dotenv not installed. Using OS environment variables only.")
    
    logger.warning("No Gemini API key found in any location")
    return ""

# Initialiser l'API Gemini avec la clé fournie
API_KEY = load_api_key()

# Flag to check if Gemini integration is available
GEMINI_AVAILABLE = False

try:
    import google.generativeai as genai
    if API_KEY:
        genai.configure(api_key=API_KEY)
        # Vérifier que GenerativeModel existe dans cette version de la bibliothèque
        if hasattr(genai, 'GenerativeModel'):
            GEMINI_AVAILABLE = True
            logger.info("Google Generative AI initialized successfully")
            # Define the model name to use
            MODEL_NAME = "models/gemini-1.5-pro"
        else:
            logger.warning("GenerativeModel not available in this version of google.generativeai. AI assistant features will be mocked.")
            GEMINI_AVAILABLE = False
            MODEL_NAME = "Non disponible (version incompatible)"
    else:
        logger.warning("GEMINI_API_KEY environment variable not set. AI assistant features will not work properly.")
        MODEL_NAME = "Non disponible (clé API manquante)"
except ImportError:
    logger.warning("Google Generative AI package not available. AI assistant features will be mocked for testing.")
    GEMINI_AVAILABLE = False
    MODEL_NAME = "Non disponible (package non installé)"

# Définir les modèles avec leurs noms complets
VISION_MODEL_NAME = "models/gemini-1.5-pro-vision"

# Medical context to provide to the AI for specialized responses..
MEDICAL_CONTEXT = """
You are DocAssist, a specialized medical AI assistant for pediatric appendicitis diagnosis.
You assist physicians by:
1. Answering questions about pediatric appendicitis diagnosis, treatment, and complications
2. Interpreting clinical features and lab results in the context of appendicitis
3. Explaining differential diagnoses for abdominal pain in children
4. Providing evidence-based guidance on diagnostic approaches
5. Helping interpret imaging findings and their significance
6. Explaining the model's predictions in clinical context

Always provide evidence-based information with appropriate medical caution.
Remind physicians that your guidance should supplement, not replace, clinical judgment.
When discussing serious conditions, emphasize the importance of timely evaluation and treatment.
"""

class MockResponse:
    """Mock response object for testing."""
    def __init__(self, text):
        self.text = text
        
class MockGenerativeModel:
    """Mock GenerativeModel for testing."""
    def __init__(self, model_name):
        self.model_name = model_name
        
    def generate_content(self, prompt):
        """Mock content generation."""
        return MockResponse(f"Mock response to: {prompt[:30]}...")
        
    def start_chat(self, history=None):
        """Mock chat session."""
        return MockChatSession()
        
class MockChatSession:
    """Mock chat session for testing."""
    def __init__(self):
        self.history = []
        
    def send_message(self, message):
        """Mock message sending."""
        self.history.append(message)
        return MockResponse(f"Mock response to: {message[:30]}...")

class MedicalAssistant:
    """Class for interacting with Gemini API in a medical context."""
    
    def __init__(self):
        """Initialize the medical assistant with Gemini models."""
        # Utilisation du format complet des noms de modèles
        if GEMINI_AVAILABLE and hasattr(genai, 'GenerativeModel'):
            try:
                self.chat_model = genai.GenerativeModel(MODEL_NAME)
                self.vision_model = genai.GenerativeModel(VISION_MODEL_NAME)
            except Exception as e:
                logger.error(f"Error initializing Gemini models: {str(e)}")
                self.chat_model = None
                self.vision_model = None
                GEMINI_AVAILABLE = False
        else:
            # Mock models for testing
            self.chat_model = MockGenerativeModel(MODEL_NAME)
            self.vision_model = MockGenerativeModel(VISION_MODEL_NAME)
        self.chat_session = None
        self._initialize_chat()
    
    def _initialize_chat(self):
        """Initialize a new chat session with medical context."""
        if GEMINI_AVAILABLE and self.chat_model is not None:
            try:
                if hasattr(self.chat_model, 'start_chat'):
                    self.chat_session = self.chat_model.start_chat(history=[])
                    # Prime the model with medical context
                    self.chat_session.send_message(MEDICAL_CONTEXT)
                else:
                    logger.warning("start_chat method not available in this version of GenerativeModel. Using direct generation instead.")
                    self.chat_session = None
            except Exception as e:
                logger.error(f"Error initializing chat: {str(e)}")
                # Fallback to using the model directly if chat session fails
                self.chat_session = None
        else:
            # Mock chat session for testing
            self.chat_session = self.chat_model.start_chat()
            self.chat_session.send_message(MEDICAL_CONTEXT)
    
    def ask_question(self, query: str, patient_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Ask a medical question to the AI assistant.
        
        Args:
            query: The question or prompt from the physician
            patient_data: Optional dictionary containing patient clinical data for context
            
        Returns:
            The AI assistant's response
        """
        if GEMINI_AVAILABLE:
            try:
                # If we're providing patient context, format it into the prompt
                if patient_data:
                    patient_context = self._format_patient_data(patient_data)
                    full_query = f"Patient Context:\n{patient_context}\n\nPhysician Question: {query}"
                else:
                    full_query = query
                
                # Use chat session if available, otherwise use model directly
                if self.chat_session is not None:
                    response = self.chat_session.send_message(full_query)
                    return response.text
                else:
                    # Alternative: generate content directly
                    response = self.chat_model.generate_content(full_query)
                    return response.text
                    
            except Exception as e:
                return f"I apologize, but I encountered an error: {str(e)}. Please try again with a different question."
        else:
            # Mock response for testing
            return "Mock response for testing"
    
    def _format_patient_data(self, patient_data: Dict[str, Any]) -> str:
        """Format patient data into a readable string for context."""
        formatted_data = []
        
        # Demographics
        if 'age' in patient_data:
            formatted_data.append(f"Age: {patient_data['age']} years")
        if 'gender' in patient_data:
            formatted_data.append(f"Gender: {patient_data['gender']}")
            
        # Clinical features
        clinical = []
        for key in ['duration', 'migration', 'anorexia', 'nausea', 'vomiting', 
                   'right_lower_quadrant_pain', 'fever', 'rebound_tenderness']:
            if key in patient_data:
                if key == 'duration':
                    clinical.append(f"Pain duration: {patient_data[key]} hours")
                else:
                    if patient_data[key]:
                        clinical.append(f"{key.replace('_', ' ').title()}: Present")
        
        if clinical:
            formatted_data.append("Clinical Features:\n- " + "\n- ".join(clinical))
            
        # Lab values
        lab_values = []
        for key in ['white_blood_cell_count', 'neutrophil_percentage', 'c_reactive_protein']:
            if key in patient_data:
                if key == 'white_blood_cell_count':
                    lab_values.append(f"WBC: {patient_data[key]} ×10³/μL")
                elif key == 'neutrophil_percentage':
                    lab_values.append(f"Neutrophils: {patient_data[key]}%")
                elif key == 'c_reactive_protein':
                    lab_values.append(f"CRP: {patient_data[key]} mg/L")
                    
        if lab_values:
            formatted_data.append("Laboratory Values:\n- " + "\n- ".join(lab_values))
            
        # Prediction results if available
        if 'prediction' in patient_data:
            formatted_data.append(f"Model Prediction: {patient_data['prediction'] * 100:.1f}% probability of appendicitis")
        
        return "\n\n".join(formatted_data)
    
    def explain_features(self, features: List[Dict[str, Any]]) -> str:
        """
        Explain the significance of clinical features in appendicitis diagnosis.
        
        Args:
            features: List of features with their importance values
            
        Returns:
            Clinical explanation of feature significance
        """
        if GEMINI_AVAILABLE:
            features_prompt = "Explain the clinical significance of these features in appendicitis diagnosis:\n"
            for feature in features:
                features_prompt += f"- {feature['name']}: Contribution value of {feature['value']:.3f}\n"
                
            features_prompt += "\nProvide a focused clinical interpretation of how these features relate to pediatric appendicitis."
            
            try:
                if self.chat_session is not None:
                    response = self.chat_session.send_message(features_prompt)
                else:
                    response = self.chat_model.generate_content(features_prompt)
                return response.text
            except Exception as e:
                return f"I apologize, but I encountered an error: {str(e)}. Please try again later."
        else:
            # Mock response for testing
            return "Mock response for testing"
    
    def recommend_next_steps(self, prediction: float, important_features: List[Dict[str, Any]]) -> str:
        """
        Recommend next clinical steps based on the model prediction and features.
        
        Args:
            prediction: Probability of appendicitis
            important_features: List of features with their importance values
            
        Returns:
            Clinical recommendations
        """
        if GEMINI_AVAILABLE:
            prompt = f"""
            Based on a machine learning model prediction of {prediction:.1%} probability of appendicitis,
            and these key contributing factors:
            {', '.join([f"{f['name']} ({f['value']:.3f})" for f in important_features[:3]])}
            
            Please provide evidence-based recommendations for next clinical steps.
            Consider imaging, consultation, observation, and clinical reassessment options.
            Frame your response for a physician audience.
            """
            
            try:
                if self.chat_session is not None:
                    response = self.chat_session.send_message(prompt)
                else:
                    response = self.chat_model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"I apologize, but I encountered an error: {str(e)}. Please try again later."
        else:
            # Mock response for testing
            return "Mock response for testing"
    
    def reset_conversation(self):
        """Reset the conversation history."""
        if GEMINI_AVAILABLE:
            self._initialize_chat()
            return "Conversation history has been reset."
        else:
            # Mock response for testing
            return "Mock response for testing"

# Create a singleton instance
assistant = MedicalAssistant()

def get_assistant_response(query: str, patient_data: Optional[Dict[str, Any]] = None) -> str:
    """Convenient function to get responses from the medical assistant."""
    return assistant.ask_question(query, patient_data)

def explain_prediction_features(features: List[Dict[str, Any]]) -> str:
    """Explain the clinical significance of prediction features."""
    return assistant.explain_features(features)

def get_clinical_recommendations(prediction: float, features: List[Dict[str, Any]]) -> str:
    """Get clinical recommendations based on prediction and features."""
    return assistant.recommend_next_steps(prediction, features)

def reset_assistant():
    """Reset the assistant's conversation history."""
    return assistant.reset_conversation()
