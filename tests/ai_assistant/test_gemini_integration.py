"""
Tests for the Gemini AI Integration module.
Tests the AI assistant functionality for answering questions and providing explanations.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import json

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ai_assistant.gemini_integration import (
    get_assistant_response,
    explain_prediction_features,
    get_clinical_recommendations,
    reset_assistant,
    MedicalAssistant
)

class TestGeminiIntegration(unittest.TestCase):
    """Test case for the Gemini AI Integration module."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        # Create mock patient data
        self.patient_data = {
            'age': 10,
            'gender': 'Male',
            'duration': 24.0,
            'migration': True,
            'anorexia': True,
            'nausea': True,
            'vomiting': False,
            'right_lower_quadrant_pain': True,
            'fever': True,
            'rebound_tenderness': True,
            'white_blood_cell_count': 15.0,
            'neutrophil_percentage': 80.0,
            'c_reactive_protein': 50.0,
            'prediction': 0.8
        }
        
        # Create mock features
        self.features = [
            {'name': 'WBC Count', 'value': 0.35},
            {'name': 'Rebound Tenderness', 'value': 0.25},
            {'name': 'Right Lower Quadrant Pain', 'value': 0.2},
            {'name': 'Neutrophil Percentage', 'value': 0.15},
            {'name': 'CRP', 'value': 0.05}
        ]
    
    @patch('src.ai_assistant.gemini_integration.assistant.ask_question')
    def test_get_assistant_response(self, mock_ask_question):
        """Test getting a response from the AI assistant."""
        # Set up the mock
        mock_ask_question.return_value = "This is a test response about appendicitis."
        
        # Call the function
        response = get_assistant_response("What are the symptoms of appendicitis?", self.patient_data)
        
        # Check that the response matches the mock
        self.assertEqual(response, "This is a test response about appendicitis.")
        
        # Check that the mock was called with the correct arguments
        mock_ask_question.assert_called_once_with("What are the symptoms of appendicitis?", self.patient_data)
    
    @patch('src.ai_assistant.gemini_integration.assistant.explain_features')
    def test_explain_prediction_features(self, mock_explain_features):
        """Test explaining prediction features."""
        # Set up the mock
        mock_explain_features.return_value = "This is an explanation of the features."
        
        # Call the function
        explanation = explain_prediction_features(self.features)
        
        # Check that the explanation matches the mock
        self.assertEqual(explanation, "This is an explanation of the features.")
        
        # Check that the mock was called with the correct arguments
        mock_explain_features.assert_called_once_with(self.features)
    
    @patch('src.ai_assistant.gemini_integration.assistant.recommend_next_steps')
    def test_get_clinical_recommendations(self, mock_recommend_next_steps):
        """Test getting clinical recommendations."""
        # Set up the mock
        mock_recommend_next_steps.return_value = "These are clinical recommendations."
        
        # Call the function
        recommendations = get_clinical_recommendations(0.8, self.features)
        
        # Check that the recommendations match the mock
        self.assertEqual(recommendations, "These are clinical recommendations.")
        
        # Check that the mock was called with the correct arguments
        mock_recommend_next_steps.assert_called_once_with(0.8, self.features)
    
    @patch('src.ai_assistant.gemini_integration.assistant.reset_conversation')
    def test_reset_assistant(self, mock_reset_conversation):
        """Test resetting the assistant."""
        # Set up the mock
        mock_reset_conversation.return_value = "Conversation history has been reset."
        
        # Call the function
        result = reset_assistant()
        
        # Check that the result matches the mock
        self.assertEqual(result, "Conversation history has been reset.")
        
        # Check that the mock was called
        mock_reset_conversation.assert_called_once()
    
    @patch('google.generativeai.GenerativeModel')
    def test_medical_assistant_initialization(self, mock_model):
        """Test initialization of the MedicalAssistant class."""
        # Set up the mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_chat_session = MagicMock()
        mock_model_instance.start_chat.return_value = mock_chat_session
        
        # Create a MedicalAssistant instance
        assistant = MedicalAssistant()
        
        # Check that the models were created
        self.assertEqual(mock_model.call_count, 2)  # Called for chat_model and vision_model
        
        # Check that the chat session was initialized
        mock_model_instance.start_chat.assert_called_once()
        mock_chat_session.send_message.assert_called_once()
    
    @patch('google.generativeai.GenerativeModel')
    def test_format_patient_data(self, mock_model):
        """Test formatting patient data."""
        # Set up the mocks
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        mock_chat_session = MagicMock()
        mock_model_instance.start_chat.return_value = mock_chat_session
        
        # Create a MedicalAssistant instance
        assistant = MedicalAssistant()
        
        # Format patient data
        formatted_data = assistant._format_patient_data(self.patient_data)
        
        # Check that the formatted data contains expected entries
        self.assertIn("Age: 10", formatted_data)
        self.assertIn("Gender: Male", formatted_data)
        self.assertIn("Pain duration: 24.0 hours", formatted_data)
        self.assertIn("WBC: 15.0", formatted_data)
        self.assertIn("Neutrophils: 80.0%", formatted_data)
        self.assertIn("CRP: 50.0", formatted_data)
        self.assertIn("80.0% probability of appendicitis", formatted_data)

if __name__ == '__main__':
    unittest.main()
