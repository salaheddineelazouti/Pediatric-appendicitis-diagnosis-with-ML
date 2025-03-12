"""
API route tests to ensure all endpoints function correctly.
Tests the various routes defined in the Flask application.
"""

import os
import sys
import unittest

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the Flask app and create a test client
from src.api.app import app

class TestApiRoutes(unittest.TestCase):
    """Test case for the API routes."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_index_route(self):
        """Test that the index route returns 200 and contains expected content."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200, "Should return status code 200")
        self.assertIn(b'Pediatric Appendicitis', response.data, 
                     "Home page should contain 'Pediatric Appendicitis'")
    
    def test_about_route(self):
        """Test that the about route returns 200 and contains expected content."""
        response = self.app.get('/about')
        self.assertEqual(response.status_code, 200, "Should return status code 200")
        self.assertIn(b'About', response.data, "About page should contain 'About'")
    
    def test_diagnose_route_get(self):
        """Test that the diagnose GET route returns 200 and form elements."""
        response = self.app.get('/diagnose')
        self.assertEqual(response.status_code, 200, "Should return status code 200")
        
        # Check form elements
        self.assertIn(b'form', response.data.lower(), "Diagnose page should contain a form")
        self.assertIn(b'age', response.data.lower(), "Form should contain age field")
        self.assertIn(b'gender', response.data.lower(), "Form should contain gender field")
        self.assertIn(b'submit', response.data.lower(), "Form should contain submit button")
    
    def test_nonexistent_route(self):
        """Test that a non-existent route returns 404."""
        response = self.app.get('/nonexistent-route')
        self.assertEqual(response.status_code, 404, "Should return status code 404")
    
    def test_static_resources(self):
        """Test that static resources can be loaded."""
        response = self.app.get('/static/css/styles.css')
        self.assertEqual(response.status_code, 200, "Should return status code 200")
        self.assertIn(b'css', response.data.lower(), "Should return CSS content")

if __name__ == '__main__':
    unittest.main()
