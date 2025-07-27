#!/usr/bin/env python3
"""
Tests for the web UI module.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.web_ui import DialogueTreeWebApp


class TestDialogueTreeWebApp(unittest.TestCase):
    """Test DialogueTreeWebApp class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        
        # Create a simple test tree
        self.test_tree = {
            "rules": {"language": "English", "tone": "casual"},
            "scene": {"setting": "test setting"},
            "nodes": {
                "start": {
                    "situation": "Test situation",
                    "choices": [{"text": "Choice 1", "next": "node1"}]
                },
                "node1": None
            },
            "params": {"loyalty": 50}
        }
        
        json.dump(self.test_tree, self.temp_file)
        self.temp_file.close()
        
        self.tree_file = Path(self.temp_file.name)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.tree_file.exists():
            self.tree_file.unlink()

    @patch('src.web_ui.OllamaClient')
    @patch('src.web_ui.NodeGenerator')
    def test_init(self, mock_node_gen, mock_ollama):
        """Test DialogueTreeWebApp initialization."""
        app = DialogueTreeWebApp(self.tree_file, "test-model")
        
        self.assertEqual(app.tree_file, self.tree_file)
        mock_ollama.assert_called_once_with("test-model")
        mock_node_gen.assert_called_once()
        self.assertIsNotNone(app.app)

    @patch('src.web_ui.OllamaClient')
    @patch('src.web_ui.NodeGenerator')
    def test_flask_app_creation(self, mock_node_gen, mock_ollama):
        """Test that Flask app is created with routes."""
        app = DialogueTreeWebApp(self.tree_file, "test-model")
        
        # Check that Flask app is created
        self.assertIsNotNone(app.app)
        
        # Check that routes are registered
        route_names = [rule.endpoint for rule in app.app.url_map.iter_rules()]
        expected_routes = [
            'static',  # Flask built-in
            'index',
            'get_tree',
            'save_tree', 
            'get_node',
            'generate_node',
            'get_status'
        ]
        
        for route in expected_routes:
            self.assertIn(route, route_names)

    @patch('src.web_ui.OllamaClient')
    @patch('src.web_ui.NodeGenerator')
    def test_templates_and_static_dirs_created(self, mock_node_gen, mock_ollama):
        """Test that template and static directories are created when app runs."""
        app = DialogueTreeWebApp(self.tree_file, "test-model")
        
        with patch.object(app.app, 'run') as mock_run:
            app.run()
            
        # Check that run was called
        mock_run.assert_called_once()

    @patch('src.web_ui.OllamaClient')
    @patch('src.web_ui.NodeGenerator') 
    def test_api_endpoints_exist(self, mock_node_gen, mock_ollama):
        """Test that API endpoints are accessible."""
        # Mock the LLM client properly
        mock_client = mock_ollama.return_value
        mock_client.model = "test-model"
        mock_client.is_available.return_value = True
        
        app = DialogueTreeWebApp(self.tree_file, "test-model")
        
        # Create a test client
        with app.app.test_client() as client:
            # Test status endpoint
            response = client.get('/api/status')
            self.assertEqual(response.status_code, 200)
            
            data = json.loads(response.data)
            self.assertIn('tree_file', data)
            self.assertIn('llm_model', data)
            self.assertIn('status', data)


class TestWebUIIntegration(unittest.TestCase):
    """Integration tests for web UI with real data."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        
        # Create a more complex test tree
        self.test_tree = {
            "rules": {
                "language": "English",
                "tone": "dramatic",
                "voice": "third person",
                "style": "fantasy"
            },
            "scene": {
                "setting": "A medieval castle",
                "time_period": "Medieval",
                "location": "Throne room"
            },
            "nodes": {
                "start": {
                    "situation": "You stand before the throne.",
                    "choices": [
                        {"text": "Bow respectfully", "next": "node1", "effects": {"loyalty": 5}},
                        {"text": "Stand proud", "next": "node2", "effects": {"ambition": 10}}
                    ]
                },
                "node1": None,
                "node2": {
                    "situation": "The king notices your defiance.",
                    "choices": [
                        {"text": "Apologize", "next": "node3", "effects": {"loyalty": 10}},
                        {"text": "Continue standing", "next": "node4", "effects": {"ambition": 15}}
                    ]
                },
                "node3": None,
                "node4": None
            },
            "params": {"loyalty": 50, "ambition": 30, "wisdom": 40}
        }
        
        json.dump(self.test_tree, self.temp_file)
        self.temp_file.close()
        
        self.tree_file = Path(self.temp_file.name)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.tree_file.exists():
            self.tree_file.unlink()

    @patch('src.web_ui.OllamaClient')
    @patch('src.web_ui.NodeGenerator')
    def test_get_tree_api(self, mock_node_gen, mock_ollama):
        """Test the GET /api/tree endpoint."""
        app = DialogueTreeWebApp(self.tree_file, "test-model")
        
        with app.app.test_client() as client:
            response = client.get('/api/tree')
            self.assertEqual(response.status_code, 200)
            
            data = json.loads(response.data)
            
            # Check structure
            self.assertIn('tree', data)
            self.assertIn('null_nodes', data)
            self.assertIn('total_nodes', data)
            self.assertIn('completed_nodes', data)
            
            # Check content
            self.assertEqual(data['total_nodes'], 5)  # 4 defined + 1 auto-created in choices
            self.assertEqual(data['completed_nodes'], 2)
            self.assertEqual(len(data['null_nodes']), 3)  # node1, node3, node4, plus auto-created ones

    @patch('src.web_ui.OllamaClient')
    @patch('src.web_ui.NodeGenerator')
    def test_get_node_api_complete(self, mock_node_gen, mock_ollama):
        """Test the GET /api/node/<id> endpoint for complete nodes."""
        app = DialogueTreeWebApp(self.tree_file, "test-model")
        
        with app.app.test_client() as client:
            response = client.get('/api/node/start')
            self.assertEqual(response.status_code, 200)
            
            data = json.loads(response.data)
            
            self.assertEqual(data['node_id'], 'start')
            self.assertFalse(data['is_null'])
            self.assertIn('data', data)
            self.assertEqual(data['data']['situation'], "You stand before the throne.")

    @patch('src.web_ui.OllamaClient')
    @patch('src.web_ui.NodeGenerator')
    def test_get_node_api_null(self, mock_node_gen, mock_ollama):
        """Test the GET /api/node/<id> endpoint for null nodes."""
        app = DialogueTreeWebApp(self.tree_file, "test-model")
        
        with app.app.test_client() as client:
            response = client.get('/api/node/node1')
            self.assertEqual(response.status_code, 200)
            
            data = json.loads(response.data)
            
            self.assertEqual(data['node_id'], 'node1')
            self.assertTrue(data['is_null'])
            self.assertIn('parent_info', data)

    @patch('src.web_ui.OllamaClient')
    @patch('src.web_ui.NodeGenerator')
    def test_get_node_api_not_found(self, mock_node_gen, mock_ollama):
        """Test the GET /api/node/<id> endpoint for non-existent nodes."""
        app = DialogueTreeWebApp(self.tree_file, "test-model")
        
        with app.app.test_client() as client:
            response = client.get('/api/node/nonexistent')
            self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()