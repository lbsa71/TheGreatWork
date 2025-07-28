#!/usr/bin/env python3
"""
Tests for the web UI module.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from web_app.app import DialogueWebApp


class TestDialogueWebApp(unittest.TestCase):
    """Test cases for DialogueWebApp class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a temporary tree file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tree_file = Path(self.temp_dir.name) / "test_tree.json"

        # Sample tree data
        self.sample_tree = {
            "rules": {
                "language": "English",
                "tone": "dramatic",
                "voice": "third person",
                "style": "medieval fantasy",
            },
            "scene": {
                "setting": "Medieval kingdom",
                "time_period": "Medieval era",
                "location": "Royal castle",
                "atmosphere": "Tense",
                "key_elements": "Political intrigue",
            },
            "nodes": {
                "start": {
                    "situation": "The king is dead.",
                    "choices": [
                        {"text": "Mourn publicly", "next": "node1", "effects": {}},
                        {"text": "Seize the throne", "next": "node2", "effects": {}},
                    ],
                },
                "node1": {
                    "situation": "You mourn publicly.",
                    "choices": [
                        {"text": "Continue mourning", "next": None, "effects": {}}
                    ],
                },
                "node2": None,
            },
            "params": {"loyalty": 45, "ambition": 80},
        }

        # Write sample tree to file
        with open(self.tree_file, "w") as f:
            json.dump(self.sample_tree, f)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    @patch("web_app.app.OllamaClient")
    @patch("web_app.app.NodeGenerator")
    def test_init(self, mock_node_gen: Any, mock_ollama: Any) -> None:
        """Test DialogueWebApp initialization."""
        web_ui = DialogueWebApp(str(self.tree_file), "llama3")

        # Check that components were initialized
        self.assertEqual(web_ui.tree_file, self.tree_file)
        self.assertEqual(web_ui.model, "llama3")
        self.assertIsNotNone(web_ui.tree_manager)
        self.assertIsNotNone(web_ui.tree)
        self.assertIsNotNone(web_ui.app)

        # Check that LLM components were initialized
        mock_ollama.assert_called_once_with(model="llama3")
        mock_node_gen.assert_called_once()

    @patch("web_app.app.OllamaClient")
    @patch("web_app.app.NodeGenerator")
    def test_routes_exist(self, mock_node_gen: Any, mock_ollama: Any) -> None:
        """Test that all expected routes exist."""
        web_ui = DialogueWebApp(str(self.tree_file), "llama3")

        # Get all routes
        routes = [rule.rule for rule in web_ui.app.url_map.iter_rules()]

        expected_routes = [
            "/",
            "/api/tree",
            "/api/node/<node_id>",
            "/api/history/<node_id>",
            "/api/tree/structure",
            "/api/generate/<node_id>",
            "/api/save",
        ]

        for expected_route in expected_routes:
            self.assertIn(
                expected_route, routes, f"Route {expected_route} not found in {routes}"
            )

    @patch("web_app.app.OllamaClient")
    @patch("web_app.app.NodeGenerator")
    def test_get_tree_endpoint(self, mock_node_gen: Any, mock_ollama: Any) -> None:
        """Test the /api/tree endpoint."""
        web_ui = DialogueWebApp(str(self.tree_file), "llama3")

        with web_ui.app.test_client() as client:
            response = client.get("/api/tree")
            self.assertEqual(response.status_code, 200)

            data = response.get_json()
            self.assertIn("nodes", data)
            self.assertIn("rules", data)
            self.assertIn("scene", data)
            self.assertIn("params", data)

            # Check that the tree data matches
            self.assertEqual(data["nodes"]["start"]["situation"], "The king is dead.")
            self.assertEqual(data["params"]["loyalty"], 45)

    @patch("web_app.app.OllamaClient")
    @patch("web_app.app.NodeGenerator")
    def test_get_node_endpoint(self, mock_node_gen: Any, mock_ollama: Any) -> None:
        """Test the /api/node/<node_id> endpoint."""
        web_ui = DialogueWebApp(str(self.tree_file), "llama3")

        with web_ui.app.test_client() as client:
            # Test valid node
            response = client.get("/api/node/start")
            self.assertEqual(response.status_code, 200)

            data = response.get_json()
            self.assertEqual(data["id"], "start")
            self.assertEqual(data["situation"], "The king is dead.")
            self.assertIn("choices", data)

            # Test non-existent node
            response = client.get("/api/node/nonexistent")
            self.assertEqual(response.status_code, 404)

            # Test null node
            response = client.get("/api/node/node2")
            self.assertEqual(response.status_code, 404)

    @patch("web_app.app.OllamaClient")
    @patch("web_app.app.NodeGenerator")
    def test_get_tree_structure_endpoint(
        self, mock_node_gen: Any, mock_ollama: Any
    ) -> None:
        """Test the /api/tree/structure endpoint."""
        web_ui = DialogueWebApp(str(self.tree_file), "llama3")

        with web_ui.app.test_client() as client:
            response = client.get("/api/tree/structure")
            self.assertEqual(response.status_code, 200)

            data = response.get_json()
            self.assertIn("start", data)
            self.assertIn("node1", data)
            self.assertIn("node2", data)

            # Check structure of valid node
            start_node = data["start"]
            self.assertIn("situation", start_node)
            self.assertIn("children", start_node)
            self.assertIn("has_choices", start_node)
            self.assertIn("is_null", start_node)
            self.assertFalse(start_node["is_null"])

            # Check structure of null node
            null_node = data["node2"]
            self.assertTrue(null_node["is_null"])
            self.assertEqual(null_node["situation"], "Incomplete node")

    @patch("web_app.app.OllamaClient")
    @patch("web_app.app.NodeGenerator")
    def test_find_parent_context(self, mock_node_gen: Any, mock_ollama: Any) -> None:
        """Test the _find_parent_context method."""
        web_ui = DialogueWebApp(str(self.tree_file), "llama3")

        # Test finding parent for node1
        context = web_ui._find_parent_context("node1")
        self.assertIsNotNone(context)
        if context is not None:  # Type guard for mypy
            self.assertEqual(context["parent_node_id"], "start")
            self.assertEqual(context["choice_text"], "Mourn publicly")
            self.assertEqual(context["choice_index"], 0)

        # Test finding parent for node2
        context = web_ui._find_parent_context("node2")
        self.assertIsNotNone(context)
        if context is not None:  # Type guard for mypy
            self.assertEqual(context["parent_node_id"], "start")
            self.assertEqual(context["choice_text"], "Seize the throne")
            self.assertEqual(context["choice_index"], 1)

        # Test non-existent node
        context = web_ui._find_parent_context("nonexistent")
        self.assertIsNone(context)

    @patch("web_app.app.OllamaClient")
    @patch("web_app.app.NodeGenerator")
    def test_get_history_endpoint(self, mock_node_gen: Any, mock_ollama: Any) -> None:
        """Test the /api/history/<node_id> endpoint."""
        web_ui = DialogueWebApp(str(self.tree_file), "llama3")

        with web_ui.app.test_client() as client:
            # Test history for root node (should be empty)
            response = client.get("/api/history/start")
            self.assertEqual(response.status_code, 200)

            data = response.get_json()
            self.assertIn("history", data)
            self.assertIn("target_node", data)
            self.assertEqual(data["target_node"], "start")
            self.assertEqual(len(data["history"]), 0)  # Root should have no history

            # Test history for child node
            response = client.get("/api/history/node1")
            self.assertEqual(response.status_code, 200)

            data = response.get_json()
            self.assertEqual(data["target_node"], "node1")
            self.assertEqual(len(data["history"]), 1)  # Should have one step

            # Check the history step structure
            step = data["history"][0]
            self.assertIn("node_id", step)
            self.assertIn("situation", step)
            self.assertIn("choice_text", step)
            self.assertIn("choice_effects", step)
            self.assertEqual(step["node_id"], "start")
            self.assertEqual(step["situation"], "The king is dead.")
            self.assertEqual(step["choice_text"], "Mourn publicly")

            # Test non-existent node
            response = client.get("/api/history/nonexistent")
            self.assertEqual(response.status_code, 404)

    @patch("web_app.app.OllamaClient")
    @patch("web_app.app.NodeGenerator")
    def test_update_node_endpoint(self, mock_node_gen: Any, mock_ollama: Any) -> None:
        """Test the PUT /api/node/<node_id> endpoint for updating nodes."""
        web_ui = DialogueWebApp(str(self.tree_file), "llama3")

        with web_ui.app.test_client() as client:
            # Test successful node update
            update_data = {
                "situation": "Updated situation text",
                "choices": [
                    {
                        "text": "Updated choice 1",
                        "next": "node1",
                        "effects": {"param1": 10}
                    },
                    {
                        "text": "Updated choice 2", 
                        "next": "node2"
                    }
                ]
            }
            
            response = client.put('/api/node/start', 
                                json=update_data,
                                content_type='application/json')
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertTrue(data['success'])
            self.assertIn('node', data)
            
            # Verify the node was actually updated
            self.assertEqual(web_ui.tree.nodes['start']['situation'], "Updated situation text")
            self.assertEqual(len(web_ui.tree.nodes['start']['choices']), 2)
            self.assertEqual(web_ui.tree.nodes['start']['choices'][0]['text'], "Updated choice 1")
            self.assertEqual(web_ui.tree.nodes['start']['choices'][0]['effects']['param1'], 10)

    @patch("web_app.app.OllamaClient") 
    @patch("web_app.app.NodeGenerator")
    def test_update_node_validation(self, mock_node_gen: Any, mock_ollama: Any) -> None:
        """Test validation for the PUT /api/node/<node_id> endpoint."""
        web_ui = DialogueWebApp(str(self.tree_file), "llama3")

        with web_ui.app.test_client() as client:
            # Test missing situation field
            response = client.put('/api/node/start',
                                json={"choices": []},
                                content_type='application/json')
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertIn("Missing 'situation' field", data['error'])
            
            # Test missing choices field
            response = client.put('/api/node/start',
                                json={"situation": "test"},
                                content_type='application/json')
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertIn("Missing 'choices' field", data['error'])
            
            # Test invalid choices structure
            response = client.put('/api/node/start',
                                json={"situation": "test", "choices": "invalid"},
                                content_type='application/json')
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertIn("'choices' must be a list", data['error'])
            
            # Test choice missing text field
            response = client.put('/api/node/start',
                                json={"situation": "test", "choices": [{"next": "node1"}]},
                                content_type='application/json')
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertIn("Choice 0 missing 'text' field", data['error'])

    @patch("web_app.app.OllamaClient")
    @patch("web_app.app.NodeGenerator")
    def test_update_nonexistent_node(self, mock_node_gen: Any, mock_ollama: Any) -> None:
        """Test updating a node that doesn't exist."""
        web_ui = DialogueWebApp(str(self.tree_file), "llama3")

        with web_ui.app.test_client() as client:
            update_data = {
                "situation": "Test situation",
                "choices": [{"text": "Test choice", "next": "test"}]
            }
            
            response = client.put('/api/node/nonexistent',
                                json=update_data,
                                content_type='application/json')
            self.assertEqual(response.status_code, 404)
            data = response.get_json()
            self.assertIn("Node not found", data['error'])


if __name__ == "__main__":
    unittest.main()
