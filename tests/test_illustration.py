#!/usr/bin/env python3
"""
Tests for the illustration module.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from requests.exceptions import RequestException

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.illustration import (
    InvokeAIClient,
    IllustrationGenerator,
    IllustrationError,
    find_nodes_without_illustrations,
    find_null_nodes,
    should_generate_illustrations_first,
)


class TestInvokeAIClient:
    """Test the InvokeAI client."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = InvokeAIClient("http://test.example.com")

    @patch("requests.get")
    def test_is_available_success(self, mock_get):
        """Test successful availability check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert self.client.is_available() is True
        mock_get.assert_called_once_with("http://test.example.com", timeout=5)

    @patch("requests.get")
    def test_is_available_failure(self, mock_get):
        """Test failed availability check."""
        mock_get.side_effect = RequestException("Connection failed")

        assert self.client.is_available() is False

    @patch("requests.post")
    def test_generate_image_success(self, mock_post):
        """Test successful image generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"image": "base64data", "id": "123"}
        mock_post.return_value = mock_response

        result = self.client.generate_image("test prompt")

        assert result == {"image": "base64data", "id": "123"}
        mock_post.assert_called_once_with(
            "http://test.example.com/api/v1/generate",
            json={
                "prompt": "test prompt",
                "width": 512,
                "height": 512,
                "steps": 30,
                "cfg_scale": 7.5
            },
            timeout=60
        )

    @patch("requests.post")
    def test_generate_image_custom_params(self, mock_post):
        """Test image generation with custom parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"image": "base64data"}
        mock_post.return_value = mock_response

        result = self.client.generate_image(
            "test prompt", width=1024, height=768, steps=50, cfg_scale=10.0
        )

        assert result == {"image": "base64data"}
        mock_post.assert_called_once_with(
            "http://test.example.com/api/v1/generate",
            json={
                "prompt": "test prompt",
                "width": 1024,
                "height": 768,
                "steps": 50,
                "cfg_scale": 10.0
            },
            timeout=60
        )

    @patch("requests.post")
    def test_generate_image_http_error(self, mock_post):
        """Test image generation with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response

        result = self.client.generate_image("test prompt")

        assert result is None

    @patch("requests.post")
    def test_generate_image_request_exception(self, mock_post):
        """Test image generation with request exception."""
        mock_post.side_effect = RequestException("Network error")

        result = self.client.generate_image("test prompt")

        assert result is None

    @patch("requests.post")
    def test_generate_image_json_decode_error(self, mock_post):
        """Test image generation with JSON decode error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response

        result = self.client.generate_image("test prompt")

        assert result is None


class TestIllustrationGenerator:
    """Test the illustration generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.images_dir = Path("/tmp/test_images")
        self.generator = IllustrationGenerator(self.mock_client, self.images_dir)

    def test_init(self):
        """Test initialization."""
        assert self.generator.client == self.mock_client
        assert self.generator.images_dir == self.images_dir

    def test_build_prompt_basic(self):
        """Test basic prompt building."""
        prompt = self.generator.build_prompt("A knight battles a dragon")

        assert "photorealistic, insane detail, masterpiece, best quality" in prompt
        assert "A knight battles a dragon" in prompt

    def test_build_prompt_with_rules(self):
        """Test prompt building with rules."""
        rules = {
            "style": "medieval fantasy",
            "tone": "dramatic and serious"
        }

        prompt = self.generator.build_prompt("A knight battles a dragon", rules=rules)

        assert "photorealistic, insane detail, masterpiece, best quality" in prompt
        assert "medieval fantasy" in prompt
        assert "dramatic and serious" in prompt
        assert "A knight battles a dragon" in prompt

    def test_build_prompt_with_scene(self):
        """Test prompt building with scene."""
        scene = {
            "setting": "A medieval castle",
            "time_period": "12th century",
            "atmosphere": "dark and foreboding"
        }

        prompt = self.generator.build_prompt("A knight battles a dragon", scene=scene)

        assert "A medieval castle" in prompt
        assert "time period: 12th century" in prompt
        assert "atmosphere: dark and foreboding" in prompt
        assert "A knight battles a dragon" in prompt

    def test_build_prompt_complete(self):
        """Test prompt building with all components."""
        rules = {"style": "fantasy", "tone": "epic"}
        scene = {
            "setting": "castle courtyard",
            "time_period": "medieval",
            "atmosphere": "tense"
        }

        prompt = self.generator.build_prompt("Battle scene", rules=rules, scene=scene)

        expected_parts = [
            "photorealistic, insane detail, masterpiece, best quality",
            "fantasy",
            "epic",
            "castle courtyard",
            "time period: medieval",
            "atmosphere: tense",
            "Battle scene"
        ]

        for part in expected_parts:
            assert part in prompt

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_save_image_data_success(self, mock_mkdir, mock_file):
        """Test successful image data saving."""
        image_data = {"image": "base64imagedata"}

        result = self.generator.save_image_data("node1", image_data)

        assert result == "images/node1/node1.png"
        mock_mkdir.assert_called_once_with(exist_ok=True)

    def test_save_image_data_no_image(self):
        """Test image data saving without image data."""
        image_data = {"id": "123"}

        result = self.generator.save_image_data("node1", image_data)

        assert result is None

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_save_image_data_exception(self, mock_mkdir, mock_file):
        """Test image data saving with exception."""
        mock_file.side_effect = IOError("File write error")
        image_data = {"image": "base64imagedata"}

        result = self.generator.save_image_data("node1", image_data)

        assert result is None

    def test_generate_illustration_success(self):
        """Test successful illustration generation."""
        self.mock_client.generate_image.return_value = {"image": "base64data"}

        with patch.object(self.generator, "save_image_data") as mock_save:
            mock_save.return_value = "images/node1/node1.png"

            result = self.generator.generate_illustration(
                "node1", "A battle scene"
            )

            assert result == "images/node1/node1.png"
            mock_save.assert_called_once_with("node1", {"image": "base64data"})

    def test_generate_illustration_client_failure(self):
        """Test illustration generation with client failure."""
        self.mock_client.generate_image.return_value = None

        result = self.generator.generate_illustration("node1", "A battle scene")

        assert result is None

    def test_generate_illustration_save_failure(self):
        """Test illustration generation with save failure."""
        self.mock_client.generate_image.return_value = {"image": "base64data"}

        with patch.object(self.generator, "save_image_data") as mock_save:
            mock_save.return_value = None

            result = self.generator.generate_illustration("node1", "A battle scene")

            assert result is None


class TestUtilityFunctions:
    """Test utility functions."""

    def test_find_nodes_without_illustrations_empty(self):
        """Test finding nodes without illustrations in empty tree."""
        nodes = {}
        result = find_nodes_without_illustrations(nodes)
        assert result == []

    def test_find_nodes_without_illustrations_all_have_illustrations(self):
        """Test finding nodes when all have illustrations."""
        nodes = {
            "node1": {"situation": "Test", "illustration": "path1.png"},
            "node2": {"situation": "Test", "illustration": "path2.png"}
        }
        result = find_nodes_without_illustrations(nodes)
        assert result == []

    def test_find_nodes_without_illustrations_some_missing(self):
        """Test finding nodes when some are missing illustrations."""
        nodes = {
            "node1": {"situation": "Test", "illustration": "path1.png"},
            "node2": {"situation": "Test"},  # No illustration
            "node3": {"situation": "Test", "illustration": "path3.png"},
            "node4": {"situation": "Test"}   # No illustration
        }
        result = find_nodes_without_illustrations(nodes)
        assert set(result) == {"node2", "node4"}

    def test_find_nodes_without_illustrations_skip_null_and_failed(self):
        """Test that null and failed nodes are skipped."""
        nodes = {
            "node1": {"situation": "Test"},  # No illustration
            "node2": None,  # Null node
            "node3": {"__failed__": True, "situation": "Failed"},  # Failed node
            "node4": {"situation": "Test", "illustration": "path.png"}
        }
        result = find_nodes_without_illustrations(nodes)
        assert result == ["node1"]

    def test_find_null_nodes_empty(self):
        """Test finding null nodes in empty tree."""
        nodes = {}
        result = find_null_nodes(nodes)
        assert result == []

    def test_find_null_nodes_none_null(self):
        """Test finding null nodes when none are null."""
        nodes = {
            "node1": {"situation": "Test"},
            "node2": {"situation": "Test"}
        }
        result = find_null_nodes(nodes)
        assert result == []

    def test_find_null_nodes_some_null(self):
        """Test finding null nodes when some are null."""
        nodes = {
            "node1": {"situation": "Test"},
            "node2": None,
            "node3": {"situation": "Test"},
            "node4": None
        }
        result = find_null_nodes(nodes)
        assert set(result) == {"node2", "node4"}

    def test_find_null_nodes_skip_failed(self):
        """Test that failed nodes are not included in null nodes."""
        nodes = {
            "node1": None,  # Null
            "node2": {"__failed__": True, "situation": "Failed"},  # Failed, not null
            "node3": {"situation": "Test"}  # Normal
        }
        result = find_null_nodes(nodes)
        assert result == ["node1"]

    def test_should_generate_illustrations_first_true(self):
        """Test should generate illustrations when nodes without illustrations exist."""
        nodes = {
            "node1": {"situation": "Test", "illustration": "path.png"},
            "node2": {"situation": "Test"}  # No illustration
        }
        result = should_generate_illustrations_first(nodes)
        assert result is True

    def test_should_generate_illustrations_first_false(self):
        """Test should not generate illustrations when all have illustrations."""
        nodes = {
            "node1": {"situation": "Test", "illustration": "path1.png"},
            "node2": {"situation": "Test", "illustration": "path2.png"}
        }
        result = should_generate_illustrations_first(nodes)
        assert result is False

    def test_should_generate_illustrations_first_empty(self):
        """Test should not generate illustrations in empty tree."""
        nodes = {}
        result = should_generate_illustrations_first(nodes)
        assert result is False