#!/usr/bin/env python3
"""
Tests for llm_integration module.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.llm_integration import LLMError, NodeGenerator, OllamaClient, PromptGenerator


class TestPromptGenerator:
    """Tests for PromptGenerator class."""

    def test_generate_node_prompt(self) -> None:
        """Test prompt generation."""
        parent_situation = "The king is dead."
        choice_text = "Mourn publicly"
        params = {"loyalty": 45, "ambition": 80}

        prompt = PromptGenerator.generate_node_prompt(
            parent_situation, choice_text, params
        )

        assert "The king is dead." in prompt
        assert "Mourn publicly" in prompt
        assert '"loyalty": 45' in prompt
        assert '"ambition": 80' in prompt
        assert "JSON object" in prompt
        assert "situation" in prompt
        assert "choices" in prompt
        assert "next" in prompt
        assert "effects" in prompt

    def test_generate_node_prompt_with_history(self) -> None:
        """Test prompt generation with dialogue history."""
        parent_situation = "The advisor looks concerned."
        choice_text = "Ask for guidance"
        params = {"loyalty": 50, "wisdom": 60}
        dialogue_history = """Dialogue History:
1. Situation: The kingdom is in chaos.
   Player chose: Seek the wise council
2. Situation: You meet with the council.
   Player chose: Express your concerns"""

        prompt = PromptGenerator.generate_node_prompt(
            parent_situation, choice_text, params, dialogue_history
        )

        # Check that both current context and history are included
        assert "The advisor looks concerned." in prompt
        assert "Ask for guidance" in prompt
        assert '"loyalty": 50' in prompt
        assert '"wisdom": 60' in prompt
        assert "Dialogue History:" in prompt
        assert "The kingdom is in chaos." in prompt
        assert "Seek the wise council" in prompt
        assert "Express your concerns" in prompt
        assert "JSON object" in prompt

    def test_generate_node_prompt_without_history(self) -> None:
        """Test prompt generation without dialogue history (legacy behavior)."""
        parent_situation = "The king is dead."
        choice_text = "Mourn publicly"
        params = {"loyalty": 45, "ambition": 80}

        prompt = PromptGenerator.generate_node_prompt(
            parent_situation, choice_text, params, None
        )

        # Should not contain history section
        assert "Dialogue History:" not in prompt
        # But should contain all other elements
        assert "The king is dead." in prompt
        assert "Mourn publicly" in prompt
        assert "current parent node" in prompt

    def test_generate_node_prompt_with_rules(self) -> None:
        """Test prompt generation with stylistic rules."""
        parent_situation = "The king is dead."
        choice_text = "Mourn publicly"
        params = {"loyalty": 45, "ambition": 80}
        rules = {
            "language": "English",
            "tone": "dramatic and serious",
            "voice": "third person narrative",
        }

        prompt = PromptGenerator.generate_node_prompt(
            parent_situation, choice_text, params, None, rules
        )

        assert "STYLISTIC RULES:" in prompt
        assert "Language: English" in prompt
        assert "Tone: dramatic and serious" in prompt
        assert "Voice: third person narrative" in prompt

    def test_generate_node_prompt_with_scene(self) -> None:
        """Test prompt generation with world-building context."""
        parent_situation = "The king is dead."
        choice_text = "Mourn publicly"
        params = {"loyalty": 45, "ambition": 80}
        scene = {
            "setting": "A medieval kingdom",
            "time_period": "Medieval era",
            "atmosphere": "Tense and uncertain",
        }

        prompt = PromptGenerator.generate_node_prompt(
            parent_situation, choice_text, params, None, None, scene
        )

        assert "WORLD-BUILDING CONTEXT:" in prompt
        assert "Setting: A medieval kingdom" in prompt
        assert "Time_Period: Medieval era" in prompt
        assert "Atmosphere: Tense and uncertain" in prompt

    def test_generate_node_prompt_with_rules_and_scene(self) -> None:
        """Test prompt generation with both rules and scene."""
        parent_situation = "The king is dead."
        choice_text = "Mourn publicly"
        params = {"loyalty": 45, "ambition": 80}
        rules = {"tone": "dramatic"}
        scene = {"setting": "Medieval kingdom"}

        prompt = PromptGenerator.generate_node_prompt(
            parent_situation, choice_text, params, None, rules, scene
        )

        assert "STYLISTIC RULES:" in prompt
        assert "Tone: dramatic" in prompt
        assert "WORLD-BUILDING CONTEXT:" in prompt
        assert "Setting: Medieval kingdom" in prompt

    def test_generate_node_prompt_with_depth(self) -> None:
        """Test prompt generation with dialogue depth."""
        parent_situation = "The king is dead."
        choice_text = "Mourn publicly"
        params = {"loyalty": 45, "ambition": 80}
        dialogue_depth = 3

        prompt = PromptGenerator.generate_node_prompt(
            parent_situation, choice_text, params, dialogue_depth=dialogue_depth
        )

        assert "DIALOGUE DEPTH: 3" in prompt
        assert "This node is 3 steps deep from the dialogue root" in prompt
        assert "Use this to decide if it's time to introduce new topics" in prompt

    def test_generate_node_prompt_without_depth(self) -> None:
        """Test prompt generation without dialogue depth."""
        parent_situation = "The king is dead."
        choice_text = "Mourn publicly"
        params = {"loyalty": 45, "ambition": 80}

        prompt = PromptGenerator.generate_node_prompt(
            parent_situation, choice_text, params
        )

        assert "DIALOGUE DEPTH" not in prompt


class TestOllamaClient:
    """Tests for OllamaClient class."""

    def test_init(self) -> None:
        """Test initialization."""
        client = OllamaClient()
        assert client.model == "qwen3:14b"
        assert client._ollama is None

        client = OllamaClient("mistral")
        assert client.model == "mistral"

    def test_get_ollama_success(self) -> None:
        """Test successful ollama import."""
        client = OllamaClient()

        mock_ollama = Mock()

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = client._get_ollama()
            assert result == mock_ollama
            assert client._ollama == mock_ollama

    def test_get_ollama_import_error(self) -> None:
        """Test ollama import error."""
        client = OllamaClient()

        with patch.dict("sys.modules", {"ollama": None}):
            with pytest.raises(LLMError, match="Ollama package not found"):
                client._get_ollama()

    def test_generate_content_success(self) -> None:
        """Test successful content generation."""
        client = OllamaClient()
        mock_ollama = Mock()
        mock_response = {"message": {"content": "Generated content"}}
        mock_ollama.chat.return_value = mock_response
        client._ollama = mock_ollama

        result = client.generate_content("Test prompt")

        assert result == "Generated content"
        mock_ollama.chat.assert_called_once_with(
            model="qwen3:14b", messages=[{"role": "user", "content": "Test prompt"}]
        )

    def test_generate_content_invalid_response(self) -> None:
        """Test generation with invalid response format."""
        client = OllamaClient()
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {"invalid": "response"}
        client._ollama = mock_ollama

        with pytest.raises(LLMError, match="Invalid response format"):
            client.generate_content("Test prompt")

    def test_generate_content_exception(self) -> None:
        """Test generation with exception."""
        client = OllamaClient()
        mock_ollama = Mock()
        mock_ollama.chat.side_effect = Exception("Network error")
        client._ollama = mock_ollama

        with pytest.raises(LLMError, match="Failed to generate content"):
            client.generate_content("Test prompt")

    def test_is_available_success(self) -> None:
        """Test availability check success."""
        client = OllamaClient()
        mock_ollama = Mock()
        mock_ollama.chat.return_value = {"message": {"content": "Hello"}}
        client._ollama = mock_ollama

        assert client.is_available() is True
        mock_ollama.chat.assert_called_once_with(
            model="qwen3:14b", messages=[{"role": "user", "content": "Hello"}]
        )

    def test_is_available_failure(self) -> None:
        """Test availability check failure."""
        client = OllamaClient()
        mock_ollama = Mock()
        mock_ollama.chat.side_effect = Exception("Connection failed")
        client._ollama = mock_ollama

        assert client.is_available() is False


class TestNodeGenerator:
    """Tests for NodeGenerator class."""

    def setup_method(self) -> None:
        """Set up test objects."""
        self.mock_client = Mock(spec=OllamaClient)
        self.mock_prompt_gen = Mock(spec=PromptGenerator)
        self.generator = NodeGenerator(self.mock_client, self.mock_prompt_gen)

    def test_init(self) -> None:
        """Test initialization."""
        client = Mock()
        generator = NodeGenerator(client)

        assert generator.llm_client == client
        assert isinstance(generator.prompt_generator, PromptGenerator)

    def test_generate_node_success(self) -> None:
        """Test successful node generation."""
        # Set up mocks
        self.mock_prompt_gen.generate_node_prompt.return_value = "Test prompt"

        node_data = {
            "situation": "Test situation",
            "choices": [
                {"text": "Choice 1", "next": None},
                {"text": "Choice 2", "next": None},
            ],
        }
        self.mock_client.generate_content.return_value = json.dumps(node_data)

        # Test
        result = self.generator.generate_node(
            "Parent situation", "Choice text", {"param": 1}
        )

        # Verify
        assert result == node_data
        self.mock_prompt_gen.generate_node_prompt.assert_called_once_with(
            "Parent situation", "Choice text", {"param": 1}, None, None, None, None
        )
        self.mock_client.generate_content.assert_called_once_with("Test prompt")

    def test_generate_node_wrapped_json(self) -> None:
        """Test node generation with JSON wrapped in text."""
        self.mock_prompt_gen.generate_node_prompt.return_value = "Test prompt"

        node_data = {
            "situation": "Test situation",
            "choices": [
                {"text": "Choice 1", "next": None},
                {"text": "Choice 2", "next": None},
            ],
        }
        wrapped_content = f"Here's the JSON:\n{json.dumps(node_data)}\nThat's it!"
        self.mock_client.generate_content.return_value = wrapped_content

        result = self.generator.generate_node(
            "Parent situation", "Choice text", {"param": 1}
        )

        assert result == node_data

    def test_generate_node_invalid_json_retry(self) -> None:
        """Test node generation with invalid JSON and retry."""
        self.mock_prompt_gen.generate_node_prompt.return_value = "Test prompt"

        # First call returns invalid JSON, second call succeeds
        node_data = {
            "situation": "Test situation",
            "choices": [
                {"text": "Choice 1", "next": None},
                {"text": "Choice 2", "next": None},
            ],
        }
        self.mock_client.generate_content.side_effect = [
            "invalid json",
            json.dumps(node_data),
        ]

        result = self.generator.generate_node(
            "Parent situation", "Choice text", {"param": 1}, max_retries=2
        )

        assert result == node_data
        assert self.mock_client.generate_content.call_count == 2

    def test_generate_node_max_retries_exceeded(self) -> None:
        """Test node generation when max retries are exceeded."""
        self.mock_prompt_gen.generate_node_prompt.return_value = "Test prompt"
        self.mock_client.generate_content.return_value = "invalid json"

        result = self.generator.generate_node(
            "Parent situation", "Choice text", {"param": 1}, max_retries=2
        )

        assert result is None
        assert self.mock_client.generate_content.call_count == 2

    def test_generate_node_llm_error(self) -> None:
        """Test node generation with LLM error."""
        self.mock_prompt_gen.generate_node_prompt.return_value = "Test prompt"
        self.mock_client.generate_content.side_effect = LLMError("Connection failed")

        result = self.generator.generate_node(
            "Parent situation", "Choice text", {"param": 1}, max_retries=1
        )

        assert result is None

    def test_generate_node_unexpected_error(self) -> None:
        """Test node generation with unexpected error."""
        self.mock_prompt_gen.generate_node_prompt.return_value = "Test prompt"
        self.mock_client.generate_content.side_effect = Exception("Unexpected error")

        result = self.generator.generate_node(
            "Parent situation", "Choice text", {"param": 1}, max_retries=1
        )

        assert result is None
