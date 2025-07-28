#!/usr/bin/env python3
"""
LLM integration for the Bootstrap Game Dialog Generator.

This module handles communication with Ollama and prompt generation
for dialogue tree completion.
"""

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM operations."""

    pass


class PromptGenerator:
    """Generates prompts for the LLM based on dialogue tree context."""

    @staticmethod
    def generate_node_prompt(
        parent_situation: str,
        choice_text: str,
        params: Dict[str, Any],
        dialogue_history: Optional[str] = None,
        rules: Optional[Dict[str, Any]] = None,
        scene: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a prompt for creating a new dialogue node.

        Args:
            parent_situation: The situation description of the parent node
            choice_text: The text of the choice that leads to this node
            params: Game parameters (loyalty, ambition, etc.)
            dialogue_history: Optional historical context from previous nodes
            rules: Optional stylistic rules (language, tone, voice)
            scene: Optional world-building information

        Returns:
            Formatted prompt string
        """
        prompt_parts = ["You are writing a branching dialogue tree for a visual novel."]

        # Include rules if available
        if rules:
            prompt_parts.extend(["", "STYLISTIC RULES:"])
            for key, value in rules.items():
                prompt_parts.append(f"- {key.title()}: {value}")

        # Include scene information if available
        if scene:
            prompt_parts.extend(["", "WORLD-BUILDING CONTEXT:"])
            for key, value in scene.items():
                prompt_parts.append(f"- {key.title()}: {value}")

        # Include dialogue history if available
        if dialogue_history:
            prompt_parts.extend(["", dialogue_history, ""])

        prompt_parts.extend(
            [
                "",
                "The current parent node has this situation:",
                f'"{parent_situation}"',
                "",
                "The player selected this choice:",
                f'"{choice_text}"',
                "",
                f"Parameters: {json.dumps(params)}",
                "",
                "Please generate a new dialogue node as a JSON object with:",
                "- 'situation': string",
                "- 'choices': a list of 2–3 options, each with:",
                "  - 'text': string",
                "  - 'next': null (placeholder)",
                "  - 'effects': dictionary of parameter changes (e.g., "
                '{{"loyalty": 10}})',
                "  - 'suggested_node_id': descriptive snake_case identifier "
                "for the choice outcome (e.g., 'investigate_entropy', "
                "'talk_to_mother_again')",
                "",
                "IMPORTANT: Use only valid JSON syntax. Numbers should be "
                "written without + prefix (e.g., use 10 not +10).",
                "Make the suggested_node_id descriptive and concise, "
                "reflecting what the choice leads to.",
                "",
                "Respond with valid JSON only.",
            ]
        )

        return "\n".join(prompt_parts)


class OllamaClient:
    """Client for interacting with Ollama LLM."""

    def __init__(self, model: str = "llama3"):
        self.model = model
        self._ollama: Optional[Any] = None

    def _get_ollama(self) -> Any:
        """Lazy load ollama to allow for easier testing."""
        if self._ollama is None:
            try:
                import ollama

                self._ollama = ollama
            except ImportError:
                raise LLMError(
                    "Ollama package not found. Please install with: pip install ollama"
                )
        return self._ollama

    def generate_content(self, prompt: str) -> str:
        """
        Generate content using the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Generated content as string

        Raises:
            LLMError: If generation fails
        """
        try:
            ollama = self._get_ollama()
            response: Any = ollama.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )

            if "message" not in response or "content" not in response["message"]:
                raise LLMError("Invalid response format from Ollama")

            content: str = response["message"]["content"].strip()
            logger.info(f"Generated content of length {len(content)}")
            return content

        except Exception as e:
            logger.error(f"Error generating content with Ollama: {e}")
            raise LLMError(f"Failed to generate content: {e}")

    def is_available(self) -> bool:
        """
        Check if Ollama is available and the model is accessible.

        Returns:
            True if available, False otherwise
        """
        try:
            ollama = self._get_ollama()
            # Try a simple request to check if the model is available
            ollama.chat(
                model=self.model, messages=[{"role": "user", "content": "Hello"}]
            )
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False


class NodeGenerator:
    """Generates dialogue nodes using LLM integration."""

    def __init__(
        self,
        llm_client: OllamaClient,
        prompt_generator: Optional[PromptGenerator] = None,
    ):
        self.llm_client = llm_client
        self.prompt_generator = prompt_generator or PromptGenerator()

    def generate_node(
        self,
        parent_situation: str,
        choice_text: str,
        params: Dict[str, Any],
        dialogue_history: Optional[str] = None,
        rules: Optional[Dict[str, Any]] = None,
        scene: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a new dialogue node.

        Args:
            parent_situation: The situation of the parent node
            choice_text: The choice text that leads to this node
            params: Game parameters
            dialogue_history: Optional historical context from previous nodes
            rules: Optional stylistic rules (language, tone, voice)
            scene: Optional world-building information
            max_retries: Maximum number of retry attempts

        Returns:
            Generated node data or None if generation failed
        """
        prompt = self.prompt_generator.generate_node_prompt(
            parent_situation, choice_text, params, dialogue_history, rules, scene
        )

        # Debug: Log the prompt being sent
        logger.info(f"Generated prompt:\n{prompt}")

        for attempt in range(max_retries):
            try:
                logger.info(f"Generating node (attempt {attempt + 1}/{max_retries})")

                content = self.llm_client.generate_content(prompt)

                # Debug: Log the raw content
                logger.info(f"Raw generated content:\n{content}")

                # Try to parse as JSON
                try:
                    node_data: Dict[str, Any] = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from the response if it's wrapped in other text
                    start_idx = content.find("{")
                    end_idx = content.rfind("}")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        json_content = content[start_idx : end_idx + 1]
                        logger.info(f"Extracted JSON content:\n{json_content}")
                        node_data = json.loads(json_content)
                    else:
                        raise

                logger.info("Successfully generated and parsed node data")
                return node_data

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error("All attempts to generate valid JSON failed")
                    return None

            except LLMError as e:
                logger.error(f"LLM error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return None

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return None

        return None
