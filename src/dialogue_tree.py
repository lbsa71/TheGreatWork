#!/usr/bin/env python3
"""
Core dialogue tree logic for the Bootstrap Game Dialog Generator.

This module contains the main functionality for processing dialogue trees,
finding incomplete nodes, and generating content using LLM integration.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DialogueTreeError(Exception):
    """Base exception for dialogue tree operations."""

    pass


class DialogueNode:
    """Represents a single dialogue node with situation and choices."""

    def __init__(self, situation: str, choices: Optional[List[Dict[str, Any]]] = None):
        self.situation = situation
        self.choices = choices or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary representation."""
        return {"situation": self.situation, "choices": self.choices}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueNode":
        """Create a DialogueNode from a dictionary."""
        return cls(situation=data["situation"], choices=data.get("choices", []))


class DialogueTree:
    """Manages a complete dialogue tree with nodes and parameters."""

    def __init__(self, nodes: Dict[str, Any], params: Dict[str, Union[int, float]]):
        self.nodes = nodes
        self.params = params

    def find_first_null_node(self) -> Optional[str]:
        """Find the first node with a null value."""
        for node_id, node_data in self.nodes.items():
            if node_data is None:
                return node_id
        return None

    def find_parent_and_choice(
        self, target_node_id: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Find the parent node and the choice that leads to the target node.

        Returns:
            Tuple of (parent_node_id, choice_dict) or None if not found
        """
        for node_id, node_data in self.nodes.items():
            if node_data is None:
                continue

            if isinstance(node_data, dict) and "choices" in node_data:
                for choice in node_data["choices"]:
                    if choice.get("next") == target_node_id:
                        return node_id, choice
        return None

    def update_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """Update a node with new data."""
        self.nodes[node_id] = node_data

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the tree to a dictionary representation."""
        return {"nodes": self.nodes, "params": self.params}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueTree":
        """Create a DialogueTree from a dictionary."""
        if "nodes" not in data or "params" not in data:
            raise DialogueTreeError(
                "Invalid dialogue tree format: missing 'nodes' or 'params'"
            )

        return cls(nodes=data["nodes"], params=data["params"])


class DialogueTreeManager:
    """Manages dialogue tree file operations and processing."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.tree: Optional[DialogueTree] = None

    def load_tree(self) -> DialogueTree:
        """Load the dialogue tree from file."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.tree = DialogueTree.from_dict(data)
            logger.info(f"Loaded dialogue tree from {self.file_path}")
            return self.tree

        except FileNotFoundError:
            raise DialogueTreeError(f"Dialogue tree file not found: {self.file_path}")
        except json.JSONDecodeError as e:
            raise DialogueTreeError(f"Invalid JSON in dialogue tree file: {e}")
        except Exception as e:
            raise DialogueTreeError(f"Error loading dialogue tree: {e}")

    def save_tree(self, tree: Optional[DialogueTree] = None) -> None:
        """Save the dialogue tree to file."""
        tree_to_save = tree or self.tree
        if tree_to_save is None:
            raise DialogueTreeError("No tree to save")

        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(tree_to_save.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Saved dialogue tree to {self.file_path}")

        except Exception as e:
            raise DialogueTreeError(f"Error saving dialogue tree: {e}")

    def create_backup(self, tree: Optional[DialogueTree] = None) -> Path:
        """Create a timestamped backup of the current tree."""
        tree_to_backup = tree or self.tree
        if tree_to_backup is None:
            raise DialogueTreeError("No tree to backup")

        # Create backup directory if it doesn't exist
        backup_dir = self.file_path.parent / "backup"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{self.file_path.stem}_backup_{timestamp}.json"

        try:
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(tree_to_backup.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Created backup at {backup_path}")
            return backup_path

        except Exception as e:
            raise DialogueTreeError(f"Error creating backup: {e}")


def validate_generated_node(node_data: Dict[str, Any]) -> bool:
    """
    Validate that a generated node has the correct structure.

    Args:
        node_data: The node data to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(node_data, dict):
        return False

    if "situation" not in node_data:
        return False

    if not isinstance(node_data["situation"], str):
        return False

    if "choices" not in node_data:
        return False

    choices = node_data["choices"]
    if not isinstance(choices, list):
        return False

    if len(choices) < 2 or len(choices) > 3:
        return False

    for choice in choices:
        if not isinstance(choice, dict):
            return False

        if "text" not in choice or not isinstance(choice["text"], str):
            return False

        if "next" not in choice:
            return False

        # "next" should be null for placeholder nodes
        if choice["next"] is not None:
            return False

        # "effects" is optional but should be a dict if present
        if "effects" in choice and not isinstance(choice["effects"], dict):
            return False

    return True
