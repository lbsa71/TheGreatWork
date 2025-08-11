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
from typing import Any, Dict, List, Optional, Tuple, Union, Set

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

    def __init__(
        self,
        nodes: Dict[str, Any],
        params: Dict[str, Union[int, float]],
        rules: Optional[Dict[str, Any]] = None,
        scene: Optional[Dict[str, Any]] = None,
    ):
        self.nodes = nodes
        self.params = params
        self.rules = rules or {}
        self.scene = scene or {}

    def find_first_null_node(self) -> Optional[str]:
        """Find the first node with a null value."""
        for node_id, node_data in self.nodes.items():
            if node_data is None:
                return node_id
            # Skip nodes that have been marked as failed
            if isinstance(node_data, dict) and node_data.get("__failed__"):
                continue
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

    def build_dialogue_history(self, target_node_id: str) -> str:
        """
        Build a dialogue history by backtracking from target node to root.

        Args:
            target_node_id: The node to build history for

        Returns:
            Formatted string containing the dialogue history
        """
        history_steps = []
        current_node_id = target_node_id

        # Backtrack through the tree to build history
        while current_node_id:
            parent_info = self.find_parent_and_choice(current_node_id)
            if parent_info is None:
                # We've reached the root or a disconnected node
                break

            parent_id, choice = parent_info
            parent_node = self.get_node(parent_id)

            if parent_node is None:
                break

            # Add this step to the history
            situation = parent_node.get("situation", "")
            choice_text = choice.get("text", "")
            history_steps.append((situation, choice_text))

            # Move to the parent for next iteration
            current_node_id = parent_id

        # Reverse to get chronological order (root to target)
        history_steps.reverse()

        # Format the history as a readable string
        if not history_steps:
            return "No previous dialogue history available."

        history_lines = ["Dialogue History:"]
        for i, (situation, choice) in enumerate(history_steps, 1):
            history_lines.append(f"{i}. Situation: {situation}")
            history_lines.append(f"   Player chose: {choice}")

        return "\n".join(history_lines)

    def generate_unique_node_id(self, suggested_id: str) -> str:
        """
        Generate a unique node ID based on a suggested ID, handling duplicates
        with suffixes.

        Args:
            suggested_id: The suggested node ID from the AI

        Returns:
            A unique node ID, possibly with a numeric suffix
        """
        return generate_unique_node_id(suggested_id, self.nodes)

    def validate_and_fix_tree(self) -> None:
        """
        Validate and fix the dialogue tree structure.

        This method performs the following validations and fixes:
        1. All referenced nodes exist (create null nodes if missing)
        2. All required fields are present on all nodes (convert to null if invalid)
        3. No extra fields are present on nodes (remove extra fields)
        4. All nodes occur in at least one 'next' reference (remove orphaned nodes)
        5. All 'effect' params exist in root level 'params' collection (add if missing)
        """
        logger.info("Starting dialogue tree validation and fixing")

        # Step 1: Ensure all referenced nodes exist
        self._ensure_referenced_nodes_exist()

        # Step 2: Fix node structure and remove extra fields
        self._fix_node_structure()

        # Step 3: Remove orphaned nodes (after structure fixing)
        self._remove_orphaned_nodes()

        # Step 4: Extract and add missing effect parameters
        self._extract_and_add_missing_params()

        logger.info("Dialogue tree validation and fixing completed")

    def _fix_node_structure(self) -> None:
        """Fix node structure and remove extra fields."""
        nodes_to_nullify = []

        for node_id, node_data in self.nodes.items():
            if node_data is None:
                continue

            if not isinstance(node_data, dict):
                nodes_to_nullify.append(node_id)
                continue

            # Check required fields
            if "situation" not in node_data or not isinstance(
                node_data["situation"], str
            ):
                nodes_to_nullify.append(node_id)
                continue

            if "choices" not in node_data or not isinstance(node_data["choices"], list):
                nodes_to_nullify.append(node_id)
                continue

            # Validate choices structure
            valid_choices = []
            for choice in node_data["choices"]:
                if not isinstance(choice, dict):
                    continue
                if "text" not in choice or not isinstance(choice["text"], str):
                    continue

                # Clean choice - only keep allowed fields
                clean_choice: Dict[str, Any] = {"text": choice["text"]}

                # Add next if present
                if "next" in choice:
                    clean_choice["next"] = choice["next"]

                # Add effects if present and valid
                if "effects" in choice and isinstance(choice["effects"], dict):
                    clean_choice["effects"] = choice["effects"]

                valid_choices.append(clean_choice)

            # Check if we have valid choices (allow empty choices for some nodes)
            # Don't nullify nodes just because they have no choices

            # Clean node - only keep allowed fields
            clean_node: Dict[str, Any] = {
                "situation": node_data["situation"],
                "choices": valid_choices,
            }

            self.nodes[node_id] = clean_node

        # Nullify invalid nodes
        for node_id in nodes_to_nullify:
            logger.warning(f"Converting invalid node '{node_id}' to null")
            self.nodes[node_id] = None

    def _ensure_referenced_nodes_exist(self) -> None:
        """Ensure all referenced nodes exist, creating null nodes if missing."""
        referenced_nodes = set()

        # Collect all referenced node IDs
        for node_id, node_data in self.nodes.items():
            if node_data is None or not isinstance(node_data, dict):
                continue

            choices = node_data.get("choices", [])
            for choice in choices:
                if isinstance(choice, dict) and "next" in choice:
                    next_node = choice["next"]
                    if next_node is not None:  # Skip null next values
                        referenced_nodes.add(next_node)

        # Create missing nodes as null
        for node_id in referenced_nodes:
            if node_id not in self.nodes:
                logger.info(f"Creating missing referenced node '{node_id}' as null")
                self.nodes[node_id] = None

    def _remove_orphaned_nodes(self) -> None:
        """Remove nodes that are not referenced by any choice and are not meaningful roots."""
        referenced_nodes = set()

        # Collect all referenced node IDs from choices
        for node_id, node_data in self.nodes.items():
            if node_data is None or not isinstance(node_data, dict):
                continue

            choices = node_data.get("choices", [])
            for choice in choices:
                if isinstance(choice, dict) and "next" in choice:
                    next_node = choice["next"]
                    if next_node is not None:
                        referenced_nodes.add(next_node)

        # Find unreferenced nodes
        all_nodes = set(self.nodes.keys())
        unreferenced_nodes = all_nodes - referenced_nodes

        # Remove unreferenced nodes that are truly orphaned
        # Keep unreferenced nodes that:
        # 1. Are called 'start' (conventional root)
        # 2. Have valid content and choices (potential alternate entry points)
        orphaned_nodes = set()
        for node_id in unreferenced_nodes:
            node_data = self.nodes[node_id]

            # Always keep 'start' node
            if node_id == "start":
                continue

            # Keep null nodes that might be placeholders
            if node_data is None:
                orphaned_nodes.add(node_id)
                continue

            # Keep valid nodes with meaningful choices (including ones that end the dialogue)
            if isinstance(node_data, dict) and "choices" in node_data:
                choices = node_data.get("choices", [])
                has_any_choices = len(choices) > 0

                # If it has any choices at all (even ending ones), it might be a legitimate entry point
                if has_any_choices:
                    continue

            # This node seems truly orphaned
            orphaned_nodes.add(node_id)

        # Remove the orphaned nodes
        for node_id in orphaned_nodes:
            logger.info(f"Removing orphaned node '{node_id}'")
            del self.nodes[node_id]

    def _extract_and_add_missing_params(self) -> None:
        """Extract effect parameters from choices and add missing ones to root params."""
        all_effect_params: set[str] = set()

        # Collect all effect parameter names
        for node_id, node_data in self.nodes.items():
            if node_data is None or not isinstance(node_data, dict):
                continue

            choices = node_data.get("choices", [])
            for choice in choices:
                if isinstance(choice, dict) and "effects" in choice:
                    effects = choice["effects"]
                    if isinstance(effects, dict):
                        all_effect_params.update(effects.keys())

        # Add missing parameters to root params with default value 0
        for param_name in all_effect_params:
            if param_name not in self.params:
                logger.info(
                    f"Adding missing parameter '{param_name}' to root params with default value 0"
                )
                self.params[param_name] = 0

    def generate_illustrations(
        self,
        images_dir: str = "images",
        style_tokens: Optional[List[str]] = None,
        max_nodes: Optional[int] = None,
        **generation_kwargs: Any,
    ) -> Tuple[int, Any]:
        """
        Generate illustrations for nodes without illustrations using BFS.

        Args:
            images_dir: Directory to save generated images
            style_tokens: Additional style tokens for better generation
            max_nodes: Maximum number of nodes to process
            **generation_kwargs: Additional arguments for image generation

        Returns:
            Tuple of (number of illustrations generated, generation statistics)
        """
        from pathlib import Path
        
        try:
            from .image_generation import generate_illustrations_for_nodes
        except ImportError:
            logger.error("Image generation module not available. Install required dependencies.")
            from .image_generation import ImageGenerationStats
            return 0, ImageGenerationStats()

        # Use scene context for prompt building
        context = self.scene.copy() if self.scene else {}
        
        # Add rules as context if available
        if self.rules:
            for key, value in self.rules.items():
                if key not in context:
                    context[key] = value

        return generate_illustrations_for_nodes(
            tree_nodes=self.nodes,
            context=context,
            images_dir=Path(images_dir),
            style_tokens=style_tokens,
            max_nodes=max_nodes,
            **generation_kwargs,
        )

    def find_nodes_without_illustrations(self) -> List[str]:
        """
        Find all nodes that don't have illustrations using breadth-first search.

        Returns:
            List of node IDs without illustrations, in BFS order
        """
        try:
            from .image_generation import DialogueTreeIllustrationGenerator, StableDiffusionXLGenerator
            
            # Create a temporary generator just for the BFS logic
            temp_generator = StableDiffusionXLGenerator()
            illustration_gen = DialogueTreeIllustrationGenerator(temp_generator)
            return illustration_gen.find_nodes_without_illustrations(self.nodes)
        except ImportError:
            logger.warning("Image generation module not available")
            return []
        except Exception as e:
            logger.warning(f"Could not initialize image generation: {e}")
            # Fallback: simple BFS without illustration generator
            return self._simple_bfs_without_illustrations()

    def _simple_bfs_without_illustrations(self) -> List[str]:
        """
        Simple BFS implementation to find nodes without illustrations.
        Used as fallback when image generation dependencies are not available.
        """
        from collections import deque

        if not self.nodes:
            return []

        nodes_without_illustrations = []
        queue = deque()
        visited: Set[str] = set()

        # Find root nodes (nodes not referenced by others)
        referenced_nodes = set()
        for node_data in self.nodes.values():
            if isinstance(node_data, dict) and "choices" in node_data:
                for choice in node_data["choices"]:
                    if choice.get("next"):
                        referenced_nodes.add(choice["next"])

        # Start BFS from root nodes
        for node_id in self.nodes.keys():
            if node_id not in referenced_nodes:  # This is a root node
                queue.append(node_id)

        # If no clear root found, start with 'start' or first node
        if not queue:
            if "start" in self.nodes:
                queue.append("start")
            elif self.nodes:
                queue.append(next(iter(self.nodes)))

        # Perform BFS
        while queue:
            current_id = queue.popleft()
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            current_node = self.nodes.get(current_id)

            # Skip null nodes
            if current_node is None:
                continue

            # Check if node needs illustration
            if isinstance(current_node, dict):
                if not current_node.get("illustration"):
                    nodes_without_illustrations.append(current_id)

                # Add child nodes to queue
                if "choices" in current_node:
                    for choice in current_node["choices"]:
                        next_node_id = choice.get("next")
                        if next_node_id and next_node_id not in visited:
                            queue.append(next_node_id)

        return nodes_without_illustrations

    def to_dict(self) -> Dict[str, Any]:
        """Convert the tree to a dictionary representation."""
        result = {"nodes": self.nodes, "params": self.params}
        if self.rules:
            result["rules"] = self.rules
        if self.scene:
            result["scene"] = self.scene
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueTree":
        """Create a DialogueTree from a dictionary."""
        if "nodes" not in data or "params" not in data:
            raise DialogueTreeError(
                "Invalid dialogue tree format: missing 'nodes' or 'params'"
            )

        return cls(
            nodes=data["nodes"],
            params=data["params"],
            rules=data.get("rules"),
            scene=data.get("scene"),
        )


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

            # Validate and fix the tree structure
            self.tree.validate_and_fix_tree()

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

        # Validate and fix the tree before saving
        tree_to_save.validate_and_fix_tree()

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


def generate_unique_node_id(suggested_id: str, existing_nodes: Dict[str, Any]) -> str:
    """
    Generate a unique node ID based on a suggested ID, handling duplicates
    with suffixes.

    Args:
        suggested_id: The suggested node ID from the AI
        existing_nodes: Dictionary of existing nodes to check against

    Returns:
        A unique node ID, possibly with a numeric suffix
    """
    if not suggested_id:
        # Fallback to generic naming if no suggestion provided
        return f"node_{len(existing_nodes) + 1}"

    # Ensure the suggested ID is in snake_case and alphanumeric
    import re

    clean_id = re.sub(r"[^a-zA-Z0-9_]", "_", suggested_id.lower())
    clean_id = re.sub(r"_+", "_", clean_id).strip("_")

    if not clean_id:
        # Fallback if cleaning resulted in empty string
        return f"node_{len(existing_nodes) + 1}"

    # Check if the base ID is available
    if clean_id not in existing_nodes:
        return clean_id

    # Find the next available suffix
    counter = 2
    while f"{clean_id}_{counter}" in existing_nodes:
        counter += 1

    return f"{clean_id}_{counter}"


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

        # "suggested_node_id" is optional but should be a string if present
        if "suggested_node_id" in choice and not isinstance(
            choice["suggested_node_id"], str
        ):
            return False

    return True
