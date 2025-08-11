#!/usr/bin/env python3
"""
Tests for dialogue_tree module.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import mock_open, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.dialogue_tree import (
    DialogueNode,
    DialogueTree,
    DialogueTreeError,
    DialogueTreeManager,
    generate_unique_node_id,
    validate_generated_node,
)


class TestDialogueNode:
    """Tests for DialogueNode class."""

    def test_init_basic(self) -> None:
        """Test basic initialization."""
        node = DialogueNode("Test situation")
        assert node.situation == "Test situation"
        assert node.choices == []

    def test_init_with_choices(self) -> None:
        """Test initialization with choices."""
        choices = [{"text": "Choice 1", "next": "node1"}]
        node = DialogueNode("Test situation", choices)
        assert node.situation == "Test situation"
        assert node.choices == choices

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        choices = [{"text": "Choice 1", "next": "node1"}]
        node = DialogueNode("Test situation", choices)
        expected = {"situation": "Test situation", "choices": choices}
        assert node.to_dict() == expected

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "situation": "Test situation",
            "choices": [{"text": "Choice 1", "next": "node1"}],
        }
        node = DialogueNode.from_dict(data)
        assert node.situation == "Test situation"
        assert node.choices == data["choices"]

    def test_from_dict_no_choices(self) -> None:
        """Test creation from dictionary without choices."""
        data = {"situation": "Test situation"}
        node = DialogueNode.from_dict(data)
        assert node.situation == "Test situation"
        assert node.choices == []


class TestDialogueTree:
    """Tests for DialogueTree class."""

    def setup_method(self) -> None:
        """Set up test data."""
        self.nodes = {
            "start": {
                "situation": "The king is dead.",
                "choices": [
                    {"text": "Mourn publicly", "next": "node1"},
                    {"text": "Seize the throne", "next": "node2"},
                ],
            },
            "node1": None,
            "node2": None,
            "node3": {
                "situation": "You are in the castle.",
                "choices": [{"text": "Exit", "next": "end"}],
            },
        }
        self.params = {"loyalty": 45.0, "ambition": 80.0}
        self.tree = DialogueTree(self.nodes, self.params)

    def test_init(self) -> None:
        """Test initialization."""
        assert self.tree.nodes == self.nodes
        assert self.tree.params == self.params
        assert self.tree.rules == {}
        assert self.tree.scene == {}

    def test_init_with_rules_and_scene(self) -> None:
        """Test initialization with rules and scene."""
        rules = {"tone": "dramatic", "voice": "third person"}
        scene = {"setting": "medieval kingdom", "atmosphere": "tense"}
        tree = DialogueTree(self.nodes, self.params, rules, scene)

        assert tree.nodes == self.nodes
        assert tree.params == self.params
        assert tree.rules == rules
        assert tree.scene == scene

    def test_find_first_null_node(self) -> None:
        """Test finding the first null node."""
        result = self.tree.find_first_null_node()
        assert result == "node1"  # Should find the first null node

    def test_find_first_null_node_none(self) -> None:
        """Test when no null nodes exist."""
        # Remove null nodes
        self.tree.nodes = {"start": self.nodes["start"], "node3": self.nodes["node3"]}
        result = self.tree.find_first_null_node()
        assert result is None

    def test_find_parent_and_choice(self) -> None:
        """Test finding parent node and choice."""
        result = self.tree.find_parent_and_choice("node1")
        assert result is not None
        parent_id, choice = result
        assert parent_id == "start"
        assert choice["text"] == "Mourn publicly"
        assert choice["next"] == "node1"

    def test_find_parent_and_choice_not_found(self) -> None:
        """Test when parent is not found."""
        result = self.tree.find_parent_and_choice("nonexistent")
        assert result is None

    def test_update_node(self) -> None:
        """Test updating a node."""
        new_data = {"situation": "New situation", "choices": []}
        self.tree.update_node("node1", new_data)
        assert self.tree.nodes["node1"] == new_data

    def test_get_node(self) -> None:
        """Test getting a node."""
        result = self.tree.get_node("start")
        assert result == self.nodes["start"]

    def test_get_node_not_found(self) -> None:
        """Test getting a non-existent node."""
        result = self.tree.get_node("nonexistent")
        assert result is None

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = self.tree.to_dict()
        expected = {"nodes": self.nodes, "params": self.params}
        assert result == expected

    def test_to_dict_with_rules_and_scene(self) -> None:
        """Test conversion to dictionary with rules and scene."""
        rules = {"tone": "dramatic"}
        scene = {"setting": "medieval kingdom"}
        tree = DialogueTree(self.nodes, self.params, rules, scene)
        result = tree.to_dict()
        expected = {
            "nodes": self.nodes,
            "params": self.params,
            "rules": rules,
            "scene": scene,
        }
        assert result == expected

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {"nodes": self.nodes, "params": self.params}
        tree = DialogueTree.from_dict(data)
        assert tree.nodes == self.nodes
        assert tree.params == self.params
        assert tree.rules == {}
        assert tree.scene == {}

    def test_from_dict_with_rules_and_scene(self) -> None:
        """Test creation from dictionary with rules and scene."""
        rules = {"tone": "dramatic"}
        scene = {"setting": "medieval kingdom"}
        data = {
            "nodes": self.nodes,
            "params": self.params,
            "rules": rules,
            "scene": scene,
        }
        tree = DialogueTree.from_dict(data)
        assert tree.nodes == self.nodes
        assert tree.params == self.params
        assert tree.rules == rules
        assert tree.scene == scene

    def test_from_dict_invalid(self) -> None:
        """Test creation from invalid dictionary."""
        with pytest.raises(DialogueTreeError):
            DialogueTree.from_dict({"invalid": "data"})

    def test_build_dialogue_history_simple_chain(self) -> None:
        """Test building dialogue history for a simple chain."""
        # Create a simple chain: start -> node1 -> node2
        nodes = {
            "start": {
                "situation": "The kingdom is in chaos.",
                "choices": [
                    {"text": "Seek the wise council", "next": "node1"},
                    {"text": "Take action immediately", "next": "other"},
                ],
            },
            "node1": {
                "situation": "You meet with the council.",
                "choices": [
                    {"text": "Ask for advice", "next": "node2"},
                ],
            },
            "node2": None,
            "other": None,
        }
        tree = DialogueTree(nodes, {"loyalty": 50.0})

        # Test history for node2 (should include start -> node1 -> node2)
        history = tree.build_dialogue_history("node2")

        expected_lines = [
            "Dialogue History:",
            "1. Situation: The kingdom is in chaos.",
            "   Player chose: Seek the wise council",
            "2. Situation: You meet with the council.",
            "   Player chose: Ask for advice",
        ]
        expected_history = "\n".join(expected_lines)

        assert history == expected_history

    def test_build_dialogue_history_no_parent(self) -> None:
        """Test building dialogue history for root node or disconnected node."""
        tree = DialogueTree(self.nodes, self.params)

        # Test history for start node (root)
        history = tree.build_dialogue_history("start")
        assert history == "No previous dialogue history available."

        # Test history for disconnected node
        tree.nodes["orphan"] = None
        history = tree.build_dialogue_history("orphan")
        assert history == "No previous dialogue history available."

    def test_build_dialogue_history_single_step(self) -> None:
        """Test building dialogue history with one step."""
        tree = DialogueTree(self.nodes, self.params)

        # Test history for node1 (direct child of start)
        history = tree.build_dialogue_history("node1")

        expected_lines = [
            "Dialogue History:",
            "1. Situation: The king is dead.",
            "   Player chose: Mourn publicly",
        ]
        expected_history = "\n".join(expected_lines)

        assert history == expected_history

    def test_calculate_dialogue_depth_simple_chain(self) -> None:
        """Test calculating dialogue depth for a simple chain."""
        # Create a simple chain: start -> node1 -> node2
        nodes = {
            "start": {
                "situation": "The kingdom is in chaos.",
                "choices": [
                    {"text": "Seek the wise council", "next": "node1"},
                    {"text": "Take action immediately", "next": "other"},
                ],
            },
            "node1": {
                "situation": "You meet with the council.",
                "choices": [
                    {"text": "Ask for advice", "next": "node2"},
                ],
            },
            "node2": None,
            "other": None,
        }
        tree = DialogueTree(nodes, {"loyalty": 50.0})

        # Test depth for each node
        assert tree.calculate_dialogue_depth("start") == 0  # Root node
        assert tree.calculate_dialogue_depth("node1") == 1  # 1 step from start
        assert tree.calculate_dialogue_depth("node2") == 2  # 2 steps from start
        assert tree.calculate_dialogue_depth("other") == 1  # 1 step from start

    def test_calculate_dialogue_depth_no_parent(self) -> None:
        """Test calculating dialogue depth for root node or disconnected node."""
        tree = DialogueTree(self.nodes, self.params)

        # Test depth for start node (root)
        assert tree.calculate_dialogue_depth("start") == 0

        # Test depth for disconnected node
        tree.nodes["orphan"] = None
        assert tree.calculate_dialogue_depth("orphan") == 0

    def test_calculate_dialogue_depth_single_step(self) -> None:
        """Test calculating dialogue depth with one step."""
        tree = DialogueTree(self.nodes, self.params)

        # Test depth for first level node
        depth = tree.calculate_dialogue_depth("node1")
        assert depth == 1


class TestDialogueTreeManager:
    """Tests for DialogueTreeManager class."""

    def setup_method(self) -> None:
        """Set up test data."""
        self.test_data = {
            "nodes": {
                "start": {
                    "situation": "Test situation",
                    "choices": [{"text": "Test choice", "next": "node1"}],
                },
                "node1": None,
            },
            "params": {"test_param": 100},
        }

    def test_init(self) -> None:
        """Test initialization."""
        manager = DialogueTreeManager("test.json")
        assert manager.file_path == Path("test.json")
        assert manager.tree is None

    def test_load_tree_success(self) -> None:
        """Test successful tree loading."""
        json_content = json.dumps(self.test_data)

        with patch("builtins.open", mock_open(read_data=json_content)):
            manager = DialogueTreeManager("test.json")
            tree = manager.load_tree()

            assert isinstance(tree, DialogueTree)
            assert tree.nodes == self.test_data["nodes"]
            assert tree.params == self.test_data["params"]
            assert manager.tree == tree

    def test_load_tree_file_not_found(self) -> None:
        """Test loading when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            manager = DialogueTreeManager("nonexistent.json")
            with pytest.raises(DialogueTreeError, match="not found"):
                manager.load_tree()

    def test_load_tree_invalid_json(self) -> None:
        """Test loading invalid JSON."""
        with patch("builtins.open", mock_open(read_data="invalid json")):
            manager = DialogueTreeManager("test.json")
            with pytest.raises(DialogueTreeError, match="Invalid JSON"):
                manager.load_tree()

    def test_save_tree_success(self) -> None:
        """Test successful tree saving."""
        tree = DialogueTree.from_dict(self.test_data)

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            manager = DialogueTreeManager("test.json")
            manager.tree = tree
            manager.save_tree()

            mock_file.assert_called_once_with(Path("test.json"), "w", encoding="utf-8")
            # Check that JSON was written
            handle = mock_file()
            written_data = "".join(call.args[0] for call in handle.write.call_args_list)
            assert json.loads(written_data) == self.test_data

    def test_save_tree_no_tree(self) -> None:
        """Test saving when no tree is loaded."""
        manager = DialogueTreeManager("test.json")
        with pytest.raises(DialogueTreeError, match="No tree to save"):
            manager.save_tree()

    def test_create_backup_success(self) -> None:
        """Test successful backup creation."""
        tree = DialogueTree.from_dict(self.test_data)

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            with patch("src.dialogue_tree.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20231225_120000"

                manager = DialogueTreeManager("test.json")
                manager.tree = tree
                backup_path = manager.create_backup()

                expected_path = Path("backup/test_backup_20231225_120000.json")
                assert backup_path == expected_path
                mock_file.assert_called_once_with(expected_path, "w", encoding="utf-8")

    def test_create_backup_no_tree(self) -> None:
        """Test backup creation when no tree is loaded."""
        manager = DialogueTreeManager("test.json")
        with pytest.raises(DialogueTreeError, match="No tree to backup"):
            manager.create_backup()


class TestValidateGeneratedNode:
    """Tests for validate_generated_node function."""

    def test_valid_node(self) -> None:
        """Test validation of a valid node."""
        node = {
            "situation": "Test situation",
            "choices": [
                {"text": "Choice 1", "next": None, "effects": {"loyalty": 10}},
                {"text": "Choice 2", "next": None},
            ],
        }
        assert validate_generated_node(node) is True

    def test_invalid_not_dict(self) -> None:
        """Test validation when node is not a dictionary."""
        assert validate_generated_node("not a dict") is False  # type: ignore
        assert validate_generated_node(None) is False  # type: ignore
        assert validate_generated_node([]) is False  # type: ignore

    def test_invalid_no_situation(self) -> None:
        """Test validation when situation is missing."""
        node: Dict[str, Any] = {"choices": []}
        assert validate_generated_node(node) is False

    def test_invalid_situation_not_string(self) -> None:
        """Test validation when situation is not a string."""
        node: Dict[str, Any] = {"situation": 123, "choices": []}
        assert validate_generated_node(node) is False

    def test_invalid_no_choices(self) -> None:
        """Test validation when choices is missing."""
        node: Dict[str, Any] = {"situation": "Test"}
        assert validate_generated_node(node) is False

    def test_invalid_choices_not_list(self) -> None:
        """Test validation when choices is not a list."""
        node: Dict[str, Any] = {"situation": "Test", "choices": "not a list"}
        assert validate_generated_node(node) is False

    def test_invalid_too_few_choices(self) -> None:
        """Test validation when there are too few choices."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": [{"text": "Only one", "next": None}],
        }
        assert validate_generated_node(node) is False

    def test_invalid_too_many_choices(self) -> None:
        """Test validation when there are too many choices."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": [
                {"text": "Choice 1", "next": None},
                {"text": "Choice 2", "next": None},
                {"text": "Choice 3", "next": None},
                {"text": "Choice 4", "next": None},
            ],
        }
        assert validate_generated_node(node) is False

    def test_invalid_choice_not_dict(self) -> None:
        """Test validation when choice is not a dictionary."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": ["not a dict", {"text": "Valid", "next": None}],
        }
        assert validate_generated_node(node) is False

    def test_invalid_choice_no_text(self) -> None:
        """Test validation when choice has no text."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": [{"next": None}, {"text": "Valid", "next": None}],
        }
        assert validate_generated_node(node) is False

    def test_invalid_choice_text_not_string(self) -> None:
        """Test validation when choice text is not a string."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": [{"text": 123, "next": None}, {"text": "Valid", "next": None}],
        }
        assert validate_generated_node(node) is False

    def test_invalid_choice_no_next(self) -> None:
        """Test validation when choice has no next field."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": [{"text": "Missing next"}, {"text": "Valid", "next": None}],
        }
        assert validate_generated_node(node) is False

    def test_invalid_choice_next_not_null(self) -> None:
        """Test validation when choice next is not null."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": [
                {"text": "Invalid", "next": "some_node"},
                {"text": "Valid", "next": None},
            ],
        }
        assert validate_generated_node(node) is False

    def test_invalid_effects_not_dict(self) -> None:
        """Test validation when effects is not a dictionary."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": [
                {"text": "Invalid", "next": None, "effects": "not a dict"},
                {"text": "Valid", "next": None},
            ],
        }
        assert validate_generated_node(node) is False

    def test_valid_effects_optional(self) -> None:
        """Test that effects field is optional."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": [
                {"text": "No effects", "next": None},
                {"text": "With effects", "next": None, "effects": {"loyalty": 5}},
            ],
        }
        assert validate_generated_node(node) is True

    def test_valid_suggested_node_id_optional(self) -> None:
        """Test that suggested_node_id field is optional."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": [
                {"text": "No suggestion", "next": None},
                {
                    "text": "With suggestion",
                    "next": None,
                    "suggested_node_id": "test_node",
                },
            ],
        }
        assert validate_generated_node(node) is True

    def test_invalid_suggested_node_id_not_string(self) -> None:
        """Test validation when suggested_node_id is not a string."""
        node: Dict[str, Any] = {
            "situation": "Test",
            "choices": [
                {"text": "Invalid", "next": None, "suggested_node_id": 123},
                {"text": "Valid", "next": None},
            ],
        }
        assert validate_generated_node(node) is False


class TestGenerateUniqueNodeId:
    """Tests for generate_unique_node_id function."""

    def test_basic_unique_id(self) -> None:
        """Test generating a unique ID when no conflicts exist."""
        existing_nodes = {"start": {}, "node1": {}}
        result = generate_unique_node_id("investigate_entropy", existing_nodes)
        assert result == "investigate_entropy"

    def test_empty_suggestion_fallback(self) -> None:
        """Test fallback behavior when suggestion is empty."""
        existing_nodes = {"start": {}, "node1": {}}
        result = generate_unique_node_id("", existing_nodes)
        assert result == "node_3"

    def test_none_suggestion_fallback(self) -> None:
        """Test fallback behavior when suggestion is None."""
        existing_nodes = {"start": {}, "node1": {}}
        result = generate_unique_node_id(None, existing_nodes)
        assert result == "node_3"

    def test_duplicate_handling_simple(self) -> None:
        """Test handling a simple duplicate."""
        existing_nodes = {"start": {}, "investigate_entropy": {}}
        result = generate_unique_node_id("investigate_entropy", existing_nodes)
        assert result == "investigate_entropy_2"

    def test_duplicate_handling_multiple(self) -> None:
        """Test handling multiple duplicates."""
        existing_nodes = {
            "start": {},
            "investigate_entropy": {},
            "investigate_entropy_2": {},
            "investigate_entropy_3": {},
        }
        result = generate_unique_node_id("investigate_entropy", existing_nodes)
        assert result == "investigate_entropy_4"

    def test_clean_special_characters(self) -> None:
        """Test cleaning special characters from suggestion."""
        existing_nodes = {"start": {}}
        result = generate_unique_node_id("talk to mother again!", existing_nodes)
        assert result == "talk_to_mother_again"

    def test_clean_multiple_underscores(self) -> None:
        """Test cleaning multiple consecutive underscores."""
        existing_nodes = {"start": {}}
        result = generate_unique_node_id("investigate___entropy", existing_nodes)
        assert result == "investigate_entropy"

    def test_clean_leading_trailing_underscores(self) -> None:
        """Test cleaning leading and trailing underscores."""
        existing_nodes = {"start": {}}
        result = generate_unique_node_id("_investigate_entropy_", existing_nodes)
        assert result == "investigate_entropy"

    def test_clean_results_in_empty_string(self) -> None:
        """Test fallback when cleaning results in empty string."""
        existing_nodes = {"start": {}}
        result = generate_unique_node_id("!!!", existing_nodes)
        assert result == "node_2"


class TestDialogueTreeGenerateUniqueNodeId:
    """Tests for DialogueTree.generate_unique_node_id method."""

    def test_generate_unique_node_id_method(self) -> None:
        """Test the method delegates correctly to standalone function."""
        tree = DialogueTree(
            nodes={"start": {"situation": "test"}, "investigate_entropy": None},
            params={"test": 1},
        )
        result = tree.generate_unique_node_id("investigate_entropy")
        assert result == "investigate_entropy_2"

    def test_generate_unique_node_id_new_name(self) -> None:
        """Test generating a completely new name."""
        tree = DialogueTree(nodes={"start": {"situation": "test"}}, params={"test": 1})
        result = tree.generate_unique_node_id("talk_to_mother")
        assert result == "talk_to_mother"
