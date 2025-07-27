#!/usr/bin/env python3
"""
Tests for dialogue_tree module.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dialogue_tree import (
    DialogueNode,
    DialogueTree,
    DialogueTreeManager,
    DialogueTreeError,
    validate_generated_node,
)


class TestDialogueNode:
    """Tests for DialogueNode class."""

    def test_init_basic(self):
        """Test basic initialization."""
        node = DialogueNode("Test situation")
        assert node.situation == "Test situation"
        assert node.choices == []

    def test_init_with_choices(self):
        """Test initialization with choices."""
        choices = [{"text": "Choice 1", "next": "node1"}]
        node = DialogueNode("Test situation", choices)
        assert node.situation == "Test situation"
        assert node.choices == choices

    def test_to_dict(self):
        """Test conversion to dictionary."""
        choices = [{"text": "Choice 1", "next": "node1"}]
        node = DialogueNode("Test situation", choices)
        expected = {"situation": "Test situation", "choices": choices}
        assert node.to_dict() == expected

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "situation": "Test situation",
            "choices": [{"text": "Choice 1", "next": "node1"}],
        }
        node = DialogueNode.from_dict(data)
        assert node.situation == "Test situation"
        assert node.choices == data["choices"]

    def test_from_dict_no_choices(self):
        """Test creation from dictionary without choices."""
        data = {"situation": "Test situation"}
        node = DialogueNode.from_dict(data)
        assert node.situation == "Test situation"
        assert node.choices == []


class TestDialogueTree:
    """Tests for DialogueTree class."""

    def setup_method(self):
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
        self.params = {"loyalty": 45, "ambition": 80}
        self.tree = DialogueTree(self.nodes, self.params)

    def test_init(self):
        """Test initialization."""
        assert self.tree.nodes == self.nodes
        assert self.tree.params == self.params

    def test_find_first_null_node(self):
        """Test finding the first null node."""
        result = self.tree.find_first_null_node()
        assert result == "node1"  # Should find the first null node

    def test_find_first_null_node_none(self):
        """Test when no null nodes exist."""
        # Remove null nodes
        self.tree.nodes = {"start": self.nodes["start"], "node3": self.nodes["node3"]}
        result = self.tree.find_first_null_node()
        assert result is None

    def test_find_parent_and_choice(self):
        """Test finding parent node and choice."""
        result = self.tree.find_parent_and_choice("node1")
        assert result is not None
        parent_id, choice = result
        assert parent_id == "start"
        assert choice["text"] == "Mourn publicly"
        assert choice["next"] == "node1"

    def test_find_parent_and_choice_not_found(self):
        """Test when parent is not found."""
        result = self.tree.find_parent_and_choice("nonexistent")
        assert result is None

    def test_update_node(self):
        """Test updating a node."""
        new_data = {"situation": "New situation", "choices": []}
        self.tree.update_node("node1", new_data)
        assert self.tree.nodes["node1"] == new_data

    def test_get_node(self):
        """Test getting a node."""
        result = self.tree.get_node("start")
        assert result == self.nodes["start"]

    def test_get_node_not_found(self):
        """Test getting a non-existent node."""
        result = self.tree.get_node("nonexistent")
        assert result is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = self.tree.to_dict()
        expected = {"nodes": self.nodes, "params": self.params}
        assert result == expected

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"nodes": self.nodes, "params": self.params}
        tree = DialogueTree.from_dict(data)
        assert tree.nodes == self.nodes
        assert tree.params == self.params

    def test_from_dict_invalid(self):
        """Test creation from invalid dictionary."""
        with pytest.raises(DialogueTreeError):
            DialogueTree.from_dict({"invalid": "data"})


class TestDialogueTreeManager:
    """Tests for DialogueTreeManager class."""

    def setup_method(self):
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

    def test_init(self):
        """Test initialization."""
        manager = DialogueTreeManager("test.json")
        assert manager.file_path == Path("test.json")
        assert manager.tree is None

    def test_load_tree_success(self):
        """Test successful tree loading."""
        json_content = json.dumps(self.test_data)

        with patch("builtins.open", mock_open(read_data=json_content)):
            manager = DialogueTreeManager("test.json")
            tree = manager.load_tree()

            assert isinstance(tree, DialogueTree)
            assert tree.nodes == self.test_data["nodes"]
            assert tree.params == self.test_data["params"]
            assert manager.tree == tree

    def test_load_tree_file_not_found(self):
        """Test loading when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            manager = DialogueTreeManager("nonexistent.json")
            with pytest.raises(DialogueTreeError, match="not found"):
                manager.load_tree()

    def test_load_tree_invalid_json(self):
        """Test loading invalid JSON."""
        with patch("builtins.open", mock_open(read_data="invalid json")):
            manager = DialogueTreeManager("test.json")
            with pytest.raises(DialogueTreeError, match="Invalid JSON"):
                manager.load_tree()

    def test_save_tree_success(self):
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

    def test_save_tree_no_tree(self):
        """Test saving when no tree is loaded."""
        manager = DialogueTreeManager("test.json")
        with pytest.raises(DialogueTreeError, match="No tree to save"):
            manager.save_tree()

    def test_create_backup_success(self):
        """Test successful backup creation."""
        tree = DialogueTree.from_dict(self.test_data)

        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            with patch("dialogue_tree.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20231225_120000"

                manager = DialogueTreeManager("test.json")
                manager.tree = tree
                backup_path = manager.create_backup()

                expected_path = Path("test_backup_20231225_120000.json")
                assert backup_path == expected_path
                mock_file.assert_called_once_with(expected_path, "w", encoding="utf-8")

    def test_create_backup_no_tree(self):
        """Test backup creation when no tree is loaded."""
        manager = DialogueTreeManager("test.json")
        with pytest.raises(DialogueTreeError, match="No tree to backup"):
            manager.create_backup()


class TestValidateGeneratedNode:
    """Tests for validate_generated_node function."""

    def test_valid_node(self):
        """Test validation of a valid node."""
        node = {
            "situation": "Test situation",
            "choices": [
                {"text": "Choice 1", "next": None, "effects": {"loyalty": 10}},
                {"text": "Choice 2", "next": None},
            ],
        }
        assert validate_generated_node(node) is True

    def test_invalid_not_dict(self):
        """Test validation when node is not a dictionary."""
        assert validate_generated_node("not a dict") is False
        assert validate_generated_node(None) is False
        assert validate_generated_node([]) is False

    def test_invalid_no_situation(self):
        """Test validation when situation is missing."""
        node = {"choices": []}
        assert validate_generated_node(node) is False

    def test_invalid_situation_not_string(self):
        """Test validation when situation is not a string."""
        node = {"situation": 123, "choices": []}
        assert validate_generated_node(node) is False

    def test_invalid_no_choices(self):
        """Test validation when choices is missing."""
        node = {"situation": "Test"}
        assert validate_generated_node(node) is False

    def test_invalid_choices_not_list(self):
        """Test validation when choices is not a list."""
        node = {"situation": "Test", "choices": "not a list"}
        assert validate_generated_node(node) is False

    def test_invalid_too_few_choices(self):
        """Test validation when there are too few choices."""
        node = {"situation": "Test", "choices": [{"text": "Only one", "next": None}]}
        assert validate_generated_node(node) is False

    def test_invalid_too_many_choices(self):
        """Test validation when there are too many choices."""
        node = {
            "situation": "Test",
            "choices": [
                {"text": "Choice 1", "next": None},
                {"text": "Choice 2", "next": None},
                {"text": "Choice 3", "next": None},
                {"text": "Choice 4", "next": None},
            ],
        }
        assert validate_generated_node(node) is False

    def test_invalid_choice_not_dict(self):
        """Test validation when choice is not a dictionary."""
        node = {
            "situation": "Test",
            "choices": ["not a dict", {"text": "Valid", "next": None}],
        }
        assert validate_generated_node(node) is False

    def test_invalid_choice_no_text(self):
        """Test validation when choice has no text."""
        node = {
            "situation": "Test",
            "choices": [{"next": None}, {"text": "Valid", "next": None}],
        }
        assert validate_generated_node(node) is False

    def test_invalid_choice_text_not_string(self):
        """Test validation when choice text is not a string."""
        node = {
            "situation": "Test",
            "choices": [{"text": 123, "next": None}, {"text": "Valid", "next": None}],
        }
        assert validate_generated_node(node) is False

    def test_invalid_choice_no_next(self):
        """Test validation when choice has no next field."""
        node = {
            "situation": "Test",
            "choices": [{"text": "Missing next"}, {"text": "Valid", "next": None}],
        }
        assert validate_generated_node(node) is False

    def test_invalid_choice_next_not_null(self):
        """Test validation when choice next is not null."""
        node = {
            "situation": "Test",
            "choices": [
                {"text": "Invalid", "next": "some_node"},
                {"text": "Valid", "next": None},
            ],
        }
        assert validate_generated_node(node) is False

    def test_invalid_effects_not_dict(self):
        """Test validation when effects is not a dictionary."""
        node = {
            "situation": "Test",
            "choices": [
                {"text": "Invalid", "next": None, "effects": "not a dict"},
                {"text": "Valid", "next": None},
            ],
        }
        assert validate_generated_node(node) is False

    def test_valid_effects_optional(self):
        """Test that effects field is optional."""
        node = {
            "situation": "Test",
            "choices": [
                {"text": "No effects", "next": None},
                {"text": "With effects", "next": None, "effects": {"loyalty": 5}},
            ],
        }
        assert validate_generated_node(node) is True
