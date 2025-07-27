#!/usr/bin/env python3
"""
Tests for the dialogue tree debugger.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.debugger import DialogueDebugger, run_debugger
from src.dialogue_tree import DialogueTree


class TestDialogueDebugger:
    """Test the DialogueDebugger class."""

    @pytest.fixture
    def sample_tree(self) -> Any:
        """Create a sample dialogue tree for testing."""
        tree_data = {
            "nodes": {
                "start": {
                    "situation": "You are at the beginning.",
                    "choices": [
                        {"text": "Go left", "next": "left", "effects": {"courage": 5}},
                        {
                            "text": "Go right",
                            "next": "right",
                            "effects": {"wisdom": 10},
                        },
                    ],
                },
                "left": {
                    "situation": "You went left and found a treasure.",
                    "choices": [
                        {
                            "text": "Take treasure",
                            "next": "treasure",
                            "effects": {"wealth": 20},
                        },
                        {"text": "Go back", "next": "start", "effects": {}},
                    ],
                },
                "right": {
                    "situation": "You went right and met a wise sage.",
                    "choices": [
                        {
                            "text": "Ask for advice",
                            "next": "advice",
                            "effects": {"wisdom": 15},
                        },
                        {"text": "Go back", "next": "start", "effects": {}},
                    ],
                },
                "treasure": None,
                "advice": None,
            },
            "params": {"courage": 50, "wisdom": 30, "wealth": 10},
        }
        return DialogueTree.from_dict(tree_data)

    def test_init_with_start_node(self, sample_tree: Any) -> None:
        """Test initialization with a specific start node."""
        debugger = DialogueDebugger(sample_tree, "left")
        assert debugger.current_node_id == "left"
        assert debugger.running is True

    def test_init_auto_find_root(self, sample_tree: Any) -> None:
        """Test automatic root node detection."""
        debugger = DialogueDebugger(sample_tree)
        assert debugger.current_node_id == "start"

    def test_init_no_valid_start_node(self) -> None:
        """Test initialization with no valid starting node."""
        empty_tree = DialogueTree({}, {})
        with pytest.raises(ValueError, match="No valid starting node found"):
            DialogueDebugger(empty_tree)

    def test_find_root_node_with_start(self, sample_tree: Any) -> None:
        """Test finding root node when 'start' exists."""
        debugger = DialogueDebugger(sample_tree)
        root = debugger._find_root_node()
        assert root == "start"

    def test_find_root_node_without_start(self) -> None:
        """Test finding root node when 'start' doesn't exist."""
        tree_data = {
            "nodes": {
                "isolated": {
                    "situation": "You are isolated.",
                    "choices": [{"text": "Stay", "next": None}],
                }
            },
            "params": {},
        }
        tree = DialogueTree.from_dict(tree_data)
        debugger = DialogueDebugger(tree)
        assert debugger.current_node_id == "isolated"

    @patch("builtins.print")
    @patch.object(DialogueTree, "build_dialogue_history", return_value="Test history")
    def test_display_node_valid(
        self, mock_history: Any, mock_print: Any, sample_tree: Any
    ) -> None:
        """Test displaying a valid node."""
        debugger = DialogueDebugger(sample_tree, "start")
        debugger._display_node()

        # Check that print was called multiple times (for the display)
        assert mock_print.call_count > 0

    @patch("builtins.print")
    @patch.object(DialogueTree, "build_dialogue_history", return_value="Test history")
    def test_display_node_null(
        self, mock_history: Any, mock_print: Any, sample_tree: Any
    ) -> None:
        """Test displaying a null node."""
        debugger = DialogueDebugger(sample_tree, "treasure")
        debugger._display_node()

        # Check that print was called multiple times (for the display)
        assert mock_print.call_count > 0

    def test_handle_choice_selection_valid(self, sample_tree: Any) -> None:
        """Test valid choice selection."""
        debugger = DialogueDebugger(sample_tree, "start")
        result = debugger._handle_choice_selection(1)
        assert result is True
        assert debugger.current_node_id == "left"

    def test_handle_choice_selection_invalid_number(self, sample_tree: Any) -> None:
        """Test choice selection with invalid number."""
        debugger = DialogueDebugger(sample_tree, "start")
        result = debugger._handle_choice_selection(99)
        assert result is False
        assert debugger.current_node_id == "start"

    def test_handle_choice_selection_null_node(self, sample_tree: Any) -> None:
        """Test choice selection from null node."""
        debugger = DialogueDebugger(sample_tree, "treasure")
        result = debugger._handle_choice_selection(1)
        assert result is False

    def test_handle_go_up_success(self, sample_tree: Any) -> None:
        """Test successful navigation to parent."""
        debugger = DialogueDebugger(sample_tree, "left")
        result = debugger._handle_go_up()
        assert result is True
        assert debugger.current_node_id == "start"

    def test_handle_go_up_no_parent(self) -> None:
        """Test going up when no parent exists."""
        tree_data = {
            "nodes": {
                "isolated": {
                    "situation": "You are isolated.",
                    "choices": [{"text": "Stay", "next": None}],
                }
            },
            "params": {},
        }
        tree = DialogueTree.from_dict(tree_data)
        debugger = DialogueDebugger(tree, "isolated")
        result = debugger._handle_go_up()
        assert result is False

    @patch("builtins.input", return_value="right")
    def test_handle_direct_navigation_success(
        self, mock_input: Any, sample_tree: Any
    ) -> None:
        """Test successful direct navigation."""
        debugger = DialogueDebugger(sample_tree, "start")
        result = debugger._handle_direct_navigation()

        assert result is True
        assert debugger.current_node_id == "right"

    @patch("builtins.input", return_value="nonexistent")
    def test_handle_direct_navigation_invalid_node(
        self, mock_input: Any, sample_tree: Any
    ) -> None:
        """Test direct navigation to invalid node."""
        debugger = DialogueDebugger(sample_tree, "start")

        with patch("builtins.input") as mock_continue_input:
            result = debugger._handle_direct_navigation()

        assert result is False
        assert debugger.current_node_id == "start"

    @patch("builtins.input", return_value="")
    def test_handle_direct_navigation_cancel(
        self, mock_input: Any, sample_tree: Any
    ) -> None:
        """Test canceling direct navigation."""
        debugger = DialogueDebugger(sample_tree, "start")
        result = debugger._handle_direct_navigation()

        assert result is False
        assert debugger.current_node_id == "start"


class TestRunDebugger:
    """Test the run_debugger function."""

    @pytest.fixture
    def sample_tree(self) -> Any:
        """Create a sample dialogue tree for testing."""
        tree_data = {
            "nodes": {
                "start": {
                    "situation": "Test situation.",
                    "choices": [{"text": "Test choice", "next": "end"}],
                },
                "end": {"situation": "The end.", "choices": []},
            },
            "params": {"test": 1},
        }
        return DialogueTree.from_dict(tree_data)

    @patch("src.debugger.DialogueDebugger")
    def test_run_debugger_success(
        self, mock_debugger_class: Any, sample_tree: Any
    ) -> None:
        """Test successful debugger run."""
        mock_debugger = MagicMock()
        mock_debugger_class.return_value = mock_debugger

        run_debugger(sample_tree, "start")

        mock_debugger_class.assert_called_once_with(sample_tree, "start")
        mock_debugger.run.assert_called_once()

    @patch("src.debugger.DialogueDebugger")
    @patch("builtins.print")
    def test_run_debugger_exception(
        self, mock_print: Any, mock_debugger_class: Any, sample_tree: Any
    ) -> None:
        """Test debugger run with exception."""
        mock_debugger_class.side_effect = Exception("Test error")

        run_debugger(sample_tree, "start")

        # Check that error was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Fatal error" in call for call in print_calls)
