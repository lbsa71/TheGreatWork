#!/usr/bin/env python3
"""
Tests for the main autofill_dialogue script.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main script modules
from autofill_dialogue import DialogueAutofiller, create_sample_tree, main


class TestDialogueAutofiller:
    """Tests for DialogueAutofiller class."""

    def setup_method(self):
        """Set up test objects."""
        self.tree_file = Path("test_tree.json")

        # Mock the dependencies
        self.mock_tree_manager = Mock()
        self.mock_llm_client = Mock()
        self.mock_node_generator = Mock()

        with patch("autofill_dialogue.DialogueTreeManager") as mock_dtm, patch(
            "autofill_dialogue.OllamaClient"
        ) as mock_oc, patch("autofill_dialogue.NodeGenerator") as mock_ng:

            mock_dtm.return_value = self.mock_tree_manager
            mock_oc.return_value = self.mock_llm_client
            mock_ng.return_value = self.mock_node_generator

            self.autofiller = DialogueAutofiller(self.tree_file, "test_model")

    def test_init(self):
        """Test initialization."""
        assert self.autofiller.tree_manager == self.mock_tree_manager
        assert self.autofiller.llm_client == self.mock_llm_client
        assert self.autofiller.node_generator == self.mock_node_generator
        assert self.autofiller.nodes_generated == 0

    def test_check_prerequisites_success(self):
        """Test successful prerequisite check."""
        self.mock_tree_manager.file_path.exists.return_value = True
        self.mock_llm_client.is_available.return_value = True

        result = self.autofiller.check_prerequisites()
        assert result is True

    def test_check_prerequisites_file_not_found(self):
        """Test prerequisite check when file doesn't exist."""
        self.mock_tree_manager.file_path.exists.return_value = False

        result = self.autofiller.check_prerequisites()
        assert result is False

    def test_check_prerequisites_llm_not_available(self):
        """Test prerequisite check when LLM is not available."""
        self.mock_tree_manager.file_path.exists.return_value = True
        self.mock_llm_client.is_available.return_value = False

        result = self.autofiller.check_prerequisites()
        assert result is False

    def test_process_tree_success(self):
        """Test successful tree processing."""
        # Mock tree with one null node
        mock_tree = Mock()
        mock_tree.nodes = {"start": {}, "node1": None}
        mock_tree.find_first_null_node.side_effect = [
            "node1",
            None,
        ]  # First call finds node1, second finds none
        mock_tree.find_parent_and_choice.return_value = (
            "start",
            {"text": "Choice", "next": "node1"},
        )
        mock_tree.get_node.return_value = {"situation": "Test situation"}
        mock_tree.params = {"test": 1}

        self.mock_tree_manager.load_tree.return_value = mock_tree
        self.mock_tree_manager.create_backup.return_value = Path("backup.json")

        # Mock node generation
        generated_node = {
            "situation": "Generated situation",
            "choices": [
                {"text": "Choice A", "next": None},
                {"text": "Choice B", "next": None},
            ],
        }
        self.mock_node_generator.generate_node.return_value = generated_node

        with patch("autofill_dialogue.validate_generated_node", return_value=True):
            result = self.autofiller.process_tree()

        assert result is True
        assert self.autofiller.nodes_generated == 1
        self.mock_tree_manager.load_tree.assert_called_once()
        self.mock_tree_manager.save_tree.assert_called()

    def test_process_tree_no_null_nodes(self):
        """Test tree processing when no null nodes exist."""
        mock_tree = Mock()
        mock_tree.find_first_null_node.return_value = None
        mock_tree.nodes = {
            "start": {"situation": "test"}
        }  # Give it a proper nodes dict

        self.mock_tree_manager.load_tree.return_value = mock_tree
        self.mock_tree_manager.create_backup.return_value = Path("backup.json")

        result = self.autofiller.process_tree()

        assert result is True
        assert self.autofiller.nodes_generated == 0

    def test_process_tree_node_generation_failure(self):
        """Test tree processing when node generation fails."""
        mock_tree = Mock()
        mock_tree.find_first_null_node.return_value = "node1"
        mock_tree.find_parent_and_choice.return_value = (
            "start",
            {"text": "Choice", "next": "node1"},
        )
        mock_tree.get_node.return_value = {"situation": "Test situation"}
        mock_tree.params = {"test": 1}

        self.mock_tree_manager.load_tree.return_value = mock_tree
        self.mock_tree_manager.create_backup.return_value = Path("backup.json")
        self.mock_node_generator.generate_node.return_value = None  # Generation fails

        result = self.autofiller.process_tree()

        assert result is False

    def test_process_node_success(self):
        """Test successful node processing."""
        mock_tree = Mock()
        mock_tree.find_parent_and_choice.return_value = (
            "start",
            {"text": "Choice", "next": "node1"},
        )
        mock_tree.get_node.return_value = {"situation": "Test situation"}
        mock_tree.params = {"test": 1}
        mock_tree.nodes = {"start": {}, "node1": None}

        generated_node = {
            "situation": "Generated situation",
            "choices": [
                {"text": "Choice A", "next": None},
                {"text": "Choice B", "next": None},
            ],
        }
        self.mock_node_generator.generate_node.return_value = generated_node

        with patch("autofill_dialogue.validate_generated_node", return_value=True):
            result = self.autofiller._process_node(mock_tree, "node1")

        assert result is True
        mock_tree.update_node.assert_called_once_with("node1", generated_node)

    def test_process_node_no_parent(self):
        """Test node processing when parent is not found."""
        mock_tree = Mock()
        mock_tree.find_parent_and_choice.return_value = None

        result = self.autofiller._process_node(mock_tree, "node1")

        assert result is False

    def test_process_node_invalid_generated_node(self):
        """Test node processing when generated node is invalid."""
        mock_tree = Mock()
        mock_tree.find_parent_and_choice.return_value = (
            "start",
            {"text": "Choice", "next": "node1"},
        )
        mock_tree.get_node.return_value = {"situation": "Test situation"}
        mock_tree.params = {"test": 1}

        generated_node = {"invalid": "node"}
        self.mock_node_generator.generate_node.return_value = generated_node

        with patch("autofill_dialogue.validate_generated_node", return_value=False):
            result = self.autofiller._process_node(mock_tree, "node1")

        assert result is False


class TestCreateSampleTree:
    """Tests for create_sample_tree function."""

    def test_create_sample_tree(self):
        """Test sample tree creation."""
        mock_file = mock_open()

        with patch("builtins.open", mock_file):
            create_sample_tree(Path("sample.json"))

        mock_file.assert_called_once_with(Path("sample.json"), "w", encoding="utf-8")

        # Check that valid JSON was written
        handle = mock_file()
        written_data = "".join(call.args[0] for call in handle.write.call_args_list)
        data = json.loads(written_data)

        assert "nodes" in data
        assert "params" in data
        assert "start" in data["nodes"]
        assert data["nodes"]["node1"] is None
        assert data["nodes"]["node2"] is None


class TestMain:
    """Tests for main function."""

    def test_main_create_sample(self):
        """Test main function with --create-sample flag."""
        test_args = ["autofill_dialogue.py", "--create-sample", "sample.json"]

        with patch("sys.argv", test_args), patch(
            "autofill_dialogue.create_sample_tree"
        ) as mock_create:

            result = main()

            assert result == 0
            mock_create.assert_called_once_with(Path("sample.json"))

    def test_main_success(self):
        """Test successful main execution."""
        test_args = ["autofill_dialogue.py", "tree.json"]

        mock_autofiller = Mock()
        mock_autofiller.check_prerequisites.return_value = True
        mock_autofiller.process_tree.return_value = True

        with patch("sys.argv", test_args), patch(
            "autofill_dialogue.DialogueAutofiller", return_value=mock_autofiller
        ):

            result = main()

            assert result == 0
            mock_autofiller.check_prerequisites.assert_called_once()
            mock_autofiller.process_tree.assert_called_once()

    def test_main_prerequisites_failed(self):
        """Test main execution when prerequisites fail."""
        test_args = ["autofill_dialogue.py", "tree.json"]

        mock_autofiller = Mock()
        mock_autofiller.check_prerequisites.return_value = False

        with patch("sys.argv", test_args), patch(
            "autofill_dialogue.DialogueAutofiller", return_value=mock_autofiller
        ):

            result = main()

            assert result == 1
            mock_autofiller.check_prerequisites.assert_called_once()
            mock_autofiller.process_tree.assert_not_called()

    def test_main_processing_failed(self):
        """Test main execution when processing fails."""
        test_args = ["autofill_dialogue.py", "tree.json"]

        mock_autofiller = Mock()
        mock_autofiller.check_prerequisites.return_value = True
        mock_autofiller.process_tree.return_value = False

        with patch("sys.argv", test_args), patch(
            "autofill_dialogue.DialogueAutofiller", return_value=mock_autofiller
        ):

            result = main()

            assert result == 1
            mock_autofiller.check_prerequisites.assert_called_once()
            mock_autofiller.process_tree.assert_called_once()

    def test_main_default_arguments(self):
        """Test main function with default arguments."""
        test_args = ["autofill_dialogue.py"]

        mock_autofiller = Mock()
        mock_autofiller.check_prerequisites.return_value = True
        mock_autofiller.process_tree.return_value = True

        with patch("sys.argv", test_args), patch(
            "autofill_dialogue.DialogueAutofiller"
        ) as mock_class:

            mock_class.return_value = mock_autofiller
            result = main()

            assert result == 0
            # Check that default arguments were used
            mock_class.assert_called_once_with(Path("tree.json"), "llama3", None)

    def test_main_verbose_flag(self):
        """Test main function with verbose flag."""
        test_args = ["autofill_dialogue.py", "--verbose", "tree.json"]

        mock_autofiller = Mock()
        mock_autofiller.check_prerequisites.return_value = True
        mock_autofiller.process_tree.return_value = True

        with patch("sys.argv", test_args), patch(
            "autofill_dialogue.DialogueAutofiller", return_value=mock_autofiller
        ), patch("autofill_dialogue.setup_logging") as mock_setup:

            result = main()

            assert result == 0
            mock_setup.assert_called_once_with(True)

    def test_main_custom_model(self):
        """Test main function with custom model."""
        test_args = ["autofill_dialogue.py", "--model", "mistral", "tree.json"]

        mock_autofiller = Mock()
        mock_autofiller.check_prerequisites.return_value = True
        mock_autofiller.process_tree.return_value = True

        with patch("sys.argv", test_args), patch(
            "autofill_dialogue.DialogueAutofiller"
        ) as mock_class:

            mock_class.return_value = mock_autofiller
            result = main()

            assert result == 0
            mock_class.assert_called_once_with(Path("tree.json"), "mistral", None)

    @patch("autofill_dialogue.DialogueTreeManager")
    @patch("autofill_dialogue.run_debugger")
    def test_main_debug_mode(self, mock_run_debugger, mock_manager_class):
        """Test main function with debug mode."""
        mock_manager = MagicMock()
        mock_tree = MagicMock()
        mock_manager.load_tree.return_value = mock_tree
        mock_manager_class.return_value = mock_manager

        test_args = ["autofill_dialogue.py", "tree.json", "--debug"]

        # Mock file exists
        with patch("sys.argv", test_args), patch(
            "pathlib.Path.exists", return_value=True
        ):
            result = main()

        assert result == 0
        mock_run_debugger.assert_called_once_with(mock_tree, None)

    @patch("autofill_dialogue.DialogueTreeManager")
    @patch("autofill_dialogue.run_debugger")
    def test_main_debug_mode_with_start_node(
        self, mock_run_debugger, mock_manager_class
    ):
        """Test main function with debug mode and start node."""
        mock_manager = MagicMock()
        mock_tree = MagicMock()
        mock_manager.load_tree.return_value = mock_tree
        mock_manager_class.return_value = mock_manager

        test_args = [
            "autofill_dialogue.py",
            "tree.json",
            "--debug",
            "--start-node",
            "node1",
        ]

        # Mock file exists
        with patch("sys.argv", test_args), patch(
            "pathlib.Path.exists", return_value=True
        ):
            result = main()

        assert result == 0
        mock_run_debugger.assert_called_once_with(mock_tree, "node1")
