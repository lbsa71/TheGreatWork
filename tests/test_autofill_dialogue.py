#!/usr/bin/env python3
"""
Tests for the main autofill_dialogue script.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main script modules
from autofill_dialogue import DialogueAutofiller, create_sample_tree, main


class TestDialogueAutofiller:
    """Tests for DialogueAutofiller class."""

    def setup_method(self) -> None:
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

    def test_init(self) -> None:
        """Test initialization."""
        assert self.autofiller.tree_manager == self.mock_tree_manager
        assert self.autofiller.llm_client == self.mock_llm_client
        assert self.autofiller.node_generator == self.mock_node_generator
        assert self.autofiller.nodes_generated == 0
        assert self.autofiller.generation_times == []
        assert self.autofiller.failed_nodes == []

    def test_check_prerequisites_success(self) -> None:
        """Test successful prerequisite check."""
        self.mock_tree_manager.file_path.exists.return_value = True
        self.mock_llm_client.is_available.return_value = True

        result = self.autofiller.check_prerequisites()
        assert result is True

    def test_check_prerequisites_file_not_found(self) -> None:
        """Test prerequisite check when file doesn't exist."""
        self.mock_tree_manager.file_path.exists.return_value = False

        result = self.autofiller.check_prerequisites()
        assert result is False

    def test_check_prerequisites_llm_not_available(self) -> None:
        """Test prerequisite check when LLM is not available."""
        self.mock_tree_manager.file_path.exists.return_value = True
        self.mock_llm_client.is_available.return_value = False

        result = self.autofiller.check_prerequisites()
        assert result is False

    def test_print_statistics_no_generations(self) -> None:
        """Test statistics printing when no nodes were generated."""
        with patch("autofill_dialogue.logger") as mock_logger:
            self.autofiller.print_statistics()
            mock_logger.info.assert_called_with(
                "No generation statistics available (no nodes were processed)"
            )

    def test_print_statistics_with_generations(self) -> None:
        """Test statistics printing with generation times."""
        # Add some mock generation times
        self.autofiller.generation_times = [1.5, 2.0, 1.0, 3.5]
        
        with patch("autofill_dialogue.logger") as mock_logger:
            self.autofiller.print_statistics()
            
            # Check that statistics were logged
            calls = mock_logger.info.call_args_list
            call_strings = [call[0][0] for call in calls]
            
            # Should contain key statistics
            assert any("Total nodes generated: 4" in call for call in call_strings)
            assert any("Mean generation time: 2.00" in call for call in call_strings)
            assert any("Fastest generation: 1.00" in call for call in call_strings)
            assert any("Slowest generation: 3.50" in call for call in call_strings)

    def test_print_statistics_with_failed_nodes(self) -> None:
        """Test statistics printing with failed nodes."""
        # Add some mock generation times and failed nodes
        self.autofiller.generation_times = [1.5, 2.0]
        self.autofiller.failed_nodes = ["node1", "node2"]
        
        with patch("autofill_dialogue.logger") as mock_logger:
            self.autofiller.print_statistics()
            
            # Check that statistics were logged
            calls = mock_logger.info.call_args_list
            call_strings = [call[0][0] for call in calls]
            
            # Should contain key statistics
            assert any("Total nodes generated: 2" in call for call in call_strings)
            assert any("Failed nodes: 2" in call for call in call_strings)
            assert any("Failed node IDs: node1, node2" in call for call in call_strings)

    def test_process_tree_success(self) -> None:
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

        with patch("autofill_dialogue.validate_generated_node", return_value=True), patch(
            "autofill_dialogue.DialogueAutofiller.print_statistics"
        ) as mock_print_stats:
            result = self.autofiller.process_tree()

        assert result is True
        assert self.autofiller.nodes_generated == 1
        assert len(self.autofiller.generation_times) == 1  # Should record generation time
        self.mock_tree_manager.load_tree.assert_called_once()
        self.mock_tree_manager.save_tree.assert_called()
        mock_print_stats.assert_called_once()  # Statistics should be printed

    def test_process_tree_no_null_nodes(self) -> None:
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

    def test_process_tree_node_generation_failure(self) -> None:
        """Test tree processing when node generation fails."""
        mock_tree = Mock()
        mock_tree.find_first_null_node.side_effect = ["node1", None]  # First call finds node1, second finds none
        mock_tree.find_parent_and_choice.return_value = (
            "start",
            {"text": "Choice", "next": "node1"},
        )
        mock_tree.get_node.return_value = {"situation": "Test situation"}
        mock_tree.params = {"test": 1}

        self.mock_tree_manager.load_tree.return_value = mock_tree
        self.mock_tree_manager.create_backup.return_value = Path("backup.json")
        self.mock_node_generator.generate_node.return_value = None  # Generation fails

        with patch("autofill_dialogue.DialogueAutofiller.print_statistics") as mock_print_stats:
            result = self.autofiller.process_tree()

        # Should continue processing and complete successfully
        assert result is True
        assert len(self.autofiller.failed_nodes) == 1
        assert "node1" in self.autofiller.failed_nodes
        assert self.autofiller.nodes_generated == 0  # No successful generations
        mock_print_stats.assert_called_once()

    def test_process_node_success(self) -> None:
        """Test successful node processing."""
        mock_tree = Mock()
        mock_tree.find_parent_and_choice.return_value = (
            "start",
            {"text": "Choice", "next": "node1"},
        )
        mock_tree.get_node.return_value = {"situation": "Test situation"}
        mock_tree.params = {"test": 1}
        mock_tree.nodes = {"start": {}, "node1": None}
        mock_tree.generate_unique_node_id.side_effect = lambda x: x or "fallback_node"

        generated_node = {
            "situation": "Generated situation",
            "choices": [
                {
                    "text": "Choice A",
                    "next": None,
                    "suggested_node_id": "choice_a_outcome",
                },
                {
                    "text": "Choice B",
                    "next": None,
                    "suggested_node_id": "choice_b_outcome",
                },
            ],
        }
        self.mock_node_generator.generate_node.return_value = generated_node

        with patch("autofill_dialogue.validate_generated_node", return_value=True):
            result = self.autofiller._process_node(mock_tree, "node1")

        assert result is True

        # Verify the node was updated with processed choices (suggested_node_id removed)
        expected_node = {
            "situation": "Generated situation",
            "choices": [
                {"text": "Choice A", "next": "choice_a_outcome"},
                {"text": "Choice B", "next": "choice_b_outcome"},
            ],
        }
        mock_tree.update_node.assert_called_once_with("node1", expected_node)

        # Verify new placeholder nodes were created
        assert mock_tree.nodes["choice_a_outcome"] is None
        assert mock_tree.nodes["choice_b_outcome"] is None

    def test_process_node_no_parent(self) -> None:
        """Test node processing when parent is not found."""
        mock_tree = Mock()
        mock_tree.find_parent_and_choice.return_value = None

        result = self.autofiller._process_node(mock_tree, "node1")

        assert result is False

    def test_process_node_invalid_generated_node(self) -> None:
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

    def test_process_node_with_suggested_node_ids(self) -> None:
        """Test node processing with suggested node IDs."""
        mock_tree = Mock()
        mock_tree.find_parent_and_choice.return_value = (
            "start",
            {"text": "Investigate the death", "next": "node1"},
        )
        mock_tree.get_node.return_value = {"situation": "Test situation"}
        mock_tree.params = {"loyalty": 50}
        mock_tree.nodes = {"start": {}, "node1": None}

        # Mock the unique ID generation to return the suggested IDs
        mock_tree.generate_unique_node_id.side_effect = [
            "investigate_entropy",
            "talk_to_mother_again",
        ]

        generated_node = {
            "situation": "You decide to investigate the circumstances...",
            "choices": [
                {
                    "text": "Look for signs of poison",
                    "next": None,
                    "suggested_node_id": "investigate_entropy",
                    "effects": {"wisdom": 5},
                },
                {
                    "text": "Question the queen",
                    "next": None,
                    "suggested_node_id": "talk_to_mother_again",
                    "effects": {"loyalty": -10},
                },
            ],
        }
        self.mock_node_generator.generate_node.return_value = generated_node

        with patch("autofill_dialogue.validate_generated_node", return_value=True):
            result = self.autofiller._process_node(mock_tree, "node1")

        assert result is True

        # Verify suggested_node_id was used for unique ID generation
        mock_tree.generate_unique_node_id.assert_any_call("investigate_entropy")
        mock_tree.generate_unique_node_id.assert_any_call("talk_to_mother_again")

        # Verify the final node has descriptive next IDs and no suggested_node_id
        expected_node = {
            "situation": "You decide to investigate the circumstances...",
            "choices": [
                {
                    "text": "Look for signs of poison",
                    "next": "investigate_entropy",
                    "effects": {"wisdom": 5},
                },
                {
                    "text": "Question the queen",
                    "next": "talk_to_mother_again",
                    "effects": {"loyalty": -10},
                },
            ],
        }
        mock_tree.update_node.assert_called_once_with("node1", expected_node)

    def test_process_node_fallback_when_no_suggested_id(self) -> None:
        """Test node processing falls back to generated ID when no suggestion provided."""
        mock_tree = Mock()
        mock_tree.find_parent_and_choice.return_value = (
            "start",
            {"text": "Choice", "next": "node1"},
        )
        mock_tree.get_node.return_value = {"situation": "Test situation"}
        mock_tree.params = {"test": 1}
        mock_tree.nodes = {"start": {}, "node1": None}

        # Mock fallback ID generation
        mock_tree.generate_unique_node_id.return_value = "node_2"

        generated_node = {
            "situation": "Generated situation",
            "choices": [
                {"text": "Choice A", "next": None},  # No suggested_node_id
            ],
        }
        self.mock_node_generator.generate_node.return_value = generated_node

        with patch("autofill_dialogue.validate_generated_node", return_value=True):
            result = self.autofiller._process_node(mock_tree, "node1")

        assert result is True

        # Verify fallback was used (empty string passed to generate_unique_node_id)
        mock_tree.generate_unique_node_id.assert_called_once_with("")

        # Verify the node has the fallback ID
        expected_node = {
            "situation": "Generated situation",
            "choices": [
                {"text": "Choice A", "next": "node_2"},
            ],
        }
        mock_tree.update_node.assert_called_once_with("node1", expected_node)


class TestCreateSampleTree:
    """Tests for create_sample_tree function."""

    def test_create_sample_tree(self) -> None:
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

    def test_main_create_sample(self) -> None:
        """Test main function with --create-sample flag."""
        test_args = ["autofill_dialogue.py", "--create-sample", "sample.json"]

        with patch("sys.argv", test_args), patch(
            "autofill_dialogue.create_sample_tree"
        ) as mock_create:

            result = main()

            assert result == 0
            mock_create.assert_called_once_with(Path("sample.json"))

    def test_main_success(self) -> None:
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

    def test_main_prerequisites_failed(self) -> None:
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

    def test_main_processing_failed(self) -> None:
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

    def test_main_default_arguments(self) -> None:
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
            mock_class.assert_called_once_with(Path("tree.json"), "qwen3:14b", None)

    def test_main_verbose_flag(self) -> None:
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

    def test_main_custom_model(self) -> None:
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
    def test_main_debug_mode(
        self, mock_run_debugger: Any, mock_manager_class: Any
    ) -> None:
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
        self, mock_run_debugger: Any, mock_manager_class: Any
    ) -> None:
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
