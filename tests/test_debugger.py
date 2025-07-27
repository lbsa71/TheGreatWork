#!/usr/bin/env python3
"""
Tests for the dialogue tree debugger.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dialogue_tree import DialogueTree
from debugger import DialogueDebugger, KeyboardInput, run_debugger


class TestKeyboardInput:
    """Test the KeyboardInput class."""
    
    @patch('debugger.WINDOWS', True)
    def test_context_manager(self):
        """Test that KeyboardInput can be used as a context manager."""
        # Test with Windows mode to avoid termios issues
        with KeyboardInput() as kb:
            assert kb is not None
    
    @patch('debugger.WINDOWS', False)
    @patch('debugger.termios')
    @patch('debugger.tty')
    def test_unix_setup(self, mock_tty, mock_termios):
        """Test Unix/Linux setup."""
        mock_termios.tcgetattr.return_value = "old_settings"
        
        with KeyboardInput() as kb:
            mock_termios.tcgetattr.assert_called_once()
            mock_tty.setraw.assert_called_once()
        
        mock_termios.tcsetattr.assert_called_once()
    
    @patch('debugger.WINDOWS', True)
    @patch('debugger.msvcrt')
    def test_windows_get_char(self, mock_msvcrt):
        """Test Windows character input."""
        mock_msvcrt.getch.return_value = b'a'
        
        with KeyboardInput() as kb:
            char = kb.get_char()
            assert char == 'a'


class TestDialogueDebugger:
    """Test the DialogueDebugger class."""
    
    @pytest.fixture
    def sample_tree(self):
        """Create a sample dialogue tree for testing."""
        tree_data = {
            "nodes": {
                "start": {
                    "situation": "You are at the beginning.",
                    "choices": [
                        {"text": "Go left", "next": "left", "effects": {"courage": 5}},
                        {"text": "Go right", "next": "right", "effects": {"wisdom": 10}},
                    ]
                },
                "left": {
                    "situation": "You went left and found a treasure.",
                    "choices": [
                        {"text": "Take treasure", "next": "treasure", "effects": {"wealth": 20}},
                        {"text": "Go back", "next": "start", "effects": {}},
                    ]
                },
                "right": {
                    "situation": "You went right and met a wise sage.",
                    "choices": [
                        {"text": "Ask for advice", "next": "advice", "effects": {"wisdom": 15}},
                        {"text": "Go back", "next": "start", "effects": {}},
                    ]
                },
                "treasure": None,
                "advice": None,
            },
            "params": {"courage": 50, "wisdom": 30, "wealth": 10}
        }
        return DialogueTree.from_dict(tree_data)
    
    def test_init_with_start_node(self, sample_tree):
        """Test debugger initialization with explicit start node."""
        debugger = DialogueDebugger(sample_tree, "left")
        assert debugger.current_node_id == "left"
        assert debugger.tree == sample_tree
        assert debugger.running is True
    
    def test_init_auto_find_root(self, sample_tree):
        """Test debugger initialization with auto-detected root."""
        debugger = DialogueDebugger(sample_tree)
        assert debugger.current_node_id == "start"
    
    def test_init_no_valid_start_node(self):
        """Test debugger initialization with no valid start node."""
        tree_data = {"nodes": {"null_node": None}, "params": {}}
        tree = DialogueTree.from_dict(tree_data)
        
        with pytest.raises(ValueError, match="No valid starting node found"):
            DialogueDebugger(tree)
    
    def test_find_root_node_with_start(self, sample_tree):
        """Test finding root node when 'start' exists."""
        debugger = DialogueDebugger(sample_tree)
        root = debugger._find_root_node()
        assert root == "start"
    
    def test_find_root_node_without_start(self):
        """Test finding root node when 'start' doesn't exist."""
        tree_data = {
            "nodes": {
                "root": {
                    "situation": "This is the root.",
                    "choices": [{"text": "Next", "next": "child"}]
                },
                "child": {
                    "situation": "This is a child.",
                    "choices": []
                }
            },
            "params": {}
        }
        tree = DialogueTree.from_dict(tree_data)
        debugger = DialogueDebugger(tree)
        assert debugger.current_node_id == "root"
    
    @patch('builtins.print')
    @patch('sys.stdout.isatty', return_value=False)
    @patch.object(DialogueTree, 'build_dialogue_history', return_value="Test history")
    def test_display_node_valid(self, mock_history, mock_isatty, mock_print, sample_tree):
        """Test displaying a valid node."""
        debugger = DialogueDebugger(sample_tree, "start")
        debugger._display_node()
        
        # Check that print was called (screen content was displayed)
        assert mock_print.called
        
        # Check that some expected content was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        print_output = " ".join(print_calls)
        assert "CURRENT NODE: start" in print_output
        assert "You are at the beginning" in print_output
        assert "Go left" in print_output
    
    @patch('builtins.print')
    @patch('sys.stdout.isatty', return_value=False)
    @patch.object(DialogueTree, 'build_dialogue_history', return_value="Test history")
    def test_display_node_null(self, mock_history, mock_isatty, mock_print, sample_tree):
        """Test displaying a null node."""
        debugger = DialogueDebugger(sample_tree, "treasure")
        debugger._display_node()
        
        print_calls = [str(call) for call in mock_print.call_args_list]
        print_output = " ".join(print_calls)
        assert "This node is NULL" in print_output
    
    def test_handle_choice_selection_valid(self, sample_tree):
        """Test handling valid choice selection."""
        debugger = DialogueDebugger(sample_tree, "start")
        result = debugger._handle_choice_selection(1)
        
        assert result is True
        assert debugger.current_node_id == "left"
    
    def test_handle_choice_selection_invalid_number(self, sample_tree):
        """Test handling invalid choice number."""
        debugger = DialogueDebugger(sample_tree, "start")
        
        with patch('builtins.input'), patch('builtins.print'):
            result = debugger._handle_choice_selection(99)
        
        assert result is False
        assert debugger.current_node_id == "start"  # Should not change
    
    def test_handle_choice_selection_null_node(self, sample_tree):
        """Test handling choice selection from null node."""
        debugger = DialogueDebugger(sample_tree, "treasure")
        
        with patch('builtins.input'), patch('builtins.print'):
            result = debugger._handle_choice_selection(1)
        
        assert result is False
        assert debugger.current_node_id == "treasure"  # Should not change
    
    def test_handle_go_up_success(self, sample_tree):
        """Test successfully going up to parent node."""
        debugger = DialogueDebugger(sample_tree, "left")
        result = debugger._handle_go_up()
        
        assert result is True
        assert debugger.current_node_id == "start"
    
    def test_handle_go_up_no_parent(self):
        """Test going up when no parent exists."""
        # Create a tree with a truly isolated node
        tree_data = {
            "nodes": {
                "isolated": {
                    "situation": "This node has no parent.",
                    "choices": []
                }
            },
            "params": {}
        }
        tree = DialogueTree.from_dict(tree_data)
        debugger = DialogueDebugger(tree, "isolated")
        
        with patch('builtins.input'), patch('builtins.print'):
            result = debugger._handle_go_up()
        
        assert result is False
        assert debugger.current_node_id == "isolated"
    
    @patch('debugger.WINDOWS', False)
    @patch('debugger.termios', create=True)
    @patch('debugger.tty', create=True)
    @patch('builtins.input', return_value="right")
    def test_handle_direct_navigation_success(self, mock_input, mock_tty, mock_termios, mock_windows, sample_tree):
        """Test successful direct navigation."""
        debugger = DialogueDebugger(sample_tree, "start")
        result = debugger._handle_direct_navigation()
        
        assert result is True
        assert debugger.current_node_id == "right"
    
    @patch('debugger.WINDOWS', False)
    @patch('debugger.termios', create=True)
    @patch('debugger.tty', create=True)
    @patch('builtins.input', return_value="nonexistent")
    def test_handle_direct_navigation_invalid_node(self, mock_input, mock_tty, mock_termios, mock_windows, sample_tree):
        """Test direct navigation to invalid node."""
        debugger = DialogueDebugger(sample_tree, "start")
        
        with patch('builtins.input') as mock_continue_input:
            result = debugger._handle_direct_navigation()
        
        assert result is False
        assert debugger.current_node_id == "start"
    
    @patch('debugger.WINDOWS', False)
    @patch('debugger.termios', create=True)
    @patch('debugger.tty', create=True)
    @patch('builtins.input', return_value="")
    def test_handle_direct_navigation_cancel(self, mock_input, mock_tty, mock_termios, mock_windows, sample_tree):
        """Test canceling direct navigation."""
        debugger = DialogueDebugger(sample_tree, "start")
        result = debugger._handle_direct_navigation()
        
        assert result is False
        assert debugger.current_node_id == "start"


class TestRunDebugger:
    """Test the run_debugger function."""
    
    @pytest.fixture
    def sample_tree(self):
        """Create a sample dialogue tree for testing."""
        tree_data = {
            "nodes": {
                "start": {
                    "situation": "Test situation.",
                    "choices": [{"text": "Test choice", "next": "end"}]
                },
                "end": {"situation": "The end.", "choices": []}
            },
            "params": {"test": 1}
        }
        return DialogueTree.from_dict(tree_data)
    
    @patch('debugger.DialogueDebugger')
    def test_run_debugger_success(self, mock_debugger_class, sample_tree):
        """Test successful debugger run."""
        mock_debugger = MagicMock()
        mock_debugger_class.return_value = mock_debugger
        
        run_debugger(sample_tree, "start")
        
        mock_debugger_class.assert_called_once_with(sample_tree, "start")
        mock_debugger.run.assert_called_once()
    
    @patch('debugger.DialogueDebugger')
    @patch('builtins.print')
    def test_run_debugger_exception(self, mock_print, mock_debugger_class, sample_tree):
        """Test debugger run with exception."""
        mock_debugger_class.side_effect = Exception("Test error")
        
        run_debugger(sample_tree, "start")
        
        # Check that error was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Failed to start debugger" in call for call in print_calls)