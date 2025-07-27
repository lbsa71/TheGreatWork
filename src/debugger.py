#!/usr/bin/env python3
"""
Interactive dialogue tree debugger.

This module provides an interactive GUI-like experience for navigating
and debugging dialogue trees.
"""

import logging
import sys
from typing import Optional, Dict, Any, List

# Import platform-specific modules for keyboard input
try:
    import msvcrt  # Windows
    WINDOWS = True
except ImportError:
    import tty
    import termios
    WINDOWS = False

from dialogue_tree import DialogueTree

logger = logging.getLogger(__name__)


class KeyboardInput:
    """Cross-platform keyboard input handler."""
    
    def __init__(self):
        self.old_settings = None
    
    def __enter__(self):
        if not WINDOWS:
            # Unix/Linux/macOS
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        return self
    
    def __exit__(self, type, value, traceback):
        if not WINDOWS and self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_char(self) -> str:
        """Get a single character from stdin without requiring Enter."""
        if WINDOWS:
            return msvcrt.getch().decode('utf-8')
        else:
            return sys.stdin.read(1)


class DialogueDebugger:
    """Interactive dialogue tree debugger."""
    
    def __init__(self, tree: DialogueTree, start_node_id: Optional[str] = None):
        """
        Initialize the debugger.
        
        Args:
            tree: The dialogue tree to debug
            start_node_id: Node ID to start from, or None for root node
        """
        self.tree = tree
        self.current_node_id = start_node_id or self._find_root_node()
        self.running = True
        
        if self.current_node_id is None:
            raise ValueError("No valid starting node found")
    
    def _find_root_node(self) -> Optional[str]:
        """Find the root node (node with no parents)."""
        # Look for 'start' node first
        if 'start' in self.tree.nodes and self.tree.nodes['start'] is not None:
            return 'start'
        
        # Find any node that has no parent
        all_children = set()
        for node_id, node_data in self.tree.nodes.items():
            if node_data is not None and isinstance(node_data, dict):
                choices = node_data.get('choices', [])
                for choice in choices:
                    if choice.get('next'):
                        all_children.add(choice['next'])
        
        # Find nodes that are not children of any other node
        root_candidates = []
        for node_id in self.tree.nodes.keys():
            if node_id not in all_children and self.tree.nodes[node_id] is not None:
                root_candidates.append(node_id)
        
        return root_candidates[0] if root_candidates else None
    
    def _clear_screen(self):
        """Clear the terminal screen."""
        # Only clear screen in interactive mode, not during tests
        try:
            if sys.stdout.isatty():
                if WINDOWS:
                    import os
                    os.system('cls')
                else:
                    print('\033[2J\033[H', end='')
        except:
            # If we can't clear screen, just continue
            pass
    
    def _display_node(self):
        """Display the current node information."""
        self._clear_screen()
        
        print("=" * 80)
        print("DIALOGUE TREE DEBUGGER")
        print("=" * 80)
        print()
        
        # Display dialogue history
        history = self.tree.build_dialogue_history(self.current_node_id)
        if history and "No previous dialogue history" not in history:
            print("HISTORY:")
            print("-" * 40)
            print(history)
            print()
        
        # Display current node
        current_node = self.tree.get_node(self.current_node_id)
        print(f"CURRENT NODE: {self.current_node_id}")
        print("-" * 40)
        
        if current_node is None:
            print("❌ This node is NULL (not yet generated)")
            print()
        else:
            situation = current_node.get('situation', 'No situation text')
            print(f"📖 {situation}")
            print()
            
            # Display choices
            choices = current_node.get('choices', [])
            if choices:
                print("AVAILABLE CHOICES:")
                print("-" * 40)
                for i, choice in enumerate(choices, 1):
                    choice_text = choice.get('text', 'No text')
                    next_node = choice.get('next', 'None')
                    effects = choice.get('effects', {})
                    
                    print(f"{i}. {choice_text}")
                    print(f"   → Next: {next_node}")
                    if effects:
                        effects_str = ", ".join([f"{k}:{v}" for k, v in effects.items()])
                        print(f"   ⚡ Effects: {effects_str}")
                    print()
            else:
                print("❌ No choices available")
                print()
        
        # Display navigation help
        print("NAVIGATION:")
        print("-" * 40)
        print("1-9     : Choose option by number")
        print("u       : Go up (to parent node)")  
        print("q       : Quit debugger")
        print("Enter   : Enter node ID directly")
        print()
        print("Current game parameters:")
        params_str = ", ".join([f"{k}:{v}" for k, v in self.tree.params.items()])
        print(f"  {params_str}")
        print()
        print("Press a key to navigate...")
    
    def _handle_choice_selection(self, choice_num: int) -> bool:
        """Handle selection of a choice by number."""
        current_node = self.tree.get_node(self.current_node_id)
        if current_node is None:
            print("❌ Cannot navigate from null node!")
            input("Press Enter to continue...")
            return False
        
        choices = current_node.get('choices', [])
        if 1 <= choice_num <= len(choices):
            next_node_id = choices[choice_num - 1].get('next')
            if next_node_id:
                self.current_node_id = next_node_id
                return True
            else:
                print("❌ Choice has no next node!")
                input("Press Enter to continue...")
                return False
        else:
            print(f"❌ Invalid choice! Choose 1-{len(choices)}")
            input("Press Enter to continue...")
            return False
    
    def _handle_go_up(self) -> bool:
        """Handle going up to parent node."""
        parent_info = self.tree.find_parent_and_choice(self.current_node_id)
        if parent_info:
            parent_id, _ = parent_info
            self.current_node_id = parent_id
            return True
        else:
            print("❌ No parent node found!")
            input("Press Enter to continue...")
            return False
    
    def _handle_direct_navigation(self) -> bool:
        """Handle direct navigation to a node by ID."""
        print("\nEnter node ID (or press Enter to cancel): ", end='', flush=True)
        
        # Restore normal input mode temporarily for non-Windows systems
        if not WINDOWS:
            try:
                # Get current settings
                old_settings = termios.tcgetattr(sys.stdin)
                # Restore normal mode temporarily
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                # If we can't restore settings, continue anyway
                pass
        
        try:
            node_id = input().strip()
            if not node_id:
                return False
            
            if node_id in self.tree.nodes:
                self.current_node_id = node_id
                return True
            else:
                print(f"❌ Node '{node_id}' not found!")
                input("Press Enter to continue...")
                return False
        except (KeyboardInterrupt, EOFError):
            return False
        finally:
            # Re-enable raw mode for continued operation
            if not WINDOWS:
                try:
                    tty.setraw(sys.stdin.fileno())
                except:
                    pass
    
    def run(self):
        """Run the interactive debugger."""
        print("Starting dialogue tree debugger...")
        print("Loading tree...")
        
        try:
            with KeyboardInput() as kb:
                while self.running:
                    self._display_node()
                    
                    try:
                        char = kb.get_char()
                        
                        # Handle different input types
                        if char.lower() == 'q':
                            self.running = False
                            break
                        elif char.lower() == 'u':
                            self._handle_go_up()
                        elif char == '\r' or char == '\n':  # Enter key
                            self._handle_direct_navigation()
                        elif char.isdigit():
                            choice_num = int(char)
                            self._handle_choice_selection(choice_num)
                        elif char == '\x03':  # Ctrl+C
                            raise KeyboardInterrupt
                        # Ignore other characters
                        
                    except KeyboardInterrupt:
                        print("\n\nExiting debugger...")
                        break
                    except Exception as e:
                        logger.error(f"Error in debugger: {e}")
                        print(f"❌ Error: {e}")
                        input("Press Enter to continue...")
        
        except Exception as e:
            print(f"❌ Fatal error in debugger: {e}")
            logger.error(f"Fatal debugger error: {e}")
        
        finally:
            print("\nDebugger session ended.")


def run_debugger(tree: DialogueTree, start_node_id: Optional[str] = None):
    """
    Run the interactive dialogue debugger.
    
    Args:
        tree: The dialogue tree to debug
        start_node_id: Node ID to start from, or None for root node
    """
    try:
        debugger = DialogueDebugger(tree, start_node_id)
        debugger.run()
    except Exception as e:
        print(f"❌ Failed to start debugger: {e}")
        logger.error(f"Failed to start debugger: {e}")