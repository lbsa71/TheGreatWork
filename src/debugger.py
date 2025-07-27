#!/usr/bin/env python3
"""
Interactive dialogue tree debugger.

This module provides an interactive console experience for navigating
and debugging dialogue trees.
"""

import logging
import sys
from typing import Optional, Dict, Any, List

from dialogue_tree import DialogueTree

logger = logging.getLogger(__name__)


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
        if "start" in self.tree.nodes and self.tree.nodes["start"] is not None:
            return "start"

        # Find any node that has no parent
        all_children = set()
        for node_id, node_data in self.tree.nodes.items():
            if node_data is not None and isinstance(node_data, dict):
                choices = node_data.get("choices", [])
                for choice in choices:
                    if choice.get("next"):
                        all_children.add(choice["next"])

        # Find nodes that are not children of any other node
        root_candidates = []
        for node_id in self.tree.nodes.keys():
            if node_id not in all_children and self.tree.nodes[node_id] is not None:
                root_candidates.append(node_id)

        return root_candidates[0] if root_candidates else None

    def _display_node(self) -> None:
        """Display the current node information."""
        print("\n" + "=" * 80)
        print("DIALOGUE TREE DEBUGGER")
        print("=" * 80)
        print()

        # Display dialogue history
        if self.current_node_id is not None:
            history = self.tree.build_dialogue_history(self.current_node_id)
            if history and "No previous dialogue history" not in history:
                print("HISTORY:")
                print("-" * 40)
                print(history)
                print()

        # Display current node
        if self.current_node_id is not None:
            current_node = self.tree.get_node(self.current_node_id)
        else:
            current_node = None
        print(f"CURRENT NODE: {self.current_node_id}")
        print("-" * 40)

        if current_node is None:
            print("❌ This node is NULL (not yet generated)")
            print()
        else:
            print(f"📖 {current_node['situation']}")
            print()
            print("AVAILABLE CHOICES:")
            print("-" * 40)
            for i, choice in enumerate(current_node.get("choices", []), 1):
                next_node = choice.get("next", "None")
                effects = choice.get("effects", {})
                effects_str = f" (Effects: {effects})" if effects else ""
                print(f"{i}. {choice['text']}")
                print(f"   → Next: {next_node}{effects_str}")
            print()

        # Display navigation options
        print("NAVIGATION:")
        print("-" * 40)
        print("1-9     : Choose option by number")
        print("u       : Go up (to parent node)")
        print("q       : Quit debugger")
        print("Enter   : Enter node ID directly")
        print()

        # Display current game parameters
        if self.tree.params:
            print("Current game parameters:")
            for key, value in self.tree.params.items():
                print(f"  {key}:{value}")
            print()

    def _handle_choice_selection(self, choice_num: int) -> bool:
        """Handle choice selection by number."""
        if self.current_node_id is None:
            print("❌ Cannot select choice: current node is NULL")
            return False

        current_node = self.tree.get_node(self.current_node_id)
        if current_node is None:
            print("❌ Cannot select choice: current node is NULL")
            return False

        choices = current_node.get("choices", [])
        if choice_num < 1 or choice_num > len(choices):
            print(f"❌ Invalid choice number: {choice_num}")
            return False

        choice = choices[choice_num - 1]
        next_node_id = choice.get("next")
        
        if next_node_id is None:
            print("❌ This choice leads to a NULL node (not yet generated)")
            return False

        if next_node_id not in self.tree.nodes:
            print(f"❌ Invalid next node: {next_node_id}")
            return False

        # Apply effects if any
        effects = choice.get("effects", {})
        if effects:
            for param, change in effects.items():
                if param in self.tree.params:
                    self.tree.params[param] += change
                    print(f"📊 {param} changed by {change}")

        self.current_node_id = next_node_id
        return True

    def _handle_go_up(self) -> bool:
        """Handle going up to parent node."""
        if self.current_node_id is None:
            print("❌ Cannot go up: current node is NULL")
            return False

        result = self.tree.find_parent_and_choice(self.current_node_id)
        if result is None:
            print("❌ No parent node found")
            return False

        parent_id, choice = result
        self.current_node_id = parent_id
        print(f"⬆️  Moved up to parent node: {parent_id}")
        return True

    def _handle_direct_navigation(self) -> bool:
        """Handle direct navigation to a specific node."""
        try:
            node_id = input("Enter node ID (or press Enter to cancel): ").strip()
            if not node_id:
                return False

            if node_id not in self.tree.nodes:
                print(f"❌ Node '{node_id}' not found")
                input("Press Enter to continue...")
                return False

            self.current_node_id = node_id
            print(f"🎯 Navigated to node: {node_id}")
            return True
        except (EOFError, KeyboardInterrupt):
            return False

    def run(self) -> None:
        """Run the interactive debugger."""
        try:
            while self.running:
                self._display_node()

                try:
                    user_input = input("Enter command: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    continue

                if user_input == "q":
                    print("👋 Goodbye!")
                    break
                elif user_input == "u":
                    self._handle_go_up()
                elif user_input.isdigit():
                    choice_num = int(user_input)
                    self._handle_choice_selection(choice_num)
                else:
                    # Try to navigate directly to the node
                    if user_input in self.tree.nodes:
                        self.current_node_id = user_input
                        print(f"🎯 Navigated to node: {user_input}")
                    else:
                        print(f"❌ Unknown command: {user_input}")

        except Exception as e:
            logger.error(f"Error in debugger: {e}")
            print(f"❌ Fatal error in debugger: {e}")


def run_debugger(tree: DialogueTree, start_node_id: Optional[str] = None) -> None:
    """
    Run the dialogue tree debugger.

    Args:
        tree: The dialogue tree to debug
        start_node_id: Node ID to start from, or None for root node
    """
    print("Starting dialogue tree debugger...")
    print("Loading tree...")
    
    try:
        debugger = DialogueDebugger(tree, start_node_id)
        debugger.run()
    except Exception as e:
        logger.error(f"Fatal debugger error: {e}")
        print(f"❌ Fatal error: {e}")
        print("Debugger session ended.")
