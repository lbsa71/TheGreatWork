#!/usr/bin/env python3
"""
Web UI for the dialogue tree debugger.

This module provides a Flask-based web interface for navigating
and debugging dialogue trees with a more interactive UI.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, render_template, jsonify, request, send_from_directory

from .dialogue_tree import DialogueTree, DialogueTreeManager
from .llm_integration import NodeGenerator, OllamaClient

logger = logging.getLogger(__name__)


class DialogueWebUI:
    """Web-based dialogue tree interface."""

    def __init__(self, tree_file: str, model: str = "llama3"):
        """
        Initialize the web UI.

        Args:
            tree_file: Path to the dialogue tree JSON file
            model: LLM model name for generation
        """
        self.tree_file = Path(tree_file)
        self.model = model

        # Load the dialogue tree
        self.tree_manager = DialogueTreeManager(str(self.tree_file))
        self.tree = self.tree_manager.load_tree()

        # Initialize LLM client for AI generation
        self.llm_client = OllamaClient(model=model)
        self.node_generator = NodeGenerator(self.llm_client)

        # Flask app setup
        self.app = Flask(
            __name__,
            template_folder=str(Path(__file__).parent.parent / "templates"),
            static_folder=str(Path(__file__).parent.parent / "static"),
        )
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup Flask routes."""

        @self.app.route("/")
        def index() -> str:
            """Main page."""
            return render_template("index.html")

        @self.app.route("/api/tree")
        def get_tree() -> Any:
            """Get the current dialogue tree."""
            return jsonify(
                {
                    "nodes": self.tree.nodes,
                    "rules": getattr(self.tree, "rules", {}),
                    "scene": getattr(self.tree, "scene", {}),
                    "params": getattr(self.tree, "params", {}),
                }
            )

        @self.app.route("/api/node/<node_id>")
        def get_node(node_id: str) -> Any:
            """Get a specific node."""
            if node_id not in self.tree.nodes:
                return jsonify({"error": "Node not found"}), 404

            node = self.tree.nodes[node_id]
            if node is None:
                return jsonify({"error": "Node is null"}), 404

            return jsonify(
                {
                    "id": node_id,
                    "situation": node.get("situation", ""),
                    "choices": node.get("choices", []),
                }
            )

        @self.app.route("/api/history/<node_id>")
        def get_dialogue_history(node_id: str) -> Any:
            """Get the dialogue history leading to a specific node."""
            if node_id not in self.tree.nodes:
                return jsonify({"error": "Node not found"}), 404

            # Build structured dialogue history
            history_steps = []
            current_node_id = node_id

            # Backtrack through the tree to build history
            while current_node_id:
                parent_info = self.tree.find_parent_and_choice(current_node_id)
                if parent_info is None:
                    # We've reached the root or a disconnected node
                    break

                parent_id, choice = parent_info
                parent_node = self.tree.get_node(parent_id)

                if parent_node is None:
                    break

                # Add this step to the history
                situation = parent_node.get("situation", "")
                choice_text = choice.get("text", "")
                choice_effects = choice.get("effects", {})

                history_steps.append(
                    {
                        "node_id": parent_id,
                        "situation": situation,
                        "choice_text": choice_text,
                        "choice_effects": choice_effects,
                    }
                )

                # Move to the parent for next iteration
                current_node_id = parent_id

            # Reverse to get chronological order (root to target)
            history_steps.reverse()

            return jsonify({"history": history_steps, "target_node": node_id})

        @self.app.route("/api/tree/structure")
        def get_tree_structure() -> Any:
            """Get the tree structure for navigation."""
            structure = {}

            for node_id, node_data in self.tree.nodes.items():
                if node_data is not None:
                    choices = node_data.get("choices", [])
                    children = [
                        choice.get("next") for choice in choices if choice.get("next")
                    ]
                    structure[node_id] = {
                        "situation": (
                            node_data.get("situation", "")[:100] + "..."
                            if len(node_data.get("situation", "")) > 100
                            else node_data.get("situation", "")
                        ),
                        "children": children,
                        "has_choices": len(choices) > 0,
                        "is_null": False,
                    }
                else:
                    structure[node_id] = {
                        "situation": "Incomplete node",
                        "children": [],
                        "has_choices": False,
                        "is_null": True,
                    }

            return jsonify(structure)

        @self.app.route("/api/generate/<node_id>", methods=["POST"])
        def generate_node(node_id: str) -> Any:
            """Generate content for a null node using AI."""
            if node_id not in self.tree.nodes:
                return jsonify({"error": "Node not found"}), 404

            if self.tree.nodes[node_id] is not None:
                return jsonify({"error": "Node already has content"}), 400

            # Check if Ollama is available before attempting generation
            if not self.llm_client.is_available():
                return (
                    jsonify(
                        {
                            "error": "AI generation is not available. Please ensure Ollama is installed and running with the required model.",
                            "details": f"Required model: {self.llm_client.model}",
                            "instructions": "Install Ollama and run: ollama pull "
                            + self.llm_client.model,
                        }
                    ),
                    503,
                )

            try:
                # Find parent context
                parent_context = self._find_parent_context(node_id)

                if not parent_context:
                    return (
                        jsonify({"error": "Could not find parent context for node"}),
                        400,
                    )

                # Get tree parameters for generation
                tree_params = getattr(self.tree, "params", {})
                tree_rules = getattr(self.tree, "rules", {})
                tree_scene = getattr(self.tree, "scene", {})

                # Generate new node content with correct parameters
                generated_content = self.node_generator.generate_node(
                    parent_situation=parent_context["parent_situation"],
                    choice_text=parent_context["choice_text"],
                    params=tree_params,
                    rules=tree_rules,
                    scene=tree_scene,
                )

                if generated_content:
                    # Update the tree
                    self.tree.nodes[node_id] = generated_content

                    # Save the updated tree
                    self.tree_manager.save_tree(self.tree)

                    return jsonify({"success": True, "node": generated_content})
                else:
                    return jsonify({"error": "Failed to generate content"}), 500

            except Exception as e:
                logger.error(f"Error generating node {node_id}: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/save", methods=["POST"])
        def save_tree() -> Any:
            """Save the current tree to file."""
            try:
                self.tree_manager.save_tree(self.tree)
                return jsonify({"success": True})
            except Exception as e:
                logger.error(f"Error saving tree: {e}")
                return jsonify({"error": str(e)}), 500

    def _find_parent_context(self, target_node_id: str) -> Optional[Dict[str, Any]]:
        """Find the parent context for a given node."""
        for node_id, node_data in self.tree.nodes.items():
            if node_data is not None and isinstance(node_data, dict):
                choices = node_data.get("choices", [])
                for i, choice in enumerate(choices):
                    if choice.get("next") == target_node_id:
                        return {
                            "parent_node_id": node_id,
                            "parent_situation": node_data.get("situation", ""),
                            "choice_index": i,
                            "choice_text": choice.get("text", ""),
                            "choice_effects": choice.get("effects", {}),
                        }
        return None

    def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
        """Run the web server."""
        logger.info(f"Starting web UI on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def run_web_ui(
    tree_file: str,
    model: str = "llama3",
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
) -> None:
    """
    Run the web UI.

    Args:
        tree_file: Path to the dialogue tree JSON file
        model: LLM model name for generation
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    try:
        web_ui = DialogueWebUI(tree_file, model)
        web_ui.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start web UI: {e}")
        raise
