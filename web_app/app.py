#!/usr/bin/env python3
"""
Web UI for the Bootstrap Game Dialog Generator.

This is a separate Flask application that uses the core business logic
from the main package as a shared library.
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the core logic
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, render_template, request
from src.dialogue_tree import DialogueTree, DialogueTreeManager, DialogueTreeError
from src.llm_integration import NodeGenerator, OllamaClient

logger = logging.getLogger(__name__)


class DialogueWebApp:
    """Flask-based web application for dialogue tree management."""

    def __init__(self, tree_file: str, model: str = "llama3"):
        """
        Initialize the web application.

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
        def index():
            """Main page."""
            return render_template("index.html")

        @self.app.route("/api/tree")
        def get_tree():
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
        def get_node(node_id: str):
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
        def get_dialogue_history(node_id: str):
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
        def get_tree_structure():
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
        def generate_node(node_id: str):
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
        def save_tree():
            """Save the current tree to file."""
            try:
                self.tree_manager.save_tree(self.tree)
                return jsonify({"success": True})
            except Exception as e:
                logger.error(f"Error saving tree: {e}")
                return jsonify({"error": str(e)}), 500

    def _find_parent_context(self, target_node_id: str):
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

    def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
        """Run the web server."""
        logger.info(f"Starting web UI on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def run_web_app(
    tree_file: str,
    model: str = "llama3",
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
):
    """
    Run the web application.

    Args:
        tree_file: Path to the dialogue tree JSON file
        model: LLM model name for generation
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    try:
        web_app = DialogueWebApp(tree_file, model)
        web_app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start web app: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dialogue Tree Web Application")
    parser.add_argument("tree_file", help="Path to the dialogue tree JSON file")
    parser.add_argument("--model", default="llama3", help="LLM model to use")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    run_web_app(
        tree_file=args.tree_file,
        model=args.model,
        host=args.host,
        port=args.port,
        debug=args.debug,
    ) 