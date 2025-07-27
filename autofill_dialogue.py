#!/usr/bin/env python3
"""
Bootstrap Game Dialog Generator

Autonomous Dialogue Tree Completion Script Using a Local LLM.

This script reads a JSON file representing a branching dialogue tree,
identifies incomplete nodes, uses Ollama to generate content, and updates
the JSON file until all nodes are filled.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dialogue_tree import (
    DialogueTree,
    DialogueTreeError,
    DialogueTreeManager,
    validate_generated_node,
)
from llm_integration import NodeGenerator, OllamaClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("autofill_dialogue.log"),
    ],
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    # Update existing loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers:
        handler.setLevel(level)


class DialogueAutofiller:
    """Main class for autofilling dialogue trees."""

    def __init__(
        self, tree_file: Path, model: str = "llama3", max_nodes: Optional[int] = None
    ):
        self.tree_manager = DialogueTreeManager(tree_file)
        self.llm_client = OllamaClient(model)
        self.node_generator = NodeGenerator(self.llm_client)
        self.nodes_generated = 0
        self.max_nodes = max_nodes

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met before starting."""
        logger.info("Checking prerequisites...")

        # Check if tree file exists
        if not self.tree_manager.file_path.exists():
            logger.error(f"Dialogue tree file not found: {self.tree_manager.file_path}")
            return False

        # Check if Ollama is available
        if not self.llm_client.is_available():
            logger.error(f"Ollama model '{self.llm_client.model}' is not available")
            logger.error("Please ensure Ollama is running and the model is installed:")
            logger.error(f"  ollama run {self.llm_client.model}")
            return False

        logger.info("All prerequisites met")
        return True

    def process_tree(self) -> bool:
        """
        Process the dialogue tree, filling all null nodes.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the tree
            tree = self.tree_manager.load_tree()
            logger.info(f"Loaded tree with {len(tree.nodes)} nodes")

            # Create initial backup
            backup_path = self.tree_manager.create_backup(tree)
            logger.info(f"Created initial backup: {backup_path}")

            # Process nodes until no null nodes remain or max_nodes limit reached
            while True:
                null_node_id = tree.find_first_null_node()
                if null_node_id is None:
                    logger.info("No more null nodes found - tree is complete!")
                    break

                # Check if we've reached the max_nodes limit
                if (
                    self.max_nodes is not None
                    and self.nodes_generated >= self.max_nodes
                ):
                    logger.info(
                        f"Reached maximum node limit ({self.max_nodes}). Stopping generation."
                    )
                    break

                logger.info(f"Processing null node: {null_node_id}")

                if not self._process_node(tree, null_node_id):
                    logger.error(f"Failed to process node: {null_node_id}")
                    return False

                # Save after each successful generation
                self.tree_manager.save_tree(tree)

                # Create backup after each node
                backup_path = self.tree_manager.create_backup(tree)
                logger.info(f"Created backup: {backup_path}")

                self.nodes_generated += 1
                logger.info(
                    f"Successfully generated {self.nodes_generated} nodes so far"
                )

            # Final save
            self.tree_manager.save_tree(tree)
            logger.info(
                f"Tree completion successful! Generated {self.nodes_generated} nodes total"
            )
            return True

        except DialogueTreeError as e:
            logger.error(f"Dialogue tree error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

    def _process_node(self, tree: DialogueTree, node_id: str) -> bool:
        """
        Process a single null node.

        Args:
            tree: The dialogue tree
            node_id: ID of the node to process

        Returns:
            True if successful, False otherwise
        """
        # Find parent and choice
        parent_info = tree.find_parent_and_choice(node_id)
        if parent_info is None:
            logger.error(f"Could not find parent for node: {node_id}")
            return False

        parent_id, choice = parent_info
        parent_node = tree.get_node(parent_id)

        if parent_node is None:
            logger.error(f"Parent node not found: {parent_id}")
            return False

        parent_situation = parent_node.get("situation", "")
        choice_text = choice.get("text", "")

        logger.info(f"Generating content for node '{node_id}'")
        logger.info(f"  Parent: {parent_id}")
        logger.info(f"  Choice: {choice_text}")

        # Build dialogue history for this node
        dialogue_history = tree.build_dialogue_history(node_id)
        logger.debug(f"Dialogue history:\n{dialogue_history}")

        # Generate the node
        generated_node = self.node_generator.generate_node(
            parent_situation=parent_situation,
            choice_text=choice_text,
            params=tree.params,
            dialogue_history=dialogue_history,
            rules=tree.rules,
            scene=tree.scene,
        )

        if generated_node is None:
            logger.error("Failed to generate node content")
            return False

        # Validate the generated node
        if not validate_generated_node(generated_node):
            logger.error("Generated node failed validation")
            logger.debug(f"Invalid node data: {generated_node}")
            return False

        # Update the tree
        tree.update_node(node_id, generated_node)

        # Update choice "next" values to create new placeholder nodes
        for choice in generated_node.get("choices", []):
            next_node_id = f"node_{len(tree.nodes) + hash(choice['text']) % 1000}"
            choice["next"] = next_node_id
            tree.nodes[next_node_id] = None

        logger.info(f"Successfully processed node: {node_id}")
        return True


def create_sample_tree(file_path: Path) -> None:
    """Create a sample dialogue tree file."""
    sample_tree = {
        "rules": {
            "language": "English",
            "tone": "dramatic and serious",
            "voice": "third person narrative",
            "style": "medieval fantasy with political intrigue",
        },
        "scene": {
            "setting": "A medieval kingdom in turmoil",
            "time_period": "Medieval era, similar to 12th century Europe",
            "location": "The royal castle and court",
            "atmosphere": "Tense and uncertain following the king's sudden death",
            "key_elements": "Political maneuvering, loyalty conflicts, succession crisis",
        },
        "nodes": {
            "start": {
                "situation": "The king is dead.",
                "choices": [
                    {"text": "Mourn publicly", "next": "node1"},
                    {"text": "Seize the throne", "next": "node2"},
                ],
            },
            "node1": None,
            "node2": None,
        },
        "params": {"loyalty": 45, "ambition": 80},
    }

    import json

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_tree, f, indent=2)

    logger.info(f"Created sample tree file: {file_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bootstrap Game Dialog Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python autofill_dialogue.py tree.json
  python autofill_dialogue.py tree.json --model mistral
  python autofill_dialogue.py tree.json --verbose
  python autofill_dialogue.py tree.json --max-nodes 10
  python autofill_dialogue.py --create-sample sample_tree.json
        """,
    )

    parser.add_argument(
        "tree_file",
        nargs="?",
        default="tree.json",
        help="Path to the dialogue tree JSON file (default: tree.json)",
    )

    parser.add_argument(
        "--model", default="llama3", help="Ollama model to use (default: llama3)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--max-nodes",
        type=int,
        help="Maximum number of nodes to generate (default: unlimited)",
    )

    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample tree file and exit",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    tree_file = Path(args.tree_file)

    # Handle sample creation
    if args.create_sample:
        create_sample_tree(tree_file)
        return 0

    logger.info("Starting Bootstrap Game Dialog Generator")
    logger.info(f"Tree file: {tree_file}")
    logger.info(f"Model: {args.model}")

    # Create autofiller
    autofiller = DialogueAutofiller(tree_file, args.model, args.max_nodes)

    # Check prerequisites
    if not autofiller.check_prerequisites():
        logger.error("Prerequisites not met. Exiting.")
        return 1

    # Process the tree
    if autofiller.process_tree():
        logger.info("Dialogue tree autofill completed successfully!")
        return 0
    else:
        logger.error("Dialogue tree autofill failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
