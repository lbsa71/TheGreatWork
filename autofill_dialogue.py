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
import time
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.debugger import run_debugger
from src.dialogue_tree import (
    DialogueTree,
    DialogueTreeError,
    DialogueTreeManager,
    validate_generated_node,
)
from src.illustration import (
    IllustrationGenerator,
    InvokeAIClient,
    IllustrationError,
    should_generate_illustrations_first,
)
from src.llm_integration import NodeGenerator, OllamaClient

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
        self, tree_file: Path, model: str = "qwen3:14b", max_nodes: Optional[int] = None,
        enable_illustrations: bool = True, invokeai_url: str = "http://localhost:9090"
    ):
        self.tree_manager = DialogueTreeManager(tree_file)
        self.llm_client = OllamaClient(model)
        self.node_generator = NodeGenerator(self.llm_client)
        
        # Illustration components
        self.enable_illustrations = enable_illustrations
        self.illustration_client = InvokeAIClient(invokeai_url) if enable_illustrations else None
        self.illustration_generator = None
        if enable_illustrations:
            images_dir = tree_file.parent / "images"
            self.illustration_generator = IllustrationGenerator(self.illustration_client, images_dir)
        
        self.nodes_generated = 0
        self.max_nodes = max_nodes
        self.generation_times = []  # Track generation times for statistics
        self.failed_nodes = []  # Track failed node IDs
        
        # Illustration statistics
        self.illustrations_generated = 0
        self.illustration_times = []
        self.failed_illustrations = []

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

        # Check if InvokeAI is available (if illustrations are enabled)
        if self.enable_illustrations and self.illustration_client:
            if not self.illustration_client.is_available():
                logger.warning("InvokeAI API is not available - illustrations will be disabled")
                logger.warning("To enable illustrations, ensure InvokeAI is running on localhost:9090")
                self.enable_illustrations = False
                self.illustration_client = None
                self.illustration_generator = None
            else:
                logger.info("InvokeAI API is available - illustrations enabled")

        logger.info("All prerequisites met")
        return True

    def process_tree(self) -> bool:
        """
        Process the dialogue tree, filling all null nodes and generating illustrations.

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
                # Check for nodes without illustrations first (breadth-first approach)
                if self.enable_illustrations and self.illustration_generator:
                    nodes_without_illustrations = tree.find_nodes_without_illustrations()
                    if nodes_without_illustrations:
                        logger.info(f"Found {len(nodes_without_illustrations)} nodes without illustrations")
                        
                        # Process first node without illustration
                        node_id = nodes_without_illustrations[0]
                        if self._process_illustration(tree, node_id):
                            self.illustrations_generated += 1
                            logger.info(f"Successfully generated {self.illustrations_generated} illustrations so far")
                        
                        # Save after each successful illustration
                        self.tree_manager.save_tree(tree)
                        continue

                # If all non-null nodes have illustrations, look for null nodes
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
                        f"Reached maximum node limit ({self.max_nodes}). "
                        "Stopping generation."
                    )
                    break

                logger.info(f"Processing null node: {null_node_id}")

                if not self._process_node(tree, null_node_id):
                    logger.warning(f"Failed to process node: {null_node_id} - skipping and continuing")
                    # Mark the node as failed by setting it to a special value
                    # This prevents infinite loops while allowing us to track failed nodes
                    tree.nodes[null_node_id] = {"__failed__": True, "situation": "Generation failed", "choices": []}
                    # Track the failed node
                    self.failed_nodes.append(null_node_id)
                    # Save the tree with the failed node marked
                    self.tree_manager.save_tree(tree)
                    continue

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
            
            # Determine completion reason
            if self.max_nodes is not None and self.nodes_generated >= self.max_nodes:
                logger.info(
                    f"Generation completed: reached maximum node limit ({self.max_nodes})"
                )
            else:
                logger.info("Generation completed: all nodes processed")
            
            logger.info(f"Generated {self.nodes_generated} nodes total")
            if self.failed_nodes:
                logger.info(f"Skipped {len(self.failed_nodes)} failed nodes")
            
            if self.enable_illustrations:
                logger.info(f"Generated {self.illustrations_generated} illustrations total")
                if self.failed_illustrations:
                    logger.info(f"Skipped {len(self.failed_illustrations)} failed illustrations")
            
            # Print generation statistics
            self.print_statistics()
            
            return True

        except DialogueTreeError as e:
            logger.error(f"Dialogue tree error: {e}")
            self.print_statistics()
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.print_statistics()
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

        # Time the generation process
        start_time = time.time()
        
        # Generate the node
        generated_node = self.node_generator.generate_node(
            parent_situation=parent_situation,
            choice_text=choice_text,
            params=tree.params,
            dialogue_history=dialogue_history,
            rules=tree.rules,
            scene=tree.scene,
        )
        
        # Record generation time
        generation_time = time.time() - start_time
        self.generation_times.append(generation_time)
        logger.info(f"Node generation completed in {generation_time:.2f} seconds")

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

        # Update choice "next" values to create new placeholder nodes using AI
        # suggestions
        for choice in generated_node.get("choices", []):
            suggested_id = choice.get("suggested_node_id", "")
            next_node_id = tree.generate_unique_node_id(suggested_id)
            choice["next"] = next_node_id
            tree.nodes[next_node_id] = None

            # Clean up the suggested_node_id field from the final choice
            # data
            if "suggested_node_id" in choice:
                del choice["suggested_node_id"]

        logger.info(f"Successfully processed node: {node_id}")
        return True

    def _process_illustration(self, tree: DialogueTree, node_id: str) -> bool:
        """
        Process illustration generation for a node.
        
        Args:
            tree: The dialogue tree
            node_id: ID of the node to generate illustration for
            
        Returns:
            True if successful, False otherwise
        """
        node_data = tree.get_node(node_id)
        if node_data is None or not isinstance(node_data, dict):
            logger.error(f"Cannot generate illustration for null or invalid node: {node_id}")
            return False
            
        if node_data.get("illustration"):
            logger.debug(f"Node {node_id} already has illustration, skipping")
            return True
            
        situation = node_data.get("situation", "")
        if not situation:
            logger.warning(f"Node {node_id} has no situation text for illustration")
            return False
            
        logger.info(f"Generating illustration for node '{node_id}'")
        logger.info(f"  Situation: {situation[:100]}...")
        
        # Time the illustration generation
        start_time = time.time()
        
        try:
            # Generate the illustration
            illustration_path = self.illustration_generator.generate_illustration(
                node_id=node_id,
                situation=situation,
                rules=tree.rules,
                scene=tree.scene
            )
            
            # Record illustration generation time
            generation_time = time.time() - start_time
            self.illustration_times.append(generation_time)
            logger.info(f"Illustration generation completed in {generation_time:.2f} seconds")
            
            if illustration_path is None:
                logger.error(f"Failed to generate illustration for node: {node_id}")
                self.failed_illustrations.append(node_id)
                return False
                
            # Add illustration to the node
            if tree.add_illustration_to_node(node_id, illustration_path):
                logger.info(f"Successfully added illustration to node: {node_id}")
                return True
            else:
                logger.error(f"Failed to add illustration to node: {node_id}")
                self.failed_illustrations.append(node_id)
                return False
                
        except Exception as e:
            generation_time = time.time() - start_time
            self.illustration_times.append(generation_time)
            logger.error(f"Error generating illustration for node {node_id}: {e}")
            self.failed_illustrations.append(node_id)
            return False

    def print_statistics(self) -> None:
        """Print generation statistics."""
        if not self.generation_times and not self.failed_nodes and not self.illustration_times and not self.failed_illustrations:
            logger.info("No generation statistics available (no nodes were processed)")
            return

        logger.info("=" * 60)
        logger.info("GENERATION STATISTICS")
        logger.info("=" * 60)
        
        # Text generation statistics
        if self.generation_times:
            total_time = sum(self.generation_times)
            mean_time = total_time / len(self.generation_times)
            min_time = min(self.generation_times)
            max_time = max(self.generation_times)
            
            logger.info(f"Total nodes generated: {len(self.generation_times)}")
            logger.info(f"Total text generation time: {total_time:.2f} seconds")
            logger.info(f"Mean text generation time: {mean_time:.2f} seconds")
            logger.info(f"Fastest text generation: {min_time:.2f} seconds")
            logger.info(f"Slowest text generation: {max_time:.2f} seconds")
            logger.info(f"Average nodes per minute: {60 / mean_time:.1f}")
        else:
            logger.info("Total nodes generated: 0")
        
        if self.failed_nodes:
            logger.info(f"Failed nodes: {len(self.failed_nodes)}")
            logger.info(f"Failed node IDs: {', '.join(self.failed_nodes)}")
        else:
            logger.info("Failed nodes: 0")
        
        # Illustration generation statistics
        if self.enable_illustrations:
            logger.info("-" * 40)
            logger.info("ILLUSTRATION STATISTICS")
            logger.info("-" * 40)
            
            if self.illustration_times:
                total_illustration_time = sum(self.illustration_times)
                mean_illustration_time = total_illustration_time / len(self.illustration_times)
                min_illustration_time = min(self.illustration_times)
                max_illustration_time = max(self.illustration_times)
                
                logger.info(f"Total illustrations generated: {len(self.illustration_times)}")
                logger.info(f"Total illustration generation time: {total_illustration_time:.2f} seconds")
                logger.info(f"Mean illustration generation time: {mean_illustration_time:.2f} seconds")
                logger.info(f"Fastest illustration generation: {min_illustration_time:.2f} seconds")
                logger.info(f"Slowest illustration generation: {max_illustration_time:.2f} seconds")
                logger.info(f"Average illustrations per minute: {60 / mean_illustration_time:.1f}")
            else:
                logger.info("Total illustrations generated: 0")
            
            if self.failed_illustrations:
                logger.info(f"Failed illustrations: {len(self.failed_illustrations)}")
                logger.info(f"Failed illustration node IDs: {', '.join(self.failed_illustrations)}")
            else:
                logger.info("Failed illustrations: 0")
            
        logger.info("=" * 60)


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
  python autofill_dialogue.py tree.json --debug
  python autofill_dialogue.py tree.json --debug --start-node node1
        """,
    )

    parser.add_argument(
        "tree_file",
        nargs="?",
        default="tree.json",
        help="Path to the dialogue tree JSON file (default: tree.json)",
    )

    parser.add_argument(
        "--model", default="qwen3:14b", help="Ollama model to use (default: qwen3:14b)"
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

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Launch interactive dialogue tree debugger",
    )

    parser.add_argument(
        "--start-node",
        type=str,
        help="Starting node ID for debugger (default: auto-detect root)",
    )

    parser.add_argument(
        "--disable-illustrations",
        action="store_true",
        help="Disable illustration generation",
    )

    parser.add_argument(
        "--invokeai-url",
        default="http://localhost:9090",
        help="InvokeAI API URL (default: http://localhost:9090)",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    tree_file = Path(args.tree_file)

    # Handle sample creation
    if args.create_sample:
        create_sample_tree(tree_file)
        return 0

    # Handle debugger mode
    if args.debug:
        logger.info("Starting interactive dialogue tree debugger")
        logger.info(f"Tree file: {tree_file}")

        if not tree_file.exists():
            logger.error(f"Dialogue tree file not found: {tree_file}")
            return 1

        try:
            tree_manager = DialogueTreeManager(tree_file)
            tree = tree_manager.load_tree()
            run_debugger(tree, args.start_node)
            return 0
        except DialogueTreeError as e:
            logger.error(f"Failed to load tree: {e}")
            return 1
        except Exception as e:
            logger.error(f"Debugger error: {e}")
            return 1

    logger.info("Starting Bootstrap Game Dialog Generator")
    logger.info(f"Tree file: {tree_file}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Illustrations: {'disabled' if args.disable_illustrations else 'enabled'}")
    if not args.disable_illustrations:
        logger.info(f"InvokeAI URL: {args.invokeai_url}")

    # Create autofiller
    autofiller = DialogueAutofiller(
        tree_file, 
        args.model, 
        args.max_nodes,
        enable_illustrations=not args.disable_illustrations,
        invokeai_url=args.invokeai_url
    )

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
