#!/usr/bin/env python3
"""
Web UI for the Bootstrap Game Dialog Generator.

This module provides a Flask-based web interface for the dialogue tree generator,
allowing interactive navigation and node generation through a web browser.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request, send_from_directory

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dialogue_tree import DialogueTree, DialogueTreeManager, validate_generated_node
from llm_integration import NodeGenerator, OllamaClient

logger = logging.getLogger(__name__)


class DialogueTreeWebApp:
    """Web application for dialogue tree management and generation."""

    def __init__(self, tree_file: Path, model: str = "llama3"):
        self.tree_file = tree_file
        self.tree_manager = DialogueTreeManager(tree_file)
        self.llm_client = OllamaClient(model)
        self.node_generator = NodeGenerator(self.llm_client)
        self.current_tree: Optional[DialogueTree] = None
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent.parent / "templates"),
                        static_folder=str(Path(__file__).parent.parent / "static"))
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the main web interface."""
            return render_template('index.html')

        @self.app.route('/api/tree', methods=['GET'])
        def get_tree():
            """Get the current dialogue tree structure."""
            try:
                if self.current_tree is None:
                    self.current_tree = self.tree_manager.load_tree()
                
                # Return tree data with additional metadata
                tree_data = self.current_tree.to_dict()
                
                # Add metadata about null nodes
                null_nodes = []
                for node_id, node_data in tree_data['nodes'].items():
                    if node_data is None:
                        null_nodes.append(node_id)
                
                result = {
                    'tree': tree_data,
                    'null_nodes': null_nodes,
                    'total_nodes': len(tree_data['nodes']),
                    'completed_nodes': len(tree_data['nodes']) - len(null_nodes)
                }
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error loading tree: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/tree', methods=['POST'])
        def save_tree():
            """Save the current dialogue tree."""
            try:
                if self.current_tree is None:
                    return jsonify({'error': 'No tree loaded'}), 400
                
                self.tree_manager.save_tree(self.current_tree)
                backup_path = self.tree_manager.create_backup(self.current_tree)
                
                return jsonify({
                    'message': 'Tree saved successfully',
                    'backup': str(backup_path)
                })
            except Exception as e:
                logger.error(f"Error saving tree: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/node/<node_id>', methods=['GET'])
        def get_node(node_id: str):
            """Get details for a specific node."""
            try:
                if self.current_tree is None:
                    self.current_tree = self.tree_manager.load_tree()
                
                node_data = self.current_tree.get_node(node_id)
                if node_data is None and node_id in self.current_tree.nodes:
                    # This is a null node
                    parent_info = self.current_tree.find_parent_and_choice(node_id)
                    result = {
                        'node_id': node_id,
                        'is_null': True,
                        'parent_info': parent_info
                    }
                    return jsonify(result)
                elif node_data is not None:
                    result = {
                        'node_id': node_id,
                        'is_null': False,
                        'data': node_data
                    }
                    return jsonify(result)
                else:
                    return jsonify({'error': 'Node not found'}), 404
                    
            except Exception as e:
                logger.error(f"Error getting node {node_id}: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/generate/<node_id>', methods=['POST'])
        def generate_node(node_id: str):
            """Generate content for a null node."""
            try:
                if self.current_tree is None:
                    self.current_tree = self.tree_manager.load_tree()
                
                # Check if this is a null node
                if node_id not in self.current_tree.nodes or self.current_tree.nodes[node_id] is not None:
                    return jsonify({'error': 'Node is not a null node or does not exist'}), 400
                
                # Find parent and choice
                parent_info = self.current_tree.find_parent_and_choice(node_id)
                if parent_info is None:
                    return jsonify({'error': 'Could not find parent for node'}), 400
                
                parent_id, choice = parent_info
                parent_node = self.current_tree.get_node(parent_id)
                
                if parent_node is None:
                    return jsonify({'error': 'Parent node not found'}), 400
                
                parent_situation = parent_node.get("situation", "")
                choice_text = choice.get("text", "")
                
                logger.info(f"Generating content for node '{node_id}'")
                
                # Build dialogue history
                dialogue_history = self.current_tree.build_dialogue_history(node_id)
                
                # Generate the node
                generated_node = self.node_generator.generate_node(
                    parent_situation=parent_situation,
                    choice_text=choice_text,
                    params=self.current_tree.params,
                    dialogue_history=dialogue_history,
                    rules=self.current_tree.rules,
                    scene=self.current_tree.scene,
                )
                
                if generated_node is None:
                    return jsonify({'error': 'Failed to generate node content'}), 500
                
                # Validate the generated node
                if not validate_generated_node(generated_node):
                    return jsonify({'error': 'Generated node failed validation'}), 500
                
                # Update the tree
                self.current_tree.update_node(node_id, generated_node)
                
                # Update choice "next" values to create new placeholder nodes
                for choice in generated_node.get("choices", []):
                    next_node_id = f"node_{len(self.current_tree.nodes) + hash(choice['text']) % 1000}"
                    choice["next"] = next_node_id
                    self.current_tree.nodes[next_node_id] = None
                
                # Save the updated tree
                self.tree_manager.save_tree(self.current_tree)
                backup_path = self.tree_manager.create_backup(self.current_tree)
                
                logger.info(f"Successfully generated node: {node_id}")
                
                return jsonify({
                    'message': 'Node generated successfully',
                    'node_data': generated_node,
                    'backup': str(backup_path)
                })
                
            except Exception as e:
                logger.error(f"Error generating node {node_id}: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            """Get application status and health check."""
            try:
                # Check if LLM is available
                llm_available = self.llm_client.is_available()
                
                # Check if tree file exists
                tree_file_exists = self.tree_file.exists()
                
                status = {
                    'tree_file': str(self.tree_file),
                    'tree_file_exists': tree_file_exists,
                    'llm_model': self.llm_client.model,
                    'llm_available': llm_available,
                    'status': 'ready' if (tree_file_exists and llm_available) else 'not_ready'
                }
                
                return jsonify(status)
                
            except Exception as e:
                logger.error(f"Error checking status: {e}")
                return jsonify({'error': str(e)}), 500

    def run(self, host: str = '127.0.0.1', port: int = 5000, debug: bool = False) -> None:
        """Run the Flask web application."""
        logger.info(f"Starting web UI on http://{host}:{port}")
        logger.info(f"Tree file: {self.tree_file}")
        
        # Create templates and static directories if they don't exist
        templates_dir = Path(__file__).parent.parent / "templates"
        static_dir = Path(__file__).parent.parent / "static"
        templates_dir.mkdir(exist_ok=True)
        static_dir.mkdir(exist_ok=True)
        
        self.app.run(host=host, port=port, debug=debug)