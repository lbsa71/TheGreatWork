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
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent.parent / 'templates'), 
                        static_folder=str(Path(__file__).parent.parent / 'static'))
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            return render_template('index.html')
        
        @self.app.route('/api/tree')
        def get_tree():
            """Get the current dialogue tree."""
            return jsonify({
                'nodes': self.tree.nodes,
                'rules': getattr(self.tree, 'rules', {}),
                'scene': getattr(self.tree, 'scene', {}),
                'params': getattr(self.tree, 'params', {})
            })
        
        @self.app.route('/api/node/<node_id>')
        def get_node(node_id: str):
            """Get a specific node."""
            if node_id not in self.tree.nodes:
                return jsonify({'error': 'Node not found'}), 404
            
            node = self.tree.nodes[node_id]
            if node is None:
                return jsonify({'error': 'Node is null'}), 404
            
            return jsonify({
                'id': node_id,
                'situation': node.get('situation', ''),
                'choices': node.get('choices', [])
            })
        
        @self.app.route('/api/tree/structure')
        def get_tree_structure():
            """Get the tree structure for navigation."""
            structure = {}
            
            for node_id, node_data in self.tree.nodes.items():
                if node_data is not None:
                    choices = node_data.get('choices', [])
                    children = [choice.get('next') for choice in choices if choice.get('next')]
                    structure[node_id] = {
                        'situation': node_data.get('situation', '')[:100] + '...' if len(node_data.get('situation', '')) > 100 else node_data.get('situation', ''),
                        'children': children,
                        'has_choices': len(choices) > 0,
                        'is_null': False
                    }
                else:
                    structure[node_id] = {
                        'situation': 'Incomplete node',
                        'children': [],
                        'has_choices': False,
                        'is_null': True
                    }
            
            return jsonify(structure)
        
        @self.app.route('/api/generate/<node_id>', methods=['POST'])
        def generate_node(node_id: str):
            """Generate content for a null node using AI."""
            if node_id not in self.tree.nodes:
                return jsonify({'error': 'Node not found'}), 404
            
            if self.tree.nodes[node_id] is not None:
                return jsonify({'error': 'Node already has content'}), 400
            
            try:
                # Find parent context
                parent_context = self._find_parent_context(node_id)
                
                # Generate new node content
                generated_content = self.node_generator.generate_node(
                    self.tree, node_id, parent_context
                )
                
                if generated_content:
                    # Update the tree
                    self.tree.nodes[node_id] = generated_content
                    
                    # Save the updated tree
                    self.tree_manager.save_tree(self.tree)
                    
                    return jsonify({
                        'success': True,
                        'node': generated_content
                    })
                else:
                    return jsonify({'error': 'Failed to generate content'}), 500
                    
            except Exception as e:
                logger.error(f"Error generating node {node_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/save', methods=['POST'])
        def save_tree():
            """Save the current tree to file."""
            try:
                self.tree_manager.save_tree(self.tree)
                return jsonify({'success': True})
            except Exception as e:
                logger.error(f"Error saving tree: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _find_parent_context(self, target_node_id: str) -> Optional[Dict[str, Any]]:
        """Find the parent context for a given node."""
        for node_id, node_data in self.tree.nodes.items():
            if node_data is not None and isinstance(node_data, dict):
                choices = node_data.get('choices', [])
                for i, choice in enumerate(choices):
                    if choice.get('next') == target_node_id:
                        return {
                            'parent_node_id': node_id,
                            'parent_situation': node_data.get('situation', ''),
                            'choice_index': i,
                            'choice_text': choice.get('text', ''),
                            'choice_effects': choice.get('effects', {})
                        }
        return None
    
    def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
        """Run the web server."""
        logger.info(f"Starting web UI on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def run_web_ui(tree_file: str, model: str = "llama3", host: str = "127.0.0.1", 
               port: int = 5000, debug: bool = False):
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