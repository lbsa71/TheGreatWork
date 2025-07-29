#!/usr/bin/env python3
"""
Illustration generation for dialogue tree nodes using InvokeAI API.

This module handles generating illustrations for dialogue nodes by calling
the InvokeAI API and managing image storage.
"""

import json
import logging
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class IllustrationError(Exception):
    """Base exception for illustration operations."""
    pass


class InvokeAIClient:
    """Client for interacting with the InvokeAI API."""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        """
        Initialize the InvokeAI client.
        
        Args:
            base_url: Base URL for the InvokeAI API server
        """
        self.base_url = base_url
        self.api_url = urljoin(base_url, "/api/v1/generate")
        
    def is_available(self) -> bool:
        """
        Check if the InvokeAI API is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            response = requests.get(self.base_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def generate_image(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        cfg_scale: float = 7.5
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an image using the InvokeAI API.
        
        Args:
            prompt: The text prompt for image generation
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of inference steps
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            API response data or None if failed
        """
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale
        }
        
        try:
            logger.debug(f"Sending image generation request to {self.api_url}")
            logger.debug(f"Payload: {payload}")
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60  # Generous timeout for image generation
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Image generation successful: {result}")
                return result
            else:
                logger.error(f"Image generation failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error calling InvokeAI API: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from InvokeAI API: {e}")
            return None


class IllustrationGenerator:
    """Generates illustrations for dialogue tree nodes."""
    
    def __init__(self, client: InvokeAIClient, images_dir: Path):
        """
        Initialize the illustration generator.
        
        Args:
            client: InvokeAI client instance
            images_dir: Directory to store generated images
        """
        self.client = client
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
        
    def build_prompt(
        self,
        situation: str,
        rules: Optional[Dict[str, Any]] = None,
        scene: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build a comprehensive prompt for image generation.
        
        Args:
            situation: The node's situation text
            rules: Dialogue rules containing style/tone information
            scene: Scene information containing setting/atmosphere
            
        Returns:
            Formatted prompt for image generation
        """
        prompt_parts = []
        
        # Add hardcoded style words for better results
        prompt_parts.append("photorealistic, insane detail, masterpiece, best quality")
        
        # Add style and tone from rules
        if rules:
            style = rules.get("style", "")
            tone = rules.get("tone", "")
            if style:
                prompt_parts.append(style)
            if tone:
                prompt_parts.append(tone)
        
        # Add scene information
        if scene:
            setting = scene.get("setting", "")
            time_period = scene.get("time_period", "")
            atmosphere = scene.get("atmosphere", "")
            
            if setting:
                prompt_parts.append(setting)
            if time_period:
                prompt_parts.append(f"time period: {time_period}")
            if atmosphere:
                prompt_parts.append(f"atmosphere: {atmosphere}")
        
        # Add the situation text (main subject)
        if situation:
            prompt_parts.append(situation)
        
        # Join all parts with commas
        return ", ".join(filter(None, prompt_parts))
    
    def save_image_data(self, node_id: str, image_data: Dict[str, Any]) -> Optional[str]:
        """
        Save image data to disk and return the relative path.
        
        Args:
            node_id: ID of the node this image belongs to
            image_data: Image data from InvokeAI API
            
        Returns:
            Relative path to saved image or None if failed
        """
        try:
            # Create node-specific directory
            node_dir = self.images_dir / node_id
            node_dir.mkdir(exist_ok=True)
            
            # TODO: Extract actual image data from API response
            # The exact format depends on how InvokeAI returns the image
            # For now, we'll create a placeholder implementation
            
            # This is a placeholder - we need to understand the actual API response format
            if "image" in image_data:
                # Assume base64 encoded image or direct binary data
                image_filename = f"{node_id}.png"
                image_path = node_dir / image_filename
                
                # TODO: Implement actual image saving based on API response format
                # This might involve base64 decoding or handling binary data
                logger.warning("Image saving not fully implemented - placeholder created")
                
                # Create placeholder file for now
                with open(image_path, "w") as f:
                    f.write("placeholder image data")
                
                # Return relative path from project root
                relative_path = f"images/{node_id}/{image_filename}"
                logger.info(f"Saved image to {relative_path}")
                return relative_path
            else:
                logger.error("No image data found in API response")
                return None
                
        except Exception as e:
            logger.error(f"Error saving image for node {node_id}: {e}")
            return None
    
    def generate_illustration(
        self,
        node_id: str,
        situation: str,
        rules: Optional[Dict[str, Any]] = None,
        scene: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate an illustration for a dialogue node.
        
        Args:
            node_id: ID of the node
            situation: The node's situation text
            rules: Dialogue rules containing style/tone information
            scene: Scene information containing setting/atmosphere
            
        Returns:
            Relative path to generated image or None if failed
        """
        logger.info(f"Generating illustration for node: {node_id}")
        
        # Build the prompt
        prompt = self.build_prompt(situation, rules, scene)
        logger.info(f"Generated prompt: {prompt}")
        
        # Generate the image
        image_data = self.client.generate_image(prompt)
        if image_data is None:
            logger.error(f"Failed to generate image for node: {node_id}")
            return None
        
        # Save the image and return path
        return self.save_image_data(node_id, image_data)


def find_nodes_without_illustrations(nodes: Dict[str, Any]) -> List[str]:
    """
    Find nodes that don't have illustrations using breadth-first search.
    
    Args:
        nodes: Dictionary of all nodes in the dialogue tree
        
    Returns:
        List of node IDs without illustrations, in breadth-first order
    """
    # Find nodes without illustrations
    nodes_without_illustrations = []
    
    for node_id, node_data in nodes.items():
        if node_data is None:
            continue
            
        # Skip failed nodes
        if isinstance(node_data, dict) and node_data.get("__failed__"):
            continue
            
        # Check if node has illustration
        if isinstance(node_data, dict) and not node_data.get("illustration"):
            nodes_without_illustrations.append(node_id)
    
    return nodes_without_illustrations


def find_null_nodes(nodes: Dict[str, Any]) -> List[str]:
    """
    Find null nodes in the dialogue tree.
    
    Args:
        nodes: Dictionary of all nodes in the dialogue tree
        
    Returns:
        List of null node IDs
    """
    null_nodes = []
    
    for node_id, node_data in nodes.items():
        if node_data is None:
            null_nodes.append(node_id)
        elif isinstance(node_data, dict) and node_data.get("__failed__"):
            # Skip failed nodes
            continue
    
    return null_nodes


def should_generate_illustrations_first(nodes: Dict[str, Any]) -> bool:
    """
    Determine if we should prioritize illustration generation over text generation.
    
    Args:
        nodes: Dictionary of all nodes in the dialogue tree
        
    Returns:
        True if there are non-null nodes without illustrations, False otherwise
    """
    nodes_without_illustrations = find_nodes_without_illustrations(nodes)
    return len(nodes_without_illustrations) > 0