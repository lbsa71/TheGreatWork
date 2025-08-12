#!/usr/bin/env python3
"""
Test script for image generation only, without requiring Ollama.
"""

import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dialogue_tree import DialogueTree
from image_generation import generate_illustrations_for_nodes, ImageGenerationStats

def create_test_tree():
    """Create a test dialogue tree with some nodes."""
    return {
        "nodes": {
            "start": {
                "situation": "You find yourself standing at the entrance of an ancient, mysterious castle shrouded in mist. The heavy wooden doors are slightly ajar, creaking ominously in the wind.",
                "choices": [
                    {"text": "Enter the castle boldly", "next": "castle_entrance"},
                    {"text": "Circle around to find another way in", "next": "castle_side"}
                ]
            },
            "castle_entrance": {
                "situation": "Inside the castle, you discover a grand hall with towering pillars and flickering torches casting dancing shadows on ancient tapestries.",
                "choices": [
                    {"text": "Examine the mysterious tapestries", "next": "tapestry_room"},
                    {"text": "Climb the spiral staircase", "next": "tower_stairs"}
                ]
            },
            "castle_side": {
                "situation": "Around the side of the castle, you discover an overgrown garden with a hidden entrance covered in thorny vines and glowing with ethereal light.",
                "choices": []
            }
        },
        "scene": {
            "setting": "A haunted medieval castle",
            "time_period": "Medieval era",
            "atmosphere": "Dark, mysterious, and magical"
        },
        "rules": {
            "setting": "Gothic fantasy medieval castle",
            "tone": "Mysterious and atmospheric"
        },
        "params": {
            "difficulty": 1,
            "magic_level": 2
        }
    }

def main():
    print("Testing Windows-compatible image generation...")
    
    # Create test tree
    tree_data = create_test_tree()
    
    # Generate images
    context = {
        **tree_data["scene"],
        **tree_data.get("rules", {})
    }
    
    print(f"Generating images for {len(tree_data['nodes'])} nodes...")
    
    count, stats = generate_illustrations_for_nodes(
        tree_nodes=tree_data["nodes"],
        context=context,
        max_nodes=3,
        width=512,
        height=512,
        num_inference_steps=15
    )
    
    print(f"\nGenerated {count} images")
    stats.print_statistics()
    
    # Show what was added to nodes
    print("\nNodes with illustrations:")
    for node_id, node_data in tree_data["nodes"].items():
        if isinstance(node_data, dict) and "illustration" in node_data:
            print(f"  {node_id}: {node_data['illustration']}")
    
    # Save updated tree
    output_file = "test-tree-with-images.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tree_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved updated tree to {output_file}")
    print("Image generation test completed successfully!")

if __name__ == "__main__":
    main()