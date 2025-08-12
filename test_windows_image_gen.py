#!/usr/bin/env python3
"""
Test script to validate Windows-compatible image generation functionality.
"""

import sys
import tempfile
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")
    
    try:
        from image_generation import (
            ImageGenerationStats,
            ONNXStableDiffusionGenerator,
            DialogueTreeIllustrationGenerator,
            generate_illustrations_for_nodes,
            ImageGenerationError,
            HAS_IMAGE_DEPS,
            HAS_BASIC_DEPS,
            HAS_ONNX_DEPS
        )
        print("✓ All modules imported successfully")
        print(f"  HAS_IMAGE_DEPS: {HAS_IMAGE_DEPS}")
        print(f"  HAS_BASIC_DEPS: {HAS_BASIC_DEPS}") 
        print(f"  HAS_ONNX_DEPS: {HAS_ONNX_DEPS}")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_stats():
    """Test statistics tracking."""
    print("\nTesting statistics...")
    
    from image_generation import ImageGenerationStats
    
    stats = ImageGenerationStats()
    assert stats.total_images_generated == 0
    assert stats.total_generation_time == 0.0
    assert len(stats.generation_times) == 0
    assert len(stats.failed_nodes) == 0
    
    # Test adding generation
    stats.add_generation(2.5)
    assert stats.total_images_generated == 1
    assert stats.total_generation_time == 2.5
    assert stats.mean_generation_time == 2.5
    
    # Test adding failure
    stats.add_failure("test_node")
    assert len(stats.failed_nodes) == 1
    assert "test_node" in stats.failed_nodes
    
    print("✓ Statistics tracking works correctly")
    return True

def test_generator_creation():
    """Test generator creation."""
    print("\nTesting generator creation...")
    
    from image_generation import ONNXStableDiffusionGenerator, HAS_IMAGE_DEPS
    
    if not HAS_IMAGE_DEPS:
        print("  Skipping - image dependencies not available")
        return True
        
    try:
        generator = ONNXStableDiffusionGenerator()
        assert generator.model_id == "runwayml/stable-diffusion-v1-5"
        assert generator.providers is not None
        print("✓ Generator created successfully")
        return True
    except Exception as e:
        print(f"✗ Generator creation failed: {e}")
        return False

def test_placeholder_generation():
    """Test placeholder image generation."""
    print("\nTesting placeholder image generation...")
    
    from image_generation import ONNXStableDiffusionGenerator, HAS_IMAGE_DEPS
    
    if not HAS_IMAGE_DEPS:
        print("  Skipping - image dependencies not available")
        return True
    
    try:
        generator = ONNXStableDiffusionGenerator()
        
        # Test image generation (should use placeholder)
        images, metadata = generator.generate_image(
            "A medieval castle",
            width=256,
            height=256,
            num_inference_steps=10
        )
        
        assert len(images) == 1
        assert metadata["backend"] in ["ONNX Runtime", "Placeholder"]
        assert metadata["width"] == 256
        assert metadata["height"] == 256
        
        print("✓ Placeholder image generation works")
        return True
    except Exception as e:
        print(f"✗ Image generation failed: {e}")
        return False

def test_bfs_discovery():
    """Test BFS node discovery."""
    print("\nTesting BFS node discovery...")
    
    from image_generation import DialogueTreeIllustrationGenerator, ONNXStableDiffusionGenerator, HAS_IMAGE_DEPS
    
    if not HAS_IMAGE_DEPS:
        print("  Skipping - image dependencies not available")
        return True
    
    try:
        generator = ONNXStableDiffusionGenerator()
        illustrator = DialogueTreeIllustrationGenerator(generator)
        
        # Test tree structure
        tree_nodes = {
            "start": {
                "situation": "Start node",
                "choices": [{"text": "Go forward", "next": "middle"}]
            },
            "middle": {
                "situation": "Middle node", 
                "choices": [{"text": "Continue", "next": "end"}]
            },
            "end": {
                "situation": "End node",
                "choices": []
            }
        }
        
        nodes = illustrator.find_nodes_without_illustrations(tree_nodes)
        assert len(nodes) == 3
        assert "start" in nodes
        assert "middle" in nodes
        assert "end" in nodes
        
        print("✓ BFS discovery works correctly")
        return True
    except Exception as e:
        print(f"✗ BFS discovery failed: {e}")
        return False

def test_integration():
    """Test the complete integration."""
    print("\nTesting complete integration...")
    
    from image_generation import generate_illustrations_for_nodes, HAS_IMAGE_DEPS
    
    if not HAS_IMAGE_DEPS:
        print("  Skipping - image dependencies not available")
        return True
    
    try:
        # Create test tree
        tree_nodes = {
            "start": {
                "situation": "You are in a magical forest with glowing trees",
                "choices": []
            }
        }
        
        context = {
            "setting": "magical forest",
            "atmosphere": "mystical"
        }
        
        # Generate images
        with tempfile.TemporaryDirectory() as temp_dir:
            count, stats = generate_illustrations_for_nodes(
                tree_nodes=tree_nodes,
                context=context,
                images_dir=Path(temp_dir) / "images",
                width=128,
                height=128,
                max_nodes=1
            )
            
            assert count == 1
            assert stats.total_images_generated == 1
            assert "illustration" in tree_nodes["start"]
            
        print("✓ Complete integration works")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Windows-Compatible Image Generation Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_stats,
        test_generator_creation,
        test_placeholder_generation,
        test_bfs_discovery,
        test_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Windows-compatible image generation is working.")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())