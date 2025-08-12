#!/usr/bin/env python3
"""
Tests for the image generation module.

These tests cover the ONNX Stable Diffusion image generation functionality
for dialogue tree nodes, including BFS node discovery, prompt building,
and image saving operations.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

import pytest

# Mock the image generation dependencies
mock_numpy = MagicMock()
mock_onnxruntime = MagicMock()
mock_pil = MagicMock()
mock_cv2 = MagicMock()
mock_transformers = MagicMock()
mock_huggingface_hub = MagicMock()

with patch.dict('sys.modules', {
    'numpy': mock_numpy,
    'onnxruntime': mock_onnxruntime,
    'PIL': mock_pil,
    'cv2': mock_cv2,
    'transformers': mock_transformers,
    'huggingface_hub': mock_huggingface_hub,
}):
    # Set HAS_IMAGE_DEPS to True for testing
    from src.image_generation import (
        ImageGenerationStats,
        ONNXStableDiffusionGenerator,
        DialogueTreeIllustrationGenerator,
        generate_illustrations_for_nodes,
        ImageGenerationError,
    )
    # Force HAS_IMAGE_DEPS to True for testing
    import src.image_generation
    src.image_generation.HAS_IMAGE_DEPS = True


class TestImageGenerationStats:
    """Test the image generation statistics tracking."""

    def test_init(self) -> None:
        """Test statistics initialization."""
        stats = ImageGenerationStats()
        
        assert stats.total_images_generated == 0
        assert stats.total_generation_time == 0.0
        assert stats.generation_times == []
        assert stats.failed_nodes == []
        assert stats.start_time > 0

    def test_add_generation(self) -> None:
        """Test adding successful generation statistics."""
        stats = ImageGenerationStats()
        
        stats.add_generation(2.5)
        stats.add_generation(3.1)
        stats.add_generation(1.8)
        
        assert stats.total_images_generated == 3
        assert abs(stats.total_generation_time - 7.4) < 0.001  # Handle floating point precision
        assert stats.generation_times == [2.5, 3.1, 1.8]

    def test_add_failure(self) -> None:
        """Test adding failed generation tracking."""
        stats = ImageGenerationStats()
        
        stats.add_failure("node1")
        stats.add_failure("node2")
        
        assert stats.failed_nodes == ["node1", "node2"]

    def test_mean_generation_time(self) -> None:
        """Test mean generation time calculation."""
        stats = ImageGenerationStats()
        
        # Empty case
        assert stats.mean_generation_time == 0.0
        
        # With data
        stats.add_generation(2.0)
        stats.add_generation(4.0)
        stats.add_generation(3.0)
        
        assert stats.mean_generation_time == 3.0

    def test_min_max_generation_time(self) -> None:
        """Test min/max generation time calculation."""
        stats = ImageGenerationStats()
        
        # Empty case
        assert stats.min_generation_time == 0.0
        assert stats.max_generation_time == 0.0
        
        # With data
        stats.add_generation(2.5)
        stats.add_generation(1.2)
        stats.add_generation(3.8)
        
        assert stats.min_generation_time == 1.2
        assert stats.max_generation_time == 3.8

    def test_throughput_calculation(self) -> None:
        """Test throughput calculation."""
        stats = ImageGenerationStats()
        
        # Mock start time for predictable results
        with patch('time.time') as mock_time:
            mock_time.return_value = 120  # 2 minutes after start
            stats.start_time = 60  # Start time
            
            stats.add_generation(1.0)
            stats.add_generation(2.0)
            
            # 2 images in 60 seconds = 2 images/minute
            assert stats.throughput_images_per_minute == 2.0

    @patch('builtins.print')
    def test_print_statistics(self, mock_print) -> None:
        """Test statistics printing."""
        stats = ImageGenerationStats()
        stats.add_generation(2.5)
        stats.add_failure("node1")
        
        stats.print_statistics()
        
        # Verify print was called with statistics
        assert mock_print.call_count > 0
        printed_text = ' '.join(call[0][0] for call in mock_print.call_args_list)
        assert "Total images generated: 1" in printed_text
        assert "Failed generations: 1" in printed_text


class TestONNXStableDiffusionGenerator:
    """Test the SDXL generator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Mock torch.cuda
        mock_torch.cuda.is_available.return_value = True
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"
        
        # Mock pipeline
        self.mock_pipeline = MagicMock()
        mock_diffusers.StableDiffusionXLPipeline.from_pretrained.return_value = self.mock_pipeline

    def test_init_default_params(self) -> None:
        """Test generator initialization with default parameters."""
        generator = ONNXStableDiffusionGenerator()
        
        assert generator.model_id == "stabilityai/stable-diffusion-xl-base-1.0"
        assert generator.device == "cuda"
        assert generator.torch_dtype == "float16"  # Compare string values
        assert generator.variant == "fp16"
        assert generator.use_safetensors is True
        assert generator.enable_xformers is True
        assert generator.enable_cpu_offload is False
        assert generator.pipeline is None
        assert generator._is_initialized is False

    def test_init_custom_params(self) -> None:
        """Test generator initialization with custom parameters."""
        generator = ONNXStableDiffusionGenerator(
            model_id="custom/model",
            device="cpu",
            torch_dtype=mock_torch.float32,
            variant="base",
            use_safetensors=False,
            enable_xformers=False,
            enable_cpu_offload=True,
        )
        
        assert generator.model_id == "custom/model"
        assert generator.device == "cpu"
        assert generator.torch_dtype == mock_torch.float32
        assert generator.variant == "base"
        assert generator.use_safetensors is False
        assert generator.enable_xformers is False
        assert generator.enable_cpu_offload is True

    def test_initialize_success(self) -> None:
        """Test successful pipeline initialization."""
        generator = ONNXStableDiffusionGenerator()
        
        generator.initialize()
        
        assert generator._is_initialized is True
        mock_diffusers.StableDiffusionXLPipeline.from_pretrained.assert_called_once()
        self.mock_pipeline.to.assert_called_once()
        self.mock_pipeline.enable_xformers_memory_efficient_attention.assert_called_once()

    def test_initialize_cuda_fallback(self) -> None:
        """Test fallback to CPU when CUDA unavailable."""
        mock_torch.cuda.is_available.return_value = False
        
        generator = ONNXStableDiffusionGenerator(device="cuda")
        generator.initialize()
        
        assert generator.device == "cpu"
        assert generator.torch_dtype == mock_torch.float32

    @patch('src.image_generation.logger')
    def test_initialize_xformers_failure(self, mock_logger) -> None:
        """Test graceful handling of xFormers failure."""
        # Mock pipeline and its methods
        self.mock_pipeline = MagicMock()
        mock_diffusers.StableDiffusionXLPipeline.from_pretrained.return_value = self.mock_pipeline
        self.mock_pipeline.enable_xformers_memory_efficient_attention.side_effect = Exception("xFormers error")
        
        generator = ONNXStableDiffusionGenerator()
        generator.initialize()
        
        # Should still initialize successfully
        assert generator._is_initialized is True
        mock_logger.warning.assert_called()

    def test_generate_image_success(self) -> None:
        """Test successful image generation."""
        generator = ONNXStableDiffusionGenerator()
        generator._is_initialized = True
        generator.pipeline = self.mock_pipeline
        
        # Mock successful generation
        mock_result = MagicMock()
        mock_image = MagicMock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [100.0, 102.5]  # 2.5 second generation
            
            images, metadata = generator.generate_image("test prompt")
            
            assert len(images) == 1
            assert images[0] == mock_image
            assert metadata["prompt"] == "test prompt"
            assert metadata["generation_time"] == 2.5
            assert metadata["width"] == 768  # Rounded to nearest 64
            assert metadata["height"] == 768

    def test_generate_image_dimension_rounding(self) -> None:
        """Test image dimension rounding to multiples of 64."""
        generator = ONNXStableDiffusionGenerator()
        generator._is_initialized = True
        generator.pipeline = self.mock_pipeline
        
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        self.mock_pipeline.return_value = mock_result
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [100.0, 101.0]
            
            # Test dimension rounding
            images, metadata = generator.generate_image("test", width=770, height=510)
            
            # Should be rounded to nearest 64
            assert metadata["width"] == 768  # 770 -> 768
            assert metadata["height"] == 448  # 510 -> 448 (nearest 64 multiple)

    def test_generate_image_oom_fallback(self) -> None:
        """Test OOM error handling with resolution fallback.""" 
        generator = ONNXStableDiffusionGenerator()
        generator._is_initialized = True
        generator.pipeline = self.mock_pipeline
        
        # Mock OOM error
        class MockOOMError(Exception):
            pass
        
        mock_torch.cuda.OutOfMemoryError = MockOOMError
        
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        
        self.mock_pipeline.side_effect = [MockOOMError(), mock_result]
        
        with patch('time.time') as mock_time:
            with patch('torch.cuda.empty_cache'):
                mock_time.side_effect = [100.0, 101.0]
                
                images, metadata = generator.generate_image("test", width=768, height=768)
                
                # Should have fallen back to lower resolution
                assert len(images) == 1
                # Second call should use reduced resolution
                assert self.mock_pipeline.call_count == 2

    def test_generate_image_not_initialized(self) -> None:
        """Test image generation when not initialized."""
        generator = ONNXStableDiffusionGenerator()
        
        # Should auto-initialize
        mock_result = MagicMock()
        mock_result.images = [MagicMock()]
        self.mock_pipeline.return_value = mock_result
        
        with patch('time.time') as mock_time:
            mock_time.side_effect = [100.0, 101.0]
            
            images, metadata = generator.generate_image("test")
            
            assert generator._is_initialized is True
            assert len(images) == 1

    def test_generate_image_pipeline_none(self) -> None:
        """Test error when pipeline is None."""
        generator = ONNXStableDiffusionGenerator()
        generator._is_initialized = True
        generator.pipeline = None
        
        with pytest.raises(ImageGenerationError):
            generator.generate_image("test")

    def test_upscale_image_no_upscaler(self) -> None:
        """Test image upscaling when upscaler unavailable."""
        generator = ONNXStableDiffusionGenerator()
        generator.upscaler = None
        
        mock_image = MagicMock()
        result = generator.upscale_image(mock_image)
        
        assert result is None

    def test_cleanup(self) -> None:
        """Test resource cleanup."""
        generator = ONNXStableDiffusionGenerator()
        generator.pipeline = self.mock_pipeline
        generator._is_initialized = True
        
        generator.cleanup()
        
        assert generator.pipeline is None
        assert generator._is_initialized is False


class TestDialogueTreeIllustrationGenerator:
    """Test the dialogue tree illustration generator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_generator = MagicMock()
        self.temp_dir = tempfile.mkdtemp()
        
        self.illustration_gen = DialogueTreeIllustrationGenerator(
            generator=self.mock_generator,
            images_dir=Path(self.temp_dir),
        )

    def test_init(self) -> None:
        """Test illustration generator initialization."""
        generator = MagicMock()
        
        illust_gen = DialogueTreeIllustrationGenerator(generator)
        
        assert illust_gen.generator == generator
        assert illust_gen.images_dir == Path("images")
        assert "photorealistic" in illust_gen.quality_enhancers
        assert "masterpiece" in illust_gen.quality_enhancers
        assert isinstance(illust_gen.stats, ImageGenerationStats)

    def test_build_prompt_complete(self) -> None:
        """Test prompt building with complete context."""
        context = {
            "setting": "medieval castle",
            "time_period": "13th century",
            "atmosphere": "dark and foreboding",
        }
        style_tokens = ["dramatic", "cinematic"]
        
        prompt = self.illustration_gen.build_prompt(
            "The king is dead.", context, style_tokens
        )
        
        assert "The king is dead." in prompt
        assert "medieval castle" in prompt
        assert "13th century" in prompt
        assert "dark and foreboding" in prompt
        assert "dramatic" in prompt
        assert "photorealistic" in prompt
        assert "masterpiece" in prompt

    def test_build_prompt_truncation(self) -> None:
        """Test prompt truncation for very long text."""
        long_text = "A" * 500  # Very long situation text
        context = {"setting": "test"}
        
        prompt = self.illustration_gen.build_prompt(long_text, context)
        
        # Should be truncated but still include quality enhancers
        assert len(prompt) <= 450  # Should be reasonably sized
        assert "photorealistic" in prompt
        assert "masterpiece" in prompt

    def test_find_nodes_without_illustrations_empty(self) -> None:
        """Test BFS with empty nodes."""
        nodes = {}
        result = self.illustration_gen.find_nodes_without_illustrations(nodes)
        assert result == []

    def test_find_nodes_without_illustrations_simple_tree(self) -> None:
        """Test BFS with simple tree structure."""
        nodes = {
            "start": {
                "situation": "Beginning",
                "choices": [{"text": "Go", "next": "middle"}]
            },
            "middle": {
                "situation": "Middle",
                "choices": [{"text": "End", "next": "end"}]
            },
            "end": {
                "situation": "End",
                "choices": []
            }
        }
        
        result = self.illustration_gen.find_nodes_without_illustrations(nodes)
        
        # Should find all nodes in BFS order
        assert result == ["start", "middle", "end"]

    def test_find_nodes_without_illustrations_with_existing(self) -> None:
        """Test BFS skipping nodes that already have illustrations."""
        nodes = {
            "start": {
                "situation": "Beginning",
                "illustration": "images/start/image.png",
                "choices": [{"text": "Go", "next": "middle"}]
            },
            "middle": {
                "situation": "Middle", 
                "choices": [{"text": "End", "next": "end"}]
            },
            "end": {
                "situation": "End",
                "choices": []
            }
        }
        
        result = self.illustration_gen.find_nodes_without_illustrations(nodes)
        
        # Should skip start node since it has illustration
        assert "start" not in result
        assert "middle" in result
        assert "end" in result

    def test_find_nodes_without_illustrations_null_nodes(self) -> None:
        """Test BFS handling null nodes."""
        nodes = {
            "start": {
                "situation": "Beginning",
                "choices": [{"text": "Go", "next": "null_node"}, {"text": "Stay", "next": "end"}]
            },
            "null_node": None,
            "end": {
                "situation": "End",
                "choices": []
            }
        }
        
        result = self.illustration_gen.find_nodes_without_illustrations(nodes)
        
        # Should skip null nodes but include others
        assert "null_node" not in result
        assert "start" in result
        assert "end" in result

    @patch('builtins.open')
    @patch('json.dump')
    @patch('pathlib.Path.cwd')
    def test_save_image_and_metadata(self, mock_cwd, mock_json_dump, mock_open) -> None:
        """Test image and metadata saving."""
        mock_cwd.return_value = Path(self.temp_dir)  # Mock current working directory
        
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        
        metadata = {"prompt": "test", "generation_time": 2.5}
        
        result_path = self.illustration_gen.save_image_and_metadata(
            mock_image, "test_node", metadata
        )
        
        # Verify image was saved
        mock_image.save.assert_called()
        
        # Verify metadata was saved
        mock_json_dump.assert_called_once()
        saved_metadata = mock_json_dump.call_args[0][0]
        assert saved_metadata["prompt"] == "test"
        assert saved_metadata["generation_time"] == 2.5
        
        # Verify return path
        assert "test_node" in result_path
        assert result_path.endswith("illustration.png")

    @patch('builtins.open')
    @patch('json.dump')
    @patch('pathlib.Path.cwd')
    def test_save_with_upscaled_image(self, mock_cwd, mock_json_dump, mock_open) -> None:
        """Test saving with upscaled image."""
        mock_cwd.return_value = Path(self.temp_dir)  # Mock current working directory
        
        mock_image = MagicMock()
        mock_upscaled = MagicMock()
        mock_image.save = MagicMock()
        mock_upscaled.save = MagicMock()
        
        metadata = {"prompt": "test"}
        
        self.illustration_gen.save_image_and_metadata(
            mock_image, "test_node", metadata, mock_upscaled
        )
        
        # Both images should be saved
        mock_image.save.assert_called()
        mock_upscaled.save.assert_called()
        
        # Metadata should include upscaled path
        saved_metadata = mock_json_dump.call_args[0][0]
        assert "upscaled_path" in saved_metadata

    @patch('time.time')
    def test_generate_illustration_success(self, mock_time) -> None:
        """Test successful illustration generation."""
        mock_time.side_effect = [100.0, 102.0]  # 2 second generation
        
        # Mock successful image generation
        mock_image = MagicMock()
        mock_metadata = {"prompt": "test", "generation_time": 2.0}
        self.mock_generator.generate_image.return_value = ([mock_image], mock_metadata)
        self.mock_generator.upscale_image.return_value = None
        
        # Mock file operations
        with patch.object(self.illustration_gen, 'save_image_and_metadata') as mock_save:
            mock_save.return_value = "images/node1/illustration.png"
            
            node_data = {"situation": "Test situation"}
            context = {"setting": "test"}
            
            result = self.illustration_gen.generate_illustration(
                "node1", node_data, context
            )
            
            assert result == "images/node1/illustration.png"
            assert self.illustration_gen.stats.total_images_generated == 1
            
            # Verify generator was called with built prompt
            self.mock_generator.generate_image.assert_called_once()
            call_args = self.mock_generator.generate_image.call_args
            assert "Test situation" in call_args[1]["prompt"]
            assert "test" in call_args[1]["prompt"]

    def test_generate_illustration_failure(self) -> None:
        """Test illustration generation failure handling."""
        # Mock generation failure
        self.mock_generator.generate_image.side_effect = Exception("Generation failed")
        
        node_data = {"situation": "Test"}
        context = {}
        
        result = self.illustration_gen.generate_illustration(
            "node1", node_data, context
        )
        
        assert result is None
        assert "node1" in self.illustration_gen.stats.failed_nodes

    def test_generate_illustration_no_images(self) -> None:
        """Test handling when no images are generated."""
        # Mock empty image result
        self.mock_generator.generate_image.return_value = ([], {})
        
        node_data = {"situation": "Test"}
        context = {}
        
        result = self.illustration_gen.generate_illustration(
            "node1", node_data, context
        )
        
        assert result is None
        assert "node1" in self.illustration_gen.stats.failed_nodes


class TestGenerateIllustrationsForNodes:
    """Test the main illustration generation function."""

    @patch('src.image_generation.HAS_IMAGE_DEPS', True)
    @patch('src.image_generation.ONNXStableDiffusionGenerator')
    @patch('src.image_generation.DialogueTreeIllustrationGenerator')
    def test_generate_illustrations_success(self, mock_illust_gen_class, mock_gen_class) -> None:
        """Test successful illustration generation for multiple nodes."""
        # Mock generator instances
        mock_generator = MagicMock()
        mock_gen_class.return_value = mock_generator
        
        mock_illust_gen = MagicMock()
        mock_illust_gen_class.return_value = mock_illust_gen
        
        # Mock BFS finding nodes
        mock_illust_gen.find_nodes_without_illustrations.return_value = ["node1", "node2"]
        
        # Mock successful generation
        mock_illust_gen.generate_illustration.side_effect = [
            "images/node1/illustration.png",
            "images/node2/illustration.png"
        ]
        
        mock_stats = ImageGenerationStats()
        mock_illust_gen.stats = mock_stats
        
        # Test data
        tree_nodes = {
            "node1": {"situation": "First node"},
            "node2": {"situation": "Second node"}
        }
        context = {"setting": "test"}
        
        generated_count, stats = generate_illustrations_for_nodes(
            tree_nodes, context, max_nodes=5
        )
        
        assert generated_count == 2
        assert stats == mock_stats
        
        # Verify illustrations were added to nodes
        assert tree_nodes["node1"]["illustration"] == "images/node1/illustration.png"
        assert tree_nodes["node2"]["illustration"] == "images/node2/illustration.png"
        
        # Verify cleanup was called
        mock_generator.cleanup.assert_called_once()

    @patch('src.image_generation.HAS_IMAGE_DEPS', False)
    def test_generate_illustrations_no_deps(self) -> None:
        """Test behavior when image dependencies unavailable."""
        tree_nodes = {"node1": {"situation": "Test"}}
        context = {}
        
        generated_count, stats = generate_illustrations_for_nodes(tree_nodes, context)
        
        assert generated_count == 0
        assert isinstance(stats, ImageGenerationStats)