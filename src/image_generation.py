#!/usr/bin/env python3
"""
Image generation module for dialogue nodes using Stable Diffusion XL.

This module provides local GPU-accelerated image generation for dialogue tree nodes
using the Stable Diffusion XL pipeline with xFormers optimization and Real-ESRGAN
upscaling capabilities.
"""

import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    from diffusers import StableDiffusionXLPipeline
    from PIL import Image
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    HAS_IMAGE_DEPS = True
except ImportError as e:
    logger.warning(f"Image generation dependencies not available: {e}")
    HAS_IMAGE_DEPS = False


class ImageGenerationError(Exception):
    """Base exception for image generation operations."""

    pass


class ImageGenerationStats:
    """Tracks image generation performance statistics."""

    def __init__(self) -> None:
        self.total_images_generated = 0
        self.total_generation_time = 0.0
        self.generation_times: List[float] = []
        self.failed_nodes: List[str] = []
        self.start_time = time.time()

    def add_generation(self, generation_time: float) -> None:
        """Record a successful image generation."""
        self.total_images_generated += 1
        self.total_generation_time += generation_time
        self.generation_times.append(generation_time)

    def add_failure(self, node_id: str) -> None:
        """Record a failed image generation."""
        self.failed_nodes.append(node_id)

    @property
    def mean_generation_time(self) -> float:
        """Average generation time per image."""
        if not self.generation_times:
            return 0.0
        return sum(self.generation_times) / len(self.generation_times)

    @property
    def min_generation_time(self) -> float:
        """Fastest generation time."""
        return min(self.generation_times) if self.generation_times else 0.0

    @property
    def max_generation_time(self) -> float:
        """Slowest generation time."""
        return max(self.generation_times) if self.generation_times else 0.0

    @property
    def throughput_images_per_minute(self) -> float:
        """Images generated per minute."""
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0.0
        return (self.total_images_generated / elapsed_time) * 60.0

    def print_statistics(self) -> None:
        """Print comprehensive generation statistics."""
        print("=" * 60)
        print("IMAGE GENERATION STATISTICS")
        print("=" * 60)
        print(f"Total images generated: {self.total_images_generated}")
        print(f"Total generation time: {self.total_generation_time:.2f} seconds")

        if self.generation_times:
            print(f"Mean generation time: {self.mean_generation_time:.2f} seconds")
            print(f"Fastest generation: {self.min_generation_time:.2f} seconds")
            print(f"Slowest generation: {self.max_generation_time:.2f} seconds")
            print(f"Throughput: {self.throughput_images_per_minute:.1f} images/minute")

        if self.failed_nodes:
            print(f"Failed generations: {len(self.failed_nodes)}")
            print(f"Failed node IDs: {', '.join(self.failed_nodes)}")
        else:
            print("Failed generations: 0")

        print("=" * 60)


class StableDiffusionXLGenerator:
    """
    GPU-accelerated Stable Diffusion XL image generator with xFormers optimization.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        torch_dtype: Any = None,  # Will be set to torch.float16 if available
        variant: str = "fp16",
        use_safetensors: bool = True,
        enable_xformers: bool = True,
        enable_cpu_offload: bool = False,
    ) -> None:
        """Initialize the SDXL pipeline with optimization settings."""
        if not HAS_IMAGE_DEPS:
            raise ImageGenerationError(
                "Image generation dependencies not installed. "
                "Please install: torch, diffusers, transformers, accelerate, xformers, pillow, realesrgan"
            )

        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype or torch.float16
        self.variant = variant
        self.use_safetensors = use_safetensors
        self.enable_xformers = enable_xformers
        self.enable_cpu_offload = enable_cpu_offload

        self.pipeline: Optional[StableDiffusionXLPipeline] = None
        self.upscaler: Optional[RealESRGANer] = None
        self._is_initialized = False

        logger.info(f"Initializing SDXL generator with model: {model_id}")
        logger.info(f"Device: {device}, dtype: {torch_dtype}, xFormers: {enable_xformers}")

    def initialize(self) -> None:
        """Load and initialize the SDXL pipeline."""
        if self._is_initialized:
            return

        try:
            logger.info("Loading Stable Diffusion XL pipeline...")
            
            # Check CUDA availability
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
                self.torch_dtype = torch.float32  # CPU doesn't support float16

            # Load pipeline
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                variant=self.variant if self.device == "cuda" else None,
                use_safetensors=self.use_safetensors,
            )

            # Move to device
            self.pipeline = self.pipeline.to(self.device)

            # Enable xFormers for memory efficiency
            if self.enable_xformers and self.device == "cuda":
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xFormers memory efficient attention enabled")
                except Exception as e:
                    logger.warning(f"Could not enable xFormers: {e}")

            # Enable CPU offload for large models
            if self.enable_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
                logger.info("Model CPU offload enabled")

            # Initialize Real-ESRGAN upscaler
            self._initialize_upscaler()

            self._is_initialized = True
            logger.info("SDXL pipeline initialization complete")

        except Exception as e:
            raise ImageGenerationError(f"Failed to initialize SDXL pipeline: {e}")

    def _initialize_upscaler(self) -> None:
        """Initialize Real-ESRGAN upscaler for optional post-processing."""
        try:
            # Use lightweight ESRGAN model for faster processing
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=32,
                upscale=4,
                act_type="prelu",
            )
            
            # Note: In a real implementation, you'd download the pretrained weights
            # For this example, we'll skip the actual upscaler initialization
            # self.upscaler = RealESRGANer(
            #     scale=4,
            #     model_path="path/to/weights",
            #     model=model,
            #     device=self.device,
            # )
            logger.info("Real-ESRGAN upscaler ready (placeholder)")
            
        except Exception as e:
            logger.warning(f"Could not initialize Real-ESRGAN upscaler: {e}")
            self.upscaler = None

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted, deformed",
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Any] = None,  # torch.Generator when available
    ) -> Tuple[List[Any], Dict[str, Any]]:  # Use Any instead of Image.Image
        """
        Generate images using the SDXL pipeline.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt to avoid unwanted features
            width: Image width (should be multiple of 64)
            height: Image height (should be multiple of 64)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            num_images_per_prompt: Number of images to generate
            generator: Random generator for reproducible results

        Returns:
            Tuple of (generated images, metadata dict)
        """
        if not self._is_initialized:
            self.initialize()

        if self.pipeline is None:
            raise ImageGenerationError("Pipeline not initialized")

        # Ensure dimensions are multiples of 64 for SDXL
        width = (width // 64) * 64
        height = (height // 64) * 64

        try:
            start_time = time.time()

            # Handle potential VRAM OOM by reducing resolution if needed
            try:
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                )
                images = result.images

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"CUDA OOM at {width}x{height}, retrying at lower resolution")
                # Reduce resolution and try again
                width = max(512, width // 2)
                height = max(512, height // 2)
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=max(15, num_inference_steps - 10),
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                )
                images = result.images

            generation_time = time.time() - start_time

            # Create metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generation_time": generation_time,
                "model_id": self.model_id,
                "timestamp": datetime.now().isoformat(),
                "seed": generator.initial_seed() if generator else None,
            }

            logger.info(f"Generated {len(images)} image(s) in {generation_time:.2f}s")
            return images, metadata

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise ImageGenerationError(f"Failed to generate image: {e}")

    def upscale_image(self, image: Any) -> Optional[Any]:  # Use Any instead of Image.Image
        """
        Upscale an image using Real-ESRGAN.

        Args:
            image: PIL Image to upscale

        Returns:
            Upscaled image or None if upscaling fails
        """
        if self.upscaler is None:
            logger.warning("Real-ESRGAN upscaler not available")
            return None

        try:
            # Convert PIL to numpy array
            import numpy as np
            img_array = np.array(image)
            
            # Upscale
            upscaled_array, _ = self.upscaler.enhance(img_array, outscale=4)
            
            # Convert back to PIL
            upscaled_image = Image.fromarray(upscaled_array)
            logger.info("Image upscaled successfully")
            return upscaled_image

        except Exception as e:
            logger.warning(f"Image upscaling failed: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up GPU memory and resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_initialized = False
        logger.info("SDXL generator cleaned up")


class DialogueTreeIllustrationGenerator:
    """
    Generates illustrations for dialogue tree nodes using breadth-first search.
    """

    def __init__(
        self,
        generator: StableDiffusionXLGenerator,
        images_dir: Path = Path("images"),
        quality_enhancers: List[str] = None,
    ) -> None:
        """
        Initialize the illustration generator.

        Args:
            generator: SDXL generator instance
            images_dir: Directory to save generated images
            quality_enhancers: Additional prompt tokens for better quality
        """
        self.generator = generator
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
        
        self.quality_enhancers = quality_enhancers or [
            "photorealistic",
            "insane detail",
            "masterpiece",
            "best quality",
            "8k",
            "ultra detailed",
        ]

        self.stats = ImageGenerationStats()

    def build_prompt(
        self,
        node_text: str,
        context: Dict[str, Any],
        style_tokens: List[str] = None,
    ) -> str:
        """
        Build a comprehensive prompt for image generation.

        Args:
            node_text: The situation text from the dialogue node
            context: Context dict with setting, time_period, atmosphere, etc.
            style_tokens: Additional style tokens

        Returns:
            Complete prompt string
        """
        prompt_parts = []

        # Start with the node text (truncated if too long)
        if node_text:
            # Take first sentence or first 100 chars, whichever is shorter
            first_sentence = node_text.split('.')[0] + '.'
            if len(first_sentence) > 100:
                first_sentence = node_text[:97] + "..."
            prompt_parts.append(first_sentence)

        # Add context elements
        if context.get("setting"):
            prompt_parts.append(context["setting"])
        
        if context.get("time_period"):
            prompt_parts.append(context["time_period"])
            
        if context.get("atmosphere"):
            prompt_parts.append(context["atmosphere"])

        # Add custom style tokens
        if style_tokens:
            prompt_parts.extend(style_tokens)

        # Add quality enhancers
        prompt_parts.extend(self.quality_enhancers)

        # Join with commas
        prompt = ", ".join(prompt_parts)
        
        # Ensure prompt isn't too long (SDXL has token limits)
        if len(prompt) > 400:
            # Truncate but keep quality enhancers
            base_prompt = ", ".join(prompt_parts[:-len(self.quality_enhancers)])
            if len(base_prompt) > 300:
                base_prompt = base_prompt[:297] + "..."
            prompt = base_prompt + ", " + ", ".join(self.quality_enhancers)

        return prompt

    def find_nodes_without_illustrations(self, tree_nodes: Dict[str, Any]) -> List[str]:
        """
        Find nodes without illustrations using breadth-first search.

        Args:
            tree_nodes: Dictionary of all tree nodes

        Returns:
            List of node IDs without illustrations, in BFS order
        """
        if not tree_nodes:
            return []

        nodes_without_illustrations = []
        queue = deque()
        visited: Set[str] = set()

        # Find root nodes (nodes not referenced by others)
        referenced_nodes = set()
        for node_data in tree_nodes.values():
            if isinstance(node_data, dict) and "choices" in node_data:
                for choice in node_data["choices"]:
                    if choice.get("next"):
                        referenced_nodes.add(choice["next"])

        # Start BFS from root nodes
        for node_id in tree_nodes.keys():
            if node_id not in referenced_nodes:  # This is a root node
                queue.append(node_id)

        # If no clear root found, start with 'start' or first node
        if not queue:
            if "start" in tree_nodes:
                queue.append("start")
            elif tree_nodes:
                queue.append(next(iter(tree_nodes)))

        # Perform BFS
        while queue:
            current_id = queue.popleft()
            
            if current_id in visited:
                continue
                
            visited.add(current_id)
            current_node = tree_nodes.get(current_id)

            # Skip null nodes
            if current_node is None:
                continue

            # Check if node needs illustration
            if isinstance(current_node, dict):
                if not current_node.get("illustration"):
                    nodes_without_illustrations.append(current_id)

                # Add child nodes to queue
                if "choices" in current_node:
                    for choice in current_node["choices"]:
                        next_node_id = choice.get("next")
                        if next_node_id and next_node_id not in visited:
                            queue.append(next_node_id)

        logger.info(f"Found {len(nodes_without_illustrations)} nodes without illustrations")
        return nodes_without_illustrations

    def save_image_and_metadata(
        self,
        image: Any,  # Use Any instead of Image.Image
        node_id: str,
        metadata: Dict[str, Any],
        upscaled_image: Optional[Any] = None,  # Use Any instead of Image.Image
    ) -> str:
        """
        Save generated image and metadata to disk.

        Args:
            image: Generated image
            node_id: Node ID for directory naming
            metadata: Generation metadata
            upscaled_image: Optional upscaled version

        Returns:
            Relative path to saved image
        """
        # Create node-specific directory
        node_dir = self.images_dir / node_id
        node_dir.mkdir(exist_ok=True)

        # Save main image
        image_filename = "illustration.png"
        image_path = node_dir / image_filename
        image.save(image_path, "PNG")

        # Save upscaled image if available
        if upscaled_image:
            upscaled_path = node_dir / "illustration_upscaled.png"
            upscaled_image.save(upscaled_path, "PNG")
            metadata["upscaled_path"] = str(upscaled_path.relative_to(Path.cwd()))

        # Save metadata
        metadata_path = node_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Return relative path
        relative_path = str(image_path.relative_to(Path.cwd()))
        logger.info(f"Saved illustration for node {node_id}: {relative_path}")
        return relative_path

    def generate_illustration(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        context: Dict[str, Any],
        style_tokens: List[str] = None,
        **generation_kwargs,
    ) -> Optional[str]:
        """
        Generate illustration for a single node.

        Args:
            node_id: Node identifier
            node_data: Node data dictionary
            context: Scene context
            style_tokens: Additional style tokens
            **generation_kwargs: Additional arguments for image generation

        Returns:
            Relative path to generated image or None if generation failed
        """
        try:
            start_time = time.time()

            # Build prompt from node situation and context
            node_text = node_data.get("situation", "")
            prompt = self.build_prompt(node_text, context, style_tokens)

            logger.info(f"Generating illustration for node {node_id}")
            logger.debug(f"Prompt: {prompt}")

            # Generate image
            images, metadata = self.generator.generate_image(
                prompt=prompt,
                **generation_kwargs
            )

            if not images:
                raise ImageGenerationError("No images generated")

            # Use first generated image
            main_image = images[0]

            # Optional upscaling
            upscaled_image = self.generator.upscale_image(main_image)

            # Save image and metadata
            image_path = self.save_image_and_metadata(
                main_image, node_id, metadata, upscaled_image
            )

            # Record success
            generation_time = time.time() - start_time
            self.stats.add_generation(generation_time)

            logger.info(f"Successfully generated illustration for {node_id} in {generation_time:.2f}s")
            return image_path

        except Exception as e:
            logger.error(f"Failed to generate illustration for node {node_id}: {e}")
            self.stats.add_failure(node_id)
            return None


def generate_illustrations_for_nodes(
    tree_nodes: Dict[str, Any],
    context: Dict[str, Any],
    images_dir: Path = Path("images"),
    style_tokens: List[str] = None,
    max_nodes: Optional[int] = None,
    **generation_kwargs,
) -> Tuple[int, ImageGenerationStats]:
    """
    Generate illustrations for all nodes without illustrations using BFS.

    Args:
        tree_nodes: Dictionary of dialogue tree nodes
        context: Scene context with setting, time_period, atmosphere
        images_dir: Directory to save images
        style_tokens: Additional style tokens for prompts
        max_nodes: Maximum number of nodes to process
        **generation_kwargs: Additional arguments for image generation

    Returns:
        Tuple of (number of illustrations generated, statistics)
    """
    if not HAS_IMAGE_DEPS:
        logger.error("Image generation dependencies not available")
        return 0, ImageGenerationStats()

    # Initialize SDXL generator
    generator = StableDiffusionXLGenerator()
    
    # Initialize illustration generator
    illustration_generator = DialogueTreeIllustrationGenerator(
        generator=generator,
        images_dir=images_dir,
    )

    try:
        # Find nodes without illustrations using BFS
        nodes_without_illustrations = illustration_generator.find_nodes_without_illustrations(tree_nodes)
        
        # Limit nodes if specified
        if max_nodes is not None:
            nodes_without_illustrations = nodes_without_illustrations[:max_nodes]

        logger.info(f"Generating illustrations for {len(nodes_without_illustrations)} nodes")

        # Generate illustrations
        generated_count = 0
        for node_id in nodes_without_illustrations:
            node_data = tree_nodes.get(node_id)
            if not isinstance(node_data, dict):
                continue

            # Generate illustration
            image_path = illustration_generator.generate_illustration(
                node_id=node_id,
                node_data=node_data,
                context=context,
                style_tokens=style_tokens,
                **generation_kwargs
            )

            if image_path:
                # Update node with illustration path
                node_data["illustration"] = image_path
                generated_count += 1
                logger.info(f"Added illustration to node {node_id}: {image_path}")

        return generated_count, illustration_generator.stats

    finally:
        # Cleanup GPU resources
        generator.cleanup()