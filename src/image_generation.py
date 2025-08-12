#!/usr/bin/env python3
"""
Image generation module for dialogue nodes using ONNX Runtime and Stable Diffusion.

This module provides Windows-compatible local image generation for dialogue tree nodes
using ONNX Runtime with DirectML acceleration and optimized Stable Diffusion models.
"""

import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import platform

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from PIL import Image as PILImage

    HAS_BASIC_DEPS = True
    
    # Try to import advanced dependencies
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import CLIPTokenizer
        HAS_ONNX_DEPS = True
    except ImportError:
        HAS_ONNX_DEPS = False
        ort = None
        
    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False
        cv2 = None

    HAS_IMAGE_DEPS = HAS_BASIC_DEPS  # Allow basic functionality with just PIL + numpy
    
    # Check for DirectML availability on Windows
    IS_WINDOWS = platform.system() == "Windows"
    
except ImportError as e:
    logger.warning(f"Image generation dependencies not available: {e}")
    HAS_IMAGE_DEPS = False
    HAS_BASIC_DEPS = False
    HAS_ONNX_DEPS = False
    HAS_CV2 = False
    IS_WINDOWS = False
    
    # Create dummy classes to allow module import
    class PILImage:
        Image = None
        
    np = None
    ort = None


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


class ONNXStableDiffusionGenerator:
    """
    Windows-compatible ONNX Runtime Stable Diffusion generator with DirectML acceleration.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "auto",
        use_directml: bool = None,
    ) -> None:
        """
        Initialize the ONNX Stable Diffusion pipeline.
        
        Args:
            model_id: Hugging Face model ID for ONNX Stable Diffusion
            device: Device preference ('auto', 'cpu', 'dml' for DirectML)
            use_directml: Whether to use DirectML on Windows (auto-detected)
        """
        if not HAS_IMAGE_DEPS:
            raise ImageGenerationError(
                "Basic image generation dependencies not installed. "
                "Please install: numpy, pillow. "
                "For full ONNX functionality: onnxruntime-directml (Windows) or onnxruntime, "
                "transformers, huggingface-hub"
            )

        self.model_id = model_id
        
        # Auto-configure optimal settings for Windows
        if use_directml is None:
            use_directml = IS_WINDOWS
        
        self.use_directml = use_directml
        self.device = device
        
        # Configure ONNX Runtime providers
        self.providers = self._configure_providers()
        
        self.text_encoder_session: Optional[ort.InferenceSession] = None
        self.unet_session: Optional[ort.InferenceSession] = None  
        self.vae_decoder_session: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[CLIPTokenizer] = None
        
        self._is_initialized = False
        self._model_cache_dir = Path.home() / ".cache" / "onnx-sd-models"
        self._model_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ONNX SD generator with model: {model_id}")
        logger.info(f"DirectML: {use_directml}, Providers: {self.providers}")

    def _configure_providers(self) -> List[str]:
        """Configure ONNX Runtime execution providers based on platform."""
        if not HAS_ONNX_DEPS or ort is None:
            logger.info("ONNX Runtime not available, will use placeholder generation")
            return ["placeholder"]
            
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX providers: {available_providers}")
        
        if self.device == "cpu":
            return ["CPUExecutionProvider"]
        elif self.device == "dml" and "DmlExecutionProvider" in available_providers:
            return ["DmlExecutionProvider"]
        elif self.use_directml and IS_WINDOWS and "DmlExecutionProvider" in available_providers:
            # DirectML for Windows GPU acceleration
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            # Fallback to CPU
            return ["CPUExecutionProvider"]

    def _download_onnx_model(self) -> Path:
        """
        Download ONNX Stable Diffusion models from Hugging Face.
        
        Returns:
            Path to the downloaded model directory
            
        Raises:
            ImageGenerationError: If model download fails
        """
        logger.debug("Attempting to download ONNX Stable Diffusion models...")
        
        if not HAS_ONNX_DEPS:
            error_msg = (
                "Cannot download models: Hugging Face Hub not available.\n"
                "Install with: pip install huggingface-hub transformers"
            )
            logger.error(error_msg)
            raise ImageGenerationError(error_msg)
            
        # Use a community ONNX conversion or convert from diffusers
        onnx_model_id = "bes-dev/stable-diffusion-v1-4-onnx"
        model_path = self._model_cache_dir / "stable-diffusion-onnx"
        
        logger.debug(f"Model ID: {onnx_model_id}")
        logger.debug(f"Target path: {model_path}")
        
        try:
            if not model_path.exists():
                logger.info(f"Downloading ONNX model '{onnx_model_id}' to {model_path}")
                logger.info("This may take several minutes on first run...")
                
                # Download the ONNX model files
                snapshot_download(
                    repo_id=onnx_model_id,
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,
                )
                logger.info(f"✓ Downloaded ONNX model to {model_path}")
            else:
                logger.info(f"Using cached ONNX model at {model_path}")
                
            # Verify the download was successful
            expected_files = [
                "text_encoder/model.onnx",
                "unet/model.onnx", 
                "vae_decoder/model.onnx"
            ]
            
            missing_files = []
            for file_path in expected_files:
                full_path = model_path / file_path
                if not full_path.exists():
                    missing_files.append(file_path)
                    
            if missing_files:
                error_msg = (
                    f"Model download incomplete. Missing files:\n"
                    + "\n".join(f"  - {f}" for f in missing_files) +
                    f"\nModel directory: {model_path}\n"
                    "This suggests:\n"
                    "  1. Download was interrupted\n"
                    "  2. Repository structure changed\n"
                    "  3. Network/permission issues\n"
                    "To fix:\n"
                    "  1. Delete the model directory and retry\n"
                    "  2. Check internet connection\n"
                    "  3. Try a different model repository"
                )
                logger.error(error_msg)
                raise ImageGenerationError(error_msg)
                
            logger.info("✓ Model download verification passed")
            return model_path
            
        except Exception as e:
            error_msg = (
                f"Failed to download ONNX model '{onnx_model_id}': {e}\n"
                "Debug information:\n"
                f"  - Target directory: {model_path}\n"
                f"  - Cache directory writable: {os.access(self._model_cache_dir, os.W_OK)}\n"
                f"  - Available space: {self._get_available_space(self._model_cache_dir)}\n"
                "Common causes:\n"
                "  1. Network connectivity issues\n"
                "  2. Hugging Face Hub authentication required\n"
                "  3. Insufficient disk space\n"
                "  4. File system permissions\n"
                "  5. Repository moved or deleted\n"
                "To troubleshoot:\n"
                "  1. Check internet connection\n"
                "  2. Verify repository exists: https://huggingface.co/bes-dev/stable-diffusion-v1-4-onnx\n"
                "  3. Free up disk space\n"
                "  4. Check directory permissions\n"
                "  5. Try: huggingface-cli login (if auth required)"
            )
            logger.error(error_msg)
            raise ImageGenerationError(error_msg) from e
            
    def _get_available_space(self, path: Path) -> str:
        """Get available disk space for debugging."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(str(path))
            return f"{free // (1024**3)} GB free"
        except Exception:
            return "Unknown"

    def initialize(self) -> None:
        """
        Load and initialize the ONNX Stable Diffusion pipeline.
        
        Raises:
            ImageGenerationError: If initialization fails
        """
        if self._is_initialized:
            return

        logger.info("Initializing ONNX Stable Diffusion pipeline...")
        logger.debug(f"Model ID: {self.model_id}")
        logger.debug(f"Device: {self.device}")
        logger.debug(f"Use DirectML: {self.use_directml}")
        logger.debug(f"Providers: {self.providers}")
        logger.debug(f"Cache directory: {self._model_cache_dir}")

        try:
            # Check basic dependencies first
            if not HAS_ONNX_DEPS:
                error_msg = (
                    "Required ONNX dependencies are missing:\n"
                    f"  - onnxruntime available: {ort is not None}\n"
                    f"  - transformers available: {'transformers' in globals()}\n"
                    f"  - huggingface_hub available: {'huggingface_hub' in globals()}\n"
                    "Install missing dependencies with:\n"
                    "  pip install onnxruntime-directml transformers huggingface-hub"
                )
                logger.error(error_msg)
                raise ImageGenerationError(error_msg)
            
            # Download/locate ONNX models
            logger.info("Attempting to download/locate ONNX models...")
            model_path = self._download_onnx_model()
            logger.debug(f"Model path: {model_path}")
            
            # Load tokenizer
            logger.debug("Loading CLIP tokenizer...")
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                logger.info("Successfully loaded CLIP tokenizer")
            except Exception as e:
                error_msg = f"Failed to load CLIP tokenizer: {e}"
                logger.error(error_msg)
                raise ImageGenerationError(error_msg) from e
                
            # Try to load ONNX models
            text_encoder_path = model_path / "text_encoder" / "model.onnx"
            unet_path = model_path / "unet" / "model.onnx"
            vae_decoder_path = model_path / "vae_decoder" / "model.onnx"
            
            logger.debug(f"Looking for models at:")
            logger.debug(f"  - Text encoder: {text_encoder_path} (exists: {text_encoder_path.exists()})")
            logger.debug(f"  - UNet: {unet_path} (exists: {unet_path.exists()})")
            logger.debug(f"  - VAE decoder: {vae_decoder_path} (exists: {vae_decoder_path.exists()})")
            
            missing_models = []
            if not text_encoder_path.exists():
                missing_models.append(f"text_encoder ({text_encoder_path})")
            if not unet_path.exists():
                missing_models.append(f"unet ({unet_path})")
            if not vae_decoder_path.exists():
                missing_models.append(f"vae_decoder ({vae_decoder_path})")
                
            if missing_models:
                error_msg = (
                    f"Required ONNX model files are missing:\n"
                    + "\n".join(f"  - {model}" for model in missing_models) +
                    f"\nModel directory: {model_path}\n"
                    "This indicates:\n"
                    "  1. Model download failed\n"
                    "  2. Incorrect model format or repository\n"
                    "  3. File system permissions issue\n"
                    "To fix this:\n"
                    "  1. Check internet connection\n"
                    "  2. Verify model repository exists on Hugging Face\n"
                    "  3. Clear cache directory and retry\n"
                    "  4. Check disk space and permissions"
                )
                logger.error(error_msg)
                raise ImageGenerationError(error_msg)
            
            # Load ONNX inference sessions
            logger.info("Loading ONNX inference sessions...")
            
            try:
                logger.debug(f"Loading text encoder with providers: {self.providers}")
                self.text_encoder_session = ort.InferenceSession(
                    str(text_encoder_path),
                    providers=self.providers
                )
                logger.info("✓ Loaded text encoder ONNX model")
                
                logger.debug(f"Loading UNet with providers: {self.providers}")
                self.unet_session = ort.InferenceSession(
                    str(unet_path),
                    providers=self.providers
                )
                logger.info("✓ Loaded UNet ONNX model")
                
                logger.debug(f"Loading VAE decoder with providers: {self.providers}")
                self.vae_decoder_session = ort.InferenceSession(
                    str(vae_decoder_path),
                    providers=self.providers
                )
                logger.info("✓ Loaded VAE decoder ONNX model")
                
            except Exception as e:
                error_msg = (
                    f"Failed to load ONNX models with ONNX Runtime: {e}\n"
                    f"Debug information:\n"
                    f"  - ONNX Runtime version: {ort.__version__ if ort else 'Not available'}\n"
                    f"  - Available providers: {ort.get_available_providers() if ort else 'Not available'}\n"
                    f"  - Requested providers: {self.providers}\n"
                    f"  - Model files exist: {text_encoder_path.exists()}, {unet_path.exists()}, {vae_decoder_path.exists()}\n"
                    "This could indicate:\n"
                    "  1. ONNX Runtime version incompatibility\n"
                    "  2. Corrupted model files\n"
                    "  3. Unsupported ONNX opset version\n"
                    "  4. Provider (DirectML/CPU) not properly configured\n"
                    "To troubleshoot:\n"
                    "  1. Update ONNX Runtime: pip install -U onnxruntime-directml\n"
                    "  2. Try CPU provider only: device='cpu'\n"
                    "  3. Re-download models (clear cache directory)\n"
                    "  4. Check model compatibility with ONNX Runtime version"
                )
                logger.error(error_msg)
                raise ImageGenerationError(error_msg) from e

            self._is_initialized = True
            logger.info("✓ ONNX SD pipeline initialization complete")
            logger.info("Ready for image generation")

        except ImageGenerationError:
            # Re-raise our own errors
            raise
        except Exception as e:
            error_msg = f"Unexpected error during ONNX SD pipeline initialization: {e}"
            logger.error(error_msg)
            raise ImageGenerationError(error_msg) from e

    # Placeholder image creation method removed - no fallback images allowed

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted, deformed",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Generate images using the ONNX Stable Diffusion pipeline.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt to avoid unwanted features  
            width: Image width (should be multiple of 64)
            height: Image height (should be multiple of 64)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            num_images_per_prompt: Number of images to generate
            seed: Random seed for reproducible results
        
        Returns:
            Tuple of (generated images, metadata dict)
            
        Raises:
            ImageGenerationError: If image generation fails or models are not available
        """
        if not self._is_initialized:
            logger.debug("Generator not initialized, calling initialize()")
            self.initialize()

        # Ensure dimensions are multiples of 64
        width = (width // 64) * 64
        height = (height // 64) * 64
        
        # Ensure minimum size
        width = max(256, width)
        height = max(256, height)
        
        logger.debug(f"Generating image with dimensions: {width}x{height}")
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Negative prompt: {negative_prompt}")
        logger.debug(f"Inference steps: {num_inference_steps}")

        # Check if we have the required dependencies first
        if not HAS_ONNX_DEPS:
            error_msg = (
                "ONNX Runtime dependencies are not available. Cannot generate images.\n"
                "Required dependencies:\n"
                "  - onnxruntime-directml (Windows GPU acceleration)\n"
                "  - onnxruntime (CPU/other platforms)\n"
                "  - transformers (for tokenizer)\n"
                "  - huggingface-hub (for model downloads)\n"
                "Install with: pip install onnxruntime-directml transformers huggingface-hub"
            )
            logger.error(error_msg)
            raise ImageGenerationError(error_msg)

        # Check if we have actual ONNX models loaded
        if (self.text_encoder_session is None or 
            self.unet_session is None or 
            self.vae_decoder_session is None):
            
            error_msg = (
                "ONNX Stable Diffusion models are not loaded. Cannot generate images.\n"
                "Debug information:\n"
                f"  - Text encoder loaded: {self.text_encoder_session is not None}\n"
                f"  - UNet loaded: {self.unet_session is not None}\n"
                f"  - VAE decoder loaded: {self.vae_decoder_session is not None}\n"
                f"  - Model ID: {self.model_id}\n"
                f"  - Cache directory: {self._model_cache_dir}\n"
                f"  - Available providers: {self.providers}\n"
                "This usually means:\n"
                "  1. ONNX models failed to download from Hugging Face\n"
                "  2. Models are not in the expected ONNX format\n"
                "  3. ONNX Runtime failed to load the models\n"
                "To fix this:\n"
                "  1. Check your internet connection for model downloads\n"
                "  2. Verify the model ID exists on Hugging Face\n"
                "  3. Check ONNX Runtime installation\n"
                "  4. Try clearing the model cache directory"
            )
            logger.error(error_msg)
            raise ImageGenerationError(error_msg)

        start_time = time.time()
        
        try:
            # Use actual ONNX Stable Diffusion pipeline
            images = self._run_onnx_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                seed=seed,
            )
            
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
                "seed": seed,
                "backend": "ONNX Runtime",
                "providers": self.providers,
            }

            logger.info(f"Successfully generated {len(images)} image(s) in {generation_time:.2f}s")
            return images, metadata
            
        except Exception as e:
            error_msg = (
                f"ONNX Stable Diffusion pipeline failed: {e}\n"
                "Debug information:\n"
                f"  - Model: {self.model_id}\n"
                f"  - Providers: {self.providers}\n"
                f"  - Prompt length: {len(prompt)} characters\n"
                f"  - Dimensions: {width}x{height}\n"
                f"  - Inference steps: {num_inference_steps}\n"
                "This could indicate:\n"
                "  1. GPU memory insufficient for the requested resolution\n"
                "  2. ONNX model corruption or format issues\n"
                "  3. Provider (DirectML/CPU) compatibility problems\n"
                "  4. Invalid generation parameters\n"
                "To troubleshoot:\n"
                "  1. Try reducing image resolution (e.g., 512x512)\n"
                "  2. Reduce inference steps to 10-15\n"
                "  3. Check GPU memory availability\n"
                "  4. Try CPU provider by setting device='cpu'"
            )
            logger.error(error_msg)
            raise ImageGenerationError(error_msg) from e

    def _run_onnx_pipeline(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int,
        seed: Optional[int],
    ) -> List[Any]:
        """
        Run the actual ONNX Stable Diffusion pipeline.
        
        Raises:
            ImageGenerationError: If the pipeline is not implemented or fails
        """
        error_msg = (
            "Full ONNX Stable Diffusion pipeline is not yet implemented.\n"
            "Current status:\n"
            f"  - Text encoder session: {self.text_encoder_session is not None}\n"
            f"  - UNet session: {self.unet_session is not None}\n"
            f"  - VAE decoder session: {self.vae_decoder_session is not None}\n"
            f"  - Tokenizer: {self.tokenizer is not None}\n"
            f"  - Model cache: {self._model_cache_dir}\n"
            "\n"
            "The ONNX SD pipeline requires:\n"
            "  1. Complete ONNX model files (text_encoder, unet, vae_decoder)\n"
            "  2. Proper tokenizer integration\n"
            "  3. Tensor operations for the diffusion process\n"
            "  4. Scheduler implementation for denoising\n"
            "\n"
            "To implement image generation:\n"
            "  1. Download ONNX SD models from Hugging Face\n"
            "  2. Implement tensor preprocessing/postprocessing\n"
            "  3. Add scheduler (DDPM, DPM-Solver, etc.)\n"
            "  4. Handle text encoding and image decoding\n"
            "\n"
            "Alternative solutions:\n"
            "  1. Use diffusers library with ONNX provider\n"
            "  2. Integrate with Automatic1111 WebUI API\n"
            "  3. Use ComfyUI with ONNX backend\n"
            "  4. Call external image generation service"
        )
        logger.error(error_msg)
        raise ImageGenerationError(error_msg)

    def upscale_image(self, image: Any) -> Optional[Any]:
        """
        Simple upscaling using PIL/OpenCV instead of Real-ESRGAN.
        """
        try:
            # Simple 2x upscale using Lanczos resampling
            width, height = image.size
            if hasattr(PILImage, 'Resampling'):
                upscaled = image.resize((width * 2, height * 2), PILImage.Resampling.LANCZOS)
            else:
                # Fallback for older PIL versions
                upscaled = image.resize((width * 2, height * 2))
            
            logger.info("Image upscaled using Lanczos resampling")
            return upscaled
            
        except Exception as e:
            logger.warning(f"Image upscaling failed: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.text_encoder_session:
            del self.text_encoder_session
            self.text_encoder_session = None
            
        if self.unet_session:
            del self.unet_session  
            self.unet_session = None
            
        if self.vae_decoder_session:
            del self.vae_decoder_session
            self.vae_decoder_session = None
            
        self._is_initialized = False
        logger.info("ONNX SD generator cleaned up")


class DialogueTreeIllustrationGenerator:
    """
    Generates illustrations for dialogue tree nodes using breadth-first search.
    """

    def __init__(
        self,
        generator: ONNXStableDiffusionGenerator,
        images_dir: Path = Path("images"),
        quality_enhancers: List[str] = None,
    ) -> None:
        """
        Initialize the illustration generator.

        Args:
            generator: ONNX SD generator instance
            images_dir: Directory to save generated images
            quality_enhancers: Additional prompt tokens for better quality
        """
        self.generator = generator
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(exist_ok=True)
        
        self.quality_enhancers = quality_enhancers or [
            "photorealistic",
            "highly detailed",
            "masterpiece",
            "best quality",
            "8k resolution",
            "professional illustration",
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
        image: Any,
        node_id: str,
        metadata: Dict[str, Any],
        upscaled_image: Optional[Any] = None,
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
            # Use relative path from images directory, not from cwd
            metadata["upscaled_path"] = str(upscaled_path.relative_to(self.images_dir.parent))

        # Save metadata
        metadata_path = node_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Return relative path from current working directory
        try:
            relative_path = str(image_path.relative_to(Path.cwd()))
        except ValueError:
            # If relative path fails, use path relative to images directory
            relative_path = str(image_path.relative_to(self.images_dir.parent))
            
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
            
        Raises:
            ImageGenerationError: If generation fails with detailed error message
        """
        start_time = time.time()

        # Build prompt from node situation and context
        node_text = node_data.get("situation", "")
        if not node_text.strip():
            error_msg = f"Node {node_id} has no situation text to generate image from"
            logger.error(error_msg)
            raise ImageGenerationError(error_msg)
            
        prompt = self.build_prompt(node_text, context, style_tokens)

        logger.info(f"Generating illustration for node {node_id}")
        logger.debug(f"Node situation: {node_text}")
        logger.debug(f"Built prompt: {prompt}")
        logger.debug(f"Generation parameters: {generation_kwargs}")

        try:
            # Generate image
            images, metadata = self.generator.generate_image(
                prompt=prompt,
                **generation_kwargs
            )

            if not images:
                error_msg = f"Image generation returned no images for node {node_id}"
                logger.error(error_msg)
                raise ImageGenerationError(error_msg)

            # Use first generated image
            main_image = images[0]

            # Optional upscaling
            upscaled_image = None
            try:
                upscaled_image = self.generator.upscale_image(main_image)
                if upscaled_image:
                    logger.debug(f"Successfully upscaled image for node {node_id}")
                else:
                    logger.debug(f"Upscaling skipped for node {node_id}")
            except Exception as e:
                logger.warning(f"Image upscaling failed for node {node_id}: {e}")
                # Continue without upscaling

            # Save image and metadata
            try:
                image_path = self.save_image_and_metadata(
                    main_image, node_id, metadata, upscaled_image
                )
            except Exception as e:
                error_msg = f"Failed to save image for node {node_id}: {e}"
                logger.error(error_msg)
                raise ImageGenerationError(error_msg) from e

            # Record success
            generation_time = time.time() - start_time
            self.stats.add_generation(generation_time)

            logger.info(f"✓ Successfully generated illustration for {node_id} in {generation_time:.2f}s")
            logger.info(f"✓ Saved to: {image_path}")
            return image_path

        except ImageGenerationError:
            # Re-raise our own errors with node context
            generation_time = time.time() - start_time
            self.stats.add_failure(node_id)
            raise
        except Exception as e:
            # Wrap unexpected errors
            generation_time = time.time() - start_time
            self.stats.add_failure(node_id)
            error_msg = (
                f"Unexpected error generating illustration for node {node_id}: {e}\n"
                f"Generation time before failure: {generation_time:.2f}s\n"
                f"Prompt length: {len(prompt)} characters\n"
                f"Parameters: {generation_kwargs}"
            )
            logger.error(error_msg)
            raise ImageGenerationError(error_msg) from e


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
        
    Raises:
        ImageGenerationError: If image generation setup or processing fails
    """
    if not HAS_IMAGE_DEPS:
        error_msg = (
            "Image generation dependencies not available.\n"
            "Required dependencies:\n"
            "  - numpy>=1.24.0\n"
            "  - pillow>=10.0.0\n"
            "Optional dependencies for full functionality:\n"
            "  - onnxruntime-directml>=1.16.0 (Windows GPU acceleration)\n"
            "  - onnxruntime>=1.16.0 (CPU/other platforms)\n"
            "  - transformers>=4.34.0 (for tokenizer)\n"
            "  - huggingface-hub>=0.17.0 (for model downloads)\n"
            "  - opencv-python>=4.8.0 (for enhanced image processing)\n"
            "Install with: pip install numpy pillow onnxruntime-directml transformers huggingface-hub opencv-python"
        )
        logger.error(error_msg)
        raise ImageGenerationError(error_msg)

    # Initialize ONNX generator - this will raise detailed errors if it fails
    logger.info("Initializing ONNX Stable Diffusion generator...")
    generator = ONNXStableDiffusionGenerator()
    
    # Initialize illustration generator
    illustration_generator = DialogueTreeIllustrationGenerator(
        generator=generator,
        images_dir=images_dir,
    )

    try:
        # Initialize the generator (downloads models, loads sessions, etc.)
        logger.info("Preparing image generation pipeline...")
        generator.initialize()
        
        # Find nodes without illustrations using BFS
        logger.info("Discovering nodes that need illustrations...")
        nodes_without_illustrations = illustration_generator.find_nodes_without_illustrations(tree_nodes)
        
        if not nodes_without_illustrations:
            logger.info("No nodes found that need illustrations")
            return 0, illustration_generator.stats
        
        # Limit nodes if specified
        if max_nodes is not None and len(nodes_without_illustrations) > max_nodes:
            logger.info(f"Limiting generation to first {max_nodes} nodes (found {len(nodes_without_illustrations)} total)")
            nodes_without_illustrations = nodes_without_illustrations[:max_nodes]

        logger.info(f"Generating illustrations for {len(nodes_without_illustrations)} nodes:")
        for i, node_id in enumerate(nodes_without_illustrations, 1):
            logger.info(f"  {i}. {node_id}")

        # Generate illustrations
        generated_count = 0
        for i, node_id in enumerate(nodes_without_illustrations, 1):
            node_data = tree_nodes.get(node_id)
            if not isinstance(node_data, dict):
                logger.warning(f"Skipping node {node_id}: not a valid node dictionary")
                continue

            logger.info(f"[{i}/{len(nodes_without_illustrations)}] Processing node: {node_id}")

            try:
                # Generate illustration - this will raise ImageGenerationError with details
                image_path = illustration_generator.generate_illustration(
                    node_id=node_id,
                    node_data=node_data,
                    context=context,
                    style_tokens=style_tokens,
                    **generation_kwargs
                )

                # Update node with illustration path
                node_data["illustration"] = image_path
                generated_count += 1
                logger.info(f"✓ [{i}/{len(nodes_without_illustrations)}] Added illustration to node {node_id}: {image_path}")

            except ImageGenerationError as e:
                # Log the detailed error but continue with other nodes
                logger.error(f"✗ [{i}/{len(nodes_without_illustrations)}] Failed to generate illustration for node {node_id}:")
                logger.error(f"    {str(e)}")
                # Don't raise here - continue processing other nodes
                continue
            except Exception as e:
                # Handle any unexpected errors
                error_msg = f"Unexpected error processing node {node_id}: {e}"
                logger.error(f"✗ [{i}/{len(nodes_without_illustrations)}] {error_msg}")
                illustration_generator.stats.add_failure(node_id)
                # Don't raise here - continue processing other nodes
                continue

        # Summary
        total_requested = len(nodes_without_illustrations)
        success_rate = (generated_count / total_requested * 100) if total_requested > 0 else 0
        
        logger.info("=" * 60)
        logger.info("IMAGE GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Successfully generated: {generated_count}/{total_requested} images ({success_rate:.1f}%)")
        if illustration_generator.stats.failed_nodes:
            logger.info(f"Failed generations: {len(illustration_generator.stats.failed_nodes)}")
            logger.info(f"Failed node IDs: {', '.join(illustration_generator.stats.failed_nodes)}")
        
        # If no images were generated successfully, this might indicate a systemic issue
        if generated_count == 0 and total_requested > 0:
            error_msg = (
                f"Failed to generate any images out of {total_requested} attempts.\n"
                "This indicates a systematic issue with the image generation setup.\n"
                "Check the error messages above for details on what went wrong."
            )
            logger.error(error_msg)
            raise ImageGenerationError(error_msg)

        return generated_count, illustration_generator.stats

    finally:
        # Cleanup resources
        try:
            generator.cleanup()
            logger.debug("Image generator resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during generator cleanup: {e}")