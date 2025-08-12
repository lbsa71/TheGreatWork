# Bootstrap Game Dialog Generator

Autonomous Dialogue Tree Completion Script Using a Local LLM

## Overview

This tool reads a JSON file representing a branching dialogue tree for a visual novel/strategy game, continuously identifies incomplete nodes (marked as `null`), uses a locally hosted LLM (via Ollama) to generate content for those nodes, and updates the JSON file until all nodes are filled.

## Features

- **Autonomous Generation**: Automatically fills incomplete dialogue nodes
- **Local LLM Integration**: Uses Ollama for privacy and offline operation
- **Local Image Generation**: Uses ONNX Stable Diffusion for Windows-compatible dialogue node illustrations
- **Controlled Generation**: Limit the number of nodes generated with `--max-nodes`
- **Comprehensive Testing**: Fully unit tested with mocked dependencies
- **Backup System**: Creates timestamped backups in a dedicated `/backup` folder
- **Validation**: Validates generated content structure before saving
- **Debug Output**: Detailed logging and debug information for monitoring progress
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Requirements

### Core Requirements
- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- An Ollama model (e.g., `qwen3:14b`, `llama3`, `mistral`)

### Image Generation Requirements (Optional)
- **Windows**: Automatically uses ONNX Runtime with DirectML for GPU acceleration
- **Linux/macOS**: Uses ONNX Runtime with CPU or available GPU providers
- Basic dependencies:
  - `numpy>=1.24.0`
  - `pillow>=10.0.0`
  - `opencv-python>=4.8.0` (optional, for enhanced text overlays)
- Enhanced functionality (optional):
  - `onnxruntime-directml>=1.16.0` (Windows GPU acceleration)
  - `onnxruntime>=1.16.0` (CPU/other platforms)
  - `transformers>=4.34.0` (for advanced models)
  - `huggingface-hub>=0.17.0` (for model downloads)

### Web Application Requirements (Optional)
- Flask (only needed for web interface)
- Install with: `pip install -r web_app/requirements.txt`

## Architecture

This project follows a clean separation of concerns:

- **Core Logic** (`src/`): Business logic for dialogue tree processing, LLM integration, and tree management
- **CLI Application** (`autofill_dialogue.py`): Command-line interface for the core functionality
- **Web Application** (`web_app/`): Dedicated Flask application that uses the core logic as a library
- **Templates & Static Files** (`templates/`, `static/`): Web UI assets

This architecture provides:
- Core functionality that works without web dependencies
- A dedicated web application for interactive tree management
- Clean separation between business logic and presentation
- Independent testing of core logic and web interface

## Quick Start

### Option 1: Automated Setup (Recommended)

**Linux/macOS:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

**Windows (PowerShell):**
```powershell
scripts\setup.ps1
```

### Option 2: Manual Setup

1. **Install dependencies (includes development tools):**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and run Ollama:**
   ```bash
   # Install Ollama (see https://ollama.ai/download)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull a model
   ollama pull qwen3:14b
   
   # Start Ollama server
   ollama serve
   ```

3. **Create a sample dialogue tree:**
   ```bash
   python autofill_dialogue.py --create-sample tree.json
   ```

4. **Run the generator:**
   ```bash
   python autofill_dialogue.py tree.json
   ```

## Usage

### Basic Usage

```bash
python autofill_dialogue.py tree.json
```

### Advanced Options

```bash
# Use a different model
python autofill_dialogue.py tree.json --model llama3

# Enable verbose logging
python autofill_dialogue.py tree.json --verbose

# Limit generation to 10 nodes
python autofill_dialogue.py tree.json --max-nodes 10

# Create a sample tree file
python autofill_dialogue.py --create-sample my_tree.json

# Launch interactive debugger
python autofill_dialogue.py tree.json --debug

# Launch debugger starting from a specific node
python autofill_dialogue.py tree.json --debug --start-node node1

# Launch web application (requires Flask)
python web_app/app.py tree.json

# Launch web app on custom port
python web_app/app.py tree.json --port 8080

# Launch web app on custom host and port
python web_app/app.py tree.json --host 0.0.0.0 --port 8080

# Generate illustrations for dialogue nodes (Windows-compatible)
python autofill_dialogue.py tree.json --generate-images

# Generate images with custom settings
python autofill_dialogue.py tree.json --generate-images --image-width 1024 --image-height 1024 --inference-steps 30

# Show help
python autofill_dialogue.py --help
```

## Image Generation

The tool supports automatic generation of illustrations for dialogue nodes using **ONNX Stable Diffusion** with full Windows compatibility.

### Windows-Compatible Features

- **DirectML Acceleration**: Uses Windows' native DirectML for GPU acceleration on any graphics card
- **CPU Fallback**: Automatically falls back to CPU if GPU is not available
- **No PyTorch Dependencies**: Avoids the notorious Windows compatibility issues with PyTorch, xFormers, and Triton
- **ONNX Runtime**: Uses Microsoft's highly optimized ONNX Runtime for reliable cross-platform operation
- **Placeholder Generation**: Creates visually appealing placeholder images when full models aren't available

### Core Features

- **Breadth-First Search**: Intelligently prioritizes which nodes need illustrations
- **Context-Aware Prompts**: Builds rich prompts from scene context, setting, atmosphere
- **Quality Enhancement**: Adds professional quality tokens for better results
- **Simple Upscaling**: Integrated 2x upscaling using high-quality resampling
- **Performance Tracking**: Detailed statistics on generation time and throughput
- **Error Recovery**: Graceful fallback to placeholder generation if models fail
- **Metadata Preservation**: Saves generation parameters with each image

### Quick Start

```bash
# Generate illustrations for all nodes without images
python autofill_dialogue.py tree.json --generate-images

# Custom image settings
python autofill_dialogue.py tree.json --generate-images \
  --image-width 1024 --image-height 1024 \
  --inference-steps 30 --images-dir my_images

# Generate limited number of images
python autofill_dialogue.py tree.json --generate-images --max-nodes 5
```

### Image Output Structure

Images are saved in organized directories:
```
images/
├── node_id_1/
│   ├── illustration.png        # Main generated image
│   ├── illustration_upscaled.png  # 4x upscaled version (optional)
│   └── metadata.json          # Generation parameters
├── node_id_2/
│   ├── illustration.png
│   └── metadata.json
└── ...
```

### Configuration

The image generation system supports various parameters:

- **Resolution**: `--image-width` and `--image-height` (default: 768x768)
- **Quality**: `--inference-steps` (default: 25, higher = better quality)
- **Output Directory**: `--images-dir` (default: "images")
- **Node Limit**: `--max-nodes` (limits how many images to generate)

### GPU Requirements

- **Recommended**: NVIDIA RTX 4090 (16GB VRAM) or equivalent
- **Minimum**: GTX 1080 Ti (11GB VRAM) or RTX 3060 (12GB VRAM)
- **Fallback**: Automatic resolution reduction if VRAM insufficient

### Performance

On RTX 4090 with xFormers optimization:
- **768x768**: ~2-3 seconds per image
- **1024x1024**: ~4-6 seconds per image  
- **Throughput**: 15-20 images/minute at 768x768

### Troubleshooting

**CUDA Out of Memory**: The system automatically reduces resolution and retries
**No GPU**: Falls back to CPU (very slow, not recommended for production)
**Dependencies**: Install missing packages with `pip install torch torchvision diffusers transformers xformers`

## Web Application

For interactive dialogue tree management, use the dedicated web application:

```bash
# Install web app dependencies
pip install -r web_app/requirements.txt

# Run the web application
python web_app/app.py tree.json

# Run with custom settings
python web_app/app.py tree.json --model qwen3:14b --host 0.0.0.0 --port 8080 --debug
```

The web application provides:
- Interactive tree visualization
- Real-time node generation
- Dialogue history tracking
- Tree structure navigation
- Save/load functionality
```

## Input Format

The dialogue tree JSON file should have this structure:

```json
{
  "rules": {
    "language": "English",
    "tone": "dramatic and serious",
    "voice": "third person narrative",
    "style": "medieval fantasy with political intrigue"
  },
  "scene": {
    "setting": "A medieval kingdom in turmoil",
    "time_period": "Medieval era",
    "location": "The royal castle and court",
    "atmosphere": "Tense and uncertain",
    "key_elements": "Political maneuvering, loyalty conflicts"
  },
  "nodes": {
    "start": {
      "situation": "The king is dead.",
      "choices": [
        { "text": "Mourn publicly", "next": "node1" },
        { "text": "Seize the throne", "next": "node2" }
      ]
    },
    "node1": null,
    "node2": null
  },
  "params": {
    "loyalty": 45,
    "ambition": 80
  }
}
```

### Fields

- **rules** *(optional)*: Stylistic rules for content generation
  - **language**: The language to use (e.g., "English", "French")
  - **tone**: The overall tone (e.g., "dramatic and serious", "lighthearted", "mysterious")
  - **voice**: The narrative voice (e.g., "third person narrative", "first person")
  - **style**: The literary/genre style (e.g., "medieval fantasy", "modern thriller")
- **scene** *(optional)*: World-building context for content generation
  - **setting**: The general setting (e.g., "A medieval kingdom")
  - **time_period**: When the story takes place (e.g., "Medieval era", "1920s")
  - **location**: Specific locations (e.g., "The royal castle")
  - **atmosphere**: The mood/atmosphere (e.g., "Tense and uncertain")
  - **key_elements**: Important story elements (e.g., "Political intrigue, succession crisis")
- **nodes**: Dictionary of dialogue nodes
  - Complete nodes have `situation` and `choices` fields
  - Incomplete nodes are marked as `null`
- **params**: Game parameters that influence generation (e.g., loyalty, ambition)

## Output Format

Generated nodes will have this structure:

```json
{
  "situation": "You decide to mourn the king publicly...",
  "choices": [
    {
      "text": "Organize a grand funeral",
      "next": null,
      "effects": {"loyalty": 10, "ambition": -5}
    },
    {
      "text": "Keep the mourning simple",
      "next": null,
      "effects": {"loyalty": 5, "resources": 5}
    }
  ]
}
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Development tools are already installed via requirements.txt
# Sort imports
isort src/ tests/ autofill_dialogue.py

# Format code
black src/ tests/ autofill_dialogue.py

# Lint code
flake8 src/ tests/ autofill_dialogue.py

# Type checking
mypy src/ --ignore-missing-imports

# Run tests
pytest tests/ -v
```

### Project Structure

```
.
├── src/                      # Source code
│   ├── dialogue_tree.py      # Core dialogue tree logic
│   ├── llm_integration.py    # LLM integration and prompt generation
│   ├── image_generation.py   # SDXL image generation for nodes
│   ├── debugger.py           # Interactive dialogue tree debugger
│   ├── web_ui.py            # Web-based dialogue tree interface
│   └── __init__.py           # Package initialization
├── templates/                # Web UI templates
│   └── index.html           # Main web interface template
├── static/                   # Web UI static files
│   ├── css/                 # Stylesheets
│   │   └── style.css        # Custom styles for web UI
│   └── js/                  # JavaScript files
├── tests/                    # Unit tests
│   ├── test_dialogue_tree.py
│   ├── test_llm_integration.py
│   ├── test_debugger.py
│   └── test_autofill_dialogue.py
├── scripts/                  # Setup scripts
│   ├── setup.sh             # Linux/macOS setup
│   └── setup.ps1            # Windows setup
├── backup/                   # Backup files (auto-created)
│   └── tree_backup_*.json   # Timestamped backups
├── images/                   # Generated illustrations (auto-created)
│   └── node_id/             # Per-node image directories
│       ├── illustration.png # Generated image
│       ├── illustration_upscaled.png # Upscaled version (optional)
│       └── metadata.json   # Generation metadata
├── .github/workflows/        # CI/CD workflows
│   └── ci.yml               # GitHub Actions CI
├── autofill_dialogue.py     # Main script
├── tree.json                # Sample dialogue tree
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md               # This file
```

## Interactive Dialogue Tree Debugger

This tool includes an interactive debugger for navigating and testing dialogue trees in real-time.

### Features

- **Interactive Navigation**: Browse through dialogue nodes using simple keyboard commands
- **History Tracking**: View the dialogue path that led to the current node
- **Real-time Information**: See current node situation, available choices, and effects
- **Quick Navigation**: Jump to specific nodes by ID or navigate by single keypress
- **Cross-platform**: Works on Windows, macOS, and Linux

### Usage

Launch the debugger with:

```bash
python autofill_dialogue.py tree.json --debug
```

Or start from a specific node:

```bash
python autofill_dialogue.py tree.json --debug --start-node node_id
```

### Debugger Controls

- **1-9**: Select a choice by number
- **u**: Go up to the parent node
- **Enter**: Enter a node ID directly to jump to that node
- **q**: Quit the debugger

### Navigation Example

```
=========================================
DIALOGUE TREE DEBUGGER
=========================================

HISTORY:
----------------------------------------
1. Situation: The king is dead.
   Player chose: Mourn publicly

CURRENT NODE: node1
----------------------------------------
📖 You decide to mourn publicly. The court watches your reaction carefully.

AVAILABLE CHOICES:
----------------------------------------
1. Organize a grand funeral
   → Next: node_124
   ⚡ Effects: loyalty:10, resources:-5

2. Keep the mourning simple
   → Next: node_125
   ⚡ Effects: loyalty:5, resources:5

NAVIGATION:
----------------------------------------
1-9     : Choose option by number
u       : Go up (to parent node)
q       : Quit debugger
Enter   : Enter node ID directly

Current game parameters:
  loyalty:50, ambition:30, wisdom:40, charisma:35

Press a key to navigate...
```

The debugger provides a complete view of:
- **Dialogue History**: The path taken to reach the current node
- **Current Situation**: The text and context of the current node
- **Available Choices**: All options with their effects and destinations
- **Game State**: Current parameter values that influence the story

This makes it easy to test dialogue flows, debug branching logic, and understand how choices affect the game state.

## Web UI for Dialogue Tree Navigation

The tool also includes a modern web-based interface for navigating and managing dialogue trees with enhanced visual features.

### Features

- **Visual Tree Navigation**: Browse the dialogue tree structure with an interactive interface
- **Real-time Content Generation**: Generate AI content for incomplete nodes directly from the web interface
- **Interactive Node Selection**: Click on nodes to view their content and make choices
- **History Tracking**: View the dialogue path and navigation history
- **Responsive Design**: Works on desktop and mobile devices
- **Live Updates**: See changes to the tree structure in real-time

### Usage

Launch the web UI with:

```bash
python autofill_dialogue.py tree.json --web
```

Or customize the host and port:

```bash
# Use a different port
python autofill_dialogue.py tree.json --web --port 8080

# Make accessible from other devices on the network
python autofill_dialogue.py tree.json --web --host 0.0.0.0 --port 8080
```

### Web UI Controls

- **Tree Structure Panel**: Shows all nodes in the dialogue tree
  - Green nodes: Complete nodes with content
  - Yellow nodes: Incomplete nodes (null)
  - Click any node to view its content
- **Node Content Panel**: Displays the selected node's situation and choices
- **Generate Button**: For null nodes, click to generate AI content
- **Navigation**: Use the tree structure to navigate between nodes
- **Save**: Automatically saves changes to the tree file

### Web Interface Example

The web UI provides a clean, modern interface with:
- **Left Panel**: Tree structure showing all nodes with visual indicators
- **Right Panel**: Current node content with situation text and available choices
- **Status Indicators**: Visual feedback for node completion status
- **Real-time Generation**: Generate AI content for incomplete nodes with a single click

This web interface makes it easy to:
- **Visualize** the entire dialogue tree structure
- **Navigate** between nodes quickly and intuitively
- **Generate** content for incomplete nodes on-demand
- **Test** dialogue flows in a browser-based environment
- **Share** the interface with team members for collaboration

1. **Load Tree**: Reads the JSON dialogue tree file
2. **Find Incomplete Nodes**: Identifies nodes marked as `null`
3. **Check Limits**: Respects the `--max-nodes` limit if specified
4. **Generate Context**: Finds the parent node and choice that leads to the incomplete node
5. **Create Prompt**: Builds a detailed prompt for the LLM including context and game parameters
6. **Generate Content**: Uses Ollama to generate new dialogue content with debug output
7. **Validate**: Ensures the generated content follows the expected format
8. **Update Tree**: Replaces the `null` node with generated content
9. **Save & Backup**: Saves the updated tree and creates a timestamped backup in `/backup` folder
10. **Repeat**: Continues until no `null` nodes remain or the node limit is reached

## New Features

### Controlled Generation
Use the `--max-nodes` argument to limit the number of nodes generated:

```bash
# Generate only 5 nodes
python autofill_dialogue.py tree.json --max-nodes 5

# Generate 10 nodes with verbose logging
python autofill_dialogue.py tree.json --max-nodes 10 --verbose
```

This is useful for:
- Testing the generation process without running it indefinitely
- Generating content in smaller batches
- Controlling resource usage and generation time

### Debug Output
The application now provides detailed debug information:
- Raw content generated by the LLM
- Extracted JSON content
- Parsing success/failure information
- Detailed logging of the generation process

Use `--verbose` for additional debug information.

### Organized Backups
Backup files are now automatically organized in a `/backup` folder:
- Prevents cluttering of the root directory
- Maintains timestamped backups for easy identification
- Automatically creates the backup directory if it doesn't exist

### Generation Statistics
The application now tracks and displays generation performance statistics:
- **Mean generation time**: Average time per node generation
- **Total generation time**: Cumulative time for all generations
- **Fastest/Slowest generation**: Performance extremes
- **Nodes per minute**: Throughput metric
- **Failed nodes tracking**: Count and list of nodes that failed generation
- **Statistics output**: Displayed on application exit (success or failure)

Example output:
```
============================================================
GENERATION STATISTICS
============================================================
Total nodes generated: 15
Total generation time: 45.23 seconds
Mean generation time: 3.02 seconds
Fastest generation: 1.85 seconds
Slowest generation: 5.67 seconds
Average nodes per minute: 19.9
Failed nodes: 2
Failed node IDs: node1, node2
============================================================
```

### Error Resilience
The application now handles generation failures gracefully:
- **JSON parsing errors**: Failed nodes are skipped and processing continues
- **LLM generation failures**: Individual node failures don't stop the entire process
- **Failed node tracking**: Failed nodes are marked and tracked for review
- **Progress preservation**: Successfully generated nodes are saved even if some fail
- **Comprehensive reporting**: Statistics include both successful and failed generations

## Troubleshooting

### Ollama Not Found
- Ensure Ollama is installed and in your PATH
- Try running `ollama --version` to verify installation

### Model Not Available
- Pull the required model: `ollama pull qwen3:14b`
- Check available models: `ollama list`

### Generation Fails
- Check if Ollama server is running: `ollama serve`
- Verify the model is loaded and accessible
- Check the logs for detailed error messages
- Use `--verbose` flag for additional debug information
- Check the debug output to see what content is being generated

### Failed Nodes
- Failed nodes are marked with `{"__failed__": True, "situation": "Generation failed", "choices": []}`
- Review failed node IDs in the statistics output
- Failed nodes can be manually edited or regenerated later
- The application continues processing other nodes even when some fail

### Tests Failing
- Ensure all dependencies are installed: `pip install pytest pytest-cov pytest-mock`
- Run individual test files to isolate issues

### Web UI Issues
- **Flask not found**: Install Flask with `pip install flask>=2.3.0`
- **Port already in use**: Use a different port with `--port 8080`
- **Cannot access from other devices**: Use `--host 0.0.0.0` to bind to all interfaces
- **Template not found**: Ensure the `templates/` and `static/` directories exist in the project root

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is released under the MIT License. See LICENSE file for details.
