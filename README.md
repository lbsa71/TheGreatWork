# Bootstrap Game Dialog Generator

Autonomous Dialogue Tree Completion Script Using a Local LLM

## Overview

This tool reads a JSON file representing a branching dialogue tree for a visual novel/strategy game, continuously identifies incomplete nodes (marked as `null`), uses a locally hosted LLM (via Ollama) to generate content for those nodes, and updates the JSON file until all nodes are filled.

## Features

- **Autonomous Generation**: Automatically fills incomplete dialogue nodes
- **Local LLM Integration**: Uses Ollama for privacy and offline operation
- **Controlled Generation**: Limit the number of nodes generated with `--max-nodes`
- **Comprehensive Testing**: Fully unit tested with mocked dependencies
- **Backup System**: Creates timestamped backups in a dedicated `/backup` folder
- **Validation**: Validates generated content structure before saving
- **Debug Output**: Detailed logging and debug information for monitoring progress
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Requirements

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- An Ollama model (e.g., `llama3`, `mistral`)

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
   ollama pull llama3
   
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
python autofill_dialogue.py tree.json --model mistral

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

# Show help
python autofill_dialogue.py --help
```

## Input Format

The dialogue tree JSON file should have this structure:

```json
{
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
│   ├── debugger.py           # Interactive dialogue tree debugger
│   └── __init__.py           # Package initialization
├── tests/                    # Unit tests
│   ├── test_dialogue_tree.py
│   ├── test_llm_integration.py
│   └── test_autofill_dialogue.py
├── scripts/                  # Setup scripts
│   ├── setup.sh             # Linux/macOS setup
│   └── setup.ps1            # Windows setup
├── backup/                   # Backup files (auto-created)
│   └── tree_backup_*.json   # Timestamped backups
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

## Troubleshooting

### Ollama Not Found
- Ensure Ollama is installed and in your PATH
- Try running `ollama --version` to verify installation

### Model Not Available
- Pull the required model: `ollama pull llama3`
- Check available models: `ollama list`

### Generation Fails
- Check if Ollama server is running: `ollama serve`
- Verify the model is loaded and accessible
- Check the logs for detailed error messages
- Use `--verbose` flag for additional debug information
- Check the debug output to see what content is being generated

### Tests Failing
- Ensure all dependencies are installed: `pip install pytest pytest-cov pytest-mock`
- Run individual test files to isolate issues

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
