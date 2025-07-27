# Bootstrap Game Dialog Generator

Autonomous Dialogue Tree Completion Script Using a Local LLM

## Overview

This tool reads a JSON file representing a branching dialogue tree for a visual novel/strategy game, continuously identifies incomplete nodes (marked as `null`), uses a locally hosted LLM (via Ollama) to generate content for those nodes, and updates the JSON file until all nodes are filled.

## Features

- **Autonomous Generation**: Automatically fills incomplete dialogue nodes
- **Local LLM Integration**: Uses Ollama for privacy and offline operation
- **Comprehensive Testing**: Fully unit tested with mocked dependencies
- **Backup System**: Creates timestamped backups after each node generation
- **Validation**: Validates generated content structure before saving
- **Logging**: Detailed logging for monitoring progress and debugging
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

1. **Install dependencies:**
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

# Create a sample tree file
python autofill_dialogue.py --create-sample my_tree.json

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
# Install quality tools
pip install black flake8 mypy isort

# Format code
black src/ tests/ autofill_dialogue.py

# Lint code
flake8 src/ tests/ autofill_dialogue.py

# Type checking
mypy src/ --ignore-missing-imports

# Sort imports
isort src/ tests/ autofill_dialogue.py
```

### Project Structure

```
.
├── src/                      # Source code
│   ├── dialogue_tree.py      # Core dialogue tree logic
│   ├── llm_integration.py    # LLM integration and prompt generation
│   └── __init__.py           # Package initialization
├── tests/                    # Unit tests
│   ├── test_dialogue_tree.py
│   ├── test_llm_integration.py
│   └── test_autofill_dialogue.py
├── scripts/                  # Setup scripts
│   ├── setup.sh             # Linux/macOS setup
│   └── setup.ps1            # Windows setup
├── .github/workflows/        # CI/CD workflows
│   └── ci.yml               # GitHub Actions CI
├── autofill_dialogue.py     # Main script
├── tree.json                # Sample dialogue tree
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md               # This file
```

## How It Works

1. **Load Tree**: Reads the JSON dialogue tree file
2. **Find Incomplete Nodes**: Identifies nodes marked as `null`
3. **Generate Context**: Finds the parent node and choice that leads to the incomplete node
4. **Create Prompt**: Builds a detailed prompt for the LLM including context and game parameters
5. **Generate Content**: Uses Ollama to generate new dialogue content
6. **Validate**: Ensures the generated content follows the expected format
7. **Update Tree**: Replaces the `null` node with generated content
8. **Save & Backup**: Saves the updated tree and creates a timestamped backup
9. **Repeat**: Continues until no `null` nodes remain

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
