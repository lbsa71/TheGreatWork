# Bootstrap Game Dialog Generator

Autonomous Dialogue Tree Completion Script Using a Local LLM

## Overview

This tool reads a JSON file representing a branching dialogue tree for a visual novel/strategy game, continuously identifies incomplete nodes (marked as `null`), uses a locally hosted LLM (via Ollama) to generate content for those nodes, and updates the JSON file until all nodes are filled.

## Features

- **Autonomous Generation**: Automatically fills incomplete dialogue nodes
- **Local LLM Integration**: Uses Ollama for privacy and offline operation
- **Web UI**: Interactive web interface for point-and-click dialogue tree management
- **Console Mode**: Traditional command-line interface for batch processing
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

### Web UI Mode (Recommended)

Start the interactive web interface:

```bash
# Start web UI on default host and port (127.0.0.1:5000)
python autofill_dialogue.py --web-ui

# Start on custom host and port
python autofill_dialogue.py --web-ui --host 0.0.0.0 --port 8080

# Use a different model with web UI
python autofill_dialogue.py --web-ui --model mistral
```

The web UI provides:
- **Tree Navigator**: Visual overview of all nodes with completion status
- **Point-and-Click Navigation**: Click any node to view its details
- **One-Click Generation**: Generate AI content for null nodes with a button
- **Real-time Updates**: Automatically refreshes tree structure after generation
- **Interactive Interface**: Better UX than console with immediate feedback

### Console Mode

For batch processing or automated workflows:

### Basic Console Usage

```bash
python autofill_dialogue.py tree.json
```

### Advanced Console Options

```bash
# Use a different model
python autofill_dialogue.py tree.json --model mistral

# Enable verbose logging
python autofill_dialogue.py tree.json --verbose

# Limit generation to 10 nodes
python autofill_dialogue.py tree.json --max-nodes 10

# Create a sample tree file
python autofill_dialogue.py --create-sample my_tree.json

# Show help
python autofill_dialogue.py --help
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

## How It Works

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
