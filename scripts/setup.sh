#!/bin/bash
# Bootstrap Game Dialog Generator - Linux/macOS Setup Script

set -e  # Exit on any error

echo "Bootstrap Game Dialog Generator Setup"
echo "======================================"

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "✓ Python $python_version found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install development dependencies for testing
echo "Installing development dependencies..."
pip install pytest pytest-cov pytest-mock black flake8 mypy

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "WARNING: Ollama is not installed or not in PATH"
    echo "To install Ollama:"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
else
    echo "✓ Ollama found"
    
    # Check if qwen3:14b model is available
    if ollama list | grep -q "qwen3:14b"; then
        echo "✓ qwen3:14b model is available"
    else
        echo ""
        echo "Installing qwen3:14b model for Ollama..."
        echo "This may take several minutes..."
        ollama pull qwen3:14b
        echo "✓ qwen3:14b model installed"
    fi
fi

# Run tests to verify installation
echo ""
echo "Running tests to verify installation..."
pytest tests/ -v

# Create sample tree if it doesn't exist
if [ ! -f "tree.json" ]; then
    echo ""
    echo "Creating sample dialogue tree..."
    python autofill_dialogue.py --create-sample tree.json
    echo "✓ Sample tree created: tree.json"
fi

echo ""
echo "Setup completed successfully!"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Ensure Ollama is running: ollama serve"
echo "  3. Run the script: python autofill_dialogue.py tree.json"
echo ""
echo "For help: python autofill_dialogue.py --help"