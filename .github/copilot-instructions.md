# TheGreatWork - Bootstrap Game Dialog Generator

TheGreatWork is a Python-based AI dialogue tree completion tool that uses local LLMs via Ollama to automatically generate interactive fiction content. The repository includes a CLI tool, web application, static HTML player, and comprehensive test suite.

**Always follow these instructions first. Only search for additional information if these instructions are incomplete or incorrect.**

## Quick Setup & Validation

Bootstrap the development environment:
```bash
# Install Python dependencies (takes ~30 seconds)
pip install -r requirements.txt

# Install Flask for web app functionality (takes ~5 seconds)  
pip install flask>=2.3.0

# Verify installation by running tests (takes ~1.5 seconds - NEVER CANCEL)
pytest tests/ -v

# Create sample dialogue tree for testing (takes ~1 second)
python autofill_dialogue.py --create-sample sample.json
```

**CRITICAL: This is a pure Python application with NO BUILD STEP required. Do not look for Makefiles, build scripts, or compilation processes.**

## Development Workflow

### Code Quality Tools (run before committing)
Run all code quality tools in this exact order:
```bash
# Fix import sorting (~0.1 seconds)
isort src/ tests/ autofill_dialogue.py

# Format code (~0.5 seconds)
black src/ tests/ autofill_dialogue.py

# Lint for errors (~0.3 seconds) - expect style warnings, focus on syntax errors
flake8 src/ tests/ autofill_dialogue.py

# Type checking (~7.5 seconds - NEVER CANCEL)
mypy src/ --ignore-missing-imports

# Run tests after formatting (~1.5 seconds)
pytest tests/ -v
```

**NEVER CANCEL the mypy command - it takes 7.5 seconds but always completes successfully.**

### Testing & Validation
- **Test suite:** `pytest tests/ -v` - Takes 1.5 seconds, 140+ tests, 91% coverage
- **One expected test failure** in `test_process_tree_node_generation_failure` - this is normal
- **All tests use mocks** - no external dependencies required for testing
- **Coverage report** generated in `htmlcov/` directory

## Core Functionality (without Ollama)

You can develop and test most functionality without installing Ollama:

### Interactive Debugger
```bash
# Navigate dialogue trees interactively (works instantly)
python autofill_dialogue.py sample.json --debug

# Start from specific node
python autofill_dialogue.py sample.json --debug --start-node node1
```

### Web Application  
```bash
# Start web interface (starts in ~3 seconds)
python web_app/app.py sample.json --port 5555

# Access at http://127.0.0.1:5555
# Provides tree visualization and navigation (no generation without Ollama)
```

### Static HTML Player
- **File:** `player.html` - fully self-contained, embeddable
- **Usage:** Open directly in browser or serve via `python -m http.server 8000`
- **Features:** Interactive navigation, history tracking, parameter updates
- **Testing verified:** Loads correctly, handles user interactions, displays game state

## LLM Integration (Ollama Required)

**Ollama is REQUIRED for content generation but NOT for development/testing:**

### Ollama Installation (if needed for generation)
```bash
# Linux/macOS - Installation takes several minutes
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download model (takes 10+ minutes for qwen3:14b - NEVER CANCEL)
ollama pull qwen3:14b

# Test generation (requires Ollama running)
python autofill_dialogue.py sample.json --max-nodes 1 --verbose
```

**CRITICAL: Model download takes 10+ minutes. Set timeout to 20+ minutes for `ollama pull` commands.**

## Repository Structure

### Core Components
- **`autofill_dialogue.py`** - Main CLI application entry point
- **`src/`** - Core business logic
  - `dialogue_tree.py` - Tree management and validation  
  - `llm_integration.py` - Ollama client and prompt generation
  - `debugger.py` - Interactive terminal navigation
- **`web_app/`** - Flask web application
  - `app.py` - Web server with REST API
  - `requirements.txt` - Flask dependencies
- **`templates/`** - HTML templates for web UI
- **`static/`** - CSS/JS assets for web UI
- **`tests/`** - Pytest test suite (comprehensive, 91% coverage)

### Key Files
- **`requirements.txt`** - All Python dependencies
- **`pyproject.toml`** - Project configuration and tool settings
- **`player.html`** - Standalone HTML dialogue player
- **`tree.json`** - Main dialogue tree example (complex sci-fi story)
- **`test-tree.json`** - Simpler tree for testing
- **`.github/workflows/ci.yml`** - CI pipeline (Python 3.8-3.11)

## Common Tasks

### Creating New Dialogue Trees
```bash
# Generate sample template
python autofill_dialogue.py --create-sample my_story.json

# Edit the JSON file manually, then test in debugger
python autofill_dialogue.py my_story.json --debug
```

### Debugging Issues
- **Import errors:** Check `sys.path.insert(0, "src")` pattern in main files
- **Flask not found:** Run `pip install flask>=2.3.0`  
- **Ollama connection errors:** Ensure `ollama serve` is running
- **Test failures:** One expected failure is normal; check for new syntax errors

### Manual Validation Scenarios

After making changes, always test these core workflows:

1. **CLI Sample Creation:**
   ```bash
   python autofill_dialogue.py --create-sample test.json
   ls -la test.json  # Should exist with ~800 bytes
   ```

2. **Interactive Debugger:**
   ```bash
   echo "q" | python autofill_dialogue.py test.json --debug
   # Should show dialogue tree interface and exit cleanly
   ```

3. **Web Application:**
   ```bash
   # Start server (background)
   python web_app/app.py test.json --port 5001 &
   sleep 3
   
   # Test endpoints
   curl -s http://127.0.0.1:5001/ | head -5  # Should return HTML
   curl -s http://127.0.0.1:5001/api/tree/structure  # Should return JSON
   
   # Cleanup
   pkill -f "web_app/app.py"
   ```

4. **Static Player:** Open `player.html` in browser - should load dialogue interface with navigation controls

## CI/CD Pipeline

The GitHub Actions workflow tests:
- **Python versions:** 3.8, 3.9, 3.10, 3.11
- **Linting:** flake8 for syntax errors
- **Type checking:** mypy with `--ignore-missing-imports`
- **Testing:** pytest with coverage reporting
- **Code style:** black and isort validation
- **Integration:** Script execution without Ollama

**Build time:** ~5 minutes total across all Python versions

## Known Limitations

- **Ollama required** for actual content generation (not for development)
- **One expected test failure** in autofill dialogue tests
- **Flake8 reports style warnings** but no syntax errors
- **Web UI generation** requires Ollama connection
- **Large model downloads** require stable internet and patience (10+ minutes)

## Troubleshooting

### "No module named 'flask'"
```bash
pip install flask>=2.3.0
```

### "Connection refused" when testing generation
```bash
# Ollama not running
ollama serve

# Model not installed  
ollama pull qwen3:14b
```

### Tests fail after code changes
```bash
# Format code first
black src/ tests/ autofill_dialogue.py

# Check for syntax errors
flake8 --select=E9,F63,F7,F82 src/ tests/ autofill_dialogue.py

# Run specific failing test
pytest tests/test_specific_file.py::TestClass::test_method -v
```

### Web app won't start
```bash
# Check port availability
python web_app/app.py sample.json --port 8080

# Check Flask installation
python -c "import flask; print(flask.__version__)"
```

Always validate changes with the core test scenarios above before committing.