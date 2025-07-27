# Bootstrap Game Dialog Generator - Windows Setup Script

Write-Host "Bootstrap Game Dialog Generator Setup" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# Check if Python 3.8+ is installed
try {
    $pythonVersion = & python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            Write-Host "Error: Python 3.8 or higher is required. Found: $pythonVersion" -ForegroundColor Red
            Write-Host "Please install Python 3.8+ from https://python.org and try again." -ForegroundColor Red
            exit 1
        }
        Write-Host "✓ $pythonVersion found" -ForegroundColor Green
    }
} catch {
    Write-Host "Error: Python not found. Please install Python 3.8+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install development dependencies for testing
Write-Host "Installing development dependencies..." -ForegroundColor Yellow
pip install pytest pytest-cov pytest-mock black flake8 mypy

# Check if Ollama is installed
try {
    $ollamaVersion = & ollama --version 2>&1
    Write-Host "✓ Ollama found" -ForegroundColor Green
    
    # Check if llama3 model is available
    $models = & ollama list 2>&1
    if ($models -match "llama3") {
        Write-Host "✓ llama3 model is available" -ForegroundColor Green
    } else {
        Write-Host "" 
        Write-Host "Installing llama3 model for Ollama..." -ForegroundColor Yellow
        Write-Host "This may take several minutes..." -ForegroundColor Yellow
        ollama pull llama3
        Write-Host "✓ llama3 model installed" -ForegroundColor Green
    }
} catch {
    Write-Host ""
    Write-Host "WARNING: Ollama is not installed or not in PATH" -ForegroundColor Yellow
    Write-Host "To install Ollama on Windows:" -ForegroundColor Yellow
    Write-Host "  1. Download from https://ollama.ai/download/windows" -ForegroundColor Yellow
    Write-Host "  2. Run the installer" -ForegroundColor Yellow
    Write-Host "  3. Restart your terminal" -ForegroundColor Yellow
    Write-Host ""
}

# Run tests to verify installation
Write-Host ""
Write-Host "Running tests to verify installation..." -ForegroundColor Yellow
pytest tests/ -v

# Create sample tree if it doesn't exist
if (-not (Test-Path "tree.json")) {
    Write-Host ""
    Write-Host "Creating sample dialogue tree..." -ForegroundColor Yellow
    python autofill_dialogue.py --create-sample tree.json
    Write-Host "✓ Sample tree created: tree.json" -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "To get started:" -ForegroundColor Cyan
Write-Host "  1. Activate the virtual environment: venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "  2. Ensure Ollama is running: ollama serve" -ForegroundColor Cyan
Write-Host "  3. Run the script: python autofill_dialogue.py tree.json" -ForegroundColor Cyan
Write-Host ""
Write-Host "For help: python autofill_dialogue.py --help" -ForegroundColor Cyan