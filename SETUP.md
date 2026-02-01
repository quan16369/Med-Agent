# Setup Guide with UV

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package management.

## 🚀 Quick Start with UV

### 1. Install UV (if not already installed)

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 2. Setup Project

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MedGemma-Agentic-Workflow.git
cd MedGemma

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (super fast!)
uv pip install -e .
```

### 3. Run the Application

```bash
# Run Gradio demo
python app.py

# Run examples
python examples.py

# Run tests
pytest tests/
```

## 📦 UV Commands Reference

```bash
# Install dependencies
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"

# Add a new dependency
uv pip install package-name
# Then add to pyproject.toml dependencies

# Update all dependencies
uv pip install --upgrade -e .

# Sync dependencies (install/uninstall to match pyproject.toml)
uv pip sync

# Create new venv
uv venv

# Run command in venv without activating
uv run python app.py
```

## 🔧 Development Setup

```bash
# Install with dev tools
uv pip install -e ".[dev]"

# Format code
black medassist/ app.py examples.py

# Lint code
ruff check medassist/ app.py examples.py

# Type check
mypy medassist/

# Run tests with coverage
pytest --cov=medassist tests/
```

## ⚡ Why UV?

### Speed Comparison
```
pip install:  ~120 seconds
poetry install: ~90 seconds
uv pip install: ~5 seconds  ← 20x faster!
```

### Benefits
- **Super Fast**: 10-100x faster than pip
- **Reliable**: Better dependency resolution
- **Compatible**: Works with existing pip/poetry projects
- **Modern**: Built with Rust, actively maintained

## 🐍 Python Version

Project requires Python 3.10+. UV will automatically use `.python-version` file.

```bash
# Check Python version
uv run python --version
```

## 📋 Alternative Setup (without UV)

If you prefer traditional pip:

```bash
# Create venv
python -m venv venv
source venv/bin/activate

# Install from requirements.txt
pip install -r requirements.txt

# Or from pyproject.toml
pip install -e .
```

## 🚢 Docker Setup

UV is also included in Docker for fast builds:

```bash
# Build Docker image
docker build -t medassist .

# Run container
docker run -p 7860:7860 medassist
```

## 🔍 Troubleshooting

### UV not found
```bash
# Make sure UV is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### CUDA/PyTorch Issues
```bash
# For CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU only
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Dependency Conflicts
```bash
# Force reinstall
uv pip install --force-reinstall -e .

# Clear cache
uv cache clean
```

## 📖 More Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [UV vs pip/poetry comparison](https://github.com/astral-sh/uv#benchmarks)
- [Python packaging guide](https://packaging.python.org/)
