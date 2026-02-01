#!/bin/bash
# Quick setup script using UV

set -e

echo "🚀 MedAssist Setup with UV"
echo "=========================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✅ UV installed successfully!"
else
    echo "✅ UV already installed"
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
uv venv

# Activate venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Install dependencies
echo ""
echo "⚡ Installing dependencies (this will be fast!)..."
uv pip install -e .

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source .venv/bin/activate  (Linux/Mac)"
echo "     .venv\\Scripts\\activate  (Windows)"
echo ""
echo "  2. Run the demo:"
echo "     python app.py"
echo ""
echo "  3. Or run examples:"
echo "     python examples.py"
