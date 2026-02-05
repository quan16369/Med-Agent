#!/bin/bash
# Production API Server Startup Script

set -e

echo "Starting MedGemma API Server..."

# Check environment
if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ_API_KEY not set"
    echo "Export your API key: export GROQ_API_KEY='your-key'"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import fastapi, langgraph, langchain" 2>/dev/null || {
    echo "Missing dependencies. Installing..."
    pip install -q -r requirements.txt
}

# Start server
echo "Environment ready"
echo "Starting API server on http://0.0.0.0:8000"
echo "API docs: http://localhost:8000/docs"
echo ""

uvicorn medassist.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
