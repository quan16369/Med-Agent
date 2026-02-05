# MedGemma - Production Medical Multi-Agent System

🏥 **LangGraph-based Medical QA with Multi-Agent Orchestration**

## 📁 Production Structure

```
MedGemma/
├── medassist/
│   ├── models/          # Data models (KG entities, multimodal)
│   ├── agents/          # Specialized agents
│   ├── tools/           # Tools (NER, PubMed, KG)
│   ├── core/            # LangGraph orchestration
│   ├── services/        # Business logic
│   ├── schemas/         # Pydantic schemas
│   └── api/             # FastAPI endpoints
├── simple_agent.py      # Simple interface
├── demo_interface.py    # Gradio UI
└── requirements.txt
```

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt
export GROQ_API_KEY="your-key"

# Simple usage
python simple_agent.py

# API server
uvicorn medassist.api.main:app --reload --port 8000

# Web demo
python demo_interface.py
```

## 🏗️ LangGraph Architecture

```
Orchestrator → Knowledge Agent → Diagnostic Agent → Treatment Agent → Evidence Agent → Validator
```

## 📚 API Endpoints

- `POST /query` - Medical question
- `POST /ingest` - Document ingestion  
- `POST /kg/explore` - Knowledge graph
- `GET /health` - Health check

See full docs at `/docs`

## 📦 Components

**Models** - Data structures
**Agents** - Specialized medical agents  
**Tools** - NER, PubMed, KG retrieval
**Core** - LangGraph orchestration
**Services** - Business logic
**API** - FastAPI endpoints

## 🧪 Test

```bash
python test_langgraph.py
pytest tests/
```

Built with LangGraph + FastAPI + BioBERT
