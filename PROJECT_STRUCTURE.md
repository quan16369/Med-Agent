# Project Structure

## Directory Organization

```
MedGemma/
│
├── medassist/              # Core Python package
│   ├── __init__.py
│   ├── config.py           # Environment-based configuration
│   ├── exceptions.py       # Custom exception hierarchy
│   ├── logging_utils.py    # Structured logging utilities
│   ├── health.py           # Health check system
│   ├── knowledge_graph.py  # Medical knowledge graph (Neo4j)
│   ├── graph_retrieval.py  # Graph traversal algorithms (BFS/DFS)
│   ├── medical_ner.py      # BioBERT medical entity recognition
│   ├── pubmed_retrieval.py # PubMed API integration
│   ├── agentic_workflow.py # Multi-agent collaboration framework
│   ├── agentic_orchestrator.py # Main orchestrator
│   └── amg_rag_orchestrator.py # Legacy orchestrator
│
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_units.py       # Unit tests for components
│   ├── test_integration.py # End-to-end integration tests
│   └── test_api.py         # API endpoint tests
│
├── examples/               # Demo and example scripts
│   ├── __init__.py
│   ├── demo_agentic.py     # Multi-agent workflow demo
│   ├── demo_amg_rag.py     # Knowledge graph reasoning demo
│   └── test_components.py  # Quick component testing
│
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # System architecture
│   ├── DEPLOYMENT.md       # Production deployment guide
│   ├── SETUP.md            # Installation and setup
│   └── REFACTORING.md      # Development history
│
├── k8s/                    # Kubernetes deployment
│   └── deployment.yaml     # K8s configuration (namespace, services, deployments)
│
├── .github/                # GitHub Actions CI/CD
│   └── workflows/
│       └── ci-cd.yml       # Automated testing and deployment
│
├── scripts/                # Utility scripts (future use)
│
├── api.py                  # FastAPI application (main entry point)
├── run_server.py           # Production server launcher
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container orchestration (app + Neo4j)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
├── README.md               # Main documentation
└── PROJECT_STRUCTURE.md    # This file
```

## Component Descriptions

### Core Package (`medassist/`)

**Configuration & Infrastructure:**
- `config.py` - Environment-based configuration with dataclasses
- `exceptions.py` - Custom exception hierarchy and error handlers
- `logging_utils.py` - Rotating file handler and structured logging
- `health.py` - Kubernetes-compatible health checks (liveness, readiness, startup)

**Medical AI Components:**
- `knowledge_graph.py` - Medical knowledge graph with Neo4j backend
- `graph_retrieval.py` - BFS/DFS traversal with confidence scoring
- `medical_ner.py` - BioBERT-based medical entity extraction
- `pubmed_retrieval.py` - Scientific literature retrieval and caching

**Agentic System:**
- `agentic_workflow.py` - 5 specialized agents (Knowledge, Diagnostic, Treatment, Evidence, Validator)
- `agentic_orchestrator.py` - Main orchestrator for multi-agent collaboration
- `amg_rag_orchestrator.py` - Legacy orchestrator (kept for compatibility)

### Tests (`tests/`)

- `test_units.py` - Unit tests for individual components (NER, retrieval, graph)
- `test_integration.py` - End-to-end workflow tests
- `test_api.py` - FastAPI endpoint tests with rate limiting

### Examples (`examples/`)

- `demo_agentic.py` - Demonstrates multi-agent workflow execution
- `demo_amg_rag.py` - Shows knowledge graph reasoning capabilities
- `test_components.py` - Quick smoke tests for all components

### Documentation (`docs/`)

- `ARCHITECTURE.md` - Detailed system design and component interactions
- `DEPLOYMENT.md` - Production deployment (Docker, Kubernetes, monitoring)
- `SETUP.md` - Installation guide and configuration
- `REFACTORING.md` - Development history and changes

### Infrastructure

**Containerization:**
- `Dockerfile` - Production-ready container with health checks
- `docker-compose.yml` - Multi-container setup (app + Neo4j database)

**Kubernetes:**
- `k8s/deployment.yaml` - Full K8s configuration with auto-scaling

**CI/CD:**
- `.github/workflows/ci-cd.yml` - Automated testing, security scans, Docker builds

**Entry Points:**
- `api.py` - FastAPI REST API server
- `run_server.py` - Production server with uvicorn

## File Naming Conventions

- **Python modules**: lowercase with underscores (`knowledge_graph.py`)
- **Test files**: `test_` prefix (`test_units.py`)
- **Documentation**: UPPERCASE.md (`ARCHITECTURE.md`)
- **Config files**: lowercase with hyphens (`docker-compose.yml`)

## Import Structure

All imports use absolute paths from package root:

```python
from medassist.knowledge_graph import MedicalKnowledgeGraph
from medassist.agentic_orchestrator import AgenticMedicalOrchestrator
```

Tests add parent directory to path:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## Development Workflow

1. **Write code** in `medassist/`
2. **Add tests** in `tests/`
3. **Create examples** in `examples/`
4. **Document** in `docs/`
5. **Deploy** with Docker or Kubernetes

## Production Deployment

- **Local**: `python run_server.py`
- **Docker**: `docker-compose up -d`
- **Kubernetes**: `kubectl apply -f k8s/deployment.yaml`
