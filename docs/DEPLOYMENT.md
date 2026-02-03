# Medical Knowledge Assistant

Production-ready multi-agent workflow system for medical question answering.

## System Overview

A sophisticated agentic workflow system that combines:
- Medical knowledge graph with multi-hop reasoning
- 5 specialized AI agents working collaboratively
- BioBERT-based medical entity recognition
- PubMed scientific evidence retrieval
- Production infrastructure (API, monitoring, health checks)

## Architecture

### Multi-Agent System

**Knowledge Agent**
- Queries medical knowledge graph
- Performs BFS/DFS traversal
- Confidence scoring with geometric mean

**Diagnostic Agent**
- Analyzes symptoms and patient data
- Provides diagnostic reasoning
- Cross-references with knowledge base

**Treatment Agent**
- Recommends evidence-based treatments
- Considers contraindications
- Provides treatment rationale

**Evidence Agent**
- Retrieves PubMed scientific literature
- Validates claims with research
- Provides citation links

**Validator Agent**
- Cross-validates all findings
- Ensures consistency
- Identifies conflicts

### Core Components

- **Knowledge Graph**: Neo4j-based medical ontology with ~19 nodes, ~19 edges
- **Graph Retrieval**: BFS/DFS algorithms with confidence scoring
- **Medical NER**: BioBERT for entity extraction
- **PubMed API**: Scientific evidence retrieval
- **Orchestrator**: Workflow planning and execution

## Quick Start

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check status
curl http://localhost:8000/health

# Query API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes diabetes and how is it treated?"}'
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run API server
python run_server.py

# Or demo directly
python demo_agentic.py
```

## API Documentation

### Endpoints

**POST /query** - Process medical question
```json
Request:
{
  "question": "What causes diabetes?",
  "include_trace": false,
  "include_stats": false
}

Response:
{
  "answer": "Detailed medical answer with reasoning...",
  "confidence": 0.85,
  "execution_time": 2.3,
  "timestamp": "2024-01-01T12:00:00"
}
```

**GET /health** - System health check
```json
{
  "status": "healthy",
  "checks": {
    "liveness": {"status": "healthy"},
    "readiness": {"status": "healthy"},
    "startup": {"status": "healthy"}
  }
}
```

**GET /stats** - System statistics
```json
{
  "total_queries": 42,
  "avg_response_time": 2.5,
  "orchestrator_loaded": true
}
```

**GET /docs** - Interactive API documentation (Swagger UI)

**GET /redoc** - Alternative API documentation (ReDoc)

### Rate Limiting

- 10 requests per minute per IP address
- Returns 429 status when exceeded

## Configuration

Environment-based configuration via `.env` file:

```bash
# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Model
MODEL_NAME=dmis-lab/biobert-v1.1
DEVICE=cuda  # or cpu
MAX_LENGTH=512

# Retrieval
MAX_DEPTH=3
MAX_WIDTH=10
CONFIDENCE_THRESHOLD=0.3

# PubMed
PUBMED_EMAIL=your@email.com
PUBMED_API_KEY=your_api_key

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/medassist.log
```

## Testing

```bash
# Run all tests
pytest

# Specific test suites
pytest tests/test_units.py          # Unit tests
pytest tests/test_integration.py    # Integration tests
pytest tests/test_api.py            # API tests

# With coverage
pytest --cov=medassist --cov-report=html

# Skip slow tests
pytest -m "not slow"

# Verbose output
pytest -v --tb=short
```

## Deployment

### Kubernetes

```bash
# Apply configuration
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n medassist

# View logs
kubectl logs -f deployment/medassist -n medassist

# Scale replicas
kubectl scale deployment/medassist --replicas=5 -n medassist
```

### Docker

```bash
# Build image
docker build -t medassist:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name medassist \
  medassist:latest
```

## Monitoring

### Health Checks

Three probe types for Kubernetes:

- **Liveness**: `/health/liveness` - Application running
- **Readiness**: `/health/readiness` - Ready to serve traffic
- **Startup**: `/health` - Initialization complete

### Logging

Structured logging with rotation:
- Console output (JSON format in production)
- File output: `logs/medassist.log`
- Max size: 10MB per file
- Retention: 5 backup files

### Metrics

Available at `/stats` endpoint:
- Total queries processed
- Average response time
- Orchestrator initialization status
- Configuration details

## Project Structure

```
MedGemma/
├── medassist/                   # Core package
│   ├── config.py                # Configuration management
│   ├── exceptions.py            # Error handling
│   ├── logging_utils.py         # Logging utilities
│   ├── health.py                # Health checks
│   ├── knowledge_graph.py       # Knowledge graph
│   ├── graph_retrieval.py       # Graph traversal
│   ├── medical_ner.py           # Medical NER
│   ├── pubmed_retrieval.py      # PubMed API
│   ├── agentic_workflow.py      # Multi-agent system
│   └── agentic_orchestrator.py  # Main orchestrator
├── tests/                       # Test suite
│   ├── test_units.py            # Unit tests
│   ├── test_integration.py      # Integration tests
│   └── test_api.py              # API tests
├── k8s/                         # Kubernetes configs
│   └── deployment.yaml          # K8s deployment
├── .github/workflows/           # CI/CD pipelines
│   └── ci-cd.yml                # GitHub Actions
├── api.py                       # FastAPI application
├── run_server.py                # Production server
├── demo_agentic.py              # Demo script
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Multi-container setup
├── requirements.txt             # Dependencies
└── .env.example                 # Environment template
```

## Development

### Code Quality

```bash
# Format code
black medassist/ tests/
isort medassist/ tests/

# Lint
flake8 medassist/ tests/

# Type checking
mypy medassist/
```

### Adding Components

**New Agent:**
```python
from medassist.agentic_workflow import BaseAgent

class NewAgent(BaseAgent):
    def process(self, message):
        # Agent logic
        return result
    
    def reflect(self, result):
        # Self-reflection
        return confidence
```

**Extend Knowledge Graph:**
```python
from medassist.knowledge_graph import MedicalKnowledgeGraph

kg = MedicalKnowledgeGraph()
kg.add_entity("NewDisease", "Disease")
kg.add_relationship("Disease1", "CAUSES", "Disease2")
```

## Performance

### Benchmarks

Typical query performance:
- Simple queries: 1-3 seconds
- Multi-hop reasoning: 3-8 seconds
- With PubMed evidence: 5-15 seconds

### Optimization

1. **GPU Acceleration**: Set `DEVICE=cuda`
2. **Cache PubMed**: Results cached automatically
3. **Adjust Depth**: Lower `MAX_DEPTH` for faster queries
4. **Connection Pool**: Configure Neo4j pool size

## Troubleshooting

**Service not starting:**
```bash
# Check logs
tail -f logs/medassist.log

# Verify health
curl http://localhost:8000/health
```

**Neo4j connection failed:**
```bash
# Verify Neo4j running
docker-compose ps neo4j

# Test connection
docker exec -it medassist-neo4j cypher-shell -u neo4j -p medassist123
```

**Out of memory:**
```bash
# Use CPU mode
export DEVICE=cpu

# Reduce batch size
export MAX_WIDTH=5
```

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

## License

MIT License

## Contact

For issues and questions, please open a GitHub issue.
