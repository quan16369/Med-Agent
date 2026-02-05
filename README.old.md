# Medical Knowledge Assistant

Production-ready multi-agent workflow system for medical question answering with knowledge graph reasoning.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

## Quick Links

- [Deployment Guide](docs/DEPLOYMENT.md) - Complete production deployment documentation
- [Architecture](docs/ARCHITECTURE.md) - System architecture and design
- [MCP Architecture](docs/MCP_ARCHITECTURE.md) - Model Context Protocol implementation
- [Setup Guide](docs/SETUP.md) - Installation and configuration
- [Refactoring History](docs/REFACTORING.md) - Development history

---

## Architecture Overview

Multi-agent workflow system with Model Context Protocol (MCP):

**Agents**:
- **Knowledge Agent**: Queries medical knowledge graph with BFS/DFS traversal
- **Diagnostic Agent**: Analyzes symptoms and provides diagnostic reasoning
- **Treatment Agent**: Recommends evidence-based treatment options
- **Evidence Agent**: Retrieves scientific literature from PubMed
- **Validator Agent**: Cross-validates findings for consistency

**MCP Components**:
- **GraphRAG Ingestion**: Document processing and knowledge extraction
- **MCP Server**: Tools (KG Search, KG Write, Web Search) and Skills (Update Memory, Write Content)
- **MCP Client**: User interface and API access

### Key Features

- Medical knowledge graph with multi-hop reasoning
- BioBERT-based named entity recognition
- Graph-conditioned retrieval with confidence scoring
- Agent collaboration and reflection
- Model Context Protocol for tool orchestration
- Document ingestion pipeline (GraphRAG)
- Production-ready infrastructure (API, monitoring, health checks)

---

## 📊 Performance

| Metric | AMG-RAG (Paper) | Baseline RAG |
|--------|----------------|--------------|
| MEDQA Accuracy | **73.9%** | 65.6% |
| MEDMCQA Accuracy | **66.3%** | 58.1% |
| F1 Score | **74.1%** | 67.2% |
| Model Size | 8B params | 70B+ params |

**Key Advantage**: Achieves GPT-4 level performance with 10× smaller model

---

## Quick Start

### Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/MedGemma.git
cd MedGemma

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

# Or run demo
python examples/demo_agentic.py
```

---

## Usage

### API Endpoints

**POST /query** - Process medical question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the symptoms of diabetes?",
    "include_trace": true,
    "include_stats": true
  }'
```

**GET /health** - System health check
```bash
curl http://localhost:8000/health
```

**GET /stats** - System statistics
```bash
curl http://localhost:8000/stats
```

**GET /docs** - Interactive API documentation

### Python API

```python
from medassist.agentic_orchestrator import AgenticMedicalOrchestrator

# Initialize orchestrator
orchestrator = AgenticMedicalOrchestrator()

# Process query
result = orchestrator.execute_workflow(
    "What causes diabetes and how is it treated?"
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Agents used: {len(result['statistics'])} agents")
```

---

## Project Structure

```
MedGemma/
├── medassist/              # Core package
│   ├── config.py           # Configuration management
│   ├── exceptions.py       # Error handling
│   ├── logging_utils.py    # Logging utilities
│   ├── health.py           # Health checks
│   ├── knowledge_graph.py  # Knowledge graph
│   ├── graph_retrieval.py  # Graph traversal
│   ├── medical_ner.py      # Medical NER
│   ├── pubmed_retrieval.py # PubMed API
│   ├── agentic_workflow.py # Multi-agent system
│   └── agentic_orchestrator.py # Main orchestrator
├── tests/                  # Test suite
│   ├── test_units.py
│   ├── test_integration.py
│   └── test_api.py
├── examples/               # Demo scripts
│   ├── demo_agentic.py
│   ├── demo_amg_rag.py
│   └── test_components.py
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── SETUP.md
│   └── REFACTORING.md
├── k8s/                    # Kubernetes configs
│   └── deployment.yaml
├── .github/workflows/      # CI/CD pipelines
│   └── ci-cd.yml
├── scripts/                # Utility scripts
├── api.py                  # FastAPI application
├── run_server.py           # Production server
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container setup
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## Testing

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_units.py          # Unit tests
pytest tests/test_integration.py    # Integration tests
pytest tests/test_api.py            # API tests

# With coverage
pytest --cov=medassist --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

---

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for complete deployment guide.

### Docker

```bash
docker build -t medassist:latest .
docker run -d -p 8000:8000 --env-file .env medassist:latest
```

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods -n medassist
```

---

## Configuration

Environment variables (see [.env.example](.env.example)):

```bash
# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Model
DEVICE=cuda  # or cpu
MAX_DEPTH=3
MAX_WIDTH=10

# API
PUBMED_EMAIL=your@email.com
LOG_LEVEL=INFO
```

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

For issues and questions, please open a GitHub issue.

- **Confidence Propagation**: Aggregates uncertainty across hops

### 4. Evidence Retrieval

- **PubMed Search**: Retrieves supporting scientific literature
- **Relevance Ranking**: Scores articles by query relevance
- **Citation Tracking**: Links evidence to knowledge graph edges

---

## 🔬 Research Foundation

Based on EMNLP 2025 paper:

**"Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge"**

*Authors*: Mohammad Reza Rezaei, Reza Saadati Fard, Jayson L. Parker, Rahul G. Krishnan, Milad Lankarany

*Key Contributions*:
- Dynamic knowledge graphs that continuously update from latest research
- Confidence-scored relationships with evidence tracking
- Multi-hop reasoning outperforms flat retrieval by 8-10%
- Achieves 74.1% F1 with only 8B parameters (vs 70B+ baselines)

**Paper**: [EMNLP 2025 Findings #679](https://aclanthology.org/2025.findings-emnlp.679/)

---

## 🧪 Benchmarking

### MEDQA Dataset

```bash
# Download MEDQA test set
mkdir -p data/medqa
wget https://github.com/jind11/MedQA/raw/master/data_clean/questions/US/test.jsonl -P data/medqa/

# Run benchmark
python scripts/benchmark_medqa.py
```

### Expected Results

- **MEDQA Accuracy**: 73.9% (target from paper)
- **MEDMCQA Accuracy**: 66.3%
- **F1 Score**: 74.1%
- **Processing Time**: <1s per query

---

## 🛠️ Development

### Requirements

- Python 3.9+
- Neo4j 5.0+ (optional, for production)
- 16GB RAM (recommended)
- CUDA GPU (optional, for BioBERT)

### Dependencies

```
torch>=2.0.0
transformers>=4.40.0
neo4j>=5.0.0
biopython>=1.81
sentence-transformers>=2.5.0
faiss-cpu>=1.8.0
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Test specific component
pytest tests/test_knowledge_graph.py -v
```

---

## 📚 Documentation

- [AMG_RAG_ARCHITECTURE.md](AMG_RAG_ARCHITECTURE.md) - Detailed architecture documentation
- [AMG_RAG_SETUP.md](AMG_RAG_SETUP.md) - Complete setup and deployment guide
- [2025.findings-emnlp.679.pdf](2025.findings-emnlp.679.pdf) - Original research paper

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **EMNLP 2025 Paper Authors**: Rezaei et al. for the AMG-RAG architecture
- **BioBERT Team**: DMIS Lab for medical NER models
- **Neo4j**: Graph database platform
- **PubMed/NCBI**: Medical literature database
- **Hugging Face**: Transformers library

---

## 📞 Contact

For questions or issues:
- Open an [issue](https://github.com/yourusername/MedGemma/issues)
- Email: your.email@example.com

---

## ⭐ Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{rezaei2025amgrag,
  title={Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge},
  author={Rezaei, Mohammad Reza and Fard, Reza Saadati and Parker, Jayson L and Krishnan, Rahul G and Lankarany, Milad},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2025},
  year={2025}
}
```

---

**Built with ❤️ for advancing medical AI**
