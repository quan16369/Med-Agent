# MedGemma - Simple Medical Agent

**Tập trung vào core agent functionality, loại bỏ production infrastructure**

## 🎯 Overview

Medical QA Agent với:
- 🧠 **Multi-agent workflow** (5 specialized agents)
- 🔗 **Knowledge Graph reasoning** (AMG-RAG)
- 📊 **Entity extraction** (BioBERT)
- 🖼️ **Multimodal support** (text + medical images)
- ⚡ **Hierarchical retrieval** (Code RAG optimization)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone
git clone https://github.com/yourusername/MedGemma.git
cd MedGemma

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys (Groq, OpenAI, etc.)
```

### 2. Run Agent

#### Interactive Mode (Recommended)

```bash
python run_agent.py
```

Commands trong interactive mode:
- `/ask <question>` - Hỏi câu hỏi medical
- `/ingest <file.txt>` - Ingest document vào knowledge graph
- `/explore <entity>` - Explore entity trong KG
- `/image <path> <question>` - Hỏi với medical image
- `/quit` - Thoát

#### Single Question Mode

```bash
python run_agent.py --ask "What are the symptoms of diabetes?"
```

#### Ingest Document Mode

```bash
python run_agent.py --ingest medical_paper.txt
```

---

## 📖 Usage Examples

### Example 1: Simple Q&A

```python
from simple_agent import SimpleAgent

# Initialize agent
agent = SimpleAgent()

# Ask question
result = agent.ask(
    question="What causes type 2 diabetes?",
    context="Patient is 45 years old, overweight"
)

print(result['answer'])
print(f"Entities: {len(result['entities'])}")
print(f"Relationships: {len(result['relationships'])}")
```

### Example 2: Multimodal Q&A (với medical image)

```python
# Ask with X-ray image
result = agent.ask_with_image(
    question="What abnormalities are visible in this chest X-ray?",
    image_path="chest_xray.jpg"
)

print(result['answer'])
print(result['visual_analysis'])
```

### Example 3: Document Ingestion

```python
# Ingest medical document vào knowledge graph
result = agent.ingest_document(
    text="Diabetes mellitus causes hyperglycemia. Metformin treats diabetes by reducing glucose production.",
    doc_id="doc_001",
    metadata={"source": "medical_textbook", "year": 2023}
)

print(f"Extracted {result['entity_count']} entities")
print(f"Found {result['relationship_count']} relationships")
```

### Example 4: Knowledge Graph Exploration

```python
# Explore entity trong knowledge graph
result = agent.explore_knowledge_graph(
    entity_name="diabetes",
    max_depth=2
)

for entity in result['related_entities']:
    print(f"- {entity['name']} ({entity['type']})")
    print(f"  Relationship: {entity['relationship']}")
```

---

## 🏗️ Architecture

### Multi-Agent Workflow (5 Agents)

```
User Question
     ↓
Orchestrator → Knowledge Agent    → Query KG
             → Diagnostic Agent   → Analyze symptoms  
             → Treatment Agent    → Recommend treatments
             → Evidence Agent     → Retrieve papers (PubMed)
             → Validator Agent    → Cross-validate findings
     ↓
Final Answer + Reasoning
```

### Knowledge Graph Pipeline (AMG-RAG)

```
Document → Entity Extraction (BioBERT)
        → Relationship Inference (Pattern matching + Proximity)
        → Bidirectional Relationships (causes ↔ caused_by)
        → Relevance Scoring (LLM-based 1-10)
        → Knowledge Graph Storage
```

### Hierarchical Retrieval (Code RAG)

```
Query → Semantic Alignment Distillation
      → Redundancy-Aware Pruning
      → Topological Proximity Metric
      → Diversity-Aware Reranking
      → Top-K Results
```

---

## 🔧 Configuration

Edit `.env` file:

```bash
# LLM Provider
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key  # optional

# Model Selection
LLM_MODEL=llama-3.1-70b-versatile
TEMPERATURE=0.7

# Knowledge Graph
KG_TYPE=networkx  # or neo4j
NEO4J_URI=bolt://localhost:7687  # if using neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Retrieval Settings
MAX_ENTITIES=50
MAX_RELATIONSHIPS=100
CONFIDENCE_THRESHOLD=0.5
```

---

## 📊 Features

### ✅ Implemented

- **Multi-Agent Workflow**: 5 specialized agents với collaboration
- **Knowledge Graph**: Entity extraction + relationship inference
- **Bidirectional Relationships**: A→B and B→A với evidence tracking
- **Relevance Scoring**: LLM-based 1-10 scoring
- **Multimodal Support**: Text + medical images (X-ray, CT, MRI)
- **Hierarchical Retrieval**: 4-stage optimization pipeline
- **PubMed Integration**: Scientific literature retrieval
- **MCP Architecture**: Tools (KG Search, KG Write, Web Search) + Skills

### 🔄 In Progress

- Entity summarization (AMG-RAG feature)
- Confidence propagation in graph traversal
- Vision model integration (CheXNet, BiomedCLIP)

---

## 📈 Performance

### Expected Improvements

| Feature | Metric | Improvement |
|---------|--------|-------------|
| Redundancy Pruning | Precision | +15-20% |
| Diversity Reranking | Result Diversity | +30% |
| Semantic Alignment | Relevance | +10% |
| Search Space Reduction | Speed | 2x faster |

### Benchmarks

| Dataset | Target Accuracy |
|---------|----------------|
| MEDQA | 74.1% F1 |
| MEDMCQA | 66.34% |

---

## 🗂️ Project Structure

```
MedGemma/
├── simple_agent.py              # Simple agent interface (main entry)
├── run_agent.py                 # CLI runner (interactive mode)
├── medassist/
│   ├── agentic_orchestrator.py  # Multi-agent orchestrator
│   ├── agentic_workflow.py      # Agent collaboration logic
│   ├── knowledge_graph.py       # KG implementation
│   ├── medical_ner.py           # BioBERT entity extraction
│   ├── ingestion_pipeline.py    # Document processing
│   ├── graph_retrieval.py       # Graph-based retrieval
│   ├── hierarchical_retrieval.py # Code RAG optimization
│   ├── multimodal.py            # Image processing
│   ├── multimodal_models.py     # Multimodal content models
│   ├── medical_image_search.py  # Medical image search
│   ├── pubmed_retrieval.py      # PubMed API integration
│   ├── mcp_server.py            # MCP server (tools + skills)
│   └── mcp_client.py            # MCP client
├── examples/
│   ├── demo_agentic.py          # Demo agentic workflow
│   └── demo_multimodal_api.py   # Demo multimodal features
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧪 Examples

### Run Demo

```bash
# Demo simple agent
python simple_agent.py

# Demo agentic workflow
python examples/demo_agentic.py

# Demo multimodal
python examples/demo_multimodal_api.py
```

### Test Ingestion Pipeline

```bash
python -c "
from simple_agent import SimpleAgent

agent = SimpleAgent()

# Ingest sample document
result = agent.ingest_document(
    text='Diabetes causes hyperglycemia. Metformin treats diabetes.',
    doc_id='sample_001'
)

print(f'Entities: {result[\"entity_count\"]}')
print(f'Relationships: {result[\"relationship_count\"]}')
"
```

---

## 📚 Documentation

- **Architecture**: Xem [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **MCP Architecture**: Xem [docs/MCP_ARCHITECTURE.md](docs/MCP_ARCHITECTURE.md)
- **Setup Guide**: Xem [docs/SETUP.md](docs/SETUP.md)

---

## 🔬 Research Papers

System kết hợp 3 papers:

1. **AMG-RAG** (arxiv:2410.03883)
   - Agentic Medical Knowledge Graphs
   - Entity extraction + relationship inference
   - Bidirectional relationships with evidence

2. **Kubrick AI Multimodal Course**
   - Multimodal content models
   - Base64 image handling
   - Medical image search patterns

3. **Code RAG** (arxiv:2508.10068)
   - Hierarchical retrieval optimization
   - Semantic alignment + redundancy pruning
   - Topological proximity + diversity reranking

---

## 🛠️ Development

### Add New Agent

```python
# In medassist/agentic_workflow.py
class CustomAgent(MedicalAgent):
    def analyze(self, query: str, context: Dict) -> AgentResponse:
        # Your logic here
        return AgentResponse(
            agent_name="CustomAgent",
            result="...",
            confidence=0.8
        )
```

### Extend Knowledge Graph

```python
# Add custom relationship type
from medassist.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()
kg.add_relationship(
    source="Disease A",
    target="Symptom B",
    relation_type="has_symptom",
    confidence=0.9,
    evidence="Clinical observation"
)
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE)

---

## 🤝 Contributing

Contributions welcome! Focus areas:
- Vision model integration
- Entity summarization
- Confidence propagation
- Benchmarking on medical datasets

---

## 💡 Tips

**Q: Agent chậm?**
- Giảm `MAX_ENTITIES` và `MAX_RELATIONSHIPS` trong `.env`
- Tăng `CONFIDENCE_THRESHOLD` để filter entities

**Q: Kết quả không accurate?**
- Ingest thêm medical documents vào KG
- Adjust `TEMPERATURE` (0.3-0.7 for medical domain)
- Enable `RELEVANCE_SCORING` trong config

**Q: Muốn dùng Neo4j thay vì NetworkX?**
- Set `KG_TYPE=neo4j` trong `.env`
- Configure Neo4j connection settings

---

## 📧 Contact

Issues: [GitHub Issues](https://github.com/yourusername/MedGemma/issues)
