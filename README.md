# 🏥 MedAssist: Agentic Medical Graph-RAG with MedGemma

**Competition**: [MedGemma Impact Challenge (Kaggle)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)  
**Category**: Agentic Workflow Prize ($10,000)  
**Deadline**: February 24, 2026  
**Base Architecture**: [AMG-RAG (EMNLP 2025)](https://github.com/MrRezaeiUofT/AMG-RAG)  
**Optimization**: SaraCoder-inspired resource optimization (arXiv:2508.10068)

An intelligent medical question answering system that uses **agentic workflows** to dynamically construct knowledge graphs and reason over medical evidence, enhanced with **hierarchical optimization** for maximum efficiency.

---

## 🎯 Project Overview

MedAssist implements **AMG-RAG (Agentic Medical Graph-RAG)** with Google's MedGemma models:

### What Makes This Agentic?

Traditional RAG: `Query → Retrieve → Generate`

Our Agentic Workflow: 
```
Query → Extract Entities → Search Evidence → Build Knowledge Graph → 
Explore Reasoning Paths → Synthesize Answer
```

Each stage is an **autonomous agent** that makes decisions about what entities to extract, what evidence to retrieve, what graph paths to explore, and how to synthesize the final answer.

### Key Innovation

- **Dynamic Knowledge Graphs**: Built at query-time from medical evidence
- **Path-Based Reasoning**: Explores graph connections for chain-of-thought
- **Multi-Source Evidence**: PubMed, Wikipedia, medical databases
- **MedGemma Integration**: Google's specialized medical AI models

---

## 🏗️ Architecture

### 5-Stage Agentic Workflow (LangGraph)

```python
Stage 1: Entity Extraction Agent
├─ Input: User query
├─ Action: Extract medical entities with 1-10 relevance scores
└─ Output: List of MedicalEntity objects

Stage 2: Evidence Retrieval Agent
├─ Input: Extracted entities
├─ Action: Search PubMed for each entity
└─ Output: Retrieved articles with abstracts

Stage 3: Knowledge Graph Construction Agent
├─ Input: Entities + Evidence
├─ Action: Extract relationships, build NetworkX graph
└─ Output: MedicalKnowledgeGraph

Stage 4: Path-Based Reasoning Agent
├─ Input: Knowledge graph + Query
├─ Action: Explore graph paths between entities
└─ Output: Reasoning paths

Stage 5: Answer Generation Agent
├─ Input: Reasoning paths + Evidence
├─ Action: Chain-of-thought synthesis
└─ Output: Final answer with confidence
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | MedGemma 3 (8B/27B) | Medical reasoning & entity extraction |
| **Orchestration** | LangGraph | Agentic workflow management |
| **Knowledge Graph** | NetworkX | Multi-directional relationship graph |
| **Evidence** | PubMed API | Peer-reviewed medical literature |
| **Embeddings** | Sentence-Transformers | Semantic similarity |
| **Vector DB** | Chroma (optional) | Efficient retrieval |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd MedGemma

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure LLM Provider

Choose one of the following:

**Option A: Google GenAI API (Recommended)**
```bash
export GOOGLE_API_KEY='your-api-key'
```

**Option B: Local Ollama**
```bash
# Install Ollama: https://ollama.ai
ollama pull medgemma
ollama serve
```

**Option C: Vertex AI (GCP)**
```bash
export GOOGLE_CLOUD_PROJECT='your-project-id'
gcloud auth application-default login
```

### 3. Run Example

```bash
# Basic test (no PubMed)
python example_usage.py

# Full test with PubMed
python test_amg_rag.py

# Simple query test
python test_amg_rag.py --simple
```

---

## 💡 Usage

### Basic Usage

```python
from medassist import AMG_RAG_System

# Initialize system (auto-detects LLM provider)
system = AMG_RAG_System(
    model_name="medgemma-3-8b",
    temperature=0.0,
    enable_pubmed=True
)

# Ask a medical question
result = system.answer_question(
    "What is the mechanism of action of metformin?"
)

print(result["answer"])
print(f"Entities: {result['entities']}")
print(f"Reasoning paths: {len(result['reasoning_paths'])}")
```

### MEDQA-Style Question

```python
query = """
A 65-year-old woman with rheumatoid arthritis is started on methotrexate.
What supplement should be prescribed with this medication?

A) Calcium
B) Folic acid
C) Vitamin D
D) Iron
"""

result = system.answer_question(query)

# Access knowledge graph
kg_stats = result["metadata"]["kg_stats"]
print(f"KG: {kg_stats['num_entities']} entities, {kg_stats['num_relations']} relations")
```

### Custom Configuration

```python
system = AMG_RAG_System(
    model_name="medgemma-3-27b",      # Larger model
    temperature=0.2,                   # Slight creativity
    pubmed_max_results=10,             # More evidence
    min_entity_relevance=7,            # Stricter filtering
    enable_pubmed=True
)
```

---

## 📊 Project Structure

```
MedGemma/
├── medassist/
│   ├── amg_rag.py              # Main AMG-RAG system with LangGraph
│   ├── __init__.py             # Package exports
│   ├── llm/
│   │   ├── medgemma.py         # MedGemma LLM integration
│   │   └── __init__.py
│   ├── core/
│   │   ├── knowledge_graph.py  # NetworkX knowledge graph
│   │   ├── chains.py           # LLM chains (entity, relation, summarization)
│   │   └── __init__.py
│   ├── models/
│   │   ├── entities.py         # MedicalEntity, MedicalRelation
│   │   └── __init__.py
│   └── tools/
│       ├── pubmed.py           # PubMed API integration
│       └── __init__.py
├── test_amg_rag.py             # Full system test
├── test_basic.py               # Unit tests
├── test_pubmed.py              # PubMed integration test
├── example_usage.py            # Usage examples
├── requirements.txt            # Dependencies
├── .env.example                # Environment configuration
├── AMG_RAG_START.md           # Development roadmap
└── README.md                   # This file
```

---

## 🎓 How It Works

### Example Workflow

**Query**: "What is the mechanism of diabetic neuropathy?"

**Stage 1: Entity Extraction**
```
Entities extracted:
- diabetic neuropathy (disease, relevance: 10)
- diabetes mellitus (disease, relevance: 9)
- hyperglycemia (symptom, relevance: 8)
- nerve damage (symptom, relevance: 8)
```

**Stage 2: Evidence Retrieval**
```
PubMed search for "diabetic neuropathy":
- 5 articles retrieved
- Abstracts extracted and parsed
```

**Stage 3: Knowledge Graph**
```
Graph built: 15 entities, 23 relationships
Key relationships:
- hyperglycemia --[causes]--> nerve damage
- diabetes mellitus --[causes]--> hyperglycemia
- hyperglycemia --[leads_to]--> oxidative stress
```

**Stage 4: Path-Based Reasoning**
```
Path 1: diabetes → hyperglycemia → oxidative stress → nerve damage
Path 2: diabetes → advanced glycation → microvascular damage
Path 3: hyperglycemia → polyol pathway → nerve dysfunction
```

**Stage 5: Answer Generation**
```
Chain of Thought:
1. Chronic hyperglycemia is the primary trigger
2. Multiple pathways contribute: polyol, oxidative stress, AGEs
3. Microvascular damage leads to ischemia
4. Result: axonal degeneration and demyelination

Final Answer: Diabetic neuropathy occurs through multiple mechanisms...
Confidence: High
```

---

## 🔬 Features

### ✅ Implemented

- [x] MedGemma LLM integration (Google GenAI, Vertex AI, Ollama, vLLM)
- [x] **Multimodal support** (X-ray, CT, MRI, histopathology image analysis)
- [x] Entity extraction with 1-10 relevance scoring
- [x] Bidirectional relationship extraction
- [x] NetworkX-based knowledge graph
- [x] PubMed evidence retrieval
- [x] LangGraph agentic workflow
- [x] Path-based reasoning
- [x] Chain-of-thought answer generation
- [x] Multi-provider LLM support (auto-detection)
- [x] Medical image report generation
- [x] Longitudinal image comparison
- [x] **SaraCoder-inspired optimization** (hierarchical entity/evidence optimization)
- [x] **Medical term disambiguation** (context-aware abbreviation resolution)
- [x] **Diversity-optimized retrieval** (MMR for maximum information coverage)

### 🚧 In Progress

- [ ] Wikipedia integration
- [ ] Vector database (Chroma) for semantic search
- [ ] MEDQA dataset evaluation (text + image questions)
- [ ] Knowledge graph visualization
- [ ] Semantic embeddings for entity clustering
- [ ] Multimodal RAG with image embeddings (MedSigLIP)

### 🎯 Competition Goals

**Target Category**: Agentic Workflow Prize

**Criteria**:
- ✅ Reimagines complex workflow with intelligent agents
- ✅ Uses HAI-DEF models (MedGemma) effectively
- ✅ Demonstrates significant efficiency improvements
- ✅ Showcases autonomous decision-making

---

## 📈 Performance Targets

Based on AMG-RAG paper (EMNLP 2025):

| Dataset | Target Score | Metric |
|---------|-------------|--------|
| MEDQA | 74%+ | F1 Score |
| MEDMCQA | 66%+ | Accuracy |
| MedQA-USMLE | 70%+ | Accuracy |

---

## 🛠️ Development

### Running Tests

```bash
# Unit tests
python test_basic.py

# PubMed integration
python test_pubmed.py

# Full system test
python test_amg_rag.py

# Simple query test
python test_amg_rag.py --simple
```

### Environment Variables

See `.env.example` for configuration options:

```bash
# Required: LLM Provider
GOOGLE_API_KEY=your-key

# Optional: PubMed
PUBMED_API_KEY=your-ncbi-key
PUBMED_EMAIL=your@email.com

# Optional: Logging
LOG_LEVEL=INFO
```

---

## 📚 References

- **AMG-RAG Paper**: [Agentic Medical Graph-RAG (EMNLP 2025)](https://github.com/MrRezaeiUofT/AMG-RAG)
- **HAI-DEF Models**: [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations)
- **MedGemma**: [Medical Gemma Models](https://developers.google.com/health-ai-developer-foundations/medgemma)
- **Competition**: [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)

---

## 📝 License

MIT License - See [terms](https://developers.google.com/health-ai-developer-foundations/terms) for HAI-DEF models usage.

---

## 🤝 Contributing

This is a competition entry. After the competition (Feb 24, 2026), contributions welcome!

**Contact**: [Your contact info]  
**Team**: [Your team members]
