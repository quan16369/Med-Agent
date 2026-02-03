# AMG-RAG Setup Guide
Complete setup instructions for the Medical Knowledge Graph system based on EMNLP 2025 paper

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Neo4j Database Setup](#neo4j-database-setup)
4. [BioBERT Model Setup](#biobert-model-setup)
5. [Knowledge Graph Construction](#knowledge-graph-construction)
6. [Running the System](#running-the-system)
7. [Testing](#testing)
8. [Benchmarking](#benchmarking)

---

## Prerequisites

- Python 3.9+
- Docker (for Neo4j)
- CUDA-capable GPU (optional, for BioBERT)
- 16GB+ RAM recommended
- 10GB+ disk space

---

## Installation

### 1. Clone and setup environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Install BioBERT models

```bash
# Download BioBERT model (will auto-download on first use)
python -c "from transformers import AutoTokenizer, AutoModelForTokenClassification; \
           AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1'); \
           AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1')"
```

Alternative models:
- `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` (PubMed-trained)
- `allenai/scibert_scivocab_cased` (Scientific papers)

---

## Neo4j Database Setup

### Option 1: Docker (Recommended)

```bash
# Pull and run Neo4j
docker run -d \
  --name neo4j-medgraph \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/medgraph123 \
  -e NEO4J_apoc_export_file_enabled=true \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_apoc_import_file_use__neo4j__config=true \
  -v $(pwd)/neo4j_data:/data \
  neo4j:5.13.0
```

Access Neo4j Browser:
- URL: http://localhost:7474
- Username: neo4j
- Password: medgraph123

### Option 2: Local Installation

Download from: https://neo4j.com/download/

### Create Database Schema

```cypher
// Create constraints for entities
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:MedicalEntity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:MedicalEntity) REQUIRE e.name IS UNIQUE;

// Create indexes for performance
CREATE INDEX entity_type IF NOT EXISTS FOR (e:MedicalEntity) ON (e.type);
CREATE INDEX entity_name_text IF NOT EXISTS FOR (e:MedicalEntity) ON (e.name);

// Create relationship indexes
CREATE INDEX rel_confidence IF NOT EXISTS FOR ()-[r:MEDICAL_RELATION]-() ON (r.confidence);
CREATE INDEX rel_type IF NOT EXISTS FOR ()-[r:MEDICAL_RELATION]-() ON (r.relation_type);
```

### Configure Connection

Create `.env` file:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=medgraph123

# PubMed API
PUBMED_EMAIL=your_email@example.com
PUBMED_API_KEY=your_api_key_here  # Optional but recommended

# Model Configuration
BIOBERT_MODEL=dmis-lab/biobert-base-cased-v1.1
DEVICE=cuda  # or cpu
```

---

## BioBERT Model Setup

### Test BioBERT NER

```python
from medassist.medical_ner import BioBERTNER

# Initialize
ner = BioBERTNER(model_name="dmis-lab/biobert-base-cased-v1.1")

# Test extraction
text = "Patient has diabetes mellitus with symptoms of numbness and blurred vision."
entities = ner.extract(text)

for entity in entities:
    print(f"[{entity.entity_type}] {entity.text} (confidence: {entity.confidence:.2f})")
```

Expected output:
```
[disease] diabetes mellitus (confidence: 0.95)
[symptom] numbness (confidence: 0.89)
[symptom] blurred vision (confidence: 0.87)
```

---

## Knowledge Graph Construction

### 1. Initialize Knowledge Graph

```python
from medassist.knowledge_graph import MedicalKnowledgeGraph
from medassist.amg_rag_orchestrator import AMGRAGOrchestrator

# Initialize with Neo4j
orchestrator = AMGRAGOrchestrator(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="medgraph123",
    use_memory_graph=False,  # Use Neo4j
    load_pretrained_graph=True
)
```

### 2. Load Medical Corpus

```python
# Create script: scripts/load_medical_corpus.py
from medassist.knowledge_graph import MedicalEntity, MedicalRelationship
import json

# Load from medical corpus (example format)
with open("data/medical_corpus.json", "r") as f:
    corpus = json.load(f)

entities = []
relationships = []

for item in corpus:
    # Parse entities
    entity = MedicalEntity(
        id=item["id"],
        name=item["name"],
        entity_type=item["type"],
        aliases=item.get("aliases", [])
    )
    entities.append(entity)
    
    # Parse relationships
    for rel in item.get("relationships", []):
        relationship = MedicalRelationship(
            source_id=item["id"],
            target_id=rel["target_id"],
            relation_type=rel["type"],
            confidence=rel["confidence"],
            evidence_count=rel["evidence_count"],
            source=rel.get("source", "pubmed")
        )
        relationships.append(relationship)

# Add to graph
orchestrator.add_knowledge(entities, relationships)

print(f"Loaded {len(entities)} entities and {len(relationships)} relationships")
```

### 3. Run Graph Construction

```bash
python scripts/load_medical_corpus.py
```

---

## Running the System

### 1. Test AMG-RAG Orchestrator

```bash
python medassist/amg_rag_orchestrator.py
```

### 2. Interactive Demo

```python
from medassist.amg_rag_orchestrator import AMGRAGOrchestrator

# Initialize
orchestrator = AMGRAGOrchestrator(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="medgraph123",
    load_pretrained_graph=True
)

# Process query
query = "What causes numbness in diabetic patients?"
response = orchestrator.process(query)

print(f"Query: {response.query}")
print(f"\nAnswer: {response.answer}")
print(f"\nKnowledge Paths:")
for path in response.knowledge_paths:
    print(f"  - {path}")
print(f"\nConfidence: {response.confidence:.2%}")
print(f"Processing Time: {response.processing_time:.3f}s")
```

### 3. Web Interface (Optional)

Create Gradio interface:

```python
import gradio as gr
from medassist.amg_rag_orchestrator import AMGRAGOrchestrator

orchestrator = AMGRAGOrchestrator(load_pretrained_graph=True)

def process_query(query):
    response = orchestrator.process(query)
    
    paths = "\n".join(response.knowledge_paths[:5])
    
    return (
        response.answer,
        paths,
        f"{response.confidence:.2%}",
        response.reasoning_trace
    )

demo = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(label="Medical Question", placeholder="Ask a medical question..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Knowledge Paths"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Reasoning Trace")
    ],
    title="AMG-RAG Medical Assistant",
    description="Medical Q&A powered by Knowledge Graph reasoning"
)

demo.launch()
```

---

## Testing

### Run Unit Tests

```bash
# Test knowledge graph
pytest tests/test_knowledge_graph.py -v

# Test NER
pytest tests/test_medical_ner.py -v

# Test retrieval
pytest tests/test_graph_retrieval.py -v

# Test orchestrator
pytest tests/test_amg_rag_orchestrator.py -v

# Run all tests
pytest tests/ -v --cov=medassist
```

### Manual Testing

```python
# Test knowledge graph
from medassist.knowledge_graph import MedicalKnowledgeGraph

kg = MedicalKnowledgeGraph(use_memory=True)
# ... add entities and test

# Test NER
from medassist.medical_ner import BioBERTNER

ner = BioBERTNER()
entities = ner.extract("diabetes causes neuropathy")
print(entities)

# Test retrieval
from medassist.graph_retrieval import GraphConditionalRetrieval

retrieval = GraphConditionalRetrieval(kg)
result = retrieval.retrieve("What causes numbness in diabetes?")
print(result)
```

---

## Benchmarking

### 1. Download MEDQA Dataset

```bash
mkdir -p data/medqa
cd data/medqa

# Download from official source
wget https://github.com/jind11/MedQA/raw/master/data_clean/questions/US/test.jsonl
```

### 2. Run Benchmark

```python
# Create script: scripts/benchmark_medqa.py
import json
from medassist.amg_rag_orchestrator import AMGRAGOrchestrator
from tqdm import tqdm

# Load MEDQA test set
with open("data/medqa/test.jsonl", "r") as f:
    test_data = [json.loads(line) for line in f]

# Initialize orchestrator
orchestrator = AMGRAGOrchestrator(load_pretrained_graph=True)

# Run benchmark
correct = 0
total = 0

for item in tqdm(test_data[:100]):  # Test on first 100
    query = item["question"]
    correct_answer = item["answer"]
    
    response = orchestrator.process(query)
    
    # Simple exact match (improve with better matching)
    if correct_answer.lower() in response.answer.lower():
        correct += 1
    
    total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%}")
print(f"Target (from paper): 73.9%")
```

### 3. Expected Performance

| Metric | Target (Paper) | Current |
|--------|---------------|---------|
| MEDQA Accuracy | 73.9% | TBD |
| MEDMCQA Accuracy | 66.3% | TBD |
| F1 Score | 74.1% | TBD |
| Processing Time | <1s | TBD |

---

## Troubleshooting

### Neo4j Connection Issues

```python
# Test connection
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "medgraph123"))

with driver.session() as session:
    result = session.run("RETURN 1")
    print(result.single()[0])  # Should print 1

driver.close()
```

### BioBERT Memory Issues

If running on CPU with limited RAM:

```python
# Use lighter model
ner = BioBERTNER(
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    device="cpu"
)

# Or use rule-based fallback
from medassist.medical_ner import BioBERTNER
ner = BioBERTNER()
ner.use_transformer = False  # Force rule-based
```

### Graph Too Large

```python
# Limit graph size
kg = MedicalKnowledgeGraph(use_memory=True)

# Add only high-confidence relationships
for rel in relationships:
    if rel.confidence > 0.7:  # Filter
        kg.add_relationship(rel)
```

---

## Next Steps

1. ✅ Complete setup and test all components
2. 🔄 Load medical corpus into Neo4j
3. 🔄 Integrate actual MedGemma model
4. 🔄 Run benchmark on MEDQA
5. 🔄 Optimize hyperparameters
6. 🔄 Deploy to production
7. 🔄 Submit to Kaggle competition

---

## Resources

- **Paper**: [EMNLP 2025 Findings #679](https://aclanthology.org/2025.findings-emnlp.679/)
- **Neo4j Docs**: https://neo4j.com/docs/
- **BioBERT**: https://github.com/dmis-lab/biobert
- **PubMed API**: https://www.ncbi.nlm.nih.gov/home/develop/api/

---

## Support

For issues, see [AMG_RAG_ARCHITECTURE.md](AMG_RAG_ARCHITECTURE.md) for detailed architecture information.

**Target Performance**: 74.1% F1 on MEDQA (competitive with GPT-4 level)
