# 🚀 Quick Start Guide - MedAssist with MedGemma

Get up and running in 5 minutes!

## Prerequisites

- Python 3.10+
- Internet connection (for PubMed API)
- One of:
  - Google API key (recommended)
  - Local Ollama
  - Vertex AI credentials

## Installation

```bash
# 1. Clone repository
git clone <your-repo>
cd MedGemma

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration

### Option A: Google GenAI (Recommended for MedGemma Challenge)

1. Get API key from: https://ai.google.dev/
2. Set environment variable:

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

### Option B: Local Ollama (Free)

1. Install Ollama: https://ollama.ai
2. Pull MedGemma model:

```bash
ollama pull medgemma  # or use gemma:7b as fallback
ollama serve
```

The system will auto-detect Ollama at `localhost:11434`.

### Option C: Vertex AI (GCP)

```bash
export GOOGLE_CLOUD_PROJECT='your-project-id'
gcloud auth application-default login
```

## Quick Test

### Test 1: Basic Functionality (No PubMed)

```bash
python -c "
from medassist import AMG_RAG_System

system = AMG_RAG_System(
    model_name='medgemma-3-8b',
    enable_pubmed=False  # Quick test
)

result = system.answer_question(
    'What is the mechanism of action of metformin?'
)

print(result['answer'])
"
```

Expected: Should return answer in ~10 seconds.

### Test 2: Full Pipeline with PubMed

```bash
python test_amg_rag.py --simple
```

Expected: 
- Extract entities
- Search PubMed
- Build knowledge graph
- Generate answer
- Takes ~30-60 seconds

### Test 3: MEDQA Question

```bash
python test_amg_rag.py
```

Expected: Complete workflow with multiple-choice question.

## Usage Examples

### Example 1: Simple Question

```python
from medassist import AMG_RAG_System

# Initialize system (auto-detects provider)
system = AMG_RAG_System()

# Ask question
result = system.answer_question(
    "What is type 2 diabetes?"
)

print(result["answer"])
```

### Example 2: Clinical Question with Evidence

```python
from medassist import AMG_RAG_System

system = AMG_RAG_System(
    model_name="medgemma-3-8b",
    temperature=0.0,
    pubmed_max_results=5,
    enable_pubmed=True
)

query = """
A 55-year-old man with type 2 diabetes has numbness in his feet.
What is the most likely mechanism?
"""

result = system.answer_question(query, verbose=True)

# Access knowledge graph
print(f"Entities: {result['entities']}")
print(f"KG Stats: {result['metadata']['kg_stats']}")
```

### Example 3: Inspect Knowledge Graph

```python
from medassist import AMG_RAG_System

system = AMG_RAG_System()
result = system.answer_question("Treatment for hypertension?")

# Get knowledge graph
kg = result["metadata"].get("knowledge_graph")

if kg:
    stats = kg.get_statistics()
    print(f"Entities: {stats['num_entities']}")
    print(f"Relations: {stats['num_relations']}")
    print(f"Entity types: {stats['entity_types']}")
    print(f"Relation types: {stats['relation_types']}")
```

## Troubleshooting

### Error: "No MedGemma provider detected"

**Solution**: Configure one of the LLM providers:
- Google API: `export GOOGLE_API_KEY='...'`
- Ollama: Start `ollama serve`
- Vertex AI: `export GOOGLE_CLOUD_PROJECT='...'`

### Error: "Module not found"

**Solution**: Install requirements:
```bash
pip install -r requirements.txt
```

### Error: PubMed API rate limit

**Solution**: Either:
- Wait 60 seconds and retry
- Get NCBI API key: https://www.ncbi.nlm.nih.gov/account/
- Disable PubMed: `enable_pubmed=False`

### Slow response times

**Solutions**:
- Use smaller model: `model_name="medgemma-2b"`
- Reduce PubMed results: `pubmed_max_results=3`
- Disable PubMed: `enable_pubmed=False`
- Use local Ollama instead of API

## Next Steps

1. Run full test suite:
   ```bash
   python test_basic.py
   python test_pubmed.py
   python test_amg_rag.py
   ```

2. Try example scripts:
   ```bash
   python example_usage.py
   ```

3. Read comprehensive docs:
   - [README.md](README.md) - Full documentation
   - [COMPETITION_WRITEUP.md](COMPETITION_WRITEUP.md) - Competition details
   - [AMG_RAG_START.md](AMG_RAG_START.md) - Development roadmap

4. Explore the code:
   - [medassist/amg_rag.py](medassist/amg_rag.py) - Main system
   - [medassist/llm/medgemma.py](medassist/llm/medgemma.py) - MedGemma integration
   - [medassist/core/knowledge_graph.py](medassist/core/knowledge_graph.py) - Knowledge graph

## Support

- **Competition**: https://www.kaggle.com/competitions/med-gemma-impact-challenge
- **HAI-DEF Models**: https://developers.google.com/health-ai-developer-foundations
- **MedGemma Docs**: https://developers.google.com/health-ai-developer-foundations/medgemma

## Performance Tips

**For fastest responses**:
```python
system = AMG_RAG_System(
    model_name="medgemma-2b",       # Smaller model
    temperature=0.0,                 # Deterministic
    pubmed_max_results=3,            # Fewer articles
    min_entity_relevance=7,          # Stricter filtering
    enable_pubmed=False              # Skip retrieval
)
```

**For best accuracy**:
```python
system = AMG_RAG_System(
    model_name="medgemma-3-27b",    # Largest model
    temperature=0.0,                 # Deterministic
    pubmed_max_results=10,           # More evidence
    min_entity_relevance=5,          # More entities
    enable_pubmed=True               # Full retrieval
)
```

Happy coding! 🚀
