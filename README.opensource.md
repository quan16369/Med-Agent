# MedGemma - Open-Source Medical Multi-Agent System

Production structure with **open-source models only** (Hackathon compliant)

## 🎯 Open-Source Stack

- **LLM**: Ollama (llama3.2 local)
- **NER**: LangChain structured output (AMG-RAG style)
- **Embeddings**: SentenceTransformers
- **Framework**: LangGraph + LangChain
- **KG**: NetworkX

## 📁 Structure

```
medassist/
├── models/          # Data models
├── agents/          # Specialized agents
├── tools/           # NER, PubMed, embeddings
├── core/            # LangGraph orchestration
├── services/        # Business logic
├── schemas/         # API validation
└── api/             # FastAPI endpoints
```

## 🚀 Setup

```bash
# Install Ollama (required for local LLM)
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2

# Install dependencies
pip install -r requirements.txt

# Test
python cli.py --query "What treats diabetes?"
```

## 🔧 Models

### 1. **Entity Extraction** (AMG-RAG style)
```python
from medassist.tools.medical_ner import AMGRAGEntityExtractor

# Uses Ollama (open-source)
extractor = AMGRAGEntityExtractor(
    use_openai=False,  # Open-source only
    model_name="llama3.2"
)

entities = extractor.extract_entities(
    "Patient has diabetes and hypertension"
)
```

### 2. **Embeddings** (SentenceTransformers)
```python
from medassist.tools.medical_ner import MedicalEmbeddings

# Open-source embeddings
embeddings = MedicalEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query("diabetes treatment")
```

### 3. **LangGraph Orchestrator** (Ollama)
```python
from medassist.core import LangGraphMedicalOrchestrator

# Uses Ollama by default
orchestrator = LangGraphMedicalOrchestrator(
    llm_provider="ollama",  # Open-source
    model_name="llama3.2"
)
```

## 📊 AMG-RAG Features

Following [AMG-RAG paper](https://github.com/MrRezaeiUofT/AMG-RAG):

- ✅ **LLM-based NER** (not BioBERT)
- ✅ **Relevance scoring** (1-10 scale)
- ✅ **Bidirectional relationships**
- ✅ **Entity summarization**
- ✅ **Context-aware descriptions**

## 🎨 Usage

```python
from simple_agent import SimpleAgent

# Initialize with Ollama
agent = SimpleAgent()

# Ask question
result = agent.ask("What is the first-line treatment for type 2 diabetes?")

print(result['answer'])
print(f"Confidence: {result['confidence']:.2%}")
```

## 🌐 API

```bash
# Start API
uvicorn medassist.api.main:app --port 8000

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What treats hypertension?"}'
```

## 📦 No Proprietary Models

- ❌ OpenAI API (optional, not required)
- ❌ Groq API (optional, not required)
- ✅ Ollama (local, open-source)
- ✅ SentenceTransformers (open-source)
- ✅ LangChain (open-source framework)

## 🔬 Models Used

| Component | Model | License |
|-----------|-------|---------|
| LLM | llama3.2 (Ollama) | Open |
| Embeddings | all-MiniLM-L6-v2 | Apache 2.0 |
| Framework | LangGraph | MIT |
| KG | NetworkX | BSD |

## 🧪 Test

```bash
# Test NER
python -c "from medassist.tools.medical_ner import AMGRAGEntityExtractor; \
           e = AMGRAGEntityExtractor(use_openai=False); \
           print(e.extract_entities('Patient has diabetes'))"

# Test full workflow
python test_langgraph.py
```

## 📝 Configuration

```python
# simple_agent.py
agent = SimpleAgent(
    use_memory_graph=True,
    load_sample_graph=True
)

# Uses Ollama by default (no API keys needed)
```

Built with open-source stack for hackathons 🚀
