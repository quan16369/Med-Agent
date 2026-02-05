# MedGemma Production Setup

Flexible LLM provider with auto-detection:
- **Groq** (preferred): Fast, free API
- **OpenAI**: Reliable fallback
- **Ollama**: Local inference

## 🚀 Quick Setup

### Option 1: Groq (Recommended for Hackathons)

```bash
# Free API: https://console.groq.com
export GROQ_API_KEY="your-groq-key"

# Test
python cli.py --query "What treats diabetes?"
```

### Option 2: OpenAI

```bash
export OPENAI_API_KEY="your-openai-key"

# Test
python cli.py --query "What treats diabetes?"
```

### Option 3: Ollama (Local)

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2

# Test (no API key needed)
python cli.py --query "What treats diabetes?"
```

## Auto-Detection

System automatically selects provider in priority order:

1. **Groq** (if `GROQ_API_KEY` is set)
2. **OpenAI** (if `OPENAI_API_KEY` is set) 
3. **Ollama** (local fallback)

```python
from simple_agent import SimpleAgent

# Auto-detect best available
agent = SimpleAgent()

# Or specify provider
from medassist.core import LangGraphMedicalOrchestrator

orchestrator = LangGraphMedicalOrchestrator(
    llm_provider="groq",  # or "openai", "ollama", "auto"
    model_name="llama-3.3-70b-versatile"
)
```

## 📊 Models

| Provider | Model | Speed | Cost | Best For |
|----------|-------|-------|------|----------|
| Groq | llama-3.3-70b-versatile | ⚡⚡⚡ | Free | Hackathons |
| OpenAI | gpt-4o-mini | ⚡⚡ | $0.15/1M | Production |
| Ollama | llama3.2 | ⚡ | Free | Offline |

## 🔧 Configuration

```python
# simple_agent.py - Auto mode
agent = SimpleAgent()  # Uses best available

# Explicit provider
orchestrator = LangGraphMedicalOrchestrator(
    llm_provider="groq",
    model_name="llama-3.3-70b-versatile"
)

# NER also auto-detects
from medassist.tools.medical_ner import AMGRAGEntityExtractor

extractor = AMGRAGEntityExtractor(
    llm_provider="auto"  # Groq > OpenAI > Ollama
)
```

## ✅ Hackathon Ready

- ✅ Groq API (free, fast)
- ✅ OpenAI fallback
- ✅ Ollama local
- ✅ Auto-detection
- ✅ No hardcoded dependencies

Tested with all 3 providers ✨
