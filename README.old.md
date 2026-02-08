# AMG-RAG - Fresh Start

Clean implementation based on: https://github.com/MrRezaeiUofT/AMG-RAG

## ✅ What's Ready

### Phase 1: Core Components

1. **Data Structures** ([medassist/models/entities.py](medassist/models/entities.py))
   - `MedicalEntity` - with confidence scoring
   - `MedicalRelation` - bidirectional relationships
   - `MEDICAL_RELATION_TYPES` - medical relationship types

2. **PubMed Integration** ([medassist/tools/pubmed.py](medassist/tools/pubmed.py))
   - `PubMedSearcher` - medical literature retrieval
   - Handles API rate limits
   - XML parsing for abstracts

3. **Tests**
   - `test_basic.py` - Test data structures ✅
   - `test_pubmed.py` - Test PubMed integration

## Quick Test

```bash
# Test basic structures
python test_basic.py

# Test PubMed (requires internet)
python test_pubmed.py
```

## Next Steps

See [AMG_RAG_START.md](AMG_RAG_START.md) for full build plan:

1. [ ] Knowledge Graph (NetworkX-based)
2. [ ] LLM Chains (entity extraction, relationship extraction) 
3. [ ] AMG-RAG System (main pipeline)
4. [ ] Testing with MEDQA dataset

## Project Structure

```
medassist/
├── models/
│   ├── entities.py        ✅ MedicalEntity, MedicalRelation
│   └── __init__.py        ✅
├── tools/
│   ├── pubmed.py          ✅ PubMedSearcher
│   └── __init__.py        ✅
├── core/
│   └── __init__.py        ✅
└── __init__.py            ✅
```

## Key Features (from AMG-RAG Paper)

1. **Entity Extraction** - with 1-10 relevance scoring
2. **Bidirectional Relations** - A→B and B→A analysis
3. **Knowledge Graph** - NetworkX-based dynamic construction
4. **Multi-source Evidence** - PubMed + Wikipedia + Vector DB
5. **Chain-of-Thought** - Structured reasoning synthesis

## Performance (AMG-RAG Paper)

| Dataset | Score | Metric |
|---------|-------|--------|
| MEDQA | 74.1% | F1 Score |
| MEDMCQA | 66.34% | Accuracy |

Build incrementally following the AMG-RAG architecture.
