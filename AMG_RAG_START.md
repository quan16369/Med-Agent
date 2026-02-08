# AMG-RAG Implementation Plan

Building from: https://github.com/MrRezaeiUofT/AMG-RAG

## Phase 1: Core Components (Start Here)

### 1. Data Structures
- [x] MedicalEntity (entity with confidence scoring)
- [x] MedicalRelation (bidirectional relationships)  
- [x] MedicalKnowledgeGraph (NetworkX-based graph)

### 2. External APIs
- [x] PubMedSearcher (medical literature)
- [ ] Wikipedia integration

### 3. LLM Chains
- [ ] Entity extraction with relevance scoring (1-10)
- [ ] Relationship extraction (bidirectional)
- [ ] Entity summarization
- [ ] Chain-of-thought reasoning
- [ ] Final answer generation

## Phase 2: AMG-RAG System

- [ ] AMG_RAG_System class
- [ ] build_knowledge_graph()
- [ ] reason_with_graph()
- [ ] answer_question()

## Phase 3: Integration

- [ ] Vector database (Chroma)
- [ ] LangGraph workflow
- [ ] Testing with MEDQA dataset

## Key Features from AMG-RAG

1. **Relevance Scoring**: 1-10 scale for entity importance
2. **Bidirectional Relationships**: A→B and B→A analysis  
3. **Context Integration**: PubMed + Wikipedia
4. **Entity Summarization**: LLM-generated descriptions
5. **Graph Reasoning**: Path exploration with confidence

## Current File Structure

```
medassist/
├── core/
│   └── amg_rag_system.py     # Main AMG-RAG implementation
├── models/
│   ├── entities.py            # MedicalEntity, MedicalRelation
│   └── knowledge_graph.py     # MedicalKnowledgeGraph
├── tools/
│   ├── pubmed.py             # PubMedSearcher
│   └── llm_chains.py         # LLM extraction chains
└── __init__.py
```

## Next Steps

1. Create minimal data structures
2. Build PubMed integration
3. Implement entity extraction
4. Build knowledge graph
5. Add reasoning layer
