# AMG-RAG: Agentic Medical Knowledge Graph Architecture

## Paper Information

**Title**: "Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge"

**Authors**: Mohammad Reza Rezaei, Reza Saadati Fard, Jayson L. Parker, Rahul G. Krishnan, Milad Lankarany

**Conference**: EMNLP 2025 Findings (Paper 679)

**Link**: https://aclanthology.org/2025.findings-emnlp.679.pdf

## Key Innovation

Instead of flat document retrieval (traditional RAG), AMG-RAG builds a **dynamic Medical Knowledge Graph** that:
- Automatically extracts medical entities and relationships
- Maintains confidence scores for relationships
- Enables multi-hop reasoning
- Continuously updates from latest research

**Result**: 74.1% F1 on MEDQA with 8B model (outperforms 70B+ models)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Medical Query                         │
│            "What causes diabetic neuropathy?"            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Medical Entity Recognizer                   │
│  Extract: ["diabetic neuropathy", "causes", "diabetes"] │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         Medical Knowledge Graph (Neo4j)                  │
│                                                          │
│   Diabetes ──[causes:0.95]──> Diabetic Neuropathy      │
│       │                              │                   │
│       └──[affects:0.90]──> Blood Vessels                │
│                              │                           │
│                              └──[leads_to:0.85]──> Nerve Damage │
│                                                          │
│   ~76,681 nodes, ~354,299 edges                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│        Graph-Conditioned Retrieval Agent                 │
│  - BFS/DFS traversal from query entities                │
│  - Multi-hop path finding (1-3 hops)                    │
│  - Confidence propagation along paths                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Evidence Retrieval Agent                    │
│  - Search PubMed for abstracts supporting paths         │
│  - Retrieve Wikipedia medical articles                  │
│  - Rerank by relevance + confidence                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           Chain-of-Thought Reasoning Agent               │
│  1. Analyze retrieved evidence                          │
│  2. Trace reasoning paths through graph                 │
│  3. Synthesize final answer with confidence             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    Final Answer                          │
│  "Diabetic neuropathy is caused by prolonged high blood │
│   sugar levels damaging blood vessels and nerves..."    │
│  Confidence: 0.92                                        │
│  Supporting evidence: [PubMed IDs, graph paths]         │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Medical Knowledge Graph (MKG)

**Database**: Neo4j (graph database)

**Structure**:
```
Node types:
- Disease (e.g., "Diabetes Mellitus")
- Symptom (e.g., "Hyperglycemia")
- Treatment (e.g., "Metformin")
- Anatomy (e.g., "Pancreas")
- Biomarker (e.g., "HbA1c")

Relationship types:
- causes (Disease → Symptom)
- treats (Treatment → Disease)
- affects (Disease → Anatomy)
- indicates (Biomarker → Disease)
- contraindicated_with (Treatment ⇄ Disease)

Relationship attributes:
- confidence: float [0-1]
- evidence_count: int
- source: ["pubmed", "wikipedia", "textbook"]
- last_updated: timestamp
```

**Statistics**:
- 76,681 nodes
- 354,299 relationships
- Average node degree: 9.2
- Max path length: 5

### 2. Automated Graph Construction Pipeline

```python
class MedicalKnowledgeGraphBuilder:
    """
    Automatically constructs and updates medical knowledge graph
    """
    
    def __init__(self):
        self.neo4j_client = Neo4jClient()
        self.entity_recognizer = MedicalNER()
        self.relation_extractor = LLMRelationExtractor()
        self.evidence_retriever = PubMedRetriever()
    
    def build_from_corpus(self, corpus_path: str):
        """Build graph from medical corpus"""
        
        # Phase 1: Extract entities
        entities = []
        for document in self.load_corpus(corpus_path):
            entities.extend(self.entity_recognizer.extract(document))
        
        # Phase 2: Infer relationships
        relationships = []
        for entity_pair in self.get_entity_pairs(entities):
            relation = self.relation_extractor.infer(entity_pair)
            if relation.confidence > 0.7:
                relationships.append(relation)
        
        # Phase 3: Validate with evidence
        for relation in relationships:
            evidence = self.evidence_retriever.search(
                f"{relation.source} {relation.type} {relation.target}"
            )
            relation.evidence_count = len(evidence)
            relation.confidence *= self.evidence_score(evidence)
        
        # Phase 4: Insert into Neo4j
        self.neo4j_client.batch_insert(entities, relationships)
        
        return {
            "nodes": len(entities),
            "edges": len(relationships)
        }
    
    def update_continuously(self):
        """Continuously update graph from latest research"""
        
        # Query PubMed for recent papers
        recent_papers = self.evidence_retriever.get_recent(days=30)
        
        for paper in recent_papers:
            # Extract new entities and relationships
            entities = self.entity_recognizer.extract(paper.abstract)
            relationships = self.relation_extractor.infer_from_text(paper.abstract)
            
            # Update graph with new knowledge
            self.neo4j_client.merge_entities(entities)
            self.neo4j_client.merge_relationships(relationships)
        
        logger.info(f"Updated graph with {len(recent_papers)} papers")
```

### 3. Graph-Conditioned Retrieval

```python
class GraphConditionalRetrieval:
    """
    Retrieve relevant knowledge using graph structure
    """
    
    def __init__(self, neo4j_client):
        self.graph = neo4j_client
    
    def retrieve(
        self,
        query_entities: List[str],
        max_hops: int = 3,
        strategy: str = "bfs"
    ) -> List[Path]:
        """
        Retrieve knowledge paths from graph
        
        Args:
            query_entities: Extracted entities from query
            max_hops: Maximum path length
            strategy: "bfs" (breadth-first) or "dfs" (depth-first)
        """
        
        paths = []
        
        for entity in query_entities:
            # Find node in graph
            start_node = self.graph.find_node(entity)
            
            if not start_node:
                continue
            
            # Traverse graph to find related entities
            if strategy == "bfs":
                related = self.bfs_traverse(start_node, max_hops)
            else:
                related = self.dfs_traverse(start_node, max_hops)
            
            # Extract paths with confidence scores
            for path in related:
                path_confidence = self.compute_path_confidence(path)
                if path_confidence > 0.5:
                    paths.append({
                        "path": path,
                        "confidence": path_confidence,
                        "hops": len(path) - 1
                    })
        
        # Sort by confidence and return top K
        paths.sort(key=lambda x: x["confidence"], reverse=True)
        return paths[:20]
    
    def compute_path_confidence(self, path: List[Node]) -> float:
        """
        Compute confidence score for a path
        Product of edge confidences (geometric mean)
        """
        confidences = []
        for i in range(len(path) - 1):
            edge = self.graph.get_edge(path[i], path[i+1])
            confidences.append(edge.confidence)
        
        # Geometric mean
        if not confidences:
            return 0.0
        
        product = 1.0
        for conf in confidences:
            product *= conf
        
        return product ** (1.0 / len(confidences))
    
    def bfs_traverse(self, start_node: Node, max_hops: int) -> List[List[Node]]:
        """Breadth-first traversal"""
        from collections import deque
        
        queue = deque([(start_node, [start_node])])
        visited = set([start_node.id])
        paths = []
        
        while queue:
            node, path = queue.popleft()
            
            if len(path) > max_hops:
                continue
            
            paths.append(path)
            
            # Explore neighbors
            for neighbor in self.graph.get_neighbors(node):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
```

### 4. Evidence Retrieval Integration

```python
class EvidenceRetrievalAgent:
    """
    Retrieve supporting evidence from PubMed and medical literature
    """
    
    def __init__(self):
        self.pubmed_client = PubMedAPI()
        self.reranker = CrossEncoderReranker()
    
    def retrieve_for_path(
        self,
        path: List[Node],
        query: str,
        max_results: int = 5
    ) -> List[Dict]:
        """
        Retrieve evidence supporting a knowledge graph path
        """
        
        # Construct search query from path
        search_query = self.path_to_query(path)
        
        # Search PubMed
        papers = self.pubmed_client.search(
            query=search_query,
            max_results=max_results * 2  # Retrieve more for reranking
        )
        
        # Rerank by relevance to original query
        reranked = self.reranker.rerank(
            query=query,
            documents=[p.abstract for p in papers]
        )
        
        # Return top K with metadata
        evidence = []
        for paper, score in reranked[:max_results]:
            evidence.append({
                "title": paper.title,
                "abstract": paper.abstract,
                "pmid": paper.pmid,
                "relevance_score": score,
                "year": paper.year
            })
        
        return evidence
    
    def path_to_query(self, path: List[Node]) -> str:
        """Convert knowledge graph path to search query"""
        
        # Extract key terms from path
        terms = [node.name for node in path]
        
        # Add relationship types
        relations = []
        for i in range(len(path) - 1):
            edge = self.graph.get_edge(path[i], path[i+1])
            relations.append(edge.type)
        
        # Combine into query
        query = " ".join(terms)
        if relations:
            query += " " + " ".join(relations)
        
        return query
```

### 5. Chain-of-Thought Reasoning

```python
class ChainOfThoughtReasoning:
    """
    Generate interpretable reasoning from graph paths and evidence
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def reason(
        self,
        query: str,
        graph_paths: List[Dict],
        evidence: List[Dict]
    ) -> Dict:
        """
        Generate step-by-step reasoning to answer query
        """
        
        # Build reasoning prompt
        prompt = self.build_prompt(query, graph_paths, evidence)
        
        # Generate reasoning with CoT
        reasoning = self.llm.generate(
            prompt,
            temperature=0.3,  # Lower for consistency
            max_tokens=512
        )
        
        # Parse reasoning steps
        steps = self.parse_reasoning(reasoning)
        
        # Extract final answer
        answer = self.extract_answer(reasoning)
        
        # Compute confidence
        confidence = self.compute_confidence(graph_paths, evidence, reasoning)
        
        return {
            "answer": answer,
            "reasoning": reasoning,
            "steps": steps,
            "confidence": confidence,
            "supporting_paths": graph_paths[:3],
            "supporting_evidence": evidence[:3]
        }
    
    def build_prompt(
        self,
        query: str,
        graph_paths: List[Dict],
        evidence: List[Dict]
    ) -> str:
        """Build prompt with graph context and evidence"""
        
        prompt = f"""Question: {query}

Knowledge Graph Context:
"""
        
        # Add top graph paths
        for i, path in enumerate(graph_paths[:3]):
            path_str = " → ".join([node.name for node in path["path"]])
            prompt += f"{i+1}. {path_str} (confidence: {path['confidence']:.2f})\n"
        
        prompt += "\nSupporting Evidence:\n"
        
        # Add evidence abstracts
        for i, ev in enumerate(evidence[:3]):
            prompt += f"{i+1}. {ev['title']}\n"
            prompt += f"   {ev['abstract'][:200]}...\n\n"
        
        prompt += """
Based on the knowledge graph relationships and supporting evidence, provide step-by-step reasoning to answer the question:

Step 1: [Identify key concepts]
Step 2: [Trace relationships in knowledge graph]
Step 3: [Integrate evidence from literature]
Step 4: [Synthesize final answer]

Answer:"""
        
        return prompt
```

## Performance Results

### MEDQA Dataset
- **Accuracy**: 73.92%
- **F1 Score**: 74.1%
- **Improvement over baseline RAG**: +8.3%

### MEDMCQA Dataset
- **Accuracy**: 66.34%
- **Improvement over baseline**: +7.2%

### Comparison with Other Models
| Model | Parameters | MEDQA Acc | MEDMCQA Acc |
|-------|-----------|-----------|-------------|
| GPT-4 | 1.76T | 81.4% | 72.3% |
| Med-PaLM 2 | 540B | 79.7% | 71.3% |
| **AMG-RAG** | **8B** | **73.9%** | **66.3%** |
| Baseline RAG | 8B | 65.6% | 59.1% |

**Key insight**: AMG-RAG with 8B parameters outperforms vanilla RAG and approaches models 100× larger.

## Implementation for MedGemma

### Required Infrastructure

1. **Neo4j Database**
```bash
# Install Neo4j
docker pull neo4j:latest
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

2. **Medical NER Model**
```python
# Use BioBERT or PubMedBERT for entity recognition
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForTokenClassification.from_pretrained("alvaroalon2/biobert_diseases_ner")
```

3. **PubMed API Integration**
```python
from Bio import Entrez

Entrez.email = "your_email@example.com"

def search_pubmed(query, max_results=10):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    results = Entrez.read(handle)
    return results["IdList"]
```

## Migration Path for MedGemma

### Phase 1: Build Initial Knowledge Graph (1-2 weeks)
- Extract entities from existing medical corpus
- Build Neo4j database with ~50K nodes
- Validate with manual review

### Phase 2: Integrate Graph Retrieval (1 week)
- Replace flat RAG with graph-conditioned retrieval
- Implement BFS traversal
- Add confidence scoring

### Phase 3: Add Evidence Retrieval (1 week)
- Integrate PubMed API
- Implement reranking
- Add citation tracking

### Phase 4: Chain-of-Thought Integration (1 week)
- Update prompts with graph context
- Add reasoning step extraction
- Validate on MEDQA benchmark

### Phase 5: Continuous Updates (ongoing)
- Automated graph updates from PubMed
- Confidence score recalibration
- Performance monitoring

## Expected Improvements

Compared to current MedGemma system:
- **+8-10% accuracy** on medical QA
- **Better interpretability** (graph paths + reasoning traces)
- **Dynamic knowledge** (auto-updates from research)
- **Multi-hop reasoning** (handle complex queries)
- **Confidence scores** (know when unsure)

## Advantages Over Flat RAG

1. **Structured Knowledge**: Graph vs flat documents
2. **Multi-hop Reasoning**: Can connect distant concepts
3. **Confidence Tracking**: Score relationships and paths
4. **Continuous Learning**: Easy to add new knowledge
5. **Interpretability**: See reasoning paths through graph
6. **Scalability**: Graph queries more efficient than full-text search

## References

1. AMG-RAG Paper (EMNLP 2025)
   - https://aclanthology.org/2025.findings-emnlp.679.pdf

2. Neo4j Graph Database
   - https://neo4j.com/docs/

3. BioBERT for Medical NER
   - https://github.com/dmis-lab/biobert

4. PubMed API (Entrez)
   - https://www.ncbi.nlm.nih.gov/books/NBK25501/

## Next Steps

1. Review architecture and confirm alignment with competition goals
2. Set up Neo4j database infrastructure
3. Extract medical entities from existing corpus
4. Build initial knowledge graph
5. Implement graph-conditioned retrieval
6. Benchmark against current system
7. Iterate and optimize

This architecture represents state-of-the-art for medical QA and should significantly improve competition performance.
