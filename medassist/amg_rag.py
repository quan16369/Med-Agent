"""
AMG-RAG System: Agentic Medical Graph-RAG
Main orchestration system combining LangGraph, MedGemma, and knowledge graphs.

Competition: https://www.kaggle.com/competitions/med-gemma-impact-challenge
Paper: Agentic Medical Graph-RAG (EMNLP 2025)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from medassist.llm.medgemma import get_medgemma_llm
from medassist.core.knowledge_graph import MedicalKnowledgeGraph
from medassist.core.chains import EntityExtractor, RelationExtractor, EntitySummarizer
from medassist.core.hierarchical_optimizer import HierarchicalOptimizer
from medassist.core.medical_disambiguator import MedicalDisambiguator
from medassist.tools.pubmed import PubMedSearcher
from medassist.models.entities import MedicalEntity, MedicalRelation


@dataclass
class AMGState:
    """
    State for AMG-RAG LangGraph workflow.
    Tracks the progression through: Query → Entities → KG → Reasoning → Answer
    """
    query: str
    entities: List[MedicalEntity]
    knowledge_graph: MedicalKnowledgeGraph
    retrieved_evidence: List[Dict]
    reasoning_paths: List[str]
    final_answer: str
    metadata: Dict


class AMG_RAG_System:
    """
    Agentic Medical Graph-RAG System.
    
    Workflow (LangGraph):
    1. Entity Extraction: Extract medical entities from query
    2. Evidence Retrieval: Search PubMed for each entity
    3. Knowledge Graph: Build graph from entities + evidence
    4. Path Reasoning: Explore graph paths with CoT reasoning
    5. Answer Generation: Synthesize final answer from reasoning
    """
    
    def __init__(
        self,
        model_name: str = "medgemma-3-8b",
        temperature: float = 0.0,
        pubmed_max_results: int = 5,
        min_entity_relevance: int = 6,
        enable_pubmed: bool = True,
        enable_optimization: bool = True,
        enable_disambiguation: bool = True
    ):
        """
        Initialize AMG-RAG system with MedGemma.
        
        Args:
            model_name: MedGemma model variant
            temperature: LLM temperature
            pubmed_max_results: Max PubMed articles per entity
            min_entity_relevance: Minimum relevance score (1-10)
            enable_pubmed: Whether to use PubMed retrieval
            enable_optimization: Use SaraCoder-inspired hierarchical optimization
            enable_disambiguation: Use medical term disambiguation
        """
        # Initialize MedGemma LLM
        self.llm = get_medgemma_llm(
            model=model_name,
            temperature=temperature
        )
        
        # Initialize components
        self.entity_extractor = EntityExtractor(self.llm)
        self.relation_extractor = RelationExtractor(self.llm)
        self.summarizer = EntitySummarizer(self.llm)
        
        # SaraCoder-inspired components
        if enable_optimization:
            self.optimizer = HierarchicalOptimizer(
                similarity_threshold=0.85,
                diversity_lambda=0.5,
                max_cluster_size=3
            )
        else:
            self.optimizer = None
        
        if enable_disambiguation:
            self.disambiguator = MedicalDisambiguator(context_window=50)
        else:
            self.disambiguator = None
        
        if enable_pubmed:
            self.pubmed = PubMedSearcher()
        else:
            self.pubmed = None
        
        # Configuration
        self.pubmed_max_results = pubmed_max_results
        self.min_entity_relevance = min_entity_relevance
        self.enable_optimization = enable_optimization
        self.enable_disambiguation = enable_disambiguation
        
        # Build LangGraph workflow
        self.graph = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow for AMG-RAG.
        
        Graph structure:
        START → extract_entities → retrieve_evidence → build_kg → 
        reason_with_paths → generate_answer → END
        """
        workflow = StateGraph(AMGState)
        
        # Add nodes
        workflow.add_node("extract_entities", self._extract_entities_node)
        workflow.add_node("retrieve_evidence", self._retrieve_evidence_node)
        workflow.add_node("build_kg", self._build_kg_node)
        workflow.add_node("reason_with_paths", self._reason_with_paths_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        
        # Add edges
        workflow.add_edge("extract_entities", "retrieve_evidence")
        workflow.add_edge("retrieve_evidence", "build_kg")
        workflow.add_edge("build_kg", "reason_with_paths")
        workflow.add_edge("reason_with_paths", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Set entry point
        workflow.set_entry_point("extract_entities")
        
        return workflow.compile()
    
    def _extract_entities_node(self, state: AMGState) -> AMGState:
        """
        Node 1: Extract medical entities from query.
        Enhanced with:
        - Medical term disambiguation (SaraCoder-inspired)
        - Hierarchical optimization for diversity
        """
        print(f"[1/5] Extracting entities from query...")
        
        entities = self.entity_extractor.extract(
            text=state.query,
            query=state.query,
            min_relevance=self.min_entity_relevance,
            source="query"
        )
        
        print(f"  → Found {len(entities)} entities")
        
        # Disambiguate ambiguous medical terms
        if self.disambiguator and entities:
            entity_dicts = [{
                'name': e.name,
                'entity_type': e.entity_type,
                'confidence': e.confidence,
                'relevance': getattr(e, 'relevance', 5)
            } for e in entities]
            
            resolved = self.disambiguator.resolve_entity_ambiguity(
                entity_dicts,
                context=state.query,
                knowledge_graph=None  # Not available yet
            )
            
            # Update entities with disambiguated names
            for i, entity_dict in enumerate(resolved):
                if 'disambiguation_confidence' in entity_dict:
                    print(f"  → Disambiguated: {entities[i].name} → {entity_dict['name']}")
                    entities[i].name = entity_dict['name']
                    entities[i].entity_type = entity_dict['entity_type']
        
        # Optimize entities for diversity (SaraCoder)
        if self.optimizer and len(entities) > 8:
            entity_dicts = [{
                'name': e.name,
                'entity_type': e.entity_type,
                'confidence': e.confidence,
                'relevance': getattr(e, 'relevance', 5)
            } for e in entities]
            
            optimized = self.optimizer.optimize_entities(
                entity_dicts,
                knowledge_graph=MedicalKnowledgeGraph(),  # Empty KG for now
                max_entities=8
            )
            
            # Reconstruct MedicalEntity objects
            entities = [
                MedicalEntity(
                    name=e['name'],
                    entity_type=e['entity_type'],
                    confidence=e['confidence']
                ) for e in optimized
            ]
            print(f"  → Optimized to {len(entities)} diverse entities")
        
        print(f"  → Final entities:")
        for e in entities:
            print(f"     - {e.name} ({e.entity_type})")
        
        state.entities = entities
        return state
    
    def _retrieve_evidence_node(self, state: AMGState) -> AMGState:
        """
        Node 2: Retrieve evidence from PubMed for each entity.
        Enhanced with:
        - Diversity-based evidence optimization (SaraCoder-inspired)
        - Deduplication and novelty scoring
        """
        print(f"[2/5] Retrieving evidence from PubMed...")
        
        retrieved_evidence = []
        
        if self.pubmed:
            for entity in state.entities:
                print(f"  → Searching PubMed for: {entity.name}")
                
                articles = self.pubmed.search(
                    query=entity.name,
                    max_results=self.pubmed_max_results * 2  # Retrieve more for optimization
                )
                
                for article in articles:
                    retrieved_evidence.append({
                        "entity": entity.name,
                        "title": article.get("title", ""),
                        "abstract": article.get("abstract", ""),
                        "pmid": article.get("pmid", ""),
                        "source": "pubmed",
                        "relevance": 1.0  # Default relevance
                    })
                
                print(f"     - Found {len(articles)} articles")
        
        # Optimize evidence for diversity (SaraCoder)
        if self.optimizer and len(retrieved_evidence) > self.pubmed_max_results * len(state.entities):
            optimized_evidence = self.optimizer.optimize_evidence(
                retrieved_evidence,
                max_evidence=self.pubmed_max_results * len(state.entities)
            )
            print(f"  → Optimized evidence: {len(retrieved_evidence)} → {len(optimized_evidence)} (diverse)")
            retrieved_evidence = optimized_evidence
        
        state.retrieved_evidence = retrieved_evidence
        return state
    
    def _build_kg_node(self, state: AMGState) -> AMGState:
        """
        Node 3: Build knowledge graph from entities and evidence.
        """
        print(f"[3/5] Building knowledge graph...")
        
        kg = MedicalKnowledgeGraph()
        
        # Add extracted entities
        for entity in state.entities:
            kg.add_entity(entity)
        
        # Extract new entities and relations from evidence
        for evidence in state.retrieved_evidence:
            text = f"{evidence['title']}. {evidence['abstract']}"
            
            # Extract entities from evidence
            new_entities = self.entity_extractor.extract(
                text=text,
                query=state.query,
                min_relevance=self.min_entity_relevance - 2,  # Lower threshold for evidence
                source=f"pubmed:{evidence['pmid']}"
            )
            
            for entity in new_entities:
                kg.add_entity(entity)
            
            # Extract relationships
            all_entities = state.entities + new_entities
            relations = self.relation_extractor.extract(
                text=text,
                entities=all_entities,
                min_confidence=0.5,
                source=f"pubmed:{evidence['pmid']}"
            )
            
            for relation in relations:
                kg.add_relation(relation)
        
        stats = kg.get_statistics()
        print(f"  → KG: {stats['num_entities']} entities, {stats['num_relations']} relations")
        
        state.knowledge_graph = kg
        return state
    
    def _reason_with_paths_node(self, state: AMGState) -> AMGState:
        """
        Node 4: Explore reasoning paths through knowledge graph.
        """
        print(f"[4/5] Reasoning with graph paths...")
        
        reasoning_paths = []
        kg = state.knowledge_graph
        
        # Get all entities from graph
        entities = list(kg.entities.keys())
        
        if len(entities) < 2:
            reasoning_paths.append("Insufficient entities for path-based reasoning.")
            state.reasoning_paths = reasoning_paths
            return state
        
        # Find interesting paths between key entities
        key_entities = [e.name.lower() for e in state.entities[:5]]  # Top 5 entities
        
        for i, source in enumerate(key_entities):
            for target in key_entities[i+1:]:
                paths = kg.explore_paths(
                    start_entity=source,
                    end_entity=target,
                    max_length=3
                )
                
                for path in paths[:2]:  # Top 2 paths per pair
                    path_str = self._format_path(path)
                    reasoning_paths.append(path_str)
        
        # Add direct connections for each key entity
        for entity_name in key_entities:
            connected = kg.get_connected_entities(
                entity_name=entity_name,
                max_depth=1
            )
            
            if connected:
                connections_str = f"{entity_name} connections:\n"
                for source, target, data in connected[:5]:  # Top 5
                    rel_type = data.get('relation_type', 'related_to')
                    connections_str += f"  - {source} --[{rel_type}]--> {target}\n"
                reasoning_paths.append(connections_str)
        
        print(f"  → Generated {len(reasoning_paths)} reasoning paths")
        
        state.reasoning_paths = reasoning_paths
        return state
    
    def _generate_answer_node(self, state: AMGState) -> AMGState:
        """
        Node 5: Generate final answer using chain-of-thought reasoning.
        """
        print(f"[5/5] Generating final answer...")
        
        # Prepare context from reasoning paths
        paths_context = "\n\n".join([
            f"Path {i+1}:\n{path}"
            for i, path in enumerate(state.reasoning_paths[:10])
        ])
        
        # Chain-of-thought answer generation
        cot_prompt = f"""Based on the following medical knowledge graph paths, answer the question with detailed reasoning.

Question: {state.query}

Knowledge Graph Reasoning Paths:
{paths_context}

Provide your answer in the following format:
1. Chain of Thought: Step-by-step reasoning process
2. Final Answer: Concise answer to the question
3. Confidence: Your confidence level (low/medium/high)
4. Supporting Evidence: Key evidence supporting your answer

Answer:"""
        
        response = self.llm.invoke(cot_prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        
        state.final_answer = answer
        state.metadata = {
            "num_entities": len(state.entities),
            "num_evidence": len(state.retrieved_evidence),
            "kg_stats": state.knowledge_graph.get_statistics(),
            "num_reasoning_paths": len(state.reasoning_paths)
        }
        
        return state
    
    def _format_path(self, path: List[Tuple[str, str, str]]) -> str:
        """Format a graph path into readable string."""
        if not path:
            return ""
        
        formatted = f"{path[0][0]}"
        for source, relation, target in path:
            formatted += f" --[{relation}]--> {target}"
        
        return formatted
    
    def answer_question(
        self,
        query: str,
        verbose: bool = True
    ) -> Dict:
        """
        Main entry point: Answer medical question using AMG-RAG.
        
        Args:
            query: Medical question
            verbose: Print progress
            
        Returns:
            Dictionary with answer, reasoning paths, and metadata
        """
        # Initialize state
        initial_state = AMGState(
            query=query,
            entities=[],
            knowledge_graph=MedicalKnowledgeGraph(),
            retrieved_evidence=[],
            reasoning_paths=[],
            final_answer="",
            metadata={}
        )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"AMG-RAG: Processing query...")
            print(f"{'='*70}\n")
        
        # Run workflow
        final_state = self.graph.invoke(initial_state)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"ANSWER:")
            print(f"{'='*70}")
            print(final_state.final_answer)
            print(f"\n{'='*70}\n")
        
        return {
            "query": query,
            "answer": final_state.final_answer,
            "entities": [e.name for e in final_state.entities],
            "reasoning_paths": final_state.reasoning_paths,
            "metadata": final_state.metadata
        }
