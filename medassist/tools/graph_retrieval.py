"""
Graph-Conditioned Retrieval Agent
Finds relevant medical knowledge by traversing the knowledge graph
"""

import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from .knowledge_graph import (
    MedicalKnowledgeGraph,
    MedicalEntity,
    KnowledgePath
)
from .medical_ner import BioBERTNER

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from graph-conditioned retrieval"""
    query: str
    paths: List[KnowledgePath]
    query_entities: List[MedicalEntity]
    confidence: float
    
    def to_context(self, max_paths: int = 5) -> str:
        """Convert retrieval result to context string for LLM"""
        context_parts = []
        
        context_parts.append("Knowledge Graph Context:")
        context_parts.append("")
        
        for i, path in enumerate(self.paths[:max_paths]):
            context_parts.append(f"{i+1}. {path.to_string()}")
            context_parts.append(f"   Confidence: {path.confidence:.3f}")
            context_parts.append("")
        
        return "\n".join(context_parts)


class GraphConditionalRetrieval:
    """
    Graph-Conditioned Retrieval Agent
    
    Based on AMG-RAG paper:
    - Extracts entities from query
    - Traverses knowledge graph to find related concepts
    - Returns high-confidence paths for reasoning
    """
    
    def __init__(
        self,
        knowledge_graph: MedicalKnowledgeGraph,
        max_hops: int = 3,
        min_confidence: float = 0.5,
        top_k_paths: int = 20,
        ner_model: Optional[str] = None
    ):
        self.kg = knowledge_graph
        self.max_hops = max_hops
        self.min_confidence = min_confidence
        self.top_k_paths = top_k_paths
        
        # Initialize NER
        if ner_model:
            self.ner = BioBERTNER(model_name=ner_model)
        else:
            self.ner = BioBERTNER()  # Use default model
    
    def retrieve(
        self,
        query: str,
        query_entities: Optional[List[str]] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge paths from graph
        
        Args:
            query: Natural language query
            query_entities: Extracted medical entities (if None, will extract)
        
        Returns:
            RetrievalResult with knowledge paths
        """
        
        # Extract entities if not provided
        if query_entities is None:
            query_entities = self.extract_entities(query)
        
        logger.info(f"Retrieving for query: {query}")
        logger.info(f"Query entities: {query_entities}")
        
        # Find entities in graph
        found_entities = []
        for entity_name in query_entities:
            entity = self.kg.find_entity(entity_name)
            if entity:
                found_entities.append(entity)
        
        if not found_entities:
            logger.warning("No query entities found in knowledge graph")
            return RetrievalResult(
                query=query,
                paths=[],
                query_entities=[],
                confidence=0.0
            )
        
        # Find paths from each query entity
        all_paths = []
        for entity in found_entities:
            paths = self.kg.find_paths(
                start_entity=entity.name,
                max_hops=self.max_hops,
                strategy="bfs",
                min_confidence=self.min_confidence
            )
            all_paths.extend(paths)
        
        # Deduplicate and sort by confidence
        unique_paths = self._deduplicate_paths(all_paths)
        unique_paths.sort(key=lambda x: x.confidence, reverse=True)
        
        # Take top K
        top_paths = unique_paths[:self.top_k_paths]
        
        # Compute overall confidence
        if top_paths:
            avg_confidence = sum(p.confidence for p in top_paths) / len(top_paths)
        else:
            avg_confidence = 0.0
        
        logger.info(f"Retrieved {len(top_paths)} paths with avg confidence {avg_confidence:.3f}")
        
        return RetrievalResult(
            query=query,
            paths=top_paths,
            query_entities=found_entities,
            confidence=avg_confidence
        )
    
    def extract_entities(self, query: str) -> List[str]:
        """
        Extract medical entities from query using BioBERT NER
        """
        
        # Use BioBERT NER to extract entities
        ner_entities = self.ner.extract(query)
        
        # Convert to list of entity texts
        extracted = [entity.text for entity in ner_entities]
        
        return list(set(extracted))  # Remove duplicates
    
    def _deduplicate_paths(self, paths: List[KnowledgePath]) -> List[KnowledgePath]:
        """Remove duplicate paths"""
        seen = set()
        unique = []
        
        for path in paths:
            # Create signature from node IDs
            signature = tuple(node.id for node in path.nodes)
            if signature not in seen:
                seen.add(signature)
                unique.append(path)
        
        return unique
    
    def retrieve_for_entities(
        self,
        entity_names: List[str]
    ) -> List[KnowledgePath]:
        """Retrieve paths starting from specific entities"""
        all_paths = []
        
        for entity_name in entity_names:
            paths = self.kg.find_paths(
                start_entity=entity_name,
                max_hops=self.max_hops,
                min_confidence=self.min_confidence
            )
            all_paths.extend(paths)
        
        # Deduplicate and sort
        unique_paths = self._deduplicate_paths(all_paths)
        unique_paths.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_paths[:self.top_k_paths]
    
    def find_connection(
        self,
        entity1: str,
        entity2: str
    ) -> Optional[KnowledgePath]:
        """
        Find connection path between two entities
        
        Args:
            entity1: First entity name
            entity2: Second entity name
        
        Returns:
            Shortest high-confidence path connecting entities
        """
        paths = self.kg.find_paths(entity1, max_hops=5)
        
        # Filter paths that end at entity2
        connecting_paths = []
        for path in paths:
            if path.nodes[-1].name.lower() == entity2.lower():
                connecting_paths.append(path)
        
        if not connecting_paths:
            return None
        
        # Return shortest path with highest confidence
        connecting_paths.sort(key=lambda x: (len(x), -x.confidence))
        return connecting_paths[0]


class MultiHopReasoning:
    """
    Multi-hop reasoning over knowledge graph
    
    Combines multiple paths to answer complex queries
    """
    
    def __init__(self, retrieval_agent: GraphConditionalRetrieval):
        self.retrieval = retrieval_agent
    
    def reason(
        self,
        query: str,
        max_reasoning_steps: int = 3
    ) -> Dict:
        """
        Perform multi-hop reasoning to answer query
        
        Args:
            query: Complex medical query
            max_reasoning_steps: Maximum reasoning hops
        
        Returns:
            Reasoning result with paths and intermediate steps
        """
        
        # Step 1: Initial retrieval
        result = self.retrieval.retrieve(query)
        
        if not result.paths:
            return {
                "success": False,
                "message": "No relevant knowledge found",
                "paths": [],
                "steps": []
            }
        
        # Step 2: Identify key entities
        key_entities = self._identify_key_entities(result.paths)
        
        # Step 3: Expand reasoning with additional hops
        reasoning_steps = []
        expanded_paths = result.paths.copy()
        
        for step in range(max_reasoning_steps):
            # Get new entities from current paths
            new_entities = []
            for path in expanded_paths[-10:]:  # Check recent paths
                for node in path.nodes:
                    if node.name not in key_entities:
                        new_entities.append(node.name)
                        key_entities.add(node.name)
            
            if not new_entities:
                break
            
            # Retrieve from new entities
            new_paths = self.retrieval.retrieve_for_entities(new_entities[:5])
            
            reasoning_steps.append({
                "step": step + 1,
                "entities_explored": new_entities[:5],
                "paths_found": len(new_paths)
            })
            
            expanded_paths.extend(new_paths)
        
        # Step 4: Rank final paths by confidence
        expanded_paths.sort(key=lambda x: x.confidence, reverse=True)
        
        return {
            "success": True,
            "paths": expanded_paths[:20],
            "reasoning_steps": reasoning_steps,
            "total_entities": len(key_entities),
            "confidence": result.confidence
        }
    
    def _identify_key_entities(self, paths: List[KnowledgePath]) -> Set[str]:
        """Identify key entities from paths"""
        entities = set()
        for path in paths:
            for node in path.nodes:
                entities.add(node.name)
        return entities


if __name__ == "__main__":
    # Demo
    from .knowledge_graph import MedicalRelationship
    
    print("Graph-Conditioned Retrieval Demo")
    print("="*60)
    
    # Setup knowledge graph
    kg = MedicalKnowledgeGraph(use_memory=True)
    
    # Add sample medical knowledge
    entities = [
        MedicalEntity("d1", "Diabetes", "disease", ["DM", "diabetes mellitus"]),
        MedicalEntity("s1", "Hyperglycemia", "symptom", ["high blood sugar"]),
        MedicalEntity("d2", "Diabetic Neuropathy", "disease", ["nerve damage"]),
        MedicalEntity("s2", "Numbness", "symptom", ["loss of sensation"]),
        MedicalEntity("t1", "Insulin", "treatment", ["insulin therapy"])
    ]
    
    for e in entities:
        kg.add_entity(e)
    
    relationships = [
        MedicalRelationship("d1", "s1", "causes", 0.95, 100),
        MedicalRelationship("d1", "d2", "causes", 0.88, 75),
        MedicalRelationship("d2", "s2", "causes", 0.92, 60),
        MedicalRelationship("t1", "d1", "treats", 0.97, 200),
        MedicalRelationship("t1", "s1", "reduces", 0.93, 150)
    ]
    
    for r in relationships:
        kg.add_relationship(r)
    
    # Initialize retrieval agent
    retrieval = GraphConditionalRetrieval(kg, max_hops=3)
    
    # Test query
    query = "What causes numbness in diabetes patients?"
    print(f"\nQuery: {query}")
    
    result = retrieval.retrieve(query)
    
    print(f"\nExtracted entities: {[e.name for e in result.query_entities]}")
    print(f"Found {len(result.paths)} paths")
    print(f"Average confidence: {result.confidence:.3f}")
    
    print("\nTop paths:")
    for i, path in enumerate(result.paths[:5]):
        print(f"\n{i+1}. {path.to_string()}")
        print(f"   Confidence: {path.confidence:.3f}")
    
    # Test multi-hop reasoning
    print("\n" + "="*60)
    print("Multi-Hop Reasoning Demo")
    
    reasoner = MultiHopReasoning(retrieval)
    reasoning_result = reasoner.reason(query, max_reasoning_steps=2)
    
    print(f"\nReasoning success: {reasoning_result['success']}")
    print(f"Total paths: {len(reasoning_result['paths'])}")
    print(f"Reasoning steps: {len(reasoning_result['reasoning_steps'])}")
