"""
Medical QA Orchestrator with Knowledge Graph
Combines graph reasoning with multi-hop retrieval for better accuracy
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from .knowledge_graph import MedicalKnowledgeGraph, MedicalEntity, MedicalRelationship
from .graph_retrieval import GraphConditionalRetrieval, MultiHopReasoning, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class AMGRAGResponse:
    """Response from AMG-RAG system"""
    query: str
    answer: str
    reasoning_trace: str
    knowledge_paths: List[str]
    confidence: float
    processing_time: float
    graph_stats: Dict


class AMGRAGOrchestrator:
    """
    Medical QA system using knowledge graph reasoning
    
    Workflow:
    1. Extract medical entities from query
    2. Find relevant paths in knowledge graph
    3. Retrieve supporting evidence from literature
    4. Generate answer with reasoning trace
    
    Combines graph-based retrieval with traditional RAG
    """
    
    def __init__(
        self,
        model_name: str = "google/medgemma-2b",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        use_memory_graph: bool = True,
        load_pretrained_graph: bool = False
    ):
        self.model_name = model_name
        
        logger.info("Initializing AMG-RAG Orchestrator")
        
        # Initialize knowledge graph
        logger.info("Setting up Medical Knowledge Graph...")
        self.knowledge_graph = MedicalKnowledgeGraph(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            use_memory=use_memory_graph
        )
        
        if load_pretrained_graph:
            self._load_pretrained_graph()
        
        # Initialize graph retrieval
        self.retrieval_agent = GraphConditionalRetrieval(
            knowledge_graph=self.knowledge_graph,
            max_hops=3,
            min_confidence=0.5,
            top_k_paths=20
        )
        
        # Initialize multi-hop reasoning
        self.reasoning_agent = MultiHopReasoning(self.retrieval_agent)
        
        # Initialize LLM (placeholder - use actual model)
        self._init_model()
        
        logger.info("AMG-RAG Orchestrator initialized successfully")
    
    def _init_model(self):
        """Initialize language model"""
        # TODO: Load actual MedGemma model
        logger.info(f"Initializing model: {self.model_name}")
        self.model = None  # Placeholder
    
    def _load_pretrained_graph(self):
        """Load pretrained medical knowledge graph"""
        logger.info("Loading pretrained medical knowledge graph...")
        
        # TODO: Load from serialized graph file or database
        # For now, add some sample medical knowledge
        self._add_sample_knowledge()
    
    def _add_sample_knowledge(self):
        """Add sample medical knowledge to graph"""
        
        # Sample diseases
        diseases = [
            MedicalEntity("disease_diabetes", "Diabetes Mellitus", "disease", 
                         ["diabetes", "DM", "diabetes mellitus type 2"]),
            MedicalEntity("disease_hypertension", "Hypertension", "disease",
                         ["high blood pressure", "HTN"]),
            MedicalEntity("disease_neuropathy", "Diabetic Neuropathy", "disease",
                         ["nerve damage", "peripheral neuropathy"]),
            MedicalEntity("disease_retinopathy", "Diabetic Retinopathy", "disease",
                         ["eye damage", "vision loss"]),
            MedicalEntity("disease_chd", "Coronary Heart Disease", "disease",
                         ["CHD", "coronary artery disease", "heart disease"]),
        ]
        
        # Sample symptoms
        symptoms = [
            MedicalEntity("symptom_hyperglycemia", "Hyperglycemia", "symptom",
                         ["high blood sugar", "elevated glucose"]),
            MedicalEntity("symptom_numbness", "Numbness", "symptom",
                         ["loss of sensation", "tingling"]),
            MedicalEntity("symptom_blurred_vision", "Blurred Vision", "symptom",
                         ["vision problems", "unclear vision"]),
            MedicalEntity("symptom_chest_pain", "Chest Pain", "symptom",
                         ["angina", "chest discomfort"]),
        ]
        
        # Sample treatments
        treatments = [
            MedicalEntity("treatment_metformin", "Metformin", "treatment",
                         ["glucophage", "metformin HCl"]),
            MedicalEntity("treatment_insulin", "Insulin", "treatment",
                         ["insulin therapy", "insulin injection"]),
            MedicalEntity("treatment_statin", "Statin", "treatment",
                         ["atorvastatin", "simvastatin", "cholesterol medication"]),
        ]
        
        # Add all entities
        for entity_list in [diseases, symptoms, treatments]:
            for entity in entity_list:
                self.knowledge_graph.add_entity(entity)
        
        # Sample relationships
        relationships = [
            # Diabetes causes symptoms
            MedicalRelationship("disease_diabetes", "symptom_hyperglycemia", 
                              "causes", 0.95, 150, "pubmed"),
            
            # Diabetes causes complications
            MedicalRelationship("disease_diabetes", "disease_neuropathy",
                              "causes", 0.88, 75, "pubmed"),
            MedicalRelationship("disease_diabetes", "disease_retinopathy",
                              "causes", 0.85, 80, "pubmed"),
            MedicalRelationship("disease_diabetes", "disease_chd",
                              "increases_risk", 0.82, 120, "pubmed"),
            
            # Neuropathy causes symptoms
            MedicalRelationship("disease_neuropathy", "symptom_numbness",
                              "causes", 0.92, 60, "pubmed"),
            
            # Retinopathy causes symptoms
            MedicalRelationship("disease_retinopathy", "symptom_blurred_vision",
                              "causes", 0.90, 55, "pubmed"),
            
            # CHD causes symptoms
            MedicalRelationship("disease_chd", "symptom_chest_pain",
                              "causes", 0.93, 100, "pubmed"),
            
            # Treatments
            MedicalRelationship("treatment_metformin", "disease_diabetes",
                              "treats", 0.97, 200, "pubmed"),
            MedicalRelationship("treatment_metformin", "symptom_hyperglycemia",
                              "reduces", 0.94, 180, "pubmed"),
            MedicalRelationship("treatment_insulin", "disease_diabetes",
                              "treats", 0.98, 250, "pubmed"),
            MedicalRelationship("treatment_statin", "disease_chd",
                              "treats", 0.91, 150, "pubmed"),
        ]
        
        for rel in relationships:
            self.knowledge_graph.add_relationship(rel)
        
        logger.info(f"Added sample knowledge: {len(diseases + symptoms + treatments)} entities, "
                   f"{len(relationships)} relationships")
    
    def process(self, query: str) -> AMGRAGResponse:
        """
        Process medical query through knowledge graph reasoning
        
        Args:
            query: Medical question
        
        Returns:
            Response with answer, reasoning paths, and confidence score
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {query}")
        
        # Step 1: Graph-conditioned retrieval
        retrieval_result = self.retrieval_agent.retrieve(query)
        
        if not retrieval_result.paths:
            logger.warning("No knowledge paths found")
            return self._generate_fallback_response(query, time.time() - start_time)
        
        # Step 2: Multi-hop reasoning
        reasoning_result = self.reasoning_agent.reason(query, max_reasoning_steps=2)
        
        # Step 3: Generate answer with chain-of-thought
        answer, reasoning_trace = self._generate_answer_with_cot(
            query,
            retrieval_result,
            reasoning_result
        )
        
        # Step 4: Extract knowledge paths as strings
        path_strings = [path.to_string() for path in retrieval_result.paths[:5]]
        
        # Get graph statistics
        graph_stats = self.knowledge_graph.get_statistics()
        
        processing_time = time.time() - start_time
        
        return AMGRAGResponse(
            query=query,
            answer=answer,
            reasoning_trace=reasoning_trace,
            knowledge_paths=path_strings,
            confidence=retrieval_result.confidence,
            processing_time=processing_time,
            graph_stats=graph_stats
        )
    
    def _generate_answer_with_cot(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        reasoning_result: Dict
    ) -> tuple[str, str]:
        """
        Generate answer using chain-of-thought reasoning
        
        Args:
            query: Original query
            retrieval_result: Graph retrieval result
            reasoning_result: Multi-hop reasoning result
        
        Returns:
            (answer, reasoning_trace)
        """
        
        # Build context from knowledge paths
        context = retrieval_result.to_context(max_paths=5)
        
        # Build prompt with chain-of-thought
        prompt = self._build_cot_prompt(query, context, reasoning_result)
        
        # Generate response
        # TODO: Use actual model inference
        answer = self._mock_generate(prompt, query, retrieval_result)
        
        # Extract reasoning trace
        reasoning_trace = self._extract_reasoning_trace(prompt, answer)
        
        return answer, reasoning_trace
    
    def _build_cot_prompt(
        self,
        query: str,
        context: str,
        reasoning_result: Dict
    ) -> str:
        """Build chain-of-thought prompt"""
        
        prompt = f"""You are a medical expert assistant. Answer the question using the knowledge graph context.

Question: {query}

{context}

Multi-hop reasoning:
- Total entities explored: {reasoning_result.get('total_entities', 0)}
- Reasoning steps: {len(reasoning_result.get('reasoning_steps', []))}

Please provide step-by-step reasoning:

Step 1: Identify key medical concepts from the question.
Step 2: Trace relationships in the knowledge graph.
Step 3: Connect relevant paths to form a complete understanding.
Step 4: Synthesize the final answer with confidence.

Reasoning:
"""
        
        return prompt
    
    def _mock_generate(
        self,
        prompt: str,
        query: str,
        retrieval_result: RetrievalResult
    ) -> str:
        """Mock answer generation (placeholder for actual model)"""
        
        # Simple rule-based response based on paths
        if not retrieval_result.paths:
            return "I don't have enough information to answer this question."
        
        # Get most confident path
        top_path = retrieval_result.paths[0]
        
        # Generate simple answer from path
        answer_parts = []
        answer_parts.append(f"Based on the medical knowledge graph:")
        answer_parts.append(f"\n{top_path.to_string()}")
        answer_parts.append(f"\nConfidence: {top_path.confidence:.2%}")
        
        return "\n".join(answer_parts)
    
    def _extract_reasoning_trace(self, prompt: str, answer: str) -> str:
        """Extract reasoning trace from response"""
        # For now, just return the context
        return prompt
    
    def _generate_fallback_response(self, query: str, processing_time: float) -> AMGRAGResponse:
        """Generate fallback response when no paths found"""
        return AMGRAGResponse(
            query=query,
            answer="I don't have sufficient knowledge to answer this question confidently.",
            reasoning_trace="No relevant knowledge paths found in the medical knowledge graph.",
            knowledge_paths=[],
            confidence=0.0,
            processing_time=processing_time,
            graph_stats=self.knowledge_graph.get_statistics()
        )
    
    def add_knowledge(
        self,
        entities: List[MedicalEntity],
        relationships: List[MedicalRelationship]
    ):
        """Add new knowledge to graph"""
        for entity in entities:
            self.knowledge_graph.add_entity(entity)
        
        for rel in relationships:
            self.knowledge_graph.add_relationship(rel)
        
        logger.info(f"Added {len(entities)} entities and {len(relationships)} relationships")
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        graph_stats = self.knowledge_graph.get_statistics()
        
        return {
            "knowledge_graph": graph_stats,
            "model": self.model_name,
            "retrieval_agent": {
                "max_hops": self.retrieval_agent.max_hops,
                "min_confidence": self.retrieval_agent.min_confidence,
                "top_k_paths": self.retrieval_agent.top_k_paths
            }
        }


if __name__ == "__main__":
    # Demo
    print("AMG-RAG Orchestrator Demo")
    print("="*60)
    
    # Initialize orchestrator
    orchestrator = AMGRAGOrchestrator(
        use_memory_graph=True,
        load_pretrained_graph=True
    )
    
    # Test queries
    queries = [
        "What causes numbness in diabetic patients?",
        "How does diabetes affect vision?",
        "What are the treatments for diabetes?",
        "What is the relationship between diabetes and heart disease?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Process query
        response = orchestrator.process(query)
        
        print(f"\nAnswer:")
        print(response.answer)
        
        print(f"\nKnowledge Paths:")
        for i, path in enumerate(response.knowledge_paths):
            print(f"{i+1}. {path}")
        
        print(f"\nConfidence: {response.confidence:.2%}")
        print(f"Processing Time: {response.processing_time:.3f}s")
    
    # System statistics
    print(f"\n{'='*60}")
    print("System Statistics:")
    print('='*60)
    stats = orchestrator.get_statistics()
    print(f"Knowledge Graph:")
    print(f"  Nodes: {stats['knowledge_graph']['num_nodes']}")
    print(f"  Edges: {stats['knowledge_graph']['num_edges']}")
    print(f"\nRetrieval Agent:")
    print(f"  Max hops: {stats['retrieval_agent']['max_hops']}")
    print(f"  Min confidence: {stats['retrieval_agent']['min_confidence']}")
