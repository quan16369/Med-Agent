"""
Advanced Medical QA Orchestrator
Multi-agent system with knowledge graph and tool usage
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass
import time

from .knowledge_graph import MedicalKnowledgeGraph
from .graph_retrieval import GraphConditionalRetrieval
from .medical_ner import BioBERTNER
from .pubmed_retrieval import PubMedRetriever
from .agentic_workflow import (
    OrchestratorAgent, KnowledgeAgent, DiagnosticAgent,
    TreatmentAgent, EvidenceAgent, ValidatorAgent,
    WorkflowState
)

logger = logging.getLogger(__name__)


@dataclass
class AgenticResponse:
    """Response from agentic workflow system"""
    query: str
    answer: str
    workflow_trace: str
    agents_used: list
    findings: Dict
    confidence: float
    processing_time: float


class AgenticMedicalOrchestrator:
    """
    Multi-Agent Medical QA System
    
    Features:
    - Specialized agents (Knowledge, Diagnostic, Treatment, Evidence, Validator)
    - Tool usage (knowledge graph, PubMed search, medical NER)
    - Agent reflection and self-correction
    - Dynamic workflow planning
    - Confidence scoring and consistency validation
    """
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        use_memory_graph: bool = True,
        load_sample_graph: bool = True,
        pubmed_email: Optional[str] = None
    ):
        logger.info("Initializing Agentic Medical Orchestrator")
        
        # Initialize knowledge graph
        self.knowledge_graph = MedicalKnowledgeGraph(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            use_memory=use_memory_graph
        )
        
        if load_sample_graph:
            self._load_sample_knowledge()
        
        # Initialize retrieval system
        self.retrieval_agent = GraphConditionalRetrieval(
            knowledge_graph=self.knowledge_graph,
            max_hops=3,
            min_confidence=0.5,
            top_k_paths=20
        )
        
        # Initialize PubMed retriever
        self.pubmed = None
        if pubmed_email:
            self.pubmed = PubMedRetriever(email=pubmed_email)
        
        # Initialize specialized agents
        self.agents = {
            "KnowledgeAgent": KnowledgeAgent(self.knowledge_graph, self.retrieval_agent),
            "DiagnosticAgent": DiagnosticAgent(),
            "TreatmentAgent": TreatmentAgent(),
            "EvidenceAgent": EvidenceAgent(self.pubmed),
            "ValidatorAgent": ValidatorAgent()
        }
        
        # Initialize orchestrator agent
        self.orchestrator = OrchestratorAgent(self.agents)
        
        logger.info("Agentic orchestrator initialized with {} agents".format(len(self.agents)))
    
    def _load_sample_knowledge(self):
        """Load sample medical knowledge"""
        from .knowledge_graph import MedicalEntity, MedicalRelationship
        
        # Add comprehensive diabetes knowledge
        entities = [
            # Diseases
            MedicalEntity("disease_diabetes", "Diabetes Mellitus", "disease", 
                         ["diabetes", "DM", "type 2 diabetes"]),
            MedicalEntity("disease_hypertension", "Hypertension", "disease",
                         ["high blood pressure", "HTN"]),
            MedicalEntity("disease_neuropathy", "Diabetic Neuropathy", "disease",
                         ["nerve damage", "peripheral neuropathy"]),
            MedicalEntity("disease_retinopathy", "Diabetic Retinopathy", "disease",
                         ["eye damage", "diabetic eye disease"]),
            MedicalEntity("disease_chd", "Coronary Heart Disease", "disease",
                         ["CHD", "heart disease"]),
            MedicalEntity("disease_kidney", "Diabetic Nephropathy", "disease",
                         ["kidney disease", "renal disease"]),
            
            # Symptoms
            MedicalEntity("symptom_hyperglycemia", "Hyperglycemia", "symptom",
                         ["high blood sugar", "elevated glucose"]),
            MedicalEntity("symptom_numbness", "Numbness", "symptom",
                         ["tingling", "loss of sensation"]),
            MedicalEntity("symptom_blurred_vision", "Blurred Vision", "symptom",
                         ["vision problems"]),
            MedicalEntity("symptom_chest_pain", "Chest Pain", "symptom",
                         ["angina"]),
            MedicalEntity("symptom_fatigue", "Fatigue", "symptom",
                         ["tiredness", "weakness"]),
            MedicalEntity("symptom_thirst", "Excessive Thirst", "symptom",
                         ["polydipsia"]),
            MedicalEntity("symptom_urination", "Frequent Urination", "symptom",
                         ["polyuria"]),
            
            # Treatments
            MedicalEntity("treatment_metformin", "Metformin", "treatment",
                         ["glucophage"]),
            MedicalEntity("treatment_insulin", "Insulin", "treatment",
                         ["insulin therapy"]),
            MedicalEntity("treatment_statin", "Statin", "treatment",
                         ["atorvastatin"]),
            MedicalEntity("treatment_ace_inhibitor", "ACE Inhibitor", "treatment",
                         ["lisinopril"]),
            
            # Biomarkers
            MedicalEntity("biomarker_hba1c", "HbA1c", "biomarker",
                         ["hemoglobin a1c"]),
            MedicalEntity("biomarker_glucose", "Blood Glucose", "biomarker",
                         ["blood sugar"])
        ]
        
        relationships = [
            # Diabetes causes symptoms
            MedicalRelationship("disease_diabetes", "symptom_hyperglycemia", "causes", 0.95, 150, "pubmed"),
            MedicalRelationship("disease_diabetes", "symptom_thirst", "causes", 0.90, 100, "pubmed"),
            MedicalRelationship("disease_diabetes", "symptom_urination", "causes", 0.89, 95, "pubmed"),
            MedicalRelationship("disease_diabetes", "symptom_fatigue", "causes", 0.82, 80, "pubmed"),
            
            # Diabetes causes complications
            MedicalRelationship("disease_diabetes", "disease_neuropathy", "causes", 0.88, 75, "pubmed"),
            MedicalRelationship("disease_diabetes", "disease_retinopathy", "causes", 0.85, 80, "pubmed"),
            MedicalRelationship("disease_diabetes", "disease_kidney", "causes", 0.83, 70, "pubmed"),
            MedicalRelationship("disease_diabetes", "disease_chd", "increases_risk", 0.82, 120, "pubmed"),
            
            # Complications cause symptoms
            MedicalRelationship("disease_neuropathy", "symptom_numbness", "causes", 0.92, 60, "pubmed"),
            MedicalRelationship("disease_neuropathy", "symptom_fatigue", "causes", 0.75, 40, "pubmed"),
            MedicalRelationship("disease_retinopathy", "symptom_blurred_vision", "causes", 0.90, 55, "pubmed"),
            MedicalRelationship("disease_chd", "symptom_chest_pain", "causes", 0.93, 100, "pubmed"),
            
            # Treatments
            MedicalRelationship("treatment_metformin", "disease_diabetes", "treats", 0.97, 200, "pubmed"),
            MedicalRelationship("treatment_metformin", "symptom_hyperglycemia", "reduces", 0.94, 180, "pubmed"),
            MedicalRelationship("treatment_insulin", "disease_diabetes", "treats", 0.98, 250, "pubmed"),
            MedicalRelationship("treatment_statin", "disease_chd", "treats", 0.91, 150, "pubmed"),
            MedicalRelationship("treatment_ace_inhibitor", "disease_hypertension", "treats", 0.93, 160, "pubmed"),
            
            # Biomarkers
            MedicalRelationship("biomarker_hba1c", "disease_diabetes", "diagnoses", 0.96, 300, "pubmed"),
            MedicalRelationship("biomarker_glucose", "symptom_hyperglycemia", "indicates", 0.98, 250, "pubmed")
        ]
        
        for entity in entities:
            self.knowledge_graph.add_entity(entity)
        
        for rel in relationships:
            self.knowledge_graph.add_relationship(rel)
        
        logger.info(f"Loaded {len(entities)} entities and {len(relationships)} relationships")
    
    def process(self, query: str) -> AgenticResponse:
        """
        Process query through multi-agent workflow
        
        Args:
            query: Medical question
        
        Returns:
            Response with answer, agent actions, and confidence score
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {query}")
        
        # Execute agentic workflow
        workflow_state = self.orchestrator.execute_workflow(query)
        
        # Generate answer from workflow findings
        answer = self.orchestrator.generate_answer(workflow_state)
        
        # Generate workflow trace
        trace = self._generate_trace(workflow_state)
        
        processing_time = time.time() - start_time
        
        response = AgenticResponse(
            query=query,
            answer=answer,
            workflow_trace=trace,
            agents_used=workflow_state.agents_completed,
            findings=workflow_state.findings,
            confidence=workflow_state.confidence,
            processing_time=processing_time
        )
        
        logger.info(f"Query processed in {processing_time:.3f}s with confidence {workflow_state.confidence:.2%}")
        
        return response
    
    def _generate_trace(self, state: WorkflowState) -> str:
        """Generate human-readable workflow trace"""
        
        lines = [f"Agentic Workflow Trace for: {state.query}", "="*80]
        
        for i, action in enumerate(state.actions, 1):
            lines.append(f"\n{i}. {action.agent}")
            lines.append(f"   Action: {action.action_type}")
            if action.tool:
                lines.append(f"   Tool: {action.tool}")
            lines.append(f"   Status: {'SUCCESS' if action.success else 'FAILED'}")
            lines.append(f"   Reasoning: {action.reasoning}")
        
        lines.append(f"\n{'='*80}")
        lines.append(f"Agents Completed: {', '.join(state.agents_completed)}")
        lines.append(f"Overall Confidence: {state.confidence:.2%}")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        kg_stats = self.knowledge_graph.get_statistics()
        
        return {
            "knowledge_graph": kg_stats,
            "agents": list(self.agents.keys()),
            "agent_count": len(self.agents),
            "retrieval_config": {
                "max_hops": self.retrieval_agent.max_hops,
                "min_confidence": self.retrieval_agent.min_confidence,
                "top_k_paths": self.retrieval_agent.top_k_paths
            }
        }


if __name__ == "__main__":
    # Demo
    print("="*80)
    print("Agentic Medical Orchestrator Demo")
    print("Multi-Agent Workflow for Medical QA")
    print("="*80)
    
    # Initialize
    orchestrator = AgenticMedicalOrchestrator(
        use_memory_graph=True,
        load_sample_graph=True
    )
    
    # Test queries
    queries = [
        "What causes numbness in diabetic patients?",
        "How does diabetes affect vision?",
        "What are the treatments for diabetes?",
        "What biomarkers are used to diagnose diabetes?"
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)
        
        response = orchestrator.process(query)
        
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nAgents Used: {', '.join(response.agents_used)}")
        print(f"Confidence: {response.confidence:.2%}")
        print(f"Processing Time: {response.processing_time:.3f}s")
        
        print(f"\nWorkflow Trace:")
        print(response.workflow_trace)
    
    # Statistics
    print(f"\n{'='*80}")
    print("System Statistics")
    print('='*80)
    stats = orchestrator.get_statistics()
    print(f"Knowledge Graph: {stats['knowledge_graph']['num_nodes']} nodes, {stats['knowledge_graph']['num_edges']} edges")
    print(f"Agents: {', '.join(stats['agents'])}")
    print("="*80)
