"""
LangGraph-based Medical Multi-Agent System
State-based orchestration with specialized agents
"""

import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from operator import add
import time

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

# Import available LLM providers
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from ..models.knowledge_graph import MedicalKnowledgeGraph, MedicalEntity, MedicalRelationship
from ..tools.graph_retrieval import GraphConditionalRetrieval
from ..tools.pubmed_retrieval import PubMedRetriever

logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State shared across all agents in the graph"""
    # Input
    query: str
    image_base64: Optional[str]
    
    # Agent workflow
    messages: Annotated[List[Dict], add]  # Append-only message history
    current_step: str
    
    # Knowledge accumulation
    entities: List[Dict]
    knowledge_paths: List[Dict]
    pubmed_evidence: List[Dict]
    
    # Diagnostic findings
    symptoms: List[str]
    conditions: List[Dict]
    differential_diagnosis: List[Dict]
    
    # Treatment recommendations
    treatments: List[Dict]
    contraindications: List[str]
    
    # Validation
    confidence: float
    validated: bool
    final_answer: str
    
    # Metadata
    agents_called: Annotated[List[str], add]
    error: Optional[str]


# ============================================================================
# AGENT NODES
# ============================================================================

class MedicalAgentNodes:
    """Collection of agent nodes for LangGraph"""
    
    def __init__(
        self,
        knowledge_graph: MedicalKnowledgeGraph,
        retrieval_agent: GraphConditionalRetrieval,
        pubmed: Optional[PubMedRetriever] = None,
        llm_provider: str = "auto",  # Auto-detect best available
        model_name: Optional[str] = None
    ):
        self.kg = knowledge_graph
        self.retrieval = retrieval_agent
        self.pubmed = pubmed
        
        # Initialize LLM (auto-detect best available provider)
        self.llm = self._init_llm(llm_provider, model_name)
    
    def _init_llm(self, provider: str, model_name: Optional[str]):
        """Initialize LLM with best available provider"""
        
        # Auto-detect: Groq > OpenAI > Ollama
        if provider == "auto":
            if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
                provider = "groq"
                model_name = model_name or "llama-3.3-70b-versatile"
            elif OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                provider = "openai"
                model_name = model_name or "gpt-4o-mini"
            elif OLLAMA_AVAILABLE:
                provider = "ollama"
                model_name = model_name or "llama3.2"
            else:
                raise RuntimeError("No LLM provider available. Install: pip install langchain-groq or langchain-openai")
        
        # Initialize based on provider
        if provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError("Groq not available. Install: pip install langchain-groq")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            logger.info(f"Using Groq: {model_name}")
            return ChatGroq(model=model_name, temperature=0, api_key=api_key)
        
        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install: pip install langchain-openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            logger.info(f"Using OpenAI: {model_name}")
            return ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
        
        elif provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not available. Install: pip install langchain-ollama")
            logger.info(f"Using Ollama: {model_name}")
            return ChatOllama(model=model_name, temperature=0)
        
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'auto', 'groq', 'openai', or 'ollama'")
    
    # ------------------------------------------------------------------------
    # ORCHESTRATOR NODE
    # ------------------------------------------------------------------------
    
    def orchestrator_node(self, state: AgentState) -> AgentState:
        """Route to appropriate agents based on query type"""
        
        query = state["query"]
        
        prompt = f"""Analyze this medical query and determine which agents to call:
        
Query: {query}

Available agents:
- knowledge: Query medical knowledge graph for entities and relationships
- diagnostic: Analyze symptoms and create differential diagnosis
- treatment: Recommend treatments and medications
- evidence: Search PubMed for research evidence
- validator: Validate findings and assess confidence

Return a routing decision in this format:
{{
    "agents_to_call": ["agent1", "agent2", ...],
    "reasoning": "Why these agents are needed"
}}
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse response (simplified - should use structured output)
        import json
        try:
            decision = json.loads(response.content)
            next_step = decision.get("agents_to_call", ["knowledge"])[0]
        except:
            next_step = "knowledge"  # Default fallback
        
        state["current_step"] = next_step
        state["messages"].append({
            "agent": "orchestrator",
            "content": response.content,
            "timestamp": time.time()
        })
        state["agents_called"].append("orchestrator")
        
        return state
    
    # ------------------------------------------------------------------------
    # KNOWLEDGE AGENT NODE
    # ------------------------------------------------------------------------
    
    def knowledge_agent_node(self, state: AgentState) -> AgentState:
        """Extract entities and query knowledge graph"""
        
        query = state["query"]
        
        # Extract entities using NER
        from ..tools.medical_ner import HybridMedicalNER
        ner = HybridMedicalNER()
        entities = ner.extract_entities(query)
        
        # Store entities
        state["entities"] = [
            {
                "text": e.text,
                "type": e.entity_type,
                "score": e.confidence,
                "relevance": getattr(e, "relevance_score", 0)
            }
            for e in entities
        ]
        
        # Query knowledge graph
        paths = []
        if len(entities) > 0:
            # Get top entities
            top_entities = sorted(entities, key=lambda x: x.confidence, reverse=True)[:3]
            
            for entity in top_entities:
                try:
                    # Find related entities
                    related = self.kg.find_related_entities(
                        entity.text,
                        max_depth=2,
                        relationship_types=None
                    )
                    
                    if related:
                        paths.append({
                            "source": entity.text,
                            "related": [
                                {
                                    "entity": r.get("entity"),
                                    "relationship": r.get("relationship"),
                                    "properties": r.get("properties", {})
                                }
                                for r in related[:5]  # Top 5 related
                            ]
                        })
                except Exception as e:
                    logger.warning(f"Error querying KG for {entity.text}: {e}")
        
        state["knowledge_paths"] = paths
        state["messages"].append({
            "agent": "knowledge",
            "content": f"Found {len(entities)} entities, {len(paths)} knowledge paths",
            "data": {"entities": state["entities"], "paths": paths},
            "timestamp": time.time()
        })
        state["agents_called"].append("knowledge")
        
        # Route to next agent
        if any("symptom" in e["type"].lower() for e in state["entities"]):
            state["current_step"] = "diagnostic"
        elif state.get("differential_diagnosis"):
            state["current_step"] = "treatment"
        else:
            state["current_step"] = "evidence"
        
        return state
    
    # ------------------------------------------------------------------------
    # DIAGNOSTIC AGENT NODE
    # ------------------------------------------------------------------------
    
    def diagnostic_agent_node(self, state: AgentState) -> AgentState:
        """Analyze symptoms and create differential diagnosis"""
        
        query = state["query"]
        entities = state.get("entities", [])
        knowledge_paths = state.get("knowledge_paths", [])
        
        # Extract symptoms
        symptoms = [e["text"] for e in entities if "symptom" in e["type"].lower()]
        state["symptoms"] = symptoms
        
        # Use LLM for differential diagnosis
        context = f"""
Medical Query: {query}

Identified Symptoms: {', '.join(symptoms)}

Knowledge Graph Context:
{json.dumps(knowledge_paths, indent=2)}
"""
        
        prompt = f"""Based on the symptoms and context, create a differential diagnosis.

{context}

Provide a ranked list of possible conditions with:
1. Condition name
2. Likelihood (high/medium/low)
3. Key supporting factors
4. Key contradicting factors

Return as JSON array."""
        
        response = self.llm.invoke([
            SystemMessage(content="You are a medical diagnostic expert."),
            HumanMessage(content=prompt)
        ])
        
        # Parse differential diagnosis
        try:
            import json
            import re
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                differential = json.loads(json_match.group())
            else:
                differential = []
        except:
            differential = []
        
        state["differential_diagnosis"] = differential
        state["messages"].append({
            "agent": "diagnostic",
            "content": response.content,
            "data": {"symptoms": symptoms, "differential": differential},
            "timestamp": time.time()
        })
        state["agents_called"].append("diagnostic")
        
        # Route to treatment
        state["current_step"] = "treatment"
        
        return state
    
    # ------------------------------------------------------------------------
    # TREATMENT AGENT NODE
    # ------------------------------------------------------------------------
    
    def treatment_agent_node(self, state: AgentState) -> AgentState:
        """Recommend treatments based on diagnosis"""
        
        query = state["query"]
        differential = state.get("differential_diagnosis", [])
        knowledge_paths = state.get("knowledge_paths", [])
        
        # Find treatment information from knowledge graph
        treatments = []
        for path in knowledge_paths:
            for related in path.get("related", []):
                if related.get("relationship") in ["treats", "treated_by", "medication_for"]:
                    treatments.append({
                        "treatment": related.get("entity"),
                        "for_condition": path.get("source"),
                        "properties": related.get("properties", {})
                    })
        
        # Use LLM for treatment recommendations
        context = f"""
Medical Query: {query}

Differential Diagnosis:
{json.dumps(differential, indent=2)}

Known Treatments from Knowledge Graph:
{json.dumps(treatments, indent=2)}
"""
        
        prompt = f"""Based on the differential diagnosis, recommend appropriate treatments.

{context}

For each likely condition, provide:
1. Treatment options (ranked by standard of care)
2. Dosage and administration
3. Contraindications
4. Monitoring requirements

Return as JSON array."""
        
        response = self.llm.invoke([
            SystemMessage(content="You are a medical treatment specialist."),
            HumanMessage(content=prompt)
        ])
        
        # Parse treatment recommendations
        try:
            import json
            import re
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                treatment_recs = json.loads(json_match.group())
            else:
                treatment_recs = []
        except:
            treatment_recs = []
        
        state["treatments"] = treatment_recs
        state["messages"].append({
            "agent": "treatment",
            "content": response.content,
            "data": {"treatments": treatment_recs},
            "timestamp": time.time()
        })
        state["agents_called"].append("treatment")
        
        # Route to evidence
        state["current_step"] = "evidence"
        
        return state
    
    # ------------------------------------------------------------------------
    # EVIDENCE AGENT NODE
    # ------------------------------------------------------------------------
    
    def evidence_agent_node(self, state: AgentState) -> AgentState:
        """Search for research evidence"""
        
        if not self.pubmed:
            # Skip if PubMed not available
            state["pubmed_evidence"] = []
            state["current_step"] = "validator"
            return state
        
        query = state["query"]
        entities = state.get("entities", [])
        
        # Create search queries from entities
        search_queries = []
        if entities:
            top_entities = sorted(entities, key=lambda x: x["score"], reverse=True)[:3]
            for e in top_entities:
                search_queries.append(e["text"])
        else:
            search_queries.append(query)
        
        # Search PubMed
        evidence = []
        for search_query in search_queries[:2]:  # Limit to 2 searches
            try:
                results = self.pubmed.search(search_query, max_results=3)
                for result in results:
                    evidence.append({
                        "query": search_query,
                        "title": result.get("title"),
                        "abstract": result.get("abstract"),
                        "pmid": result.get("pmid"),
                        "year": result.get("publication_date", {}).get("year")
                    })
            except Exception as e:
                logger.warning(f"PubMed search failed for '{search_query}': {e}")
        
        state["pubmed_evidence"] = evidence
        state["messages"].append({
            "agent": "evidence",
            "content": f"Found {len(evidence)} research articles",
            "data": {"evidence": evidence},
            "timestamp": time.time()
        })
        state["agents_called"].append("evidence")
        
        # Route to validator
        state["current_step"] = "validator"
        
        return state
    
    # ------------------------------------------------------------------------
    # VALIDATOR AGENT NODE
    # ------------------------------------------------------------------------
    
    def validator_agent_node(self, state: AgentState) -> AgentState:
        """Validate findings and generate final answer"""
        
        query = state["query"]
        
        # Collect all findings
        findings = {
            "entities": state.get("entities", []),
            "knowledge_paths": state.get("knowledge_paths", []),
            "symptoms": state.get("symptoms", []),
            "differential_diagnosis": state.get("differential_diagnosis", []),
            "treatments": state.get("treatments", []),
            "evidence": state.get("pubmed_evidence", [])
        }
        
        # Use LLM to synthesize final answer
        prompt = f"""Based on all gathered information, provide a comprehensive answer to the medical query.

Query: {query}

Gathered Information:
{json.dumps(findings, indent=2)}

Provide:
1. Direct answer to the query
2. Supporting evidence from knowledge graph and research
3. Important caveats or limitations
4. Confidence level (0-100%)

Format as a clear, professional medical response."""
        
        response = self.llm.invoke([
            SystemMessage(content="You are a medical expert synthesizing information from multiple sources."),
            HumanMessage(content=prompt)
        ])
        
        # Extract confidence (simplified)
        confidence = 0.7  # Default
        if "high confidence" in response.content.lower():
            confidence = 0.9
        elif "medium confidence" in response.content.lower():
            confidence = 0.7
        elif "low confidence" in response.content.lower():
            confidence = 0.5
        
        state["final_answer"] = response.content
        state["confidence"] = confidence
        state["validated"] = True
        state["messages"].append({
            "agent": "validator",
            "content": response.content,
            "data": {"confidence": confidence},
            "timestamp": time.time()
        })
        state["agents_called"].append("validator")
        
        # End workflow
        state["current_step"] = "end"
        
        return state


# ============================================================================
# LANGGRAPH ORCHESTRATOR
# ============================================================================

class LangGraphMedicalOrchestrator:
    """
    LangGraph-based Medical Multi-Agent System
    
    Features:
    - State-based workflow with StateGraph
    - Specialized agent nodes (Knowledge, Diagnostic, Treatment, Evidence, Validator)
    - Dynamic routing based on query type
    - Tool integration (KG, PubMed, NER)
    - Comprehensive state tracking
    """
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        use_memory_graph: bool = True,
        load_sample_graph: bool = True,
        pubmed_email: Optional[str] = None,
        llm_provider: str = "auto",  # Auto-detect: Groq > OpenAI > Ollama
        model_name: Optional[str] = None
    ):
        logger.info("Initializing LangGraph Medical Orchestrator")
        
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
        
        # Initialize agent nodes
        self.agent_nodes = MedicalAgentNodes(
            knowledge_graph=self.knowledge_graph,
            retrieval_agent=self.retrieval_agent,
            pubmed=self.pubmed,
            llm_provider=llm_provider,
            model_name=model_name
        )
        
        # Build LangGraph
        self.graph = self._build_graph()
        
        logger.info("LangGraph orchestrator initialized")
    
    def _load_sample_knowledge(self):
        """Load sample medical knowledge"""
        from ..models.knowledge_graph import MedicalEntity, MedicalRelationship
        
        # Sample entities
        entities = [
            MedicalEntity(name="Type 2 Diabetes", entity_type="Disease", properties={"icd10": "E11"}),
            MedicalEntity(name="Metformin", entity_type="Medication", properties={"drug_class": "Biguanide"}),
            MedicalEntity(name="Hypertension", entity_type="Disease", properties={"icd10": "I10"}),
            MedicalEntity(name="ACE Inhibitors", entity_type="Medication", properties={"drug_class": "Antihypertensive"}),
            MedicalEntity(name="Chest Pain", entity_type="Symptom", properties={}),
            MedicalEntity(name="Myocardial Infarction", entity_type="Disease", properties={"icd10": "I21"}),
        ]
        
        # Sample relationships
        relationships = [
            MedicalRelationship(source="Metformin", target="Type 2 Diabetes", relationship_type="treats", properties={}),
            MedicalRelationship(source="ACE Inhibitors", target="Hypertension", relationship_type="treats", properties={}),
            MedicalRelationship(source="Chest Pain", target="Myocardial Infarction", relationship_type="symptom_of", properties={}),
        ]
        
        # Add to graph
        for entity in entities:
            self.knowledge_graph.add_entity(entity)
        for rel in relationships:
            self.knowledge_graph.add_relationship(rel)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("orchestrator", self.agent_nodes.orchestrator_node)
        workflow.add_node("knowledge", self.agent_nodes.knowledge_agent_node)
        workflow.add_node("diagnostic", self.agent_nodes.diagnostic_agent_node)
        workflow.add_node("treatment", self.agent_nodes.treatment_agent_node)
        workflow.add_node("evidence", self.agent_nodes.evidence_agent_node)
        workflow.add_node("validator", self.agent_nodes.validator_agent_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Add conditional edges (routing logic)
        def route_from_orchestrator(state: AgentState) -> str:
            return state["current_step"]
        
        def route_from_knowledge(state: AgentState) -> str:
            return state["current_step"]
        
        def route_from_diagnostic(state: AgentState) -> str:
            return state["current_step"]
        
        def route_from_treatment(state: AgentState) -> str:
            return state["current_step"]
        
        def route_from_evidence(state: AgentState) -> str:
            return state["current_step"]
        
        def route_from_validator(state: AgentState) -> str:
            return END if state.get("validated") else "validator"
        
        # Add edges
        workflow.add_conditional_edges(
            "orchestrator",
            route_from_orchestrator,
            {
                "knowledge": "knowledge",
                "diagnostic": "diagnostic",
                "treatment": "treatment",
                "evidence": "evidence",
                "validator": "validator"
            }
        )
        
        workflow.add_conditional_edges(
            "knowledge",
            route_from_knowledge,
            {
                "diagnostic": "diagnostic",
                "treatment": "treatment",
                "evidence": "evidence",
                "validator": "validator"
            }
        )
        
        workflow.add_conditional_edges(
            "diagnostic",
            route_from_diagnostic,
            {
                "treatment": "treatment",
                "evidence": "evidence",
                "validator": "validator"
            }
        )
        
        workflow.add_conditional_edges(
            "treatment",
            route_from_treatment,
            {
                "evidence": "evidence",
                "validator": "validator"
            }
        )
        
        workflow.add_conditional_edges(
            "evidence",
            route_from_evidence,
            {
                "validator": "validator"
            }
        )
        
        workflow.add_conditional_edges(
            "validator",
            route_from_validator
        )
        
        # Compile graph
        return workflow.compile()
    
    def query(self, question: str, image_base64: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a medical query through the LangGraph workflow
        
        Args:
            question: Medical question
            image_base64: Optional base64-encoded image
            
        Returns:
            Dict with answer, workflow trace, confidence, etc.
        """
        start_time = time.time()
        
        # Initialize state
        initial_state: AgentState = {
            "query": question,
            "image_base64": image_base64,
            "messages": [],
            "current_step": "orchestrator",
            "entities": [],
            "knowledge_paths": [],
            "pubmed_evidence": [],
            "symptoms": [],
            "conditions": [],
            "differential_diagnosis": [],
            "treatments": [],
            "contraindications": [],
            "confidence": 0.0,
            "validated": False,
            "final_answer": "",
            "agents_called": [],
            "error": None
        }
        
        try:
            # Run graph
            final_state = self.graph.invoke(initial_state)
            
            processing_time = time.time() - start_time
            
            return {
                "query": question,
                "answer": final_state.get("final_answer", "Unable to generate answer"),
                "confidence": final_state.get("confidence", 0.0),
                "workflow_trace": self._format_trace(final_state),
                "agents_used": final_state.get("agents_called", []),
                "findings": {
                    "entities": final_state.get("entities", []),
                    "knowledge_paths": final_state.get("knowledge_paths", []),
                    "symptoms": final_state.get("symptoms", []),
                    "differential_diagnosis": final_state.get("differential_diagnosis", []),
                    "treatments": final_state.get("treatments", []),
                    "evidence": final_state.get("pubmed_evidence", [])
                },
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in LangGraph workflow: {e}", exc_info=True)
            return {
                "query": question,
                "answer": f"Error processing query: {str(e)}",
                "confidence": 0.0,
                "workflow_trace": "Error occurred",
                "agents_used": [],
                "findings": {},
                "processing_time": time.time() - start_time
            }
    
    def _format_trace(self, state: AgentState) -> str:
        """Format workflow trace for display"""
        
        trace_lines = ["=== Workflow Trace ===\n"]
        
        for msg in state.get("messages", []):
            agent = msg.get("agent", "unknown")
            content = msg.get("content", "")
            trace_lines.append(f"\n[{agent.upper()}]")
            trace_lines.append(content[:200] + "..." if len(content) > 200 else content)
        
        return "\n".join(trace_lines)
