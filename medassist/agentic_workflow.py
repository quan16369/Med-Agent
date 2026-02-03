"""
Multi-Agent Workflow System
Specialized agents collaborate to answer medical questions
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in the workflow"""
    ORCHESTRATOR = "orchestrator"
    KNOWLEDGE = "knowledge"
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    EVIDENCE = "evidence"
    VALIDATOR = "validator"


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    receiver: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class AgentAction:
    """Action taken by an agent"""
    agent: str
    action_type: str  # "query", "analyze", "search", "validate"
    tool: Optional[str]
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    reasoning: str
    
    def to_dict(self) -> Dict:
        return {
            "agent": self.agent,
            "action_type": self.action_type,
            "tool": self.tool,
            "input": self.input_data,
            "output": self.output_data,
            "success": self.success,
            "reasoning": self.reasoning
        }


@dataclass
class WorkflowState:
    """Current state of the workflow"""
    query: str
    current_agent: str
    agents_completed: List[str] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)
    findings: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "current_agent": self.current_agent,
            "agents_completed": self.agents_completed,
            "messages": [m.to_dict() for m in self.messages],
            "actions": [a.to_dict() for a in self.actions],
            "findings": self.findings,
            "confidence": self.confidence
        }


class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, name: str, role: AgentRole, tools: Optional[List[str]] = None):
        self.name = name
        self.role = role
        self.tools = tools or []
        self.memory: List[Dict] = []
        
    def process(self, message: AgentMessage, state: WorkflowState) -> AgentAction:
        """Process message and take action"""
        raise NotImplementedError
        
    def send_message(self, receiver: str, content: str, metadata: Dict = None) -> AgentMessage:
        """Send message to another agent"""
        import time
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            metadata=metadata or {},
            timestamp=time.time()
        )
    
    def use_tool(self, tool_name: str, input_data: Dict) -> Dict:
        """Use a tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not available to {self.name}")
        return {"error": "Tool not implemented"}
    
    def reflect(self, state: WorkflowState) -> str:
        """Reflect on current state and suggest next steps"""
        return f"{self.name} reflection: Continue workflow"


class KnowledgeAgent(BaseAgent):
    """Agent that queries medical knowledge graph"""
    
    def __init__(self, knowledge_graph, retrieval_agent):
        super().__init__("KnowledgeAgent", AgentRole.KNOWLEDGE, ["graph_query", "entity_extraction"])
        self.kg = knowledge_graph
        self.retrieval = retrieval_agent
        
    def process(self, message: AgentMessage, state: WorkflowState) -> AgentAction:
        """Query knowledge graph for relevant medical knowledge"""
        
        query = message.content
        logger.info(f"{self.name} processing: {query}")
        
        # Use graph retrieval tool
        result = self.use_tool("graph_query", {"query": query})
        
        reasoning = f"Searched knowledge graph for '{query}'. Found {len(result.get('paths', []))} knowledge paths."
        
        return AgentAction(
            agent=self.name,
            action_type="query",
            tool="graph_query",
            input_data={"query": query},
            output_data=result,
            success=len(result.get("paths", [])) > 0,
            reasoning=reasoning
        )
    
    def use_tool(self, tool_name: str, input_data: Dict) -> Dict:
        """Use knowledge graph tools"""
        if tool_name == "graph_query":
            result = self.retrieval.retrieve(input_data["query"])
            return {
                "paths": [p.to_string() for p in result.paths[:5]],
                "entities": result.query_entities,
                "confidence": result.confidence
            }
        elif tool_name == "entity_extraction":
            entities = self.retrieval.extract_entities(input_data["text"])
            return {"entities": entities}
        return super().use_tool(tool_name, input_data)
    
    def reflect(self, state: WorkflowState) -> str:
        """Reflect on knowledge findings"""
        if not state.findings.get("knowledge"):
            return "Need more knowledge graph exploration"
        
        confidence = state.findings["knowledge"].get("confidence", 0)
        if confidence < 0.5:
            return "Low confidence in knowledge paths. Suggest evidence validation."
        return "Knowledge findings sufficient. Proceed to diagnosis."


class DiagnosticAgent(BaseAgent):
    """Agent that performs diagnostic analysis"""
    
    def __init__(self):
        super().__init__("DiagnosticAgent", AgentRole.DIAGNOSTIC, ["symptom_analysis", "differential_diagnosis"])
        
    def process(self, message: AgentMessage, state: WorkflowState) -> AgentAction:
        """Analyze symptoms and perform diagnosis"""
        
        logger.info(f"{self.name} analyzing symptoms")
        
        knowledge = state.findings.get("knowledge", {})
        paths = knowledge.get("paths", [])
        
        # Analyze symptoms from knowledge paths
        diseases = []
        symptoms = []
        for path in paths:
            if "disease" in path.lower():
                diseases.extend([w for w in path.split() if w.istitle()])
            if "symptom" in path.lower() or "causes" in path.lower():
                symptoms.extend([w for w in path.split() if w.istitle()])
        
        diagnosis = {
            "possible_diseases": list(set(diseases[:3])),
            "key_symptoms": list(set(symptoms[:5])),
            "reasoning": f"Identified {len(set(diseases))} possible diseases from knowledge paths"
        }
        
        reasoning = f"Diagnostic analysis complete. Found {len(diagnosis['possible_diseases'])} possible conditions."
        
        return AgentAction(
            agent=self.name,
            action_type="analyze",
            tool="symptom_analysis",
            input_data={"knowledge_paths": paths},
            output_data=diagnosis,
            success=len(diagnosis["possible_diseases"]) > 0,
            reasoning=reasoning
        )
    
    def reflect(self, state: WorkflowState) -> str:
        """Reflect on diagnostic findings"""
        if not state.findings.get("diagnosis"):
            return "Need diagnostic analysis first"
        
        diseases = state.findings["diagnosis"].get("possible_diseases", [])
        if len(diseases) == 0:
            return "No clear diagnosis. Need more knowledge or evidence."
        elif len(diseases) > 5:
            return "Too many possibilities. Need differential diagnosis."
        return "Diagnosis clear. Proceed to treatment planning."


class TreatmentAgent(BaseAgent):
    """Agent that recommends treatments"""
    
    def __init__(self):
        super().__init__("TreatmentAgent", AgentRole.TREATMENT, ["treatment_search", "medication_check"])
        
    def process(self, message: AgentMessage, state: WorkflowState) -> AgentAction:
        """Recommend treatments based on diagnosis"""
        
        logger.info(f"{self.name} planning treatment")
        
        diagnosis = state.findings.get("diagnosis", {})
        diseases = diagnosis.get("possible_diseases", [])
        knowledge = state.findings.get("knowledge", {})
        paths = knowledge.get("paths", [])
        
        # Extract treatments from knowledge paths
        treatments = []
        for path in paths:
            if "treats" in path.lower() or "treatment" in path.lower():
                words = path.split()
                for i, word in enumerate(words):
                    if word.lower() in ["treats", "treatment"]:
                        if i > 0:
                            treatments.append(words[i-1])
        
        treatment_plan = {
            "recommended_treatments": list(set(treatments[:3])),
            "target_conditions": diseases[:2],
            "rationale": "Treatments extracted from validated knowledge paths"
        }
        
        reasoning = f"Treatment planning complete. {len(treatment_plan['recommended_treatments'])} treatments recommended."
        
        return AgentAction(
            agent=self.name,
            action_type="plan",
            tool="treatment_search",
            input_data={"diseases": diseases},
            output_data=treatment_plan,
            success=len(treatment_plan["recommended_treatments"]) > 0,
            reasoning=reasoning
        )


class EvidenceAgent(BaseAgent):
    """Agent that retrieves scientific evidence"""
    
    def __init__(self, pubmed_retriever=None):
        super().__init__("EvidenceAgent", AgentRole.EVIDENCE, ["pubmed_search", "evidence_ranking"])
        self.pubmed = pubmed_retriever
        
    def process(self, message: AgentMessage, state: WorkflowState) -> AgentAction:
        """Retrieve supporting evidence from literature"""
        
        logger.info(f"{self.name} searching for evidence")
        
        query = message.metadata.get("search_query", message.content)
        
        # Mock evidence retrieval (replace with actual PubMed search)
        evidence = {
            "articles_found": 0,
            "top_articles": [],
            "confidence": 0.0
        }
        
        if self.pubmed:
            try:
                articles = self.pubmed.search(query, max_results=3)
                evidence = {
                    "articles_found": len(articles),
                    "top_articles": [
                        {"title": a.title, "pmid": a.pmid, "year": a.year}
                        for a in articles[:3]
                    ],
                    "confidence": 0.8 if articles else 0.0
                }
            except Exception as e:
                logger.error(f"PubMed search failed: {e}")
        
        reasoning = f"Evidence search for '{query}'. Found {evidence['articles_found']} supporting articles."
        
        return AgentAction(
            agent=self.name,
            action_type="search",
            tool="pubmed_search",
            input_data={"query": query},
            output_data=evidence,
            success=evidence["articles_found"] > 0,
            reasoning=reasoning
        )


class ValidatorAgent(BaseAgent):
    """Agent that validates findings and checks consistency"""
    
    def __init__(self):
        super().__init__("ValidatorAgent", AgentRole.VALIDATOR, ["consistency_check", "confidence_scoring"])
        
    def process(self, message: AgentMessage, state: WorkflowState) -> AgentAction:
        """Validate all findings for consistency"""
        
        logger.info(f"{self.name} validating findings")
        
        # Check consistency across findings
        validation = {
            "knowledge_valid": "knowledge" in state.findings,
            "diagnosis_valid": "diagnosis" in state.findings,
            "treatment_valid": "treatment" in state.findings,
            "evidence_valid": "evidence" in state.findings,
            "consistency_score": 0.0
        }
        
        # Calculate consistency score
        valid_count = sum([validation["knowledge_valid"], validation["diagnosis_valid"], 
                          validation["treatment_valid"], validation["evidence_valid"]])
        validation["consistency_score"] = valid_count / 4.0
        
        # Check for contradictions
        issues = []
        if validation["diagnosis_valid"] and validation["treatment_valid"]:
            diagnosis = state.findings["diagnosis"]
            treatment = state.findings["treatment"]
            diseases = set(diagnosis.get("possible_diseases", []))
            targets = set(treatment.get("target_conditions", []))
            if diseases and targets and not diseases.intersection(targets):
                issues.append("Treatment targets don't match diagnosed conditions")
        
        validation["issues"] = issues
        validation["valid"] = len(issues) == 0 and validation["consistency_score"] >= 0.75
        
        reasoning = f"Validation complete. Consistency: {validation['consistency_score']:.0%}, Issues: {len(issues)}"
        
        return AgentAction(
            agent=self.name,
            action_type="validate",
            tool="consistency_check",
            input_data={"findings": list(state.findings.keys())},
            output_data=validation,
            success=validation["valid"],
            reasoning=reasoning
        )
    
    def reflect(self, state: WorkflowState) -> str:
        """Reflect on validation results"""
        if not state.findings.get("validation"):
            return "Need to run validation"
        
        validation = state.findings["validation"]
        if not validation.get("valid"):
            issues = validation.get("issues", [])
            return f"Validation failed: {', '.join(issues)}. Need revision."
        return "All validations passed. Ready to generate final answer."


class OrchestratorAgent(BaseAgent):
    """Meta-agent that orchestrates the workflow"""
    
    def __init__(self, agents: Dict[str, BaseAgent]):
        super().__init__("OrchestratorAgent", AgentRole.ORCHESTRATOR, ["workflow_planning", "agent_coordination"])
        self.agents = agents
        
    def plan_workflow(self, query: str) -> List[str]:
        """Plan the sequence of agents to execute"""
        
        # Standard medical QA workflow
        workflow = [
            "KnowledgeAgent",      # Get knowledge from graph
            "DiagnosticAgent",     # Analyze and diagnose
            "TreatmentAgent",      # Plan treatment
            "EvidenceAgent",       # Validate with evidence
            "ValidatorAgent"       # Final validation
        ]
        
        return workflow
    
    def execute_workflow(self, query: str) -> WorkflowState:
        """Execute the multi-agent workflow"""
        
        logger.info(f"Orchestrator starting workflow for: {query}")
        
        # Initialize workflow state
        state = WorkflowState(
            query=query,
            current_agent="OrchestratorAgent"
        )
        
        # Plan workflow
        workflow = self.plan_workflow(query)
        logger.info(f"Planned workflow: {' -> '.join(workflow)}")
        
        # Execute each agent in sequence
        for agent_name in workflow:
            if agent_name not in self.agents:
                logger.warning(f"Agent {agent_name} not found, skipping")
                continue
            
            agent = self.agents[agent_name]
            state.current_agent = agent_name
            
            # Create message for agent
            if len(state.messages) == 0:
                # First message from orchestrator
                message = self.send_message(agent_name, query)
            else:
                # Message from previous agent
                prev_action = state.actions[-1] if state.actions else None
                content = f"Process based on previous findings from {prev_action.agent}" if prev_action else query
                message = AgentMessage("OrchestratorAgent", agent_name, content, {}, 0.0)
            
            state.messages.append(message)
            
            # Execute agent
            try:
                action = agent.process(message, state)
                state.actions.append(action)
                
                # Store findings
                finding_key = agent.role.value
                state.findings[finding_key] = action.output_data
                
                logger.info(f"{agent_name} completed: {action.reasoning}")
                
                # Agent reflection
                reflection = agent.reflect(state)
                logger.info(f"{agent_name} reflection: {reflection}")
                
                state.agents_completed.append(agent_name)
                
            except Exception as e:
                logger.error(f"{agent_name} failed: {e}")
                state.actions.append(AgentAction(
                    agent=agent_name,
                    action_type="error",
                    tool=None,
                    input_data={"message": str(message.content)},
                    output_data={"error": str(e)},
                    success=False,
                    reasoning=f"Agent failed with error: {e}"
                ))
        
        # Calculate final confidence
        state.confidence = self._calculate_confidence(state)
        
        logger.info(f"Workflow complete. Confidence: {state.confidence:.2%}")
        
        return state
    
    def _calculate_confidence(self, state: WorkflowState) -> float:
        """Calculate overall confidence from agent actions"""
        
        successes = sum(1 for action in state.actions if action.success)
        total = len(state.actions)
        
        if total == 0:
            return 0.0
        
        base_confidence = successes / total
        
        # Bonus for validation passing
        if state.findings.get("validation", {}).get("valid"):
            base_confidence *= 1.2
        
        # Penalty for no evidence
        if not state.findings.get("evidence", {}).get("articles_found"):
            base_confidence *= 0.8
        
        return min(base_confidence, 1.0)
    
    def generate_answer(self, state: WorkflowState) -> str:
        """Generate final answer from workflow findings"""
        
        answer_parts = []
        
        # Add diagnosis
        if "diagnosis" in state.findings:
            diagnosis = state.findings["diagnosis"]
            diseases = diagnosis.get("possible_diseases", [])
            if diseases:
                answer_parts.append(f"Based on the analysis, possible conditions include: {', '.join(diseases)}.")
        
        # Add knowledge context
        if "knowledge" in state.findings:
            knowledge = state.findings["knowledge"]
            if knowledge.get("paths"):
                answer_parts.append(f"\nKnowledge graph analysis reveals {len(knowledge['paths'])} relevant medical relationships.")
        
        # Add treatment
        if "treatment" in state.findings:
            treatment = state.findings["treatment"]
            treatments = treatment.get("recommended_treatments", [])
            if treatments:
                answer_parts.append(f"\nRecommended treatments: {', '.join(treatments)}.")
        
        # Add evidence
        if "evidence" in state.findings:
            evidence = state.findings["evidence"]
            article_count = evidence.get("articles_found", 0)
            if article_count > 0:
                answer_parts.append(f"\nSupported by {article_count} scientific articles.")
        
        if not answer_parts:
            return "Insufficient information to provide a comprehensive answer."
        
        return " ".join(answer_parts)
