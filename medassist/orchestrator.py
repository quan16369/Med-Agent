"""
Advanced Orchestrator Agent - Coordinates sophisticated medical workflow
Features: Dynamic workflow adaptation, inter-agent communication, confidence aggregation
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import json
import asyncio
from collections import defaultdict

from .base_agent import BaseAgent, AgentMessage
from .specialized_agents import (
    MedicalHistoryAgent,
    DiagnosticAgent,
    TreatmentAgent,
    KnowledgeAgent
)
from .reasoning_engine import UncertaintyQuantifier, EvidenceTracker
from .confidence_aggregator import ConfidenceAggregator
from .semantic_router import SemanticRouter, ConfidenceEscalator
from .deep_confidence import TokenConfidenceTracker, ParallelThinkingFilter
from .monitoring import PerformanceMonitor
from .adaptive_models import AdaptiveModelSelector, detect_query_complexity, QueryComplexity
from .offline_rag import OfflineRAG
from config import ORCHESTRATOR_CONFIG, WORKFLOW_TEMPLATES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedAssistOrchestrator:
    """
    Advanced Orchestrator with dynamic workflow adaptation
    Coordinates multiple specialized medical agents with inter-agent communication
    """
    
    def __init__(
        self,
        model_name: str = "google/medgemma-2b",
        device: str = "auto",
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        rural_mode: bool = False,
        offline_mode: bool = False
    ):
        """
        Initialize the sophisticated orchestrator
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run model on
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization (for rural/low-resource)
            rural_mode: Enable optimizations for rural/resource-constrained settings
            offline_mode: Enable offline-first operation (no internet required)
        """
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.rural_mode = rural_mode
        self.offline_mode = offline_mode
        
        # Apply rural optimizations if enabled
        if self.rural_mode:
            logger.info("Rural mode enabled: Optimizing for resource-constrained settings")
            self.load_in_4bit = True  # Force 4-bit quantization
            self.offline_mode = True  # Force offline mode
            if device == "auto":
                self.device = "cpu"  # Prefer CPU for rural deployment
            logger.info(f"  - 4-bit quantization enabled (50% less RAM)")
            logger.info(f"  - Offline mode enabled")
            logger.info(f"  - Device: {self.device}")
        
        logger.info("Initializing Advanced MedAssist Orchestrator...")
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model(
            model_name, self.device, self.load_in_8bit, self.load_in_4bit
        )
        
        # Initialize specialized agents
        self.agents = {
            'history': MedicalHistoryAgent(self.model, self.tokenizer),
            'diagnostic': DiagnosticAgent(self.model, self.tokenizer),
            'treatment': TreatmentAgent(self.model, self.tokenizer),
            'knowledge': KnowledgeAgent(self.model, self.tokenizer)
        }
        
        # Advanced orchestration features
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.global_evidence_tracker = EvidenceTracker()
        self.agent_communication_log = []
        self.confidence_aggregator = ConfidenceAggregator()
        
        # Semantic routing and deep confidence
        self.semantic_router = SemanticRouter()
        self.confidence_escalator = ConfidenceEscalator(confidence_threshold=0.75)
        self.parallel_filter = ParallelThinkingFilter(num_samples=8)
        self.performance_monitor = PerformanceMonitor(window_size=100)
        
        # Adaptive multi-model selection (uses full HAI-DEF collection intelligently)
        import psutil
        available_ram = psutil.virtual_memory().total / (1024**3) if hasattr(psutil.virtual_memory(), 'total') else 4.0
        self.model_selector = AdaptiveModelSelector(
            available_ram_gb=available_ram,
            max_inference_time_sec=10.0,
            rural_mode=self.rural_mode
        )
        logger.info(f"Adaptive model selector initialized (RAM: {available_ram:.1f}GB, Rural: {self.rural_mode})")
        
        # Offline RAG system for knowledge augmentation
        if not self.rural_mode or self.offline_mode:
            try:
                self.rag = OfflineRAG()
                logger.info("Offline RAG system initialized")
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}. Continuing without RAG.")
                self.rag = None
        else:
            self.rag = None
        
        # Orchestrator state
        self.case_history = []
        self.current_case = None
        self.dynamic_workflow_enabled = True
        
        logger.info("Advanced Orchestrator initialized with Semantic Router + DeepConfidence")
    
    def _load_model(self, model_name: str, device: str, load_in_8bit: bool):
        """Load the language model and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Model loading options
            model_kwargs = {
                "device_map": device,
                "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
            }
            
            if load_in_8bit and device != "cpu":
                model_kwargs["load_in_8bit"] = True
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            logger.info("Model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Falling back to mock model for demonstration")
            return None, None
    
    def process_case(
        self,
        case: Dict[str, Any],
        workflow: str = "standard_diagnosis"
    ) -> Dict[str, Any]:
        """
        Process a patient case through sophisticated agentic workflow
        Features: Dynamic adaptation, inter-agent communication, confidence aggregation
        
        Args:
            case: Dictionary with patient information
                Required: symptoms, age, gender
                Optional: history, medications, allergies, vital_signs
            workflow: Workflow template (can be overridden dynamically)
            
        Returns:
            Dictionary with comprehensive assessment, confidence metrics, and reasoning traces
        """
        logger.info(f"Processing case with adaptive workflow (initial: {workflow})")
        
        self.current_case = case
        start_time = datetime.now()
        
        # Assess case complexity and adapt workflow
        if self.dynamic_workflow_enabled:
            complexity_score = self._assess_case_complexity(case)
            workflow = self._adapt_workflow(workflow, complexity_score, case)
            logger.info(f"Complexity: {complexity_score:.2f}, Selected workflow: {workflow}")
        
        # Get workflow steps
        workflow_steps = WORKFLOW_TEMPLATES.get(
            workflow,
            WORKFLOW_TEMPLATES['standard_diagnosis']
        )
        
        # Result container with advanced metrics
        results = {
            "case_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "workflow": workflow,
            "complexity_score": self._assess_case_complexity(case),
            "steps": [],
            "findings": {},
            "recommendations": {},
            "reasoning": [],
            "inter_agent_communications": [],
            "confidence_metrics": {},
            "evidence_trail": []
        }
        
        # Execute workflow with inter-agent communication
        for step_idx, step in enumerate(workflow_steps):
            logger.info(f"Executing step {step_idx+1}/{len(workflow_steps)}: {step}")
            
            agent = self.agents.get(step)
            if agent is None:
                logger.warning(f"Agent {step} not found, skipping")
                continue
            
            try:
                # Prepare input with inter-agent context
                agent_input = self._prepare_sophisticated_input(
                    step, case, results, step_idx
                )
                
                # Check if agent needs to query other agents
                if self._should_consult_other_agent(step, results):
                    consultation = self._facilitate_agent_consultation(
                        current_agent=step,
                        results=results
                    )
                    agent_input['consultation_results'] = consultation
                    results['inter_agent_communications'].append(consultation)
                
                # Execute agent with timing
                step_start = datetime.now()
                agent_output = agent.process(agent_input)
                step_duration = (datetime.now() - step_start).total_seconds()
                
                # Store comprehensive results
                step_result = {
                    "agent": step,
                    "timestamp": datetime.now().isoformat(),
                    "output": agent_output,
                    "duration": step_duration,
                    "confidence_score": agent_output.get('confidence_score', 0.5),
                    "uncertainty_level": agent_output.get('uncertainty_level', 'Unknown')
                }
                
                results['steps'].append(step_result)
                results['findings'][step] = agent_output.get('findings', {})
                
                # Aggregate evidence
                if 'evidence_summary' in agent_output:
                    results['evidence_trail'].append({
                        'agent': step,
                        'evidence': agent_output['evidence_summary']
                    })
                
                # Add reasoning trace
                if 'reasoning' in agent_output:
                    results['reasoning'].append({
                        "agent": step,
                        "reasoning": agent_output['reasoning'],
                        "confidence": agent_output.get('confidence_score', 0.5)
                    })
                
                # Dynamic workflow adaptation based on findings
                if self.dynamic_workflow_enabled:
                    additional_steps = self._determine_additional_steps(
                        step, agent_output, workflow_steps
                    )
                    if additional_steps:
                        logger.info(f"Adding dynamic steps: {additional_steps}")
                        workflow_steps.extend(additional_steps)
                
                logger.info(f"Step {step} completed in {step_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in step {step}: {e}")
                results['steps'].append({
                    "agent": step,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Synthesize final assessment with confidence aggregation
        results['final_assessment'] = self._synthesize_sophisticated_assessment(results)
        
        # Calculate aggregate confidence metrics
        results['confidence_metrics'] = self._aggregate_confidence_metrics(results)
        
        # Generate meta-reasoning summary
        results['meta_reasoning'] = self._generate_meta_reasoning(results)
        
        # Calculate processing time
        end_time = datetime.now()
        results['processing_time'] = (end_time - start_time).total_seconds()
        
        # Store in history
        self.case_history.append(results)
        
        logger.info(f"Case processing completed in {results['processing_time']:.2f}s")
        logger.info(f"Overall confidence: {results['confidence_metrics'].get('aggregate_confidence', 0):.2f}")
        
        return results
    
    def _prepare_agent_input(
        self,
        agent_type: str,
        case: Dict,
        current_results: Dict
    ) -> Dict:
        """
        Prepare input for a specific agent based on case and previous results
        
        Args:
            agent_type: Type of agent
            case: Original case data
            current_results: Results from previous agents
            
        Returns:
            Input dictionary for the agent
        """
        base_input = {
            "case": case,
            "previous_findings": current_results.get('findings', {})
        }
        
        # Add agent-specific context
        if agent_type == 'history':
            base_input['focus'] = ['medical_history', 'risk_factors', 'medications']
        elif agent_type == 'diagnostic':
            base_input['focus'] = ['differential_diagnosis', 'recommended_tests']
        elif agent_type == 'treatment':
            base_input['focus'] = ['treatment_plan', 'patient_education']
        elif agent_type == 'knowledge':
            # Knowledge agent queries based on findings
            if 'diagnostic' in current_results.get('findings', {}):
                diagnosis = current_results['findings']['diagnostic']
                base_input['query'] = diagnosis.get('top_diagnosis', '')
        
        return base_input
    
    def _synthesize_assessment(self, results: Dict) -> Dict:
        """
        Synthesize findings from all agents into final assessment
        
        Args:
            results: Results dictionary with all agent outputs
            
        Returns:
            Final assessment dictionary
        """
        assessment = {
            "summary": "",
            "key_findings": [],
            "diagnosis": {},
            "treatment_plan": {},
            "follow_up": []
        }
        
        findings = results.get('findings', {})
        
        # Extract history findings
        if 'history' in findings:
            history = findings['history']
            assessment['key_findings'].extend(
                history.get('risk_factors', [])
            )
        
        # Extract diagnostic findings
        if 'diagnostic' in findings:
            diagnostic = findings['diagnostic']
            assessment['diagnosis'] = {
                "differential": diagnostic.get('differential_diagnosis', []),
                "most_likely": diagnostic.get('top_diagnosis', ''),
                "recommended_tests": diagnostic.get('recommended_tests', [])
            }
        
        # Extract treatment recommendations
        if 'treatment' in findings:
            treatment = findings['treatment']
            assessment['treatment_plan'] = {
                "medications": treatment.get('medications', []),
                "non_pharmacologic": treatment.get('non_pharmacologic', []),
                "patient_education": treatment.get('patient_education', [])
            }
            assessment['follow_up'] = treatment.get('follow_up', [])
        
        # Generate summary
        assessment['summary'] = self._generate_summary(assessment)
        
        return assessment
    
    def _assess_case_complexity(self, case: Dict) -> float:
        """Assess complexity of case to adapt workflow"""
        complexity = 0.0
        
        # Multiple symptoms increase complexity
        symptoms = str(case.get('symptoms', ''))
        complexity += min(symptoms.count(',') * 0.1, 0.3)
        
        # Chronic conditions increase complexity
        history = str(case.get('history', '')).lower()
        chronic_conditions = ['diabetes', 'hypertension', 'copd', 'heart failure', 'renal']
        complexity += sum(0.15 for cond in chronic_conditions if cond in history)
        
        # Multiple medications increase complexity
        medications = str(case.get('medications', ''))
        complexity += min(medications.count(',') * 0.1, 0.3)
        
        # Age extremes increase complexity
        age = case.get('age', 45)
        if age < 18 or age > 75:
            complexity += 0.2
        
        # Vital sign abnormalities
        vitals = case.get('vital_signs', '')
        if vitals:
            complexity += 0.15
        
        return min(complexity, 1.0)
    
    def _adapt_workflow(self, initial_workflow: str, complexity: float, case: Dict) -> str:
        """Dynamically adapt workflow based on case complexity"""
        if complexity > 0.7:
            return 'complex_case'  # Use full workflow with knowledge agent
        elif complexity < 0.3 and 'urgent' not in str(case.get('symptoms', '')).lower():
            return 'triage'  # Simplified workflow
        else:
            return initial_workflow
    
    def _prepare_sophisticated_input(
        self,
        agent_type: str,
        case: Dict,
        current_results: Dict,
        step_idx: int
    ) -> Dict:
        """Prepare sophisticated input with full context for agent"""
        base_input = {
            \"case\": case,
            \"previous_findings\": current_results.get('findings', {}),
            \"step_index\": step_idx,
            \"context\": {
                \"complexity\": current_results.get('complexity_score', 0.5),
                \"previous_confidence\": self._get_previous_confidence(current_results)
            }
        }
        
        # Add agent-specific sophisticated context
        if agent_type == 'history':
            base_input['focus'] = ['comprehensive_risk_assessment', 'medication_interactions']
        elif agent_type == 'diagnostic':
            base_input['focus'] = ['differential_with_confidence', 'red_flags', 'rare_conditions']
            base_input['use_multi_strategy'] = True
        elif agent_type == 'treatment':
            base_input['focus'] = ['evidence_based_plan', 'contraindications', 'patient_specific']
            base_input['safety_priority'] = 'high'
        elif agent_type == 'knowledge':
            if 'diagnostic' in current_results.get('findings', {}):
                diagnosis = current_results['findings']['diagnostic'].get('top_diagnosis', '')
                base_input['query'] = f\"Latest evidence and guidelines for {diagnosis}\"
                base_input['evidence_level_required'] = 'A or B'
        
        return base_input
    
    def _should_consult_other_agent(self, current_agent: str, results: Dict) -> bool:
        """Determine if agent should consult another agent"""
        # Diagnostic agent should consult knowledge agent for rare conditions
        if current_agent == 'diagnostic':
            symptoms = self.current_case.get('symptoms', '').lower()
            rare_keywords = ['rare', 'unusual', 'atypical', 'complex']
            return any(kw in symptoms for kw in rare_keywords)
        
        # Treatment agent should consult diagnostic if low confidence
        if current_agent == 'treatment':
            if 'diagnostic' in results.get('findings', {}):
                diag_confidence = results['findings']['diagnostic'].get('confidence_score', 1.0)
                return diag_confidence < 0.6
        
        return False
    
    def _facilitate_agent_consultation(
        self,
        current_agent: str,
        results: Dict
    ) -> Dict:
        """Facilitate inter-agent consultation"""
        consultation_log = {
            'requesting_agent': current_agent,
            'timestamp': datetime.now().isoformat(),
            'consultations': []
        }
        
        if current_agent == 'diagnostic' and 'knowledge' in self.agents:
            # Diagnostic consults knowledge agent
            knowledge_agent = self.agents['knowledge']
            top_dx = results['findings'].get('diagnostic', {}).get('top_diagnosis', '')
            
            if top_dx:
                knowledge_input = {
                    'query': f\"Evidence and differential for {top_dx}\",
                    'context': {'urgency': 'high'}
                }
                knowledge_output = knowledge_agent.process(knowledge_input)
                consultation_log['consultations'].append({
                    'consulted_agent': 'knowledge',
                    'query': knowledge_input['query'],
                    'response': knowledge_output
                })
        
        return consultation_log
    
    def _determine_additional_steps(
        self,
        completed_step: str,
        agent_output: Dict,
        current_workflow: List[str]
    ) -> List[str]:
        """Determine if additional agent steps needed based on findings"""
        additional = []
        
        # If diagnostic uncertainty is high, add knowledge agent
        if completed_step == 'diagnostic':
            uncertainty = agent_output.get('uncertainty_level', '')
            if uncertainty == 'High' and 'knowledge' not in current_workflow:
                additional.append('knowledge')
        
        # If contraindications found, might need knowledge consultation
        if completed_step == 'treatment':
            contraindications = agent_output.get('findings', {}).get('contraindications_checked', [])
            if contraindications and 'knowledge' not in current_workflow:
                additional.append('knowledge')
        
        return additional
    
    def _synthesize_sophisticated_assessment(self, results: Dict) -> Dict:
        """Synthesize findings with confidence and evidence"""
        assessment = {
            \"summary\": \"\",
            \"key_findings\": [],
            \"diagnosis\": {},
            \"treatment_plan\": {},
            \"follow_up\": [],
            \"confidence_analysis\": {},
            \"safety_alerts\": []
        }
        
        findings = results.get('findings', {})
        
        # Extract with confidence
        if 'history' in findings:
            history = findings['history']
            assessment['key_findings'].extend(history.get('risk_factors', []))
        
        if 'diagnostic' in findings:
            diagnostic = findings['diagnostic']
            assessment['diagnosis'] = {
                \"differential\": diagnostic.get('differential_diagnosis', []),
                \"most_likely\": diagnostic.get('top_diagnosis', ''),
                \"confidence\": diagnostic.get('top_diagnosis_confidence', 0.5),
                \"differential_with_confidence\": diagnostic.get('differential_with_confidence', []),
                \"recommended_tests\": diagnostic.get('recommended_tests', []),
                \"urgency\": diagnostic.get('urgency', 'Routine')
            }
        
        if 'treatment' in findings:
            treatment = findings['treatment']
            assessment['treatment_plan'] = {
                \"medications\": treatment.get('medications', []),
                \"non_pharmacologic\": treatment.get('non_pharmacologic', []),
                \"patient_education\": treatment.get('patient_education', []),
                \"contraindications\": treatment.get('contraindications_checked', [])
            }
            assessment['follow_up'] = treatment.get('follow_up', [])
            
            # Extract safety alerts
            if treatment.get('contraindications_checked'):
                assessment['safety_alerts'] = treatment['contraindications_checked']
        
        # Generate sophisticated summary
        assessment['summary'] = self._generate_sophisticated_summary(assessment, results)
        
        return assessment
    
    def _aggregate_confidence_metrics(self, results: Dict) -> Dict:
        """Aggregate confidence across all agents"""
        confidences = []
        uncertainties = []
        
        for step in results.get('steps', []):
            if 'confidence_score' in step:
                confidences.append(step['confidence_score'])
            if 'uncertainty_level' in step:
                uncertainties.append(step['uncertainty_level'])
        
        if confidences:
            aggregate = sum(confidences) / len(confidences)
            min_conf = min(confidences)
            max_conf = max(confidences)
        else:
            aggregate = 0.5
            min_conf = 0.5
            max_conf = 0.5
        
        # Determine overall uncertainty
        high_uncertainty_count = uncertainties.count('High')
        if high_uncertainty_count > len(uncertainties) / 2:
            overall_uncertainty = 'High'
        elif high_uncertainty_count > 0:
            overall_uncertainty = 'Moderate'
        else:
            overall_uncertainty = 'Low'
        
        return {
            'aggregate_confidence': round(aggregate, 3),
            'min_confidence': round(min_conf, 3),
            'max_confidence': round(max_conf, 3),
            'overall_uncertainty': overall_uncertainty,
            'confidence_variance': round(sum((c - aggregate)**2 for c in confidences) / len(confidences), 3) if confidences else 0
        }
    
    def _generate_meta_reasoning(self, results: Dict) -> Dict:
        """Generate meta-level reasoning about the diagnostic process"""
        return {
            'workflow_adapted': results['workflow'] != 'standard_diagnosis',
            'inter_agent_consultations': len(results.get('inter_agent_communications', [])),
            'reasoning_strategies_used': self._count_reasoning_strategies(results),
            'total_steps': len(results.get('steps', [])),
            'average_step_duration': sum(s.get('duration', 0) for s in results.get('steps', [])) / max(len(results.get('steps', [])), 1),
            'evidence_sources': len(results.get('evidence_trail', [])),
            'quality_assessment': self._assess_overall_quality(results)
        }
    
    def _count_reasoning_strategies(self, results: Dict) -> Dict:
        """Count different reasoning strategies used"""
        strategies = defaultdict(int)
        for step in results.get('steps', []):
            reasoning = step.get('output', {}).get('reasoning', '')
            if 'ReAct' in reasoning:
                strategies['ReAct'] += 1
            if 'Chain-of-Thought' in reasoning or 'chain-of-thought' in reasoning.lower():
                strategies['Chain-of-Thought'] += 1
            if 'Reflective' in reasoning:
                strategies['Reflective'] += 1
            if 'Socratic' in reasoning:
                strategies['Socratic'] += 1
        return dict(strategies)
    
    def _assess_overall_quality(self, results: Dict) -> str:
        """Assess overall quality of assessment"""
        conf = results['confidence_metrics'].get('aggregate_confidence', 0.5)
        if conf > 0.8:
            return 'High'
        elif conf > 0.6:
            return 'Good'
        elif conf > 0.4:
            return 'Moderate'
        else:
            return 'Low - Consider specialist consultation'
    
    def _get_previous_confidence(self, results: Dict) -> float:
        """Get confidence from previous steps"""
        steps = results.get('steps', [])
        if not steps:
            return 0.5
        confidences = [s.get('confidence_score', 0.5) for s in steps]
        return sum(confidences) / len(confidences)
    
    def _generate_sophisticated_summary(self, assessment: Dict, results: Dict) -> str:
        """Generate sophisticated human-readable summary"""
        summary_parts = []
        
        # Diagnosis with confidence
        if assessment.get('diagnosis', {}).get('most_likely'):
            dx = assessment['diagnosis']['most_likely']
            conf = assessment['diagnosis'].get('confidence', 0.5)
            summary_parts.append(f\"Primary diagnosis: {dx} (confidence: {conf:.0%})\")
