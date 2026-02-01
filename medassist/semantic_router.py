"""
Semantic Router for Intelligent Agent Selection
Inspired by vLLM Semantic Router - dynamically routes queries to optimal agents
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging


@dataclass
class AgentCapability:
    """Represents an agent's capabilities"""
    agent_name: str
    specialization: str
    complexity_range: Tuple[float, float]  # (min, max) complexity
    avg_latency: float  # seconds
    avg_confidence: float  # 0-1
    success_rate: float  # 0-1
    keywords: List[str]  # Signal keywords
    embedding: Optional[np.ndarray] = None  # Agent capability embedding


class SemanticRouter:
    """
    Intelligent routing system that selects optimal agent based on:
    - Signal detection (keyword, semantic similarity)
    - Confidence-based escalation
    - Dynamic workflow adaptation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define agent capabilities
        self.agents = {
            'history': AgentCapability(
                agent_name='history',
                specialization='patient_history',
                complexity_range=(0.0, 1.0),
                avg_latency=2.0,
                avg_confidence=0.92,
                success_rate=0.95,
                keywords=['symptoms', 'history', 'onset', 'duration', 'previous', 
                         'medications', 'allergies', 'family history']
            ),
            'diagnostic': AgentCapability(
                agent_name='diagnostic',
                specialization='diagnosis',
                complexity_range=(0.3, 1.0),
                avg_latency=3.5,
                avg_confidence=0.88,
                success_rate=0.91,
                keywords=['diagnosis', 'differential', 'disease', 'condition',
                         'lab results', 'imaging', 'test', 'findings']
            ),
            'treatment': AgentCapability(
                agent_name='treatment',
                specialization='treatment_planning',
                complexity_range=(0.3, 1.0),
                avg_latency=3.0,
                avg_confidence=0.90,
                success_rate=0.93,
                keywords=['treatment', 'therapy', 'medication', 'drug',
                         'prescription', 'dose', 'management', 'protocol']
            ),
            'knowledge': AgentCapability(
                agent_name='knowledge',
                specialization='medical_knowledge',
                complexity_range=(0.5, 1.0),
                avg_latency=2.5,
                avg_confidence=0.95,
                success_rate=0.97,
                keywords=['evidence', 'guidelines', 'research', 'literature',
                         'mechanism', 'pathophysiology', 'rare', 'uncommon']
            )
        }
        
        # Routing statistics
        self.routing_history: List[Dict] = []
        
    def extract_signals(self, query: str, case_data: Dict) -> Dict[str, float]:
        """
        Extract routing signals from query and case data
        Returns signal scores for each agent type
        """
        signals = {}
        query_lower = query.lower()
        
        # Keyword signal detection
        for agent_name, capability in self.agents.items():
            keyword_matches = sum(
                1 for kw in capability.keywords 
                if kw in query_lower
            )
            keyword_score = min(keyword_matches / 3.0, 1.0)
            signals[f'{agent_name}_keyword'] = keyword_score
        
        # Context signals from case data
        has_symptoms = len(case_data.get('symptoms', '').split(',')) > 2
        has_history = bool(case_data.get('history', ''))
        has_medications = bool(case_data.get('medications', ''))
        
        # Boost specific agents based on context
        if has_symptoms:
            signals['diagnostic_context'] = 0.8
        if has_history:
            signals['history_context'] = 0.9
        if has_medications:
            signals['treatment_context'] = 0.7
        
        return signals
    
    def calculate_agent_scores(self, 
                               signals: Dict[str, float],
                               complexity: float,
                               context: str = "") -> Dict[str, float]:
        """
        Calculate routing scores for each agent based on signals and complexity
        """
        scores = {}
        
        for agent_name, capability in self.agents.items():
            score = 0.0
            
            # 1. Keyword signal match
            keyword_signal = signals.get(f'{agent_name}_keyword', 0.0)
            score += keyword_signal * 0.4
            
            # 2. Context signal match
            context_signal = signals.get(f'{agent_name}_context', 0.0)
            score += context_signal * 0.3
            
            # 3. Complexity fit
            min_complexity, max_complexity = capability.complexity_range
            if min_complexity <= complexity <= max_complexity:
                complexity_fit = 1.0
            elif complexity < min_complexity:
                complexity_fit = complexity / min_complexity
            else:
                complexity_fit = max_complexity / complexity
            score += complexity_fit * 0.2
            
            # 4. Historical performance
            performance = (capability.success_rate + capability.avg_confidence) / 2.0
            score += performance * 0.1
            
            scores[agent_name] = min(score, 1.0)
        
        return scores
    
    def route_query(self, 
                   query: str,
                   case_data: Dict,
                   complexity: float) -> Dict[str, any]:
        """
        Main routing function - selects optimal agent(s) for query
        
        Returns:
            {
                'primary_agent': str,
                'workflow': List[str],
                'confidence': float,
                'reasoning': str,
                'signals': Dict
            }
        """
        # Extract signals
        signals = self.extract_signals(query, case_data)
        
        # Calculate scores
        agent_scores = self.calculate_agent_scores(signals, complexity, query)
        
        # Select primary agent (highest score)
        primary_agent = max(agent_scores.items(), key=lambda x: x[1])
        
        # Build workflow based on complexity and scores
        workflow = self._build_workflow(agent_scores, complexity)
        
        # Calculate routing confidence
        confidence = self._calculate_routing_confidence(
            agent_scores, primary_agent[1]
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            primary_agent[0], agent_scores, signals, complexity
        )
        
        # Record routing decision
        routing_decision = {
            'primary_agent': primary_agent[0],
            'primary_score': primary_agent[1],
            'workflow': workflow,
            'confidence': confidence,
            'reasoning': reasoning,
            'signals': signals,
            'all_scores': agent_scores,
            'complexity': complexity
        }
        
        self.routing_history.append(routing_decision)
        
        self.logger.info(
            f"Routed to {primary_agent[0]} (score={primary_agent[1]:.2f}, "
            f"confidence={confidence:.2f}, workflow={workflow})"
        )
        
        return routing_decision
    
    def _build_workflow(self, 
                       agent_scores: Dict[str, float],
                       complexity: float) -> List[str]:
        """
        Build execution workflow based on scores and complexity
        """
        # Sort agents by score
        sorted_agents = sorted(
            agent_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if complexity < 0.3:
            # Simple case: top 2 agents
            return [agent for agent, score in sorted_agents[:2] if score > 0.3]
        
        elif complexity < 0.7:
            # Standard case: top 3 agents
            return [agent for agent, score in sorted_agents[:3] if score > 0.2]
        
        else:
            # Complex case: all relevant agents
            workflow = [agent for agent, score in sorted_agents if score > 0.1]
            
            # Always include knowledge agent for complex cases
            if 'knowledge' not in workflow:
                workflow.append('knowledge')
            
            return workflow
    
    def _calculate_routing_confidence(self,
                                     agent_scores: Dict[str, float],
                                     primary_score: float) -> float:
        """
        Calculate confidence in routing decision
        Higher confidence when:
        - Primary agent has high score
        - Clear winner (large gap to second place)
        """
        scores = sorted(agent_scores.values(), reverse=True)
        
        # High primary score
        confidence = primary_score * 0.6
        
        # Clear winner (margin to second place)
        if len(scores) > 1:
            margin = scores[0] - scores[1]
            confidence += margin * 0.4
        else:
            confidence += 0.4
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(self,
                           primary_agent: str,
                           agent_scores: Dict[str, float],
                           signals: Dict[str, float],
                           complexity: float) -> str:
        """Generate human-readable routing reasoning"""
        
        # Find top signals
        top_signals = sorted(
            signals.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        signal_str = ", ".join(f"{k}={v:.2f}" for k, v in top_signals)
        
        reasoning = (
            f"Routed to {primary_agent} agent based on: "
            f"complexity={complexity:.2f}, top_signals=[{signal_str}]"
        )
        
        return reasoning
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics"""
        if not self.routing_history:
            return {}
        
        agent_counts = {}
        for decision in self.routing_history:
            agent = decision['primary_agent']
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        avg_confidence = np.mean([
            d['confidence'] for d in self.routing_history
        ])
        
        return {
            'total_routings': len(self.routing_history),
            'agent_distribution': agent_counts,
            'avg_routing_confidence': avg_confidence,
            'last_10_decisions': self.routing_history[-10:]
        }


class ConfidenceEscalator:
    """
    Implements confidence-based escalation (inspired by DeepConf)
    Routes to simpler/faster agents first, escalates if confidence is low
    """
    
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Agent hierarchy (fast → slow, simple → complex)
        self.agent_hierarchy = [
            'history',      # Fastest, simplest
            'treatment',    # Fast, moderate complexity
            'diagnostic',   # Moderate speed, complex
            'knowledge'     # Slower, most complex/accurate
        ]
    
    def should_escalate(self, 
                       agent_result: Dict,
                       current_level: int) -> Tuple[bool, str]:
        """
        Determine if we should escalate to next agent level
        
        Args:
            agent_result: Result from current agent
            current_level: Current position in hierarchy (0-based)
        
        Returns:
            (should_escalate: bool, reason: str)
        """
        confidence = agent_result.get('confidence', 0.0)
        
        # Don't escalate if at highest level
        if current_level >= len(self.agent_hierarchy) - 1:
            return False, "At highest agent level"
        
        # Escalate if confidence below threshold
        if confidence < self.confidence_threshold:
            reason = (
                f"Confidence {confidence:.2f} below threshold "
                f"{self.confidence_threshold:.2f}"
            )
            return True, reason
        
        # Check for uncertainty signals in reasoning
        uncertainty_keywords = [
            'uncertain', 'unclear', 'ambiguous', 'possibly',
            'might', 'could be', 'not sure'
        ]
        
        reasoning = agent_result.get('reasoning_trace', '').lower()
        has_uncertainty = any(kw in reasoning for kw in uncertainty_keywords)
        
        if has_uncertainty and current_level < len(self.agent_hierarchy) - 1:
            return True, "Detected uncertainty in reasoning"
        
        return False, "Confidence acceptable"
    
    def get_escalation_plan(self, 
                           initial_agent: str,
                           max_agents: int = 3) -> List[str]:
        """
        Generate escalation plan starting from initial agent
        
        Returns ordered list of agents to try
        """
        if initial_agent not in self.agent_hierarchy:
            return [initial_agent]
        
        start_idx = self.agent_hierarchy.index(initial_agent)
        
        # Return initial agent + potential escalation targets
        return self.agent_hierarchy[start_idx:start_idx + max_agents]
    
    def execute_with_escalation(self,
                               query: str,
                               agents: Dict,
                               escalation_plan: List[str]) -> Dict:
        """
        Execute query with automatic escalation
        
        Args:
            query: Query to process
            agents: Dictionary of available agents
            escalation_plan: Ordered list of agents to try
        
        Returns:
            Final result with escalation history
        """
        escalation_history = []
        final_result = None
        
        for level, agent_name in enumerate(escalation_plan):
            self.logger.info(
                f"Level {level}: Executing {agent_name} agent"
            )
            
            agent = agents.get(agent_name)
            if not agent:
                continue
            
            # Execute agent (pseudo-code, actual implementation varies)
            result = self._execute_agent(agent, query)
            
            escalation_history.append({
                'level': level,
                'agent': agent_name,
                'confidence': result.get('confidence', 0.0),
                'result': result
            })
            
            # Check if escalation needed
            should_escalate, reason = self.should_escalate(result, level)
            
            if not should_escalate:
                final_result = result
                self.logger.info(
                    f"Stopping at {agent_name}: {reason}"
                )
                break
            else:
                self.logger.info(
                    f"Escalating from {agent_name}: {reason}"
                )
        
        # Use last result if all escalations triggered
        if final_result is None:
            final_result = escalation_history[-1]['result']
        
        return {
            **final_result,
            'escalation_history': escalation_history,
            'escalation_levels': len(escalation_history)
        }
    
    def _execute_agent(self, agent, query: str) -> Dict:
        """Placeholder for actual agent execution"""
        # This would call the actual agent's process method
        return {
            'confidence': 0.8,
            'result': {},
            'reasoning_trace': ""
        }


# Example usage
if __name__ == "__main__":
    # Initialize router
    router = SemanticRouter()
    
    # Example case
    case = {
        'symptoms': 'fever, cough, chest pain',
        'history': 'hypertension, diabetes',
        'medications': 'metformin, lisinopril'
    }
    
    # Route query
    decision = router.route_query(
        query="Diagnose patient with respiratory symptoms",
        case_data=case,
        complexity=0.65
    )
    
    print(f"Primary Agent: {decision['primary_agent']}")
    print(f"Workflow: {decision['workflow']}")
    print(f"Confidence: {decision['confidence']:.2%}")
    print(f"Reasoning: {decision['reasoning']}")
    
    # Test confidence escalation
    escalator = ConfidenceEscalator(confidence_threshold=0.80)
    
    # Simulate low-confidence result
    result = {
        'confidence': 0.65,
        'reasoning_trace': 'Diagnosis is uncertain, possibly pneumonia'
    }
    
    should_escalate, reason = escalator.should_escalate(result, current_level=0)
    print(f"\nEscalate: {should_escalate} - {reason}")
    
    # Get escalation plan
    plan = escalator.get_escalation_plan('history', max_agents=3)
    print(f"Escalation plan: {plan}")
