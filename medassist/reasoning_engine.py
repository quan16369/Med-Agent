"""
Advanced Reasoning Engine for MedAssist
Implements sophisticated reasoning patterns for medical AI agents
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """Types of reasoning strategies"""
    REACT = "react"  # Reasoning + Acting
    CHAIN_OF_THOUGHT = "cot"  # Step-by-step reasoning
    TREE_OF_THOUGHT = "tot"  # Multi-path exploration
    REFLECTIVE = "reflective"  # Self-critique and improvement
    SOCRATIC = "socratic"  # Question-driven reasoning


@dataclass
class ThoughtStep:
    """Single step in reasoning process"""
    step_number: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReasoningTrace:
    """Complete reasoning trace with metadata"""
    strategy: ReasoningStrategy
    steps: List[ThoughtStep]
    final_answer: str
    confidence_score: float
    evidence_sources: List[str]
    reasoning_time: float
    reflection: Optional[str] = None
    alternatives_considered: List[str] = field(default_factory=list)


class ReActReasoner:
    """
    ReAct (Reasoning + Acting) Pattern Implementation
    Interleaves reasoning and action for complex problem-solving
    """
    
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.trace: List[ThoughtStep] = []
    
    def reason(
        self,
        query: str,
        context: Dict,
        agent,
        available_tools: List
    ) -> ReasoningTrace:
        """
        Execute ReAct reasoning loop
        
        Args:
            query: Question or problem to solve
            context: Relevant context information
            agent: Agent instance for generating thoughts
            available_tools: List of tools agent can use
            
        Returns:
            ReasoningTrace with complete reasoning path
        """
        start_time = datetime.now()
        
        # Initial thought
        step_num = 1
        current_query = query
        final_answer = None
        
        while step_num <= self.max_iterations:
            # Generate thought
            thought_prompt = self._build_thought_prompt(
                current_query, context, self.trace, available_tools
            )
            
            thought_text = agent.generate_response(thought_prompt, max_new_tokens=256)
            
            # Parse thought for action
            action_info = self._parse_action(thought_text)
            
            thought = ThoughtStep(
                step_number=step_num,
                thought=thought_text,
                action=action_info.get('action'),
                confidence=self._estimate_confidence(thought_text)
            )
            
            # Execute action if present
            if action_info and action_info.get('action') != 'Final Answer':
                observation = self._execute_action(
                    action_info, available_tools, agent
                )
                thought.observation = observation
                current_query = f"Given observation: {observation}, continue reasoning..."
            else:
                # Reached final answer
                final_answer = action_info.get('action_input', thought_text)
                break
            
            self.trace.append(thought)
            step_num += 1
        
        # If no final answer yet, generate one
        if final_answer is None:
            final_answer = self._generate_final_answer(agent, query, context)
        
        end_time = datetime.now()
        reasoning_time = (end_time - start_time).total_seconds()
        
        # Build reasoning trace
        trace = ReasoningTrace(
            strategy=ReasoningStrategy.REACT,
            steps=self.trace,
            final_answer=final_answer,
            confidence_score=self._calculate_overall_confidence(),
            evidence_sources=self._extract_evidence(),
            reasoning_time=reasoning_time
        )
        
        return trace
    
    def _build_thought_prompt(
        self,
        query: str,
        context: Dict,
        previous_steps: List[ThoughtStep],
        tools: List
    ) -> str:
        """Build prompt for next thought"""
        prompt_parts = [
            "Use ReAct reasoning pattern (Thought → Action → Observation).",
            "",
            f"Question: {query}",
            ""
        ]
        
        if context:
            prompt_parts.append("Context:")
            for key, value in context.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")
        
        if tools:
            prompt_parts.append("Available Tools:")
            for tool in tools:
                prompt_parts.append(f"- {tool.name}: {tool.description}")
            prompt_parts.append("")
        
        if previous_steps:
            prompt_parts.append("Previous Reasoning Steps:")
            for step in previous_steps[-3:]:  # Last 3 steps
                prompt_parts.append(f"\nThought {step.step_number}: {step.thought}")
                if step.action:
                    prompt_parts.append(f"Action: {step.action}")
                if step.observation:
                    prompt_parts.append(f"Observation: {step.observation}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "What is your next thought?",
            "Format:",
            "Thought: [your reasoning]",
            "Action: [tool name] or 'Final Answer'",
            "Action Input: [tool parameters or final answer]"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_action(self, thought_text: str) -> Optional[Dict]:
        """Parse action from thought text"""
        lines = thought_text.split('\n')
        action = None
        action_input = None
        
        for line in lines:
            if line.strip().startswith('Action:'):
                action = line.split('Action:')[1].strip()
            elif line.strip().startswith('Action Input:'):
                action_input = line.split('Action Input:')[1].strip()
        
        if action:
            return {'action': action, 'action_input': action_input}
        return None
    
    def _execute_action(self, action_info: Dict, tools: List, agent) -> str:
        """Execute an action using available tools"""
        action_name = action_info['action']
        action_input = action_info['action_input']
        
        for tool in tools:
            if tool.name.lower() == action_name.lower():
                try:
                    result = tool.run(action_input)
                    return str(result)
                except Exception as e:
                    return f"Error executing {action_name}: {str(e)}"
        
        return f"Tool {action_name} not found"
    
    def _estimate_confidence(self, thought_text: str) -> float:
        """Estimate confidence in thought"""
        # Simple heuristic based on language
        confidence_keywords = {
            'certain': 0.9, 'clearly': 0.85, 'definitely': 0.9,
            'likely': 0.7, 'probably': 0.7, 'possibly': 0.5,
            'uncertain': 0.3, 'unclear': 0.3, 'maybe': 0.4
        }
        
        text_lower = thought_text.lower()
        confidences = [score for keyword, score in confidence_keywords.items() 
                      if keyword in text_lower]
        
        return sum(confidences) / len(confidences) if confidences else 0.6
    
    def _generate_final_answer(self, agent, query: str, context: Dict) -> str:
        """Generate final answer from reasoning"""
        prompt = f"Based on the reasoning above, provide a final answer to: {query}"
        return agent.generate_response(prompt, max_new_tokens=256)
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence from all steps"""
        if not self.trace:
            return 0.5
        confidences = [step.confidence for step in self.trace]
        return sum(confidences) / len(confidences)
    
    def _extract_evidence(self) -> List[str]:
        """Extract evidence from observations"""
        evidence = []
        for step in self.trace:
            if step.observation:
                evidence.append(step.observation)
        return evidence


class ChainOfThoughtReasoner:
    """
    Chain-of-Thought (CoT) Reasoning
    Step-by-step logical progression
    """
    
    def reason(
        self,
        problem: str,
        context: Dict,
        agent,
        num_steps: int = 5
    ) -> ReasoningTrace:
        """Execute chain-of-thought reasoning"""
        start_time = datetime.now()
        
        steps = []
        current_reasoning = problem
        
        prompt = f"""Solve this medical problem step-by-step:

Problem: {problem}

Context: {json.dumps(context, indent=2)}

Provide {num_steps} clear reasoning steps, each building on the previous:

Step 1: [First consideration]
Step 2: [Building on step 1]
...

Let's think through this carefully:"""
        
        # Generate full reasoning chain
        response = agent.generate_response(prompt, max_new_tokens=768)
        
        # Parse steps
        lines = response.split('\n')
        step_num = 1
        current_step_text = []
        
        for line in lines:
            if line.strip().startswith(f'Step {step_num}:'):
                if current_step_text:
                    # Save previous step
                    step_content = '\n'.join(current_step_text)
                    steps.append(ThoughtStep(
                        step_number=step_num - 1,
                        thought=step_content,
                        confidence=self._assess_step_quality(step_content)
                    ))
                    current_step_text = []
                
                current_step_text.append(line)
                step_num += 1
            elif current_step_text:
                current_step_text.append(line)
        
        # Add final step
        if current_step_text:
            step_content = '\n'.join(current_step_text)
            steps.append(ThoughtStep(
                step_number=len(steps) + 1,
                thought=step_content,
                confidence=self._assess_step_quality(step_content)
            ))
        
        # Extract final answer from last step
        final_answer = steps[-1].thought if steps else response
        
        end_time = datetime.now()
        
        return ReasoningTrace(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            steps=steps,
            final_answer=final_answer,
            confidence_score=sum(s.confidence for s in steps) / len(steps) if steps else 0.5,
            evidence_sources=[],
            reasoning_time=(end_time - start_time).total_seconds()
        )
    
    def _assess_step_quality(self, step_text: str) -> float:
        """Assess quality/confidence of reasoning step"""
        # Check for medical terminology and logical connectors
        quality_indicators = [
            'therefore', 'because', 'thus', 'indicates',
            'suggests', 'consistent with', 'rule out', 'differential'
        ]
        
        score = 0.5
        text_lower = step_text.lower()
        
        for indicator in quality_indicators:
            if indicator in text_lower:
                score += 0.05
        
        # Length indicates thoroughness
        if len(step_text) > 100:
            score += 0.1
        
        return min(score, 1.0)


class ReflectiveReasoner:
    """
    Reflective Reasoning with Self-Critique
    Generate answer, critique it, improve it
    """
    
    def reason(
        self,
        query: str,
        context: Dict,
        agent,
        max_reflections: int = 2
    ) -> ReasoningTrace:
        """Execute reflective reasoning with self-improvement"""
        start_time = datetime.now()
        
        steps = []
        
        # Initial answer
        initial_prompt = f"""Provide an initial answer to this medical question:

Question: {query}

Context: {json.dumps(context, indent=2)}

Initial Answer:"""
        
        current_answer = agent.generate_response(initial_prompt, max_new_tokens=512)
        
        steps.append(ThoughtStep(
            step_number=1,
            thought="Initial answer generated",
            observation=current_answer,
            confidence=0.6
        ))
        
        # Reflection cycles
        for reflection_num in range(max_reflections):
            # Critique current answer
            critique_prompt = f"""Critically evaluate this medical answer:

Question: {query}
Answer: {current_answer}

Identify:
1. Strengths
2. Weaknesses or gaps
3. Missing considerations
4. Potential errors

Critique:"""
            
            critique = agent.generate_response(critique_prompt, max_new_tokens=384)
            
            steps.append(ThoughtStep(
                step_number=len(steps) + 1,
                thought=f"Reflection {reflection_num + 1}: Self-critique",
                observation=critique,
                confidence=0.7
            ))
            
            # Improve answer based on critique
            improve_prompt = f"""Improve the answer based on this critique:

Original Question: {query}
Previous Answer: {current_answer}
Critique: {critique}

Provide an improved, more comprehensive answer:"""
            
            improved_answer = agent.generate_response(improve_prompt, max_new_tokens=512)
            
            steps.append(ThoughtStep(
                step_number=len(steps) + 1,
                thought=f"Improved answer after reflection {reflection_num + 1}",
                observation=improved_answer,
                confidence=0.75 + (reflection_num * 0.1)
            ))
            
            current_answer = improved_answer
        
        end_time = datetime.now()
        
        return ReasoningTrace(
            strategy=ReasoningStrategy.REFLECTIVE,
            steps=steps,
            final_answer=current_answer,
            confidence_score=0.85,  # Higher confidence after reflection
            evidence_sources=[],
            reasoning_time=(end_time - start_time).total_seconds(),
            reflection=steps[-2].observation if len(steps) > 1 else None
        )


class SocraticReasoner:
    """
    Socratic Reasoning through Guided Questions
    Uses questions to explore the problem space
    """
    
    def reason(
        self,
        problem: str,
        context: Dict,
        agent
    ) -> ReasoningTrace:
        """Execute Socratic reasoning through questions"""
        start_time = datetime.now()
        
        questions = [
            "What are the key facts in this case?",
            "What are the most likely explanations?",
            "What evidence supports or refutes each possibility?",
            "What additional information would be helpful?",
            "What is the most reasonable conclusion?"
        ]
        
        steps = []
        accumulated_knowledge = f"Problem: {problem}\nContext: {json.dumps(context)}\n\n"
        
        for i, question in enumerate(questions):
            prompt = f"""{accumulated_knowledge}

Question {i+1}: {question}

Answer:"""
            
            answer = agent.generate_response(prompt, max_new_tokens=384)
            
            steps.append(ThoughtStep(
                step_number=i + 1,
                thought=question,
                observation=answer,
                confidence=0.6 + (i * 0.05)
            ))
            
            accumulated_knowledge += f"Q{i+1}: {question}\nA{i+1}: {answer}\n\n"
        
        # Final synthesis
        final_prompt = f"""{accumulated_knowledge}

Based on the Socratic dialogue above, provide a final comprehensive answer to:
{problem}

Final Answer:"""
        
        final_answer = agent.generate_response(final_prompt, max_new_tokens=512)
        
        end_time = datetime.now()
        
        return ReasoningTrace(
            strategy=ReasoningStrategy.SOCRATIC,
            steps=steps,
            final_answer=final_answer,
            confidence_score=0.8,
            evidence_sources=[step.observation for step in steps],
            reasoning_time=(end_time - start_time).total_seconds()
        )


class UncertaintyQuantifier:
    """
    Quantify uncertainty in medical reasoning
    Provides confidence intervals and epistemic uncertainty
    """
    
    @staticmethod
    def calculate_confidence(
        reasoning_trace: ReasoningTrace,
        context: Dict
    ) -> Dict[str, Any]:
        """Calculate comprehensive confidence metrics"""
        
        # Factors affecting confidence
        factors = {
            'reasoning_depth': len(reasoning_trace.steps) / 10.0,  # More steps = more thorough
            'evidence_quality': len(reasoning_trace.evidence_sources) / 5.0,
            'consistency': UncertaintyQuantifier._check_consistency(reasoning_trace),
            'completeness': UncertaintyQuantifier._assess_completeness(context)
        }
        
        # Weight factors
        weights = {
            'reasoning_depth': 0.25,
            'evidence_quality': 0.30,
            'consistency': 0.25,
            'completeness': 0.20
        }
        
        # Calculate weighted confidence
        weighted_confidence = sum(
            min(factors[k], 1.0) * weights[k]
            for k in factors
        )
        
        # Determine uncertainty category
        if weighted_confidence > 0.8:
            uncertainty_level = "Low"
            recommendation = "High confidence in assessment"
        elif weighted_confidence > 0.6:
            uncertainty_level = "Moderate"
            recommendation = "Consider additional evaluation"
        else:
            uncertainty_level = "High"
            recommendation = "Recommend specialist consultation"
        
        return {
            'confidence_score': round(weighted_confidence, 3),
            'confidence_interval': (
                round(weighted_confidence - 0.1, 3),
                round(weighted_confidence + 0.05, 3)
            ),
            'uncertainty_level': uncertainty_level,
            'contributing_factors': factors,
            'recommendation': recommendation
        }
    
    @staticmethod
    def _check_consistency(trace: ReasoningTrace) -> float:
        """Check consistency across reasoning steps"""
        if len(trace.steps) < 2:
            return 0.7
        
        # Check if conclusions align (simplified)
        confidences = [step.confidence for step in trace.steps]
        variance = sum((c - trace.confidence_score) ** 2 for c in confidences) / len(confidences)
        
        consistency = 1.0 - min(variance, 0.5) * 2
        return consistency
    
    @staticmethod
    def _assess_completeness(context: Dict) -> float:
        """Assess if sufficient information available"""
        required_fields = ['age', 'gender', 'symptoms']
        optional_fields = ['history', 'medications', 'vital_signs', 'labs']
        
        case = context.get('case', {})
        
        required_score = sum(1 for f in required_fields if case.get(f)) / len(required_fields)
        optional_score = sum(1 for f in optional_fields if case.get(f)) / len(optional_fields)
        
        return required_score * 0.7 + optional_score * 0.3


class EvidenceTracker:
    """
    Track and cite evidence for medical claims
    Maintain provenance of information
    """
    
    def __init__(self):
        self.evidence_database = []
    
    def add_evidence(
        self,
        claim: str,
        source: str,
        confidence: float,
        reasoning: str
    ):
        """Add evidence for a claim"""
        self.evidence_database.append({
            'claim': claim,
            'source': source,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_evidence_for_claim(self, claim: str) -> List[Dict]:
        """Retrieve evidence supporting a claim"""
        return [e for e in self.evidence_database if claim.lower() in e['claim'].lower()]
    
    def generate_evidence_summary(self) -> str:
        """Generate formatted summary of evidence"""
        if not self.evidence_database:
            return "No evidence tracked"
        
        summary = ["Evidence Summary:", ""]
        
        for i, evidence in enumerate(self.evidence_database, 1):
            summary.append(f"{i}. {evidence['claim']}")
            summary.append(f"   Source: {evidence['source']}")
            summary.append(f"   Confidence: {evidence['confidence']:.2f}")
            summary.append(f"   Reasoning: {evidence['reasoning']}")
            summary.append("")
        
        return "\n".join(summary)
