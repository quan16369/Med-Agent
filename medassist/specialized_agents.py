"""
Specialized Medical Agents with Advanced Reasoning
Each agent handles a specific aspect of the medical workflow with sophisticated reasoning capabilities
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import json

from .base_agent import BaseAgent, AgentTools
from .reasoning_engine import (
    ReActReasoner, ChainOfThoughtReasoner, ReflectiveReasoner,
    SocraticReasoner, UncertaintyQuantifier, EvidenceTracker,
    ReasoningStrategy, ReasoningTrace
)
from config import AGENT_ROLES

logger = logging.getLogger(__name__)


class MedicalHistoryAgent(BaseAgent):
    """
    Agent specialized in analyzing patient medical history
    Uses Chain-of-Thought reasoning for systematic analysis
    """
    
    def __init__(self, model, tokenizer):
        role_config = AGENT_ROLES['history']
        super().__init__(
            name=role_config['name'],
            role='history',
            system_prompt=role_config['system_prompt'],
            model=model,
            tokenizer=tokenizer,
            tools=[]
        )
        self.cot_reasoner = ChainOfThoughtReasoner()
        self.evidence_tracker = EvidenceTracker()
        self.uncertainty_quantifier = UncertaintyQuantifier()
    
    def process(self, input_data: Dict) -> Dict:
        """
        Analyze patient medical history with sophisticated reasoning
        
        Args:
            input_data: Contains case information
            
        Returns:
            Dictionary with comprehensive history analysis including confidence and evidence
        """
        case = input_data.get('case', {})
        
        # Use Chain-of-Thought reasoning for systematic analysis
        if self.model is not None:
            reasoning_trace = self.cot_reasoner.reason(
                problem=f"Analyze medical history for: {case.get('age')}yo {case.get('gender')} with symptoms: {case.get('symptoms', 'Not specified')}",
                context={'case': case},
                agent=self,
                num_steps=4
            )
            analysis = reasoning_trace.final_answer
            confidence_metrics = self.uncertainty_quantifier.calculate_confidence(
                reasoning_trace, {'case': case}
            )
        else:
            analysis = self._mock_history_analysis(case)
            confidence_metrics = {'confidence_score': 0.6, 'uncertainty_level': 'Moderate'}
        
        # Parse and structure findings
        findings = self._parse_history_findings(analysis, case)
        
        # Extract and track evidence
        risk_factors = findings.get('risk_factors', [])
        for risk in risk_factors:
            self.evidence_tracker.add_evidence(
                claim=risk,
                source='Patient history',
                confidence=confidence_metrics.get('confidence_score', 0.6),
                reasoning='Identified from medical history review'
            )
        
        return {
            "findings": findings,
            "raw_analysis": analysis,
            "reasoning": "Systematic chain-of-thought analysis of patient history",
            "confidence_score": confidence_metrics.get('confidence_score'),
            "uncertainty_level": confidence_metrics.get('uncertainty_level'),
            "evidence_summary": self.evidence_tracker.generate_evidence_summary(),
            "reasoning_trace": reasoning_trace if self.model else None
        }
    
    def _build_history_prompt(self, case: Dict) -> str:
        """Build prompt for history analysis"""
        prompt_parts = [
            "Analyze the following patient information and identify relevant risk factors:",
            "",
            f"Age: {case.get('age', 'Unknown')}",
            f"Gender: {case.get('gender', 'Unknown')}",
        ]
        
        if 'history' in case:
            prompt_parts.append(f"Medical History: {case['history']}")
        
        if 'medications' in case:
            prompt_parts.append(f"Current Medications: {case['medications']}")
        
        if 'allergies' in case:
            prompt_parts.append(f"Allergies: {case['allergies']}")
        
        if 'family_history' in case:
            prompt_parts.append(f"Family History: {case['family_history']}")
        
        if 'social_history' in case:
            prompt_parts.append(f"Social History: {case['social_history']}")
        
        prompt_parts.extend([
            "",
            "Please provide:",
            "1. Relevant risk factors",
            "2. Pertinent positive findings",
            "3. Pertinent negative findings",
            "4. Medication considerations"
        ])
        
        return "\n".join(prompt_parts)
    
    def _mock_history_analysis(self, case: Dict) -> str:
        """Mock analysis when model not available"""
        age = case.get('age', 0)
        gender = case.get('gender', 'unknown')
        
        analysis = f"Patient is a {age}-year-old {gender}. "
        
        if 'history' in case:
            analysis += f"Medical history notable for: {case['history']}. "
        
        # Add basic risk assessment
        if age > 50:
            analysis += "Age-related cardiovascular risk factors present. "
        
        if 'smoking' in str(case.get('social_history', '')).lower():
            analysis += "Smoking history increases respiratory and cardiovascular risks. "
        
        return analysis
    
    def _parse_history_findings(self, analysis: str, case: Dict) -> Dict:
        """Parse analysis into structured findings"""
        return {
            "age": case.get('age'),
            "gender": case.get('gender'),
            "risk_factors": self._extract_risk_factors(case),
            "medications": case.get('medications', []),
            "allergies": case.get('allergies', []),
            "summary": analysis
        }
    
    def _extract_risk_factors(self, case: Dict) -> List[str]:
        """Extract risk factors from case"""
        risk_factors = []
        
        age = case.get('age', 0)
        if age > 65:
            risk_factors.append("Advanced age")
        
        history = str(case.get('history', '')).lower()
        if 'diabetes' in history:
            risk_factors.append("Diabetes mellitus")
        if 'hypertension' in history:
            risk_factors.append("Hypertension")
        if 'smoking' in history or 'smoker' in history:
            risk_factors.append("Tobacco use")
        
        return risk_factors


class DiagnosticAgent(BaseAgent):
    """
    Agent specialized in diagnostic reasoning
    Uses ReAct pattern with Reflective reasoning for differential diagnosis
    """
    
    def __init__(self, model, tokenizer):
        role_config = AGENT_ROLES['diagnostic']
        super().__init__(
            name=role_config['name'],
            role='diagnostic',
            system_prompt=role_config['system_prompt'],
            model=model,
            tokenizer=tokenizer,
            tools=[]
        )
        self.react_reasoner = ReActReasoner(max_iterations=8)
        self.reflective_reasoner = ReflectiveReasoner()
        self.socratic_reasoner = SocraticReasoner()
        self.evidence_tracker = EvidenceTracker()
        self.uncertainty_quantifier = UncertaintyQuantifier()
    
    def process(self, input_data: Dict) -> Dict:
        """
        Generate differential diagnosis with multi-strategy reasoning
        
        Args:
            input_data: Contains case and previous findings
            
        Returns:
            Dictionary with comprehensive diagnostic assessment, confidence, and alternatives
        """
        case = input_data.get('case', {})
        previous_findings = input_data.get('previous_findings', {})
        
        if self.model is not None:
            # Use Socratic reasoning for initial exploration
            socratic_trace = self.socratic_reasoner.reason(
                problem=f"Diagnose: {case.get('symptoms', 'Not specified')}",
                context={'case': case, 'previous_findings': previous_findings},
                agent=self
            )
            
            # Use Reflective reasoning to critique and improve
            reflective_trace = self.reflective_reasoner.reason(
                query=f"What is the most likely diagnosis for: {case.get('symptoms')}?",
                context={'case': case, 'socratic_findings': socratic_trace.final_answer},
                agent=self,
                max_reflections=2
            )
            
            diagnosis = reflective_trace.final_answer
            
            # Calculate comprehensive confidence
            confidence_metrics = self.uncertainty_quantifier.calculate_confidence(
                reflective_trace, {'case': case}
            )
            
            # Track alternative diagnoses considered
            alternatives = self._extract_alternatives(socratic_trace.steps)
        else:
            diagnosis = self._mock_diagnostic_reasoning(case)
            confidence_metrics = {'confidence_score': 0.65, 'uncertainty_level': 'Moderate'}
            alternatives = []
        
        # Structure findings
        findings = self._parse_diagnostic_findings(diagnosis, case)
        
        # Track evidence for each diagnosis
        for dx in findings.get('differential_diagnosis', [])[:3]:
            self.evidence_tracker.add_evidence(
                claim=dx,
                source='Clinical reasoning',
                confidence=confidence_metrics.get('confidence_score', 0.65),
                reasoning='Based on symptom pattern and clinical presentation'
            )
        
        return {
            "findings": findings,
            "raw_diagnosis": diagnosis,
            "reasoning": "Multi-strategy reasoning: Socratic exploration + Reflective refinement",
            "confidence_score": confidence_metrics.get('confidence_score'),
            "uncertainty_level": confidence_metrics.get('uncertainty_level'),
            "confidence_interval": confidence_metrics.get('confidence_interval'),
            "alternatives_considered": alternatives,
            "evidence_summary": self.evidence_tracker.generate_evidence_summary(),
            "reasoning_traces": {
                'socratic': socratic_trace if self.model else None,
                'reflective': reflective_trace if self.model else None
            }
        }
    
    def _build_diagnostic_prompt(self, case: Dict, previous_findings: Dict) -> str:
        """Build prompt for diagnostic reasoning"""
        prompt_parts = [
            "Based on the following patient presentation, generate a differential diagnosis:",
            "",
            f"Chief Complaint: {case.get('symptoms', 'Not specified')}",
        ]
        
        if 'vital_signs' in case:
            prompt_parts.append(f"Vital Signs: {case['vital_signs']}")
        
        if 'physical_exam' in case:
            prompt_parts.append(f"Physical Exam: {case['physical_exam']}")
        
        # Add context from history agent
        if 'history' in previous_findings:
            history = previous_findings['history']
            if history.get('risk_factors'):
                prompt_parts.append(f"Risk Factors: {', '.join(history['risk_factors'])}")
        
        prompt_parts.extend([
            "",
            "Please provide:",
            "1. Top 3-5 differential diagnoses (ordered by likelihood)",
            "2. Key distinguishing features for each",
            "3. Recommended diagnostic tests",
            "4. Red flags or warning signs"
        ])
        
        return "\n".join(prompt_parts)
    
    def _mock_diagnostic_reasoning(self, case: Dict) -> str:
        """Mock diagnostic reasoning"""
        symptoms = str(case.get('symptoms', '')).lower()
        
        diagnoses = []
        
        # Simple pattern matching for demonstration
        if 'cough' in symptoms and 'fever' in symptoms:
            diagnoses.extend([
                "1. Community-acquired pneumonia",
                "2. Acute bronchitis",
                "3. Tuberculosis (if chronic)"
            ])
        elif 'chest pain' in symptoms:
            diagnoses.extend([
                "1. Acute coronary syndrome",
                "2. Pulmonary embolism",
                "3. Musculoskeletal pain"
            ])
        elif 'headache' in symptoms:
            diagnoses.extend([
                "1. Tension headache",
                "2. Migraine",
                "3. Meningitis (if fever present)"
            ])
        else:
            diagnoses.append("1. Further evaluation needed")
        
        return "\n".join(diagnoses)
    
    def _extract_alternatives(self, steps: List) -> List[str]:
        """Extract alternative diagnoses from reasoning steps"""
        alternatives = []
        for step in steps:
            text = step.observation or step.thought
            if 'diagnosis' in text.lower() or 'differential' in text.lower():
                # Extract diagnosis names (simplified)
                words = text.split()
                for i, word in enumerate(words):
                    if i < len(words) - 1 and word.lower() in ['diagnosis:', 'consider:', 'rule out:']:
                        alternatives.append(' '.join(words[i+1:i+4]))
        return list(set(alternatives))[:5]
    
    def _parse_diagnostic_findings(self, diagnosis: str, case: Dict) -> Dict:
        """Parse diagnosis into structured format with confidence per diagnosis"""
        lines = diagnosis.strip().split('\n')
        differential = []
        differential_with_confidence = []
        
        for line in lines:
            if line.strip() and any(char.isdigit() for char in line[:3]):
                # Remove numbering
                clean_line = line.split('.', 1)[-1].strip()
                if clean_line:
                    differential.append(clean_line)
                    # Assign decreasing confidence to lower-ranked diagnoses
                    confidence = 0.85 - (len(differential) * 0.1)
                    differential_with_confidence.append({
                        'diagnosis': clean_line,
                        'confidence': max(confidence, 0.4),
                        'rank': len(differential)
                    })
        
        return {
            "differential_diagnosis": differential,
            "differential_with_confidence": differential_with_confidence,
            "top_diagnosis": differential[0] if differential else "Unknown",
            "top_diagnosis_confidence": differential_with_confidence[0]['confidence'] if differential_with_confidence else 0.5,
            "recommended_tests": self._suggest_tests(case),
            "urgency": self._assess_urgency(case),
            "reasoning_strategy": "Socratic + Reflective"
        }
    
    def _suggest_tests(self, case: Dict) -> List[str]:
        """Suggest relevant diagnostic tests"""
        tests = ["Complete Blood Count (CBC)", "Basic Metabolic Panel"]
        
        symptoms = str(case.get('symptoms', '')).lower()
        
        if 'chest' in symptoms or 'cough' in symptoms:
            tests.extend(["Chest X-ray", "Sputum culture"])
        if 'fever' in symptoms:
            tests.append("Blood cultures")
        if 'pain' in symptoms:
            tests.append("Inflammatory markers (ESR, CRP)")
        
        return tests
    
    def _assess_urgency(self, case: Dict) -> str:
        """Assess case urgency"""
        symptoms = str(case.get('symptoms', '')).lower()
        
        urgent_keywords = ['chest pain', 'shortness of breath', 'severe', 'acute']
        
        if any(keyword in symptoms for keyword in urgent_keywords):
            return "High - Prompt evaluation recommended"
        else:
            return "Routine"


class TreatmentAgent(BaseAgent):
    """
    Agent specialized in treatment planning
    Uses ReAct pattern with tool integration for evidence-based recommendations
    """
    
    def __init__(self, model, tokenizer):
        role_config = AGENT_ROLES['treatment']
        super().__init__(
            name=role_config['name'],
            role='treatment',
            system_prompt=role_config['system_prompt'],
            model=model,
            tokenizer=tokenizer,
            tools=[AgentTools()]
        )
        self.react_reasoner = ReActReasoner(max_iterations=6)
        self.reflective_reasoner = ReflectiveReasoner()
        self.evidence_tracker = EvidenceTracker()
        self.uncertainty_quantifier = UncertaintyQuantifier()
    
    def process(self, input_data: Dict) -> Dict:
        """
        Generate treatment plan with evidence-based reasoning and safety checks
        
        Args:
            input_data: Contains case and previous findings
            
        Returns:
            Dictionary with comprehensive treatment recommendations with confidence and contraindications
        """
        case = input_data.get('case', {})
        previous_findings = input_data.get('previous_findings', {})
        
        if self.model is not None:
            # Use ReAct for treatment planning with tool integration
            diagnosis = previous_findings.get('diagnostic', {}).get('top_diagnosis', 'Unknown')
            
            react_trace = self.react_reasoner.reason(
                query=f"Develop treatment plan for {diagnosis} in {case.get('age')}yo {case.get('gender')}",
                context={'case': case, 'diagnosis': diagnosis, 'previous': previous_findings},
                agent=self,
                available_tools=self.tools
            )
            
            # Reflect on treatment plan for safety and completeness
            reflective_trace = self.reflective_reasoner.reason(
                query=f"Review and optimize treatment plan for {diagnosis}",
                context={'initial_plan': react_trace.final_answer, 'case': case},
                agent=self,
                max_reflections=1
            )
            
            treatment = reflective_trace.final_answer
            
            # Calculate confidence with safety emphasis
            confidence_metrics = self.uncertainty_quantifier.calculate_confidence(
                reflective_trace, {'case': case}
            )
            
            # Check contraindications
            contraindications = self._check_contraindications(case, treatment)
        else:
            treatment = self._mock_treatment_plan(case, previous_findings)
            confidence_metrics = {'confidence_score': 0.7, 'uncertainty_level': 'Moderate'}
            contraindications = []
        
        # Structure recommendations
        findings = self._parse_treatment_findings(treatment, case)
        findings['contraindications_checked'] = contraindications
        
        # Track evidence for treatment recommendations
        for med in findings.get('medications', [])[:3]:
            med_name = med.get('name', str(med)) if isinstance(med, dict) else str(med)
            self.evidence_tracker.add_evidence(
                claim=f"Recommend {med_name}",
                source='Evidence-based guidelines',
                confidence=confidence_metrics.get('confidence_score', 0.7),
                reasoning='Standard treatment for confirmed diagnosis'
            )
        
        return {
            "findings": findings,
            "raw_treatment": treatment,
            "reasoning": "ReAct reasoning with safety reflection and contraindication checking",
            "confidence_score": confidence_metrics.get('confidence_score'),
            "uncertainty_level": confidence_metrics.get('uncertainty_level'),
            "safety_checks": contraindications,
            "evidence_summary": self.evidence_tracker.generate_evidence_summary(),
            "reasoning_traces": {
                'react': react_trace if self.model else None,
                'reflective': reflective_trace if self.model else None
            }
        }
    
    def _build_treatment_prompt(self, case: Dict, previous_findings: Dict) -> str:
        """Build prompt for treatment planning"""
        prompt_parts = [
            "Develop a treatment plan for the following case:",
            ""
        ]
        
        # Add diagnosis if available
        if 'diagnostic' in previous_findings:
            diagnosis = previous_findings['diagnostic'].get('top_diagnosis', 'Unknown')
            prompt_parts.append(f"Working Diagnosis: {diagnosis}")
        
        # Add patient context
        prompt_parts.append(f"Patient: {case.get('age', 'Unknown')} year old {case.get('gender', 'patient')}")
        
        # Add contraindications
        if 'allergies' in case:
            prompt_parts.append(f"Allergies: {case['allergies']}")
        
        if 'history' in previous_findings:
            history = previous_findings['history']
            if history.get('medications'):
                prompt_parts.append(f"Current Medications: {history['medications']}")
        
        prompt_parts.extend([
            "",
            "Please provide:",
            "1. Pharmacologic treatment recommendations",
            "2. Non-pharmacologic interventions",
            "3. Patient education points",
            "4. Follow-up recommendations"
        ])
        
        return "\n".join(prompt_parts)
    
    def _mock_treatment_plan(self, case: Dict, previous_findings: Dict) -> str:
        """Mock treatment plan"""
        diagnosis = "unknown condition"
        if 'diagnostic' in previous_findings:
            diagnosis = previous_findings['diagnostic'].get('top_diagnosis', diagnosis)
        
        plan = f"Treatment plan for {diagnosis}:\n\n"
        plan += "Pharmacologic:\n"
        plan += "- Appropriate antibiotic or medication as indicated\n"
        plan += "- Symptomatic relief medications\n\n"
        plan += "Non-pharmacologic:\n"
        plan += "- Rest and hydration\n"
        plan += "- Lifestyle modifications as appropriate\n\n"
        plan += "Patient education:\n"
        plan += "- Warning signs to watch for\n"
        plan += "- When to seek emergency care\n"
        
        return plan
    
    def _parse_treatment_findings(self, treatment: str, case: Dict) -> Dict:
        """Parse treatment plan into structured format"""
        return {
            "medications": self._extract_medications(treatment),
            "non_pharmacologic": self._extract_non_pharm(treatment),
            "patient_education": [
                "Monitor symptoms and report worsening",
                "Complete full course of prescribed medications",
                "Follow-up as recommended"
            ],
            "follow_up": ["Follow up in 1-2 weeks or sooner if symptoms worsen"],
            "summary": treatment
        }
    
    def _extract_medications(self, treatment: str) -> List[Dict]:
        """Extract medication recommendations"""
        # Simplified extraction
        meds = []
        if 'antibiotic' in treatment.lower():
            meds.append({
                "name": "Appropriate antibiotic",
                "indication": "As clinically indicated"
            })
        return meds
    
    def _check_contraindications(self, case: Dict, treatment: str) -> List[str]:
        """Check for potential contraindications"""
        contraindications = []
        
        allergies = case.get('allergies', '')
        medications = case.get('medications', '')
        history = case.get('history', '')
        
        text_lower = treatment.lower()
        
        # Check drug allergies
        if allergies:
            allergy_list = allergies.lower().split(',')
            for allergy in allergy_list:
                allergy = allergy.strip()
                if allergy in text_lower:
                    contraindications.append(f"ALERT: Patient allergic to {allergy}")
        
        # Check drug-disease interactions
        if 'renal' in history.lower() or 'kidney' in history.lower():
            if 'nsaid' in text_lower or 'ibuprofen' in text_lower:
                contraindications.append("CAUTION: NSAIDs in renal disease")
        
        if 'liver' in history.lower() or 'hepatic' in history.lower():
            if 'acetaminophen' in text_lower or 'tylenol' in text_lower:
                contraindications.append("CAUTION: Acetaminophen dosing in liver disease")
        
        # Check pregnancy
        if case.get('gender', '').lower() == 'female' and case.get('age', 100) < 50:
            if 'ace inhibitor' in text_lower or 'arb' in text_lower:
                contraindications.append("Consider: ACE/ARB contraindicated in pregnancy")
        
        return contraindications
    
    def _extract_non_pharm(self, treatment: str) -> List[str]:
        """Extract non-pharmacologic interventions"""
        interventions = []
        text = treatment.lower()
        
        if 'rest' in text:
            interventions.append("Adequate rest")
        if 'hydration' in text:
            interventions.append("Maintain hydration")
        if 'exercise' in text:
            interventions.append("Regular exercise as tolerated")
        if 'diet' in text:
            interventions.append("Dietary modifications")
        if 'physical therapy' in text:
            interventions.append("Physical therapy")
        
        return interventions if interventions else ["Supportive care"]


class KnowledgeAgent(BaseAgent):
    """
    Agent specialized in medical knowledge retrieval and synthesis
    Uses Chain-of-Thought for comprehensive literature review
    """
    
    def __init__(self, model, tokenizer):
        role_config = AGENT_ROLES['knowledge']
        super().__init__(
            name=role_config['name'],
            role='knowledge',
            system_prompt=role_config['system_prompt'],
            model=model,
            tokenizer=tokenizer,
            tools=[]
        )
        self.cot_reasoner = ChainOfThoughtReasoner()
        self.evidence_tracker = EvidenceTracker()
    
    def process(self, input_data: Dict) -> Dict:
        """
        Retrieve and synthesize medical knowledge with evidence grading
        
        Args:
            input_data: Contains query and context
            
        Returns:
            Dictionary with comprehensive knowledge synthesis and evidence quality
        """
        query = input_data.get('query', '')
        context = input_data.get('context', {})
        
        if not query:
            return {
                "findings": {"summary": "No specific query provided"},
                "reasoning": "Knowledge lookup requires a specific query",
                "confidence_score": 0.0
            }
        
        if self.model is not None:
            # Use Chain-of-Thought for systematic knowledge synthesis
            reasoning_trace = self.cot_reasoner.reason(
                problem=f"Provide comprehensive evidence-based information about: {query}",
                context={'query': query, 'clinical_context': context},
                agent=self,
                num_steps=5
            )
            
            knowledge = reasoning_trace.final_answer
            
            # Grade evidence quality
            evidence_grade = self._grade_evidence(knowledge)
        else:
            knowledge = self._mock_knowledge(query)
            evidence_grade = 'C'
        
        # Track knowledge sources
        self.evidence_tracker.add_evidence(
            claim=f"Evidence for {query}",
            source='Medical literature and guidelines',
            confidence=0.8 if evidence_grade in ['A', 'B'] else 0.6,
            reasoning='Based on systematic knowledge synthesis'
        )
        
        # Extract key findings
        key_points = self._extract_key_points(knowledge)
        guidelines = self._identify_relevant_guidelines(query)
        
        return {
            "findings": {
                "query": query,
                "summary": knowledge,
                "key_points": key_points,
                "evidence_grade": evidence_grade,
                "guidelines": guidelines,
                "confidence": 0.8 if evidence_grade in ['A', 'B'] else 0.6
            },
            "raw_knowledge": knowledge,
            "reasoning": "Systematic chain-of-thought knowledge synthesis with evidence grading",
            "evidence_summary": self.evidence_tracker.generate_evidence_summary(),
            "reasoning_trace": reasoning_trace if self.model else None
        }
    
    def _grade_evidence(self, knowledge: str) -> str:
        """Grade quality of evidence (A-D scale)"""
        text_lower = knowledge.lower()
        
        high_quality_terms = ['randomized', 'meta-analysis', 'systematic review', 'rct']
        moderate_quality_terms = ['cohort', 'case-control', 'observational']
        
        if any(term in text_lower for term in high_quality_terms):
            return 'A'  # High-quality evidence
        elif any(term in text_lower for term in moderate_quality_terms):
            return 'B'  # Moderate-quality evidence
        elif 'expert' in text_lower or 'consensus' in text_lower:
            return 'C'  # Expert opinion
        else:
            return 'D'  # Low-quality evidence
    
    def _extract_key_points(self, knowledge: str) -> List[str]:
        """Extract key points from knowledge synthesis"""
        # Simple extraction based on sentences
        sentences = knowledge.split('.')
        key_points = [s.strip() for s in sentences if len(s.strip()) > 50][:5]
        return key_points
    
    def _identify_relevant_guidelines(self, query: str) -> List[str]:
        """Identify relevant clinical guidelines"""
        guidelines_map = {
            'hypertension': 'ACC/AHA Hypertension Guidelines',
            'diabetes': 'ADA Diabetes Guidelines',
            'pneumonia': 'IDSA/ATS Pneumonia Guidelines',
            'copd': 'GOLD COPD Guidelines',
            'asthma': 'GINA Asthma Guidelines',
            'heart failure': 'ACC/AHA Heart Failure Guidelines'
        }
        
        query_lower = query.lower()
        relevant = [guideline for condition, guideline in guidelines_map.items() 
                   if condition in query_lower]
        
        return relevant if relevant else ['Refer to current specialty guidelines']
    
    def _mock_knowledge(self, query: str) -> str:
        """Mock knowledge retrieval"""
        return f"Clinical information about {query}: This would include evidence-based " \
               f"recommendations, current guidelines, and relevant research findings. " \
               f"In production, this would query medical databases and literature."
