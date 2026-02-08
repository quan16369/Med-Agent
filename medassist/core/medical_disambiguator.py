"""
Medical Term Disambiguation
Inspired by SaraCoder's External-Aware Identifier Disambiguator.

Resolves ambiguous medical terminology using:
1. Context-aware disambiguation
2. Medical ontology (UMLS-like)
3. Co-occurrence patterns
4. Knowledge graph relationships
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MedicalTermSense:
    """A single sense/meaning of an ambiguous medical term."""
    term: str
    sense_id: int
    full_name: str
    category: str  # disease, symptom, procedure, drug, etc.
    aliases: List[str]
    context_keywords: List[str]
    confidence: float


# Common ambiguous medical terms and their senses
MEDICAL_AMBIGUITY_DICT = {
    "MI": [
        MedicalTermSense(
            term="MI",
            sense_id=0,
            full_name="Myocardial Infarction",
            category="disease",
            aliases=["heart attack", "cardiac infarction"],
            context_keywords=["chest pain", "cardiac", "troponin", "ECG", "coronary", "ST elevation"],
            confidence=0.9
        ),
        MedicalTermSense(
            term="MI",
            sense_id=1,
            full_name="Mitral Insufficiency",
            category="disease",
            aliases=["mitral regurgitation", "MR"],
            context_keywords=["valve", "murmur", "regurgitation", "mitral", "echocardiography"],
            confidence=0.9
        ),
    ],
    "MS": [
        MedicalTermSense(
            term="MS",
            sense_id=0,
            full_name="Multiple Sclerosis",
            category="disease",
            aliases=["disseminated sclerosis"],
            context_keywords=["neurological", "demyelination", "CNS", "brain", "MRI", "lesions"],
            confidence=0.9
        ),
        MedicalTermSense(
            term="MS",
            sense_id=1,
            full_name="Mitral Stenosis",
            category="disease",
            aliases=["mitral valve stenosis"],
            context_keywords=["valve", "stenosis", "mitral", "heart", "murmur", "rheumatic"],
            confidence=0.9
        ),
    ],
    "PE": [
        MedicalTermSense(
            term="PE",
            sense_id=0,
            full_name="Pulmonary Embolism",
            category="disease",
            aliases=["lung embolism"],
            context_keywords=["chest pain", "dyspnea", "D-dimer", "CT angiography", "DVT", "lung"],
            confidence=0.9
        ),
        MedicalTermSense(
            term="PE",
            sense_id=1,
            full_name="Physical Examination",
            category="procedure",
            aliases=["clinical examination"],
            context_keywords=["examination", "inspection", "palpation", "auscultation", "vitals"],
            confidence=0.9
        ),
    ],
    "RA": [
        MedicalTermSense(
            term="RA",
            sense_id=0,
            full_name="Rheumatoid Arthritis",
            category="disease",
            aliases=["rheumatoid disease"],
            context_keywords=["joint", "arthritis", "inflammation", "RF", "anti-CCP", "autoimmune"],
            confidence=0.9
        ),
        MedicalTermSense(
            term="RA",
            sense_id=1,
            full_name="Right Atrium",
            category="anatomy",
            aliases=["right atrial"],
            context_keywords=["heart", "atrium", "cardiac", "chamber", "ECG", "P wave"],
            confidence=0.9
        ),
    ],
    "AS": [
        MedicalTermSense(
            term="AS",
            sense_id=0,
            full_name="Aortic Stenosis",
            category="disease",
            aliases=["aortic valve stenosis"],
            context_keywords=["valve", "stenosis", "aortic", "murmur", "heart", "syncope"],
            confidence=0.9
        ),
        MedicalTermSense(
            term="AS",
            sense_id=1,
            full_name="Ankylosing Spondylitis",
            category="disease",
            aliases=["Bechterew disease"],
            context_keywords=["spine", "spondylitis", "back pain", "HLA-B27", "inflammation"],
            confidence=0.9
        ),
    ],
    "DM": [
        MedicalTermSense(
            term="DM",
            sense_id=0,
            full_name="Diabetes Mellitus",
            category="disease",
            aliases=["diabetes"],
            context_keywords=["glucose", "insulin", "hyperglycemia", "HbA1c", "blood sugar"],
            confidence=0.9
        ),
        MedicalTermSense(
            term="DM",
            sense_id=1,
            full_name="Dermatomyositis",
            category="disease",
            aliases=["inflammatory myopathy"],
            context_keywords=["muscle", "myositis", "weakness", "rash", "autoimmune"],
            confidence=0.9
        ),
    ],
    "CA": [
        MedicalTermSense(
            term="CA",
            sense_id=0,
            full_name="Cancer",
            category="disease",
            aliases=["carcinoma", "malignancy"],
            context_keywords=["tumor", "malignant", "metastasis", "oncology", "biopsy"],
            confidence=0.9
        ),
        MedicalTermSense(
            term="CA",
            sense_id=1,
            full_name="Calcium",
            category="lab_value",
            aliases=["serum calcium"],
            context_keywords=["electrolyte", "hypercalcemia", "hypocalcemia", "mg/dL", "parathyroid"],
            confidence=0.9
        ),
    ],
}


class MedicalDisambiguator:
    """
    Disambiguates ambiguous medical terms using context.
    
    Inspired by SaraCoder's External-Aware Identifier Disambiguator:
    "accurately resolves cross-file symbol ambiguity via dependency analysis"
    
    For medical text, we resolve:
    - Abbreviations (MI, MS, PE, RA, etc.)
    - Polysemous terms (depression: psychiatric vs anatomical)
    - Context-dependent terms
    """
    
    def __init__(
        self,
        ambiguity_dict: Optional[Dict[str, List[MedicalTermSense]]] = None,
        context_window: int = 50
    ):
        """
        Args:
            ambiguity_dict: Dictionary of ambiguous terms and their senses
            context_window: Number of words to consider for context
        """
        self.ambiguity_dict = ambiguity_dict or MEDICAL_AMBIGUITY_DICT
        self.context_window = context_window
        
        # Build reverse index: full_name → abbreviation
        self.full_name_to_abbrev = {}
        for abbrev, senses in self.ambiguity_dict.items():
            for sense in senses:
                self.full_name_to_abbrev[sense.full_name.lower()] = abbrev
    
    def is_ambiguous(self, term: str) -> bool:
        """Check if a term is ambiguous."""
        return term.upper() in self.ambiguity_dict
    
    def get_senses(self, term: str) -> List[MedicalTermSense]:
        """Get all possible senses of a term."""
        return self.ambiguity_dict.get(term.upper(), [])
    
    def disambiguate(
        self,
        term: str,
        context: str,
        knowledge_graph: Optional[any] = None
    ) -> Tuple[MedicalTermSense, float]:
        """
        Disambiguate a medical term using context.
        
        Returns:
            (best_sense, confidence_score)
        """
        senses = self.get_senses(term)
        if not senses:
            # Not ambiguous or unknown term
            return None, 0.0
        
        if len(senses) == 1:
            return senses[0], 1.0
        
        # Score each sense based on context
        scores = []
        context_lower = context.lower()
        
        for sense in senses:
            # Count keyword matches in context
            keyword_matches = sum(
                1 for keyword in sense.context_keywords
                if keyword.lower() in context_lower
            )
            
            # Count alias matches
            alias_matches = sum(
                1 for alias in sense.aliases
                if alias.lower() in context_lower
            )
            
            # Check knowledge graph co-occurrence
            kg_score = 0.0
            if knowledge_graph is not None:
                kg_score = self._compute_kg_coherence(sense, context, knowledge_graph)
            
            # Combined score
            score = (
                0.5 * keyword_matches +
                0.3 * alias_matches +
                0.2 * kg_score
            )
            scores.append((sense, score))
        
        # Select best sense
        if scores:
            best_sense, best_score = max(scores, key=lambda x: x[1])
            # Normalize confidence
            total_score = sum(s for _, s in scores)
            confidence = best_score / total_score if total_score > 0 else 0.0
            return best_sense, confidence
        
        # Default to first sense
        return senses[0], 0.5
    
    def _compute_kg_coherence(
        self,
        sense: MedicalTermSense,
        context: str,
        knowledge_graph: any
    ) -> float:
        """
        Compute how well a sense fits with the knowledge graph.
        
        Checks if sense is connected to other entities mentioned in context.
        """
        if not hasattr(knowledge_graph, 'graph'):
            return 0.0
        
        graph = knowledge_graph.graph
        sense_name = sense.full_name
        
        if sense_name not in graph.nodes:
            return 0.0
        
        # Find other medical entities in context
        context_entities = []
        for node in graph.nodes:
            if node.lower() in context.lower():
                context_entities.append(node)
        
        if not context_entities:
            return 0.0
        
        # Count connections to context entities
        connections = 0
        for entity in context_entities:
            if entity == sense_name:
                continue
            if graph.has_edge(sense_name, entity) or graph.has_edge(entity, sense_name):
                connections += 1
        
        # Normalize by number of context entities
        return connections / len(context_entities)
    
    def expand_abbreviations(
        self,
        text: str,
        knowledge_graph: Optional[any] = None
    ) -> Dict[str, Tuple[str, float]]:
        """
        Expand all abbreviations in text to full names.
        
        Returns:
            Dict mapping abbreviation → (full_name, confidence)
        """
        expansions = {}
        
        # Find all potential abbreviations
        words = text.split()
        for word in words:
            clean_word = word.strip('.,;:!?()[]{}')
            if self.is_ambiguous(clean_word):
                # Get context window around the word
                word_idx = text.index(word)
                start = max(0, word_idx - self.context_window * 5)  # ~5 chars per word
                end = min(len(text), word_idx + self.context_window * 5)
                context = text[start:end]
                
                # Disambiguate
                sense, confidence = self.disambiguate(clean_word, context, knowledge_graph)
                if sense:
                    expansions[clean_word] = (sense.full_name, confidence)
        
        return expansions
    
    def resolve_entity_ambiguity(
        self,
        entities: List[Dict[str, any]],
        context: str,
        knowledge_graph: Optional[any] = None
    ) -> List[Dict[str, any]]:
        """
        Resolve ambiguous entity names using context.
        
        Updates entity names with disambiguated full forms.
        """
        resolved_entities = []
        
        for entity in entities:
            name = entity.get('name', '')
            
            if self.is_ambiguous(name):
                sense, confidence = self.disambiguate(name, context, knowledge_graph)
                if sense and confidence > 0.6:
                    # Update entity with full name
                    entity = entity.copy()
                    entity['name'] = sense.full_name
                    entity['abbreviation'] = name
                    entity['disambiguation_confidence'] = confidence
                    entity['entity_type'] = sense.category
                    
                    logger.info(f"Disambiguated '{name}' → '{sense.full_name}' (confidence: {confidence:.2f})")
            
            resolved_entities.append(entity)
        
        return resolved_entities
    
    def add_custom_sense(
        self,
        term: str,
        full_name: str,
        category: str,
        context_keywords: List[str],
        aliases: Optional[List[str]] = None
    ):
        """
        Add a custom ambiguous term to the dictionary.
        
        Useful for domain-specific abbreviations.
        """
        term_upper = term.upper()
        
        if term_upper not in self.ambiguity_dict:
            self.ambiguity_dict[term_upper] = []
        
        sense = MedicalTermSense(
            term=term_upper,
            sense_id=len(self.ambiguity_dict[term_upper]),
            full_name=full_name,
            category=category,
            aliases=aliases or [],
            context_keywords=context_keywords,
            confidence=0.8
        )
        
        self.ambiguity_dict[term_upper].append(sense)
        logger.info(f"Added custom sense: {term} → {full_name}")
