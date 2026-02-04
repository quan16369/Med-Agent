"""
Medical Named Entity Recognition
Extracts medical entities (diseases, symptoms, treatments) from text
"""

import logging
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

# Try to import transformers for BioBERT
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available, using fallback NER")
    TRANSFORMERS_AVAILABLE = False


@dataclass
class MedicalEntity:
    """Extracted medical entity with AMG-RAG enhancements"""
    text: str
    entity_type: str  # disease, symptom, treatment, anatomy, biomarker
    start: int
    end: int
    confidence: float
    relevance_score: float = 5.0  # AMG-RAG: 1-10 scale, 10=most relevant
    description: Optional[str] = None  # AMG-RAG: Context-aware description


class BioBERTNER:
    """
    Medical NER using BioBERT transformer models
    
    Supported entity types:
    - Diseases (Diabetes, Hypertension, Cancer)
    - Symptoms (Fever, Cough, Pain)
    - Treatments (Metformin, Surgery)
    - Anatomy (Heart, Lung, Brain)
    - Biomarkers (HbA1c, Glucose)
    """
    
    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading BioBERT model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(model_name)
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if device == "cuda" else -1,
                    aggregation_strategy="simple"
                )
                logger.info("BioBERT NER initialized successfully")
                self.use_transformer = True
            except Exception as e:
                logger.error(f"Failed to load BioBERT: {e}")
                logger.info("Falling back to rule-based NER")
                self.use_transformer = False
        else:
            logger.info("Using rule-based NER (transformers not available)")
            self.use_transformer = False
        
        # Medical vocabulary for fallback
        self._init_medical_vocabulary()
    
    def _init_medical_vocabulary(self):
        """Initialize medical vocabulary for rule-based NER"""
        
        self.disease_patterns = {
            "diabetes", "diabetes mellitus", "type 2 diabetes", "hypertension",
            "heart disease", "coronary heart disease", "chd", "cancer",
            "neuropathy", "diabetic neuropathy", "retinopathy", "diabetic retinopathy",
            "kidney disease", "renal disease", "stroke", "obesity",
            "asthma", "copd", "pneumonia", "tuberculosis", "tb",
            "alzheimer", "parkinson", "epilepsy", "depression", "anxiety"
        }
        
        self.symptom_patterns = {
            "fever", "cough", "pain", "chest pain", "headache", "fatigue",
            "numbness", "tingling", "weakness", "dizziness", "nausea",
            "vomiting", "diarrhea", "constipation", "shortness of breath",
            "blurred vision", "vision loss", "weight loss", "weight gain",
            "thirst", "frequent urination", "confusion", "memory loss"
        }
        
        self.treatment_patterns = {
            "metformin", "insulin", "aspirin", "statin", "atorvastatin",
            "lisinopril", "amlodipine", "omeprazole", "albuterol",
            "chemotherapy", "radiation", "surgery", "physical therapy",
            "antibiotics", "antivirals", "vaccine", "immunotherapy"
        }
        
        self.anatomy_patterns = {
            "heart", "brain", "lung", "liver", "kidney", "pancreas",
            "stomach", "intestine", "colon", "bladder", "eye", "retina",
            "nerve", "blood vessel", "artery", "vein", "muscle", "bone"
        }
        
        self.biomarker_patterns = {
            "glucose", "blood sugar", "hba1c", "hemoglobin a1c",
            "cholesterol", "ldl", "hdl", "triglycerides",
            "blood pressure", "heart rate", "bmi", "creatinine",
            "egfr", "alt", "ast", "hemoglobin", "white blood cell"
        }
    
    def extract(self, text: str) -> List[MedicalEntity]:
        """
        Extract medical entities from text
        
        Args:
            text: Input text
        
        Returns:
            List of MedicalEntity objects
        """
        if self.use_transformer:
            return self._extract_transformer(text)
        else:
            return self._extract_rule_based(text)
    
    def _extract_transformer(self, text: str) -> List[MedicalEntity]:
        """Extract using BioBERT transformer"""
        
        try:
            # Run NER pipeline
            ner_results = self.ner_pipeline(text)
            
            # Convert to MedicalEntity
            entities = []
            for result in ner_results:
                entity_type = self._map_label_to_type(result['entity_group'])
                
                entity = MedicalEntity(
                    text=result['word'],
                    entity_type=entity_type,
                    start=result['start'],
                    end=result['end'],
                    confidence=result['score']
                )
                entities.append(entity)
            
            return entities
        
        except Exception as e:
            logger.error(f"BioBERT extraction failed: {e}")
            return self._extract_rule_based(text)
    
    def _map_label_to_type(self, label: str) -> str:
        """Map BioBERT label to medical entity type"""
        
        label_lower = label.lower()
        
        if 'disease' in label_lower or 'condition' in label_lower:
            return 'disease'
        elif 'symptom' in label_lower or 'sign' in label_lower:
            return 'symptom'
        elif 'drug' in label_lower or 'medication' in label_lower or 'treatment' in label_lower:
            return 'treatment'
        elif 'anatomy' in label_lower or 'organ' in label_lower:
            return 'anatomy'
        elif 'biomarker' in label_lower or 'lab' in label_lower:
            return 'biomarker'
        else:
            return 'disease'  # Default
    
    def _extract_rule_based(self, text: str) -> List[MedicalEntity]:
        """Extract using rule-based pattern matching"""
        
        text_lower = text.lower()
        entities = []
        
        # Find diseases
        for pattern in self.disease_patterns:
            for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', text_lower):
                entities.append(MedicalEntity(
                    text=pattern,
                    entity_type='disease',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Find symptoms
        for pattern in self.symptom_patterns:
            for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', text_lower):
                entities.append(MedicalEntity(
                    text=pattern,
                    entity_type='symptom',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Find treatments
        for pattern in self.treatment_patterns:
            for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', text_lower):
                entities.append(MedicalEntity(
                    text=pattern,
                    entity_type='treatment',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Find anatomy
        for pattern in self.anatomy_patterns:
            for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', text_lower):
                entities.append(MedicalEntity(
                    text=pattern,
                    entity_type='anatomy',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Find biomarkers
        for pattern in self.biomarker_patterns:
            for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', text_lower):
                entities.append(MedicalEntity(
                    text=pattern,
                    entity_type='biomarker',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # Remove duplicates (keep highest confidence)
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove overlapping entities, keeping highest confidence"""
        
        if not entities:
            return []
        
        # Sort by start position
        entities_sorted = sorted(entities, key=lambda e: (e.start, -e.confidence))
        
        deduplicated = []
        last_end = -1
        
        for entity in entities_sorted:
            if entity.start >= last_end:
                deduplicated.append(entity)
                last_end = entity.end
        
        return deduplicated
    
    def extract_entity_types(self, text: str) -> Dict[str, Set[str]]:
        """
        Extract entities grouped by type
        
        Returns:
            Dict mapping entity_type -> Set of entity texts
        """
        entities = self.extract(text)
        
        entity_types = {
            'disease': set(),
            'symptom': set(),
            'treatment': set(),
            'anatomy': set(),
            'biomarker': set()
        }
        
        for entity in entities:
            entity_types[entity.entity_type].add(entity.text)
        
        return entity_types


class HybridMedicalNER:
    """
    Hybrid NER combining BioBERT with medical vocabulary.
    Uses BioBERT for primary extraction, falls back to rules for missed entities.
    Enhanced with AMG-RAG relevance scoring.
    """
    
    def __init__(
        self,
        biobert_model: str = "dmis-lab/biobert-base-cased-v1.1",
        device: str = "cpu",
        use_hybrid: bool = True,
        llm=None  # AMG-RAG: Optional LLM for relevance scoring
    ):
        self.biobert = BioBERTNER(biobert_model, device)
        self.use_hybrid = use_hybrid
        self.llm = llm
    
    def extract(self, text: str, context: Optional[str] = None) -> List[MedicalEntity]:
        """
        Extract with hybrid approach and optional relevance scoring.
        
        Args:
            text: Text to extract entities from
            context: Optional context for relevance scoring (AMG-RAG)
            
        Returns:
            List of MedicalEntity with relevance scores
        """
        
        # Get BioBERT entities
        entities = self.biobert.extract(text)
        
        if self.use_hybrid and self.biobert.use_transformer:
            # Get rule-based entities
            rule_entities = self.biobert._extract_rule_based(text)
            
            # Merge, avoiding duplicates
            entities = self._merge_entities(entities, rule_entities)
        
        # AMG-RAG: Score relevance if context provided
        if context and self.llm and entities:
            entities = self._score_relevance(entities, text, context)
        
        return entities
    
    def _score_relevance(
        self,
        entities: List[MedicalEntity],
        text: str,
        context: str
    ) -> List[MedicalEntity]:
        """
        Score entity relevance (1-10 scale) using LLM (AMG-RAG pattern).
        
        Relevance scale:
        10 = directly related to question
        7-9 = moderately relevant
        4-6 = weakly relevant
        1-3 = minimally relevant
        """
        try:
            from langchain.prompts import PromptTemplate
            from langchain.output_parsers import ResponseSchema, StructuredOutputParser
            
            entity_names = [e.text for e in entities]
            
            # Create prompt for relevance scoring
            schemas = [
                ResponseSchema(
                    name="scores",
                    description="Relevance scores (1-10) for each entity",
                    type="array"
                ),
                ResponseSchema(
                    name="descriptions",
                    description="Brief context-aware descriptions for each entity",
                    type="array"
                )
            ]
            
            parser = StructuredOutputParser.from_response_schemas(schemas)
            
            prompt = PromptTemplate(
                template="""Rate the relevance of each medical entity to the given context.
                
                Context: {context}
                Text: {text}
                Entities: {entities}
                
                For each entity, provide:
                1. Relevance score (1-10): 10=directly related to question, 7-9=moderately relevant, 4-6=weakly relevant, 1-3=minimally relevant
                2. Brief description of the entity in the context of the question (2-3 sentences)
                
                {format_instructions}""",
                input_variables=["context", "text", "entities"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            chain = prompt | self.llm | parser
            result = chain.invoke({
                "context": context,
                "text": text,
                "entities": entity_names
            })
            
            scores = result.get("scores", [])
            descriptions = result.get("descriptions", [])
            
            # Update entities with scores and descriptions
            for i, entity in enumerate(entities):
                if i < len(scores):
                    entity.relevance_score = float(scores[i])
                if i < len(descriptions):
                    entity.description = descriptions[i]
            
        except Exception as e:
            logger.warning(f"Relevance scoring failed: {e}")
            # Keep default scores (5.0)
        
        return entities
    
    def extract(self, text: str) -> List[MedicalEntity]:
        """Extract with hybrid approach"""
        
        # Get BioBERT entities
        entities = self.biobert.extract(text)
        
        if self.use_hybrid and self.biobert.use_transformer:
            # Get rule-based entities
            rule_entities = self.biobert._extract_rule_based(text)
            
            # Merge, avoiding duplicates
            entities = self._merge_entities(entities, rule_entities)
        
        return entities
    
    def _merge_entities(
        self,
        entities1: List[MedicalEntity],
        entities2: List[MedicalEntity]
    ) -> List[MedicalEntity]:
        """Merge two entity lists, removing duplicates"""
        
        # Create set of (text, type) for deduplication
        seen = set()
        merged = []
        
        for entity in entities1 + entities2:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                merged.append(entity)
        
        return merged


if __name__ == "__main__":
    # Demo
    print("Medical NER Demo")
    print("="*60)
    
    # Initialize NER
    ner = BioBERTNER()
    
    # Test queries
    test_texts = [
        "Patient has diabetes mellitus with symptoms of numbness and blurred vision.",
        "Prescribed metformin 500mg for blood sugar control.",
        "Diabetic neuropathy causing tingling in feet, check HbA1c levels.",
        "History of hypertension and heart disease, on aspirin and statin therapy.",
        "Chest pain and shortness of breath, possible coronary heart disease."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        print("-"*60)
        
        entities = ner.extract(text)
        
        print(f"Found {len(entities)} entities:")
        for entity in entities:
            print(f"  [{entity.entity_type}] {entity.text} (conf: {entity.confidence:.2f})")
        
        # Group by type
        entity_types = ner.extract_entity_types(text)
        print("\nGrouped by type:")
        for etype, texts in entity_types.items():
            if texts:
                print(f"  {etype}: {', '.join(texts)}")
    
    print("\n" + "="*60)
    print("NER Demo Complete")
