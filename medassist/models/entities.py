"""
AMG-RAG: Agentic Medical Graph-RAG
Core data structures based on https://github.com/MrRezaeiUofT/AMG-RAG

Following the paper's approach:
- MedicalEntity with confidence scoring (relevance 1-10)
- MedicalRelation with bidirectional relationships
- Dynamic knowledge graph construction
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MedicalEntity:
    """
    Represents a medical entity in the knowledge graph
    
    Attributes:
        name: Entity name (disease, drug, symptom, etc.)
        description: LLM-generated description with context
        entity_type: Type of medical entity (disease, treatment, symptom, imaging_finding, etc.)
        confidence: Confidence score (0-1) based on relevance
        sources: List of evidence sources (PubMed, Wikipedia, etc.)
        image_path: Optional path to associated medical image (for multimodal)
        image_findings: Optional list of findings from medical image analysis
    """
    name: str
    description: str
    entity_type: str = "medical_concept"
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)
    image_path: Optional[str] = None
    image_findings: Optional[List[str]] = None


@dataclass
class MedicalRelation:
    """
    Represents a relationship between medical entities
    
    AMG-RAG uses bidirectional analysis (A→B and B→A)
    
    Attributes:
        source: Source entity name
        target: Target entity name  
        relation_type: Type of relationship (treats, causes, symptom_of, etc.)
        confidence: Confidence score (0-1) for this relationship
        evidence: Supporting evidence text
        sources: Evidence sources
    """
    source: str
    target: str
    relation_type: str
    confidence: float = 0.5
    evidence: str = ""
    sources: List[str] = field(default_factory=list)


# Medical relationship types from AMG-RAG paper
MEDICAL_RELATION_TYPES = [
    "treats",
    "causes",
    "symptom_of",
    "risk_factor_for",
    "contraindicated_with",
    "differential_diagnosis",
    "associated_with",
    "prevents",
    "diagnoses",
    "related_to"
]

# Medical imaging modalities (for multimodal support)
IMAGING_MODALITIES = [
    "chest_xray",
    "ct_scan",
    "mri",
    "ultrasound",
    "pet_scan",
    "mammography",
    "dermatology",
    "histopathology",
    "fundoscopy",
    "endoscopy",
]
