"""
Pydantic Schemas for API requests/responses
Production-ready data validation
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class EntityType(str, Enum):
    """Medical entity types"""
    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    MEDICATION = "Medication"
    PROCEDURE = "Procedure"
    ANATOMY = "Anatomy"
    BIOMARKER = "Biomarker"
    GENE = "Gene"


class RelationshipType(str, Enum):
    """Medical relationship types"""
    TREATS = "treats"
    CAUSES = "causes"
    SYMPTOM_OF = "symptom_of"
    CONTRAINDICATED = "contraindicated"
    INTERACTS_WITH = "interacts_with"
    LOCATED_IN = "located_in"


class QueryType(str, Enum):
    """Query classification"""
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    INFORMATION = "information"
    EXPLANATION = "explanation"


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class QueryRequest(BaseModel):
    """Medical query request"""
    question: str = Field(..., min_length=10, max_length=1000, description="Medical question")
    context: Optional[str] = Field(None, max_length=5000, description="Additional context")
    image_base64: Optional[str] = Field(None, description="Base64-encoded medical image")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class DocumentIngestionRequest(BaseModel):
    """Document ingestion request"""
    text: str = Field(..., min_length=50, description="Document text to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")
    extract_relationships: bool = Field(True, description="Extract entity relationships")
    
    @validator('text')
    def validate_text(cls, v):
        if len(v.split()) < 10:
            raise ValueError("Text must contain at least 10 words")
        return v


class KnowledgeGraphQueryRequest(BaseModel):
    """Knowledge graph exploration request"""
    entity_name: str = Field(..., min_length=2, max_length=200, description="Entity to explore")
    max_depth: int = Field(2, ge=1, le=5, description="Maximum traversal depth")
    relationship_types: Optional[List[RelationshipType]] = Field(None, description="Filter by relationship types")


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class EntitySchema(BaseModel):
    """Medical entity"""
    text: str
    type: EntityType
    confidence: float = Field(ge=0.0, le=1.0)
    relevance_score: Optional[float] = Field(None, ge=0, le=10)
    description: Optional[str] = None


class RelationshipSchema(BaseModel):
    """Medical relationship"""
    source: str
    target: str
    relationship_type: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[str] = None


class KnowledgePathSchema(BaseModel):
    """Knowledge graph path"""
    source: str
    related: List[Dict[str, Any]]


class DifferentialDiagnosisSchema(BaseModel):
    """Differential diagnosis entry"""
    condition: str
    likelihood: str  # high, medium, low
    supporting_factors: List[str]
    contradicting_factors: List[str]


class TreatmentRecommendationSchema(BaseModel):
    """Treatment recommendation"""
    treatment: str
    condition: str
    dosage: Optional[str] = None
    contraindications: List[str] = Field(default_factory=list)
    monitoring: Optional[str] = None


class EvidenceSchema(BaseModel):
    """Research evidence"""
    title: str
    abstract: Optional[str] = None
    pmid: Optional[str] = None
    year: Optional[int] = None
    relevance: Optional[float] = None


class FindingsSchema(BaseModel):
    """Gathered findings from agents"""
    entities: List[EntitySchema] = Field(default_factory=list)
    knowledge_paths: List[KnowledgePathSchema] = Field(default_factory=list)
    symptoms: List[str] = Field(default_factory=list)
    differential_diagnosis: List[DifferentialDiagnosisSchema] = Field(default_factory=list)
    treatments: List[TreatmentRecommendationSchema] = Field(default_factory=list)
    evidence: List[EvidenceSchema] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Medical query response"""
    query: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    query_type: Optional[QueryType] = None
    agents_used: List[str] = Field(default_factory=list)
    findings: FindingsSchema
    workflow_trace: Optional[str] = None
    processing_time: float = Field(ge=0.0)
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the first-line treatment for type 2 diabetes?",
                "answer": "Metformin is the first-line medication...",
                "confidence": 0.85,
                "query_type": "treatment",
                "agents_used": ["knowledge", "treatment", "validator"],
                "findings": {
                    "entities": [
                        {"text": "Type 2 Diabetes", "type": "Disease", "confidence": 0.95}
                    ],
                    "treatments": [
                        {"treatment": "Metformin", "condition": "Type 2 Diabetes"}
                    ]
                },
                "processing_time": 3.2
            }
        }


class IngestionResponse(BaseModel):
    """Document ingestion response"""
    success: bool
    entities_extracted: int
    relationships_extracted: int
    message: str
    entities: Optional[List[EntitySchema]] = None
    relationships: Optional[List[RelationshipSchema]] = None


class KnowledgeGraphResponse(BaseModel):
    """Knowledge graph exploration response"""
    entity: str
    connections: List[Dict[str, Any]]
    total_connections: int
    depth_reached: int


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    trace_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    components: Dict[str, str]
    timestamp: str
