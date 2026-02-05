"""
Production-ready API endpoints
FastAPI with proper error handling, validation, and documentation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from typing import Optional
import time

from medassist.schemas import (
    QueryRequest, QueryResponse, 
    DocumentIngestionRequest, IngestionResponse,
    KnowledgeGraphQueryRequest, KnowledgeGraphResponse,
    ErrorResponse, HealthResponse
)
from medassist.core.langgraph_orchestrator import LangGraphMedicalOrchestrator
from medassist.services.ingestion_pipeline import IngestionPipeline
from medassist.models.knowledge_graph import MedicalKnowledgeGraph

logger = logging.getLogger(__name__)

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="MedGemma API",
    description="Medical Multi-Agent System with LangGraph",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
orchestrator: Optional[LangGraphMedicalOrchestrator] = None
ingestion_pipeline: Optional[IngestionPipeline] = None
knowledge_graph: Optional[MedicalKnowledgeGraph] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global orchestrator, ingestion_pipeline, knowledge_graph
    
    logger.info("Initializing MedGemma API...")
    
    try:
        # Initialize orchestrator
        orchestrator = LangGraphMedicalOrchestrator(
            use_memory_graph=True,
            load_sample_graph=True,
            llm_provider="groq",
            model_name="llama-3.3-70b-versatile"
        )
        
        # Initialize ingestion pipeline
        knowledge_graph = orchestrator.knowledge_graph
        ingestion_pipeline = IngestionPipeline(config=None)
        ingestion_pipeline.kg = knowledge_graph
        
        logger.info("MedGemma API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down MedGemma API...")


# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "MedGemma API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        components={
            "orchestrator": "operational" if orchestrator else "unavailable",
            "knowledge_graph": "operational" if knowledge_graph else "unavailable",
            "ingestion": "operational" if ingestion_pipeline else "unavailable"
        },
        timestamp=datetime.utcnow().isoformat()
    )


# ============================================================================
# QUERY ENDPOINTS
# ============================================================================

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process medical query through LangGraph multi-agent workflow
    
    - **question**: Medical question (required)
    - **context**: Additional context (optional)
    - **image_base64**: Base64-encoded medical image (optional)
    """
    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")
        
        logger.info(f"📥 Query: {request.question[:100]}...")
        
        # Add context to question if provided
        full_question = request.question
        if request.context:
            full_question = f"{request.question}\n\nContext: {request.context}"
        
        # Execute workflow
        result = orchestrator.query(
            question=full_question,
            image_base64=request.image_base64
        )
        
        # Convert to response schema
        from medassist.schemas import (
            FindingsSchema, EntitySchema, KnowledgePathSchema,
            DifferentialDiagnosisSchema, TreatmentRecommendationSchema,
            EvidenceSchema
        )
        
        findings_data = result.get('findings', {})
        
        response = QueryResponse(
            query=request.question,
            answer=result.get('answer', ''),
            confidence=result.get('confidence', 0.0),
            agents_used=result.get('agents_used', []),
            findings=FindingsSchema(
                entities=[
                    EntitySchema(**e) for e in findings_data.get('entities', [])
                ],
                knowledge_paths=[
                    KnowledgePathSchema(**kp) for kp in findings_data.get('knowledge_paths', [])
                ],
                symptoms=findings_data.get('symptoms', []),
                differential_diagnosis=[
                    DifferentialDiagnosisSchema(**dx) for dx in findings_data.get('differential_diagnosis', [])
                ] if isinstance(findings_data.get('differential_diagnosis', []), list) else [],
                treatments=[
                    TreatmentRecommendationSchema(**t) for t in findings_data.get('treatments', [])
                ] if isinstance(findings_data.get('treatments', []), list) else [],
                evidence=[
                    EvidenceSchema(**e) for e in findings_data.get('evidence', [])
                ]
            ),
            workflow_trace=result.get('workflow_trace'),
            processing_time=result.get('processing_time', 0.0)
        )
        
        logger.info(f"Response generated (confidence: {response.confidence:.2%})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    Stream query response (for future SSE implementation)
    """
    raise HTTPException(status_code=501, detail="Streaming not yet implemented")


# ============================================================================
# INGESTION ENDPOINTS
# ============================================================================

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(request: DocumentIngestionRequest, background_tasks: BackgroundTasks):
    """
    Ingest medical document into knowledge graph
    
    - **text**: Document text (min 50 chars)
    - **metadata**: Optional metadata
    - **extract_relationships**: Extract relationships (default: true)
    """
    try:
        if not ingestion_pipeline:
            raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")
        
        logger.info(f"Ingesting document ({len(request.text)} chars)...")
        
        # Extract entities and relationships
        result = ingestion_pipeline.process_document(
            text=request.text,
            metadata=request.metadata,
            extract_relationships=request.extract_relationships
        )
        
        response = IngestionResponse(
            success=True,
            entities_extracted=len(result.get('entities', [])),
            relationships_extracted=len(result.get('relationships', [])),
            message="Document ingested successfully",
            entities=[EntitySchema(**e) for e in result.get('entities', [])],
            relationships=None  # Optional: include if needed
        )
        
        logger.info(f"Ingested: {response.entities_extracted} entities, {response.relationships_extracted} relationships")
        return response
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# KNOWLEDGE GRAPH ENDPOINTS
# ============================================================================

@app.post("/kg/explore", response_model=KnowledgeGraphResponse)
async def explore_knowledge_graph(request: KnowledgeGraphQueryRequest):
    """
    Explore knowledge graph relationships
    
    - **entity_name**: Entity to explore
    - **max_depth**: Max traversal depth (1-5)
    - **relationship_types**: Filter by types (optional)
    """
    try:
        if not knowledge_graph:
            raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
        
        logger.info(f"Exploring: {request.entity_name}")
        
        # Find related entities
        related = knowledge_graph.find_related_entities(
            entity_name=request.entity_name,
            max_depth=request.max_depth,
            relationship_types=[rt.value for rt in request.relationship_types] if request.relationship_types else None
        )
        
        response = KnowledgeGraphResponse(
            entity=request.entity_name,
            connections=related,
            total_connections=len(related),
            depth_reached=request.max_depth
        )
        
        logger.info(f"Found {response.total_connections} connections")
        return response
        
    except Exception as e:
        logger.error(f"KG exploration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/stats", response_model=dict)
async def knowledge_graph_stats():
    """Get knowledge graph statistics"""
    try:
        if not knowledge_graph:
            raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
        
        stats = knowledge_graph.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
