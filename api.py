"""
FastAPI application for medical question answering service.
Provides REST API endpoints for agentic workflow execution.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
import time
from datetime import datetime
import logging
import base64
from pathlib import Path

from medassist.config import get_config
from medassist.logging_utils import setup_logging, get_logger
from medassist.health import HealthChecker
from medassist.exceptions import MedAssistError, handle_error
from medassist.core.agentic_orchestrator import AgenticMedicalOrchestrator
from medassist.services.mcp_server import MCPServer, MCPRequest
from medassist.services.ingestion_pipeline import IngestionPipeline
from medassist.models.multimodal_models import (
    MultimodalMessage, 
    TextContent, 
    ImageUrlContent,
    MedicalImageInput
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Knowledge Assistant API",
    description="Agentic workflow system for medical question answering with knowledge graph reasoning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[AgenticMedicalOrchestrator] = None
mcp_server: Optional[MCPServer] = None
ingestion_pipeline: Optional[IngestionPipeline] = None
health_checker = HealthChecker()

# Request/Response models
class QueryRequest(BaseModel):
    """Medical question request"""
    question: str = Field(..., min_length=10, max_length=1000, description="Medical question")
    include_trace: bool = Field(default=False, description="Include workflow trace in response")
    include_stats: bool = Field(default=False, description="Include agent statistics")
    image_base64: Optional[str] = Field(None, description="Optional medical image (base64 encoded)")
    image_metadata: Optional[Dict[str, Any]] = Field(None, description="Image metadata (type, body_part, etc.)")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

class AgentStatistics(BaseModel):
    """Agent execution statistics"""
    agent_name: str
    calls: int
    avg_confidence: float

class WorkflowTrace(BaseModel):
    """Workflow execution trace"""
    step: int
    agent: str
    action: str
    confidence: Optional[float]
    timestamp: str

class QueryResponse(BaseModel):
    """Medical question response"""
    question: str
    answer: str
    confidence: float
    execution_time: float
    timestamp: str
    trace: Optional[List[WorkflowTrace]] = None
    statistics: Optional[List[AgentStatistics]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    checks: Dict[str, Any]

class StatsResponse(BaseModel):
    """System statistics response"""
    total_queries: int
    avg_response_time: float
    orchestrator_loaded: bool
    config: Dict[str, Any]

# Rate limiting state
request_counts: Dict[str, List[float]] = {}
RATE_LIMIT = 10  # requests per minute per IP

def check_rate_limit(ip: str) -> bool:
    """Check if IP has exceeded rate limit"""
    now = time.time()
    minute_ago = now - 60
    
    if ip not in request_counts:
        request_counts[ip] = []
    
    # Clean old requests
    request_counts[ip] = [t for t in request_counts[ip] if t > minute_ago]
    
    if len(request_counts[ip]) >= RATE_LIMIT:
        return False
    
    request_counts[ip].append(now)
    return True

# Statistics tracking
query_count = 0
total_execution_time = 0.0

@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    global orchestrator, mcp_server, ingestion_pipeline
    try:
        logger.info("Starting Medical Knowledge Assistant API")
        config = get_config()
        logger.info(f"Configuration loaded: environment={config.logging.environment}")
        
        # Initialize orchestrator
        orchestrator = AgenticMedicalOrchestrator()
        logger.info("Orchestrator initialized successfully")
        
        # Initialize MCP Server
        mcp_server = MCPServer(
            knowledge_graph=orchestrator.kg,
            graph_retriever=orchestrator.graph_retriever,
            pubmed_retriever=orchestrator.pubmed,
            ner=orchestrator.ner
        )
        logger.info("MCP Server initialized successfully")
        
        # Initialize ingestion pipeline
        ingestion_pipeline = IngestionPipeline(kg=orchestrator.kg)
        logger.info("Ingestion pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Medical Knowledge Assistant API")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Check rate limit
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Maximum 10 requests per minute."}
        )
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={process_time:.3f}s"
    )
    
    return response

@app.post("/query", response_model=QueryResponse)
async def query_medical_question(request: QueryRequest) -> QueryResponse:
    """
    Process a medical question using the agentic workflow.
    
    The system uses 5 specialized agents:
    - Knowledge Agent: Queries medical knowledge graph
    - Diagnostic Agent: Analyzes symptoms
    - Treatment Agent: Recommends treatments
    - Evidence Agent: Retrieves scientific evidence
    - Validator Agent: Validates findings
    """
    global query_count, total_execution_time
    
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        start_time = time.time()
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Execute workflow
        result = orchestrator.execute_workflow(request.question)
        
        execution_time = time.time() - start_time
        query_count += 1
        total_execution_time += execution_time
        
        # Build response
        response = QueryResponse(
            question=request.question,
            answer=result["answer"],
            confidence=result["confidence"],
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Add optional trace
        if request.include_trace and "trace" in result:
            response.trace = [
                WorkflowTrace(
                    step=i+1,
                    agent=step.get("agent", "unknown"),
                    action=step.get("action", "unknown"),
                    confidence=step.get("confidence"),
                    timestamp=step.get("timestamp", "")
                )
                for i, step in enumerate(result["trace"])
            ]
        
        # Add optional statistics
        if request.include_stats and "statistics" in result:
            response.statistics = [
                AgentStatistics(**stat)
                for stat in result["statistics"]
            ]
        
        logger.info(f"Query completed in {execution_time:.2f}s with confidence {result['confidence']:.2f}")
        return response
        
    except MedAssistError as e:
        logger.error(f"Application error: {e}")
        raise HTTPException(status_code=400, detail=handle_error(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring.
    Returns liveness, readiness, and startup status.
    """
    try:
        liveness = health_checker.liveness()
        readiness = health_checker.readiness(orchestrator)
        startup = health_checker.startup(orchestrator)
        
        overall_status = "healthy"
        if any(check.status.value != "healthy" for check in [liveness, readiness, startup]):
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            checks={
                "liveness": {
                    "status": liveness.status.value,
                    "message": liveness.message
                },
                "readiness": {
                    "status": readiness.status.value,
                    "message": readiness.message,
                    "details": readiness.details
                },
                "startup": {
                    "status": startup.status.value,
                    "message": startup.message
                }
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

@app.get("/health/liveness")
async def liveness_probe():
    """Kubernetes liveness probe"""
    result = health_checker.liveness()
    if result.status.value != "healthy":
        raise HTTPException(status_code=503, detail=result.message)
    return {"status": "ok"}

@app.get("/health/readiness")
async def readiness_probe():
    """Kubernetes readiness probe"""
    result = health_checker.readiness(orchestrator)
    if result.status.value != "healthy":
        raise HTTPException(status_code=503, detail=result.message)
    return {"status": "ok"}

@app.get("/stats", response_model=StatsResponse)
async def get_statistics() -> StatsResponse:
    """Get system statistics"""
    config = get_config()
    
    return StatsResponse(
        total_queries=query_count,
        avg_response_time=total_execution_time / query_count if query_count > 0 else 0.0,
        orchestrator_loaded=orchestrator is not None,
        config={
            "environment": config.logging.environment,
            "device": config.model.device,
            "max_depth": config.retrieval.max_depth,
            "max_width": config.retrieval.max_width
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Medical Knowledge Assistant API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "features": {
            "agentic_workflow": True,
            "mcp_server": True,
            "document_ingestion": True,
            "knowledge_graph": True
        }
    }


# MCP Server Endpoints

class MCPToolRequest(BaseModel):
    """MCP tool request"""
    tool: str = Field(..., description="Tool or skill name")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


@app.post("/mcp")
async def mcp_request(request: MCPToolRequest):
    """
    MCP Server endpoint.
    Execute tools and skills via Model Context Protocol.
    """
    if mcp_server is None:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")
    
    try:
        mcp_req = MCPRequest(
            tool=request.tool,
            parameters=request.parameters,
            context=request.context
        )
        
        response = mcp_server.process_request(mcp_req)
        
        return {
            "success": response.success,
            "result": response.result,
            "metadata": response.metadata,
            "error": response.error
        }
    except Exception as e:
        logger.error(f"MCP request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/capabilities")
async def mcp_capabilities():
    """Get MCP Server capabilities"""
    if mcp_server is None:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")
    
    return mcp_server.get_capabilities()


@app.get("/mcp/tools")
async def list_mcp_tools():
    """List available MCP tools"""
    if mcp_server is None:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")
    
    return {"tools": mcp_server.list_tools()}


@app.get("/mcp/skills")
async def list_mcp_skills():
    """List available MCP skills"""
    if mcp_server is None:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")
    
    return {"skills": mcp_server.list_skills()}


# Document Ingestion Endpoints

class DocumentIngestionRequest(BaseModel):
    """Document ingestion request with optional multimodal data"""
    content: str = Field(..., min_length=10, description="Document content")
    document_id: Optional[str] = Field(default=None, description="Document identifier")
    images: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="List of images with base64 data and metadata"
    )


class BatchIngestionRequest(BaseModel):
    """Batch document ingestion request"""
    documents: List[Dict[str, Any]] = Field(
        ..., 
        description="List of documents with content, optional id, and optional images"
    )


@app.post("/ingest/document")
async def ingest_document(request: DocumentIngestionRequest):
    """
    Ingest a single document into the knowledge graph.
    Extracts entities and relationships.
    Supports multimodal documents with text and images.
    """
    if ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")
    
    try:
        # Prepare document data
        doc_data = request.content
        if request.images:
            # Convert to multimodal format
            doc_data = {
                "text": request.content,
                "images": request.images
            }
        
        kg_object = ingestion_pipeline.ingest_document(
            doc_data,
            request.document_id
        )
        
        return {
            "success": True,
            "document_id": kg_object.source_document,
            "entities_extracted": len(kg_object.entities),
            "relationships_extracted": len(kg_object.relationships),
            "timestamp": kg_object.timestamp
        }
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/batch")
async def ingest_batch(request: BatchIngestionRequest):
    """
    Ingest multiple documents in batch.
    """
    if ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")
    
    try:
        kg_objects = ingestion_pipeline.ingest_batch(request.documents)
        
        return {
            "success": True,
            "documents_processed": len(kg_objects),
            "total_entities": sum(len(obj.entities) for obj in kg_objects),
            "total_relationships": sum(len(obj.relationships) for obj in kg_objects)
        }
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/stats")
async def ingestion_stats():
    """Get ingestion pipeline statistics"""
    if ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialized")
    
    return ingestion_pipeline.get_statistics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
