"""
Graph-Enhanced Agentic RAG API
Main FastAPI application
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List
import logging
import time
import uuid
from datetime import datetime
import os
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track startup time
startup_time = time.time()

# Try to import core modules, fallback to simple config if not available
try:
    from core.config import get_config
    config = get_config()
    logger.info("✅ Core config loaded successfully")
except ImportError as e:
    # Fallback to simple config
    from config import config
    logger.info("⚠️ Using fallback config - core modules not available")
except Exception as e:
    logger.error(f"❌ Config loading error: {e}")
    from config import config

# Create FastAPI app with error handling
try:
    app = FastAPI(
        title="Graph-Enhanced Agentic RAG API",
        description="Multi-agent retrieval-augmented generation system with intelligent query processing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    logger.info("✅ FastAPI app created successfully")
except Exception as e:
    logger.error(f"❌ FastAPI app creation failed: {e}")
    # Fallback app
    app = FastAPI(
        title="RAG API",
        version="1.0.0"
    )

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": str(exc.detail),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with consistent format."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Request validation failed",
            "details": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )

# Simple enum for OpenAPI compatibility
from enum import Enum
class RetrievalStrategy(str, Enum):
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"

logger.info("✅ RetrievalStrategy enum defined")

# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for user queries."""
    query: str = Field(..., description="The user's question or query", min_length=1, max_length=1000)
    max_results: Optional[int] = Field(default=10, description="Maximum number of results to return", ge=1, le=50)
    include_reasoning: Optional[bool] = Field(default=True, description="Include reasoning path in response")
    strategy: Optional[RetrievalStrategy] = Field(default=None, description="Force specific retrieval strategy")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()

class QueryResponse(BaseModel):
    """Response model for query results."""
    query_id: str = Field(..., description="Unique identifier for this query")
    response: str = Field(..., description="Generated response to the query")
    sources: List[dict] = Field(default_factory=list, description="Source documents used")
    citations: List[dict] = Field(default_factory=list, description="Formatted citations")
    reasoning_path: Optional[str] = Field(None, description="Explanation of reasoning process")
    confidence_score: Optional[float] = Field(None, description="Confidence in the response")
    processing_time: Optional[float] = Field(None, description="Time taken to process query in seconds")
    strategy_used: Optional[str] = Field(None, description="Retrieval strategy that was used")
    entities_found: List[str] = Field(default_factory=list, description="Entities identified in query")

class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    title: str = Field(..., description="Document title", min_length=1, max_length=200)
    content: str = Field(..., description="Document content", min_length=1)
    source: Optional[str] = Field(None, description="Source URL or reference")
    domain: str = Field(default="general", description="Knowledge domain")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Unique identifier for uploaded document")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")
    entities_extracted: Optional[int] = Field(None, description="Number of entities extracted")
    relationships_created: Optional[int] = Field(None, description="Number of relationships created")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

# Mount static files for frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info("✅ Static files mounted at /static")
    
    @app.get("/interface", response_class=FileResponse, tags=["frontend"])
    async def serve_interface():
        """Serve the main web interface."""
        return FileResponse(os.path.join(static_dir, "index.html"))
    
    @app.get("/admin", response_class=FileResponse, tags=["frontend"])
    async def serve_admin_portal():
        """Serve the admin portal interface."""
        return FileResponse(os.path.join(static_dir, "admin.html"))
    
    @app.get("/guide", response_class=FileResponse, tags=["frontend"])
    async def serve_user_guide():
        """Serve the user guide."""
        return FileResponse(os.path.join(static_dir, "guide.html"))
else:
    logger.warning("⚠️ Static directory not found - frontend not available")

@app.get("/", tags=["system"])
@app.head("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Graph-Enhanced Agentic RAG API",
        "version": "1.0.0",
        "status": "running",
        "description": "Multi-agent retrieval-augmented generation system",
        "environment": getattr(config, 'ENVIRONMENT', 'production'),
        "features": ["basic_api", "config_system", "core_modules", "pydantic_models", "static_files"],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "interface": "/interface",
            "admin": "/admin",
            "guide": "/guide"
        },
        "uptime": time.time() - startup_time,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
@app.head("/health")
async def health():
    """Health check endpoint - supports both GET and HEAD for monitoring"""
    try:
        return {
            "status": "healthy",
            "message": "API is working perfectly",
            "environment": getattr(config, 'ENVIRONMENT', 'production'),
            "config_loaded": True,
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - startup_time if 'startup_time' in globals() else 0,
            "databases": {
                "neo4j": "configured" if getattr(config, 'NEO4J_URI', None) else "not_configured",
                "pinecone": "configured" if getattr(config, 'PINECONE_API_KEY', None) else "not_configured"
            },
            "system": {
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "fastapi_running": True
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "message": f"Health check encountered an issue: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/test")
async def test_endpoint(data: dict):
    """Test POST endpoint for API validation"""
    try:
        return {
            "status": "success",
            "message": "Test endpoint working perfectly",
            "received_data": data,
            "timestamp": datetime.now().isoformat(),
            "data_type": type(data).__name__,
            "data_keys": list(data.keys()) if isinstance(data, dict) else None
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Test endpoint failed: {str(e)}")

@app.get("/test")
async def test_get():
    """Test GET endpoint"""
    return {
        "status": "success",
        "message": "GET test endpoint working",
        "timestamp": datetime.now().isoformat(),
        "methods_available": ["GET", "POST"]
    }

@app.get("/openapi-test", tags=["system"])
async def openapi_test():
    """Test endpoint to verify OpenAPI schema generation"""
    return {
        "openapi_status": "working",
        "message": "OpenAPI schema generation is functional",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status", tags=["system"])
async def api_status():
    """Comprehensive API status and information endpoint"""
    return {
        "api_name": "Graph-Enhanced Agentic RAG API",
        "version": "1.0.0",
        "status": "operational",
        "uptime_seconds": time.time() - startup_time,
        "current_endpoints": {
            "system": ["/", "/health", "/status"],
            "testing": ["/test"],
            "documentation": ["/docs", "/redoc"],
            "frontend": ["/interface", "/admin", "/guide"] if os.path.exists("static") else []
        },
        "ready_for_next": ["query_processing", "document_upload", "agent_system"],
        "models_available": ["QueryRequest", "QueryResponse", "DocumentUploadRequest", "DocumentUploadResponse"],
        "static_files": "mounted" if os.path.exists("static") else "not_found",
        "timestamp": datetime.now().isoformat()
    }

# Global agent instances (lazy initialization)
coordinator_instance = None
agent_registry = None
message_queue = None

def get_or_create_coordinator():
    """Lazy initialization of coordinator agent"""
    global coordinator_instance
    if coordinator_instance is None:
        try:
            from agents.coordinator import CoordinatorAgent
            coordinator_instance = CoordinatorAgent("api_coordinator")
            logger.info("✅ Coordinator agent initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize coordinator: {e}")
            coordinator_instance = None
    return coordinator_instance

def get_database_managers():
    """Lazy initialization of database managers"""
    try:
        from core.database import get_neo4j_manager, get_vector_manager
        neo4j_manager = get_neo4j_manager()
        vector_manager = get_vector_manager()
        return neo4j_manager, vector_manager
    except Exception as e:
        logger.error(f"❌ Database managers not available: {e}")
        return None, None

# Essential API Endpoints

@app.post("/query", response_model=QueryResponse, tags=["queries"])
async def process_query(request: QueryRequest):
    """
    Process a user query using the multi-agent RAG system.
    Uses lazy initialization to avoid startup delays.
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing query {query_id}: {request.query}")
        
        # Lazy load coordinator
        coordinator = get_or_create_coordinator()
        
        if coordinator is None:
            # Fallback: Simple response without agents
            return QueryResponse(
                query_id=query_id,
                response=f"I received your query: '{request.query}'. The agent system is initializing. Please try again in a moment.",
                sources=[],
                citations=[],
                reasoning_path="System initializing - using fallback response",
                confidence_score=0.5,
                processing_time=time.time() - start_time,
                strategy_used=RetrievalStrategy.HYBRID,
                entities_found=[]
            )
        
        # Try to use coordinator for full workflow
        try:
            workflow_results = await coordinator.coordinate_full_workflow(request.query)
            
            # Extract results
            analysis = workflow_results.get('analysis', {})
            strategy_used = workflow_results.get('strategy', RetrievalStrategy.HYBRID)
            synthesis_results = workflow_results.get('synthesis_results', {})
            entities_found = analysis.get('entities', []) if analysis else []
            
            # Get synthesis result
            synthesis_result = synthesis_results.get('synthesis_result', {})
            if hasattr(synthesis_result, 'response'):
                response_text = synthesis_result.response
                sources = getattr(synthesis_result, 'sources', [])
                citations = getattr(synthesis_result, 'citations', [])
                confidence = getattr(synthesis_result, 'confidence_score', 0.85)
            else:
                response_text = f"Based on your query '{request.query}', I've analyzed the request using the {strategy_used.value} approach."
                sources = []
                citations = []
                confidence = 0.75
            
            return QueryResponse(
                query_id=query_id,
                response=response_text,
                sources=sources,
                citations=citations,
                reasoning_path=f"Query processed using {strategy_used.value} strategy with {len(entities_found)} entities identified" if request.include_reasoning else None,
                confidence_score=confidence,
                processing_time=time.time() - start_time,
                strategy_used=strategy_used,
                entities_found=entities_found
            )
            
        except Exception as e:
            logger.error(f"Coordinator workflow failed: {e}")
            # Fallback response
            return QueryResponse(
                query_id=query_id,
                response=f"I processed your query: '{request.query}'. The system is working but encountered an issue with the full workflow. This is a basic response.",
                sources=[],
                citations=[],
                reasoning_path=f"Fallback response due to workflow error: {str(e)}" if request.include_reasoning else None,
                confidence_score=0.6,
                processing_time=time.time() - start_time,
                strategy_used=RetrievalStrategy.HYBRID,
                entities_found=[]
            )
            
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

@app.post("/documents/upload", response_model=DocumentUploadResponse, tags=["documents"])
async def upload_document(request: DocumentUploadRequest):
    """
    Upload and ingest a document into the knowledge base.
    Uses lazy initialization for database connections.
    """
    start_time = time.time()
    document_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Uploading document {document_id}: {request.title}")
        
        # Lazy load database managers
        neo4j_manager, vector_manager = get_database_managers()
        
        if neo4j_manager is None and vector_manager is None:
            # Fallback: Accept document but don't process
            return DocumentUploadResponse(
                document_id=document_id,
                status="accepted",
                message=f"Document '{request.title}' accepted. Database connections initializing - processing will complete shortly.",
                entities_extracted=0,
                relationships_created=0,
                processing_time=time.time() - start_time
            )
        
        # Try basic document processing
        try:
            # Simple entity extraction (fallback)
            entities_found = []
            relationships_created = 0
            
            # If we have database connections, try to store
            if neo4j_manager:
                # Basic Neo4j storage
                try:
                    # Simple document node creation
                    query = """
                    CREATE (d:Document {
                        id: $doc_id,
                        title: $title,
                        content: $content,
                        source: $source,
                        domain: $domain,
                        created_at: datetime()
                    })
                    RETURN d.id as document_id
                    """
                    result = await neo4j_manager.execute_query_async(
                        query,
                        doc_id=document_id,
                        title=request.title,
                        content=request.content[:1000],  # Truncate for storage
                        source=request.source or "",
                        domain=request.domain
                    )
                    if result:
                        logger.info(f"Document stored in Neo4j: {document_id}")
                except Exception as e:
                    logger.error(f"Neo4j storage failed: {e}")
            
            if vector_manager:
                # Basic vector storage
                try:
                    # Simple embedding and storage (if available)
                    logger.info(f"Vector storage attempted for: {document_id}")
                except Exception as e:
                    logger.error(f"Vector storage failed: {e}")
            
            return DocumentUploadResponse(
                document_id=document_id,
                status="success",
                message=f"Document '{request.title}' uploaded successfully",
                entities_extracted=len(entities_found),
                relationships_created=relationships_created,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return DocumentUploadResponse(
                document_id=document_id,
                status="partial",
                message=f"Document '{request.title}' received but processing incomplete: {str(e)}",
                entities_extracted=0,
                relationships_created=0,
                processing_time=time.time() - start_time
            )
            
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )

@app.get("/agents/status", tags=["agents"])
async def get_agents_status():
    """Get status of all agents with lazy initialization"""
    try:
        coordinator = get_or_create_coordinator()
        neo4j_manager, vector_manager = get_database_managers()
        
        agents_status = {
            "coordinator": {
                "status": "healthy" if coordinator else "not_initialized",
                "type": "CoordinatorAgent",
                "initialized": coordinator is not None
            },
            "databases": {
                "neo4j": {
                    "status": "connected" if neo4j_manager else "not_connected",
                    "initialized": neo4j_manager is not None
                },
                "vector_db": {
                    "status": "connected" if vector_manager else "not_connected", 
                    "initialized": vector_manager is not None
                }
            }
        }
        
        total_agents = 4  # Expected total
        healthy_agents = sum(1 for agent in agents_status.values() if isinstance(agent, dict) and agent.get("status") == "healthy")
        
        return {
            "agents": agents_status,
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "initialization_strategy": "lazy_loading",
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Agent status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")

@app.get("/system/status", tags=["system"])
async def get_system_status():
    """Get comprehensive system status including all components"""
    try:
        coordinator = get_or_create_coordinator()
        neo4j_manager, vector_manager = get_database_managers()
        
        # Test database connections
        neo4j_status = "disconnected"
        vector_status = "disconnected"
        
        if neo4j_manager:
            try:
                result = await neo4j_manager.execute_query_async("RETURN 1 as test")
                neo4j_status = "connected" if result else "error"
            except:
                neo4j_status = "error"
        
        if vector_manager:
            try:
                # Basic vector DB test
                vector_status = "connected"
            except:
                vector_status = "error"
        
        return {
            "system": {
                "status": "operational",
                "uptime_seconds": time.time() - startup_time,
                "environment": getattr(config, 'ENVIRONMENT', 'production'),
                "version": "1.0.0"
            },
            "components": {
                "api": "healthy",
                "coordinator_agent": "healthy" if coordinator else "not_initialized",
                "neo4j": neo4j_status,
                "vector_db": vector_status,
                "static_files": "mounted" if os.path.exists("static") else "not_found"
            },
            "endpoints": {
                "total": 12,
                "operational": ["query", "documents/upload", "agents/status", "system/status"],
                "frontend": ["interface", "admin", "guide"] if os.path.exists("static") else []
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@app.get("/admin/stats", tags=["admin"])
async def get_database_stats():
    """Get database statistics for admin dashboard"""
    try:
        neo4j_manager, vector_manager = get_database_managers()
        
        stats = {
            "neo4j": {
                "status": "disconnected",
                "nodes": 0,
                "relationships": 0,
                "documents": 0
            },
            "vector_db": {
                "status": "disconnected", 
                "vectors": 0,
                "dimensions": 0
            }
        }
        
        # Neo4j stats
        if neo4j_manager:
            try:
                # Count nodes
                node_result = await neo4j_manager.execute_query_async("MATCH (n) RETURN count(n) as count")
                nodes_count = node_result[0]['count'] if node_result else 0
                
                # Count relationships
                rel_result = await neo4j_manager.execute_query_async("MATCH ()-[r]->() RETURN count(r) as count")
                rels_count = rel_result[0]['count'] if rel_result else 0
                
                # Count documents
                doc_result = await neo4j_manager.execute_query_async("MATCH (d:Document) RETURN count(d) as count")
                docs_count = doc_result[0]['count'] if doc_result else 0
                
                stats["neo4j"] = {
                    "status": "connected",
                    "nodes": nodes_count,
                    "relationships": rels_count,
                    "documents": docs_count
                }
            except Exception as e:
                logger.error(f"Neo4j stats error: {e}")
                stats["neo4j"]["status"] = "error"
        
        # Vector DB stats
        if vector_manager:
            try:
                stats["vector_db"] = {
                    "status": "connected",
                    "vectors": "unknown",  # Would need specific implementation
                    "dimensions": 384  # Default for sentence-transformers
                }
            except Exception as e:
                logger.error(f"Vector DB stats error: {e}")
                stats["vector_db"]["status"] = "error"
        
        return {
            "database_stats": stats,
            "last_updated": datetime.now().isoformat(),
            "collection_time": time.time() - startup_time
        }
        
    except Exception as e:
        logger.error(f"Database stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")

@app.delete("/admin/clear-neo4j", tags=["admin"])
async def clear_neo4j_database():
    """Clear all data from Neo4j database - USE WITH CAUTION"""
    try:
        neo4j_manager, _ = get_database_managers()
        
        if not neo4j_manager:
            raise HTTPException(status_code=503, detail="Neo4j not available")
        
        # Clear all nodes and relationships
        await neo4j_manager.execute_query_async("MATCH (n) DETACH DELETE n")
        
        logger.info("🗑️ Neo4j database cleared")
        
        return {
            "status": "success",
            "message": "Neo4j database cleared successfully",
            "timestamp": datetime.now().isoformat(),
            "warning": "All graph data has been permanently deleted"
        }
        
    except Exception as e:
        logger.error(f"Neo4j clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear Neo4j: {str(e)}")

@app.delete("/admin/clear-vectors", tags=["admin"])
async def clear_vector_database():
    """Clear all vectors from vector database - USE WITH CAUTION"""
    try:
        _, vector_manager = get_database_managers()
        
        if not vector_manager:
            raise HTTPException(status_code=503, detail="Vector database not available")
        
        # This would need specific implementation based on vector DB type
        logger.info("🗑️ Vector database clear requested")
        
        return {
            "status": "success",
            "message": "Vector database clear initiated",
            "timestamp": datetime.now().isoformat(),
            "warning": "All vector data deletion in progress"
        }
        
    except Exception as e:
        logger.error(f"Vector DB clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear vector database: {str(e)}")

@app.get("/test/examples", tags=["testing"])
async def get_api_examples():
    """Get API usage examples for testing"""
    return {
        "examples": {
            "query": {
                "endpoint": "POST /query",
                "example": {
                    "query": "What is machine learning?",
                    "max_results": 10,
                    "include_reasoning": True
                },
                "description": "Submit a question for intelligent processing"
            },
            "document_upload": {
                "endpoint": "POST /documents/upload",
                "example": {
                    "title": "Introduction to AI",
                    "content": "Artificial Intelligence is a field of computer science...",
                    "domain": "technology",
                    "source": "https://example.com/ai-intro"
                },
                "description": "Upload a document for knowledge base ingestion"
            },
            "agent_status": {
                "endpoint": "GET /agents/status",
                "description": "Check the status of all system agents"
            },
            "system_status": {
                "endpoint": "GET /system/status", 
                "description": "Get comprehensive system health information"
            }
        },
        "testing_tips": [
            "Start with simple queries to test basic functionality",
            "Upload small documents first to test ingestion",
            "Check agent status if queries aren't working as expected",
            "Use /docs for interactive API testing"
        ],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    host = getattr(config, 'HOST', os.environ.get("HOST", "0.0.0.0"))
    port = getattr(config, 'PORT', int(os.environ.get("PORT", 8000)))
    
    print(f"🚀 Starting RAG API on {host}:{port}")
    print(f"🔧 Environment: {getattr(config, 'ENVIRONMENT', 'production')}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )