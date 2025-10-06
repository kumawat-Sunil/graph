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

# Create FastAPI app
app = FastAPI(
    title="Graph-Enhanced Agentic RAG API",
    description="Multi-agent retrieval-augmented generation system with intelligent query processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to import core interfaces for type hints
try:
    from core.interfaces import RetrievalStrategy
    logger.info("✅ Core interfaces loaded")
except ImportError:
    # Create fallback enum
    from enum import Enum
    class RetrievalStrategy(str, Enum):
        VECTOR_ONLY = "vector_only"
        GRAPH_ONLY = "graph_only"
        HYBRID = "hybrid"
    logger.info("⚠️ Using fallback RetrievalStrategy enum")

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
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Formatted citations")
    reasoning_path: Optional[str] = Field(None, description="Explanation of reasoning process")
    confidence_score: Optional[float] = Field(None, description="Confidence in the response", ge=0.0, le=1.0)
    processing_time: Optional[float] = Field(None, description="Time taken to process query in seconds")
    strategy_used: Optional[RetrievalStrategy] = Field(None, description="Retrieval strategy that was used")
    entities_found: Optional[List[str]] = Field(default_factory=list, description="Entities identified in query")

class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    title: str = Field(..., description="Document title", min_length=1, max_length=200)
    content: str = Field(..., description="Document content", min_length=1)
    source: Optional[str] = Field(None, description="Source URL or reference")
    domain: Optional[str] = Field(default="general", description="Knowledge domain")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

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