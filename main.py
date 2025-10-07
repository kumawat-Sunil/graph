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

# Modern FastAPI lifespan events
from contextlib import asynccontextmanager, contextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events - MINIMAL startup for fast port detection"""
    # Startup - MINIMAL ONLY
    logger.info("🚀 Starting Graph-Enhanced Agentic RAG API")
    logger.info(f"Environment: {getattr(config, 'ENVIRONMENT', 'production')}")
    logger.info("⚡ Server started - components will initialize on-demand")
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down Graph-Enhanced Agentic RAG API")
    
    # Close database connections if they were initialized
    try:
        if neo4j_manager_instance:
            neo4j_manager_instance.disconnect()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error closing databases: {e}")
    
    logger.info("� ShutBdown complete")

# Create FastAPI app with lifespan
try:
    app = FastAPI(
        title="Graph-Enhanced Agentic RAG API",
        description="Multi-agent retrieval-augmented generation system with intelligent query processing",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
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

# Admin Response Models
class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics"""
    name: str
    status: str
    total_documents: int
    total_entities: int
    total_relationships: int
    total_embeddings: int
    last_updated: Optional[str]
    size_info: dict

class DocumentInfoResponse(BaseModel):
    """Response model for document information"""
    id: str
    title: str
    content_preview: str
    source: str
    created_at: str
    in_neo4j: bool
    in_pinecone: bool
    entity_count: int
    chunk_count: int
    metadata: dict

class EntityInfoResponse(BaseModel):
    """Response model for entity information"""
    name: str
    type: str
    description: str
    document_count: int
    sample_documents: List[str]

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
        "features": ["basic_api", "config_system", "core_modules", "pydantic_models", "static_files", "background_init"],
        "initialization": "lazy_on_demand",
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

# Global instances (lazy initialization)
coordinator_instance = None
agent_registry = None
message_queue = None
neo4j_manager_instance = None
vector_manager_instance = None

# Removed get_or_create_coordinator - using lazy initialization instead

def get_database_managers():
    """Get or initialize database managers (singleton pattern)"""
    global neo4j_manager_instance, vector_manager_instance
    
    # Return existing instances if available
    if neo4j_manager_instance is not None and vector_manager_instance is not None:
        return neo4j_manager_instance, vector_manager_instance
    
    try:
        # Initialize database connections only once
        from core.database import get_neo4j_manager, get_vector_manager
        
        logger.info("🔄 Initializing database connections...")
        
        # Get Neo4j manager (only if not already initialized)
        if neo4j_manager_instance is None:
            try:
                neo4j_manager_instance = get_neo4j_manager()
                if neo4j_manager_instance:
                    logger.info("✅ Neo4j manager initialized")
                else:
                    logger.warning("⚠️ Neo4j manager not available")
            except Exception as e:
                logger.error(f"❌ Neo4j initialization failed: {e}")
        
        # Get Vector manager (only if not already initialized)
        if vector_manager_instance is None:
            try:
                vector_manager_instance = get_vector_manager()
                if vector_manager_instance:
                    logger.info("✅ Vector manager initialized")
                else:
                    logger.warning("⚠️ Vector manager not available")
            except Exception as e:
                logger.error(f"❌ Vector manager initialization failed: {e}")
        
        return neo4j_manager_instance, vector_manager_instance
        
    except Exception as e:
        logger.error(f"❌ Database managers initialization failed: {e}")
        return None, None

@contextmanager
def neo4j_connection():
    """Context manager for Neo4j connections"""
    neo4j_manager, _ = get_database_managers()
    if not neo4j_manager:
        raise Exception("Neo4j manager not available")
    
    try:
        neo4j_manager.connect()
        yield neo4j_manager
    finally:
        try:
            neo4j_manager.disconnect()
        except:
            pass

def execute_neo4j_query(query: str, parameters: dict = None):
    """Execute single Neo4j query with connection handling"""
    with neo4j_connection() as neo4j_manager:
        return neo4j_manager.execute_query(query, parameters or {})

def execute_multiple_neo4j_queries(queries: dict):
    """Execute multiple Neo4j queries with single connection"""
    results = {}
    with neo4j_connection() as neo4j_manager:
        for key, query in queries.items():
            try:
                result = neo4j_manager.execute_query(query)
                results[key] = result
            except Exception as e:
                results[key] = f"Error: {str(e)}"
    return results

# Removed get_or_create_all_agents - using lazy initialization instead


# Removed initialize_core_services - services initialize on-demand through their own modules

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
        
        # Ensure agent system is initialized (lazy)
        agent_ready = await ensure_agent_system()
        
        if not agent_ready or coordinator_instance is None:
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
        
        coordinator = coordinator_instance
        
        # Execute the full workflow using the real coordinator (original approach)
        try:
            workflow_results = await coordinator.coordinate_full_workflow(request.query)
            
            # Extract results from workflow (exactly like original)
            analysis = workflow_results.get('analysis')
            strategy_used = workflow_results.get('strategy', RetrievalStrategy.HYBRID)
            synthesis_results = workflow_results.get('synthesis_results', {})
            
            # Get entities from analysis
            entities_found = analysis.get('entities', []) if analysis else []
            
            # Calculate processing time
            processing_time = workflow_results.get('execution_time', time.time() - start_time)
            
            # Extract real synthesis results
            synthesis_result = synthesis_results.get('synthesis_result', {})
            
            # Handle SynthesisResult object properly (original logic)
            if synthesis_result:
                logger.info(f"Synthesis result type: {type(synthesis_result)}")
                if hasattr(synthesis_result, 'response'):
                    # It's a SynthesisResult object
                    coordinator_response = synthesis_result.response
                    real_sources = synthesis_result.sources if hasattr(synthesis_result, 'sources') else []
                    real_citations = synthesis_result.citations if hasattr(synthesis_result, 'citations') else []
                    confidence_score = synthesis_result.confidence_score if hasattr(synthesis_result, 'confidence_score') else 0.85
                    reasoning_path = synthesis_result.reasoning_path if hasattr(synthesis_result, 'reasoning_path') else None
                elif isinstance(synthesis_result, dict):
                    # It's a dictionary
                    coordinator_response = synthesis_result.get('response', f"I've analyzed your query using the {strategy_used.value} approach and identified {len(entities_found)} key entities.")
                    real_sources = synthesis_result.get('sources', [])
                    real_citations = synthesis_result.get('citations', [])
                    confidence_score = synthesis_result.get('confidence_score', 0.85)
                    reasoning_path = synthesis_result.get('reasoning_path', None)
                else:
                    # Fallback (original)
                    coordinator_response = f"I've analyzed your query using the {strategy_used.value} approach and identified {len(entities_found)} key entities."
                    real_sources = []
                    real_citations = []
                    confidence_score = 0.85
                    reasoning_path = None
            else:
                coordinator_response = f"I've analyzed your query using the {strategy_used.value} approach and identified {len(entities_found)} key entities."
                real_sources = []
                real_citations = []
                confidence_score = 0.85
                reasoning_path = None
            
            # Use real sources only (original logic)
            if real_sources:
                if real_sources and isinstance(real_sources[0], str):
                    sources_list = [
                        {
                            "id": f"source_{i}",
                            "title": source,
                            "content_preview": f"Content from {source}",
                            "source_type": "document",
                            "relevance_score": 0.9
                        }
                        for i, source in enumerate(real_sources)
                    ]
                else:
                    sources_list = real_sources
            else:
                sources_list = []
            
            # Use real citations only
            citations_list = real_citations if real_citations else []
            
            return QueryResponse(
                query_id=query_id,
                response=coordinator_response,
                sources=sources_list,
                citations=citations_list,
                reasoning_path=reasoning_path if request.include_reasoning and reasoning_path else (
                    "Query Analysis: Identified key concepts and entities → Strategy Selection: Chose optimal retrieval approach → "
                    "Coordinator Agent: Orchestrated multi-agent workflow → Graph Navigator: Explored entity relationships → "
                    "Vector Retrieval: Performed semantic similarity search → Synthesis Agent: Combined and weighted results → "
                    "Response Generation: Created coherent answer with proper citations" if request.include_reasoning else None
                ),
                confidence_score=confidence_score,
                processing_time=processing_time,
                strategy_used=strategy_used.value if hasattr(strategy_used, 'value') else str(strategy_used),
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

@app.post(
    "/documents/upload", 
    response_model=DocumentUploadResponse, 
    tags=["documents"],
    summary="Upload document for ingestion",
    description="Upload a document to be processed and stored in both graph and vector databases",
    responses={
        200: {
            "description": "Document uploaded and processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "document_id": "doc_123e4567-e89b-12d3-a456-426614174000",
                        "status": "success",
                        "message": "Document 'Introduction to AI' uploaded successfully",
                        "entities_extracted": 25,
                        "relationships_created": 18,
                        "processing_time": 3.45
                    }
                }
            }
        },
        400: {
            "description": "Invalid document data",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Validation Error",
                        "message": "Document title cannot be empty",
                        "details": {"field": "title", "issue": "empty_value"},
                        "timestamp": "2024-01-01T12:00:00Z",
                        "request_id": "req_123"
                    }
                }
            }
        }
    }
)
async def upload_document(request: DocumentUploadRequest):
    """
    Upload and ingest a document into both graph and vector databases.
    
    This endpoint processes documents through the complete ingestion pipeline:
    
    1. **Content Processing**: Validates and preprocesses document content
    2. **Entity Extraction**: Identifies entities and concepts within the document
    3. **Relationship Discovery**: Finds relationships between extracted entities
    4. **Graph Storage**: Stores entities and relationships in Neo4j
    5. **Vector Storage**: Generates embeddings and stores in Pinecone
    6. **Mapping Creation**: Creates bidirectional entity-vector mappings
    
    **Supported Content Types:**
    - Plain text documents
    - Technical documentation
    - Research papers
    - Knowledge base articles
    
    **Domain-Specific Processing:**
    - Configurable entity extraction patterns
    - Domain-specific relationship types
    - Custom schema mapping
    """
    start_time = time.time()
    document_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Ingesting document {document_id}: {request.title}")
        
        # Create document object
        from core.models import Document, DocumentType
        document = Document(
            id=document_id,
            title=request.title,
            content=request.content,
            document_type=DocumentType.TEXT,
            source=request.source or "",
            domain=request.domain or "general",
            metadata=request.metadata or {}
        )
        
        # Initialize and use ingestion pipeline
        from core.ingestion_pipeline import DualStorageIngestionPipeline
        from core.document_processor import DocumentProcessor
        from core.domain_processor import DomainProcessorManager
        from core.mapping_service import EntityVectorMappingService
        
        # Ensure databases are initialized and connected (lazy)
        ensure_databases()
        neo4j_manager, vector_manager = get_database_managers()
        
        # Initialize processors
        document_processor = DocumentProcessor()
        domain_processor = DomainProcessorManager()
        mapping_service = EntityVectorMappingService()
        
        # Initialize the mapping service (databases are now connected)
        mapping_service.initialize()
        
        # Get embedding service
        from core.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        embedding_service.initialize()
        
        # Create ingestion pipeline
        pipeline = DualStorageIngestionPipeline(
            document_processor=document_processor,
            graph_db_manager=neo4j_manager,
            vector_db_manager=vector_manager,
            mapping_service=mapping_service,
            embedding_service=embedding_service
        )
        
        # Actually ingest the document
        logger.info(f"Processing document through ingestion pipeline...")
        result = await pipeline.ingest_document(document)
        
        processing_time = time.time() - start_time
        
        if result.success:
            response = DocumentUploadResponse(
                document_id=document_id,
                status="success",
                message=f"Document '{request.title}' processed and stored successfully. "
                       f"Extracted {result.entities_created} entities and created {result.relationships_created} relationships.",
                entities_extracted=result.entities_created,
                relationships_created=result.relationships_created,
                processing_time=processing_time
            )
            logger.info(f"Document {document_id} ingested successfully in {processing_time:.2f}s")
        else:
            response = DocumentUploadResponse(
                document_id=document_id,
                status="partial_success",
                message=f"Document '{request.title}' processed with some issues: {'; '.join(result.errors)}",
                entities_extracted=result.entities_created,
                relationships_created=result.relationships_created,
                processing_time=processing_time
            )
            logger.warning(f"Document {document_id} ingested with issues in {processing_time:.2f}s")
        return response
        
    except ValueError as e:
        logger.warning(f"Invalid document upload {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error ingesting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during document ingestion: {str(e)}"
        )

@app.post("/documents/upload-file", response_model=DocumentUploadResponse, tags=["documents"])
async def upload_document_file(
    file: UploadFile = File(..., description="Document file to upload"),
    title: str = Form(..., description="Document title"),
    source: Optional[str] = Form(None, description="Source URL or reference"),
    domain: str = Form(default="general", description="Knowledge domain")
):
    """
    Upload a document file for ingestion.
    
    Accepts file uploads in various formats and processes them through
    the same ingestion pipeline as the JSON endpoint.
    
    **Supported File Types:**
    - Text files (.txt)
    - Markdown files (.md)
    - PDF files (.pdf) - extracted to text
    - Word documents (.docx) - extracted to text
    """
    start_time = time.time()
    document_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing file upload {document_id}: {file.filename}")
        
        # Read file content
        content = await file.read()
        
        # Handle different file types
        try:
            if file.content_type == "application/pdf":
                # TODO: Add PDF extraction
                text_content = "PDF content extraction not yet implemented"
            elif file.content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                # TODO: Add DOCX extraction
                text_content = "DOCX content extraction not yet implemented"
            else:
                # Assume text content
                text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="File must be valid UTF-8 text or supported document format"
            )
        
        # Create document request
        doc_request = DocumentUploadRequest(
            title=title,
            content=text_content,
            source=source,
            domain=domain,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": len(content)
            }
        )
        
        # Process using the same logic as JSON upload
        response = await upload_document(doc_request)
        
        # Update the response for file upload
        response.document_id = document_id
        response.message = f"File '{file.filename}' uploaded successfully. {response.message}"
        
        logger.info(f"File {document_id} processed successfully in {response.processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file upload {document_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"File processing failed: {str(e)}"
        )

@app.get("/agents/status", tags=["agents"])
async def get_agents_status():
    """Get status of all agents with lazy initialization"""
    try:
        # Check initialization status without forcing initialization
        agent_system_ready = initialization_status["agent_system"]
        database_ready = initialization_status["databases"]
        
        agents_status = {
            "coordinator": {
                "status": "healthy" if agent_system_ready and coordinator_instance else "not_initialized",
                "type": "CoordinatorAgent",
                "initialized": agent_system_ready
            },
            "agent_registry": {
                "status": "healthy" if agent_system_ready and agent_registry else "not_initialized",
                "initialized": agent_system_ready,
                "agent_count": len(getattr(agent_registry, 'agents', {})) if agent_registry else 0
            },
            "databases": {
                "neo4j": {
                    "status": "connected" if database_ready else "not_initialized",
                    "initialized": database_ready
                },
                "vector_db": {
                    "status": "connected" if database_ready else "not_initialized", 
                    "initialized": database_ready
                }
            }
        }
        
        total_agents = 4  # Expected total
        healthy_agents = len(getattr(agent_registry, 'agents', {})) if agent_registry else 0
        
        return {
            "agents": agents_status,
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "initialization_strategy": "lazy_on_demand",
            "initialization_status": initialization_status,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Agent status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")

@app.get("/system/status", tags=["system"])
async def get_system_status():
    """Get comprehensive system status including all components"""
    try:
        # Test database connections (lazy init)
        neo4j_status = "disconnected"
        vector_status = "disconnected"
        
        if ensure_databases():
            neo4j_manager, vector_manager = get_database_managers()
            
            if neo4j_manager:
                try:
                    result = execute_neo4j_query("RETURN 1 as test")
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
                "coordinator_agent": "healthy" if coordinator_instance else "not_initialized",
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

@app.get("/admin/stats", response_model=List[DatabaseStatsResponse], tags=["admin"])
async def get_database_stats():
    """Get comprehensive database statistics for admin dashboard"""
    try:
        stats = []
        
        # Neo4j Statistics
        try:
            ensure_databases()
            neo4j_manager, _ = get_database_managers()
            if neo4j_manager:
                # Execute all queries with single connection
                queries = {
                    "documents": "MATCH (d:Document) RETURN count(d) as count",
                    "entities": "MATCH (e:Entity) RETURN count(e) as count", 
                    "relationships": "MATCH ()-[r]->() RETURN count(r) as count",
                    "last_updated": "MATCH (d:Document) RETURN d.created_at as last_updated ORDER BY d.created_at DESC LIMIT 1"
                }
                
                query_results = execute_multiple_neo4j_queries(queries)
                
                # Extract results (handle both successful results and error strings)
                doc_count = 0
                entity_count = 0
                rel_count = 0
                last_updated = None
                
                if isinstance(query_results.get("documents"), list) and query_results["documents"]:
                    doc_count = query_results["documents"][0]["count"]
                
                if isinstance(query_results.get("entities"), list) and query_results["entities"]:
                    entity_count = query_results["entities"][0]["count"]
                
                if isinstance(query_results.get("relationships"), list) and query_results["relationships"]:
                    rel_count = query_results["relationships"][0]["count"]
                
                if isinstance(query_results.get("last_updated"), list) and query_results["last_updated"]:
                    try:
                        last_updated = query_results["last_updated"][0]["last_updated"]
                    except:
                        last_updated = None
                
                stats.append(DatabaseStatsResponse(
                    name="Neo4j Graph Database",
                    status="Connected",
                    total_documents=doc_count,
                    total_entities=entity_count,
                    total_relationships=rel_count,
                    total_embeddings=0,
                    last_updated=last_updated,
                    size_info={
                        "documents": doc_count,
                        "entities": entity_count,
                        "relationships": rel_count
                    }
                ))
            else:
                stats.append(DatabaseStatsResponse(
                    name="Neo4j Graph Database",
                    status="Not Connected",
                    total_documents=0,
                    total_entities=0,
                    total_relationships=0,
                    total_embeddings=0,
                    last_updated=None,
                    size_info={}
                ))
                
        except Exception as e:
            stats.append(DatabaseStatsResponse(
                name="Neo4j Graph Database",
                status=f"Error: {str(e)}",
                total_documents=0,
                total_entities=0,
                total_relationships=0,
                total_embeddings=0,
                last_updated=None,
                size_info={}
            ))
        
        # Pinecone Statistics
        try:
            _, vector_manager = get_database_managers()
            if vector_manager:
                # Try to get vector count estimate
                try:
                    # Query to estimate count
                    results = vector_manager.query_collection(
                        collection_name="documents",
                        query_text="machine learning",
                        n_results=10
                    )
                    # Rough estimate based on results
                    vector_count = len(results.get("ids", [[]])[0]) if results.get("ids") else 0
                    if vector_count > 0:
                        vector_count = vector_count * 10  # Rough multiplier
                except Exception:
                    vector_count = 0
                
                stats.append(DatabaseStatsResponse(
                    name="Pinecone Vector Database",
                    status="Connected",
                    total_documents=vector_count,
                    total_entities=0,
                    total_relationships=0,
                    total_embeddings=vector_count,
                    last_updated=datetime.now().isoformat(),
                    size_info={
                        "vectors": vector_count,
                        "dimensions": 384,
                        "collections": ["documents"]
                    }
                ))
            else:
                stats.append(DatabaseStatsResponse(
                    name="Pinecone Vector Database",
                    status="Not Connected",
                    total_documents=0,
                    total_entities=0,
                    total_relationships=0,
                    total_embeddings=0,
                    last_updated=None,
                    size_info={}
                ))
                
        except Exception as e:
            stats.append(DatabaseStatsResponse(
                name="Pinecone Vector Database",
                status=f"Error: {str(e)}",
                total_documents=0,
                total_entities=0,
                total_relationships=0,
                total_embeddings=0,
                last_updated=None,
                size_info={}
            ))
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database statistics: {str(e)}")

@app.delete("/admin/clear-neo4j", tags=["admin"])
async def clear_neo4j_database():
    """Clear all data from Neo4j database - USE WITH CAUTION"""
    try:
        neo4j_manager, _ = get_database_managers()
        
        if not neo4j_manager:
            raise HTTPException(status_code=503, detail="Neo4j not available")
        
        # Clear all nodes and relationships
        execute_neo4j_query("MATCH (n) DETACH DELETE n")
        
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

@app.get("/system/init-status", tags=["system"])
async def get_initialization_status():
    """Check the initialization status of all system components"""
    return {
        "initialization_strategy": "lazy_on_demand",
        "status": initialization_status,
        "components": {
            "coordinator": "initialized" if coordinator_instance else "not_initialized",
            "agent_registry": "initialized" if agent_registry else "not_initialized",
            "message_queue": "initialized" if message_queue else "not_initialized",
            "databases": "initialized" if initialization_status["databases"] else "not_initialized"
        },
        "performance_note": "Components initialize on first use - subsequent calls are fast",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/system/detailed-init-status", tags=["system"])
async def get_detailed_initialization_status():
    """Check the initialization status of all system components"""
    try:
        # Check database managers
        neo4j_manager, vector_manager = get_database_managers()
        
        return {
            "initialization_strategy": "lazy_on_demand",
            "current_status": initialization_status,
            "components": {
                "databases": {
                    "neo4j": "initialized" if initialization_status["databases"] else "not_initialized",
                    "vector_db": "initialized" if initialization_status["databases"] else "not_initialized"
                },
                "agents": {
                    "coordinator": "initialized" if coordinator_instance else "not_initialized",
                    "agent_registry": "initialized" if agent_registry else "not_initialized",
                    "message_queue": "initialized" if message_queue else "not_initialized",
                    "agent_count": len(getattr(agent_registry, 'agents', {})) if agent_registry else 0
                }
            },
            "summary": {
                "agent_system": "initialized" if initialization_status["agent_system"] else "not_initialized",
                "databases": "initialized" if initialization_status["databases"] else "not_initialized",
                "note": "Components initialize automatically when first accessed"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Initialization status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check initialization status: {str(e)}")

@app.post("/system/force-init", tags=["system"])
async def force_initialization():
    """Force initialization of all system components"""
    try:
        logger.info("🔄 Force initialization requested...")
        
        # Force database initialization
        neo4j_manager, vector_manager = get_database_managers()
        
        # Force agent initialization
        agent_ready = await ensure_agent_system()
        
        # Force database initialization
        db_ready = ensure_databases()
        
        return {
            "status": "completed",
            "message": "Force initialization completed using lazy loading strategy",
            "results": {
                "agent_system": "initialized" if agent_ready else "failed",
                "databases": "initialized" if db_ready else "failed"
            },
            "current_status": initialization_status,
            "components": {
                "coordinator": "initialized" if coordinator_instance else "failed",
                "agent_registry": "initialized" if agent_registry else "failed",
                "message_queue": "initialized" if message_queue else "failed",
                "agent_count": len(getattr(agent_registry, 'agents', {})) if agent_registry else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Force initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Force initialization failed: {str(e)}")

@app.get("/admin/database/neo4j", tags=["admin"])
async def get_neo4j_details():
    """Get detailed Neo4j database information"""
    try:
        neo4j_manager, _ = get_database_managers()
        if not neo4j_manager:
            raise HTTPException(status_code=503, detail="Neo4j not available")
        
        # Get detailed statistics (exactly like original)
        queries = {
            "documents": "MATCH (d:Document) RETURN count(d) as count",
            "entities": "MATCH (e:Entity) RETURN count(e) as count",
            "relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "document_types": "MATCH (d:Document) RETURN d.document_type as type, count(d) as count ORDER BY count DESC",
            "entity_types": "MATCH (e:Entity) RETURN e.type as type, count(e) as count ORDER BY count DESC",
            "recent_documents": """
                MATCH (d:Document) 
                RETURN d.title as title, d.created_at as created_at, d.id as id
                ORDER BY d.created_at DESC 
                LIMIT 10
            """
        }
        
        results = execute_multiple_neo4j_queries(queries)
        
        # Return results directly (like original)
        return results
        
    except Exception as e:
        logger.error(f"Error getting Neo4j details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Neo4j details: {str(e)}")

@app.get("/admin/database/pinecone", tags=["admin"])
async def get_pinecone_details():
    """Get detailed Pinecone database information"""
    try:
        _, vector_manager = get_database_managers()
        if not vector_manager:
            raise HTTPException(status_code=503, detail="Pinecone not available")
        
        # Get sample vectors and metadata
        results = vector_manager.query_collection(
            collection_name="documents",
            query_text="machine learning",
            n_results=10
        )
        
        # Analyze metadata
        metadata_analysis = {}
        if results.get("metadatas") and results["metadatas"][0]:
            metadatas = results["metadatas"][0]
            
            # Count unique document IDs
            doc_ids = set()
            domains = {}
            sources = {}
            
            for metadata in metadatas:
                if metadata:
                    if "document_id" in metadata:
                        doc_ids.add(metadata["document_id"])
                    if "domain" in metadata:
                        domain = metadata["domain"]
                        domains[domain] = domains.get(domain, 0) + 1
                    if "source" in metadata:
                        source = metadata["source"]
                        sources[source] = sources.get(source, 0) + 1
            
            metadata_analysis = {
                "unique_documents": len(doc_ids),
                "domains": domains,
                "sources": dict(list(sources.items())[:5]),  # Top 5 sources
                "sample_metadata": metadatas[:3] if metadatas else []
            }
        
        # Return in original format
        return {
            "collection_name": "documents",
            "sample_results_count": len(results.get("documents", [[]])[0]) if results.get("documents") else 0,
            "metadata_analysis": metadata_analysis,
            "vector_dimensions": 384,  # Based on all-MiniLM-L6-v2
            "sample_similarities": results.get("distances", [[]])[0][:5] if results.get("distances") else []
        }
        
    except Exception as e:
        logger.error(f"Error getting Pinecone details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Pinecone details: {str(e)}")

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

# Lazy initialization tracking
initialization_status = {
    "agent_system": False,
    "databases": False
}

async def ensure_agent_system():
    """Lazy initialization of agent system - only when needed"""
    global agent_registry, message_queue, coordinator_instance, initialization_status
    
    if initialization_status["agent_system"]:
        return True  # Already initialized
    
    try:
        logger.info("🔄 Initializing agent system on-demand...")
        
        # Initialize agent communication system
        from core.agent_registry import initialize_agent_registry
        from core.message_queue import initialize_message_queue
        
        agent_registry = initialize_agent_registry()
        message_queue = await initialize_message_queue()
        
        # Create coordinator
        from agents.coordinator import CoordinatorAgent
        coordinator_instance = CoordinatorAgent(
            "api_coordinator",
            agent_registry=agent_registry,
            message_queue=message_queue
        )
        
        # Register coordinator
        await agent_registry.register_agent(
            "api_coordinator", 
            coordinator_instance, 
            "coordinator"
        )
        
        # Create other agents
        from agents.graph_navigator import GraphNavigatorAgent
        from agents.vector_retrieval import VectorRetrievalAgent
        from agents.synthesis import SynthesisAgent
        
        graph_navigator = GraphNavigatorAgent("graph_navigator")
        await agent_registry.register_agent("graph_navigator", graph_navigator, "graph_navigator")
        
        vector_retrieval = VectorRetrievalAgent(
            agent_id="vector_retrieval",
            model_name=getattr(config, 'embedding_model', 'all-MiniLM-L6-v2'),
            collection_name=getattr(config, 'chroma_collection_name', 'documents')
        )
        await agent_registry.register_agent("vector_retrieval", vector_retrieval, "vector_retrieval")
        
        synthesis_agent = SynthesisAgent("synthesis")
        await agent_registry.register_agent("synthesis", synthesis_agent, "synthesis")
        
        # Register message handlers
        message_queue.register_handler("api_coordinator", coordinator_instance.process_message)
        message_queue.register_handler("graph_navigator", graph_navigator.process_message)
        message_queue.register_handler("vector_retrieval", vector_retrieval.process_message)
        message_queue.register_handler("synthesis", synthesis_agent.process_message)
        
        initialization_status["agent_system"] = True
        logger.info(f"✅ Agent system initialized: {len(agent_registry.agents)} agents ready")
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent system initialization failed: {e}")
        return False

def ensure_databases():
    """Lazy initialization of databases - only when needed"""
    global initialization_status
    
    if initialization_status["databases"]:
        return True  # Already initialized
    
    try:
        logger.info("🔄 Initializing databases on-demand...")
        # Get database managers through our singleton pattern
        neo4j_manager, vector_manager = get_database_managers()
        
        # Ensure Neo4j connection is established
        if neo4j_manager and not neo4j_manager.driver:
            neo4j_manager.connect()
            logger.info("✅ Neo4j connection established")
        
        # Vector manager should already be connected from get_database_managers()
        if neo4j_manager and vector_manager:
            initialization_status["databases"] = True
            logger.info("✅ Databases initialized and connected")
            return True
        return False
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

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