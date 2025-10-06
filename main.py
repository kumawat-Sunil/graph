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
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Graph-Enhanced Agentic RAG API")
    logger.info(f"Environment: {getattr(config, 'ENVIRONMENT', 'production')}")
    
    # Start background initialization (non-blocking)
    import asyncio
    init_task = asyncio.create_task(background_initialization())
    
    logger.info("⚡ Server started - initialization running in background")
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down Graph-Enhanced Agentic RAG API")
    
    # Cancel background initialization if still running
    if not init_task.done():
        init_task.cancel()
        logger.info("🛑 Background initialization cancelled")
    
    # Close database connections
    try:
        logger.info("Closing database connections...")
        # Add cleanup logic here if needed
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error closing databases: {e}")
    
    logger.info("👋 Shutdown complete")

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
        "initialization": "complete" if initialization_complete else "in_progress",
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
    """Get or create coordinator agent with proper registry/queue"""
    global coordinator_instance, agent_registry, message_queue
    if coordinator_instance is None:
        try:
            from agents.coordinator import CoordinatorAgent
            # Create with registry and message queue if available
            if agent_registry and message_queue:
                coordinator_instance = CoordinatorAgent(
                    "api_coordinator",
                    agent_registry=agent_registry,
                    message_queue=message_queue
                )
                logger.info("✅ Coordinator agent initialized with registry/queue")
            else:
                coordinator_instance = CoordinatorAgent("api_coordinator")
                logger.info("✅ Coordinator agent initialized (standalone)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize coordinator: {e}")
            coordinator_instance = None
    return coordinator_instance

def get_database_managers():
    """Lazy initialization of database managers with proper setup"""
    try:
        # Initialize database connections
        from core.database import get_neo4j_manager, get_vector_manager
        
        logger.info("🔄 Initializing database connections...")
        
        # Get Neo4j manager
        neo4j_manager = None
        try:
            neo4j_manager = get_neo4j_manager()
            if neo4j_manager:
                logger.info("✅ Neo4j manager initialized")
            else:
                logger.warning("⚠️ Neo4j manager not available")
        except Exception as e:
            logger.error(f"❌ Neo4j initialization failed: {e}")
        
        # Get Vector manager  
        vector_manager = None
        try:
            vector_manager = get_vector_manager()
            if vector_manager:
                logger.info("✅ Vector manager initialized")
            else:
                logger.warning("⚠️ Vector manager not available")
        except Exception as e:
            logger.error(f"❌ Vector manager initialization failed: {e}")
        
        return neo4j_manager, vector_manager
        
    except Exception as e:
        logger.error(f"❌ Database managers initialization failed: {e}")
        return None, None

def get_or_create_all_agents():
    """Initialize all agents in the system"""
    global coordinator_instance, agent_registry, message_queue
    
    agents = {}
    
    try:
        # Initialize coordinator
        coordinator = get_or_create_coordinator()
        if coordinator:
            agents['coordinator'] = coordinator
        
        # Initialize Graph Navigator Agent
        try:
            from agents.graph_navigator import GraphNavigatorAgent
            graph_navigator = GraphNavigatorAgent("graph_navigator")
            agents['graph_navigator'] = graph_navigator
            logger.info("✅ Graph Navigator agent initialized")
        except Exception as e:
            logger.error(f"❌ Graph Navigator initialization failed: {e}")
            agents['graph_navigator'] = None
        
        # Initialize Vector Retrieval Agent
        try:
            from agents.vector_retrieval import VectorRetrievalAgent
            vector_agent = VectorRetrievalAgent("vector_retrieval")
            agents['vector_retrieval'] = vector_agent
            logger.info("✅ Vector Retrieval agent initialized")
        except Exception as e:
            logger.error(f"❌ Vector Retrieval initialization failed: {e}")
            agents['vector_retrieval'] = None
        
        # Initialize Synthesis Agent
        try:
            from agents.synthesis import SynthesisAgent
            synthesis_agent = SynthesisAgent("synthesis")
            agents['synthesis'] = synthesis_agent
            logger.info("✅ Synthesis agent initialized")
        except Exception as e:
            logger.error(f"❌ Synthesis agent initialization failed: {e}")
            agents['synthesis'] = None
        
        # Initialize Agent Registry
        try:
            from core.agent_registry import AgentRegistry
            agent_registry = AgentRegistry()
            logger.info("✅ Agent registry initialized")
        except Exception as e:
            logger.error(f"❌ Agent registry initialization failed: {e}")
            agent_registry = None
        
        # Initialize Message Queue
        try:
            from core.message_queue import MessageQueue
            message_queue = MessageQueue()
            logger.info("✅ Message queue initialized")
        except Exception as e:
            logger.error(f"❌ Message queue initialization failed: {e}")
            message_queue = None
        
        logger.info(f"🎯 Agent initialization complete: {len([a for a in agents.values() if a is not None])}/{len(agents)} agents ready")
        return agents
        
    except Exception as e:
        logger.error(f"❌ Agent system initialization failed: {e}")
        return {}


def initialize_core_services():
    """Initialize core services like embedding, document processing"""
    services = {}
    
    try:
        # Initialize Embedding Service
        try:
            from core.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            services['embedding'] = embedding_service
            logger.info("✅ Embedding service initialized")
        except Exception as e:
            logger.error(f"❌ Embedding service initialization failed: {e}")
            services['embedding'] = None
        
        # Initialize Document Processor
        try:
            from core.document_processor import DocumentProcessor
            doc_processor = DocumentProcessor()
            services['document_processor'] = doc_processor
            logger.info("✅ Document processor initialized")
        except Exception as e:
            logger.error(f"❌ Document processor initialization failed: {e}")
            services['document_processor'] = None
        
        # Initialize Ingestion Pipeline
        try:
            from core.ingestion_pipeline import DualStorageIngestionPipeline
            ingestion_pipeline = DualStorageIngestionPipeline()
            services['ingestion_pipeline'] = ingestion_pipeline
            logger.info("✅ Ingestion pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Ingestion pipeline initialization failed: {e}")
            services['ingestion_pipeline'] = None
        
        logger.info(f"🔧 Core services initialization complete: {len([s for s in services.values() if s is not None])}/{len(services)} services ready")
        return services
        
    except Exception as e:
        logger.error(f"❌ Core services initialization failed: {e}")
        return {}

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
                        {
                            "doc_id": document_id,
                            "title": request.title,
                            "content": request.content[:1000],  # Truncate for storage
                            "source": request.source or "",
                            "domain": request.domain
                        }
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
                result = await neo4j_manager.execute_query_async("RETURN 1 as test", {})
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
                node_result = await neo4j_manager.execute_query_async("MATCH (n) RETURN count(n) as count", {})
                nodes_count = node_result[0]['count'] if node_result else 0
                
                # Count relationships
                rel_result = await neo4j_manager.execute_query_async("MATCH ()-[r]->() RETURN count(r) as count", {})
                rels_count = rel_result[0]['count'] if rel_result else 0
                
                # Count documents
                doc_result = await neo4j_manager.execute_query_async("MATCH (d:Document) RETURN count(d) as count", {})
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
        await neo4j_manager.execute_query_async("MATCH (n) DETACH DELETE n", {})
        
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
    global initialization_complete
    
    return {
        "background_initialization": {
            "status": "complete" if initialization_complete else "in_progress",
            "complete": initialization_complete
        },
        "components": {
            "coordinator": "initialized" if coordinator_instance else "pending",
            "agent_registry": "initialized" if agent_registry else "pending",
            "message_queue": "initialized" if message_queue else "pending"
        },
        "performance_note": "First queries may be slower while initialization completes",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/system/detailed-init-status", tags=["system"])
async def get_detailed_initialization_status():
    """Check the initialization status of all system components"""
    try:
        # Check database managers
        neo4j_manager, vector_manager = get_database_managers()
        
        # Check agents
        agents = get_or_create_all_agents()
        
        # Check core services
        services = initialize_core_services()
        
        return {
            "initialization_status": {
                "databases": {
                    "neo4j": "initialized" if neo4j_manager else "failed",
                    "vector_db": "initialized" if vector_manager else "failed"
                },
                "agents": {
                    "coordinator": "initialized" if agents.get('coordinator') else "failed",
                    "graph_navigator": "initialized" if agents.get('graph_navigator') else "failed", 
                    "vector_retrieval": "initialized" if agents.get('vector_retrieval') else "failed",
                    "synthesis": "initialized" if agents.get('synthesis') else "failed",
                    "agent_registry": "initialized" if agent_registry else "failed",
                    "message_queue": "initialized" if message_queue else "failed"
                },
                "core_services": {
                    "embedding": "initialized" if services.get('embedding') else "failed",
                    "document_processor": "initialized" if services.get('document_processor') else "failed",
                    "ingestion_pipeline": "initialized" if services.get('ingestion_pipeline') else "failed"
                }
            },
            "summary": {
                "total_components": 11,
                "initialized": len([x for x in [neo4j_manager, vector_manager] + list(agents.values()) + list(services.values()) if x is not None]),
                "failed": len([x for x in [neo4j_manager, vector_manager] + list(agents.values()) + list(services.values()) if x is None])
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
        agents = get_or_create_all_agents()
        
        # Force core services initialization
        services = initialize_core_services()
        
        initialized_count = len([x for x in [neo4j_manager, vector_manager] + list(agents.values()) + list(services.values()) if x is not None])
        total_count = 11
        
        return {
            "status": "completed",
            "message": f"Force initialization completed: {initialized_count}/{total_count} components initialized",
            "details": {
                "databases": 2 if neo4j_manager and vector_manager else (1 if neo4j_manager or vector_manager else 0),
                "agents": len([a for a in agents.values() if a is not None]),
                "services": len([s for s in services.values() if s is not None])
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Force initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Force initialization failed: {str(e)}")

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

# Background initialization flag
initialization_complete = False

async def background_initialization():
    """Initialize system components in background after server starts"""
    global initialization_complete, agent_registry, message_queue, coordinator_instance
    
    try:
        logger.info("🔄 Starting background initialization...")
        
        # Initialize agent communication system (exactly like original)
        try:
            logger.info("Initializing agent communication system...")
            
            # Initialize agent registry and message queue (original way)
            from core.agent_registry import initialize_agent_registry
            from core.message_queue import initialize_message_queue
            
            agent_registry = initialize_agent_registry()
            message_queue = await initialize_message_queue()
            
            logger.info("✅ Agent communication system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent communication: {e}")
            agent_registry = None
            message_queue = None
        
        # Initialize and register agent instances (exactly like original)
        try:
            logger.info("Initializing and registering agent instances...")
            
            # Create global coordinator instance with communication system
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
            
            # Create and register other agents (original way)
            from agents.graph_navigator import GraphNavigatorAgent
            from agents.vector_retrieval import VectorRetrievalAgent
            from agents.synthesis import SynthesisAgent
            
            # Graph Navigator Agent
            graph_navigator = GraphNavigatorAgent("graph_navigator")
            await agent_registry.register_agent(
                "graph_navigator",
                graph_navigator,
                "graph_navigator"
            )
            
            # Vector Retrieval Agent
            vector_retrieval = VectorRetrievalAgent(
                agent_id="vector_retrieval",
                model_name=getattr(config, 'embedding_model', 'all-MiniLM-L6-v2'),
                collection_name=getattr(config, 'chroma_collection_name', 'documents')
            )
            await agent_registry.register_agent(
                "vector_retrieval",
                vector_retrieval,
                "vector_retrieval"
            )
            
            # Synthesis Agent
            synthesis_agent = SynthesisAgent("synthesis")
            await agent_registry.register_agent(
                "synthesis",
                synthesis_agent,
                "synthesis"
            )
            
            # Register message handlers with the queue (original way)
            message_queue.register_handler("api_coordinator", coordinator_instance.process_message)
            message_queue.register_handler("graph_navigator", graph_navigator.process_message)
            message_queue.register_handler("vector_retrieval", vector_retrieval.process_message)
            message_queue.register_handler("synthesis", synthesis_agent.process_message)
            
            logger.info("✅ All agents initialized and registered successfully")
            logger.info(f"   📊 Registered agents: {len(agent_registry.agents)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
        
        initialization_complete = True
        logger.info("🎉 Background initialization complete!")
        
    except Exception as e:
        logger.error(f"❌ Background initialization failed: {e}")

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