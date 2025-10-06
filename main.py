"""
Graph-Enhanced Agentic RAG API
Main FastAPI application
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
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
    title="RAG API - Minimal",
    description="Simple FastAPI deployment test",
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

@app.get("/")
@app.head("/")
async def root():
    """Root endpoint - supports both GET and HEAD for health checks"""
    return {
        "message": "RAG API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "environment": getattr(config, 'ENVIRONMENT', 'production'),
        "features": ["basic_api", "config_system", "core_modules"],
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

@app.get("/status")
async def api_status():
    """API status and information endpoint"""
    return {
        "api_name": "Graph-Enhanced Agentic RAG API",
        "version": "1.0.0",
        "status": "operational",
        "uptime_seconds": time.time() - startup_time,
        "endpoints": {
            "root": "/",
            "health": "/health", 
            "test": "/test (GET/POST)",
            "status": "/status",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "next_features": ["query_processing", "document_upload", "agent_system"],
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