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

# Try to import core modules, fallback to simple config if not available
try:
    from core.config import get_config
    config = get_config()
except ImportError:
    # Fallback to simple config
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
async def root():
    """Root endpoint"""
    return {
        "message": "RAG API is running!",
        "status": "healthy",
        "version": "1.0.0",
        "environment": getattr(config, 'ENVIRONMENT', 'production'),
        "features": ["basic_api", "config_system", "core_modules"]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is working perfectly",
        "environment": getattr(config, 'ENVIRONMENT', 'production'),
        "config_loaded": True,
        "databases": {
            "neo4j": "configured" if getattr(config, 'NEO4J_URI', None) else "not_configured",
            "pinecone": "configured" if getattr(config, 'PINECONE_API_KEY', None) else "not_configured"
        }
    }

@app.post("/test")
async def test_endpoint(data: dict):
    """Test POST endpoint"""
    return {
        "received": data,
        "message": "Test successful"
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