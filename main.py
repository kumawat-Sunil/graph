"""
RAG API - Gradually building up features
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
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
        "environment": config.ENVIRONMENT,
        "features": ["basic_api", "config_system"]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is working perfectly",
        "environment": config.ENVIRONMENT,
        "config_loaded": True,
        "databases": {
            "neo4j": "configured" if config.NEO4J_URI else "not_configured",
            "pinecone": "configured" if config.PINECONE_API_KEY else "not_configured"
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
    print(f"🚀 Starting RAG API on {config.HOST}:{config.PORT}")
    print(f"🔧 Environment: {config.ENVIRONMENT}")
    
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info"
    )