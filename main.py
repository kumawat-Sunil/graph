"""
Minimal FastAPI app for deployment testing
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

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
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "API is working perfectly"
    }

@app.post("/test")
async def test_endpoint(data: dict):
    """Test POST endpoint"""
    return {
        "received": data,
        "message": "Test successful"
    }

if __name__ == "__main__":
    # Get port from environment (Render provides this)
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting minimal RAG API on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )