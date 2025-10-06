#!/usr/bin/env python3
"""
Start the Graph-Enhanced Agentic RAG API server from src directory.
"""
import uvicorn
import sys
import os
import logging

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # Get configuration from environment
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    print("🚀 Starting Graph-Enhanced Agentic RAG API from src...")
    print(f"🌐 BINDING TO: {host}:{port}")
    
    # Run the API
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )