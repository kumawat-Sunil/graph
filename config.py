"""
Simple configuration for the RAG API
"""
import os
from typing import Optional

class Config:
    """Basic configuration class"""
    
    # API Settings
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", 8000))
    ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "production")
    
    # Database URLs (will be set via environment variables in Render)
    NEO4J_URI: Optional[str] = os.environ.get("NEO4J_URI")
    PINECONE_API_KEY: Optional[str] = os.environ.get("PINECONE_API_KEY")
    GEMINI_API_KEY: Optional[str] = os.environ.get("GEMINI_API_KEY")

# Global config instance
config = Config()