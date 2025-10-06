"""
Configuration management for the Graph-Enhanced Agentic RAG system.
"""

import os
from typing import Optional
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", env="NEO4J_DATABASE")
    
    # Chroma Configuration
    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")
    chroma_port: int = Field(default=8000, env="CHROMA_PORT")
    chroma_collection_name: str = Field(default="documents", env="CHROMA_COLLECTION_NAME")
    chroma_persist_directory: Optional[str] = Field(default="./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    
    # Pinecone Configuration
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="rag-documents", env="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field(default="us-east-1", env="PINECONE_ENVIRONMENT")
    
    # Vector Database Type Selection
    vector_db_type: str = Field(default="chroma", env="VECTOR_DB_TYPE")  # "chroma" or "pinecone"
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class LLMConfig(BaseSettings):
    """Language model configuration settings."""
    
    # Gemini API Configuration
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash-exp", env="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, env="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=2048, env="GEMINI_MAX_TOKENS")
    
    # Embedding Model Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class AgentConfig(BaseSettings):
    """Agent-specific configuration settings."""
    
    # Coordinator Agent
    coordinator_timeout: int = Field(default=30, env="COORDINATOR_TIMEOUT")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    
    # Graph Navigator Agent
    graph_traversal_depth: int = Field(default=3, env="GRAPH_TRAVERSAL_DEPTH")
    max_entities_per_query: int = Field(default=50, env="MAX_ENTITIES_PER_QUERY")
    
    # Vector Retrieval Agent
    vector_search_k: int = Field(default=10, env="VECTOR_SEARCH_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Synthesis Agent
    max_context_length: int = Field(default=4000, env="MAX_CONTEXT_LENGTH")
    citation_format: str = Field(default="apa", env="CITATION_FORMAT")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class APIConfig(BaseSettings):
    """API server configuration settings."""
    
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    
    # CORS settings
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: list = Field(default=["*"], env="CORS_METHODS")
    cors_headers: list = Field(default=["*"], env="CORS_HEADERS")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class AppConfig(BaseSettings):
    """Main application configuration."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from environment
    )
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        return DatabaseConfig()
    
    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration."""
        return LLMConfig()
    
    @property
    def agents(self) -> AgentConfig:
        """Get agents configuration."""
        return AgentConfig()
    
    @property
    def api(self) -> APIConfig:
        """Get API configuration."""
        return APIConfig()


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables and .env file."""
    return AppConfig()