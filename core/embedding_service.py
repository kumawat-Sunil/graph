"""
Embedding service for generating vector embeddings from text.

This module provides a simple embedding service that can be used with the
dual storage ingestion pipeline to generate vector embeddings for documents and entities.
"""

import logging
from typing import List, Optional, Union
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Simple embedding service for generating vector embeddings.
    
    This is a basic implementation that can be extended to use actual
    embedding models like sentence-transformers, OpenAI embeddings, etc.
    """
    
    def __init__(self, model_name: str = "dummy", embedding_dimension: int = 384):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use
            embedding_dimension: Dimension of the output embeddings
        """
        self.model_name = model_name
        self.embedding_dimension = embedding_dimension
        self._initialized = False
        
        # For a real implementation, you would initialize the actual model here
        # For example:
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def initialize(self) -> None:
        """Initialize the embedding model."""
        if self._initialized:
            return
        
        # In a real implementation, this would load the actual model
        logger.info(f"Initializing embedding service with model: {self.model_name}")
        self._initialized = True
        
    def encode(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s).
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Numpy array(s) containing the embeddings
        """
        if not self._initialized:
            self.initialize()
        
        if isinstance(text, str):
            return self._encode_single(text)
        else:
            return [self._encode_single(t) for t in text]
    
    def _encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to encode
            
        Returns:
            Numpy array containing the embedding
        """
        # This is a dummy implementation that creates a simple hash-based embedding
        # In a real implementation, you would use an actual embedding model
        
        # Create a simple hash-based embedding for demonstration
        # This is NOT suitable for production use
        text_hash = hash(text.lower().strip())
        np.random.seed(abs(text_hash) % (2**32))  # Use hash as seed for reproducibility
        
        # Generate a random embedding with some structure based on text characteristics
        embedding = np.random.normal(0, 1, self.embedding_dimension)
        
        # Add some text-based features to make it slightly more meaningful
        text_length_feature = min(len(text) / 1000.0, 1.0)  # Normalize text length
        word_count_feature = min(len(text.split()) / 100.0, 1.0)  # Normalize word count
        
        # Modify first few dimensions based on text features
        if len(embedding) > 0:
            embedding[0] = text_length_feature
        if len(embedding) > 1:
            embedding[1] = word_count_feature
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to encode
            batch_size: Size of batches for processing
            
        Returns:
            List of numpy arrays containing the embeddings
        """
        if not self._initialized:
            self.initialize()
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = [self._encode_single(text) for text in batch_texts]
            embeddings.extend(batch_embeddings)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimension")
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "initialized": self._initialized,
            "type": "dummy_embedding_service"
        }


class SentenceTransformerEmbeddingService(EmbeddingService):
    """
    Embedding service using sentence-transformers library.
    
    This is a more realistic implementation that uses actual pre-trained models.
    Requires: pip install sentence-transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize the sentence-transformers model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension from model
            test_embedding = self.model.encode("test")
            self.embedding_dimension = len(test_embedding)
            
            self._initialized = True
            logger.info(f"Sentence-transformers model loaded successfully. Dimension: {self.embedding_dimension}")
            
        except ImportError:
            logger.error("sentence-transformers library not found. Please install it: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize sentence-transformers model: {e}")
            raise
    
    def encode(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings using sentence-transformers."""
        if not self._initialized:
            self.initialize()
        
        try:
            embeddings = self.model.encode(text)
            
            if isinstance(text, str):
                return embeddings
            else:
                return [emb for emb in embeddings]
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for a batch using sentence-transformers."""
        if not self._initialized:
            self.initialize()
        
        try:
            # sentence-transformers handles batching internally
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
            return [emb for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """
    Get the global embedding service instance.
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    
    if _embedding_service is None:
        # Try to use sentence-transformers if available, otherwise use dummy
        try:
            import sentence_transformers
            # Use the configured model name or default
            actual_model = model_name if model_name != "dummy" else "all-MiniLM-L6-v2"
            _embedding_service = SentenceTransformerEmbeddingService(actual_model)
            logger.info(f"Using SentenceTransformer embedding service with model: {actual_model}")
        except ImportError:
            _embedding_service = EmbeddingService(model_name)
            logger.info("Using dummy embedding service (install sentence-transformers for better embeddings)")
    
    return _embedding_service


def initialize_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> None:
    """
    Initialize the global embedding service.
    
    Args:
        model_name: Name of the embedding model to use
    """
    service = get_embedding_service(model_name)
    service.initialize()


def create_embedding_service(service_type: str = "auto", model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """
    Create an embedding service of the specified type.
    
    Args:
        service_type: Type of service ("auto", "sentence_transformers", "dummy")
        model_name: Name of the model to use
        
    Returns:
        EmbeddingService instance
    """
    if service_type == "auto":
        try:
            import sentence_transformers
            return SentenceTransformerEmbeddingService(model_name)
        except ImportError:
            logger.warning("sentence-transformers not available, using dummy service")
            return EmbeddingService(model_name)
    
    elif service_type == "sentence_transformers":
        return SentenceTransformerEmbeddingService(model_name)
    
    elif service_type == "dummy":
        return EmbeddingService(model_name)
    
    else:
        raise ValueError(f"Unknown service type: {service_type}")