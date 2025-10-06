"""
Vector Retrieval Agent for the Graph-Enhanced Agentic RAG system.

This agent handles text embedding generation, semantic similarity search,
and hybrid search capabilities using sentence transformers and Pinecone vector database.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer
# Vector database imports handled through database manager

from core.interfaces import VectorRetrievalInterface, MessageType, VectorResult, Document
from core.protocols import AgentMessage
from core.vector_models import DocumentEmbedding, EmbeddingVector, EmbeddingValidationResult
from core.models import ValidationResult


logger = logging.getLogger(__name__)


class EmbeddingGenerationService:
    """
    Service for generating text embeddings using sentence transformers.
    
    Provides text embedding generation with caching and batch processing capabilities.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 1000,
        device: str = "cpu"
    ):
        """
        Initialize the embedding generation service.
        
        Args:
            model_name: Name of the sentence transformer model to use
            cache_size: Maximum number of embeddings to cache
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self.device = device
        self._model = None
        self._embedding_cache: Dict[str, EmbeddingVector] = {}
        self._cache_order: List[str] = []
        
        logger.info(f"Initializing EmbeddingGenerationService with model: {model_name}")
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded sentence transformer model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _add_to_cache(self, text: str, embedding: EmbeddingVector) -> None:
        """Add embedding to cache with LRU eviction."""
        cache_key = self._get_cache_key(text)
        
        # Remove if already exists to update order
        if cache_key in self._embedding_cache:
            self._cache_order.remove(cache_key)
        
        # Add to cache
        self._embedding_cache[cache_key] = embedding
        self._cache_order.append(cache_key)
        
        # Evict oldest if cache is full
        while len(self._embedding_cache) > self.cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._embedding_cache[oldest_key]
    
    def _get_from_cache(self, text: str) -> Optional[EmbeddingVector]:
        """Get embedding from cache."""
        cache_key = self._get_cache_key(text)
        if cache_key in self._embedding_cache:
            # Move to end (most recently used)
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._embedding_cache[cache_key]
        return None
    
    async def generate_embedding(self, text: str) -> EmbeddingVector:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingVector: Generated embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        
        # Check cache first
        cached_embedding = self._get_from_cache(text)
        if cached_embedding is not None:
            logger.debug(f"Retrieved embedding from cache for text length: {len(text)}")
            return cached_embedding
        
        # Load model if not loaded
        self._load_model()
        
        try:
            # Generate embedding
            embedding_array = await asyncio.get_event_loop().run_in_executor(
                None, self._model.encode, text
            )
            
            # Create EmbeddingVector
            embedding_vector = EmbeddingVector(
                vector=embedding_array,
                dimension=len(embedding_array),
                model_name=self.model_name,
                created_at=datetime.now()
            )
            
            # Add to cache
            self._add_to_cache(text, embedding_vector)
            
            logger.debug(f"Generated embedding for text length: {len(text)}, dimension: {embedding_vector.dimension}")
            return embedding_vector
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[EmbeddingVector]: List of generated embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts and check cache
        texts_to_process = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
                
            text = text.strip()
            cached = self._get_from_cache(text)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                texts_to_process.append((i, text))
        
        # Load model if needed
        if texts_to_process:
            self._load_model()
        
        # Process uncached texts in batch
        new_embeddings = {}
        if texts_to_process:
            try:
                batch_texts = [text for _, text in texts_to_process]
                embedding_arrays = await asyncio.get_event_loop().run_in_executor(
                    None, self._model.encode, batch_texts
                )
                
                for (original_idx, text), embedding_array in zip(texts_to_process, embedding_arrays):
                    embedding_vector = EmbeddingVector(
                        vector=embedding_array,
                        dimension=len(embedding_array),
                        model_name=self.model_name,
                        created_at=datetime.now()
                    )
                    
                    # Add to cache
                    self._add_to_cache(text, embedding_vector)
                    new_embeddings[original_idx] = embedding_vector
                
                logger.debug(f"Generated {len(new_embeddings)} embeddings in batch")
                
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                raise RuntimeError(f"Batch embedding generation failed: {e}")
        
        # Combine cached and new embeddings in original order
        result = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
                
            if i in cached_embeddings:
                result.append(cached_embeddings[i])
            elif i in new_embeddings:
                result.append(new_embeddings[i])
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"model_name": self.model_name, "loaded": False}
        
        return {
            "model_name": self.model_name,
            "loaded": True,
            "device": self.device,
            "max_seq_length": getattr(self._model, 'max_seq_length', None),
            "embedding_dimension": self._model.get_sentence_embedding_dimension()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._embedding_cache),
            "max_cache_size": self.cache_size,
            "cache_hit_ratio": len(self._embedding_cache) / max(1, len(self._cache_order))
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_order.clear()
        logger.info("Embedding cache cleared")


class VectorRetrievalAgent(VectorRetrievalInterface):
    """
    Vector Retrieval Agent implementation.
    
    Handles semantic similarity search, embedding generation, and hybrid search
    capabilities using Pinecone vector database and sentence transformers.
    """
    
    def __init__(
        self,
        agent_id: str = "vector_retrieval_agent",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "documents",
        embedding_cache_size: int = 1000
    ):
        """
        Initialize the Vector Retrieval Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            model_name: Sentence transformer model name
            collection_name: Name of the vector collection
            embedding_cache_size: Size of embedding cache
        """
        super().__init__(agent_id)
        
        self.model_name = model_name
        self.collection_name = collection_name
        
        # Initialize embedding service
        self.embedding_service = EmbeddingGenerationService(
            model_name=model_name,
            cache_size=embedding_cache_size
        )
        
        # Initialize vector database manager
        from core.database import get_vector_manager
        self.vector_manager = get_vector_manager()
        
        logger.info(f"Initialized VectorRetrievalAgent: {agent_id}")
    
    def _initialize_vector_db(self) -> None:
        """Initialize vector database connection."""
        if not hasattr(self, '_vector_initialized'):
            try:
                # Create collection in vector database
                self.vector_manager.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Document embeddings for RAG system"}
                )
                
                self._vector_initialized = True
                logger.info(f"Initialized vector database collection: {self.collection_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize vector database: {e}")
                raise RuntimeError(f"Vector database initialization failed: {e}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector as list of floats
        """
        try:
            embedding_vector = await self.embedding_service.generate_embedding(text)
            return embedding_vector.to_list()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            embedding_vectors = await self.embedding_service.generate_embeddings_batch(texts)
            return [emb.to_list() for emb in embedding_vectors]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    async def similarity_search(
        self, 
        query: str, 
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> VectorResult:
        """
        Perform similarity search in vector database.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            VectorResult: Search results with documents and similarities
        """
        try:
            # Initialize vector database if needed
            self._initialize_vector_db()
            
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if isinstance(value, (str, int, float, bool)):
                        where_clause[key] = value
                    elif isinstance(value, list):
                        where_clause[key] = {"$in": value}
            
            # Perform similarity search using vector manager
            results = self.vector_manager.query_collection(
                collection_name=self.collection_name,
                query_text=query,  # Let the manager handle embedding generation
                n_results=k
            )
            
            # Convert results to VectorResult format
            documents = []
            similarities = []
            
            if results["documents"] and results["documents"][0]:
                for i, (doc_content, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0] or [{}] * len(results["documents"][0]),
                    results["distances"][0]
                )):
                    # Convert distance to similarity (Pinecone uses cosine similarity)
                    similarity = 1.0 / (1.0 + distance)
                    
                    # Ensure metadata is a dict and handle title properly
                    safe_metadata = metadata if isinstance(metadata, dict) else {}
                    title = safe_metadata.get("title") or f"Document Chunk {i + 1}"
                    source = safe_metadata.get("source") or "Unknown Source"
                    
                    document = Document(
                        id=safe_metadata.get("id", f"doc_{i}"),
                        content=doc_content,
                        title=title,
                        metadata=safe_metadata,
                        source=source
                    )

                    
                    documents.append(document)
                    similarities.append(similarity)
            
            logger.debug(f"Similarity search returned {len(documents)} results for query length: {len(query)}")
            
            return VectorResult(
                documents=documents,
                similarities=similarities,
                query_embedding=query_embedding
            )
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Vector similarity search failed: {e}")
    
    async def add_documents(
        self, 
        documents: List[DocumentEmbedding],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of DocumentEmbedding objects to add
            batch_size: Number of documents to process in each batch
        """
        try:
            # Initialize vector database if needed
            self._initialize_vector_db()
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Prepare batch data
                ids = [doc.id for doc in batch]
                embeddings = [doc.embedding for doc in batch]
                documents_content = [doc.content for doc in batch]
                metadatas = []
                
                for doc in batch:
                    metadata = doc.metadata.copy()
                    
                    # Convert list fields to strings for vector database compatibility
                    graph_entity_ids_str = ",".join(doc.graph_entity_ids) if doc.graph_entity_ids else ""
                    
                    metadata.update({
                        "id": doc.id,
                        "title": doc.title or "",
                        "source": doc.source or "",
                        "document_type": doc.document_type or "",
                        "embedding_model": doc.embedding_model,
                        "embedding_dimension": doc.embedding_dimension,
                        "graph_entity_ids": graph_entity_ids_str,
                        "created_at": doc.created_at.isoformat(),
                        "updated_at": doc.updated_at.isoformat()
                    })
                    
                    # Ensure all metadata values are vector database-compatible types
                    for key, value in list(metadata.items()):
                        if isinstance(value, list):
                            metadata[key] = ",".join(str(v) for v in value)
                        elif value is None:
                            metadata[key] = ""
                        elif not isinstance(value, (str, int, float, bool)):
                            metadata[key] = str(value)
                    
                    metadatas.append(metadata)
                
                # Add to vector database using manager
                self.vector_manager.add_documents(
                    collection_name=self.collection_name,
                    documents=documents_content,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.debug(f"Added batch of {len(batch)} documents to vector database")
            
            logger.info(f"Successfully added {len(documents)} documents to vector database")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector database: {e}")
            raise RuntimeError(f"Document addition failed: {e}")
    
    async def update_document(self, document: DocumentEmbedding) -> None:
        """
        Update a document in the vector database.
        
        Args:
            document: DocumentEmbedding object to update
        """
        try:
            # Initialize vector database if needed
            self._initialize_vector_db()
            
            # Prepare metadata
            metadata = document.metadata.copy()
            
            # Convert list fields to strings for vector database compatibility
            graph_entity_ids_str = ",".join(document.graph_entity_ids) if document.graph_entity_ids else ""
            
            metadata.update({
                "id": document.id,
                "title": document.title or "",
                "source": document.source or "",
                "document_type": document.document_type or "",
                "embedding_model": document.embedding_model,
                "embedding_dimension": document.embedding_dimension,
                "graph_entity_ids": graph_entity_ids_str,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat()
            })
            
            # Ensure all metadata values are vector database-compatible types
            for key, value in list(metadata.items()):
                if isinstance(value, list):
                    metadata[key] = ",".join(str(v) for v in value)
                elif value is None:
                    metadata[key] = ""
                elif not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
            
            # Update in vector database (Pinecone uses upsert for updates)
            self.vector_manager.add_documents(
                collection_name=self.collection_name,
                documents=[document.content],
                metadatas=[metadata],
                ids=[document.id]
            )
            
            logger.debug(f"Updated document {document.id} in vector database")
            
        except Exception as e:
            logger.error(f"Failed to update document {document.id}: {e}")
            raise RuntimeError(f"Document update failed: {e}")
    
    async def delete_document(self, document_id: str) -> None:
        """
        Delete a document from the vector database.
        
        Args:
            document_id: ID of the document to delete
        """
        try:
            # Initialize vector database if needed
            self._initialize_vector_db()
            
            # Delete from vector database
            self.vector_manager.delete_documents([document_id])
            
            logger.debug(f"Deleted document {document_id} from vector database")
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            raise RuntimeError(f"Document deletion failed: {e}")
    
    async def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector database.
        
        Returns:
            int: Number of documents
        """
        try:
            # Initialize vector database if needed
            self._initialize_vector_db()
            
            if hasattr(self.vector_manager, 'get_index_stats'):
                # Pinecone
                stats = self.vector_manager.get_index_stats()
                return stats.get('total_vectors', 0)
            else:
                # Vector database fallback
                return getattr(self.vector_manager, 'count', lambda: 0)()
            
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def _calculate_relevance_scores(
        self, 
        documents: List[Document], 
        similarities: List[float],
        query: str
    ) -> List[float]:
        """
        Calculate enhanced relevance scores based on multiple factors.
        
        Args:
            documents: List of documents
            similarities: List of similarity scores
            query: Original query text
            
        Returns:
            List[float]: Enhanced relevance scores
        """
        relevance_scores = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for doc, similarity in zip(documents, similarities):
            # Start with similarity score
            relevance = similarity
            
            # Boost for exact phrase matches
            if query_lower in doc.content.lower():
                relevance *= 1.2
            
            # Boost for keyword matches
            doc_words = set(doc.content.lower().split())
            keyword_overlap = len(query_words.intersection(doc_words)) / len(query_words)
            relevance *= (1.0 + keyword_overlap * 0.3)
            
            # Boost for title matches
            if doc.metadata.get("title"):
                title_lower = doc.metadata["title"].lower()
                if query_lower in title_lower:
                    relevance *= 1.15
                
                title_words = set(title_lower.split())
                title_overlap = len(query_words.intersection(title_words)) / len(query_words)
                relevance *= (1.0 + title_overlap * 0.2)
            
            # Penalize very short documents
            if len(doc.content) < 100:
                relevance *= 0.9
            
            relevance_scores.append(min(relevance, 1.0))  # Cap at 1.0
        
        return relevance_scores
    
    async def rerank_results(
        self, 
        documents: List[Document], 
        similarities: List[float],
        query: str
    ) -> Tuple[List[Document], List[float]]:
        """
        Re-rank search results based on enhanced relevance scoring.
        
        Args:
            documents: List of documents to re-rank
            similarities: Original similarity scores
            query: Original query text
            
        Returns:
            Tuple[List[Document], List[float]]: Re-ranked documents and scores
        """
        if not documents:
            return documents, similarities
        
        # Calculate enhanced relevance scores
        relevance_scores = self._calculate_relevance_scores(documents, similarities, query)
        
        # Sort by relevance scores (descending)
        sorted_pairs = sorted(
            zip(documents, relevance_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        reranked_documents, reranked_scores = zip(*sorted_pairs)
        
        logger.debug(f"Re-ranked {len(documents)} documents based on enhanced relevance")
        
        return list(reranked_documents), list(reranked_scores)
    
    async def hybrid_search(
        self, 
        query: str, 
        semantic_weight: float = 0.7,
        k: int = 10
    ) -> VectorResult:
        """
        Perform hybrid semantic and keyword search.
        
        Args:
            query: Query text
            semantic_weight: Weight for semantic similarity (0.0 to 1.0)
            k: Number of results to return
            
        Returns:
            VectorResult: Hybrid search results
        """
        try:
            # Perform semantic search
            semantic_results = await self.similarity_search(query, k=k*2)  # Get more for filtering
            
            # Re-rank results with hybrid scoring
            reranked_docs, reranked_scores = await self.rerank_results(
                semantic_results.documents,
                semantic_results.similarities,
                query
            )
            
            # Apply semantic weight to combine scores
            final_scores = []
            for semantic_score, relevance_score in zip(semantic_results.similarities, reranked_scores):
                hybrid_score = (semantic_weight * semantic_score + 
                              (1 - semantic_weight) * relevance_score)
                final_scores.append(hybrid_score)
            
            # Take top k results
            top_k_docs = reranked_docs[:k]
            top_k_scores = final_scores[:k]
            
            logger.debug(f"Hybrid search returned {len(top_k_docs)} results")
            
            return VectorResult(
                documents=top_k_docs,
                similarities=top_k_scores,
                query_embedding=semantic_results.query_embedding
            )
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise RuntimeError(f"Hybrid search failed: {e}")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming messages from other agents.
        
        Args:
            message: Incoming agent message
            
        Returns:
            Optional[AgentMessage]: Response message if needed
        """
        try:
            if message.message_type == MessageType.VECTOR_SEARCH:
                payload = message.payload
                query = payload.get("query", "")
                k = payload.get("k", 10)
                filters = payload.get("filters")
                search_type = payload.get("search_type", "similarity")
                
                if search_type == "hybrid":
                    semantic_weight = payload.get("semantic_weight", 0.7)
                    result = await self.hybrid_search(query, semantic_weight, k)
                else:
                    result = await self.similarity_search(query, k, filters)
                
                # Convert result to message payload
                response_payload = {
                    "documents": [doc.dict() for doc in result.documents],
                    "similarities": result.similarities,
                    "query_embedding": result.query_embedding,
                    "search_type": search_type
                }
                
                return self.create_message(
                    MessageType.RESPONSE,
                    response_payload,
                    message.correlation_id
                )
            
            else:
                logger.warning(f"Unsupported message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_payload = {
                "error": str(e),
                "message_type": message.message_type.value
            }
            return self.create_message(
                MessageType.ERROR,
                error_payload,
                message.correlation_id
            )
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent and its configuration."""
        model_info = self.embedding_service.get_model_info()
        cache_stats = self.embedding_service.get_cache_stats()
        
        return {
            "agent_id": self.agent_id,
            "agent_type": "VectorRetrievalAgent",
            "model_info": model_info,
            "cache_stats": cache_stats,
            "vector_db_config": {
                "type": "pinecone",
                "collection_name": self.collection_name
            },
            "document_count": asyncio.run(self.get_document_count()) if hasattr(self, '_vector_initialized') else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the agent and its dependencies."""
        health_status = {
            "agent_id": self.agent_id,
            "status": "healthy",
            "checks": {}
        }
        
        try:
            # Check embedding service
            test_embedding = await self.embedding_service.generate_embedding("test")
            health_status["checks"]["embedding_service"] = {
                "status": "healthy",
                "embedding_dimension": test_embedding.dimension
            }
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["checks"]["embedding_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            # Check vector database connection
            self._initialize_vector_db()
            doc_count = await self.get_document_count()
            health_status["checks"]["vector_database"] = {
                "status": "healthy",
                "document_count": doc_count
            }
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["checks"]["vector_database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        return health_status