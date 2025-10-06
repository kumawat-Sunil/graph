"""
Vector embedding models for the Graph-Enhanced Agentic RAG system.

This module defines vector document classes, embedding models, and entity-vector
mapping data structures to maintain consistency between graph and vector representations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid
import numpy as np
import json
from dataclasses import dataclass


class EmbeddingType(str, Enum):
    """Types of embeddings supported by the system."""
    DOCUMENT = "document"
    ENTITY = "entity"
    CONCEPT = "concept"
    QUERY = "query"


class VectorStoreType(str, Enum):
    """Types of vector stores supported."""
    PINECONE = "pinecone"
    CHROMA = "chroma"  # Legacy support
    FAISS = "faiss"
    WEAVIATE = "weaviate"


@dataclass
class EmbeddingVector:
    """
    Represents a single embedding vector with metadata.
    
    This is a lightweight data structure for handling raw embedding vectors
    with their associated metadata.
    """
    vector: np.ndarray
    dimension: int
    model_name: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        
        # Validate vector dimensions
        if len(self.vector) != self.dimension:
            raise ValueError(f"Vector length {len(self.vector)} doesn't match dimension {self.dimension}")
    
    def to_list(self) -> List[float]:
        """Convert numpy array to list for serialization."""
        return self.vector.tolist()
    
    def cosine_similarity(self, other: 'EmbeddingVector') -> float:
        """Calculate cosine similarity with another embedding vector."""
        if self.dimension != other.dimension:
            raise ValueError("Cannot compare vectors of different dimensions")
        
        dot_product = np.dot(self.vector, other.vector)
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


class DocumentEmbedding(BaseModel):
    """
    Represents a document with its vector embedding and metadata.
    
    This class maintains the connection between document content and its
    vector representation, including links to graph entities.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., min_length=1)
    title: Optional[str] = Field(None, max_length=1000)
    embedding: List[float] = Field(..., description="Vector embedding of the document")
    embedding_model: str = Field(..., description="Model used to generate embedding")
    embedding_dimension: int = Field(..., gt=0, description="Dimension of the embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    graph_entity_ids: List[str] = Field(default_factory=list, description="Linked graph entity IDs")
    source: Optional[str] = Field(None, max_length=2000)
    document_type: str = Field(default="text")
    chunk_index: Optional[int] = Field(None, description="Index if document is chunked")
    parent_document_id: Optional[str] = Field(None, description="Parent document if this is a chunk")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('embedding')
    def validate_embedding(cls, v, values):
        """Validate embedding vector consistency."""
        if not isinstance(v, list):
            raise ValueError("Embedding must be a list of floats")
        
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("All embedding values must be numeric")
        
        if len(v) == 0:
            raise ValueError("Embedding cannot be empty")
        
        return v
    
    @validator('embedding_dimension')
    def validate_embedding_dimension(cls, v, values):
        """Validate embedding dimension consistency."""
        if 'embedding' in values:
            embedding = values['embedding']
            if len(embedding) != v:
                raise ValueError(f"Embedding length {len(embedding)} doesn't match specified dimension {v}")
        
        return v
    
    @validator('graph_entity_ids')
    def validate_entity_ids(cls, v):
        """Validate entity ID references."""
        if v is None:
            return []
        
        if not isinstance(v, list):
            raise ValueError("Entity IDs must be a list")
        
        # Validate UUID format for each ID
        for entity_id in v:
            if not isinstance(entity_id, str):
                raise ValueError("Entity IDs must be strings")
            try:
                uuid.UUID(entity_id)
            except ValueError:
                raise ValueError(f"Invalid UUID format for entity ID: {entity_id}")
        
        return v
    
    @validator('content')
    def validate_content(cls, v):
        """Validate document content."""
        if not v or not v.strip():
            raise ValueError("Document content cannot be empty")
        return v.strip()
    
    def get_embedding_vector(self) -> EmbeddingVector:
        """Get the embedding as an EmbeddingVector object."""
        return EmbeddingVector(
            vector=np.array(self.embedding),
            dimension=self.embedding_dimension,
            model_name=self.embedding_model,
            created_at=self.created_at
        )
    
    def add_entity_link(self, entity_id: str) -> None:
        """Link this document to a graph entity."""
        try:
            uuid.UUID(entity_id)
        except ValueError:
            raise ValueError(f"Invalid UUID format for entity ID: {entity_id}")
        
        if entity_id not in self.graph_entity_ids:
            self.graph_entity_ids.append(entity_id)
            self.updated_at = datetime.now()
    
    def remove_entity_link(self, entity_id: str) -> None:
        """Remove link to a graph entity."""
        if entity_id in self.graph_entity_ids:
            self.graph_entity_ids.remove(entity_id)
            self.updated_at = datetime.now()
    
    def similarity_to(self, other: 'DocumentEmbedding') -> float:
        """Calculate cosine similarity to another document embedding."""
        if self.embedding_dimension != other.embedding_dimension:
            raise ValueError("Cannot compare embeddings of different dimensions")
        
        self_vector = self.get_embedding_vector()
        other_vector = other.get_embedding_vector()
        
        return self_vector.cosine_similarity(other_vector)


class EntityEmbedding(BaseModel):
    """
    Represents an entity with its vector embeddings from different contexts.
    
    Entities can have multiple embeddings based on different contexts
    where they appear in documents.
    """
    entity_id: str = Field(..., description="ID of the associated graph entity")
    entity_name: str = Field(..., min_length=1, max_length=500)
    description_embedding: List[float] = Field(..., description="Embedding of entity description")
    context_embeddings: List[List[float]] = Field(default_factory=list, description="Embeddings from different contexts")
    context_sources: List[str] = Field(default_factory=list, description="Sources of context embeddings")
    embedding_model: str = Field(..., description="Model used to generate embeddings")
    embedding_dimension: int = Field(..., gt=0, description="Dimension of embedding vectors")
    graph_node_id: str = Field(..., description="ID of the node in graph database")
    aggregated_embedding: Optional[List[float]] = Field(None, description="Aggregated embedding from all contexts")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('entity_id', 'graph_node_id')
    def validate_ids(cls, v):
        """Validate ID format."""
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError(f"Invalid UUID format: {v}")
        return v
    
    @validator('description_embedding')
    def validate_description_embedding(cls, v, values):
        """Validate description embedding."""
        if not isinstance(v, list) or not v:
            raise ValueError("Description embedding must be a non-empty list")
        
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("All embedding values must be numeric")
        
        if 'embedding_dimension' in values and len(v) != values['embedding_dimension']:
            raise ValueError(f"Description embedding length doesn't match dimension")
        
        return v
    
    @validator('context_embeddings')
    def validate_context_embeddings(cls, v, values):
        """Validate context embeddings."""
        if not isinstance(v, list):
            raise ValueError("Context embeddings must be a list")
        
        embedding_dim = values.get('embedding_dimension')
        for i, embedding in enumerate(v):
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"Context embedding {i} must be a non-empty list")
            
            if not all(isinstance(x, (int, float)) for x in embedding):
                raise ValueError(f"Context embedding {i} contains non-numeric values")
            
            if embedding_dim and len(embedding) != embedding_dim:
                raise ValueError(f"Context embedding {i} length doesn't match dimension")
        
        return v
    
    @validator('context_sources')
    def validate_context_sources(cls, v, values):
        """Validate that context sources match context embeddings."""
        context_embeddings = values.get('context_embeddings', [])
        if len(v) != len(context_embeddings):
            raise ValueError("Number of context sources must match number of context embeddings")
        return v
    
    def add_context_embedding(self, embedding: List[float], source: str) -> None:
        """Add a new context embedding."""
        if len(embedding) != self.embedding_dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embedding)}")
        
        if not all(isinstance(x, (int, float)) for x in embedding):
            raise ValueError("All embedding values must be numeric")
        
        self.context_embeddings.append(embedding)
        self.context_sources.append(source)
        self.updated_at = datetime.now()
        
        # Recalculate aggregated embedding
        self._update_aggregated_embedding()
    
    def _update_aggregated_embedding(self) -> None:
        """Update the aggregated embedding from all contexts."""
        if not self.context_embeddings:
            self.aggregated_embedding = self.description_embedding.copy()
            return
        
        # Simple average of all embeddings (description + contexts)
        all_embeddings = [self.description_embedding] + self.context_embeddings
        aggregated = np.mean(all_embeddings, axis=0)
        self.aggregated_embedding = aggregated.tolist()
    
    def get_primary_embedding(self) -> EmbeddingVector:
        """Get the primary (aggregated or description) embedding."""
        embedding_data = self.aggregated_embedding or self.description_embedding
        return EmbeddingVector(
            vector=np.array(embedding_data),
            dimension=self.embedding_dimension,
            model_name=self.embedding_model,
            created_at=self.created_at
        )
    
    def similarity_to_text(self, text_embedding: List[float]) -> float:
        """Calculate similarity to a text embedding."""
        if len(text_embedding) != self.embedding_dimension:
            raise ValueError("Embedding dimension mismatch")
        
        primary_embedding = self.get_primary_embedding()
        text_vector = EmbeddingVector(
            vector=np.array(text_embedding),
            dimension=self.embedding_dimension,
            model_name=self.embedding_model
        )
        
        return primary_embedding.cosine_similarity(text_vector)


class EntityVectorMapping(BaseModel):
    """
    Maintains bidirectional mapping between graph entities and vector embeddings.
    
    This class ensures consistency between the graph database and vector store
    by tracking relationships between entities and their vector representations.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_to_vectors: Dict[str, List[str]] = Field(default_factory=dict, description="Entity ID to vector IDs mapping")
    vector_to_entities: Dict[str, List[str]] = Field(default_factory=dict, description="Vector ID to entity IDs mapping")
    entity_to_documents: Dict[str, List[str]] = Field(default_factory=dict, description="Entity ID to document IDs mapping")
    document_to_entities: Dict[str, List[str]] = Field(default_factory=dict, description="Document ID to entity IDs mapping")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def link_entity_vector(self, entity_id: str, vector_id: str) -> None:
        """
        Create bidirectional link between entity and vector.
        
        Args:
            entity_id: ID of the graph entity
            vector_id: ID of the vector embedding
        """
        # Validate IDs
        try:
            uuid.UUID(entity_id)
            uuid.UUID(vector_id)
        except ValueError as e:
            raise ValueError(f"Invalid ID format: {e}")
        
        # Add entity -> vector mapping
        if entity_id not in self.entity_to_vectors:
            self.entity_to_vectors[entity_id] = []
        if vector_id not in self.entity_to_vectors[entity_id]:
            self.entity_to_vectors[entity_id].append(vector_id)
        
        # Add vector -> entity mapping
        if vector_id not in self.vector_to_entities:
            self.vector_to_entities[vector_id] = []
        if entity_id not in self.vector_to_entities[vector_id]:
            self.vector_to_entities[vector_id].append(entity_id)
        
        self.updated_at = datetime.now()
    
    def unlink_entity_vector(self, entity_id: str, vector_id: str) -> None:
        """
        Remove bidirectional link between entity and vector.
        
        Args:
            entity_id: ID of the graph entity
            vector_id: ID of the vector embedding
        """
        # Remove from entity -> vector mapping
        if entity_id in self.entity_to_vectors:
            if vector_id in self.entity_to_vectors[entity_id]:
                self.entity_to_vectors[entity_id].remove(vector_id)
            if not self.entity_to_vectors[entity_id]:
                del self.entity_to_vectors[entity_id]
        
        # Remove from vector -> entity mapping
        if vector_id in self.vector_to_entities:
            if entity_id in self.vector_to_entities[vector_id]:
                self.vector_to_entities[vector_id].remove(entity_id)
            if not self.vector_to_entities[vector_id]:
                del self.vector_to_entities[vector_id]
        
        self.updated_at = datetime.now()
    
    def link_entity_document(self, entity_id: str, document_id: str) -> None:
        """
        Create bidirectional link between entity and document.
        
        Args:
            entity_id: ID of the graph entity
            document_id: ID of the document
        """
        # Validate IDs
        try:
            uuid.UUID(entity_id)
            uuid.UUID(document_id)
        except ValueError as e:
            raise ValueError(f"Invalid ID format: {e}")
        
        # Add entity -> document mapping
        if entity_id not in self.entity_to_documents:
            self.entity_to_documents[entity_id] = []
        if document_id not in self.entity_to_documents[entity_id]:
            self.entity_to_documents[entity_id].append(document_id)
        
        # Add document -> entity mapping
        if document_id not in self.document_to_entities:
            self.document_to_entities[document_id] = []
        if entity_id not in self.document_to_entities[document_id]:
            self.document_to_entities[document_id].append(entity_id)
        
        self.updated_at = datetime.now()
    
    def get_vectors_for_entity(self, entity_id: str) -> List[str]:
        """Get all vector IDs associated with an entity."""
        return self.entity_to_vectors.get(entity_id, [])
    
    def get_entities_for_vector(self, vector_id: str) -> List[str]:
        """Get all entity IDs associated with a vector."""
        return self.vector_to_entities.get(vector_id, [])
    
    def get_documents_for_entity(self, entity_id: str) -> List[str]:
        """Get all document IDs associated with an entity."""
        return self.entity_to_documents.get(entity_id, [])
    
    def get_entities_for_document(self, document_id: str) -> List[str]:
        """Get all entity IDs associated with a document."""
        return self.document_to_entities.get(document_id, [])
    
    def validate_consistency(self) -> Tuple[bool, List[str]]:
        """
        Validate the consistency of all mappings.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check entity -> vector consistency
        for entity_id, vector_ids in self.entity_to_vectors.items():
            for vector_id in vector_ids:
                if vector_id not in self.vector_to_entities:
                    errors.append(f"Entity {entity_id} links to vector {vector_id}, but reverse mapping missing")
                elif entity_id not in self.vector_to_entities[vector_id]:
                    errors.append(f"Entity {entity_id} links to vector {vector_id}, but reverse mapping incomplete")
        
        # Check vector -> entity consistency
        for vector_id, entity_ids in self.vector_to_entities.items():
            for entity_id in entity_ids:
                if entity_id not in self.entity_to_vectors:
                    errors.append(f"Vector {vector_id} links to entity {entity_id}, but reverse mapping missing")
                elif vector_id not in self.entity_to_vectors[entity_id]:
                    errors.append(f"Vector {vector_id} links to entity {entity_id}, but reverse mapping incomplete")
        
        # Check entity -> document consistency
        for entity_id, document_ids in self.entity_to_documents.items():
            for document_id in document_ids:
                if document_id not in self.document_to_entities:
                    errors.append(f"Entity {entity_id} links to document {document_id}, but reverse mapping missing")
                elif entity_id not in self.document_to_entities[document_id]:
                    errors.append(f"Entity {entity_id} links to document {document_id}, but reverse mapping incomplete")
        
        # Check document -> entity consistency
        for document_id, entity_ids in self.document_to_entities.items():
            for entity_id in entity_ids:
                if entity_id not in self.entity_to_documents:
                    errors.append(f"Document {document_id} links to entity {entity_id}, but reverse mapping missing")
                elif document_id not in self.entity_to_documents[entity_id]:
                    errors.append(f"Document {document_id} links to entity {entity_id}, but reverse mapping incomplete")
        
        return len(errors) == 0, errors
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the mappings."""
        return {
            "total_entities": len(self.entity_to_vectors),
            "total_vectors": len(self.vector_to_entities),
            "total_documents": len(self.document_to_entities),
            "entity_vector_links": sum(len(vectors) for vectors in self.entity_to_vectors.values()),
            "entity_document_links": sum(len(docs) for docs in self.entity_to_documents.values())
        }


# Validation functions for embedding consistency

def validate_embedding_consistency(
    document_embedding: DocumentEmbedding,
    entity_embeddings: List[EntityEmbedding],
    mapping: EntityVectorMapping
) -> Tuple[bool, List[str]]:
    """
    Validate consistency between document embeddings, entity embeddings, and mappings.
    
    Args:
        document_embedding: Document embedding to validate
        entity_embeddings: List of entity embeddings to check against
        mapping: Entity-vector mapping to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if document's linked entities exist in entity embeddings
    entity_embedding_ids = {emb.entity_id for emb in entity_embeddings}
    for entity_id in document_embedding.graph_entity_ids:
        if entity_id not in entity_embedding_ids:
            errors.append(f"Document {document_embedding.id} links to entity {entity_id} which has no embedding")
    
    # Check if document is properly mapped to its entities
    for entity_id in document_embedding.graph_entity_ids:
        mapped_docs = mapping.get_documents_for_entity(entity_id)
        if document_embedding.id not in mapped_docs:
            errors.append(f"Document {document_embedding.id} links to entity {entity_id} but mapping is missing")
    
    # Check embedding dimensions consistency
    for entity_emb in entity_embeddings:
        if entity_emb.embedding_dimension != document_embedding.embedding_dimension:
            errors.append(f"Dimension mismatch: document {document_embedding.embedding_dimension} vs entity {entity_emb.entity_id} {entity_emb.embedding_dimension}")
    
    return len(errors) == 0, errors


def validate_embedding_quality(embedding: List[float], min_norm: float = 0.1, max_norm: float = 10.0) -> Tuple[bool, List[str]]:
    """
    Validate the quality of an embedding vector.
    
    Args:
        embedding: The embedding vector to validate
        min_norm: Minimum acceptable norm
        max_norm: Maximum acceptable norm
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    if not embedding:
        return False, ["Embedding is empty"]
    
    # Convert to numpy array for calculations
    emb_array = np.array(embedding)
    
    # Check for NaN or infinite values
    if np.any(np.isnan(emb_array)):
        return False, ["Embedding contains NaN values"]
    
    if np.any(np.isinf(emb_array)):
        return False, ["Embedding contains infinite values"]
    
    # Check norm
    norm = np.linalg.norm(emb_array)
    if norm < min_norm:
        warnings.append(f"Embedding norm {norm:.4f} is very small (< {min_norm})")
    elif norm > max_norm:
        warnings.append(f"Embedding norm {norm:.4f} is very large (> {max_norm})")
    
    # Check for zero vector
    if norm == 0:
        return False, ["Embedding is a zero vector"]
    
    # Check for unusual distributions
    std_dev = np.std(emb_array)
    if std_dev < 0.01:
        warnings.append(f"Embedding has very low variance (std: {std_dev:.4f})")
    
    return True, warnings


class EmbeddingValidationResult(BaseModel):
    """Result of embedding validation operations."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    
    def add_error(self, error: str) -> None:
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(warning)
    
    def merge(self, other: 'EmbeddingValidationResult') -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.statistics.update(other.statistics)


def comprehensive_embedding_validation(
    document_embeddings: List[DocumentEmbedding],
    entity_embeddings: List[EntityEmbedding],
    mapping: EntityVectorMapping
) -> EmbeddingValidationResult:
    """
    Perform comprehensive validation of all embedding components.
    
    Args:
        document_embeddings: List of document embeddings
        entity_embeddings: List of entity embeddings
        mapping: Entity-vector mapping
        
    Returns:
        EmbeddingValidationResult: Comprehensive validation result
    """
    result = EmbeddingValidationResult(is_valid=True)
    
    # Validate mapping consistency
    mapping_valid, mapping_errors = mapping.validate_consistency()
    if not mapping_valid:
        result.is_valid = False
        result.errors.extend(mapping_errors)
    
    # Validate individual document embeddings
    for doc_emb in document_embeddings:
        quality_valid, quality_warnings = validate_embedding_quality(doc_emb.embedding)
        if not quality_valid:
            result.add_error(f"Document {doc_emb.id} has invalid embedding quality")
        result.warnings.extend([f"Document {doc_emb.id}: {w}" for w in quality_warnings])
        
        # Validate consistency with entities and mapping
        consistency_valid, consistency_errors = validate_embedding_consistency(
            doc_emb, entity_embeddings, mapping
        )
        if not consistency_valid:
            result.errors.extend(consistency_errors)
    
    # Validate individual entity embeddings
    for ent_emb in entity_embeddings:
        # Validate description embedding
        desc_valid, desc_warnings = validate_embedding_quality(ent_emb.description_embedding)
        if not desc_valid:
            result.add_error(f"Entity {ent_emb.entity_id} has invalid description embedding")
        result.warnings.extend([f"Entity {ent_emb.entity_id} description: {w}" for w in desc_warnings])
        
        # Validate context embeddings
        for i, context_emb in enumerate(ent_emb.context_embeddings):
            ctx_valid, ctx_warnings = validate_embedding_quality(context_emb)
            if not ctx_valid:
                result.add_error(f"Entity {ent_emb.entity_id} context {i} has invalid embedding")
            result.warnings.extend([f"Entity {ent_emb.entity_id} context {i}: {w}" for w in ctx_warnings])
    
    # Add statistics
    result.statistics = {
        "document_embeddings_count": len(document_embeddings),
        "entity_embeddings_count": len(entity_embeddings),
        "mapping_statistics": mapping.get_statistics(),
        "total_errors": len(result.errors),
        "total_warnings": len(result.warnings)
    }
    
    return result


class VectorConsistencyError(Exception):
    """Exception raised when vector consistency validation fails."""
    pass


class EmbeddingQualityError(Exception):
    """Exception raised when embedding quality validation fails."""
    pass