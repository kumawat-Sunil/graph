"""
Core data model classes for the Graph-Enhanced Agentic RAG system.

This module defines the fundamental data structures used throughout the system,
including entities, documents, concepts, and validation functions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid
import re


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    LOCATION = "location"
    TECHNOLOGY = "technology"
    DOCUMENT = "document"
    EVENT = "event"
    GENERIC = "generic"


class DocumentType(str, Enum):
    """Types of documents in the system."""
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"


class ConceptDomain(str, Enum):
    """Domains for concept classification."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    SCIENTIFIC = "scientific"
    GENERAL = "general"
    DOMAIN_SPECIFIC = "domain_specific"


class Entity(BaseModel):
    """
    Represents an entity in the knowledge graph.
    
    Entities are the core nodes in our graph database, representing
    real-world objects, concepts, or abstract ideas.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=500)
    type: EntityType = Field(default=EntityType.GENERIC)
    description: Optional[str] = Field(None, max_length=2000)
    properties: Dict[str, Any] = Field(default_factory=dict)
    vector_id: Optional[str] = Field(None, description="Link to vector embedding")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('name')
    def validate_name(cls, v):
        """Validate entity name format."""
        if not v or not v.strip():
            raise ValueError("Entity name cannot be empty")
        # Remove excessive whitespace
        return re.sub(r'\s+', ' ', v.strip())
    
    @validator('properties')
    def validate_properties(cls, v):
        """Validate entity properties."""
        if v is None:
            return {}
        # Ensure all keys are strings and values are serializable
        validated = {}
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError("Property keys must be strings")
            # Check if value is JSON serializable
            try:
                import json
                json.dumps(value)
                validated[key] = value
            except (TypeError, ValueError):
                raise ValueError(f"Property value for '{key}' is not JSON serializable")
        return validated
    
    def add_property(self, key: str, value: Any) -> None:
        """Add a property to the entity."""
        if not isinstance(key, str):
            raise ValueError("Property key must be a string")
        self.properties[key] = value
        self.updated_at = datetime.now()
    
    def remove_property(self, key: str) -> None:
        """Remove a property from the entity."""
        if key in self.properties:
            del self.properties[key]
            self.updated_at = datetime.now()


class Document(BaseModel):
    """
    Represents a document in the vector store and knowledge graph.
    
    Documents contain the actual content that gets embedded and stored
    in both the vector database and referenced in the graph.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=1, max_length=1000)
    content: str = Field(..., min_length=1)
    document_type: DocumentType = Field(default=DocumentType.TEXT)
    source: Optional[str] = Field(None, max_length=2000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(None)
    entity_ids: List[str] = Field(default_factory=list, description="Linked entity IDs")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('content')
    def validate_content(cls, v):
        """Validate document content."""
        if not v or not v.strip():
            raise ValueError("Document content cannot be empty")
        return v.strip()
    
    @validator('title')
    def validate_title(cls, v):
        """Validate document title."""
        if not v or not v.strip():
            raise ValueError("Document title cannot be empty")
        return re.sub(r'\s+', ' ', v.strip())
    
    @validator('embedding')
    def validate_embedding(cls, v):
        """Validate embedding vector."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("Embedding must be a list of floats")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("All embedding values must be numeric")
            if len(v) == 0:
                raise ValueError("Embedding cannot be empty")
        return v
    
    @validator('entity_ids')
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
    
    def add_entity_link(self, entity_id: str) -> None:
        """Link this document to an entity."""
        try:
            uuid.UUID(entity_id)
        except ValueError:
            raise ValueError(f"Invalid UUID format for entity ID: {entity_id}")
        
        if entity_id not in self.entity_ids:
            self.entity_ids.append(entity_id)
            self.updated_at = datetime.now()
    
    def remove_entity_link(self, entity_id: str) -> None:
        """Remove link to an entity."""
        if entity_id in self.entity_ids:
            self.entity_ids.remove(entity_id)
            self.updated_at = datetime.now()


class Concept(BaseModel):
    """
    Represents a concept in the knowledge domain.
    
    Concepts are specialized entities that represent abstract ideas,
    principles, or domain-specific knowledge.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=500)
    definition: str = Field(..., min_length=1, max_length=5000)
    domain: ConceptDomain = Field(default=ConceptDomain.GENERAL)
    synonyms: List[str] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list, description="Related concept IDs")
    parent_concepts: List[str] = Field(default_factory=list, description="Parent concept IDs")
    child_concepts: List[str] = Field(default_factory=list, description="Child concept IDs")
    entity_id: Optional[str] = Field(None, description="Linked entity ID")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('name')
    def validate_name(cls, v):
        """Validate concept name."""
        if not v or not v.strip():
            raise ValueError("Concept name cannot be empty")
        return re.sub(r'\s+', ' ', v.strip())
    
    @validator('definition')
    def validate_definition(cls, v):
        """Validate concept definition."""
        if not v or not v.strip():
            raise ValueError("Concept definition cannot be empty")
        return v.strip()
    
    @validator('synonyms')
    def validate_synonyms(cls, v):
        """Validate synonyms list."""
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("Synonyms must be a list")
        # Remove duplicates and empty strings
        return list(set(s.strip() for s in v if s and s.strip()))
    
    @validator('related_concepts', 'parent_concepts', 'child_concepts')
    def validate_concept_ids(cls, v):
        """Validate concept ID references."""
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("Concept IDs must be a list")
        # Validate UUID format for each ID
        for concept_id in v:
            if not isinstance(concept_id, str):
                raise ValueError("Concept IDs must be strings")
            try:
                uuid.UUID(concept_id)
            except ValueError:
                raise ValueError(f"Invalid UUID format for concept ID: {concept_id}")
        return v
    
    def add_synonym(self, synonym: str) -> None:
        """Add a synonym to the concept."""
        if synonym and synonym.strip() and synonym.strip() not in self.synonyms:
            self.synonyms.append(synonym.strip())
            self.updated_at = datetime.now()
    
    def add_related_concept(self, concept_id: str) -> None:
        """Add a related concept."""
        try:
            uuid.UUID(concept_id)
        except ValueError:
            raise ValueError(f"Invalid UUID format for concept ID: {concept_id}")
        
        if concept_id not in self.related_concepts:
            self.related_concepts.append(concept_id)
            self.updated_at = datetime.now()


# Validation functions for data integrity

def validate_entity_data(entity_data: Dict[str, Any]) -> bool:
    """
    Validate entity data before creating Entity instance.
    
    Args:
        entity_data: Dictionary containing entity data
        
    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    try:
        Entity(**entity_data)
        return True
    except Exception as e:
        raise ValueError(f"Invalid entity data: {str(e)}")


def validate_document_data(document_data: Dict[str, Any]) -> bool:
    """
    Validate document data before creating Document instance.
    
    Args:
        document_data: Dictionary containing document data
        
    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    try:
        Document(**document_data)
        return True
    except Exception as e:
        raise ValueError(f"Invalid document data: {str(e)}")


def validate_concept_data(concept_data: Dict[str, Any]) -> bool:
    """
    Validate concept data before creating Concept instance.
    
    Args:
        concept_data: Dictionary containing concept data
        
    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    try:
        Concept(**concept_data)
        return True
    except Exception as e:
        raise ValueError(f"Invalid concept data: {str(e)}")


def validate_entity_vector_consistency(entity: Entity, vector_id: str) -> bool:
    """
    Validate consistency between entity and its vector representation.
    
    Args:
        entity: Entity instance
        vector_id: Vector ID to validate against
        
    Returns:
        bool: True if consistent
    """
    if entity.vector_id is None and vector_id is None:
        return True
    
    if entity.vector_id is None or vector_id is None:
        return False
    
    return entity.vector_id == vector_id


def validate_document_entity_links(document: Document, entity_ids: List[str]) -> bool:
    """
    Validate that document entity links are consistent.
    
    Args:
        document: Document instance
        entity_ids: List of entity IDs to validate against
        
    Returns:
        bool: True if all links are valid
    """
    if not document.entity_ids:
        return True
    
    # Check if all document entity IDs exist in the provided entity IDs
    return all(entity_id in entity_ids for entity_id in document.entity_ids)


def validate_concept_hierarchy(concept: Concept, all_concept_ids: List[str]) -> bool:
    """
    Validate concept hierarchy relationships.
    
    Args:
        concept: Concept instance
        all_concept_ids: List of all valid concept IDs
        
    Returns:
        bool: True if hierarchy is valid
    """
    # Check parent concepts exist
    for parent_id in concept.parent_concepts:
        if parent_id not in all_concept_ids:
            return False
    
    # Check child concepts exist
    for child_id in concept.child_concepts:
        if child_id not in all_concept_ids:
            return False
    
    # Check related concepts exist
    for related_id in concept.related_concepts:
        if related_id not in all_concept_ids:
            return False
    
    # Check for circular references in hierarchy
    if concept.id in concept.parent_concepts or concept.id in concept.child_concepts:
        return False
    
    return True


class DataIntegrityError(Exception):
    """Exception raised when data integrity validation fails."""
    pass


class ValidationResult(BaseModel):
    """Result of data validation operations."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(warning)


def comprehensive_data_validation(
    entities: List[Entity],
    documents: List[Document],
    concepts: List[Concept]
) -> ValidationResult:
    """
    Perform comprehensive validation across all data models.
    
    Args:
        entities: List of entities to validate
        documents: List of documents to validate
        concepts: List of concepts to validate
        
    Returns:
        ValidationResult: Comprehensive validation result
    """
    result = ValidationResult(is_valid=True)
    
    # Collect all IDs for cross-reference validation
    entity_ids = [entity.id for entity in entities]
    concept_ids = [concept.id for concept in concepts]
    
    # Validate entity-vector consistency
    for entity in entities:
        if entity.vector_id:
            # Check if vector_id format is valid (basic check)
            try:
                uuid.UUID(entity.vector_id)
            except ValueError:
                result.add_error(f"Entity {entity.id} has invalid vector_id format")
    
    # Validate document-entity links
    for document in documents:
        for entity_id in document.entity_ids:
            if entity_id not in entity_ids:
                result.add_error(f"Document {document.id} references non-existent entity {entity_id}")
    
    # Validate concept hierarchies
    for concept in concepts:
        if not validate_concept_hierarchy(concept, concept_ids):
            result.add_error(f"Concept {concept.id} has invalid hierarchy relationships")
    
    # Check for orphaned concepts (concepts with entity_id but no matching entity)
    for concept in concepts:
        if concept.entity_id and concept.entity_id not in entity_ids:
            result.add_warning(f"Concept {concept.id} references non-existent entity {concept.entity_id}")
    
    return result