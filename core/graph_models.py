"""
Graph schema models for Neo4j integration in the Graph-Enhanced Agentic RAG system.

This module defines Neo4j node and relationship models, Cypher query builders,
and graph entity validation and constraints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid

from .models import Entity, Document, Concept


class NodeType(str, Enum):
    """Types of nodes in the Neo4j graph."""
    ENTITY = "Entity"
    DOCUMENT = "Document"
    CONCEPT = "Concept"
    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    TECHNOLOGY = "Technology"
    EVENT = "Event"


class RelationshipType(str, Enum):
    """Types of relationships in the Neo4j graph."""
    RELATED_TO = "RELATED_TO"
    MENTIONED_IN = "MENTIONED_IN"
    PART_OF = "PART_OF"
    REFERENCES = "REFERENCES"
    CONTAINS = "CONTAINS"
    DEPENDS_ON = "DEPENDS_ON"
    SIMILAR_TO = "SIMILAR_TO"
    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"
    AUTHORED_BY = "AUTHORED_BY"
    LOCATED_IN = "LOCATED_IN"
    WORKS_FOR = "WORKS_FOR"


class GraphNode(BaseModel):
    """
    Base class for Neo4j graph nodes.
    
    Represents a node in the Neo4j graph database with common properties
    and methods for all node types.
    """
    id: str = Field(..., description="Unique identifier for the node")
    labels: List[str] = Field(default_factory=list, description="Neo4j node labels")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('id')
    def validate_id(cls, v):
        """Validate node ID format."""
        if not v or not v.strip():
            raise ValueError("Node ID cannot be empty")
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError(f"Invalid UUID format for node ID: {v}")
        return v
    
    @validator('labels')
    def validate_labels(cls, v):
        """Validate node labels."""
        if not v:
            return []
        if not isinstance(v, list):
            raise ValueError("Labels must be a list")
        # Ensure all labels are valid strings
        validated_labels = []
        for label in v:
            if not isinstance(label, str) or not label.strip():
                raise ValueError("All labels must be non-empty strings")
            validated_labels.append(label.strip())
        return validated_labels
    
    def add_label(self, label: str) -> None:
        """Add a label to the node."""
        if label and label.strip() and label not in self.labels:
            self.labels.append(label.strip())
            self.updated_at = datetime.now()
    
    def remove_label(self, label: str) -> None:
        """Remove a label from the node."""
        if label in self.labels:
            self.labels.remove(label)
            self.updated_at = datetime.now()
    
    def set_property(self, key: str, value: Any) -> None:
        """Set a property on the node."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Property key must be a non-empty string")
        self.properties[key] = value
        self.updated_at = datetime.now()
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property from the node."""
        return self.properties.get(key, default)
    
    def remove_property(self, key: str) -> None:
        """Remove a property from the node."""
        if key in self.properties:
            del self.properties[key]
            self.updated_at = datetime.now()


class EntityNode(GraphNode):
    """
    Neo4j node representation for Entity objects.
    
    Extends GraphNode with entity-specific properties and methods.
    """
    name: str = Field(..., min_length=1, max_length=500)
    entity_type: str = Field(..., description="Type of entity")
    description: Optional[str] = Field(None, max_length=2000)
    vector_id: Optional[str] = Field(None, description="Link to vector embedding")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure Entity label is present
        if NodeType.ENTITY not in self.labels:
            self.labels.append(NodeType.ENTITY)
        # Add specific entity type as label if provided
        if self.entity_type and self.entity_type not in self.labels:
            self.labels.append(self.entity_type)
    
    @classmethod
    def from_entity(cls, entity: Entity) -> 'EntityNode':
        """Create EntityNode from Entity model."""
        return cls(
            id=entity.id,
            name=entity.name,
            entity_type=entity.type.value,
            description=entity.description,
            vector_id=entity.vector_id,
            properties=entity.properties.copy(),
            created_at=entity.created_at,
            updated_at=entity.updated_at
        )
    
    def to_entity(self) -> Entity:
        """Convert EntityNode back to Entity model."""
        from .models import EntityType
        return Entity(
            id=self.id,
            name=self.name,
            type=EntityType(self.entity_type),
            description=self.description,
            properties=self.properties.copy(),
            vector_id=self.vector_id,
            created_at=self.created_at,
            updated_at=self.updated_at
        )


class DocumentNode(GraphNode):
    """
    Neo4j node representation for Document objects.
    
    Extends GraphNode with document-specific properties and methods.
    """
    title: str = Field(..., min_length=1, max_length=1000)
    content: str = Field(..., min_length=1)
    document_type: str = Field(..., description="Type of document")
    source: Optional[str] = Field(None, max_length=2000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure Document label is present
        if NodeType.DOCUMENT not in self.labels:
            self.labels.append(NodeType.DOCUMENT)
    
    @classmethod
    def from_document(cls, document: Document) -> 'DocumentNode':
        """Create DocumentNode from Document model."""
        return cls(
            id=document.id,
            title=document.title,
            content=document.content,
            document_type=document.document_type.value,
            source=document.source,
            metadata=document.metadata.copy(),
            created_at=document.created_at,
            updated_at=document.updated_at
        )
    
    def to_document(self) -> Document:
        """Convert DocumentNode back to Document model."""
        from .models import DocumentType
        return Document(
            id=self.id,
            title=self.title,
            content=self.content,
            document_type=DocumentType(self.document_type),
            source=self.source,
            metadata=self.metadata.copy(),
            created_at=self.created_at,
            updated_at=self.updated_at
        )


class ConceptNode(GraphNode):
    """
    Neo4j node representation for Concept objects.
    
    Extends GraphNode with concept-specific properties and methods.
    """
    name: str = Field(..., min_length=1, max_length=500)
    definition: str = Field(..., min_length=1, max_length=5000)
    domain: str = Field(..., description="Domain of the concept")
    synonyms: List[str] = Field(default_factory=list)
    entity_id: Optional[str] = Field(None, description="Linked entity ID")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure Concept label is present
        if NodeType.CONCEPT not in self.labels:
            self.labels.append(NodeType.CONCEPT)
    
    @classmethod
    def from_concept(cls, concept: Concept) -> 'ConceptNode':
        """Create ConceptNode from Concept model."""
        return cls(
            id=concept.id,
            name=concept.name,
            definition=concept.definition,
            domain=concept.domain.value,
            synonyms=concept.synonyms.copy(),
            entity_id=concept.entity_id,
            created_at=concept.created_at,
            updated_at=concept.updated_at
        )
    
    def to_concept(self) -> Concept:
        """Convert ConceptNode back to Concept model."""
        from .models import ConceptDomain
        return Concept(
            id=self.id,
            name=self.name,
            definition=self.definition,
            domain=ConceptDomain(self.domain),
            synonyms=self.synonyms.copy(),
            entity_id=self.entity_id,
            created_at=self.created_at,
            updated_at=self.updated_at
        )


class GraphRelationship(BaseModel):
    """
    Represents a relationship between two nodes in the Neo4j graph.
    
    Contains relationship type, properties, and references to source and target nodes.
    """
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    source_node_id: str = Field(..., description="ID of the source node")
    target_node_id: str = Field(..., description="ID of the target node")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('source_node_id', 'target_node_id')
    def validate_node_ids(cls, v):
        """Validate node ID format."""
        if not v or not v.strip():
            raise ValueError("Node ID cannot be empty")
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError(f"Invalid UUID format for node ID: {v}")
        return v
    
    def set_property(self, key: str, value: Any) -> None:
        """Set a property on the relationship."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Property key must be a non-empty string")
        self.properties[key] = value
        self.updated_at = datetime.now()
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property from the relationship."""
        return self.properties.get(key, default)


class CypherQueryBuilder:
    """
    Builder class for generating Cypher queries for common graph operations.
    
    Provides methods to build CREATE, MATCH, UPDATE, and DELETE queries
    with proper parameterization and validation.
    """
    
    def __init__(self):
        self.query_parts = []
        self.parameters = {}
    
    def reset(self) -> 'CypherQueryBuilder':
        """Reset the query builder for a new query."""
        self.query_parts = []
        self.parameters = {}
        return self
    
    def create_node(self, node: GraphNode, variable_name: str = "n") -> 'CypherQueryBuilder':
        """
        Generate CREATE query for a node.
        
        Args:
            node: GraphNode instance to create
            variable_name: Variable name for the node in the query
            
        Returns:
            CypherQueryBuilder: Self for method chaining
        """
        labels_str = ":".join(node.labels) if node.labels else ""
        if labels_str:
            labels_str = ":" + labels_str
        
        self.query_parts.append(f"CREATE ({variable_name}{labels_str} $node_props)")
        self.parameters["node_props"] = {
            "id": node.id,
            **node.properties,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat()
        }
        
        # Add specific properties based on node type
        if isinstance(node, EntityNode):
            self.parameters["node_props"].update({
                "name": node.name,
                "entity_type": node.entity_type,
                "description": node.description,
                "vector_id": node.vector_id
            })
        elif isinstance(node, DocumentNode):
            self.parameters["node_props"].update({
                "title": node.title,
                "content": node.content,
                "document_type": node.document_type,
                "source": node.source,
                "metadata": node.metadata
            })
        elif isinstance(node, ConceptNode):
            self.parameters["node_props"].update({
                "name": node.name,
                "definition": node.definition,
                "domain": node.domain,
                "synonyms": node.synonyms,
                "entity_id": node.entity_id
            })
        
        return self
    
    def match_node_by_id(self, node_id: str, labels: List[str] = None, variable_name: str = "n") -> 'CypherQueryBuilder':
        """
        Generate MATCH query for a node by ID.
        
        Args:
            node_id: ID of the node to match
            labels: Optional labels to filter by
            variable_name: Variable name for the node in the query
            
        Returns:
            CypherQueryBuilder: Self for method chaining
        """
        labels_str = ":".join(labels) if labels else ""
        if labels_str:
            labels_str = ":" + labels_str
        
        self.query_parts.append(f"MATCH ({variable_name}{labels_str} {{id: $node_id}})")
        self.parameters["node_id"] = node_id
        return self
    
    def match_nodes_by_property(self, property_name: str, property_value: Any, 
                               labels: List[str] = None, variable_name: str = "n") -> 'CypherQueryBuilder':
        """
        Generate MATCH query for nodes by property.
        
        Args:
            property_name: Name of the property to match
            property_value: Value of the property to match
            labels: Optional labels to filter by
            variable_name: Variable name for the nodes in the query
            
        Returns:
            CypherQueryBuilder: Self for method chaining
        """
        labels_str = ":".join(labels) if labels else ""
        if labels_str:
            labels_str = ":" + labels_str
        
        param_name = f"{property_name}_value"
        self.query_parts.append(f"MATCH ({variable_name}{labels_str} {{{property_name}: ${param_name}}})")
        self.parameters[param_name] = property_value
        return self
    
    def create_relationship(self, relationship: GraphRelationship, 
                          source_var: str = "a", target_var: str = "b") -> 'CypherQueryBuilder':
        """
        Generate CREATE query for a relationship.
        
        Args:
            relationship: GraphRelationship instance to create
            source_var: Variable name for source node
            target_var: Variable name for target node
            
        Returns:
            CypherQueryBuilder: Self for method chaining
        """
        rel_type = relationship.relationship_type.value
        self.query_parts.append(f"CREATE ({source_var})-[r:{rel_type} $rel_props]->({target_var})")
        self.parameters["rel_props"] = {
            **relationship.properties,
            "created_at": relationship.created_at.isoformat(),
            "updated_at": relationship.updated_at.isoformat()
        }
        return self
    
    def match_relationship(self, source_id: str, target_id: str, 
                          relationship_type: RelationshipType = None,
                          source_var: str = "a", target_var: str = "b", 
                          rel_var: str = "r") -> 'CypherQueryBuilder':
        """
        Generate MATCH query for a relationship.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relationship_type: Optional relationship type to filter by
            source_var: Variable name for source node
            target_var: Variable name for target node
            rel_var: Variable name for relationship
            
        Returns:
            CypherQueryBuilder: Self for method chaining
        """
        rel_type_str = f":{relationship_type.value}" if relationship_type else ""
        
        self.query_parts.extend([
            f"MATCH ({source_var} {{id: $source_id}})",
            f"MATCH ({target_var} {{id: $target_id}})",
            f"MATCH ({source_var})-[{rel_var}{rel_type_str}]->({target_var})"
        ])
        self.parameters.update({
            "source_id": source_id,
            "target_id": target_id
        })
        return self
    
    def traverse_relationships(self, start_node_id: str, relationship_types: List[RelationshipType] = None,
                             max_depth: int = 3, direction: str = "both") -> 'CypherQueryBuilder':
        """
        Generate MATCH query for relationship traversal.
        
        Args:
            start_node_id: ID of the starting node
            relationship_types: Optional list of relationship types to traverse
            max_depth: Maximum traversal depth
            direction: Direction of traversal ("outgoing", "incoming", "both")
            
        Returns:
            CypherQueryBuilder: Self for method chaining
        """
        # Build relationship type filter
        rel_types_str = ""
        if relationship_types:
            types = "|".join([rt.value for rt in relationship_types])
            rel_types_str = f":{types}"
        
        # Build direction pattern
        if direction == "outgoing":
            pattern = f"-[{rel_types_str}*1..{max_depth}]->"
        elif direction == "incoming":
            pattern = f"<-[{rel_types_str}*1..{max_depth}]-"
        else:  # both
            pattern = f"-[{rel_types_str}*1..{max_depth}]-"
        
        self.query_parts.append(f"MATCH path = (start {{id: $start_id}}){pattern}(end)")
        self.parameters["start_id"] = start_node_id
        return self
    
    def where(self, condition: str, **params) -> 'CypherQueryBuilder':
        """
        Add WHERE clause to the query.
        
        Args:
            condition: WHERE condition string
            **params: Parameters for the condition
            
        Returns:
            CypherQueryBuilder: Self for method chaining
        """
        self.query_parts.append(f"WHERE {condition}")
        self.parameters.update(params)
        return self
    
    def return_clause(self, *items) -> 'CypherQueryBuilder':
        """
        Add RETURN clause to the query.
        
        Args:
            *items: Items to return
            
        Returns:
            CypherQueryBuilder: Self for method chaining
        """
        return_str = ", ".join(items)
        self.query_parts.append(f"RETURN {return_str}")
        return self
    
    def limit(self, count: int) -> 'CypherQueryBuilder':
        """
        Add LIMIT clause to the query.
        
        Args:
            count: Maximum number of results
            
        Returns:
            CypherQueryBuilder: Self for method chaining
        """
        self.query_parts.append(f"LIMIT {count}")
        return self
    
    def order_by(self, *items) -> 'CypherQueryBuilder':
        """
        Add ORDER BY clause to the query.
        
        Args:
            *items: Items to order by
            
        Returns:
            CypherQueryBuilder: Self for method chaining
        """
        order_str = ", ".join(items)
        self.query_parts.append(f"ORDER BY {order_str}")
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """
        Build the final Cypher query and parameters.
        
        Returns:
            Tuple[str, Dict[str, Any]]: Query string and parameters
        """
        query = "\n".join(self.query_parts)
        return query, self.parameters.copy()


class GraphConstraints:
    """
    Defines Neo4j constraints and indexes for the graph schema.
    
    Provides methods to create and manage database constraints
    for data integrity and performance optimization.
    """
    
    @staticmethod
    def get_constraint_queries() -> List[str]:
        """
        Get list of Cypher queries to create all necessary constraints.
        
        Returns:
            List[str]: List of constraint creation queries
        """
        constraints = [
            # Unique constraints for node IDs
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            
            # Node existence constraints
            "CREATE CONSTRAINT entity_name_exists IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS NOT NULL",
            "CREATE CONSTRAINT document_title_exists IF NOT EXISTS FOR (d:Document) REQUIRE d.title IS NOT NULL",
            "CREATE CONSTRAINT concept_name_exists IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS NOT NULL",
            
            # Property constraints
            "CREATE CONSTRAINT entity_type_exists IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_type IS NOT NULL",
            "CREATE CONSTRAINT document_type_exists IF NOT EXISTS FOR (d:Document) REQUIRE d.document_type IS NOT NULL",
            "CREATE CONSTRAINT concept_domain_exists IF NOT EXISTS FOR (c:Concept) REQUIRE c.domain IS NOT NULL",
        ]
        return constraints
    
    @staticmethod
    def get_index_queries() -> List[str]:
        """
        Get list of Cypher queries to create all necessary indexes.
        
        Returns:
            List[str]: List of index creation queries
        """
        indexes = [
            # Text search indexes
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            
            # Type-based indexes
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX document_type_index IF NOT EXISTS FOR (d:Document) ON (d.document_type)",
            "CREATE INDEX concept_domain_index IF NOT EXISTS FOR (c:Concept) ON (c.domain)",
            
            # Vector ID index for entity-vector mapping
            "CREATE INDEX entity_vector_id_index IF NOT EXISTS FOR (e:Entity) ON (e.vector_id)",
            
            # Timestamp indexes for temporal queries
            "CREATE INDEX entity_created_index IF NOT EXISTS FOR (e:Entity) ON (e.created_at)",
            "CREATE INDEX document_created_index IF NOT EXISTS FOR (d:Document) ON (d.created_at)",
            "CREATE INDEX concept_created_index IF NOT EXISTS FOR (c:Concept) ON (c.created_at)",
        ]
        return indexes


class GraphValidator:
    """
    Validates graph entities and relationships for consistency and integrity.
    
    Provides methods to validate nodes, relationships, and overall graph structure.
    """
    
    @staticmethod
    def validate_node(node: GraphNode) -> List[str]:
        """
        Validate a graph node for consistency and integrity.
        
        Args:
            node: GraphNode instance to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate basic node structure
        if not node.id:
            errors.append("Node ID is required")
        else:
            try:
                uuid.UUID(node.id)
            except ValueError:
                errors.append(f"Invalid UUID format for node ID: {node.id}")
        
        if not node.labels:
            errors.append("Node must have at least one label")
        
        # Validate specific node types
        if isinstance(node, EntityNode):
            if not node.name or not node.name.strip():
                errors.append("Entity node must have a name")
            if not node.entity_type:
                errors.append("Entity node must have an entity_type")
            if node.vector_id:
                try:
                    uuid.UUID(node.vector_id)
                except ValueError:
                    errors.append(f"Invalid UUID format for vector_id: {node.vector_id}")
        
        elif isinstance(node, DocumentNode):
            if not node.title or not node.title.strip():
                errors.append("Document node must have a title")
            if not node.content or not node.content.strip():
                errors.append("Document node must have content")
            if not node.document_type:
                errors.append("Document node must have a document_type")
        
        elif isinstance(node, ConceptNode):
            if not node.name or not node.name.strip():
                errors.append("Concept node must have a name")
            if not node.definition or not node.definition.strip():
                errors.append("Concept node must have a definition")
            if not node.domain:
                errors.append("Concept node must have a domain")
            if node.entity_id:
                try:
                    uuid.UUID(node.entity_id)
                except ValueError:
                    errors.append(f"Invalid UUID format for entity_id: {node.entity_id}")
        
        return errors
    
    @staticmethod
    def validate_relationship(relationship: GraphRelationship) -> List[str]:
        """
        Validate a graph relationship for consistency and integrity.
        
        Args:
            relationship: GraphRelationship instance to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate relationship structure
        if not relationship.source_node_id:
            errors.append("Relationship must have a source_node_id")
        else:
            try:
                uuid.UUID(relationship.source_node_id)
            except ValueError:
                errors.append(f"Invalid UUID format for source_node_id: {relationship.source_node_id}")
        
        if not relationship.target_node_id:
            errors.append("Relationship must have a target_node_id")
        else:
            try:
                uuid.UUID(relationship.target_node_id)
            except ValueError:
                errors.append(f"Invalid UUID format for target_node_id: {relationship.target_node_id}")
        
        if relationship.source_node_id == relationship.target_node_id:
            errors.append("Relationship cannot have the same source and target node")
        
        if not relationship.relationship_type:
            errors.append("Relationship must have a relationship_type")
        
        return errors
    
    @staticmethod
    def validate_entity_vector_mapping(entity_node: EntityNode, vector_id: str) -> List[str]:
        """
        Validate consistency between entity node and vector ID.
        
        Args:
            entity_node: EntityNode instance
            vector_id: Vector ID to validate against
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        if entity_node.vector_id and vector_id:
            if entity_node.vector_id != vector_id:
                errors.append(f"Entity vector_id mismatch: {entity_node.vector_id} != {vector_id}")
        elif entity_node.vector_id and not vector_id:
            errors.append("Entity has vector_id but no corresponding vector found")
        elif not entity_node.vector_id and vector_id:
            errors.append("Vector exists but entity has no vector_id reference")
        
        return errors


# Common query patterns for the Graph-Enhanced Agentic RAG system

class CommonQueries:
    """
    Pre-built common Cypher queries for the Graph-Enhanced Agentic RAG system.
    
    Provides static methods for frequently used query patterns.
    """
    
    @staticmethod
    def find_entity_by_name(name: str) -> Tuple[str, Dict[str, Any]]:
        """Find entity by name with fuzzy matching."""
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        RETURN e
        ORDER BY e.name
        LIMIT 10
        """
        return query, {"name": name}
    
    @staticmethod
    def find_related_entities(entity_id: str, max_depth: int = 2) -> Tuple[str, Dict[str, Any]]:
        """Find entities related to a given entity."""
        query = f"""
        MATCH (start:Entity {{id: $entity_id}})
        MATCH path = (start)-[*1..{max_depth}]-(related:Entity)
        RETURN DISTINCT related, length(path) as distance
        ORDER BY distance, related.name
        LIMIT 20
        """
        return query, {"entity_id": entity_id}
    
    @staticmethod
    def find_documents_mentioning_entity(entity_id: str) -> Tuple[str, Dict[str, Any]]:
        """Find documents that mention a specific entity."""
        query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (e)-[:MENTIONED_IN]->(d:Document)
        RETURN d
        ORDER BY d.created_at DESC
        LIMIT 10
        """
        return query, {"entity_id": entity_id}
    
    @staticmethod
    def find_concept_hierarchy(concept_id: str) -> Tuple[str, Dict[str, Any]]:
        """Find concept hierarchy (parents and children)."""
        query = """
        MATCH (c:Concept {id: $concept_id})
        OPTIONAL MATCH (c)-[:CHILD_OF]->(parent:Concept)
        OPTIONAL MATCH (child:Concept)-[:CHILD_OF]->(c)
        RETURN c, collect(DISTINCT parent) as parents, collect(DISTINCT child) as children
        """
        return query, {"concept_id": concept_id}
    
    @staticmethod
    def find_shortest_path(source_id: str, target_id: str) -> Tuple[str, Dict[str, Any]]:
        """Find shortest path between two entities."""
        query = """
        MATCH (source {id: $source_id}), (target {id: $target_id})
        MATCH path = shortestPath((source)-[*..6]-(target))
        RETURN path, length(path) as path_length
        """
        return query, {"source_id": source_id, "target_id": target_id}