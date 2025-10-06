"""
Entity-Vector Mapping Service for maintaining consistency between graph and vector databases.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from .database import get_neo4j_manager, get_vector_manager, Neo4jConnectionError

logger = logging.getLogger(__name__)


@dataclass
class EntityVectorLink:
    """Represents a link between a graph entity and vector embedding."""
    entity_id: str
    entity_type: str
    vector_id: str
    collection_name: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityVectorLink':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class MappingValidationError(Exception):
    """Exception raised for mapping validation errors."""
    pass


class EntityVectorMappingService:
    """
    Service for managing bidirectional mapping between graph entities and vector embeddings.
    Maintains data consistency and provides validation methods.
    """
    
    def __init__(self):
        self.neo4j_manager = get_neo4j_manager()
        self.vector_manager = get_vector_manager()
        self._mapping_collection = "entity_vector_mappings"
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the mapping service and create necessary collections."""
        if self._initialized:
            return
        
        try:
            # Create mapping collection in vector database for storing mapping metadata
            self.vector_manager.create_collection(
                name=self._mapping_collection,
                metadata={"description": "Entity-vector mapping metadata"}
            )
            
            # Create constraints and indexes in Neo4j for mapping efficiency
            self._create_neo4j_constraints()
            
            self._initialized = True
            logger.info("Entity-Vector Mapping Service initialized successfully")
            
        except Exception as e:
            raise MappingValidationError(f"Failed to initialize mapping service: {e}")
    
    def _create_neo4j_constraints(self) -> None:
        """Create Neo4j constraints and indexes for efficient mapping queries."""
        constraints_and_indexes = [
            # Ensure unique entity IDs
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            
            # Create index on vector_id property for fast lookups
            "CREATE INDEX entity_vector_id_index IF NOT EXISTS FOR (e:Entity) ON (e.vector_id)",
            
            # Create index on entity type for filtering
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            
            # Create index on updated_at for synchronization queries
            "CREATE INDEX entity_updated_at_index IF NOT EXISTS FOR (e:Entity) ON (e.updated_at)"
        ]
        
        for query in constraints_and_indexes:
            try:
                self.neo4j_manager.execute_query(query)
            except Exception as e:
                logger.warning(f"Failed to create constraint/index: {query}. Error: {e}")
    
    def create_mapping(self, entity_id: str, entity_type: str, vector_id: str, 
                      collection_name: str, metadata: Optional[Dict[str, Any]] = None) -> EntityVectorLink:
        """
        Create a new mapping between a graph entity and vector embedding.
        
        Args:
            entity_id: Unique identifier for the graph entity
            entity_type: Type of the entity (e.g., 'Document', 'Concept', 'Entity')
            vector_id: Unique identifier for the vector embedding
            collection_name: Name of the vector database collection containing the vector
            metadata: Optional metadata for the mapping
            
        Returns:
            EntityVectorLink object representing the created mapping
        """
        if not self._initialized:
            self.initialize()
        
        now = datetime.now()
        link = EntityVectorLink(
            entity_id=entity_id,
            entity_type=entity_type,
            vector_id=vector_id,
            collection_name=collection_name,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        try:
            # Update the graph entity with vector reference
            self._update_entity_vector_reference(entity_id, vector_id, collection_name, now)
            
            # Store mapping metadata in vector database
            self._store_mapping_metadata(link)
            
            logger.info(f"Created mapping: entity {entity_id} -> vector {vector_id}")
            return link
            
        except Exception as e:
            raise MappingValidationError(f"Failed to create mapping for entity {entity_id}: {e}")
    
    def _update_entity_vector_reference(self, entity_id: str, vector_id: str, 
                                      collection_name: str, updated_at: datetime) -> None:
        """Update the graph entity with vector reference information."""
        query = """
        MATCH (e:Entity {id: $entity_id})
        SET e.vector_id = $vector_id,
            e.vector_collection = $collection_name,
            e.updated_at = $updated_at
        RETURN e
        """
        
        result = self.neo4j_manager.execute_query(query, {
            "entity_id": entity_id,
            "vector_id": vector_id,
            "collection_name": collection_name,
            "updated_at": updated_at.isoformat()
        })
        
        if not result:
            raise MappingValidationError(f"Entity {entity_id} not found in graph database")
    
    def _store_mapping_metadata(self, link: EntityVectorLink) -> None:
        """Store mapping metadata in vector database collection."""
        mapping_id = self._generate_mapping_id(link.entity_id, link.vector_id)
        
        self.vector_manager.add_documents(
            collection_name=self._mapping_collection,
            documents=[json.dumps(link.to_dict())],
            metadatas=[{
                "entity_id": link.entity_id,
                "entity_type": link.entity_type,
                "vector_id": link.vector_id,
                "collection_name": link.collection_name,
                "created_at": link.created_at.isoformat(),
                "updated_at": link.updated_at.isoformat()
            }],
            ids=[mapping_id]
        )
    
    def _generate_mapping_id(self, entity_id: str, vector_id: str) -> str:
        """Generate a unique mapping ID from entity and vector IDs."""
        combined = f"{entity_id}:{vector_id}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_vectors_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get all vector embeddings associated with an entity.
        
        Args:
            entity_id: The entity ID to look up
            
        Returns:
            List of dictionaries containing vector information
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Query Neo4j for entity vector references
            query = """
            MATCH (e:Entity {id: $entity_id})
            WHERE e.vector_id IS NOT NULL
            RETURN e.vector_id as vector_id, e.vector_collection as collection_name
            """
            
            result = self.neo4j_manager.execute_query(query, {"entity_id": entity_id})
            
            vectors = []
            for record in result:
                vector_info = {
                    "vector_id": record["vector_id"],
                    "collection_name": record["collection_name"],
                    "entity_id": entity_id
                }
                vectors.append(vector_info)
            
            return vectors
            
        except Exception as e:
            raise MappingValidationError(f"Failed to get vectors for entity {entity_id}: {e}")
    
    def get_entities_for_vector(self, vector_id: str, collection_name: str) -> List[Dict[str, Any]]:
        """
        Get all entities associated with a vector embedding.
        
        Args:
            vector_id: The vector ID to look up
            collection_name: The collection containing the vector
            
        Returns:
            List of dictionaries containing entity information
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Query Neo4j for entities with this vector reference
            query = """
            MATCH (e:Entity)
            WHERE e.vector_id = $vector_id AND e.vector_collection = $collection_name
            RETURN e.id as entity_id, e.type as entity_type, e.name as entity_name
            """
            
            result = self.neo4j_manager.execute_query(query, {
                "vector_id": vector_id,
                "collection_name": collection_name
            })
            
            entities = []
            for record in result:
                entity_info = {
                    "entity_id": record["entity_id"],
                    "entity_type": record["entity_type"],
                    "entity_name": record.get("entity_name"),
                    "vector_id": vector_id,
                    "collection_name": collection_name
                }
                entities.append(entity_info)
            
            return entities
            
        except Exception as e:
            raise MappingValidationError(f"Failed to get entities for vector {vector_id}: {e}")
    
    def update_mapping(self, entity_id: str, vector_id: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> EntityVectorLink:
        """
        Update an existing mapping with new metadata.
        
        Args:
            entity_id: The entity ID
            vector_id: The vector ID
            metadata: New metadata to update
            
        Returns:
            Updated EntityVectorLink object
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Get existing mapping
            existing_mappings = self.get_mapping_details(entity_id, vector_id)
            if not existing_mappings:
                raise MappingValidationError(f"No mapping found for entity {entity_id} and vector {vector_id}")
            
            existing_link = existing_mappings[0]
            
            # Update metadata
            if metadata:
                existing_link.metadata.update(metadata)
            existing_link.updated_at = datetime.now()
            
            # Update in storage
            self._update_entity_vector_reference(
                entity_id, vector_id, existing_link.collection_name, existing_link.updated_at
            )
            self._store_mapping_metadata(existing_link)
            
            logger.info(f"Updated mapping: entity {entity_id} -> vector {vector_id}")
            return existing_link
            
        except Exception as e:
            raise MappingValidationError(f"Failed to update mapping for entity {entity_id}: {e}")
    
    def delete_mapping(self, entity_id: str, vector_id: str) -> None:
        """
        Delete a mapping between an entity and vector.
        
        Args:
            entity_id: The entity ID
            vector_id: The vector ID
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Remove vector reference from graph entity
            query = """
            MATCH (e:Entity {id: $entity_id})
            WHERE e.vector_id = $vector_id
            REMOVE e.vector_id, e.vector_collection
            SET e.updated_at = $updated_at
            RETURN e
            """
            
            result = self.neo4j_manager.execute_query(query, {
                "entity_id": entity_id,
                "vector_id": vector_id,
                "updated_at": datetime.now().isoformat()
            })
            
            if not result:
                logger.warning(f"No entity found with ID {entity_id} and vector {vector_id}")
            
            # Remove mapping metadata from vector database
            mapping_id = self._generate_mapping_id(entity_id, vector_id)
            try:
                self.vector_manager.delete_documents([mapping_id])
            except Exception as e:
                logger.warning(f"Failed to delete mapping metadata: {e}")
            
            logger.info(f"Deleted mapping: entity {entity_id} -> vector {vector_id}")
            
        except Exception as e:
            raise MappingValidationError(f"Failed to delete mapping for entity {entity_id}: {e}")
    
    def get_mapping_details(self, entity_id: str, vector_id: str) -> List[EntityVectorLink]:
        """
        Get detailed information about a specific mapping.
        
        Args:
            entity_id: The entity ID
            vector_id: The vector ID
            
        Returns:
            List of EntityVectorLink objects (should be 0 or 1)
        """
        if not self._initialized:
            self.initialize()
        
        try:
            mapping_id = self._generate_mapping_id(entity_id, vector_id)
            
            # Query mapping metadata from vector database
            results = self.vector_manager.query_collection(
                collection_name=self._mapping_collection,
                query_texts=[f"entity_id:{entity_id} vector_id:{vector_id}"],
                n_results=1,
                where={"entity_id": entity_id, "vector_id": vector_id}
            )
            
            links = []
            if results.get("documents") and results["documents"][0]:
                for doc in results["documents"][0]:
                    try:
                        link_data = json.loads(doc)
                        links.append(EntityVectorLink.from_dict(link_data))
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse mapping data: {e}")
            
            return links
            
        except Exception as e:
            raise MappingValidationError(f"Failed to get mapping details: {e}")
    
    def validate_mapping_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of all entity-vector mappings.
        
        Returns:
            Dictionary containing validation results and any issues found
        """
        if not self._initialized:
            self.initialize()
        
        validation_results = {
            "total_entities": 0,
            "entities_with_vectors": 0,
            "orphaned_entities": [],
            "orphaned_vectors": [],
            "inconsistent_mappings": [],
            "validation_passed": True
        }
        
        try:
            # Get all entities with vector references from Neo4j
            query = """
            MATCH (e:Entity)
            RETURN e.id as entity_id, e.vector_id as vector_id, e.vector_collection as collection_name
            """
            
            entities = self.neo4j_manager.execute_query(query)
            validation_results["total_entities"] = len(entities)
            
            for entity in entities:
                entity_id = entity["entity_id"]
                vector_id = entity.get("vector_id")
                collection_name = entity.get("vector_collection")
                
                if vector_id and collection_name:
                    validation_results["entities_with_vectors"] += 1
                    
                    # Check if vector exists in vector database
                    try:
                        # For Pinecone, we'll use a different approach to check vector existence
                        collection = getattr(self.vector_manager, 'get_collection', lambda x: None)(collection_name)
                        # Try to get the specific vector
                        results = collection.get(ids=[vector_id])
                        
                        if not results.get("ids") or vector_id not in results["ids"]:
                            validation_results["orphaned_entities"].append({
                                "entity_id": entity_id,
                                "vector_id": vector_id,
                                "collection_name": collection_name,
                                "issue": "Vector not found in vector database"
                            })
                            validation_results["validation_passed"] = False
                            
                    except Exception as e:
                        validation_results["inconsistent_mappings"].append({
                            "entity_id": entity_id,
                            "vector_id": vector_id,
                            "collection_name": collection_name,
                            "issue": f"Error accessing vector: {e}"
                        })
                        validation_results["validation_passed"] = False
            
            logger.info(f"Mapping validation completed. Found {len(validation_results['orphaned_entities'])} orphaned entities")
            return validation_results
            
        except Exception as e:
            validation_results["validation_passed"] = False
            validation_results["error"] = str(e)
            raise MappingValidationError(f"Failed to validate mapping integrity: {e}")
    
    def synchronize_mappings(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Synchronize mappings between graph and vector databases.
        
        Args:
            dry_run: If True, only report what would be changed without making changes
            
        Returns:
            Dictionary containing synchronization results
        """
        if not self._initialized:
            self.initialize()
        
        sync_results = {
            "entities_processed": 0,
            "mappings_created": 0,
            "mappings_updated": 0,
            "mappings_deleted": 0,
            "errors": [],
            "dry_run": dry_run
        }
        
        try:
            validation_results = self.validate_mapping_integrity()
            
            # Handle orphaned entities (entities pointing to non-existent vectors)
            for orphaned in validation_results["orphaned_entities"]:
                entity_id = orphaned["entity_id"]
                vector_id = orphaned["vector_id"]
                
                if not dry_run:
                    try:
                        # Remove the invalid vector reference
                        query = """
                        MATCH (e:Entity {id: $entity_id})
                        REMOVE e.vector_id, e.vector_collection
                        SET e.updated_at = $updated_at
                        """
                        
                        self.neo4j_manager.execute_query(query, {
                            "entity_id": entity_id,
                            "updated_at": datetime.now().isoformat()
                        })
                        
                        sync_results["mappings_deleted"] += 1
                        
                    except Exception as e:
                        sync_results["errors"].append(f"Failed to clean orphaned entity {entity_id}: {e}")
                else:
                    sync_results["mappings_deleted"] += 1
            
            sync_results["entities_processed"] = validation_results["total_entities"]
            
            logger.info(f"Mapping synchronization completed ({'dry run' if dry_run else 'actual'})")
            return sync_results
            
        except Exception as e:
            sync_results["errors"].append(str(e))
            raise MappingValidationError(f"Failed to synchronize mappings: {e}")
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current mappings.
        
        Returns:
            Dictionary containing mapping statistics
        """
        if not self._initialized:
            self.initialize()
        
        try:
            stats = {}
            
            # Get entity statistics from Neo4j
            entity_query = """
            MATCH (e:Entity)
            OPTIONAL MATCH (e) WHERE e.vector_id IS NOT NULL
            RETURN 
                count(e) as total_entities,
                count(e.vector_id) as entities_with_vectors,
                collect(DISTINCT e.type) as entity_types
            """
            
            entity_result = self.neo4j_manager.execute_query(entity_query)
            if entity_result:
                stats.update(entity_result[0])
            
            # Get collection statistics from vector database
            collections = getattr(self.vector_manager, 'list_collections', lambda: [])() or []
            collection_stats = {}
            
            for collection_name in collections:
                if collection_name != self._mapping_collection:
                    try:
                        collection_info = getattr(self.vector_manager, 'get_collection_stats', lambda x: {})(collection_name)
                        collection_stats[collection_name] = collection_info
                    except Exception as e:
                        logger.warning(f"Failed to get stats for collection {collection_name}: {e}")
            
            stats["collections"] = collection_stats
            stats["total_collections"] = len(collections) - 1  # Exclude mapping collection
            
            return stats
            
        except Exception as e:
            raise MappingValidationError(f"Failed to get mapping statistics: {e}")


# Global mapping service instance
_mapping_service: Optional[EntityVectorMappingService] = None


def get_mapping_service() -> EntityVectorMappingService:
    """Get the global entity-vector mapping service instance."""
    global _mapping_service
    
    if _mapping_service is None:
        _mapping_service = EntityVectorMappingService()
    
    return _mapping_service


def initialize_mapping_service() -> None:
    """Initialize the entity-vector mapping service."""
    service = get_mapping_service()
    service.initialize()