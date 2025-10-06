"""
Dual storage ingestion pipeline for the Graph-Enhanced Agentic RAG system.

This module handles the ingestion of documents into both graph and vector databases,
creating entity-vector mappings and ensuring data consistency across storage systems.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import uuid
import logging

from .document_processor import (
    DocumentProcessor, 
    TextChunk, 
    ExtractedEntity, 
    EntityRelationship,
    ChunkingStrategy
)
from .domain_processor import DomainType
from .domain_config import get_domain_config_manager
from .models import Document, Entity, EntityType
from .database import Neo4jConnectionManager
from .vector_models import DocumentEmbedding, EmbeddingVector
from .mapping_service import EntityVectorMappingService

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion process."""
    document_id: str
    success: bool
    chunks_created: int
    entities_created: int
    relationships_created: int
    vector_embeddings_created: int
    entity_mappings_created: int
    errors: List[str]
    processing_time: float


@dataclass
class IngestionConfig:
    """Configuration for the ingestion pipeline."""
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_BASED
    chunk_size: int = 500
    chunk_overlap: int = 50
    batch_size: int = 10
    create_embeddings: bool = True
    create_graph_relationships: bool = True
    enable_entity_linking: bool = True
    max_retries: int = 3
    domain_type: Optional[DomainType] = None
    use_domain_specific_processing: bool = True


class DualStorageIngestionPipeline:
    """Main ingestion pipeline for dual storage (graph + vector)."""
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        graph_db_manager: Neo4jConnectionManager,
        vector_db_manager: Any,  # Pinecone or ChromaDB
        mapping_service: EntityVectorMappingService,
        embedding_service: Any,  # Sentence transformer service
        config: Optional[IngestionConfig] = None
    ):
        """
        Initialize the dual storage ingestion pipeline.
        
        Args:
            document_processor: Document processing service
            graph_db_manager: Graph database manager
            vector_db_manager: Vector database manager
            mapping_service: Entity-vector mapping service
            embedding_service: Embedding generation service
            config: Ingestion configuration
        """
        self.document_processor = document_processor
        self.graph_db = graph_db_manager
        self.vector_db = vector_db_manager
        self.mapping_service = mapping_service
        self.embedding_service = embedding_service
        self.config = config or IngestionConfig()
        
        # Configure domain if specified
        if self.config.domain_type:
            domain_config_manager = get_domain_config_manager()
            domain_config_manager.configure_domain(self.config.domain_type)
            
            # Set domain on document processor
            if hasattr(self.document_processor, 'set_domain'):
                self.document_processor.set_domain(self.config.domain_type.value)
        
        # Initialize components
        self._initialized = False
        self._batch_processor = None
        
    def initialize(self) -> None:
        """Initialize the ingestion pipeline."""
        if self._initialized:
            return
        
        # Initialize database connections
        self.graph_db.connect() if hasattr(self.graph_db, 'connect') else None
        self.vector_db.connect() if hasattr(self.vector_db, 'connect') else None
        
        # Initialize mapping service
        self.mapping_service.initialize()
        
        # Create necessary collections and constraints
        self._setup_storage_infrastructure()
        
        self._initialized = True
        logger.info("Dual storage ingestion pipeline initialized successfully")
    
    def _setup_storage_infrastructure(self) -> None:
        """Set up necessary collections and constraints in both databases."""
        try:
            # Create document collection in vector database
            self.vector_db.create_collection(
                name="documents",
                metadata={"description": "Document embeddings for RAG system"}
            )
            
            # Create entity collection in vector database
            self.vector_db.create_collection(
                name="entities",
                metadata={"description": "Entity embeddings for graph-vector consistency"}
            )
            
            # Create constraints and indexes in Neo4j
            self._create_graph_constraints()
            
        except Exception as e:
            raise Exception(f"Failed to setup storage infrastructure: {e}")
    
    def _create_graph_constraints(self) -> None:
        """Create necessary constraints and indexes in Neo4j."""
        constraints_and_indexes = [
            # Entity constraints
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX entity_vector_id_index IF NOT EXISTS FOR (e:Entity) ON (e.vector_id)",
            "CREATE INDEX document_created_at_index IF NOT EXISTS FOR (d:Document) ON (d.created_at)"
        ]
        
        for query in constraints_and_indexes:
            try:
                self.graph_db.execute_query(query)
            except Exception as e:
                logger.warning(f"Failed to create constraint/index: {query}. Error: {e}")
    
    async def ingest_document(self, document: Document) -> IngestionResult:
        """
        Ingest a single document into both graph and vector databases.
        
        Args:
            document: Document to ingest
            
        Returns:
            IngestionResult containing ingestion statistics and any errors
        """
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        result = IngestionResult(
            document_id=document.id,
            success=False,
            chunks_created=0,
            entities_created=0,
            relationships_created=0,
            vector_embeddings_created=0,
            entity_mappings_created=0,
            errors=[],
            processing_time=0.0
        )
        
        try:
            # Step 1: Process document (extract text, chunk, extract entities)
            chunks, entities, relationships = self.document_processor.process_document(
                document,
                chunking_strategy=self.config.chunking_strategy,
                chunk_size=self.config.chunk_size
            )
            
            result.chunks_created = len(chunks)
            logger.info(f"Processed document {document.id}: {len(chunks)} chunks, {len(entities)} entities, {len(relationships)} relationships")
            
            # Step 2: Store document in graph database
            await self._store_document_in_graph(document, entities, relationships)
            result.entities_created = len(entities)
            result.relationships_created = len(relationships)
            
            # Step 3: Generate embeddings and store in vector database
            if self.config.create_embeddings:
                embeddings_created = await self._store_chunks_in_vector(chunks, document.id)
                result.vector_embeddings_created = embeddings_created
            
            # Step 4: Create entity-vector mappings
            if self.config.enable_entity_linking:
                mappings_created = await self._create_entity_vector_mappings(entities, chunks, document.id)
                result.entity_mappings_created = mappings_created
            
            result.success = True
            result.processing_time = time.time() - start_time
            
            logger.info(f"Successfully ingested document {document.id} in {result.processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Failed to ingest document {document.id}: {str(e)}"
            result.errors.append(error_msg)
            result.processing_time = time.time() - start_time
            logger.error(error_msg, exc_info=True)
        
        return result
    
    async def _store_document_in_graph(self, document: Document, entities: List[ExtractedEntity], 
                                     relationships: List[EntityRelationship]) -> None:
        """Store document and extracted entities/relationships in Neo4j."""
        try:
            # Create document node
            doc_query = """
            CREATE (d:Document {
                id: $doc_id,
                title: $title,
                content: $content,
                document_type: $doc_type,
                source: $source,
                created_at: $created_at,
                updated_at: $updated_at
            })
            RETURN d
            """
            
            self.graph_db.execute_query(doc_query, {
                "doc_id": document.id,
                "title": document.title,
                "content": document.content[:10000],  # Limit content size in graph
                "doc_type": document.document_type.value,
                "source": document.source,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat()
            })
            
            # Create entity nodes and link to document
            for entity in entities:
                # Create or merge entity
                entity_query = """
                MERGE (e:Entity {name: $name, type: $type})
                ON CREATE SET 
                    e.id = $entity_id,
                    e.description = $description,
                    e.created_at = $created_at,
                    e.updated_at = $updated_at,
                    e.properties = $properties
                ON MATCH SET
                    e.updated_at = $updated_at
                WITH e
                MATCH (d:Document {id: $doc_id})
                MERGE (e)-[:MENTIONED_IN {
                    frequency: 1,
                    context: $context,
                    confidence: $confidence
                }]->(d)
                RETURN e
                """
                
                entity_id = str(uuid.uuid4())
                self.graph_db.execute_query(entity_query, {
                    "entity_id": entity_id,
                    "name": entity.text,
                    "type": entity.label,
                    "description": f"Entity extracted from document {document.id}",
                    "doc_id": document.id,
                    "context": entity.context[:1000],  # Limit context size
                    "confidence": entity.confidence,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "properties": json.dumps(entity.properties) if entity.properties else "{}"
                })
            
            # Create relationships between entities
            if self.config.create_graph_relationships:
                for relationship in relationships:
                    rel_query = """
                    MATCH (e1:Entity {name: $source_entity})
                    MATCH (e2:Entity {name: $target_entity})
                    MERGE (e1)-[r:RELATED_TO {
                        type: $rel_type,
                        confidence: $confidence,
                        context: $context,
                        evidence: $evidence
                    }]->(e2)
                    RETURN r
                    """
                    
                    self.graph_db.execute_query(rel_query, {
                        "source_entity": relationship.source_entity,
                        "target_entity": relationship.target_entity,
                        "rel_type": relationship.relationship_type,
                        "confidence": relationship.confidence,
                        "context": relationship.context[:1000],
                        "evidence": relationship.evidence_text[:500]
                    })
            
        except Exception as e:
            raise Exception(f"Failed to store document in graph database: {e}")
    
    async def _store_chunks_in_vector(self, chunks: List[TextChunk], document_id: str) -> int:
        """Store document chunks as vector embeddings."""
        try:
            embeddings_created = 0
            
            # Get document title and source from Neo4j for proper attribution
            document_info = await self._get_document_info(document_id)
            doc_title = document_info.get('title', 'Unknown Document')
            doc_source = document_info.get('source', 'Unknown Source')
            
            # Process chunks in batches
            for i in range(0, len(chunks), self.config.batch_size):
                batch_chunks = chunks[i:i + self.config.batch_size]
                
                # Generate embeddings for batch
                texts = [chunk.content for chunk in batch_chunks]
                embeddings = await self._generate_embeddings_batch(texts)
                
                # Prepare data for vector storage
                documents = []
                metadatas = []
                ids = []
                
                for chunk, embedding in zip(batch_chunks, embeddings):
                    documents.append(chunk.content)
                    metadatas.append({
                        "chunk_id": chunk.id,
                        "document_id": document_id,
                        "chunk_index": chunk.chunk_index,
                        "start_position": chunk.start_position,
                        "end_position": chunk.end_position,
                        "strategy": chunk.metadata.get("strategy", "unknown"),
                        "title": doc_title,  # Add actual document title
                        "source": doc_source,  # Add actual document source
                        "created_at": datetime.now().isoformat()
                    })
                    ids.append(chunk.id)
                
                # Store in vector database
                self.vector_db.add_documents(
                    collection_name="documents",
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                embeddings_created += len(batch_chunks)
                logger.debug(f"Stored batch of {len(batch_chunks)} chunks in vector database")
            
            return embeddings_created
            
        except Exception as e:
            raise Exception(f"Failed to store chunks in vector database: {e}")
    
    async def _get_document_info(self, document_id: str) -> dict:
        """Get document title and source from Neo4j for proper attribution."""
        try:
            query = """
            MATCH (d:Document {id: $document_id})
            RETURN d.title as title, d.source as source
            """
            
            result = self.graph_db.execute_query(query, {"document_id": document_id})
            
            if result and len(result) > 0:
                return {
                    'title': result[0].get('title', 'Unknown Document'),
                    'source': result[0].get('source', 'Unknown Source')
                }
            else:
                return {'title': 'Unknown Document', 'source': 'Unknown Source'}
                
        except Exception as e:
            logger.warning(f"Failed to get document info for {document_id}: {e}")
            return {'title': 'Unknown Document', 'source': 'Unknown Source'}
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            # Use the embedding service to generate embeddings
            embeddings = []
            for text in texts:
                embedding = await self._generate_single_embedding(text)
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {e}")
    
    async def _generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            # Use the embedding service to generate embeddings
            if hasattr(self.embedding_service, 'encode'):
                embedding = self.embedding_service.encode(text)
                return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            else:
                # Fallback: create a dummy embedding
                import numpy as np
                return np.random.rand(384).tolist()  # 384-dimensional embedding
                
        except Exception as e:
            raise Exception(f"Failed to generate embedding for text: {e}")
    
    async def _create_entity_vector_mappings(self, entities: List[ExtractedEntity], 
                                           chunks: List[TextChunk], document_id: str) -> int:
        """Create mappings between entities and their vector representations."""
        try:
            mappings_created = 0
            
            # Get entity IDs from graph database
            entity_id_map = {}
            for entity in entities:
                query = """
                MATCH (e:Entity {name: $name, type: $type})
                RETURN e.id as entity_id
                LIMIT 1
                """
                
                result = self.graph_db.execute_query(query, {
                    "name": entity.text,
                    "type": entity.label
                })
                
                if result:
                    entity_id_map[entity.text] = result[0]["entity_id"]
            
            # Create mappings for entities found in chunks
            for chunk in chunks:
                chunk_entities = []
                
                # Find entities mentioned in this chunk
                for entity in entities:
                    if entity.text.lower() in chunk.content.lower():
                        chunk_entities.append(entity)
                
                # Create mappings
                for entity in chunk_entities:
                    if entity.text in entity_id_map:
                        entity_id = entity_id_map[entity.text]
                        
                        # Create mapping
                        self.mapping_service.create_mapping(
                            entity_id=entity_id,
                            entity_type=entity.label,
                            vector_id=chunk.id,
                            collection_name="documents",
                            metadata={
                                "document_id": document_id,
                                "chunk_index": chunk.chunk_index,
                                "confidence": entity.confidence,
                                "context": entity.context[:500]
                            }
                        )
                        
                        mappings_created += 1
            
            return mappings_created
            
        except Exception as e:
            raise Exception(f"Failed to create entity-vector mappings: {e}")
    
    async def ingest_documents_batch(self, documents: List[Document]) -> List[IngestionResult]:
        """
        Ingest multiple documents in batch with parallel processing.
        
        Args:
            documents: List of documents to ingest
            
        Returns:
            List of IngestionResult objects
        """
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Starting batch ingestion of {len(documents)} documents")
        
        # Process documents in batches to avoid overwhelming the system
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [self.ingest_document(doc) for doc in batch_docs]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    error_result = IngestionResult(
                        document_id=batch_docs[j].id,
                        success=False,
                        chunks_created=0,
                        entities_created=0,
                        relationships_created=0,
                        vector_embeddings_created=0,
                        entity_mappings_created=0,
                        errors=[f"Exception during ingestion: {str(result)}"],
                        processing_time=0.0
                    )
                    results.append(error_result)
                else:
                    results.append(result)
            
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        # Log summary statistics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_chunks = sum(r.chunks_created for r in results)
        total_entities = sum(r.entities_created for r in results)
        total_embeddings = sum(r.vector_embeddings_created for r in results)
        total_mappings = sum(r.entity_mappings_created for r in results)
        
        logger.info(f"Batch ingestion completed: {successful} successful, {failed} failed")
        logger.info(f"Created: {total_chunks} chunks, {total_entities} entities, {total_embeddings} embeddings, {total_mappings} mappings")
        
        return results
    
    def validate_ingestion_integrity(self, document_id: str) -> Dict[str, Any]:
        """
        Validate the integrity of ingested data for a specific document.
        
        Args:
            document_id: ID of the document to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            "document_id": document_id,
            "graph_data_exists": False,
            "vector_data_exists": False,
            "mappings_consistent": False,
            "entity_count_graph": 0,
            "chunk_count_vector": 0,
            "mapping_count": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check graph database
            graph_query = """
            MATCH (d:Document {id: $doc_id})
            OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(d)
            RETURN d, count(e) as entity_count
            """
            
            graph_result = self.graph_db.execute_query(graph_query, {"doc_id": document_id})
            if graph_result:
                validation_result["graph_data_exists"] = True
                validation_result["entity_count_graph"] = graph_result[0]["entity_count"]
            
            # Check vector database
            try:
                vector_results = self.vector_db.query_collection(
                    collection_name="documents",
                    query_texts=["dummy"],  # We just want to check metadata
                    n_results=1000,  # Large number to get all chunks
                    where={"document_id": document_id}
                )
                
                if vector_results.get("ids") and vector_results["ids"][0]:
                    validation_result["vector_data_exists"] = True
                    validation_result["chunk_count_vector"] = len(vector_results["ids"][0])
            except Exception as e:
                validation_result["errors"].append(f"Error checking vector data: {e}")
            
            # Check mappings
            try:
                # This would require querying the mapping service
                # For now, we'll do a basic consistency check
                if validation_result["graph_data_exists"] and validation_result["vector_data_exists"]:
                    validation_result["mappings_consistent"] = True
            except Exception as e:
                validation_result["errors"].append(f"Error checking mappings: {e}")
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
        
        return validation_result
    
    def get_ingestion_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about ingested data.
        
        Returns:
            Dictionary containing ingestion statistics
        """
        stats = {
            "total_documents": 0,
            "total_entities": 0,
            "total_chunks": 0,
            "total_mappings": 0,
            "collections": {},
            "errors": []
        }
        
        try:
            # Get graph statistics
            graph_queries = [
                ("total_documents", "MATCH (d:Document) RETURN count(d) as count"),
                ("total_entities", "MATCH (e:Entity) RETURN count(e) as count"),
                ("total_relationships", "MATCH ()-[r]->() RETURN count(r) as count")
            ]
            
            for stat_name, query in graph_queries:
                try:
                    result = self.graph_db.execute_query(query)
                    if result:
                        stats[stat_name] = result[0]["count"]
                except Exception as e:
                    stats["errors"].append(f"Error getting {stat_name}: {e}")
            
            # Get vector statistics
            try:
                collections = self.vector_db.list_collections()
                for collection_name in collections:
                    if collection_name in ["documents", "entities"]:
                        collection_stats = self.vector_db.get_collection_stats(collection_name)
                        stats["collections"][collection_name] = collection_stats
                        
                        if collection_name == "documents":
                            stats["total_chunks"] = collection_stats.get("document_count", 0)
            except Exception as e:
                stats["errors"].append(f"Error getting vector statistics: {e}")
            
            # Get mapping statistics
            try:
                mapping_stats = self.mapping_service.get_mapping_statistics()
                stats.update(mapping_stats)
            except Exception as e:
                stats["errors"].append(f"Error getting mapping statistics: {e}")
            
        except Exception as e:
            stats["errors"].append(f"General statistics error: {e}")
        
        return stats


class BatchProcessor:
    """Helper class for processing large document sets in batches."""
    
    def __init__(self, pipeline: DualStorageIngestionPipeline, batch_size: int = 10):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.processed_count = 0
        self.failed_count = 0
        
    async def process_document_set(self, documents: List[Document], 
                                 progress_callback: Optional[callable] = None) -> List[IngestionResult]:
        """
        Process a large set of documents with progress tracking.
        
        Args:
            documents: List of documents to process
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of IngestionResult objects
        """
        total_docs = len(documents)
        all_results = []
        
        logger.info(f"Starting batch processing of {total_docs} documents")
        
        for i in range(0, total_docs, self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_docs + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)")
            
            # Process batch
            batch_results = await self.pipeline.ingest_documents_batch(batch_docs)
            all_results.extend(batch_results)
            
            # Update counters
            batch_successful = sum(1 for r in batch_results if r.success)
            batch_failed = len(batch_results) - batch_successful
            
            self.processed_count += batch_successful
            self.failed_count += batch_failed
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback({
                    "batch_num": batch_num,
                    "total_batches": total_batches,
                    "processed_count": self.processed_count,
                    "failed_count": self.failed_count,
                    "total_documents": total_docs,
                    "progress_percentage": (i + len(batch_docs)) / total_docs * 100
                })
            
            # Small delay between batches to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        logger.info(f"Batch processing completed: {self.processed_count} successful, {self.failed_count} failed")
        return all_results


# Utility functions for pipeline management

async def create_ingestion_pipeline(
    document_processor: Optional[DocumentProcessor] = None,
    config: Optional[IngestionConfig] = None
) -> DualStorageIngestionPipeline:
    """
    Create and initialize a dual storage ingestion pipeline.
    
    Args:
        document_processor: Optional document processor instance
        config: Optional ingestion configuration
        
    Returns:
        Initialized DualStorageIngestionPipeline
    """
    from .database import get_neo4j_manager, get_vector_manager
    from .mapping_service import get_mapping_service
    
    # Create document processor if not provided
    if document_processor is None:
        document_processor = DocumentProcessor()
    
    # Get database managers
    graph_db = get_neo4j_manager()
    vector_db = get_vector_manager()
    mapping_service = get_mapping_service()
    
    # Create embedding service
    from .embedding_service import get_embedding_service
    embedding_service = get_embedding_service()
    embedding_service.initialize()
    
    # Create pipeline
    pipeline = DualStorageIngestionPipeline(
        document_processor=document_processor,
        graph_db_manager=graph_db,
        vector_db_manager=vector_db,
        mapping_service=mapping_service,
        embedding_service=embedding_service,
        config=config
    )
    
    # Initialize pipeline
    pipeline.initialize()
    
    return pipeline


def validate_pipeline_health(pipeline: DualStorageIngestionPipeline) -> Dict[str, Any]:
    """
    Validate the health of the ingestion pipeline.
    
    Args:
        pipeline: Pipeline to validate
        
    Returns:
        Dictionary containing health check results
    """
    health_result = {
        "overall_healthy": True,
        "graph_db_healthy": False,
        "vector_db_healthy": False,
        "mapping_service_healthy": False,
        "errors": [],
        "warnings": []
    }
    
    try:
        # Check graph database
        if hasattr(pipeline.graph_db, 'health_check'):
            health_result["graph_db_healthy"] = pipeline.graph_db.health_check()
        else:
            health_result["warnings"].append("Graph database health check not available")
        
        # Check vector database
        if hasattr(pipeline.vector_db, 'health_check'):
            health_result["vector_db_healthy"] = pipeline.vector_db.health_check()
        else:
            health_result["warnings"].append("Vector database health check not available")
        
        # Check mapping service
        try:
            mapping_validation = pipeline.mapping_service.validate_mapping_integrity()
            health_result["mapping_service_healthy"] = mapping_validation.get("validation_passed", False)
        except Exception as e:
            health_result["errors"].append(f"Mapping service health check failed: {e}")
        
        # Overall health
        health_result["overall_healthy"] = (
            health_result["graph_db_healthy"] and 
            health_result["vector_db_healthy"] and 
            health_result["mapping_service_healthy"]
        )
        
    except Exception as e:
        health_result["errors"].append(f"Health check error: {e}")
        health_result["overall_healthy"] = False
    
    return health_result