"""
Database connection management for Neo4j and Vector Databases (Pinecone/Chroma).
"""

import time
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager, contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

from neo4j import GraphDatabase, Driver, Session, Result
from neo4j.exceptions import ServiceUnavailable, TransientError, ClientError
import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from .config import get_config

logger = logging.getLogger(__name__)

# Check if Pinecone is available
try:
    from database.pinecone_manager import PineconeManager
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available. Install with: pip install pinecone-client[grpc]")


class Neo4jConnectionError(Exception):
    """Exception raised for Neo4j connection errors."""
    pass


class ChromaConnectionError(Exception):
    """Exception raised for Chroma connection errors."""
    pass


class Neo4jConnectionManager:
    """
    Neo4j connection manager for local Neo4j instances.
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            # Simple driver configuration for local Neo4j
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            
            # Test the connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1").single()
            
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            
        except Exception as e:
            raise Neo4jConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def disconnect(self) -> None:
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")
    
    @contextmanager
    def session(self, **kwargs):
        """Context manager for Neo4j sessions."""
        if not self.driver:
            raise Neo4jConnectionError("Not connected to Neo4j. Call connect() first.")
        
        database = kwargs.pop('database', self.database)
        session = self.driver.session(database=database, **kwargs)
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        if not self.driver:
            raise Neo4jConnectionError("Not connected to Neo4j. Call connect() first.")
        
        try:
            with self.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise Neo4jConnectionError(f"Query failed: {e}")
    
    async def execute_query_async(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query asynchronously.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.execute_query, query, parameters)
    
    def execute_write_transaction(self, transaction_function, **kwargs) -> Any:
        """
        Execute a write transaction.
        
        Args:
            transaction_function: Function that takes a transaction as parameter
            **kwargs: Additional arguments to pass to the transaction function
            
        Returns:
            Result of the transaction function
        """
        if not self.driver:
            raise Neo4jConnectionError("Not connected to Neo4j. Call connect() first.")
        
        try:
            with self.session() as session:
                return session.execute_write(transaction_function, **kwargs)
        except Exception as e:
            raise Neo4jConnectionError(f"Write transaction failed: {e}")
    
    def execute_read_transaction(self, transaction_function, **kwargs) -> Any:
        """
        Execute a read transaction.
        
        Args:
            transaction_function: Function that takes a transaction as parameter
            **kwargs: Additional arguments to pass to the transaction function
            
        Returns:
            Result of the transaction function
        """
        if not self.driver:
            raise Neo4jConnectionError("Not connected to Neo4j. Call connect() first.")
        
        try:
            with self.session() as session:
                return session.execute_read(transaction_function, **kwargs)
        except Exception as e:
            raise Neo4jConnectionError(f"Read transaction failed: {e}")
    
    def health_check(self) -> bool:
        """
        Check if the Neo4j connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            with self.session() as session:
                result = session.run("RETURN 1 as health_check")
                return result.single()["health_check"] == 1
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the Neo4j database.
        
        Returns:
            Dictionary containing database information
        """
        try:
            info = {}
            
            # Get Neo4j version
            with self.session() as session:
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                for record in result:
                    if record["name"] == "Neo4j Kernel":
                        info["version"] = record["versions"][0]
                        info["edition"] = record["edition"]
            
            # Get database statistics
            with self.session() as session:
                result = session.run("""
                    MATCH (n) 
                    RETURN count(n) as node_count
                """)
                info["node_count"] = result.single()["node_count"]
                
                result = session.run("""
                    MATCH ()-[r]->() 
                    RETURN count(r) as relationship_count
                """)
                info["relationship_count"] = result.single()["relationship_count"]
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}


# Global Neo4j connection manager instance
_neo4j_manager: Optional[Neo4jConnectionManager] = None


def get_neo4j_manager() -> Neo4jConnectionManager:
    """Get the global Neo4j connection manager instance."""
    global _neo4j_manager
    
    if _neo4j_manager is None:
        config = get_config()
        _neo4j_manager = Neo4jConnectionManager(
            uri=config.database.neo4j_uri,
            user=config.database.neo4j_user,
            password=config.database.neo4j_password,
            database=config.database.neo4j_database  # Use the actual database name from config
        )
    
    return _neo4j_manager


def initialize_neo4j() -> None:
    """Initialize the Neo4j connection."""
    manager = get_neo4j_manager()
    manager.connect()


def close_neo4j() -> None:
    """Close the Neo4j connection."""
    global _neo4j_manager
    if _neo4j_manager:
        _neo4j_manager.disconnect()
        _neo4j_manager = None


class ChromaConnectionManager:
    """
    Chroma vector database connection manager with collection management and optimization.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8000, persist_directory: Optional[str] = None):
        self.host = host
        self.port = port
        self.persist_directory = persist_directory
        self.client: Optional[ClientAPI] = None
        self.collections: Dict[str, Any] = {}
        self.max_retries = get_config().agents.max_retries
        
    def connect(self) -> None:
        """Establish connection to Chroma database with retry logic."""
        for attempt in range(self.max_retries):
            try:
                if self.persist_directory:
                    # Use persistent storage
                    self.client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
                else:
                    # Use HTTP client for remote Chroma server
                    self.client = chromadb.HttpClient(
                        host=self.host,
                        port=self.port,
                        settings=Settings(
                            anonymized_telemetry=False
                        )
                    )
                
                # Test the connection
                self.client.heartbeat()
                
                logger.info(f"Successfully connected to Chroma database")
                return
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise ChromaConnectionError(f"Failed to connect to Chroma after {self.max_retries} attempts: {e}")
                
                wait_time = 2 ** attempt
                logger.warning(f"Chroma connection attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
    
    def disconnect(self) -> None:
        """Close the Chroma client connection."""
        if self.client:
            self.client = None
            self.collections.clear()
            logger.info("Disconnected from Chroma")
    
    def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None, 
                         embedding_function: Optional[Any] = None) -> Any:
        """
        Create a new collection in Chroma.
        
        Args:
            name: Collection name
            metadata: Optional metadata for the collection
            embedding_function: Optional custom embedding function
            
        Returns:
            Chroma collection object
        """
        if not self.client:
            raise ChromaConnectionError("Not connected to Chroma. Call connect() first.")
        
        try:
            # Check if collection already exists
            try:
                collection = self.client.get_collection(name=name, embedding_function=embedding_function)
                logger.info(f"Collection '{name}' already exists, returning existing collection")
                self.collections[name] = collection
                return collection
            except Exception:
                # Collection doesn't exist, create it
                pass
            
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {},
                embedding_function=embedding_function
            )
            
            self.collections[name] = collection
            logger.info(f"Created collection '{name}' in Chroma")
            return collection
            
        except Exception as e:
            raise ChromaConnectionError(f"Failed to create collection '{name}': {e}")
    
    def get_collection(self, name: str, embedding_function: Optional[Any] = None) -> Any:
        """
        Get an existing collection from Chroma.
        
        Args:
            name: Collection name
            embedding_function: Optional custom embedding function
            
        Returns:
            Chroma collection object
        """
        if not self.client:
            raise ChromaConnectionError("Not connected to Chroma. Call connect() first.")
        
        # Check if collection is cached
        if name in self.collections:
            return self.collections[name]
        
        try:
            collection = self.client.get_collection(name=name, embedding_function=embedding_function)
            self.collections[name] = collection
            return collection
            
        except Exception as e:
            raise ChromaConnectionError(f"Failed to get collection '{name}': {e}")
    
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection from Chroma.
        
        Args:
            name: Collection name to delete
        """
        if not self.client:
            raise ChromaConnectionError("Not connected to Chroma. Call connect() first.")
        
        try:
            self.client.delete_collection(name=name)
            if name in self.collections:
                del self.collections[name]
            logger.info(f"Deleted collection '{name}' from Chroma")
            
        except Exception as e:
            raise ChromaConnectionError(f"Failed to delete collection '{name}': {e}")
    
    def list_collections(self) -> List[str]:
        """
        List all collections in Chroma.
        
        Returns:
            List of collection names
        """
        if not self.client:
            raise ChromaConnectionError("Not connected to Chroma. Call connect() first.")
        
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
            
        except Exception as e:
            raise ChromaConnectionError(f"Failed to list collections: {e}")
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None) -> None:
        """
        Add documents to a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """
        collection = self.get_collection(collection_name)
        
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}_{int(time.time())}" for i in range(len(documents))]
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
            
        except Exception as e:
            raise ChromaConnectionError(f"Failed to add documents to collection '{collection_name}': {e}")
    
    def query_collection(self, collection_name: str, query_texts: List[str], 
                        n_results: int = 10, where: Optional[Dict[str, Any]] = None,
                        where_document: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query a collection for similar documents.
        
        Args:
            collection_name: Name of the collection to query
            query_texts: List of query texts
            n_results: Number of results to return
            where: Optional metadata filter
            where_document: Optional document content filter
            
        Returns:
            Query results dictionary
        """
        collection = self.get_collection(collection_name)
        
        try:
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            return results
            
        except Exception as e:
            raise ChromaConnectionError(f"Failed to query collection '{collection_name}': {e}")
    
    def update_documents(self, collection_name: str, ids: List[str],
                        documents: Optional[List[str]] = None,
                        metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Update documents in a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of document IDs to update
            documents: Optional list of new document texts
            metadatas: Optional list of new metadata dictionaries
        """
        collection = self.get_collection(collection_name)
        
        try:
            collection.update(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Updated {len(ids)} documents in collection '{collection_name}'")
            
        except Exception as e:
            raise ChromaConnectionError(f"Failed to update documents in collection '{collection_name}': {e}")
    
    def delete_documents(self, collection_name: str, ids: List[str]) -> None:
        """
        Delete documents from a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of document IDs to delete
        """
        collection = self.get_collection(collection_name)
        
        try:
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from collection '{collection_name}'")
            
        except Exception as e:
            raise ChromaConnectionError(f"Failed to delete documents from collection '{collection_name}': {e}")
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary containing collection statistics
        """
        collection = self.get_collection(collection_name)
        
        try:
            count = collection.count()
            metadata = collection.metadata
            
            return {
                "name": collection_name,
                "document_count": count,
                "metadata": metadata
            }
            
        except Exception as e:
            raise ChromaConnectionError(f"Failed to get stats for collection '{collection_name}': {e}")
    
    def health_check(self) -> bool:
        """
        Check if the Chroma connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Test the connection with heartbeat
            heartbeat = self.client.heartbeat()
            return heartbeat is not None
            
        except Exception as e:
            logger.error(f"Chroma health check failed: {e}")
            return False
    
    def reset_database(self) -> None:
        """
        Reset the entire Chroma database (use with caution).
        """
        if not self.client:
            raise ChromaConnectionError("Not connected to Chroma. Call connect() first.")
        
        try:
            self.client.reset()
            self.collections.clear()
            logger.warning("Chroma database has been reset - all data deleted")
            
        except Exception as e:
            raise ChromaConnectionError(f"Failed to reset Chroma database: {e}")


# Global Chroma connection manager instance
_chroma_manager: Optional[ChromaConnectionManager] = None


def get_chroma_manager() -> ChromaConnectionManager:
    """Get the global Chroma connection manager instance."""
    global _chroma_manager
    
    if _chroma_manager is None:
        config = get_config()
        _chroma_manager = ChromaConnectionManager(
            host=config.database.chroma_host,
            port=config.database.chroma_port,
            persist_directory=config.database.chroma_persist_directory
        )
    
    return _chroma_manager


def initialize_chroma() -> None:
    """Initialize the Chroma connection."""
    manager = get_chroma_manager()
    manager.connect()


def close_chroma() -> None:
    """Close the Chroma connection."""
    global _chroma_manager
    if _chroma_manager:
        _chroma_manager.disconnect()
        _chroma_manager = None


def initialize_databases() -> None:
    """Initialize both Neo4j and Chroma connections."""
    initialize_neo4j()
    initialize_chroma()


def close_databases() -> None:
    """Close both Neo4j and Chroma connections."""
    close_neo4j()
    close_chroma()


# Global Pinecone manager instance
_pinecone_manager = None


def get_pinecone_manager() -> PineconeManager:
    """Get the global Pinecone manager instance."""
    global _pinecone_manager
    
    if not PINECONE_AVAILABLE:
        raise ImportError("Pinecone not available. Install with: pip install pinecone-client[grpc]")
    
    if _pinecone_manager is None:
        _pinecone_manager = PineconeManager()
    
    return _pinecone_manager


def get_vector_manager():
    """
    Get vector database manager based on configuration.
    Returns Pinecone if configured, otherwise Chroma.
    """
    config = get_config()
    vector_db_type = getattr(config.database, 'vector_db_type', 'chroma').lower()
    
    if vector_db_type == 'pinecone' and PINECONE_AVAILABLE:
        return get_pinecone_manager()
    else:
        return get_chroma_manager()


def initialize_vector_db() -> None:
    """Initialize the configured vector database."""
    config = get_config()
    vector_db_type = getattr(config.database, 'vector_db_type', 'chroma').lower()
    
    if vector_db_type == 'pinecone' and PINECONE_AVAILABLE:
        # Pinecone doesn't need explicit initialization
        logger.info("✅ Pinecone manager ready")
    else:
        initialize_chroma()


def close_vector_db() -> None:
    """Close the configured vector database."""
    global _pinecone_manager
    config = get_config()
    vector_db_type = getattr(config.database, 'vector_db_type', 'chroma').lower()
    
    if vector_db_type == 'pinecone':
        _pinecone_manager = None
        logger.info("✅ Pinecone manager closed")
    else:
        close_chroma()


def initialize_databases() -> None:
    """Initialize both Neo4j and the configured vector database."""
    initialize_neo4j()
    initialize_vector_db()


def close_databases() -> None:
    """Close both Neo4j and the configured vector database."""
    close_neo4j()
    close_vector_db()