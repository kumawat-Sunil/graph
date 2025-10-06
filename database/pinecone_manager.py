"""
Pinecone Vector Database Manager
Handles vector storage and retrieval operations using Pinecone cloud service
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer
import uuid
import time

logger = logging.getLogger(__name__)

class PineconeManager:
    """Manages Pinecone vector database operations"""
    
    def __init__(self):
        """Initialize Pinecone client and index"""
        # Use config system instead of direct os.getenv
        from core.config import get_config
        config = get_config()
        
        self.api_key = config.database.pinecone_api_key
        self.index_name = config.database.pinecone_index_name
        self.environment = config.database.pinecone_environment
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Get index
        try:
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Pinecone index: {e}")
            raise
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = self.embedding_model.encode([text])[0].tolist()
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def upsert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Upsert documents to Pinecone
        
        Args:
            documents: List of documents with 'id', 'text', and 'metadata'
        """
        try:
            vectors = []
            
            for doc in documents:
                # Generate embedding
                embedding = self.generate_embedding(doc['text'])
                
                # Prepare vector
                vector = {
                    "id": doc['id'],
                    "values": embedding,
                    "metadata": {
                        "text": doc['text'][:1000],  # Pinecone metadata limit
                        **doc.get('metadata', {})
                    }
                }
                vectors.append(vector)
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(batch)
                logger.debug(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors)")
            
            logger.info(f"✅ Successfully upserted {len(vectors)} documents to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upsert documents: {e}")
            return False
    
    def search_similar(self, query_text: str, top_k: int = 10, 
                      filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_text: Text to search for
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query_text)
            
            # Search
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata
            )
            
            # Format results
            documents = []
            for match in results['matches']:
                doc = {
                    'id': match['id'],
                    'score': match['score'],
                    'text': match['metadata'].get('text', ''),
                    'metadata': match['metadata']
                }
                documents.append(doc)
            
            logger.info(f"✅ Found {len(documents)} similar documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.index.delete(ids=document_ids)
            logger.info(f"✅ Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete documents: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats['total_vector_count'],
                'dimension': stats['dimension'],
                'index_fullness': stats.get('index_fullness', 0)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get index stats: {e}")
            return {}
    
    def clear_all_vectors(self) -> bool:
        """Clear all vectors from the index"""
        try:
            # Delete all vectors in the index
            self.index.delete(delete_all=True)
            logger.info("✅ All vectors cleared from Pinecone index")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear all vectors: {e}")
            return False
    
    def create_collection(self, name: str, metadata: Optional[Dict] = None):
        """Compatibility method - Pinecone uses namespaces instead of collections"""
        logger.info(f"Pinecone uses namespaces. Collection '{name}' noted for namespace usage.")
        return True
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     metadatas: List[Dict], ids: List[str]):
        """
        Compatibility method for ChromaDB-style interface
        """
        try:
            # Convert to Pinecone format
            pinecone_docs = []
            for i, (doc_id, text, metadata) in enumerate(zip(ids, documents, metadatas)):
                pinecone_docs.append({
                    'id': doc_id,
                    'text': text,
                    'metadata': {
                        'collection': collection_name,
                        **metadata
                    }
                })
            
            return self.upsert_documents(pinecone_docs)
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False
    
    def query_collection(self, collection_name: str, query_text: str, 
                        n_results: int = 10) -> Dict[str, Any]:
        """
        Compatibility method for ChromaDB-style querying
        """
        try:
            # Search with collection filter
            filter_metadata = {"collection": collection_name}
            results = self.search_similar(query_text, top_k=n_results, 
                                        filter_metadata=filter_metadata)
            
            # Convert to ChromaDB-style format
            return {
                'ids': [[doc['id'] for doc in results]],
                'documents': [[doc['text'] for doc in results]],
                'metadatas': [[doc['metadata'] for doc in results]],
                'distances': [[1 - doc['score'] for doc in results]]  # Convert similarity to distance
            }
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}

# Factory function for compatibility
def get_pinecone_manager() -> PineconeManager:
    """Get Pinecone manager instance"""
    return PineconeManager()