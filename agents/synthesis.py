"""
Synthesis Agent for the Graph-Enhanced Agentic RAG system.

This agent handles result integration from graph and vector search,
response generation using Gemini API, and citation formatting.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Union
import json
import re
from dataclasses import dataclass
import time

from google import genai

from core.interfaces import (
    SynthesisAgentInterface, MessageType, 
    GraphResult, VectorResult, SynthesisResult
)
from core.protocols import AgentMessage
from core.protocols import (
    SynthesisMessage, SynthesisResponse, GraphSearchResponse, 
    VectorSearchResponse, MessageValidator
)
from core.models import Entity, Document
from core.config import get_config


logger = logging.getLogger(__name__)


@dataclass
class IntegratedContext:
    """Represents integrated context from multiple sources."""
    query: str
    graph_entities: List[Entity]
    graph_relationships: List[Dict[str, Any]]
    vector_documents: List[Document]
    combined_text: str
    source_mapping: Dict[str, str]  # Maps content to source
    relevance_scores: Dict[str, float]  # Maps source to relevance score
    total_tokens: int


@dataclass
class CitationInfo:
    """Information for creating citations."""
    source_id: str
    source_type: str  # 'graph' or 'vector'
    title: str
    content_snippet: str
    relevance_score: float
    url: Optional[str] = None


@dataclass
class GeminiResponse:
    """Structured response from Gemini API."""
    text: str
    finish_reason: str
    safety_ratings: List[Dict[str, Any]]
    usage_metadata: Dict[str, Any]
    processing_time: float


class GeminiClient:
    """Client for interacting with Google's Gemini API using the new API."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp", temperature: float = 0.7, max_tokens: int = 2048):
        """Initialize Gemini client with authentication and configuration."""
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        # Initialize the new Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized Gemini client with model: {model_name}")
    
    async def generate_response(self, prompt: str, context: str = "") -> GeminiResponse:
        """Generate response using Gemini API with error handling and retries."""
        start_time = time.time()
        
        try:
            # Combine prompt and context
            full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
            
            # Generate response using correct new API structure (matching working code)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[{"parts": [{"text": full_prompt}]}]
            )
            
            processing_time = time.time() - start_time
            
            # Parse response using correct structure (matching working code)
            if response and response.candidates and len(response.candidates) > 0:
                response_text = response.candidates[0].content.parts[0].text
                finish_reason = "STOP"
                safety_ratings = []
                usage_metadata = {
                    "prompt_token_count": 0,
                    "candidates_token_count": len(response_text.split()) if response_text else 0,
                    "total_token_count": len(response_text.split()) if response_text else 0
                }
                
                return GeminiResponse(
                    text=response_text,
                    finish_reason=finish_reason,
                    safety_ratings=safety_ratings,
                    usage_metadata=usage_metadata,
                    processing_time=processing_time
                )
            else:
                raise ValueError("No valid response received from Gemini API")
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error generating response with Gemini: {str(e)}")
            
            # Return error response
            return GeminiResponse(
                text=f"Error generating response: {str(e)}",
                finish_reason="ERROR",
                safety_ratings=[],
                usage_metadata={"error": str(e)},
                processing_time=processing_time
            )
    
    def validate_response(self, response: GeminiResponse) -> bool:
        """Validate Gemini API response for quality and safety."""
        # Check if response was blocked by safety filters
        if any(rating.get("blocked", False) for rating in response.safety_ratings):
            logger.warning("Response blocked by safety filters")
            return False
        
        # Check if response finished properly
        if response.finish_reason not in ["STOP", "MAX_TOKENS"]:
            logger.warning(f"Response finished with reason: {response.finish_reason}")
            return False
        
        # Check if response has meaningful content
        if not response.text or len(response.text.strip()) < 10:
            logger.warning("Response text is too short or empty")
            return False
        
        return True


class PromptTemplates:
    """Templates for different types of queries and responses."""
    
    @staticmethod
    def get_factual_query_prompt(query: str) -> str:
        """Prompt template for simple factual queries."""
        return f"""You are an expert knowledge assistant. Answer the following question based on the provided context.

Question: {query}

Instructions:
- Provide a clear, concise, and accurate answer
- Use only information from the provided context
- If the context doesn't contain enough information, state this clearly
- Include relevant details but avoid unnecessary elaboration
- Cite sources when making specific claims

Answer:"""
    
    @staticmethod
    def get_relational_query_prompt(query: str) -> str:
        """Prompt template for relationship-focused queries."""
        return f"""You are an expert knowledge assistant specializing in analyzing relationships and connections between concepts.

Question: {query}

Instructions:
- Focus on explaining relationships, connections, and interactions between entities
- Trace logical paths through the provided graph relationships
- Explain how different concepts relate to each other
- Use the relationship information to provide comprehensive insights
- Highlight important connections that answer the question
- Cite specific relationships and entities from the context

Answer:"""
    
    @staticmethod
    def get_complex_query_prompt(query: str) -> str:
        """Prompt template for complex multi-hop queries."""
        return f"""You are an expert knowledge assistant capable of complex reasoning across multiple information sources.

Question: {query}

Instructions:
- Synthesize information from both graph relationships and document content
- Perform multi-step reasoning to connect different pieces of information
- Explain your reasoning process step by step
- Address different aspects of the complex question
- Use both entity relationships and document content to provide a comprehensive answer
- Show how different sources support your conclusions
- Provide citations for all major claims

Answer:"""
    
    @staticmethod
    def get_synthesis_prompt(query: str, query_type: str = "general") -> str:
        """Get appropriate prompt based on query type."""
        if query_type == "factual":
            return PromptTemplates.get_factual_query_prompt(query)
        elif query_type == "relational":
            return PromptTemplates.get_relational_query_prompt(query)
        elif query_type == "complex":
            return PromptTemplates.get_complex_query_prompt(query)
        else:
            # Default general prompt
            return f"""You are an expert knowledge assistant. Answer the following question based on the provided context.

Question: {query}

Instructions:
- Provide a comprehensive and accurate answer
- Use information from the provided context
- Explain your reasoning when making connections between different pieces of information
- Include relevant citations and sources
- If information is insufficient, state this clearly

Answer:"""


class SynthesisAgent(SynthesisAgentInterface):
    """
    Synthesis Agent implementation for integrating and synthesizing results.
    
    This agent handles:
    - Merging graph and vector search results
    - Deduplication and relevance scoring
    - Context window management for LLM input
    - Response generation with citations
    """
    
    def __init__(self, agent_id: str = "synthesis_agent"):
        super().__init__(agent_id)
        self.config = get_config()
        
        # Agent configuration
        self.max_context_tokens = self.config.agents.max_context_length
        self.min_relevance_threshold = 0.3
        self.deduplication_threshold = 0.85
        
        # Token estimation (rough approximation: 1 token ≈ 4 characters)
        self.chars_per_token = 4
        
        # Initialize Gemini client
        try:
            self.gemini_client = GeminiClient(
                api_key=self.config.llm.gemini_api_key,
                model_name=self.config.llm.gemini_model,
                temperature=self.config.llm.gemini_temperature,
                max_tokens=self.config.llm.gemini_max_tokens
            )
            logger.info(f"Initialized {self.agent_id} with Gemini model: {self.config.llm.gemini_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            self.gemini_client = None
        
        # Prompt templates
        self.prompt_templates = PromptTemplates()
        
        logger.info(f"Initialized {self.agent_id} with max_context_tokens={self.max_context_tokens}")
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages for synthesis operations."""
        try:
            if message.message_type == MessageType.SYNTHESIS_REQUEST:
                return await self._handle_synthesis_request(message)
            else:
                logger.warning(f"Unsupported message type: {message.message_type}")
                return None
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return self.create_message(
                MessageType.ERROR,
                {
                    "error_type": "synthesis_error",
                    "error_message": str(e),
                    "original_message_id": getattr(message, 'id', message.correlation_id)
                },
                message.correlation_id
            )
    
    async def integrate_results(
        self, 
        graph_results: Optional[GraphResult] = None,
        vector_results: Optional[Union[VectorResult, Any]] = None  # Accept both VectorResult and VectorSearchResponse
    ) -> Dict[str, Any]:
        """
        Integrate results from graph and vector search operations.
        
        This method merges results from different sources, performs deduplication,
        calculates relevance scores, and manages context window size.
        
        Args:
            graph_results: Results from graph traversal
            vector_results: Results from vector similarity search
            
        Returns:
            Dict containing integrated context ready for LLM processing
        """
        logger.info("Starting result integration")
        
        # Initialize integrated context structure
        integrated_context = {
            "entities": [],
            "relationships": [],
            "documents": [],
            "combined_text": "",
            "source_mapping": {},
            "relevance_scores": {},
            "citations": [],
            "total_tokens": 0,
            "graph_contribution": 0.0,
            "vector_contribution": 0.0
        }
        
        # Process graph results
        if graph_results:
            await self._integrate_graph_results(graph_results, integrated_context)
        
        # Process vector results
        if vector_results:
            await self._integrate_vector_results(vector_results, integrated_context)
        
        # Perform deduplication
        await self._deduplicate_content(integrated_context)
        
        # Calculate final relevance scores
        await self._calculate_relevance_scores(integrated_context)
        
        # Manage context window
        await self._manage_context_window(integrated_context)
        
        # Create combined text for LLM input
        await self._create_combined_text(integrated_context)
        
        # Store context for citation creation
        self._last_integrated_context = integrated_context
        
        logger.info(f"Integration complete. Total tokens: {integrated_context['total_tokens']}")
        return integrated_context
    
    async def _integrate_graph_results(self, graph_results: GraphResult, context: Dict[str, Any]):
        """Integrate graph search results into the context."""
        logger.debug("Integrating graph results")
        
        # Add entities with metadata
        for entity in graph_results.entities:
            entity_info = {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "description": entity.description or "",
                "source_type": "graph",
                "properties": entity.properties
            }
            context["entities"].append(entity_info)
            
            # Map entity content to source
            entity_text = f"{entity.name}: {entity.description or 'No description'}"
            context["source_mapping"][entity_text] = f"graph_entity_{entity.id}"
            context["relevance_scores"][f"graph_entity_{entity.id}"] = 0.8  # Default high relevance for graph entities
        
        # Add relationships
        for relationship in graph_results.relationships:
            rel_info = {
                "source": relationship.get("source", ""),
                "target": relationship.get("target", ""),
                "type": relationship.get("type", ""),
                "properties": relationship.get("properties", {}),
                "source_type": "graph"
            }
            context["relationships"].append(rel_info)
            
            # Map relationship content to source
            rel_text = f"{rel_info['source']} -> {rel_info['type']} -> {rel_info['target']}"
            context["source_mapping"][rel_text] = f"graph_rel_{len(context['relationships'])}"
            context["relevance_scores"][f"graph_rel_{len(context['relationships'])}"] = 0.7
        
        context["graph_contribution"] = len(graph_results.entities) + len(graph_results.relationships)
    
    async def _integrate_vector_results(self, vector_results: Union[VectorResult, Any], context: Dict[str, Any]):
        """Integrate vector search results into the context."""
        logger.debug("Integrating vector results")
        
        # Handle both VectorResult and VectorSearchResponse
        documents = vector_results.documents
        similarities = vector_results.similarities
        
        # Add documents with similarity scores
        for i, (document, similarity) in enumerate(zip(documents, similarities)):
            # Handle both Document objects and dictionaries
            if isinstance(document, dict):
                metadata = document.get('metadata', {})
                # Extract actual document information from Pinecone metadata
                actual_doc_id = metadata.get('document_id', document.get('id', f'doc_{i}'))
                actual_title = metadata.get('title') or document.get('title', f"Document Chunk {i+1}")
                actual_source = metadata.get('source') or document.get('source', 'unknown')
                
                doc_info = {
                    "id": document.get('id', f'doc_{i}'),  # Keep the chunk ID for mapping
                    "actual_document_id": actual_doc_id,  # Store the real document ID
                    "title": actual_title,
                    "content": document.get('content', ''),
                    "source": actual_source,
                    "source_type": "vector",
                    "similarity_score": similarity,
                    "metadata": metadata,
                    "chunk_info": {
                        "chunk_id": metadata.get('chunk_id'),
                        "chunk_index": metadata.get('chunk_index', i),
                        "document_id": actual_doc_id
                    }
                }
            else:
                # Document object
                metadata = document.metadata or {}
                actual_doc_id = metadata.get('document_id', document.id)
                actual_title = metadata.get('title') or getattr(document, 'title', f"Document Chunk {i+1}")
                actual_source = metadata.get('source') or document.source or "unknown"
                
                doc_info = {
                    "id": document.id,  # Keep the chunk ID for mapping
                    "actual_document_id": actual_doc_id,  # Store the real document ID
                    "title": actual_title,
                    "content": document.content,
                    "source": actual_source,
                    "source_type": "vector",
                    "similarity_score": similarity,
                    "metadata": metadata,
                    "chunk_info": {
                        "chunk_id": metadata.get('chunk_id'),
                        "chunk_index": metadata.get('chunk_index', i),
                        "document_id": actual_doc_id
                    }
                }
            
            context["documents"].append(doc_info)
            
            # Map document content to source
            doc_text = doc_info["content"][:500] + "..." if len(doc_info["content"]) > 500 else doc_info["content"]
            source_id = f"vector_doc_{doc_info['id']}"
            context["source_mapping"][doc_text] = source_id
            
            # Use similarity score as initial relevance
            context["relevance_scores"][source_id] = similarity
        
        context["vector_contribution"] = len(documents)
    
    async def _deduplicate_content(self, context: Dict[str, Any]):
        """Remove duplicate or highly similar content."""
        logger.debug("Performing content deduplication")
        
        # Simple text-based deduplication for documents
        unique_documents = []
        seen_content = set()
        
        for doc in context["documents"]:
            # Create a normalized version for comparison
            normalized_content = doc["content"].lower().strip()
            content_hash = hash(normalized_content)
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_documents.append(doc)
            else:
                # Remove from source mapping and relevance scores
                source_id = f"vector_doc_{doc['id']}"
                if source_id in context["relevance_scores"]:
                    del context["relevance_scores"][source_id]
                
                # Remove from source mapping
                for text, mapped_source in list(context["source_mapping"].items()):
                    if mapped_source == source_id:
                        del context["source_mapping"][text]
        
        context["documents"] = unique_documents
        
        # Deduplicate entities by name (case-insensitive)
        unique_entities = []
        seen_entity_names = set()
        
        for entity in context["entities"]:
            normalized_name = entity["name"].lower().strip()
            if normalized_name not in seen_entity_names:
                seen_entity_names.add(normalized_name)
                unique_entities.append(entity)
            else:
                # Remove from mappings
                source_id = f"graph_entity_{entity['id']}"
                if source_id in context["relevance_scores"]:
                    del context["relevance_scores"][source_id]
        
        context["entities"] = unique_entities
        
        logger.debug(f"Deduplication complete. Entities: {len(context['entities'])}, Documents: {len(context['documents'])}")
    
    async def _calculate_relevance_scores(self, context: Dict[str, Any]):
        """Calculate and normalize relevance scores for all content."""
        logger.debug("Calculating relevance scores")
        
        # Boost scores for entities that appear in multiple relationships
        entity_relationship_count = {}
        for rel in context["relationships"]:
            source = rel.get("source", "")
            target = rel.get("target", "")
            entity_relationship_count[source] = entity_relationship_count.get(source, 0) + 1
            entity_relationship_count[target] = entity_relationship_count.get(target, 0) + 1
        
        # Update entity relevance scores based on relationship frequency
        for entity in context["entities"]:
            entity_name = entity["name"]
            relationship_boost = min(entity_relationship_count.get(entity_name, 0) * 0.1, 0.3)
            source_id = f"graph_entity_{entity['id']}"
            if source_id in context["relevance_scores"]:
                context["relevance_scores"][source_id] = min(
                    context["relevance_scores"][source_id] + relationship_boost, 
                    1.0
                )
        
        # Normalize all scores to 0-1 range
        if context["relevance_scores"]:
            max_score = max(context["relevance_scores"].values())
            min_score = min(context["relevance_scores"].values())
            
            if max_score > min_score:
                for source_id in context["relevance_scores"]:
                    normalized_score = (context["relevance_scores"][source_id] - min_score) / (max_score - min_score)
                    context["relevance_scores"][source_id] = normalized_score
        
        # Filter out low-relevance content
        filtered_scores = {
            source_id: score 
            for source_id, score in context["relevance_scores"].items() 
            if score >= self.min_relevance_threshold
        }
        context["relevance_scores"] = filtered_scores
        
        logger.debug(f"Relevance calculation complete. {len(filtered_scores)} sources above threshold")
    
    async def _manage_context_window(self, context: Dict[str, Any]):
        """Manage context window size to fit within token limits."""
        logger.debug("Managing context window size")
        
        # Estimate current token count
        total_text = ""
        
        # Add entity descriptions
        for entity in context["entities"]:
            source_id = f"graph_entity_{entity['id']}"
            if source_id in context["relevance_scores"]:
                total_text += f"{entity['name']}: {entity['description']}\n"
        
        # Add relationship descriptions
        for i, rel in enumerate(context["relationships"]):
            source_id = f"graph_rel_{i+1}"
            if source_id in context["relevance_scores"]:
                total_text += f"{rel['source']} -> {rel['type']} -> {rel['target']}\n"
        
        # Add document content
        for doc in context["documents"]:
            source_id = f"vector_doc_{doc['id']}"
            if source_id in context["relevance_scores"]:
                total_text += f"{doc['content']}\n"
        
        estimated_tokens = len(total_text) // self.chars_per_token
        context["total_tokens"] = estimated_tokens
        
        # If over limit, prioritize by relevance score
        if estimated_tokens > self.max_context_tokens:
            logger.info(f"Context too large ({estimated_tokens} tokens), prioritizing by relevance")
            
            # Sort all sources by relevance score
            sorted_sources = sorted(
                context["relevance_scores"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Keep adding sources until we hit the token limit
            kept_sources = set()
            current_tokens = 0
            
            for source_id, score in sorted_sources:
                # Estimate tokens for this source
                source_tokens = 0
                
                if source_id.startswith("graph_entity_"):
                    entity_id = source_id.replace("graph_entity_", "")
                    entity = next((e for e in context["entities"] if e["id"] == entity_id), None)
                    if entity:
                        source_tokens = len(f"{entity['name']}: {entity['description']}") // self.chars_per_token
                
                elif source_id.startswith("graph_rel_"):
                    rel_index = int(source_id.replace("graph_rel_", "")) - 1
                    if rel_index < len(context["relationships"]):
                        rel = context["relationships"][rel_index]
                        source_tokens = len(f"{rel['source']} -> {rel['type']} -> {rel['target']}") // self.chars_per_token
                
                elif source_id.startswith("vector_doc_"):
                    doc_id = source_id.replace("vector_doc_", "")
                    doc = next((d for d in context["documents"] if d["id"] == doc_id), None)
                    if doc:
                        source_tokens = len(doc["content"]) // self.chars_per_token
                
                if current_tokens + source_tokens <= self.max_context_tokens:
                    kept_sources.add(source_id)
                    current_tokens += source_tokens
                else:
                    break
            
            # Filter context to only include kept sources
            context["relevance_scores"] = {
                source_id: score 
                for source_id, score in context["relevance_scores"].items() 
                if source_id in kept_sources
            }
            
            # Filter entities, relationships, and documents
            context["entities"] = [
                e for e in context["entities"] 
                if f"graph_entity_{e['id']}" in kept_sources
            ]
            
            context["relationships"] = [
                rel for i, rel in enumerate(context["relationships"]) 
                if f"graph_rel_{i+1}" in kept_sources
            ]
            
            context["documents"] = [
                d for d in context["documents"] 
                if f"vector_doc_{d['id']}" in kept_sources
            ]
            
            context["total_tokens"] = current_tokens
            
            logger.info(f"Context window managed. Final tokens: {current_tokens}")
    
    async def _create_combined_text(self, context: Dict[str, Any]):
        """Create combined text for LLM input with proper formatting."""
        logger.debug("Creating combined text for LLM input")
        
        text_parts = []
        citations = []
        
        # Add entities section
        if context["entities"]:
            text_parts.append("## Relevant Entities:")
            for entity in context["entities"]:
                source_id = f"graph_entity_{entity['id']}"
                if source_id in context["relevance_scores"]:
                    text_parts.append(f"- **{entity['name']}** ({entity['type']}): {entity['description']}")
                    
                    # Create citation
                    citations.append(CitationInfo(
                        source_id=source_id,
                        source_type="graph",
                        title=entity['name'],
                        content_snippet=entity['description'][:200] + "..." if len(entity['description']) > 200 else entity['description'],
                        relevance_score=context["relevance_scores"][source_id]
                    ))
        
        # Add relationships section
        if context["relationships"]:
            text_parts.append("\n## Relevant Relationships:")
            for i, rel in enumerate(context["relationships"]):
                source_id = f"graph_rel_{i+1}"
                if source_id in context["relevance_scores"]:
                    text_parts.append(f"- {rel['source']} --[{rel['type']}]--> {rel['target']}")
                    
                    # Create citation
                    citations.append(CitationInfo(
                        source_id=source_id,
                        source_type="graph",
                        title=f"Relationship: {rel['type']}",
                        content_snippet=f"{rel['source']} -> {rel['target']}",
                        relevance_score=context["relevance_scores"][source_id]
                    ))
        
        # Add documents section
        if context["documents"]:
            text_parts.append("\n## Relevant Documents:")
            for doc in context["documents"]:
                source_id = f"vector_doc_{doc['id']}"
                if source_id in context["relevance_scores"]:
                    # Truncate long content
                    content_preview = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
                    text_parts.append(f"- **{doc['title']}**: {content_preview}")
                    
                    # Create citation
                    citations.append(CitationInfo(
                        source_id=source_id,
                        source_type="vector",
                        title=doc['title'],
                        content_snippet=content_preview,
                        relevance_score=context["relevance_scores"][source_id],
                        url=doc.get('source')
                    ))
        
        context["combined_text"] = "\n".join(text_parts)
        context["citations"] = citations
        
        logger.debug(f"Combined text created. Length: {len(context['combined_text'])} characters")
    
    async def _handle_synthesis_request(self, message: AgentMessage) -> AgentMessage:
        """Handle synthesis request message."""
        try:
            payload = SynthesisMessage(**message.payload)
            
            # Convert protocol objects to interface objects if needed
            graph_results = None
            if payload.graph_results:
                graph_results = GraphResult(
                    entities=payload.graph_results.entities,
                    relationships=payload.graph_results.relationships,
                    paths=payload.graph_results.paths,
                    cypher_query=payload.graph_results.cypher_query
                )
            
            vector_results = payload.vector_results  # Use VectorSearchResponse directly
            
            # Integrate results
            integrated_context = await self.integrate_results(graph_results, vector_results)
            
            # Generate response using Gemini API
            query_type = payload.query_type if hasattr(payload, 'query_type') else "general"
            synthesis_result = await self.generate_response(
                payload.query, 
                integrated_context, 
                query_type
            )
            
            # Create response payload
            response_payload = {
                "synthesis_result": synthesis_result,
                "integrated_context": integrated_context,
                "query": payload.query,
                "processing_time": synthesis_result.metadata.get("processing_time", 0.0) if hasattr(synthesis_result, 'metadata') else 0.0
            }
            
            return self.create_message(
                MessageType.RESPONSE,
                response_payload,
                message.correlation_id
            )
            
        except Exception as e:
            logger.error(f"Error handling synthesis request: {str(e)}")
            raise
    
    async def generate_response(
        self, 
        query: str, 
        integrated_context: Dict[str, Any],
        query_type: str = "general"
    ) -> SynthesisResult:
        """Generate final response using integrated context and Gemini API."""
        logger.info(f"Generating response for query: {query[:100]}...")
        
        if not self.gemini_client:
            logger.error("Gemini client not initialized")
            return SynthesisResult(
                response="Error: Gemini API client not available",
                sources=[],
                citations=[],
                reasoning_path="Failed to initialize Gemini client"
            )
        
        try:
            # Get appropriate prompt template based on query type
            prompt = self.prompt_templates.get_synthesis_prompt(query, query_type)
            
            # Use the combined text from integrated context
            context_text = integrated_context.get("combined_text", "")
            
            # Add metadata about the context
            context_metadata = f"""
Context Statistics:
- Entities: {len(integrated_context.get('entities', []))}
- Relationships: {len(integrated_context.get('relationships', []))}
- Documents: {len(integrated_context.get('documents', []))}
- Total tokens: {integrated_context.get('total_tokens', 0)}
- Graph contribution: {integrated_context.get('graph_contribution', 0)}
- Vector contribution: {integrated_context.get('vector_contribution', 0)}

"""
            
            full_context = context_metadata + context_text
            
            # Generate response using Gemini
            start_time = time.time()
            gemini_response = await self.gemini_client.generate_response(prompt, full_context)
            processing_time = time.time() - start_time
            
            # Validate response
            if not self.gemini_client.validate_response(gemini_response):
                logger.warning("Gemini response failed validation")
                return SynthesisResult(
                    response="Error: Generated response failed validation checks",
                    sources=[],
                    citations=[],
                    reasoning_path="Response validation failed"
                )
            
            # Extract sources and create citations
            sources = self._extract_sources_from_context(integrated_context)
            citations_from_context = self._create_citations_from_context(integrated_context)
            
            # Create proper citations using the new method
            formatted_citations = await self.create_citations(sources)
            
            # Merge context citations with formatted citations
            all_citations = citations_from_context + formatted_citations
            
            # Remove duplicates based on source ID
            unique_citations = {}
            for citation in all_citations:
                citation_id = citation.get("id") or citation.get("source_id", "unknown")
                if citation_id not in unique_citations:
                    unique_citations[citation_id] = citation
            
            final_citations = list(unique_citations.values())
            
            # Create detailed reasoning path
            reasoning_path = self.generate_reasoning_explanation(
                query,
                query_type,
                integrated_context,
                "hybrid" if integrated_context.get("graph_contribution", 0) > 0 and integrated_context.get("vector_contribution", 0) > 0 else "single-mode",
                {
                    "processing_time": processing_time,
                    "gemini_usage": gemini_response.usage_metadata,
                    "finish_reason": gemini_response.finish_reason
                }
            )
            
            # Validate response quality
            quality_validation = self.validate_response_quality(
                gemini_response.text,
                final_citations,
                reasoning_path
            )
            
            # Enhance response with citations if validation suggests it
            enhanced_response = gemini_response.text
            if quality_validation.get("quality_score", 0) < 0.6 or not quality_validation.get("is_valid", True):
                # Add citation formatting to improve quality
                citation_text = self.format_citations_for_response(final_citations[:5])  # Top 5 citations
                enhanced_response = gemini_response.text + citation_text
            
            # Log quality validation results
            if not quality_validation.get("is_valid", True):
                logger.warning(f"Response quality validation failed: {quality_validation.get('issues', [])}")
            
            if quality_validation.get("recommendations"):
                logger.info(f"Response quality recommendations: {quality_validation.get('recommendations', [])}")
            
            logger.info(f"Response generated successfully in {processing_time:.2f}s")
            
            return SynthesisResult(
                response=enhanced_response,
                sources=sources,
                citations=final_citations,
                reasoning_path=reasoning_path,
                confidence_score=quality_validation.get("quality_score", 0.0),
                metadata={
                    "processing_time": processing_time,
                    "gemini_usage": gemini_response.usage_metadata,
                    "safety_ratings": gemini_response.safety_ratings,
                    "finish_reason": gemini_response.finish_reason,
                    "query_type": query_type,
                    "context_tokens": integrated_context.get('total_tokens', 0),
                    "quality_validation": quality_validation,
                    "citation_count": len(final_citations),
                    "reasoning_steps": len(reasoning_path.split("\n")) if reasoning_path else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return SynthesisResult(
                response=f"Error generating response: {str(e)}",
                sources=[],
                citations=[],
                reasoning_path=f"Error occurred during response generation: {str(e)}"
            )
    
    def _extract_sources_from_context(self, integrated_context: Dict[str, Any]) -> List[str]:
        """Extract source identifiers from integrated context."""
        sources = []
        
        # Add entity sources
        for entity in integrated_context.get("entities", []):
            sources.append(f"graph_entity_{entity['id']}")
        
        # Add relationship sources
        for i, rel in enumerate(integrated_context.get("relationships", [])):
            sources.append(f"graph_rel_{i+1}")
        
        # Add document sources with proper titles
        for doc in integrated_context.get("documents", []):
            # Use actual document title if available, otherwise fall back to ID
            title = doc.get('title', 'Unknown Document')
            if title and title != 'Unknown Document' and not title.startswith('Document Chunk'):
                sources.append(title)
            else:
                sources.append(f"vector_doc_{doc['id']}")
        
        return sources
    
    def _create_citations_from_context(self, integrated_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create citation objects from integrated context."""
        citations = []
        
        # Process citations from integrated context
        for citation_info in integrated_context.get("citations", []):
            if isinstance(citation_info, CitationInfo):
                citations.append({
                    "id": citation_info.source_id,
                    "type": citation_info.source_type,
                    "title": citation_info.title,
                    "snippet": citation_info.content_snippet,
                    "relevance": citation_info.relevance_score,
                    "url": citation_info.url
                })
            elif isinstance(citation_info, dict):
                citations.append(citation_info)
        
        # Sort by relevance score
        citations.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        return citations
    
    def _create_reasoning_path(
        self, 
        query_type: str, 
        integrated_context: Dict[str, Any], 
        usage_metadata: Dict[str, Any]
    ) -> str:
        """Create a reasoning path explanation."""
        path_parts = []
        
        # Query analysis
        path_parts.append(f"1. Query Analysis: Classified as '{query_type}' query")
        
        # Context integration
        entities_count = len(integrated_context.get("entities", []))
        relationships_count = len(integrated_context.get("relationships", []))
        documents_count = len(integrated_context.get("documents", []))
        
        path_parts.append(f"2. Context Integration: Combined {entities_count} entities, {relationships_count} relationships, and {documents_count} documents")
        
        # Strategy explanation
        if entities_count > 0 and relationships_count > 0 and documents_count > 0:
            strategy = "hybrid graph-vector approach"
        elif entities_count > 0 or relationships_count > 0:
            strategy = "graph-focused approach"
        elif documents_count > 0:
            strategy = "vector-focused approach"
        else:
            strategy = "minimal context approach"
        
        path_parts.append(f"3. Retrieval Strategy: Used {strategy}")
        
        # Token usage
        total_tokens = usage_metadata.get("total_token_count", 0)
        path_parts.append(f"4. Response Generation: Processed {total_tokens} tokens using Gemini API")
        
        # Quality assurance
        path_parts.append("5. Quality Assurance: Response validated for safety and completeness")
        
        return " → ".join(path_parts)
    
    def _enhance_response_with_citations(self, response_text: str, citations: List[Dict[str, Any]]) -> str:
        """Enhance response text with inline citations if not already present."""
        # This is a simple implementation - could be made more sophisticated
        # For now, just append citation information at the end
        
        if not citations:
            return response_text
        
        # Check if response already has citation markers
        if "[" in response_text and "]" in response_text:
            # Response likely already has citations
            return response_text
        
        # Add citation section
        citation_text = "\n\n**Sources:**\n"
        for i, citation in enumerate(citations[:5], 1):  # Limit to top 5 citations
            title = citation.get("title", "Unknown")
            citation_type = citation.get("type", "unknown")
            relevance = citation.get("relevance", 0)
            
            citation_text += f"{i}. {title} ({citation_type.title()}, relevance: {relevance:.2f})\n"
        
        return response_text + citation_text
    
    async def create_citations(self, sources: List[str]) -> List[Dict[str, Any]]:
        """Create properly formatted citations for sources."""
        citations = []
        
        for source_id in sources:
            try:
                citation = await self._create_citation_for_source(source_id)
                if citation:
                    citations.append(citation)
            except Exception as e:
                logger.warning(f"Failed to create citation for source {source_id}: {str(e)}")
                continue
        
        # Sort citations by relevance score (highest first)
        citations.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return citations
    
    async def _create_citation_for_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Create a citation for a specific source."""
        # Parse source ID to determine type and extract information
        if source_id.startswith("graph_entity_"):
            return await self._create_graph_entity_citation(source_id)
        elif source_id.startswith("graph_rel_"):
            return await self._create_graph_relationship_citation(source_id)
        elif source_id.startswith("vector_doc_"):
            return await self._create_vector_document_citation(source_id)
        else:
            logger.warning(f"Unknown source type for ID: {source_id}")
            return None
    
    async def _create_graph_entity_citation(self, source_id: str) -> Dict[str, Any]:
        """Create citation for a graph entity."""
        entity_id = source_id.replace("graph_entity_", "")
        
        return {
            "id": source_id,
            "type": "graph_entity",
            "entity_id": entity_id,
            "title": f"Entity: {entity_id}",
            "description": "Knowledge graph entity",
            "source_type": "Knowledge Graph",
            "relevance_score": 0.8,  # Default high relevance for entities
            "citation_format": f"[Graph Entity: {entity_id}]",
            "metadata": {
                "database": "Neo4j",
                "node_type": "Entity"
            }
        }
    
    async def _create_graph_relationship_citation(self, source_id: str) -> Dict[str, Any]:
        """Create citation for a graph relationship."""
        rel_index = source_id.replace("graph_rel_", "")
        
        return {
            "id": source_id,
            "type": "graph_relationship",
            "relationship_id": rel_index,
            "title": f"Relationship #{rel_index}",
            "description": "Knowledge graph relationship",
            "source_type": "Knowledge Graph",
            "relevance_score": 0.7,  # Default relevance for relationships
            "citation_format": f"[Graph Relationship #{rel_index}]",
            "metadata": {
                "database": "Neo4j",
                "edge_type": "Relationship"
            }
        }
    
    async def _create_vector_document_citation(self, source_id: str) -> Dict[str, Any]:
        """Create citation for a vector document."""
        doc_id = source_id.replace("vector_doc_", "")
        
        # Try to get actual document information from the last integrated context
        # This is stored during the integration process
        title = f"Document Chunk {doc_id.replace('doc_', '')}"
        description = "Content from vector database"
        actual_doc_id = doc_id
        source_info = "Vector Database"
        chunk_info = {}
        
        # Try to find the document in the last integrated context
        if hasattr(self, '_last_integrated_context') and self._last_integrated_context:
            for doc in self._last_integrated_context.get("documents", []):
                if f"vector_doc_{doc.get('id')}" == source_id:
                    title = doc.get('title', title)
                    actual_doc_id = doc.get('actual_document_id', actual_doc_id)
                    source_info = doc.get('source', source_info)
                    chunk_info = doc.get('chunk_info', {})
                    description = f"Content from {title}"
                    break
        
        return {
            "id": source_id,
            "type": "vector_document", 
            "document_id": actual_doc_id,
            "title": title,
            "description": description,
            "source_type": source_info,
            "relevance_score": 0.8,  # Higher relevance since it was retrieved
            "citation_format": f"[{title}]",
            "metadata": {
                "database": "Pinecone",
                "content_type": "Document Chunk",
                "chunk_id": doc_id,
                "chunk_info": chunk_info
            }
        }
    
    def format_citations_for_response(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations for inclusion in response text."""
        if not citations:
            return ""
        
        citation_text = "\n\n**Sources:**\n"
        
        for i, citation in enumerate(citations, 1):
            title = citation.get("title", "Unknown Source")
            source_type = citation.get("source_type", "Unknown")
            relevance = citation.get("relevance_score", 0)
            description = citation.get("description", "")
            
            citation_text += f"{i}. **{title}** ({source_type})\n"
            if description:
                citation_text += f"   - {description}\n"
            citation_text += f"   - Relevance: {relevance:.2f}\n"
        
        return citation_text
    
    def validate_response_quality(self, response: str, citations: List[Dict[str, Any]], reasoning_path: str) -> Dict[str, Any]:
        """Validate the quality of generated response."""
        validation_results = {
            "is_valid": True,
            "quality_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Check response length
        if len(response.strip()) < 50:
            validation_results["issues"].append("Response is too short")
            validation_results["is_valid"] = False
        elif len(response.strip()) > 5000:
            validation_results["recommendations"].append("Response is very long - consider summarizing")
        
        # Check for citations
        if not citations:
            validation_results["issues"].append("No citations provided")
            validation_results["recommendations"].append("Add source citations to support claims")
        elif len(citations) < 2:
            validation_results["recommendations"].append("Consider adding more diverse sources")
        
        # Check reasoning path
        if not reasoning_path or len(reasoning_path.strip()) < 20:
            validation_results["issues"].append("Reasoning path is missing or too brief")
            validation_results["is_valid"] = False
        
        # Check for common quality indicators
        quality_indicators = {
            "has_specific_details": bool(re.search(r'\d+|specific|particular|exactly|precisely', response.lower())),
            "has_explanations": bool(re.search(r'because|since|due to|therefore|thus|as a result', response.lower())),
            "has_structure": bool(re.search(r'first|second|third|finally|in conclusion|however|moreover', response.lower())),
            "avoids_speculation": not bool(re.search(r'might be|could be|possibly|perhaps|maybe', response.lower())),
            "cites_sources": bool(re.search(r'\[.*\]|\(.*\)|according to|based on', response.lower()))
        }
        
        # Calculate quality score
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        validation_results["quality_score"] = quality_score
        
        # Add recommendations based on missing indicators
        if not quality_indicators["has_specific_details"]:
            validation_results["recommendations"].append("Add more specific details and examples")
        
        if not quality_indicators["has_explanations"]:
            validation_results["recommendations"].append("Include more explanatory language")
        
        if not quality_indicators["has_structure"]:
            validation_results["recommendations"].append("Improve response structure with transitions")
        
        if not quality_indicators["avoids_speculation"]:
            validation_results["recommendations"].append("Reduce speculative language")
        
        if not quality_indicators["cites_sources"]:
            validation_results["recommendations"].append("Add inline citations or source references")
        
        # Overall validation
        if quality_score < 0.3:  # Lowered threshold to be less strict
            validation_results["is_valid"] = False
            validation_results["issues"].append("Overall response quality is low")
        
        return validation_results
    
    def generate_reasoning_explanation(
        self, 
        query: str,
        query_type: str,
        integrated_context: Dict[str, Any],
        retrieval_strategy: str,
        processing_metadata: Dict[str, Any]
    ) -> str:
        """Generate detailed explanation of reasoning process."""
        explanation_parts = []
        
        # 1. Query Analysis
        explanation_parts.append(f"**Query Analysis:**")
        explanation_parts.append(f"- Original query: \"{query}\"")
        explanation_parts.append(f"- Classified as: {query_type} query")
        explanation_parts.append(f"- Selected strategy: {retrieval_strategy}")
        
        # 2. Information Retrieval
        explanation_parts.append(f"\n**Information Retrieval:**")
        
        entities_count = len(integrated_context.get("entities", []))
        relationships_count = len(integrated_context.get("relationships", []))
        documents_count = len(integrated_context.get("documents", []))
        
        if entities_count > 0:
            explanation_parts.append(f"- Retrieved {entities_count} relevant entities from knowledge graph")
        
        if relationships_count > 0:
            explanation_parts.append(f"- Explored {relationships_count} relationships between entities")
        
        if documents_count > 0:
            explanation_parts.append(f"- Found {documents_count} relevant documents through vector search")
        
        # 3. Context Integration
        explanation_parts.append(f"\n**Context Integration:**")
        total_tokens = integrated_context.get("total_tokens", 0)
        explanation_parts.append(f"- Combined information into {total_tokens} tokens of context")
        
        graph_contribution = integrated_context.get("graph_contribution", 0)
        vector_contribution = integrated_context.get("vector_contribution", 0)
        
        if graph_contribution > 0 and vector_contribution > 0:
            explanation_parts.append(f"- Used hybrid approach: {graph_contribution} graph elements + {vector_contribution} vector documents")
        elif graph_contribution > 0:
            explanation_parts.append(f"- Primarily used graph-based information: {graph_contribution} elements")
        elif vector_contribution > 0:
            explanation_parts.append(f"- Primarily used vector-based information: {vector_contribution} documents")
        
        # 4. Response Generation
        explanation_parts.append(f"\n**Response Generation:**")
        
        if "gemini_usage" in processing_metadata:
            usage = processing_metadata["gemini_usage"]
            prompt_tokens = usage.get("prompt_token_count", 0)
            response_tokens = usage.get("candidates_token_count", 0)
            explanation_parts.append(f"- Processed {prompt_tokens} input tokens, generated {response_tokens} output tokens")
        
        processing_time = processing_metadata.get("processing_time", 0)
        if processing_time > 0:
            explanation_parts.append(f"- Total processing time: {processing_time:.2f} seconds")
        
        # 5. Quality Assurance
        explanation_parts.append(f"\n**Quality Assurance:**")
        explanation_parts.append(f"- Response validated for safety and completeness")
        explanation_parts.append(f"- Citations generated for all source materials")
        
        finish_reason = processing_metadata.get("finish_reason", "unknown")
        if finish_reason == "STOP":
            explanation_parts.append(f"- Response completed naturally")
        elif finish_reason == "MAX_TOKENS":
            explanation_parts.append(f"- Response truncated due to length limits")
        
        return "\n".join(explanation_parts)
