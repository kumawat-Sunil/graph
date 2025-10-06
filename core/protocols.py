"""
Message protocol classes for agent communication in the Graph-Enhanced Agentic RAG system.

This module defines the structured message formats and communication protocols
used between different agents in the system.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import uuid

from .interfaces import MessageType, QueryType, RetrievalStrategy
from .models import Entity, Document, Concept


class MessagePriority(str, Enum):
    """Priority levels for agent messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageStatus(str, Enum):
    """Status of message processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AgentMessage(BaseModel):
    """
    Enhanced message format for agent communication.
    
    This is the base message structure used for all inter-agent communication
    in the system, providing standardized format and metadata.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., description="ID of the sending agent")
    target_agent_id: Optional[str] = Field(None, description="ID of the target agent")
    message_type: MessageType = Field(..., description="Type of message")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
    status: MessageStatus = Field(default=MessageStatus.PENDING)
    timestamp: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = Field(None, description="ID to track related messages")
    parent_message_id: Optional[str] = Field(None, description="ID of parent message if this is a response")
    timeout_seconds: Optional[int] = Field(default=30, description="Message timeout in seconds")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    @validator('agent_id', 'target_agent_id')
    def validate_agent_ids(cls, v):
        """Validate agent ID format."""
        if v is not None and not v.strip():
            raise ValueError("Agent ID cannot be empty")
        return v
    
    @validator('payload')
    def validate_payload(cls, v):
        """Validate message payload."""
        if v is None:
            return {}
        # Ensure payload is JSON serializable
        try:
            import json
            json.dumps(v)
        except (TypeError, ValueError):
            raise ValueError("Message payload must be JSON serializable")
        return v
    
    def is_expired(self) -> bool:
        """Check if message has expired based on timeout."""
        if self.timeout_seconds is None:
            return False
        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.timeout_seconds
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1


class QueryAnalysisMessage(BaseModel):
    """Message payload for query analysis requests."""
    query: str = Field(..., min_length=1)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    user_id: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query text."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class QueryAnalysisResponse(BaseModel):
    """Response payload for query analysis."""
    query: str
    entities: List[str] = Field(default_factory=list)
    query_type: QueryType
    complexity_score: float = Field(ge=0.0, le=1.0)
    requires_graph: bool = False
    requires_vector: bool = False
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)
    extracted_keywords: List[str] = Field(default_factory=list)
    suggested_strategy: Optional[RetrievalStrategy] = Field(None)


class StrategySelectionMessage(BaseModel):
    """Message payload for strategy selection requests."""
    query_analysis: QueryAnalysisResponse
    available_strategies: List[RetrievalStrategy] = Field(default_factory=list)
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict)


class StrategySelectionResponse(BaseModel):
    """Response payload for strategy selection."""
    selected_strategy: RetrievalStrategy
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    fallback_strategies: List[RetrievalStrategy] = Field(default_factory=list)
    estimated_execution_time: Optional[float] = Field(None, description="Estimated time in seconds")


class GraphSearchMessage(BaseModel):
    """Message payload for graph search requests."""
    entities: List[str] = Field(default_factory=list)
    query: str
    max_depth: int = Field(default=2, ge=1, le=5)
    max_results: int = Field(default=50, ge=1, le=1000)
    relationship_types: Optional[List[str]] = Field(None)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GraphSearchResponse(BaseModel):
    """Response payload for graph search results."""
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    paths: List[List[str]] = Field(default_factory=list)
    cypher_query: Optional[str] = Field(None)
    execution_time: Optional[float] = Field(None, description="Query execution time in seconds")
    total_results: int = Field(default=0)
    has_more_results: bool = Field(default=False)


class VectorSearchMessage(BaseModel):
    """Message payload for vector search requests."""
    query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    include_embeddings: bool = Field(default=False)
    rerank: bool = Field(default=True)


class VectorSearchResponse(BaseModel):
    """Response payload for vector search results."""
    documents: List[Document] = Field(default_factory=list)
    similarities: List[float] = Field(default_factory=list)
    query_embedding: Optional[List[float]] = Field(None)
    execution_time: Optional[float] = Field(None, description="Search execution time in seconds")
    total_results: int = Field(default=0)
    reranked: bool = Field(default=False)


class SynthesisMessage(BaseModel):
    """Message payload for synthesis requests."""
    query: str = Field(..., min_length=1)
    graph_results: Optional[GraphSearchResponse] = Field(None)
    vector_results: Optional[VectorSearchResponse] = Field(None)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    response_format: str = Field(default="natural", description="Format for response generation")
    max_response_length: int = Field(default=1000, ge=100, le=5000)
    include_citations: bool = Field(default=True)
    include_reasoning: bool = Field(default=True)


class SynthesisResponse(BaseModel):
    """Response payload for synthesis results."""
    response: str
    sources: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_path: Optional[str] = Field(None)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    used_graph_results: bool = Field(default=False)
    used_vector_results: bool = Field(default=False)
    generation_time: Optional[float] = Field(None, description="Response generation time in seconds")


class ErrorMessage(BaseModel):
    """Message payload for error notifications."""
    error_type: str
    error_message: str
    error_code: Optional[str] = Field(None)
    stack_trace: Optional[str] = Field(None)
    original_message_id: Optional[str] = Field(None)
    recoverable: bool = Field(default=True)
    suggested_action: Optional[str] = Field(None)


class HealthCheckMessage(BaseModel):
    """Message payload for health check requests."""
    check_type: str = Field(default="basic")
    include_metrics: bool = Field(default=False)


class HealthCheckResponse(BaseModel):
    """Response payload for health check results."""
    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    timestamp: datetime = Field(default_factory=datetime.now)
    response_time_ms: float
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Status of dependencies")


class MessageValidator:
    """Validator for agent messages and payloads."""
    
    @staticmethod
    def validate_message(message: AgentMessage) -> bool:
        """
        Validate a complete agent message.
        
        Args:
            message: AgentMessage to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If message is invalid
        """
        try:
            # Basic validation is handled by Pydantic
            # Additional custom validation can be added here
            
            # Check if message has expired
            if message.is_expired():
                raise ValueError("Message has expired")
            
            # Validate payload based on message type
            MessageValidator._validate_payload_for_type(message.message_type, message.payload)
            
            return True
        except Exception as e:
            raise ValueError(f"Invalid message: {str(e)}")
    
    @staticmethod
    def _validate_payload_for_type(message_type: MessageType, payload: Dict[str, Any]) -> None:
        """Validate payload based on message type."""
        try:
            if message_type == MessageType.QUERY_ANALYSIS:
                QueryAnalysisMessage(**payload)
            elif message_type == MessageType.STRATEGY_SELECTION:
                StrategySelectionMessage(**payload)
            elif message_type == MessageType.GRAPH_SEARCH:
                GraphSearchMessage(**payload)
            elif message_type == MessageType.VECTOR_SEARCH:
                VectorSearchMessage(**payload)
            elif message_type == MessageType.SYNTHESIS_REQUEST:
                SynthesisMessage(**payload)
            elif message_type == MessageType.ERROR:
                ErrorMessage(**payload)
            # Add more validations as needed
        except Exception as e:
            raise ValueError(f"Invalid payload for message type {message_type}: {str(e)}")
    
    @staticmethod
    def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize message payload to remove potentially harmful content.
        
        Args:
            payload: Original payload
            
        Returns:
            Dict: Sanitized payload
        """
        sanitized = {}
        
        for key, value in payload.items():
            # Remove potentially harmful keys
            if key.startswith('_') or key in ['__class__', '__module__']:
                continue
            
            # Sanitize string values
            if isinstance(value, str):
                # Remove potential script tags or other harmful content
                import re
                value = re.sub(r'<script.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
                value = re.sub(r'javascript:', '', value, flags=re.IGNORECASE)
            
            sanitized[key] = value
        
        return sanitized


class MessageQueue:
    """Simple message queue for agent communication."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._messages: List[AgentMessage] = []
        self._processed_messages: List[str] = []  # Track processed message IDs
    
    def enqueue(self, message: AgentMessage) -> bool:
        """
        Add a message to the queue.
        
        Args:
            message: Message to enqueue
            
        Returns:
            bool: True if successfully enqueued
        """
        if len(self._messages) >= self.max_size:
            # Remove oldest message
            self._messages.pop(0)
        
        # Validate message before enqueuing
        try:
            MessageValidator.validate_message(message)
            self._messages.append(message)
            return True
        except ValueError:
            return False
    
    def dequeue(self, agent_id: str) -> Optional[AgentMessage]:
        """
        Get the next message for a specific agent.
        
        Args:
            agent_id: ID of the agent requesting messages
            
        Returns:
            Optional[AgentMessage]: Next message or None
        """
        for i, message in enumerate(self._messages):
            if (message.target_agent_id == agent_id or message.target_agent_id is None) and \
               message.id not in self._processed_messages:
                # Mark as processing
                message.status = MessageStatus.PROCESSING
                return message
        
        return None
    
    def mark_processed(self, message_id: str) -> None:
        """Mark a message as processed."""
        if message_id not in self._processed_messages:
            self._processed_messages.append(message_id)
        
        # Remove from active messages
        self._messages = [msg for msg in self._messages if msg.id != message_id]
    
    def get_pending_count(self, agent_id: str) -> int:
        """Get count of pending messages for an agent."""
        return len([
            msg for msg in self._messages 
            if (msg.target_agent_id == agent_id or msg.target_agent_id is None) and
               msg.id not in self._processed_messages
        ])
    
    def cleanup_expired(self) -> int:
        """Remove expired messages and return count of removed messages."""
        initial_count = len(self._messages)
        self._messages = [msg for msg in self._messages if not msg.is_expired()]
        return initial_count - len(self._messages)


# Factory functions for creating common message types

def create_query_analysis_message(
    agent_id: str,
    query: str,
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> AgentMessage:
    """Create a query analysis message."""
    payload = QueryAnalysisMessage(
        query=query,
        context=context or {}
    ).dict()
    
    return AgentMessage(
        agent_id=agent_id,
        message_type=MessageType.QUERY_ANALYSIS,
        payload=payload,
        correlation_id=correlation_id
    )


def create_graph_search_message(
    agent_id: str,
    entities: List[str],
    query: str,
    max_depth: int = 2,
    correlation_id: Optional[str] = None
) -> AgentMessage:
    """Create a graph search message."""
    payload = GraphSearchMessage(
        entities=entities,
        query=query,
        max_depth=max_depth
    ).dict()
    
    return AgentMessage(
        agent_id=agent_id,
        message_type=MessageType.GRAPH_SEARCH,
        payload=payload,
        correlation_id=correlation_id
    )


def create_vector_search_message(
    agent_id: str,
    query: str,
    k: int = 10,
    correlation_id: Optional[str] = None
) -> AgentMessage:
    """Create a vector search message."""
    payload = VectorSearchMessage(
        query=query,
        k=k
    ).dict()
    
    return AgentMessage(
        agent_id=agent_id,
        message_type=MessageType.VECTOR_SEARCH,
        payload=payload,
        correlation_id=correlation_id
    )


def create_synthesis_message(
    agent_id: str,
    query: str,
    graph_results: Optional[GraphSearchResponse] = None,
    vector_results: Optional[VectorSearchResponse] = None,
    correlation_id: Optional[str] = None
) -> AgentMessage:
    """Create a synthesis message."""
    payload = SynthesisMessage(
        query=query,
        graph_results=graph_results,
        vector_results=vector_results
    ).dict()
    
    return AgentMessage(
        agent_id=agent_id,
        message_type=MessageType.SYNTHESIS_REQUEST,
        payload=payload,
        correlation_id=correlation_id
    )


def create_error_message(
    agent_id: str,
    error_type: str,
    error_message: str,
    original_message_id: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> AgentMessage:
    """Create an error message."""
    payload = ErrorMessage(
        error_type=error_type,
        error_message=error_message,
        original_message_id=original_message_id
    ).dict()
    
    return AgentMessage(
        agent_id=agent_id,
        message_type=MessageType.ERROR,
        payload=payload,
        correlation_id=correlation_id
    )