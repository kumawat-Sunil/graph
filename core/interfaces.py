"""
Base agent interfaces and message protocols for the Graph-Enhanced Agentic RAG system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class MessageType(str, Enum):
    """Types of messages exchanged between agents."""
    QUERY_ANALYSIS = "query_analysis"
    STRATEGY_SELECTION = "strategy_selection"
    GRAPH_SEARCH = "graph_search"
    VECTOR_SEARCH = "vector_search"
    SYNTHESIS_REQUEST = "synthesis_request"
    RESPONSE = "response"
    ERROR = "error"


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    FACTUAL = "factual"
    RELATIONAL = "relational"
    MULTI_HOP = "multi_hop"
    COMPLEX = "complex"


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"


class AgentMessage(BaseModel):
    """Standard message format for agent communication."""
    agent_id: str = Field(..., description="ID of the sending agent")
    message_type: MessageType = Field(..., description="Type of message")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    correlation_id: Optional[str] = Field(None, description="ID to track related messages")


class QueryAnalysis(BaseModel):
    """Result of query analysis by the coordinator agent."""
    query: str
    entities: List[str] = Field(default_factory=list)
    query_type: QueryType
    complexity_score: float = Field(ge=0.0, le=1.0)
    requires_graph: bool = False
    requires_vector: bool = False


class Entity(BaseModel):
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Represents a document in the vector store."""
    id: str
    title: str = Field(..., min_length=1, max_length=1000)  # Required title
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    source: Optional[str] = None


class GraphResult(BaseModel):
    """Result from graph traversal operations."""
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    paths: List[List[str]] = Field(default_factory=list)
    cypher_query: Optional[str] = None


class VectorResult(BaseModel):
    """Result from vector similarity search."""
    documents: List[Document] = Field(default_factory=list)
    similarities: List[float] = Field(default_factory=list)
    query_embedding: Optional[List[float]] = None


class SynthesisResult(BaseModel):
    """Final synthesized response."""
    response: str
    sources: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_path: Optional[str] = None
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.message_handlers: Dict[MessageType, callable] = {}
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process an incoming message and return a response if needed."""
        pass
    
    def create_message(
        self, 
        message_type: MessageType, 
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """Create a standardized message."""
        return AgentMessage(
            agent_id=self.agent_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id
        )


class CoordinatorAgentInterface(BaseAgent):
    """Interface for the coordinator agent."""
    
    @abstractmethod
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a user query to determine entities and strategy."""
        pass
    
    @abstractmethod
    async def select_strategy(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """Select the optimal retrieval strategy based on query analysis."""
        pass
    
    @abstractmethod
    async def orchestrate_retrieval(
        self, 
        query: str, 
        strategy: RetrievalStrategy
    ) -> Union[GraphResult, VectorResult, tuple]:
        """Orchestrate the retrieval process using selected strategy."""
        pass


class GraphNavigatorInterface(BaseAgent):
    """Interface for the graph navigator agent."""
    
    @abstractmethod
    async def find_entities(self, query: str) -> List[Entity]:
        """Find entities in the graph matching the query."""
        pass
    
    @abstractmethod
    async def traverse_relationships(
        self, 
        entities: List[Entity], 
        depth: int = 2
    ) -> GraphResult:
        """Traverse relationships from given entities."""
        pass
    
    @abstractmethod
    async def execute_cypher_query(self, cypher: str, parameters: Dict = None) -> GraphResult:
        """Execute a Cypher query against the graph database."""
        pass


class VectorRetrievalInterface(BaseAgent):
    """Interface for the vector retrieval agent."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        pass
    
    @abstractmethod
    async def similarity_search(
        self, 
        query: str, 
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> VectorResult:
        """Perform similarity search in vector database."""
        pass
    
    @abstractmethod
    async def hybrid_search(
        self, 
        query: str, 
        semantic_weight: float = 0.7,
        k: int = 10
    ) -> VectorResult:
        """Perform hybrid semantic and keyword search."""
        pass


class SynthesisAgentInterface(BaseAgent):
    """Interface for the synthesis agent."""
    
    @abstractmethod
    async def integrate_results(
        self, 
        graph_results: Optional[GraphResult] = None,
        vector_results: Optional[VectorResult] = None
    ) -> Dict[str, Any]:
        """Integrate results from different retrieval methods."""
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        query: str, 
        integrated_context: Dict[str, Any]
    ) -> SynthesisResult:
        """Generate final response using integrated context."""
        pass
    
    @abstractmethod
    async def create_citations(self, sources: List[str]) -> List[Dict[str, Any]]:
        """Create properly formatted citations for sources."""
        pass


class AgentRegistry:
    """Registry for managing agent instances."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent instance."""
        self._agents[agent.agent_id] = agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """List all registered agent IDs."""
        return list(self._agents.keys())
    
    async def send_message(
        self, 
        target_agent_id: str, 
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Send a message to a specific agent."""
        agent = self.get_agent(target_agent_id)
        if agent:
            return await agent.process_message(message)
        return None

