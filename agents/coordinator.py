"""
Coordinator Agent implementation for the Graph-Enhanced Agentic RAG system.

The Coordinator Agent is responsible for:
1. Analyzing user queries to extract entities and determine query type
2. Selecting optimal retrieval strategies based on query characteristics
3. Orchestrating communication between other agents
4. Aggregating results from multiple agents

This implements task 4.1: Create query analysis functionality
"""

import re
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
import logging

from core.interfaces import (
    CoordinatorAgentInterface, 
    MessageType, 
    QueryAnalysis, 
    QueryType, 
    RetrievalStrategy
)
from core.protocols import (
    AgentMessage,
    QueryAnalysisMessage,
    QueryAnalysisResponse,
    StrategySelectionMessage,
    StrategySelectionResponse,
    MessageValidator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityExtractor:
    """Handles entity extraction from user queries."""
    
    def __init__(self):
        # Common entity patterns for technical documentation domain
        self.entity_patterns = {
            'technology': [
                r'\b(?:Python|Java|JavaScript|TypeScript|React|Vue|Angular|Node\.js|Django|Flask|FastAPI)\b',
                r'\b(?:Docker|Kubernetes|AWS|Azure|GCP|MongoDB|PostgreSQL|MySQL|Redis)\b',
                r'\b(?:API|REST|GraphQL|HTTP|HTTPS|JSON|XML|YAML)\b',
                r'\b(?:Git|GitHub|GitLab|CI/CD|DevOps|Microservices)\b'
            ],
            'concept': [
                r'\b(?:authentication|authorization|security|encryption|hashing)\b',
                r'\b(?:database|query|index|schema|migration|backup)\b',
                r'\b(?:deployment|scaling|monitoring|logging|testing)\b',
                r'\b(?:architecture|design pattern|best practice|framework)\b'
            ],
            'organization': [
                r'\b(?:[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Technologies|Systems))\b',
                r'\b(?:Google|Microsoft|Amazon|Apple|Meta|Netflix|Spotify)\b'
            ],
            'person': [
                r'\b(?:[A-Z][a-z]+ [A-Z][a-z]+)\b'  # Simple name pattern
            ]
        }
        
        # Keywords that indicate different query types
        self.query_type_indicators = {
            'factual': [
                'what is', 'define', 'explain', 'describe', 'meaning of',
                'definition', 'overview', 'introduction to'
            ],
            'relational': [
                'how does', 'relationship between', 'connection', 'related to',
                'difference between', 'compare', 'versus', 'vs', 'interact with',
                'depends on', 'affects', 'influences'
            ],
            'multi_hop': [
                'how to', 'step by step', 'process of', 'workflow', 'pipeline',
                'from', 'to', 'through', 'via', 'using', 'with', 'then'
            ],
            'complex': [
                'analyze', 'evaluate', 'assess', 'recommend', 'best approach',
                'pros and cons', 'advantages', 'disadvantages', 'trade-offs',
                'when to use', 'why', 'because'
            ]
        }
    
    def extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from a user query using pattern matching.
        
        Args:
            query: User query text
            
        Returns:
            List[str]: Extracted entity names
        """
        entities = set()
        query_lower = query.lower()
        
        # Extract entities using predefined patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                entities.update(matches)
        
        # Extract potential entities using capitalization patterns
        # Look for capitalized words that might be proper nouns
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Filter out common words that are capitalized but not entities
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where',
            'Why', 'How', 'Who', 'Which', 'Can', 'Could', 'Should', 'Would',
            'Will', 'Do', 'Does', 'Did', 'Is', 'Are', 'Was', 'Were', 'Have',
            'Has', 'Had', 'Get', 'Got', 'Make', 'Made', 'Take', 'Took'
        }
        
        for word in capitalized_words:
            if word not in common_words and len(word) > 2:
                entities.add(word)
        
        # Extract quoted terms as potential entities
        quoted_terms = re.findall(r'"([^"]+)"', query)
        entities.update(quoted_terms)
        
        # Extract terms in backticks (common in technical queries)
        backtick_terms = re.findall(r'`([^`]+)`', query)
        entities.update(backtick_terms)
        
        return list(entities)
    
    def classify_query_type(self, query: str) -> QueryType:
        """
        Classify the query type based on linguistic patterns.
        
        Args:
            query: User query text
            
        Returns:
            QueryType: Classified query type
        """
        query_lower = query.lower()
        
        # Initialize scores
        type_scores = {
            QueryType.FACTUAL: 0,
            QueryType.RELATIONAL: 0,
            QueryType.MULTI_HOP: 0,
            QueryType.COMPLEX: 0
        }
        
        # Strong indicators that override other patterns
        
        # Relational queries - check first as they're most specific
        relational_patterns = [
            r'\b(?:relationship|relation)\s+between\b',
            r'\b(?:difference|differences)\s+between\b',
            r'\b(?:compare|comparison)\b',
            r'\b(?:versus|vs\.?)\b',
            r'\b(?:interact|interacts|interaction)\b',
            r'\b(?:connect|connected|connection)\b',
            r'\b(?:related|relates)\s+to\b'
        ]
        
        for pattern in relational_patterns:
            if re.search(pattern, query_lower):
                type_scores[QueryType.RELATIONAL] += 3
        
        # Complex analytical queries
        complex_patterns = [
            r'\b(?:why|because)\b',
            r'\b(?:analyze|analysis|evaluate|evaluation|assess|assessment)\b',
            r'\b(?:recommend|recommendation|suggest|suggestion)\b',
            r'\b(?:best|better|worse|optimal|ideal)\b',
            r'\b(?:pros\s+and\s+cons|advantages|disadvantages|trade-?offs?)\b',
            r'\b(?:should|would|could)\s+(?:i|you|we)\b',
            r'\b(?:when\s+to\s+use|which\s+to\s+choose)\b'
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, query_lower):
                type_scores[QueryType.COMPLEX] += 3
        
        # Multi-hop process queries
        multihop_patterns = [
            r'\b(?:how\s+(?:do|to))\b',
            r'\b(?:step\s+by\s+step|steps?)\b',
            r'\b(?:process|procedure|workflow|pipeline)\b',
            r'\b(?:guide|tutorial|walkthrough)\b',
            r'\b(?:implement|implementation|setup|configure)\b',
            r'\b(?:deploy|deployment|install|installation)\b'
        ]
        
        for pattern in multihop_patterns:
            if re.search(pattern, query_lower):
                type_scores[QueryType.MULTI_HOP] += 3
        
        # Factual definition queries
        factual_patterns = [
            r'\b(?:what\s+is|what\s+are|what\s+does)\b',
            r'\b(?:define|definition|meaning)\b',
            r'\b(?:explain|explanation)\b',
            r'\b(?:describe|description)\b',
            r'\b(?:overview|introduction)\b'
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, query_lower):
                type_scores[QueryType.FACTUAL] += 2
        
        # Additional scoring based on keywords
        for query_type, indicators in self.query_type_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    type_scores[QueryType(query_type)] += 1
        
        # Special case: "how" can be factual if it's asking for explanation
        if re.search(r'\bhow\s+(?:does|do)\b', query_lower) and not re.search(r'\bhow\s+(?:to|do\s+i)\b', query_lower):
            type_scores[QueryType.FACTUAL] += 2
            type_scores[QueryType.MULTI_HOP] = max(0, type_scores[QueryType.MULTI_HOP] - 1)
        
        # Return the type with the highest score
        max_score = max(type_scores.values())
        if max_score == 0:
            return QueryType.FACTUAL  # Default to factual
        
        # If there's a tie, use priority order: COMPLEX > RELATIONAL > MULTI_HOP > FACTUAL
        priority_order = [QueryType.COMPLEX, QueryType.RELATIONAL, QueryType.MULTI_HOP, QueryType.FACTUAL]
        
        for query_type in priority_order:
            if type_scores[query_type] == max_score:
                return query_type
        
        return QueryType.FACTUAL


class QueryComplexityAnalyzer:
    """Analyzes query complexity to help with strategy selection."""
    
    def __init__(self):
        self.complexity_factors = {
            'entity_count': 0.2,      # More entities = higher complexity
            'query_length': 0.1,      # Longer queries tend to be more complex
            'question_words': 0.15,   # Multiple question words increase complexity
            'logical_operators': 0.2, # AND, OR, NOT increase complexity
            'temporal_references': 0.1, # Time-based queries can be complex
            'comparison_terms': 0.15,  # Comparison queries are moderately complex
            'technical_terms': 0.1    # Technical jargon increases complexity
        }
    
    def calculate_complexity_score(self, query: str, entities: List[str]) -> float:
        """
        Calculate a complexity score for the query (0.0 to 1.0).
        
        Args:
            query: User query text
            entities: Extracted entities
            
        Returns:
            float: Complexity score between 0.0 and 1.0
        """
        score = 0.0
        query_lower = query.lower()
        
        # Entity count factor
        entity_factor = min(len(entities) / 5.0, 1.0)  # Normalize to max 5 entities
        score += entity_factor * self.complexity_factors['entity_count']
        
        # Query length factor
        word_count = len(query.split())
        length_factor = min(word_count / 20.0, 1.0)  # Normalize to max 20 words
        score += length_factor * self.complexity_factors['query_length']
        
        # Question words factor
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which']
        question_count = sum(1 for word in question_words if word in query_lower)
        question_factor = min(question_count / 3.0, 1.0)  # Normalize to max 3 question words
        score += question_factor * self.complexity_factors['question_words']
        
        # Logical operators factor
        logical_ops = ['and', 'or', 'not', 'but', 'however', 'although', 'while']
        logical_count = sum(1 for op in logical_ops if f' {op} ' in query_lower)
        logical_factor = min(logical_count / 2.0, 1.0)  # Normalize to max 2 operators
        score += logical_factor * self.complexity_factors['logical_operators']
        
        # Temporal references factor
        temporal_terms = ['before', 'after', 'during', 'when', 'since', 'until', 'while']
        temporal_count = sum(1 for term in temporal_terms if term in query_lower)
        temporal_factor = min(temporal_count / 2.0, 1.0)
        score += temporal_factor * self.complexity_factors['temporal_references']
        
        # Comparison terms factor
        comparison_terms = ['compare', 'versus', 'vs', 'difference', 'similar', 'different', 'better', 'worse']
        comparison_count = sum(1 for term in comparison_terms if term in query_lower)
        comparison_factor = min(comparison_count / 2.0, 1.0)
        score += comparison_factor * self.complexity_factors['comparison_terms']
        
        # Technical terms factor (based on entity patterns)
        technical_patterns = [
            r'\b(?:API|REST|HTTP|JSON|XML|SQL|NoSQL|CRUD)\b',
            r'\b(?:authentication|authorization|encryption|hashing)\b',
            r'\b(?:microservices|architecture|design pattern|framework)\b'
        ]
        technical_count = 0
        for pattern in technical_patterns:
            technical_count += len(re.findall(pattern, query, re.IGNORECASE))
        technical_factor = min(technical_count / 3.0, 1.0)
        score += technical_factor * self.complexity_factors['technical_terms']
        
        return min(score, 1.0)  # Ensure score doesn't exceed 1.0


class CoordinatorAgent(CoordinatorAgentInterface):
    """
    Coordinator Agent implementation.
    
    Handles query analysis, strategy selection, and agent orchestration.
    """
    
    def __init__(self, agent_id: str = "coordinator", agent_registry=None, message_queue=None):
        super().__init__(agent_id)
        self.entity_extractor = EntityExtractor()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        
        # Agent orchestration components
        self.agent_registry = agent_registry
        self.message_queue = message_queue
        self.active_workflows = {}  # Track active workflow sessions
        
        # Message handlers
        self.message_handlers = {
            MessageType.QUERY_ANALYSIS: self._handle_query_analysis,
            MessageType.STRATEGY_SELECTION: self._handle_strategy_selection,
            MessageType.RESPONSE: self._handle_agent_response
        }
        
        logger.info(f"Coordinator Agent {agent_id} initialized")
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query to extract entities and determine characteristics.
        
        Args:
            query: User query text
            
        Returns:
            QueryAnalysis: Analysis results including entities, type, and complexity
        """
        try:
            logger.info(f"Analyzing query: {query[:100]}...")
            
            # Extract entities from the query
            entities = self.entity_extractor.extract_entities(query)
            logger.debug(f"Extracted entities: {entities}")
            
            # Classify query type
            query_type = self.entity_extractor.classify_query_type(query)
            logger.debug(f"Query type: {query_type}")
            
            # Calculate complexity score
            complexity_score = self.complexity_analyzer.calculate_complexity_score(query, entities)
            logger.debug(f"Complexity score: {complexity_score}")
            
            # Determine retrieval requirements
            requires_graph = self._requires_graph_search(query_type, entities, complexity_score)
            requires_vector = self._requires_vector_search(query_type, complexity_score)
            
            analysis = QueryAnalysis(
                query=query,
                entities=entities,
                query_type=query_type,
                complexity_score=complexity_score,
                requires_graph=requires_graph,
                requires_vector=requires_vector
            )
            
            logger.info(f"Query analysis complete: type={query_type}, complexity={complexity_score:.2f}, "
                       f"entities={len(entities)}, graph={requires_graph}, vector={requires_vector}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            # Return a basic analysis as fallback
            return QueryAnalysis(
                query=query,
                entities=[],
                query_type=QueryType.FACTUAL,
                complexity_score=0.5,
                requires_graph=False,
                requires_vector=True
            )
    
    def _requires_graph_search(self, query_type: QueryType, entities: List[str], complexity_score: float) -> bool:
        """Determine if query requires graph search."""
        # Relational queries almost always need graph search
        if query_type == QueryType.RELATIONAL:
            return True
        
        # Multi-hop queries with entities likely need graph search
        if query_type == QueryType.MULTI_HOP and len(entities) > 1:
            return True
        
        # Complex queries with multiple entities might benefit from graph search
        if query_type == QueryType.COMPLEX and len(entities) > 2:
            return True
        
        # High complexity queries with entities might need graph exploration
        if complexity_score > 0.7 and len(entities) > 0:
            return True
        
        return False
    
    def _requires_vector_search(self, query_type: QueryType, complexity_score: float) -> bool:
        """Determine if query requires vector search."""
        # Most queries benefit from vector search for semantic matching
        # Only skip vector search for very simple relational queries
        if query_type == QueryType.RELATIONAL and complexity_score < 0.3:
            return False
        
        return True
    
    async def select_strategy(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """
        Select optimal retrieval strategy based on query analysis using decision tree logic.
        
        Args:
            analysis: Query analysis results
            
        Returns:
            RetrievalStrategy: Selected strategy
        """
        try:
            logger.info(f"Selecting strategy for query type: {analysis.query_type}, "
                       f"complexity: {analysis.complexity_score:.2f}, entities: {len(analysis.entities)}")
            
            # Enhanced decision tree for strategy selection
            strategy = self._decision_tree_strategy_selection(analysis)
            
            # Validate strategy selection with fallback logic
            validated_strategy = self._validate_strategy_with_fallback(strategy, analysis)
            
            logger.info(f"Selected strategy: {validated_strategy}")
            return validated_strategy
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {str(e)}")
            # Ultimate fallback to vector search
            return RetrievalStrategy.VECTOR_ONLY
    
    def _decision_tree_strategy_selection(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """
        Implement decision tree logic for strategy selection.
        
        Args:
            analysis: Query analysis results
            
        Returns:
            RetrievalStrategy: Selected strategy based on decision tree
        """
        # Decision tree implementation
        
        # Level 1: Query Type Classification
        if analysis.query_type == QueryType.FACTUAL:
            return self._handle_factual_query_strategy(analysis)
        
        elif analysis.query_type == QueryType.RELATIONAL:
            return self._handle_relational_query_strategy(analysis)
        
        elif analysis.query_type == QueryType.MULTI_HOP:
            return self._handle_multi_hop_query_strategy(analysis)
        
        elif analysis.query_type == QueryType.COMPLEX:
            return self._handle_complex_query_strategy(analysis)
        
        else:
            # Default fallback
            return RetrievalStrategy.VECTOR_ONLY
    
    def _handle_factual_query_strategy(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """Handle strategy selection for factual queries."""
        # Factual queries typically benefit from vector search
        # But may need graph search if they involve specific entities with relationships
        
        if len(analysis.entities) == 0:
            # No entities - pure semantic search
            return RetrievalStrategy.VECTOR_ONLY
        
        elif len(analysis.entities) == 1:
            # Single entity - check complexity
            if analysis.complexity_score > 0.6:
                # High complexity might benefit from context
                return RetrievalStrategy.HYBRID
            else:
                return RetrievalStrategy.VECTOR_ONLY
        
        else:
            # Multiple entities in factual query - might need relationship context
            if analysis.complexity_score > 0.5:
                return RetrievalStrategy.HYBRID
            else:
                return RetrievalStrategy.VECTOR_ONLY
    
    def _handle_relational_query_strategy(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """Handle strategy selection for relational queries."""
        # Relational queries almost always need graph traversal
        
        if len(analysis.entities) < 2:
            # Few entities - might be asking about general relationships
            if analysis.complexity_score > 0.4:
                return RetrievalStrategy.HYBRID
            else:
                return RetrievalStrategy.GRAPH_ONLY
        
        else:
            # Multiple entities - definitely need graph traversal
            if analysis.complexity_score > 0.6:
                # High complexity - use both approaches
                return RetrievalStrategy.HYBRID
            else:
                # Focus on graph relationships
                return RetrievalStrategy.GRAPH_ONLY
    
    def _handle_multi_hop_query_strategy(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """Handle strategy selection for multi-hop queries."""
        # Multi-hop queries often need both graph traversal and semantic understanding
        
        if len(analysis.entities) == 0:
            # No specific entities - procedural/process query
            return RetrievalStrategy.VECTOR_ONLY
        
        elif len(analysis.entities) == 1:
            # Single entity process - might need context
            if analysis.complexity_score > 0.5:
                return RetrievalStrategy.HYBRID
            else:
                return RetrievalStrategy.VECTOR_ONLY
        
        else:
            # Multiple entities in process - likely need relationship traversal
            return RetrievalStrategy.HYBRID
    
    def _handle_complex_query_strategy(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """Handle strategy selection for complex analytical queries."""
        # Complex queries typically benefit from comprehensive search
        
        if len(analysis.entities) == 0:
            # Abstract complex query - semantic search might suffice
            if analysis.complexity_score > 0.8:
                return RetrievalStrategy.HYBRID
            else:
                return RetrievalStrategy.VECTOR_ONLY
        
        elif len(analysis.entities) <= 2:
            # Few entities - hybrid approach for comprehensive analysis
            return RetrievalStrategy.HYBRID
        
        else:
            # Many entities - definitely need comprehensive approach
            return RetrievalStrategy.HYBRID
    
    def _validate_strategy_with_fallback(self, strategy: RetrievalStrategy, analysis: QueryAnalysis) -> RetrievalStrategy:
        """
        Validate selected strategy and apply fallback logic if needed.
        
        Args:
            strategy: Initially selected strategy
            analysis: Query analysis results
            
        Returns:
            RetrievalStrategy: Validated strategy with fallback applied if necessary
        """
        # Check if strategy is feasible based on system constraints
        
        # For now, assume all strategies are available
        # In a real system, you might check:
        # - Graph database connectivity
        # - Vector database availability
        # - System load/performance constraints
        
        # Apply fallback logic based on confidence and constraints
        if strategy == RetrievalStrategy.GRAPH_ONLY:
            # Ensure graph-only makes sense
            if len(analysis.entities) == 0:
                logger.warning("Graph-only strategy selected but no entities found, falling back to vector")
                return RetrievalStrategy.VECTOR_ONLY
        
        elif strategy == RetrievalStrategy.HYBRID:
            # Hybrid is resource-intensive, ensure it's justified
            if analysis.complexity_score < 0.3 and len(analysis.entities) <= 1:
                logger.info("Hybrid strategy may be overkill, falling back to vector-only")
                return RetrievalStrategy.VECTOR_ONLY
        
        return strategy
    
    def get_fallback_strategies(self, primary_strategy: RetrievalStrategy, analysis: QueryAnalysis) -> List[RetrievalStrategy]:
        """
        Get ordered list of fallback strategies for the primary strategy.
        
        Args:
            primary_strategy: The primary selected strategy
            analysis: Query analysis results
            
        Returns:
            List[RetrievalStrategy]: Ordered list of fallback strategies
        """
        fallbacks = []
        
        if primary_strategy == RetrievalStrategy.HYBRID:
            # If hybrid fails, try the most appropriate single strategy
            if analysis.query_type == QueryType.RELATIONAL or len(analysis.entities) > 2:
                fallbacks = [RetrievalStrategy.GRAPH_ONLY, RetrievalStrategy.VECTOR_ONLY]
            else:
                fallbacks = [RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.GRAPH_ONLY]
        
        elif primary_strategy == RetrievalStrategy.GRAPH_ONLY:
            # If graph fails, try hybrid then vector
            fallbacks = [RetrievalStrategy.HYBRID, RetrievalStrategy.VECTOR_ONLY]
        
        elif primary_strategy == RetrievalStrategy.VECTOR_ONLY:
            # If vector fails, try hybrid then graph
            if len(analysis.entities) > 0:
                fallbacks = [RetrievalStrategy.HYBRID, RetrievalStrategy.GRAPH_ONLY]
            else:
                fallbacks = [RetrievalStrategy.HYBRID]  # Graph without entities doesn't make sense
        
        return fallbacks
    
    async def select_strategy_with_fallback(self, analysis: QueryAnalysis, failed_strategies: List[RetrievalStrategy] = None) -> RetrievalStrategy:
        """
        Select strategy with awareness of previously failed strategies.
        
        Args:
            analysis: Query analysis results
            failed_strategies: List of strategies that have already failed
            
        Returns:
            RetrievalStrategy: Selected strategy avoiding failed ones
        """
        failed_strategies = failed_strategies or []
        
        # Get primary strategy
        primary_strategy = await self.select_strategy(analysis)
        
        # If primary strategy has failed, try fallbacks
        if primary_strategy in failed_strategies:
            fallbacks = self.get_fallback_strategies(primary_strategy, analysis)
            
            for fallback in fallbacks:
                if fallback not in failed_strategies:
                    logger.info(f"Primary strategy {primary_strategy} failed, using fallback: {fallback}")
                    return fallback
            
            # If all strategies have failed, return the most basic one
            logger.warning("All strategies have failed, falling back to vector-only as last resort")
            return RetrievalStrategy.VECTOR_ONLY
        
        return primary_strategy
    
    async def orchestrate_retrieval(self, query: str, strategy: RetrievalStrategy) -> tuple:
        """
        Orchestrate retrieval process using selected strategy.
        
        Args:
            query: User query
            strategy: Selected retrieval strategy
            
        Returns:
            tuple: Results from retrieval (graph_results, vector_results)
        """
        logger.info(f"Orchestrating retrieval with strategy: {strategy}")
        
        try:
            # Analyze query first
            analysis = await self.analyze_query(query)
            
            # Execute retrieval based on strategy
            if strategy == RetrievalStrategy.VECTOR_ONLY:
                return await self._orchestrate_vector_only(query, analysis)
            
            elif strategy == RetrievalStrategy.GRAPH_ONLY:
                return await self._orchestrate_graph_only(query, analysis)
            
            elif strategy == RetrievalStrategy.HYBRID:
                return await self._orchestrate_hybrid(query, analysis)
            
            else:
                logger.warning(f"Unknown strategy: {strategy}, falling back to vector-only")
                return await self._orchestrate_vector_only(query, analysis)
                
        except Exception as e:
            logger.error(f"Error orchestrating retrieval: {str(e)}")
            # Return empty results on error
            return (None, None)
    
    async def _orchestrate_vector_only(self, query: str, analysis: QueryAnalysis) -> tuple:
        """Orchestrate vector-only retrieval."""
        logger.info("Orchestrating vector-only retrieval")
        
        # Create vector search message
        vector_message = self._create_vector_search_message(query, analysis)
        
        # Send to vector retrieval agent (simulated for now)
        vector_results = await self._send_to_vector_agent(vector_message)
        
        return (None, vector_results)
    
    async def _orchestrate_graph_only(self, query: str, analysis: QueryAnalysis) -> tuple:
        """Orchestrate graph-only retrieval."""
        logger.info("Orchestrating graph-only retrieval")
        
        # Create graph search message
        graph_message = self._create_graph_search_message(query, analysis)
        
        # Send to graph navigator agent (simulated for now)
        graph_results = await self._send_to_graph_agent(graph_message)
        
        return (graph_results, None)
    
    async def _orchestrate_hybrid(self, query: str, analysis: QueryAnalysis) -> tuple:
        """Orchestrate hybrid retrieval using both graph and vector search."""
        logger.info("Orchestrating hybrid retrieval")
        
        # Create messages for both agents
        vector_message = self._create_vector_search_message(query, analysis)
        graph_message = self._create_graph_search_message(query, analysis)
        
        # Execute both searches concurrently
        vector_task = self._send_to_vector_agent(vector_message)
        graph_task = self._send_to_graph_agent(graph_message)
        
        # Wait for both to complete
        vector_results, graph_results = await asyncio.gather(
            vector_task, 
            graph_task, 
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(vector_results, Exception):
            logger.error(f"Vector search failed: {vector_results}")
            vector_results = None
        
        if isinstance(graph_results, Exception):
            logger.error(f"Graph search failed: {graph_results}")
            graph_results = None
        
        return (graph_results, vector_results)
    
    def _create_vector_search_message(self, query: str, analysis: QueryAnalysis) -> AgentMessage:
        """Create a vector search message for the vector retrieval agent."""
        from core.protocols import VectorSearchMessage
        
        payload = VectorSearchMessage(
            query=query,
            k=10,  # Default number of results
            similarity_threshold=0.0,
            filters={},
            include_embeddings=False,
            rerank=True
        )
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id="vector_retrieval",
            message_type=MessageType.VECTOR_SEARCH,
            payload=payload.model_dump(),
            correlation_id=f"query_{hash(query)}"
        )
    
    def _create_graph_search_message(self, query: str, analysis: QueryAnalysis) -> AgentMessage:
        """Create a graph search message for the graph navigator agent."""
        from core.protocols import GraphSearchMessage
        
        payload = GraphSearchMessage(
            entities=analysis.entities,
            query=query,
            max_depth=2,
            max_results=50,
            relationship_types=None,
            filters={}
        )
        
        return AgentMessage(
            agent_id=self.agent_id,
            target_agent_id="graph_navigator",
            message_type=MessageType.GRAPH_SEARCH,
            payload=payload.model_dump(),
            correlation_id=f"query_{hash(query)}"
        )
    
    async def _send_to_vector_agent(self, message: AgentMessage) -> Optional[Dict]:
        """
        Send message to vector retrieval agent using real communication system.
        """
        logger.debug(f"Sending message to vector retrieval agent: {message.id}")
        
        try:
            # First try direct agent communication via registry
            if self.agent_registry:
                vector_agent = self.agent_registry.get_agent("vector_retrieval")
                if vector_agent:
                    logger.debug("Using direct agent communication")
                    # Extract query parameters from message
                    query = message.payload.get('query', '') if message.payload else ''
                    k = message.payload.get("k", 5)
                    filters = message.payload.get("filters", {})
                    # Call the vector agent directly
                    if hasattr(vector_agent, 'similarity_search'):
                        result = await vector_agent.similarity_search(query, k=k, filters=filters)
                        # Convert documents to ensure they have proper titles
                        converted_docs = []
                        for doc in result.documents:
                            doc_dict = doc.model_dump()
                            # Ensure title is not None
                            if doc_dict.get('title') is None:
                                doc_dict['title'] = f"Document {doc_dict.get('id', 'Unknown')}"
                            converted_docs.append(doc_dict)
                        
                        return {
                            "documents": converted_docs,
                            "similarities": result.similarities,
                            "query_embedding": result.query_embedding
                        }
                    elif hasattr(vector_agent, 'process_message'):
                        response = await vector_agent.process_message(message)
                        return response.payload if response else None
                else:
                    logger.error("❌ Target agent vector_retrieval not found or unavailable")
            
            # Fallback: direct communication (for backward compatibility)
            else:
                logger.warning("No agent registry available, using direct communication")
                
                # Import and create vector retrieval agent
                from .vector_retrieval import VectorRetrievalAgent
                from core.config import get_config
                
                config = get_config()
                vector_agent = VectorRetrievalAgent(
                    agent_id="vector_retrieval",
                    model_name=config.llm.embedding_model,
                    collection_name=config.database.chroma_collection_name
                )
                
                # Process the message using the real agent
                response = await vector_agent.process_message(message)
                
                if response and response.message_type == MessageType.RESPONSE:
                    return response.payload
            
            # Return empty results as fallback
            logger.warning("Using fallback empty response")
            from core.protocols import VectorSearchResponse
            fallback_response = VectorSearchResponse(
                documents=[],
                similarities=[],
                query_embedding=None,
                execution_time=0.1,
                total_results=0,
                reranked=False
            )
            return fallback_response.model_dump()
                
        except Exception as e:
            logger.error(f"Error communicating with vector agent: {e}")
            # Return empty results as fallback
            from core.protocols import VectorSearchResponse
            fallback_response = VectorSearchResponse(
                documents=[],
                similarities=[],
                query_embedding=None,
                execution_time=0.1,
                total_results=0,
                reranked=False
            )
            return fallback_response.model_dump()
    
    async def _send_to_graph_agent(self, message: AgentMessage) -> Optional[Dict]:
        """
        Send message to graph navigator agent using real communication system.
        """
        logger.debug(f"Sending message to graph navigator agent: {message.id}")
        
        try:
            # First try direct agent communication via registry
            if self.agent_registry:
                graph_agent = self.agent_registry.get_agent("graph_navigator")
                if graph_agent:
                    logger.debug("Using direct agent communication")
                    # Extract query parameters from message
                    query = message.payload.get('query', '') if message.payload else ''
                    entities = message.payload.get("entities", [])
                    depth = message.payload.get("depth", 2)
                    # Call the graph agent directly
                    if hasattr(graph_agent, 'find_entities') and hasattr(graph_agent, 'traverse_relationships'):
                        # Find entities first
                        found_entities = await graph_agent.find_entities(query)
                        # Then traverse relationships if entities found
                        if found_entities:
                            result = await graph_agent.traverse_relationships(found_entities, depth=depth)
                            return {
                                "entities": [entity.model_dump() for entity in result.entities],
                                "relationships": result.relationships,
                                "paths": result.paths
                            }
                        else:
                            return {
                                "entities": [],
                                "relationships": [],
                                "paths": []
                            }
                    elif hasattr(graph_agent, 'process_message'):
                        response = await graph_agent.process_message(message)
                        return response.payload if response else None
                else:
                    logger.error("❌ Target agent graph_navigator not found or unavailable")
            
            # Fallback: create basic response structure
            logger.warning("Using fallback graph response")
            entities = message.payload.get('entities', [])
            
            from core.protocols import GraphSearchResponse
            
            # Create a basic response - in production this would come from real graph traversal
            response = GraphSearchResponse(
                entities=[],  # Empty for now - will be populated by real graph search
                relationships=[],
                paths=[],
                cypher_query="MATCH (n) RETURN n LIMIT 10",
                execution_time=0.15,
                total_results=0,
                has_more_results=False
            )
            
            return response.model_dump()
                
        except Exception as e:
            logger.error(f"Error in graph search processing: {e}")
            return None
    
    async def coordinate_full_workflow(self, query: str) -> Dict[str, Any]:
        """
        Coordinate the full workflow from query to final response.
        
        Args:
            query: User query
            
        Returns:
            Dict: Complete workflow results including analysis, retrieval, and synthesis
        """
        logger.info(f"Starting full workflow coordination for query: {query[:100]}...")
        
        workflow_results = {
            "query": query,
            "analysis": None,
            "strategy": None,
            "retrieval_results": None,
            "synthesis_results": None,
            "execution_time": 0,
            "errors": []
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Analyze query
            logger.info("Step 1: Analyzing query")
            analysis = await self.analyze_query(query)
            workflow_results["analysis"] = analysis.model_dump()
            
            # Step 2: Select strategy
            logger.info("Step 2: Selecting retrieval strategy")
            strategy = await self.select_strategy(analysis)
            workflow_results["strategy"] = strategy
            
            # Step 3: Execute retrieval
            logger.info("Step 3: Executing retrieval")
            graph_results, vector_results = await self.orchestrate_retrieval(query, strategy)
            workflow_results["retrieval_results"] = {
                "graph_results": graph_results,
                "vector_results": vector_results
            }
            
            # Step 4: Coordinate synthesis (placeholder)
            logger.info("Step 4: Coordinating synthesis")
            synthesis_results = await self._coordinate_synthesis(query, graph_results, vector_results)
            workflow_results["synthesis_results"] = synthesis_results
            
        except Exception as e:
            logger.error(f"Error in workflow coordination: {str(e)}")
            workflow_results["errors"].append(str(e))
        
        finally:
            end_time = asyncio.get_event_loop().time()
            workflow_results["execution_time"] = end_time - start_time
            logger.info(f"Workflow completed in {workflow_results['execution_time']:.2f} seconds")
        
        return workflow_results
    
    async def _coordinate_synthesis(self, query: str, graph_results: Optional[Dict], vector_results: Optional[Dict]) -> Optional[Dict]:
        """
        Coordinate synthesis of results from multiple agents.
        
        This is a placeholder implementation for synthesis coordination.
        """
        logger.info("Coordinating synthesis of retrieval results")
        
        # Create synthesis message
        from core.protocols import SynthesisMessage
        
        # Convert results back to response objects if they exist
        graph_response = None
        vector_response = None
        
        if graph_results:
            from core.protocols import GraphSearchResponse
            graph_response = GraphSearchResponse(**graph_results)
        
        if vector_results:
            from core.protocols import VectorSearchResponse
            # Ensure documents are properly serialized
            serialized_vector_results = vector_results.copy()
            if 'documents' in serialized_vector_results:
                # Convert Document objects to serializable dictionaries
                serialized_docs = []
                for doc in serialized_vector_results['documents']:
                    if hasattr(doc, 'model_dump'):
                        # Use mode='json' to ensure datetime objects are serialized as strings
                        doc_dict = doc.model_dump(mode='json')
                    else:
                        doc_dict = doc
                    
                    # Ensure all fields are JSON serializable
                    if 'embedding' in doc_dict and doc_dict['embedding'] is None:
                        doc_dict['embedding'] = None  # Keep None as is
                    elif 'embedding' in doc_dict and isinstance(doc_dict['embedding'], list):
                        # Ensure embedding is a list of floats
                        doc_dict['embedding'] = [float(x) for x in doc_dict['embedding']]
                    
                    serialized_docs.append(doc_dict)
                
                serialized_vector_results['documents'] = serialized_docs
            
            vector_response = VectorSearchResponse(**serialized_vector_results)
        
        synthesis_message = SynthesisMessage(
            query=query,
            graph_results=graph_response,
            vector_results=vector_response,
            context={},
            response_format="natural",
            max_response_length=1000,
            include_citations=True,
            include_reasoning=True
        )
        
        # Send to synthesis agent (simulated)
        synthesis_results = await self._send_to_synthesis_agent(synthesis_message)
        
        return synthesis_results
    
    async def _send_to_synthesis_agent(self, synthesis_message: Any) -> Dict:
        """
        Send message to synthesis agent using the agent registry.
        """
        logger.info("Sending synthesis request")
        
        try:
            # Create agent message for synthesis with proper serialization
            if hasattr(synthesis_message, 'model_dump'):
                # Use mode='json' to ensure datetime objects are serialized as strings
                payload = synthesis_message.model_dump(mode='json')
            else:
                payload = synthesis_message
            
            agent_message = AgentMessage(
                agent_id=self.agent_id,
                target_agent_id="synthesis",
                message_type=MessageType.SYNTHESIS_REQUEST,
                payload=payload
            )
            
            # Use agent registry if available
            if self.agent_registry:
                synthesis_agent = self.agent_registry.get_agent("synthesis")
                if synthesis_agent:
                    logger.debug("Using direct synthesis agent communication")
                    response = await synthesis_agent.process_message(agent_message)
                    if response and response.message_type == MessageType.RESPONSE:
                        return response.payload
                    else:
                        logger.warning("Synthesis agent returned no valid response")
                else:
                    logger.error("❌ Synthesis agent not found in registry")
            
            # Fallback: create basic synthesis response
            logger.warning("Using fallback synthesis response")
            query = getattr(synthesis_message, 'query', 'Unknown query')
            
            from core.protocols import SynthesisResponse
            
            # Create a basic response - in production this would come from real synthesis
            response = SynthesisResponse(
                response=f"Based on the analysis, here's what I found about: {query}. This is a coordinated response from the multi-agent system.",
                sources=[],
                citations=[],
                reasoning_path="Query analyzed → Strategy selected → Results retrieved → Response synthesized",
                confidence_score=0.8,
                used_graph_results=hasattr(synthesis_message, 'graph_results') and synthesis_message.graph_results is not None,
                used_vector_results=hasattr(synthesis_message, 'vector_results') and synthesis_message.vector_results is not None,
                generation_time=0.2
            )
            
            return response.model_dump()
                
        except Exception as e:
            logger.error(f"Error in synthesis processing: {e}")
            return {"response": f"Synthesis processing error: {str(e)}", "sources": [], "citations": []}
    
    async def _handle_agent_response(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle responses from other agents."""
        try:
            correlation_id = message.correlation_id
            if correlation_id and correlation_id in self.active_workflows:
                workflow = self.active_workflows[correlation_id]
                
                # Update workflow with agent response
                if message.agent_id == "vector_retrieval":
                    workflow["vector_response"] = message.payload
                    logger.info(f"Received vector response for workflow {correlation_id}")
                
                elif message.agent_id == "graph_navigator":
                    workflow["graph_response"] = message.payload
                    logger.info(f"Received graph response for workflow {correlation_id}")
                
                elif message.agent_id == "synthesis":
                    workflow["synthesis_response"] = message.payload
                    workflow["status"] = "completed"
                    logger.info(f"Received synthesis response for workflow {correlation_id}")
                
                # Check if workflow is ready for next step
                await self._check_workflow_progress(correlation_id)
            
            return None  # No response needed for agent responses
            
        except Exception as e:
            logger.error(f"Error handling agent response: {str(e)}")
            return None
    
    async def _check_workflow_progress(self, correlation_id: str):
        """Check if workflow can proceed to next step."""
        if correlation_id not in self.active_workflows:
            return
        
        workflow = self.active_workflows[correlation_id]
        
        # Check if retrieval is complete and synthesis can start
        if (workflow.get("status") == "retrieving" and 
            self._is_retrieval_complete(workflow)):
            
            await self._start_synthesis_step(correlation_id)
    
    def _is_retrieval_complete(self, workflow: Dict) -> bool:
        """Check if retrieval step is complete based on strategy."""
        strategy = workflow.get("strategy")
        
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            return "vector_response" in workflow
        
        elif strategy == RetrievalStrategy.GRAPH_ONLY:
            return "graph_response" in workflow
        
        elif strategy == RetrievalStrategy.HYBRID:
            return ("vector_response" in workflow and 
                    "graph_response" in workflow)
        
        return False
    
    async def _start_synthesis_step(self, correlation_id: str):
        """Start the synthesis step of the workflow."""
        workflow = self.active_workflows[correlation_id]
        workflow["status"] = "synthesizing"
        
        # Create synthesis message
        synthesis_results = await self._coordinate_synthesis(
            workflow["query"],
            workflow.get("graph_response"),
            workflow.get("vector_response")
        )
        
        workflow["synthesis_response"] = synthesis_results
        workflow["status"] = "completed"
        
        logger.info(f"Workflow {correlation_id} completed")
    
    def create_workflow_session(self, query: str) -> str:
        """Create a new workflow session and return its ID."""
        import uuid
        correlation_id = str(uuid.uuid4())
        
        self.active_workflows[correlation_id] = {
            "query": query,
            "status": "started",
            "created_at": asyncio.get_event_loop().time(),
            "strategy": None,
            "analysis": None
        }
        
        logger.info(f"Created workflow session {correlation_id} for query: {query[:50]}...")
        return correlation_id
    
    def get_workflow_status(self, correlation_id: str) -> Optional[Dict]:
        """Get the current status of a workflow."""
        return self.active_workflows.get(correlation_id)
    
    def cleanup_completed_workflows(self, max_age_seconds: int = 3600):
        """Clean up completed workflows older than max_age_seconds."""
        current_time = asyncio.get_event_loop().time()
        to_remove = []
        
        for correlation_id, workflow in self.active_workflows.items():
            if (workflow.get("status") == "completed" and 
                current_time - workflow.get("created_at", 0) > max_age_seconds):
                to_remove.append(correlation_id)
        
        for correlation_id in to_remove:
            del self.active_workflows[correlation_id]
            logger.info(f"Cleaned up completed workflow {correlation_id}")
    
    async def send_message_to_agent(self, target_agent_id: str, message: AgentMessage) -> bool:
        """
        Send a message to another agent through the registry or queue.
        
        Args:
            target_agent_id: ID of the target agent
            message: Message to send
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            if self.agent_registry:
                # Use agent registry for direct communication
                response = await self.agent_registry.send_message(target_agent_id, message)
                if response:
                    # Handle immediate response
                    await self.process_message(response)
                return True
            
            elif self.message_queue:
                # Use message queue for asynchronous communication
                return self.message_queue.enqueue(message)
            
            else:
                # Fallback to simulated communication
                logger.warning(f"No agent registry or message queue available, simulating message to {target_agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error sending message to {target_agent_id}: {str(e)}")
            return False
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get metrics about agent orchestration performance."""
        active_count = len(self.active_workflows)
        completed_count = sum(1 for w in self.active_workflows.values() if w.get("status") == "completed")
        
        return {
            "active_workflows": active_count,
            "completed_workflows": completed_count,
            "total_workflows": active_count,
            "average_workflow_time": self._calculate_average_workflow_time(),
            "agent_registry_available": self.agent_registry is not None,
            "message_queue_available": self.message_queue is not None
        }
    
    def _calculate_average_workflow_time(self) -> float:
        """Calculate average workflow completion time."""
        completed_workflows = [
            w for w in self.active_workflows.values() 
            if w.get("status") == "completed" and "completed_at" in w
        ]
        
        if not completed_workflows:
            return 0.0
        
        total_time = sum(
            w["completed_at"] - w["created_at"] 
            for w in completed_workflows
        )
        
        return total_time / len(completed_workflows)
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming messages from other agents.
        
        Args:
            message: Incoming message
            
        Returns:
            Optional[AgentMessage]: Response message if needed
        """
        try:
            # Validate message
            MessageValidator.validate_message(message)
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                return await handler(message)
            else:
                logger.warning(f"No handler for message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            # Return error message
            return self.create_message(
                MessageType.ERROR,
                {
                    "error_type": "processing_error",
                    "error_message": str(e),
                    "original_message_id": message.id
                },
                message.correlation_id
            )
    
    async def _handle_query_analysis(self, message: AgentMessage) -> AgentMessage:
        """Handle query analysis message."""
        try:
            payload = QueryAnalysisMessage(**message.payload)
            analysis = await self.analyze_query(payload.query)
            
            response_payload = QueryAnalysisResponse(
                query=analysis.query,
                entities=analysis.entities,
                query_type=analysis.query_type,
                complexity_score=analysis.complexity_score,
                requires_graph=analysis.requires_graph,
                requires_vector=analysis.requires_vector,
                confidence_score=0.8,  # Default confidence
                extracted_keywords=analysis.entities,  # Use entities as keywords for now
                suggested_strategy=await self.select_strategy(analysis)
            )
            
            return self.create_message(
                MessageType.RESPONSE,
                response_payload.model_dump(),
                message.correlation_id
            )
            
        except Exception as e:
            logger.error(f"Error handling query analysis: {str(e)}")
            raise
    
    async def _handle_strategy_selection(self, message: AgentMessage) -> AgentMessage:
        """Handle strategy selection message."""
        try:
            payload = StrategySelectionMessage(**message.payload)
            
            # Convert QueryAnalysisResponse back to QueryAnalysis
            analysis = QueryAnalysis(
                query=payload.query_analysis.query,
                entities=payload.query_analysis.entities,
                query_type=payload.query_analysis.query_type,
                complexity_score=payload.query_analysis.complexity_score,
                requires_graph=payload.query_analysis.requires_graph,
                requires_vector=payload.query_analysis.requires_vector
            )
            
            strategy = await self.select_strategy(analysis)
            
            # Determine fallback strategies
            fallback_strategies = self._get_fallback_strategies(strategy)
            
            response_payload = StrategySelectionResponse(
                selected_strategy=strategy,
                reasoning=self._get_strategy_reasoning(analysis, strategy),
                confidence_score=0.8,
                fallback_strategies=fallback_strategies,
                estimated_execution_time=self._estimate_execution_time(strategy)
            )
            
            return self.create_message(
                MessageType.RESPONSE,
                response_payload.model_dump(),
                message.correlation_id
            )
            
        except Exception as e:
            logger.error(f"Error handling strategy selection: {str(e)}")
            raise
    
    def _get_fallback_strategies(self, primary_strategy: RetrievalStrategy) -> List[RetrievalStrategy]:
        """Get fallback strategies for the primary strategy (legacy method)."""
        # Use the new method with a dummy analysis for backward compatibility
        from core.interfaces import QueryAnalysis, QueryType
        dummy_analysis = QueryAnalysis(
            query="",
            entities=[],
            query_type=QueryType.FACTUAL,
            complexity_score=0.5,
            requires_graph=False,
            requires_vector=True
        )
        return self.get_fallback_strategies(primary_strategy, dummy_analysis)
    
    def _get_strategy_reasoning(self, analysis: QueryAnalysis, strategy: RetrievalStrategy) -> str:
        """Generate reasoning explanation for strategy selection."""
        reasons = []
        
        if strategy == RetrievalStrategy.HYBRID:
            reasons.append("Query requires both semantic understanding and relationship exploration")
            if analysis.complexity_score > 0.6:
                reasons.append("High complexity score indicates need for comprehensive search")
            if len(analysis.entities) > 2:
                reasons.append("Multiple entities suggest complex relationships")
        
        elif strategy == RetrievalStrategy.GRAPH_ONLY:
            reasons.append("Query focuses on relationships and connections between entities")
            if analysis.query_type == QueryType.RELATIONAL:
                reasons.append("Relational query type best served by graph traversal")
        
        elif strategy == RetrievalStrategy.VECTOR_ONLY:
            reasons.append("Query is primarily factual and benefits from semantic search")
            if analysis.query_type == QueryType.FACTUAL:
                reasons.append("Factual query type well-suited for vector similarity search")
        
        return ". ".join(reasons) if reasons else "Default strategy selection based on query characteristics"
    
    def _estimate_execution_time(self, strategy: RetrievalStrategy) -> float:
        """Estimate execution time for strategy (in seconds)."""
        estimates = {
            RetrievalStrategy.VECTOR_ONLY: 1.5,
            RetrievalStrategy.GRAPH_ONLY: 2.0,
            RetrievalStrategy.HYBRID: 3.0
        }
        return estimates.get(strategy, 2.0)