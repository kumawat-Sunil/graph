"""
Graph Navigator Agent implementation for the Graph-Enhanced Agentic RAG system.

The Graph Navigator Agent is responsible for:
1. Finding entities in the graph database using fuzzy matching
2. Traversing relationships between entities with configurable depth
3. Generating and executing optimized Cypher queries
4. Extracting relevant subgraphs for context

This implements task 5: Implement Graph Navigator Agent
"""

import re
import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime
from difflib import SequenceMatcher

from core.interfaces import (
    GraphNavigatorInterface, 
    Entity, 
    GraphResult,
    MessageType
)
from core.protocols import AgentMessage
from core.graph_models import (
    EntityNode,
    DocumentNode,
    ConceptNode,
    GraphRelationship,
    CypherQueryBuilder,
    RelationshipType,
    NodeType,
    CommonQueries
)
from core.database import get_neo4j_manager, Neo4jConnectionError
from core.protocols import (
    GraphSearchMessage,
    GraphSearchResponse,
    MessageValidator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityMatcher:
    """Handles fuzzy entity matching and disambiguation in the graph database."""
    
    def __init__(self, neo4j_manager):
        self.neo4j_manager = neo4j_manager
        self.similarity_threshold = 0.6  # Minimum similarity for fuzzy matching
        
    async def find_entities_fuzzy(self, query_text: str, limit: int = 10) -> List[Entity]:
        """
        Find entities using fuzzy string matching.
        
        Args:
            query_text: Text to search for entities
            limit: Maximum number of entities to return
            
        Returns:
            List[Entity]: Matched entities sorted by relevance
        """
        try:
            # Extract potential entity names from query text
            potential_entities = self._extract_entity_candidates(query_text)
            
            if not potential_entities:
                return []
            
            # Search for each potential entity
            all_matches = []
            for candidate in potential_entities:
                matches = await self._search_entity_by_name(candidate, limit)
                all_matches.extend(matches)
            
            # Remove duplicates and sort by relevance
            unique_matches = self._deduplicate_entities(all_matches)
            scored_matches = self._score_entity_matches(unique_matches, query_text)
            
            # Return top matches
            return [entity for entity, score in scored_matches[:limit]]
            
        except Exception as e:
            logger.error(f"Error in fuzzy entity matching: {str(e)}")
            return []
    
    def _extract_entity_candidates(self, text: str) -> List[str]:
        """Extract potential entity names from text."""
        candidates = set()
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', text)
        candidates.update(quoted_terms)
        
        # Extract terms in backticks
        backtick_terms = re.findall(r'`([^`]+)`', text)
        candidates.update(backtick_terms)
        
        # Extract capitalized words/phrases
        capitalized_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b[A-Z][a-z]*[A-Z][a-zA-Z]*\b'  # CamelCase
        ]
        
        for pattern in capitalized_patterns:
            matches = re.findall(pattern, text)
            candidates.update(matches)
        
        # Extract technical terms
        tech_patterns = [
            r'\b(?:API|REST|HTTP|JSON|XML|SQL|NoSQL|CRUD)\b',
            r'\b(?:Python|Java|JavaScript|TypeScript|React|Vue|Angular)\b',
            r'\b(?:Docker|Kubernetes|AWS|Azure|GCP|MongoDB|PostgreSQL)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            candidates.update(matches)
        
        # Filter out common words
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'What', 'When', 'Where',
            'Why', 'How', 'Who', 'Which', 'Can', 'Could', 'Should', 'Would',
            'Will', 'Do', 'Does', 'Did', 'Is', 'Are', 'Was', 'Were', 'Have',
            'Has', 'Had', 'Get', 'Got', 'Make', 'Made', 'Take', 'Took'
        }
        
        filtered_candidates = [
            candidate for candidate in candidates 
            if candidate not in common_words and len(candidate.strip()) > 2
        ]
        
        return filtered_candidates
    
    async def _search_entity_by_name(self, name: str, limit: int) -> List[Entity]:
        """Search for entities by name in the graph database."""
        try:
            # Use exact match first
            exact_query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower($name)
            RETURN e
            LIMIT $limit
            """
            
            exact_results = await self.neo4j_manager.execute_query_async(
                exact_query, 
                {"name": name, "limit": limit}
            )
            
            entities = []
            for record in exact_results:
                entity_data = record.get('e', {})
                if entity_data:
                    entities.append(self._create_entity_from_record(entity_data))
            
            # If no exact matches, try fuzzy matching
            if not entities:
                fuzzy_query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($name)
                   OR toLower($name) CONTAINS toLower(e.name)
                RETURN e, e.name as entity_name
                ORDER BY length(e.name)
                LIMIT $limit
                """
                
                fuzzy_results = await self.neo4j_manager.execute_query_async(
                    fuzzy_query,
                    {"name": name, "limit": limit * 2}  # Get more for filtering
                )
                
                for record in fuzzy_results:
                    entity_data = record.get('e', {})
                    entity_name = record.get('entity_name', '')
                    
                    if entity_data and entity_name:
                        # Calculate similarity score
                        similarity = SequenceMatcher(None, name.lower(), entity_name.lower()).ratio()
                        if similarity >= self.similarity_threshold:
                            entity = self._create_entity_from_record(entity_data)
                            entity.similarity_score = similarity  # Add for sorting
                            entities.append(entity)
                
                # Sort by similarity score
                entities.sort(key=lambda e: getattr(e, 'similarity_score', 0), reverse=True)
            
            return entities[:limit]
            
        except Exception as e:
            logger.error(f"Error searching entity by name '{name}': {str(e)}")
            return []
    
    def _create_entity_from_record(self, record: Dict[str, Any]) -> Entity:
        """Create Entity object from Neo4j record."""
        return Entity(
            id=record.get('id', ''),
            name=record.get('name', ''),
            type=record.get('entity_type', 'unknown'),
            description=record.get('description'),
            properties=record.get('properties', {})
        )
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on ID."""
        seen_ids = set()
        unique_entities = []
        
        for entity in entities:
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _score_entity_matches(self, entities: List[Entity], query_text: str) -> List[Tuple[Entity, float]]:
        """Score entity matches based on relevance to query text."""
        scored_entities = []
        query_lower = query_text.lower()
        
        for entity in entities:
            score = 0.0
            
            # Base similarity score if available
            if hasattr(entity, 'similarity_score'):
                score += entity.similarity_score * 0.5
            
            # Exact name match bonus
            if entity.name.lower() in query_lower:
                score += 0.3
            
            # Description relevance
            if entity.description:
                desc_words = entity.description.lower().split()
                query_words = query_lower.split()
                common_words = set(desc_words) & set(query_words)
                if common_words:
                    score += len(common_words) / len(query_words) * 0.2
            
            scored_entities.append((entity, score))
        
        # Sort by score descending
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        return scored_entities
    
    async def disambiguate_entities(self, entities: List[Entity], context: str = "") -> List[Entity]:
        """
        Disambiguate entities when multiple matches are found.
        
        Args:
            entities: List of potential entity matches
            context: Additional context to help with disambiguation
            
        Returns:
            List[Entity]: Disambiguated entities
        """
        if len(entities) <= 1:
            return entities
        
        try:
            # Get additional context for each entity
            enriched_entities = []
            for entity in entities:
                # Get related entities and relationships for context
                related_info = await self._get_entity_context(entity.id)
                entity.context_info = related_info
                enriched_entities.append(entity)
            
            # Score entities based on context relevance
            if context:
                context_scored = self._score_by_context(enriched_entities, context)
                return [entity for entity, score in context_scored[:3]]  # Top 3
            
            return enriched_entities[:3]  # Return top 3 if no context
            
        except Exception as e:
            logger.error(f"Error in entity disambiguation: {str(e)}")
            return entities[:3]  # Fallback to first 3
    
    async def _get_entity_context(self, entity_id: str) -> Dict[str, Any]:
        """Get contextual information for an entity."""
        try:
            context_query = """
            MATCH (e:Entity {id: $entity_id})
            OPTIONAL MATCH (e)-[r]-(related)
            RETURN e, 
                   collect(DISTINCT type(r)) as relationship_types,
                   collect(DISTINCT labels(related)) as related_labels,
                   count(related) as connection_count
            """
            
            result = await self.neo4j_manager.execute_query_async(
                context_query,
                {"entity_id": entity_id}
            )
            
            if result:
                record = result[0]
                return {
                    'relationship_types': record.get('relationship_types', []),
                    'related_labels': record.get('related_labels', []),
                    'connection_count': record.get('connection_count', 0)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting entity context for {entity_id}: {str(e)}")
            return {}
    
    def _score_by_context(self, entities: List[Entity], context: str) -> List[Tuple[Entity, float]]:
        """Score entities based on context relevance."""
        scored_entities = []
        context_lower = context.lower()
        context_words = set(context_lower.split())
        
        for entity in entities:
            score = 0.0
            context_info = getattr(entity, 'context_info', {})
            
            # Score based on relationship types
            rel_types = context_info.get('relationship_types', [])
            for rel_type in rel_types:
                if any(word in rel_type.lower() for word in context_words):
                    score += 0.2
            
            # Score based on connection count (more connected = more important)
            connection_count = context_info.get('connection_count', 0)
            score += min(connection_count / 10.0, 0.3)  # Cap at 0.3
            
            # Score based on entity description
            if entity.description:
                desc_words = set(entity.description.lower().split())
                common_words = desc_words & context_words
                if common_words:
                    score += len(common_words) / len(context_words) * 0.3
            
            scored_entities.append((entity, score))
        
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        return scored_entities


class GraphTraverser:
    """Handles graph traversal algorithms and path finding."""
    
    def __init__(self, neo4j_manager):
        self.neo4j_manager = neo4j_manager
        self.max_traversal_depth = 4
        self.max_paths_per_query = 50
    
    async def traverse_relationships(
        self, 
        start_entities: List[Entity], 
        depth: int = 2,
        relationship_types: Optional[List[RelationshipType]] = None,
        direction: str = "both"
    ) -> GraphResult:
        """
        Traverse relationships from starting entities.
        
        Args:
            start_entities: Starting entities for traversal
            depth: Maximum traversal depth
            relationship_types: Optional filter for relationship types
            direction: Direction of traversal ("outgoing", "incoming", "both")
            
        Returns:
            GraphResult: Traversal results with entities, relationships, and paths
        """
        try:
            if not start_entities:
                return GraphResult()
            
            depth = min(depth, self.max_traversal_depth)  # Safety limit
            
            all_entities = []
            all_relationships = []
            all_paths = []
            
            for start_entity in start_entities:
                result = await self._traverse_from_entity(
                    start_entity.id, 
                    depth, 
                    relationship_types, 
                    direction
                )
                
                all_entities.extend(result['entities'])
                all_relationships.extend(result['relationships'])
                all_paths.extend(result['paths'])
            
            # Deduplicate results
            unique_entities = self._deduplicate_entities_by_id(all_entities)
            unique_relationships = self._deduplicate_relationships(all_relationships)
            
            return GraphResult(
                entities=unique_entities,
                relationships=unique_relationships,
                paths=all_paths[:self.max_paths_per_query]
            )
            
        except Exception as e:
            logger.error(f"Error in relationship traversal: {str(e)}")
            return GraphResult()
    
    async def _traverse_from_entity(
        self, 
        entity_id: str, 
        depth: int,
        relationship_types: Optional[List[RelationshipType]] = None,
        direction: str = "both"
    ) -> Dict[str, List]:
        """Traverse relationships from a single entity."""
        try:
            # Build relationship type filter
            rel_filter = ""
            if relationship_types:
                types_str = "|".join([rt.value for rt in relationship_types])
                rel_filter = f":{types_str}"
            
            # Build direction pattern
            if direction == "outgoing":
                pattern = f"-[r{rel_filter}*1..{depth}]->"
            elif direction == "incoming":
                pattern = f"<-[r{rel_filter}*1..{depth}]-"
            else:  # both
                pattern = f"-[r{rel_filter}*1..{depth}]-"
            
            query = f"""
            MATCH (start:Entity {{id: $entity_id}})
            MATCH path = (start){pattern}(end)
            WHERE start <> end
            RETURN path,
                   nodes(path) as path_nodes,
                   relationships(path) as path_relationships,
                   length(path) as path_length
            ORDER BY path_length
            LIMIT {self.max_paths_per_query}
            """
            
            results = await self.neo4j_manager.execute_query_async(
                query,
                {"entity_id": entity_id}
            )
            
            entities = []
            relationships = []
            paths = []
            
            for record in results:
                # Extract entities from path
                path_nodes = record.get('path_nodes', [])
                for node_data in path_nodes:
                    if node_data.get('id'):
                        entity = self._create_entity_from_node(node_data)
                        entities.append(entity)
                
                # Extract relationships from path
                path_rels = record.get('path_relationships', [])
                for rel_data in path_rels:
                    relationship = self._create_relationship_from_record(rel_data)
                    relationships.append(relationship)
                
                # Extract path as list of entity IDs
                path_ids = [node.get('id') for node in path_nodes if node.get('id')]
                if path_ids:
                    paths.append(path_ids)
            
            return {
                'entities': entities,
                'relationships': relationships,
                'paths': paths
            }
            
        except Exception as e:
            logger.error(f"Error traversing from entity {entity_id}: {str(e)}")
            return {'entities': [], 'relationships': [], 'paths': []}
    
    async def find_paths_between_entities(
        self, 
        source_entity_id: str, 
        target_entity_id: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """
        Find paths between two specific entities.
        
        Args:
            source_entity_id: ID of source entity
            target_entity_id: ID of target entity
            max_depth: Maximum path length
            
        Returns:
            List[List[str]]: List of paths (each path is a list of entity IDs)
        """
        try:
            max_depth = min(max_depth, self.max_traversal_depth)
            
            query = f"""
            MATCH (source:Entity {{id: $source_id}})
            MATCH (target:Entity {{id: $target_id}})
            MATCH path = shortestPath((source)-[*1..{max_depth}]-(target))
            RETURN path, nodes(path) as path_nodes, length(path) as path_length
            ORDER BY path_length
            LIMIT 10
            """
            
            results = await self.neo4j_manager.execute_query_async(
                query,
                {"source_id": source_entity_id, "target_id": target_entity_id}
            )
            
            paths = []
            for record in results:
                path_nodes = record.get('path_nodes', [])
                path_ids = [node.get('id') for node in path_nodes if node.get('id')]
                if path_ids:
                    paths.append(path_ids)
            
            return paths
            
        except Exception as e:
            logger.error(f"Error finding paths between {source_entity_id} and {target_entity_id}: {str(e)}")
            return []
    
    async def extract_subgraph(
        self, 
        center_entities: List[Entity], 
        radius: int = 2,
        max_nodes: int = 100
    ) -> GraphResult:
        """
        Extract a subgraph around center entities.
        
        Args:
            center_entities: Entities at the center of the subgraph
            radius: Maximum distance from center entities
            max_nodes: Maximum number of nodes in subgraph
            
        Returns:
            GraphResult: Extracted subgraph
        """
        try:
            if not center_entities:
                return GraphResult()
            
            center_ids = [entity.id for entity in center_entities]
            
            query = f"""
            MATCH (center:Entity)
            WHERE center.id IN $center_ids
            MATCH (center)-[*0..{radius}]-(node)
            WITH DISTINCT node
            LIMIT {max_nodes}
            MATCH (node)-[rel]-(connected)
            WHERE connected.id IN [n.id | n IN collect(node)]
            RETURN DISTINCT node, rel, connected
            """
            
            results = await self.neo4j_manager.execute_query_async(
                query,
                {"center_ids": center_ids}
            )
            
            entities = set()
            relationships = []
            
            for record in results:
                # Add nodes
                node_data = record.get('node')
                if node_data and node_data.get('id'):
                    entity = self._create_entity_from_node(node_data)
                    entities.add(entity.id)  # Use ID for deduplication
                
                connected_data = record.get('connected')
                if connected_data and connected_data.get('id'):
                    entity = self._create_entity_from_node(connected_data)
                    entities.add(entity.id)
                
                # Add relationships
                rel_data = record.get('rel')
                if rel_data:
                    relationship = self._create_relationship_from_record(rel_data)
                    relationships.append(relationship)
            
            # Convert entity IDs back to Entity objects
            entity_objects = []
            for entity_id in entities:
                # Get full entity data
                entity_data = await self._get_entity_by_id(entity_id)
                if entity_data:
                    entity_objects.append(entity_data)
            
            return GraphResult(
                entities=entity_objects,
                relationships=self._deduplicate_relationships(relationships),
                paths=[]  # Paths not relevant for subgraph extraction
            )
            
        except Exception as e:
            logger.error(f"Error extracting subgraph: {str(e)}")
            return GraphResult()
    
    def _create_entity_from_node(self, node_data: Dict[str, Any]) -> Entity:
        """Create Entity object from Neo4j node data."""
        return Entity(
            id=node_data.get('id', ''),
            name=node_data.get('name', ''),
            type=node_data.get('entity_type', 'unknown'),
            description=node_data.get('description'),
            properties=node_data.get('properties', {})
        )
    
    def _create_relationship_from_record(self, rel_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create relationship dictionary from Neo4j relationship data."""
        return {
            'type': rel_data.get('type', ''),
            'properties': rel_data.get('properties', {}),
            'start_node': rel_data.get('start_node', ''),
            'end_node': rel_data.get('end_node', '')
        }
    
    def _deduplicate_entities_by_id(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on ID."""
        seen_ids = set()
        unique_entities = []
        
        for entity in entities:
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships."""
        seen_rels = set()
        unique_rels = []
        
        for rel in relationships:
            # Create a unique key for the relationship
            key = (rel.get('type', ''), rel.get('start_node', ''), rel.get('end_node', ''))
            if key not in seen_rels:
                seen_rels.add(key)
                unique_rels.append(rel)
        
        return unique_rels
    
    async def _get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID from the database."""
        try:
            query = "MATCH (e:Entity {id: $entity_id}) RETURN e"
            results = await self.neo4j_manager.execute_query_async(
                query,
                {"entity_id": entity_id}
            )
            
            if results:
                entity_data = results[0].get('e', {})
                return self._create_entity_from_node(entity_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting entity by ID {entity_id}: {str(e)}")
            return None


class CypherQueryGenerator:
    """Generates and optimizes Cypher queries dynamically."""
    
    def __init__(self, neo4j_manager):
        self.neo4j_manager = neo4j_manager
        self.query_builder = CypherQueryBuilder()
        
        # Query templates for common patterns
        self.query_templates = {
            'find_entity': """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($name)
                RETURN e
                ORDER BY e.name
                LIMIT $limit
            """,
            
            'entity_relationships': """
                MATCH (e:Entity {id: $entity_id})-[r]-(related)
                RETURN e, r, related, type(r) as rel_type
                ORDER BY rel_type
                LIMIT $limit
            """,
            
            'multi_hop_traversal': """
                MATCH (start:Entity {id: $start_id})
                MATCH path = (start)-[*1..$depth]-(end:Entity)
                WHERE start <> end
                RETURN path, length(path) as depth
                ORDER BY depth
                LIMIT $limit
            """,
            
            'shortest_path': """
                MATCH (start:Entity {id: $start_id})
                MATCH (end:Entity {id: $end_id})
                MATCH path = shortestPath((start)-[*1..$max_depth]-(end))
                RETURN path
            """,
            
            'subgraph_extraction': """
                MATCH (center:Entity {id: $center_id})
                MATCH (center)-[*0..$radius]-(node)
                WITH DISTINCT node
                MATCH (node)-[rel]-(connected)
                WHERE connected IN collect(node)
                RETURN node, rel, connected
                LIMIT $max_nodes
            """
        }
    
    def generate_entity_search_query(
        self, 
        search_term: str, 
        entity_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate optimized query for entity search.
        
        Args:
            search_term: Term to search for
            entity_types: Optional filter for entity types
            limit: Maximum results
            
        Returns:
            Tuple[str, Dict[str, Any]]: Query string and parameters
        """
        self.query_builder.reset()
        
        # Build WHERE conditions
        where_conditions = ["toLower(e.name) CONTAINS toLower($search_term)"]
        parameters = {"search_term": search_term, "limit": limit}
        
        if entity_types:
            where_conditions.append("e.entity_type IN $entity_types")
            parameters["entity_types"] = entity_types
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        MATCH (e:Entity)
        WHERE {where_clause}
        RETURN e, e.name as entity_name
        ORDER BY 
            CASE WHEN toLower(e.name) = toLower($search_term) THEN 0 ELSE 1 END,
            length(e.name),
            e.name
        LIMIT $limit
        """
        
        return query, parameters
    
    def generate_relationship_traversal_query(
        self,
        start_entity_ids: List[str],
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
        depth: int = 2,
        limit: int = 50
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate optimized query for relationship traversal.
        
        Args:
            start_entity_ids: Starting entity IDs
            relationship_types: Optional relationship type filter
            direction: Traversal direction
            depth: Maximum depth
            limit: Maximum results
            
        Returns:
            Tuple[str, Dict[str, Any]]: Query string and parameters
        """
        # Build relationship pattern
        rel_filter = ""
        if relationship_types:
            types_str = "|".join(relationship_types)
            rel_filter = f":{types_str}"
        
        if direction == "outgoing":
            pattern = f"-[r{rel_filter}*1..{depth}]->"
        elif direction == "incoming":
            pattern = f"<-[r{rel_filter}*1..{depth}]-"
        else:  # both
            pattern = f"-[r{rel_filter}*1..{depth}]-"
        
        query = f"""
        MATCH (start:Entity)
        WHERE start.id IN $start_ids
        MATCH path = (start){pattern}(end:Entity)
        WHERE start <> end
        RETURN path,
               start,
               end,
               relationships(path) as path_rels,
               length(path) as path_length
        ORDER BY path_length, start.name, end.name
        LIMIT $limit
        """
        
        parameters = {
            "start_ids": start_entity_ids,
            "limit": limit
        }
        
        return query, parameters
    
    def generate_path_finding_query(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 4,
        relationship_types: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate query to find paths between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum path length
            relationship_types: Optional relationship type filter
            
        Returns:
            Tuple[str, Dict[str, Any]]: Query string and parameters
        """
        rel_filter = ""
        if relationship_types:
            types_str = "|".join(relationship_types)
            rel_filter = f":{types_str}"
        
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MATCH path = allShortestPaths((source)-[{rel_filter}*1..{max_depth}]-(target))
        RETURN path,
               nodes(path) as path_nodes,
               relationships(path) as path_rels,
               length(path) as path_length
        ORDER BY path_length
        LIMIT 10
        """
        
        parameters = {
            "source_id": source_id,
            "target_id": target_id
        }
        
        return query, parameters
    
    def optimize_query_for_performance(self, query: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize a Cypher query for better performance.
        
        Args:
            query: Original query
            parameters: Query parameters
            
        Returns:
            Tuple[str, Dict[str, Any]]: Optimized query and parameters
        """
        optimized_query = query
        
        # Add USING INDEX hints where appropriate
        if "MATCH (e:Entity)" in query and "e.name" in query:
            optimized_query = optimized_query.replace(
                "MATCH (e:Entity)",
                "MATCH (e:Entity) USING INDEX e:Entity(name)"
            )
        
        # Add limits to prevent runaway queries
        if "LIMIT" not in optimized_query.upper():
            optimized_query += "\nLIMIT 1000"
        
        # Optimize relationship traversals
        if "*" in optimized_query and "LIMIT" in optimized_query.upper():
            # Already has limit, good
            pass
        elif "*" in optimized_query:
            # Add limit to traversal queries
            optimized_query += "\nLIMIT 100"
        
        return optimized_query, parameters
    
    async def execute_parameterized_query(
        self, 
        template_name: str, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute a parameterized query from templates.
        
        Args:
            template_name: Name of the query template
            parameters: Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        try:
            if template_name not in self.query_templates:
                raise ValueError(f"Unknown query template: {template_name}")
            
            query = self.query_templates[template_name]
            optimized_query, opt_params = self.optimize_query_for_performance(query, parameters)
            
            results = await self.neo4j_manager.execute_query_async(optimized_query, opt_params)
            return results
            
        except Exception as e:
            logger.error(f"Error executing parameterized query '{template_name}': {str(e)}")
            return []


class GraphNavigatorAgent(GraphNavigatorInterface):
    """
    Graph Navigator Agent implementation.
    
    Handles entity lookup, relationship traversal, and Cypher query generation.
    """
    
    def __init__(self, agent_id: str = "graph_navigator"):
        super().__init__(agent_id)
        self.neo4j_manager = get_neo4j_manager()
        
        # Initialize components
        self.entity_matcher = EntityMatcher(self.neo4j_manager)
        self.graph_traverser = GraphTraverser(self.neo4j_manager)
        self.query_generator = CypherQueryGenerator(self.neo4j_manager)
        
        # Message handlers
        self.message_handlers = {
            MessageType.GRAPH_SEARCH: self._handle_graph_search
        }
        
        logger.info(f"Graph Navigator Agent {agent_id} initialized")
    
    async def find_entities(self, query: str) -> List[Entity]:
        """
        Find entities in the graph matching the query.
        
        Args:
            query: Search query text
            
        Returns:
            List[Entity]: Matching entities
        """
        try:
            logger.info(f"Finding entities for query: {query[:100]}...")
            
            # Use fuzzy matching to find entities
            entities = await self.entity_matcher.find_entities_fuzzy(query, limit=10)
            
            # Disambiguate if multiple matches
            if len(entities) > 1:
                entities = await self.entity_matcher.disambiguate_entities(entities, query)
            
            logger.info(f"Found {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Error finding entities: {str(e)}")
            return []
    
    async def traverse_relationships(
        self, 
        entities: List[Entity], 
        depth: int = 2
    ) -> GraphResult:
        """
        Traverse relationships from given entities.
        
        Args:
            entities: Starting entities
            depth: Maximum traversal depth
            
        Returns:
            GraphResult: Traversal results
        """
        try:
            logger.info(f"Traversing relationships from {len(entities)} entities, depth={depth}")
            
            result = await self.graph_traverser.traverse_relationships(
                entities, 
                depth=depth,
                direction="both"
            )
            
            logger.info(f"Traversal found {len(result.entities)} entities, "
                       f"{len(result.relationships)} relationships, "
                       f"{len(result.paths)} paths")
            
            return result
            
        except Exception as e:
            logger.error(f"Error traversing relationships: {str(e)}")
            return GraphResult()
    
    async def execute_cypher_query(
        self, 
        cypher: str, 
        parameters: Dict = None
    ) -> GraphResult:
        """
        Execute a Cypher query against the graph database.
        
        Args:
            cypher: Cypher query string
            parameters: Query parameters
            
        Returns:
            GraphResult: Query results
        """
        try:
            logger.info(f"Executing Cypher query: {cypher[:100]}...")
            
            # Optimize query for performance
            optimized_query, opt_params = self.query_generator.optimize_query_for_performance(
                cypher, 
                parameters or {}
            )
            
            # Execute query
            results = await self.neo4j_manager.execute_query_async(optimized_query, opt_params)
            
            # Convert results to GraphResult
            entities = []
            relationships = []
            
            for record in results:
                # Extract entities from record
                for key, value in record.items():
                    if isinstance(value, dict) and value.get('id'):
                        # This looks like an entity
                        entity = Entity(
                            id=value.get('id', ''),
                            name=value.get('name', ''),
                            type=value.get('entity_type', 'unknown'),
                            description=value.get('description'),
                            properties=value.get('properties', {})
                        )
                        entities.append(entity)
            
            # Deduplicate entities
            unique_entities = self.graph_traverser._deduplicate_entities_by_id(entities)
            
            result = GraphResult(
                entities=unique_entities,
                relationships=relationships,
                cypher_query=cypher
            )
            
            logger.info(f"Query returned {len(unique_entities)} entities")
            return result
            
        except Exception as e:
            logger.error(f"Error executing Cypher query: {str(e)}")
            return GraphResult(cypher_query=cypher)
    
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages."""
        try:
            if message.message_type in self.message_handlers:
                return await self.message_handlers[message.message_type](message)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return self.create_message(
                MessageType.ERROR,
                {"error": str(e), "original_message": message.dict()}
            )
    
    async def _handle_graph_search(self, message: AgentMessage) -> AgentMessage:
        """Handle graph search requests."""
        try:
            payload = message.payload
            query = payload.get('query', '')
            entities = payload.get('entities', [])
            depth = payload.get('depth', 2)
            
            if query and not entities:
                # Find entities first
                found_entities = await self.find_entities(query)
                entities = [entity.dict() for entity in found_entities]
            
            # Convert entity dicts back to Entity objects
            entity_objects = [
                Entity(**entity_data) for entity_data in entities
            ]
            
            # Traverse relationships
            result = await self.traverse_relationships(entity_objects, depth)
            
            # Create response
            response_payload = {
                'entities': [entity.dict() for entity in result.entities],
                'relationships': result.relationships,
                'paths': result.paths,
                'cypher_query': result.cypher_query
            }
            
            return self.create_message(
                MessageType.RESPONSE,
                response_payload,
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            logger.error(f"Error handling graph search: {str(e)}")
            return self.create_message(
                MessageType.ERROR,
                {"error": str(e)},
                correlation_id=message.correlation_id
            )