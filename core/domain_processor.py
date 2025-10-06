"""
Domain-specific processing system for the Graph-Enhanced Agentic RAG system.

This module provides pluggable entity extraction, relationship patterns, and schema mapping
for different knowledge domains like technical documentation, research papers, etc.
"""

import re
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .document_processor import ExtractedEntity, EntityRelationship
from .models import EntityType


class DomainType(str, Enum):
    """Supported knowledge domains."""
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    RESEARCH_PAPERS = "research_papers"
    ENTERPRISE_KNOWLEDGE = "enterprise_knowledge"
    LEGAL_DOCUMENTS = "legal_documents"
    MEDICAL_LITERATURE = "medical_literature"
    GENERAL = "general"


@dataclass
class EntityPattern:
    """Pattern for extracting domain-specific entities."""
    name: str
    pattern: str
    entity_type: EntityType
    confidence: float
    flags: int = re.IGNORECASE
    description: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class RelationshipPattern:
    """Pattern for identifying domain-specific relationships."""
    name: str
    pattern: str
    relationship_type: str
    confidence: float
    source_group: int = 1  # Regex group for source entity
    target_group: int = 2  # Regex group for target entity
    flags: int = re.IGNORECASE
    description: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class DomainSchema:
    """Schema definition for a specific domain."""
    domain_type: DomainType
    name: str
    description: str
    entity_types: List[EntityType]
    entity_patterns: List[EntityPattern]
    relationship_patterns: List[RelationshipPattern]
    graph_constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDomainExtractor(ABC):
    """Abstract base class for domain-specific entity extractors."""
    
    def __init__(self, domain_schema: DomainSchema):
        self.domain_schema = domain_schema
        self.entity_patterns = {p.name: p for p in domain_schema.entity_patterns}
        self.relationship_patterns = {p.name: p for p in domain_schema.relationship_patterns}
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract domain-specific entities from text."""
        pass
    
    @abstractmethod
    def identify_relationships(self, text: str, entities: List[ExtractedEntity]) -> List[EntityRelationship]:
        """Identify domain-specific relationships between entities."""
        pass
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get list of entity types supported by this domain."""
        return self.domain_schema.entity_types
    
    def get_graph_constraints(self) -> List[str]:
        """Get Neo4j constraints for this domain."""
        return self.domain_schema.graph_constraints


class TechnicalDocumentationExtractor(BaseDomainExtractor):
    """Entity extractor for technical documentation domain."""
    
    def __init__(self, domain_schema: Optional[DomainSchema] = None):
        if domain_schema is None:
            domain_schema = self._create_default_schema()
        super().__init__(domain_schema)
    
    def _create_default_schema(self) -> DomainSchema:
        """Create default schema for technical documentation."""
        entity_patterns = [
            EntityPattern(
                name="programming_language",
                pattern=r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin|Scala|R|MATLAB|SQL|HTML|CSS)\b',
                entity_type=EntityType.TECHNOLOGY,
                confidence=0.9,
                description="Programming languages and markup languages",
                examples=["Python", "JavaScript", "C++"]
            ),
            EntityPattern(
                name="framework_library",
                pattern=r'\b(?:React|Angular|Vue\.js|Django|Flask|Spring|Express\.js|Node\.js|TensorFlow|PyTorch|Pandas|NumPy|jQuery|Bootstrap)\b',
                entity_type=EntityType.TECHNOLOGY,
                confidence=0.9,
                description="Software frameworks and libraries",
                examples=["React", "Django", "TensorFlow"]
            ),
            EntityPattern(
                name="database_system",
                pattern=r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Neo4j|SQLite|Oracle|SQL Server|Cassandra|DynamoDB)\b',
                entity_type=EntityType.TECHNOLOGY,
                confidence=0.9,
                description="Database management systems",
                examples=["MySQL", "MongoDB", "Neo4j"]
            ),
            EntityPattern(
                name="cloud_service",
                pattern=r'\b(?:AWS|Azure|Google Cloud|GCP|Docker|Kubernetes|Heroku|Vercel|Netlify|Firebase)\b',
                entity_type=EntityType.TECHNOLOGY,
                confidence=0.8,
                description="Cloud platforms and services",
                examples=["AWS", "Docker", "Kubernetes"]
            ),
            EntityPattern(
                name="api_endpoint",
                pattern=r'(?:GET|POST|PUT|DELETE|PATCH)\s+/[\w/\-\{\}]+',
                entity_type=EntityType.CONCEPT,
                confidence=0.8,
                description="API endpoints and HTTP methods",
                examples=["GET /api/users", "POST /auth/login"]
            ),
            EntityPattern(
                name="function_method",
                pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\(\)',
                entity_type=EntityType.CONCEPT,
                confidence=0.7,
                description="Function and method names",
                examples=["getData()", "processRequest()"]
            ),
            EntityPattern(
                name="class_interface",
                pattern=r'\b[A-Z][a-zA-Z0-9_]*(?:Service|Interface|Controller|Repository|Manager|Handler|Processor|Factory|Builder|Adapter|Proxy|Decorator|Observer|Strategy|Command|State|Visitor|Template|Bridge|Composite|Facade|Flyweight|Singleton|Abstract|Base|Impl)\b|\bI[A-Z][a-zA-Z0-9_]+\b',
                entity_type=EntityType.CONCEPT,
                confidence=0.8,
                description="Class and interface names",
                examples=["UserService", "IUserService", "DataController"]
            ),
            EntityPattern(
                name="configuration_file",
                pattern=r'\b[\w\-]+\.(?:json|yaml|yml|xml|ini|conf|config|env|properties)\b',
                entity_type=EntityType.DOCUMENT,
                confidence=0.8,
                description="Configuration files",
                examples=["config.json", "docker-compose.yml"]
            ),
            EntityPattern(
                name="version_number",
                pattern=r'\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9\-]+)?\b',
                entity_type=EntityType.CONCEPT,
                confidence=0.7,
                description="Version numbers",
                examples=["v1.2.3", "2.0.0-beta"]
            )
        ]
        
        relationship_patterns = [
            RelationshipPattern(
                name="implements_interface",
                pattern=r'(\w+)\s+implements\s+(\w+)',
                relationship_type="IMPLEMENTS",
                confidence=0.9,
                source_group=1,
                target_group=2,
                description="Class implements interface relationship",
                examples=["UserService implements IUserService"]
            ),
            RelationshipPattern(
                name="extends_class",
                pattern=r'(\w+)\s+extends\s+(\w+)',
                relationship_type="EXTENDS",
                confidence=0.9,
                source_group=1,
                target_group=2,
                description="Class inheritance relationship",
                examples=["AdminUser extends User"]
            ),
            RelationshipPattern(
                name="uses_technology",
                pattern=r'(?:uses?|utilizing?|built with|powered by)\s+([A-Za-z][A-Za-z0-9\.\+\-\s]+)',
                relationship_type="USES",
                confidence=0.8,
                source_group=0,  # Will be set contextually
                target_group=1,
                description="Technology usage relationship",
                examples=["built with React", "uses PostgreSQL"]
            ),
            RelationshipPattern(
                name="depends_on",
                pattern=r'(\w+)\s+(?:depends on|requires|needs)\s+(\w+)',
                relationship_type="DEPENDS_ON",
                confidence=0.8,
                source_group=1,
                target_group=2,
                description="Dependency relationship",
                examples=["Frontend depends on API", "Service requires Database"]
            ),
            RelationshipPattern(
                name="configures",
                pattern=r'(\w+)\s+(?:configures?|sets up|initializes)\s+(\w+)',
                relationship_type="CONFIGURES",
                confidence=0.7,
                source_group=1,
                target_group=2,
                description="Configuration relationship",
                examples=["Docker configures environment", "Webpack configures build"]
            )
        ]
        
        graph_constraints = [
            "CREATE CONSTRAINT tech_name_unique IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT api_endpoint_unique IF NOT EXISTS FOR (a:APIEndpoint) REQUIRE a.path IS UNIQUE",
            "CREATE INDEX tech_category_index IF NOT EXISTS FOR (t:Technology) ON (t.category)",
            "CREATE INDEX version_number_index IF NOT EXISTS FOR (v:Version) ON (v.number)"
        ]
        
        return DomainSchema(
            domain_type=DomainType.TECHNICAL_DOCUMENTATION,
            name="Technical Documentation",
            description="Schema for technical documentation, API docs, and software development content",
            entity_types=[EntityType.TECHNOLOGY, EntityType.CONCEPT, EntityType.DOCUMENT, EntityType.ORGANIZATION],
            entity_patterns=entity_patterns,
            relationship_patterns=relationship_patterns,
            graph_constraints=graph_constraints,
            metadata={
                "version": "1.0",
                "author": "Graph-Enhanced RAG System",
                "created_at": "2024-01-01"
            }
        )
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract technical documentation entities from text."""
        entities = []
        seen_entities = set()
        
        for pattern_name, pattern_info in self.entity_patterns.items():
            matches = re.finditer(pattern_info.pattern, text, pattern_info.flags)
            
            for match in matches:
                entity_text = match.group().strip()
                entity_key = (entity_text.lower(), pattern_info.entity_type.value)
                
                # Skip duplicates
                if entity_key in seen_entities:
                    continue
                seen_entities.add(entity_key)
                
                # Get context around the entity
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end].strip()
                
                entity = ExtractedEntity(
                    text=entity_text,
                    label=pattern_info.entity_type.value,
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=pattern_info.confidence,
                    context=context,
                    properties={
                        "domain": self.domain_schema.domain_type.value,
                        "pattern_name": pattern_name,
                        "extraction_method": "domain_specific"
                    }
                )
                entities.append(entity)
        
        return entities
    
    def identify_relationships(self, text: str, entities: List[ExtractedEntity]) -> List[EntityRelationship]:
        """Identify technical documentation relationships."""
        relationships = []
        
        # Create entity lookup for quick access
        entity_lookup = {ent.text.lower(): ent.text for ent in entities}
        
        for pattern_name, pattern_info in self.relationship_patterns.items():
            matches = re.finditer(pattern_info.pattern, text, pattern_info.flags)
            
            for match in matches:
                # Handle special case for "uses_technology" pattern
                if pattern_name == "uses_technology":
                    # Find the subject entity from context
                    context_start = max(0, match.start() - 100)
                    context_text = text[context_start:match.start()]
                    
                    # Look for entities in the preceding context
                    source_entity = None
                    for entity in entities:
                        if entity.text.lower() in context_text.lower():
                            source_entity = entity.text
                            break
                    
                    target_text = match.group(pattern_info.target_group).strip()
                    target_entity = self._find_matching_entity(target_text, entities)
                    
                    if source_entity and target_entity:
                        relationship = EntityRelationship(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            relationship_type=pattern_info.relationship_type,
                            confidence=pattern_info.confidence,
                            context=match.group(0),
                            evidence_text=match.group(0)
                        )
                        relationships.append(relationship)
                else:
                    # Standard pattern processing
                    source_text = match.group(pattern_info.source_group).strip()
                    target_text = match.group(pattern_info.target_group).strip()
                    
                    source_entity = self._find_matching_entity(source_text, entities)
                    target_entity = self._find_matching_entity(target_text, entities)
                    
                    if source_entity and target_entity and source_entity != target_entity:
                        relationship = EntityRelationship(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            relationship_type=pattern_info.relationship_type,
                            confidence=pattern_info.confidence,
                            context=match.group(0),
                            evidence_text=match.group(0)
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _find_matching_entity(self, text: str, entities: List[ExtractedEntity]) -> Optional[str]:
        """Find an entity that matches the given text."""
        text_lower = text.lower().strip()
        
        # Exact match first
        for entity in entities:
            if entity.text.lower() == text_lower:
                return entity.text
        
        # Partial match
        for entity in entities:
            entity_text_lower = entity.text.lower()
            if text_lower in entity_text_lower or entity_text_lower in text_lower:
                return entity.text
        
        return None


class ResearchPapersExtractor(BaseDomainExtractor):
    """Entity extractor for research papers domain."""
    
    def __init__(self, domain_schema: Optional[DomainSchema] = None):
        if domain_schema is None:
            domain_schema = self._create_default_schema()
        super().__init__(domain_schema)
    
    def _create_default_schema(self) -> DomainSchema:
        """Create default schema for research papers."""
        entity_patterns = [
            EntityPattern(
                name="research_method",
                pattern=r'\b(?:machine learning|deep learning|neural network|regression|classification|clustering|reinforcement learning|supervised learning|unsupervised learning|semi-supervised learning)\b',
                entity_type=EntityType.CONCEPT,
                confidence=0.9,
                description="Research methods and techniques",
                examples=["machine learning", "neural network"]
            ),
            EntityPattern(
                name="dataset",
                pattern=r'\b(?:ImageNet|MNIST|CIFAR|COCO|Wikipedia|Common Crawl|OpenWebText|BookCorpus)\b',
                entity_type=EntityType.DOCUMENT,
                confidence=0.9,
                description="Research datasets",
                examples=["ImageNet", "MNIST"]
            ),
            EntityPattern(
                name="metric",
                pattern=r'\b(?:accuracy|precision|recall|F1[- ]score|BLEU|ROUGE|perplexity|AUC|ROC)\b',
                entity_type=EntityType.CONCEPT,
                confidence=0.8,
                description="Evaluation metrics",
                examples=["accuracy", "F1-score", "BLEU"]
            ),
            EntityPattern(
                name="model_architecture",
                pattern=r'\b(?:transformer|BERT|GPT|ResNet|VGG|AlexNet|LSTM|GRU|CNN|RNN)\b',
                entity_type=EntityType.TECHNOLOGY,
                confidence=0.9,
                description="Model architectures",
                examples=["transformer", "BERT", "ResNet"]
            ),
            EntityPattern(
                name="author_name",
                pattern=r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s+et\s+al\.?)?\b',
                entity_type=EntityType.PERSON,
                confidence=0.7,
                description="Author names in academic format",
                examples=["John Smith", "Jane Doe et al."]
            )
        ]
        
        relationship_patterns = [
            RelationshipPattern(
                name="achieves_performance",
                pattern=r'(\w+(?:\s+\w+)*)\s+achieves?\s+(\d+(?:\.\d+)?%?\s+\w+)',
                relationship_type="ACHIEVES",
                confidence=0.8,
                description="Model performance achievement",
                examples=["BERT achieves 92% accuracy"]
            ),
            RelationshipPattern(
                name="trained_on",
                pattern=r'(\w+(?:\s+\w+)*)\s+(?:trained on|fine-tuned on)\s+(\w+(?:\s+\w+)*)',
                relationship_type="TRAINED_ON",
                confidence=0.9,
                description="Model training dataset relationship",
                examples=["GPT trained on Common Crawl"]
            ),
            RelationshipPattern(
                name="outperforms",
                pattern=r'(\w+(?:\s+\w+)*)\s+outperforms?\s+(\w+(?:\s+\w+)*)',
                relationship_type="OUTPERFORMS",
                confidence=0.8,
                description="Performance comparison relationship",
                examples=["BERT outperforms LSTM"]
            )
        ]
        
        return DomainSchema(
            domain_type=DomainType.RESEARCH_PAPERS,
            name="Research Papers",
            description="Schema for academic research papers and scientific literature",
            entity_types=[EntityType.CONCEPT, EntityType.TECHNOLOGY, EntityType.PERSON, EntityType.DOCUMENT],
            entity_patterns=entity_patterns,
            relationship_patterns=relationship_patterns,
            graph_constraints=[
                "CREATE CONSTRAINT author_name_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
                "CREATE CONSTRAINT dataset_name_unique IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE"
            ]
        )
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract research paper entities from text."""
        entities = []
        seen_entities = set()
        
        for pattern_name, pattern_info in self.entity_patterns.items():
            matches = re.finditer(pattern_info.pattern, text, pattern_info.flags)
            
            for match in matches:
                entity_text = match.group().strip()
                entity_key = (entity_text.lower(), pattern_info.entity_type.value)
                
                if entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end].strip()
                    
                    entity = ExtractedEntity(
                        text=entity_text,
                        label=pattern_info.entity_type.value,
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=pattern_info.confidence,
                        context=context,
                        properties={
                            "domain": self.domain_schema.domain_type.value,
                            "pattern_name": pattern_name,
                            "extraction_method": "domain_specific"
                        }
                    )
                    entities.append(entity)
        
        return entities
    
    def identify_relationships(self, text: str, entities: List[ExtractedEntity]) -> List[EntityRelationship]:
        """Identify research paper relationships."""
        relationships = []
        
        for pattern_name, pattern_info in self.relationship_patterns.items():
            matches = re.finditer(pattern_info.pattern, text, pattern_info.flags)
            
            for match in matches:
                source_text = match.group(pattern_info.source_group).strip()
                target_text = match.group(pattern_info.target_group).strip()
                
                source_entity = self._find_matching_entity(source_text, entities)
                target_entity = self._find_matching_entity(target_text, entities)
                
                if source_entity and target_entity:
                    relationship = EntityRelationship(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        relationship_type=pattern_info.relationship_type,
                        confidence=pattern_info.confidence,
                        context=match.group(0),
                        evidence_text=match.group(0)
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _find_matching_entity(self, text: str, entities: List[ExtractedEntity]) -> Optional[str]:
        """Find an entity that matches the given text."""
        text_lower = text.lower().strip()
        
        for entity in entities:
            if entity.text.lower() == text_lower:
                return entity.text
        
        for entity in entities:
            entity_text_lower = entity.text.lower()
            if text_lower in entity_text_lower or entity_text_lower in text_lower:
                return entity.text
        
        return None


class DomainProcessorManager:
    """Manager for domain-specific processing capabilities."""
    
    def __init__(self):
        self.extractors: Dict[DomainType, BaseDomainExtractor] = {}
        self.schemas: Dict[DomainType, DomainSchema] = {}
        self.current_domain: Optional[DomainType] = None
        
        # Register default extractors
        self._register_default_extractors()
    
    def _register_default_extractors(self):
        """Register default domain extractors."""
        # Technical documentation extractor
        tech_extractor = TechnicalDocumentationExtractor()
        self.register_extractor(DomainType.TECHNICAL_DOCUMENTATION, tech_extractor)
        
        # Research papers extractor
        research_extractor = ResearchPapersExtractor()
        self.register_extractor(DomainType.RESEARCH_PAPERS, research_extractor)
    
    def register_extractor(self, domain_type: DomainType, extractor: BaseDomainExtractor):
        """Register a domain-specific extractor."""
        self.extractors[domain_type] = extractor
        self.schemas[domain_type] = extractor.domain_schema
    
    def set_domain(self, domain_type: DomainType):
        """Set the current active domain."""
        if domain_type not in self.extractors:
            raise ValueError(f"Domain {domain_type} not registered")
        self.current_domain = domain_type
    
    def get_current_extractor(self) -> Optional[BaseDomainExtractor]:
        """Get the current active domain extractor."""
        if self.current_domain:
            return self.extractors.get(self.current_domain)
        return None
    
    def extract_entities(self, text: str, domain_type: Optional[DomainType] = None) -> List[ExtractedEntity]:
        """Extract entities using domain-specific extractor."""
        domain = domain_type or self.current_domain
        if not domain:
            raise ValueError("No domain specified and no current domain set")
        
        extractor = self.extractors.get(domain)
        if not extractor:
            raise ValueError(f"No extractor registered for domain {domain}")
        
        return extractor.extract_entities(text)
    
    def identify_relationships(self, text: str, entities: List[ExtractedEntity], 
                            domain_type: Optional[DomainType] = None) -> List[EntityRelationship]:
        """Identify relationships using domain-specific extractor."""
        domain = domain_type or self.current_domain
        if not domain:
            raise ValueError("No domain specified and no current domain set")
        
        extractor = self.extractors.get(domain)
        if not extractor:
            raise ValueError(f"No extractor registered for domain {domain}")
        
        return extractor.identify_relationships(text, entities)
    
    def get_domain_schema(self, domain_type: DomainType) -> Optional[DomainSchema]:
        """Get schema for a specific domain."""
        return self.schemas.get(domain_type)
    
    def list_available_domains(self) -> List[DomainType]:
        """List all available domains."""
        return list(self.extractors.keys())
    
    def load_custom_schema(self, schema_path: str) -> DomainSchema:
        """Load a custom domain schema from file."""
        schema_file = Path(schema_path)
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        # Convert JSON data to DomainSchema
        entity_patterns = [
            EntityPattern(**pattern_data) for pattern_data in schema_data.get('entity_patterns', [])
        ]
        
        relationship_patterns = [
            RelationshipPattern(**pattern_data) for pattern_data in schema_data.get('relationship_patterns', [])
        ]
        
        schema = DomainSchema(
            domain_type=DomainType(schema_data['domain_type']),
            name=schema_data['name'],
            description=schema_data['description'],
            entity_types=[EntityType(et) for et in schema_data.get('entity_types', [])],
            entity_patterns=entity_patterns,
            relationship_patterns=relationship_patterns,
            graph_constraints=schema_data.get('graph_constraints', []),
            metadata=schema_data.get('metadata', {})
        )
        
        return schema
    
    def save_schema(self, domain_type: DomainType, output_path: str):
        """Save a domain schema to file."""
        schema = self.schemas.get(domain_type)
        if not schema:
            raise ValueError(f"No schema found for domain {domain_type}")
        
        # Convert schema to JSON-serializable format
        schema_data = {
            'domain_type': schema.domain_type.value,
            'name': schema.name,
            'description': schema.description,
            'entity_types': [et.value for et in schema.entity_types],
            'entity_patterns': [
                {
                    'name': p.name,
                    'pattern': p.pattern,
                    'entity_type': p.entity_type.value,
                    'confidence': p.confidence,
                    'flags': p.flags,
                    'description': p.description,
                    'examples': p.examples
                }
                for p in schema.entity_patterns
            ],
            'relationship_patterns': [
                {
                    'name': p.name,
                    'pattern': p.pattern,
                    'relationship_type': p.relationship_type,
                    'confidence': p.confidence,
                    'source_group': p.source_group,
                    'target_group': p.target_group,
                    'flags': p.flags,
                    'description': p.description,
                    'examples': p.examples
                }
                for p in schema.relationship_patterns
            ],
            'graph_constraints': schema.graph_constraints,
            'metadata': schema.metadata
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)


# Global domain processor manager instance
_domain_manager = None


def get_domain_manager() -> DomainProcessorManager:
    """Get the global domain processor manager instance."""
    global _domain_manager
    if _domain_manager is None:
        _domain_manager = DomainProcessorManager()
    return _domain_manager


def configure_domain(domain_type: DomainType):
    """Configure the system for a specific domain."""
    manager = get_domain_manager()
    manager.set_domain(domain_type)


def register_custom_extractor(domain_type: DomainType, extractor: BaseDomainExtractor):
    """Register a custom domain extractor."""
    manager = get_domain_manager()
    manager.register_extractor(domain_type, extractor)