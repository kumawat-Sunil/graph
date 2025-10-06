"""
Document processing system for the Graph-Enhanced Agentic RAG system.

This module handles text extraction, chunking, entity extraction, and relationship
identification from documents during the ingestion pipeline.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

from .models import Document, Entity, EntityType, DocumentType


class ChunkingStrategy(str, Enum):
    """Available text chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC_BASED = "semantic_based"


@dataclass
class TextChunk:
    """Represents a chunk of text from a document."""
    id: str
    content: str
    start_position: int
    end_position: int
    document_id: str
    chunk_index: int
    metadata: Dict[str, Any]


@dataclass
class ExtractedEntity:
    """Represents an entity extracted from text."""
    text: str
    label: str
    start_position: int
    end_position: int
    confidence: float
    context: str
    properties: Dict[str, Any]


@dataclass
class EntityRelationship:
    """Represents a relationship between two entities."""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    context: str
    evidence_text: str


class DocumentProcessor:
    """Main document processing system."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm", domain_type: Optional[str] = None):
        """
        Initialize the document processor.
        
        Args:
            spacy_model: Name of the spaCy model to use for NLP processing
            domain_type: Optional domain type for domain-specific processing
        """
        self.nlp = None
        self.domain_type = domain_type
        self.domain_manager = None
        
        # Initialize domain manager lazily to avoid circular imports
        self._initialize_domain_manager()
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                # Fallback to a basic model if the specified one isn't available
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print(f"Warning: No spaCy model found. Using fallback entity extraction.")
                    self.nlp = None
            
            # Configure NLP pipeline
            if self.nlp and "ner" not in self.nlp.pipe_names:
                try:
                    self.nlp.add_pipe("ner")
                except ValueError:
                    # Pipeline already has NER
                    pass
        else:
            print("Warning: spaCy not available. Using fallback entity extraction.")
        
        # Entity type mapping from spaCy labels to our EntityType enum
        self.entity_type_mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.TECHNOLOGY,
            "WORK_OF_ART": EntityType.DOCUMENT,
            "LAW": EntityType.CONCEPT,
            "LANGUAGE": EntityType.CONCEPT,
            "NORP": EntityType.CONCEPT,
            "FAC": EntityType.LOCATION,
            "MISC": EntityType.GENERIC
        }
        
        # Relationship patterns for identifying relationships between entities
        self.relationship_patterns = [
            {
                "pattern": r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "type": "IS_A",
                "confidence": 0.8
            },
            {
                "pattern": r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:works for|employed by|part of)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "type": "WORKS_FOR",
                "confidence": 0.9
            },
            {
                "pattern": r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:is located in|located in|based in|from)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "type": "LOCATED_IN",
                "confidence": 0.8
            },
            {
                "pattern": r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:created|developed|invented|built)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "type": "CREATED",
                "confidence": 0.9
            },
            {
                "pattern": r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:uses|utilizes|implements|employs)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "type": "USES",
                "confidence": 0.7
            },
            {
                "pattern": r"([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:related to|associated with|connected to)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
                "type": "RELATED_TO",
                "confidence": 0.6
            }
        ]
    
    def extract_text(self, content: str, document_type: DocumentType) -> str:
        """
        Extract plain text from document content based on document type.
        
        Args:
            content: Raw document content
            document_type: Type of the document
            
        Returns:
            Extracted plain text
        """
        if document_type == DocumentType.TEXT:
            return content
        
        elif document_type == DocumentType.MARKDOWN:
            # Remove markdown formatting
            text = re.sub(r'#{1,6}\s+', '', content)  # Headers
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
            text = re.sub(r'`(.*?)`', r'\1', text)  # Inline code
            text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Code blocks
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
            return text
        
        elif document_type == DocumentType.HTML:
            # Basic HTML tag removal
            text = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
            text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # HTML entities
            return text
        
        elif document_type == DocumentType.JSON:
            # Extract text values from JSON (basic implementation)
            import json
            try:
                data = json.loads(content)
                text_parts = []
                self._extract_text_from_json(data, text_parts)
                return ' '.join(text_parts)
            except json.JSONDecodeError:
                return content
        
        else:
            # For other types, return as-is
            return content
    
    def _extract_text_from_json(self, obj: Any, text_parts: List[str]) -> None:
        """Recursively extract text from JSON objects."""
        if isinstance(obj, dict):
            for value in obj.values():
                self._extract_text_from_json(value, text_parts)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_text_from_json(item, text_parts)
        elif isinstance(obj, str) and len(obj.strip()) > 0:
            text_parts.append(obj.strip())
    
    def chunk_text(
        self, 
        text: str, 
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_BASED,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[TextChunk]:
        """
        Split text into chunks using the specified strategy.
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy to use
            chunk_size: Target size for chunks (in characters)
            overlap: Overlap between chunks (in characters)
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        if strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(text, chunk_size, overlap)
        
        elif strategy == ChunkingStrategy.SENTENCE_BASED:
            chunks = self._chunk_sentence_based(text, chunk_size)
        
        elif strategy == ChunkingStrategy.PARAGRAPH_BASED:
            chunks = self._chunk_paragraph_based(text, chunk_size)
        
        elif strategy == ChunkingStrategy.SEMANTIC_BASED:
            # For now, fall back to sentence-based
            chunks = self._chunk_sentence_based(text, chunk_size)
        
        return chunks
    
    def _chunk_fixed_size(self, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """Create fixed-size chunks with overlap."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at word boundary
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunk = TextChunk(
                    id=str(uuid.uuid4()),
                    content=chunk_content,
                    start_position=start,
                    end_position=end,
                    document_id="",  # Will be set by caller
                    chunk_index=chunk_index,
                    metadata={"strategy": "fixed_size", "size": len(chunk_content)}
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start = max(start + chunk_size - overlap, end)
        
        return chunks
    
    def _chunk_sentence_based(self, text: str, max_chunk_size: int) -> List[TextChunk]:
        """Create chunks based on sentence boundaries."""
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback: simple sentence splitting
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = text.find(sentence)
            else:
                if current_chunk:
                    chunk = TextChunk(
                        id=str(uuid.uuid4()),
                        content=current_chunk,
                        start_position=current_start,
                        end_position=current_start + len(current_chunk),
                        document_id="",
                        chunk_index=chunk_index,
                        metadata={"strategy": "sentence_based", "sentence_count": current_chunk.count('.') + 1}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = sentence
                current_start = text.find(sentence, current_start + len(current_chunk) if chunks else 0)
        
        # Add the last chunk
        if current_chunk:
            chunk = TextChunk(
                id=str(uuid.uuid4()),
                content=current_chunk,
                start_position=current_start,
                end_position=current_start + len(current_chunk),
                document_id="",
                chunk_index=chunk_index,
                metadata={"strategy": "sentence_based", "sentence_count": current_chunk.count('.') + 1}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_paragraph_based(self, text: str, max_chunk_size: int) -> List[TextChunk]:
        """Create chunks based on paragraph boundaries."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    current_start = text.find(paragraph)
            else:
                if current_chunk:
                    chunk = TextChunk(
                        id=str(uuid.uuid4()),
                        content=current_chunk,
                        start_position=current_start,
                        end_position=current_start + len(current_chunk),
                        document_id="",
                        chunk_index=chunk_index,
                        metadata={"strategy": "paragraph_based", "paragraph_count": current_chunk.count('\n\n') + 1}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = paragraph
                current_start = text.find(paragraph, current_start + len(current_chunk) if chunks else 0)
        
        # Add the last chunk
        if current_chunk:
            chunk = TextChunk(
                id=str(uuid.uuid4()),
                content=current_chunk,
                start_position=current_start,
                end_position=current_start + len(current_chunk),
                document_id="",
                chunk_index=chunk_index,
                metadata={"strategy": "paragraph_based", "paragraph_count": current_chunk.count('\n\n') + 1}
            )
            chunks.append(chunk)
        
        return chunks
    
    def extract_entities(self, text: str, use_domain_specific: bool = True) -> List[ExtractedEntity]:
        """
        Extract entities from text using NLP processing and domain-specific patterns.
        
        Args:
            text: Text to extract entities from
            use_domain_specific: Whether to use domain-specific extraction
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Initialize domain manager if needed
        if use_domain_specific:
            self._initialize_domain_manager()
        
        # First, try domain-specific extraction if enabled and domain is set
        if use_domain_specific and self.domain_manager and self.domain_manager.current_domain:
            try:
                domain_entities = self.domain_manager.extract_entities(text)
                entities.extend(domain_entities)
            except Exception as e:
                print(f"Warning: Domain-specific extraction failed: {e}")
        
        # Then use general NLP extraction
        if self.nlp:
            # Use spaCy for entity extraction
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Get context around the entity (50 characters before and after)
                context_start = max(0, ent.start_char - 50)
                context_end = min(len(text), ent.end_char + 50)
                context = text[context_start:context_end].strip()
                
                # Map spaCy label to our EntityType
                entity_type = self.entity_type_mapping.get(ent.label_, EntityType.GENERIC)
                
                extracted_entity = ExtractedEntity(
                    text=ent.text.strip(),
                    label=entity_type.value,
                    start_position=ent.start_char,
                    end_position=ent.end_char,
                    confidence=0.8,  # Default confidence for spaCy entities
                    context=context,
                    properties={
                        "spacy_label": ent.label_,
                        "lemma": ent.lemma_ if hasattr(ent, 'lemma_') else ent.text,
                        "pos": ent.root.pos_ if hasattr(ent, 'root') else None,
                        "extraction_method": "spacy"
                    }
                )
                entities.append(extracted_entity)
        else:
            # Fallback: simple pattern-based entity extraction
            fallback_entities = self._extract_entities_fallback(text)
            entities.extend(fallback_entities)
        
        # Remove duplicates while preserving domain-specific entities
        unique_entities = self._deduplicate_entities(entities)
        
        return unique_entities
    
    def _extract_entities_fallback(self, text: str) -> List[ExtractedEntity]:
        """Fallback entity extraction using simple patterns."""
        entities = []
        
        # Enhanced patterns for common entity types (order matters - more specific first)
        patterns = [
            # AI/ML specific terms
            (r'\b(?:Transformer|BERT|GPT|T5|Attention|Self-Attention|Multi-Head|Encoder|Decoder|Neural Network|Deep Learning|Machine Learning|AI|Artificial Intelligence)\b', EntityType.TECHNOLOGY, 0.9),
            # General technologies
            (r'\b(?:Python|Java|JavaScript|C\+\+|Go|Rust|Ruby|Node\.js|React|Angular|Vue)\b', EntityType.TECHNOLOGY, 0.8),
            # Organizations
            (r'\b[A-Z][a-zA-Z]+ Inc\.?\b|\b[A-Z][a-zA-Z]+ Corp\.?\b|\b[A-Z][a-zA-Z]+ LLC\b|\b[A-Z][a-zA-Z]+ Ltd\.?\b', EntityType.ORGANIZATION, 0.7),
            # Person names
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', EntityType.PERSON, 0.6),
            # Locations
            (r'\b[A-Z][a-z]+(?:, [A-Z][A-Z])?\b', EntityType.LOCATION, 0.5),
            # Concepts (capitalized words that might be important)
            (r'\b[A-Z][a-z]{2,}\b', EntityType.CONCEPT, 0.4),
        ]
        
        seen_entities = set()  # To avoid duplicates
        
        print(f"DEBUG: Running fallback extraction on text length {len(text)}")
        
        for pattern, entity_type, confidence in patterns:
            matches = re.finditer(pattern, text)
            
            for match in matches:
                entity_text = match.group().strip()
                entity_key = (entity_text.lower(), entity_type.value)
                
                # Skip if we've already found this entity
                if entity_key in seen_entities:
                    continue
                
                seen_entities.add(entity_key)
                
                # Get context around the entity
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                context = text[context_start:context_end].strip()
                
                extracted_entity = ExtractedEntity(
                    text=entity_text,
                    label=entity_type.value,
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=confidence,
                    context=context,
                    properties={"extraction_method": "pattern_based"}
                )
                entities.append(extracted_entity)
        
        print(f"DEBUG: Fallback extraction found {len(entities)} entities")
        return entities
    
    def identify_relationships(
        self, 
        text: str, 
        entities: List[ExtractedEntity],
        use_domain_specific: bool = True
    ) -> List[EntityRelationship]:
        """
        Identify relationships between entities in the text.
        
        Args:
            text: Source text
            entities: List of extracted entities
            use_domain_specific: Whether to use domain-specific relationship patterns
            
        Returns:
            List of identified relationships
        """
        relationships = []
        
        # Initialize domain manager if needed
        if use_domain_specific:
            self._initialize_domain_manager()
        
        # First, try domain-specific relationship identification
        if use_domain_specific and self.domain_manager and self.domain_manager.current_domain:
            try:
                domain_relationships = self.domain_manager.identify_relationships(text, entities)
                relationships.extend(domain_relationships)
            except Exception as e:
                print(f"Warning: Domain-specific relationship identification failed: {e}")
        
        # Create entity lookup for quick access
        entity_positions = {
            (ent.start_position, ent.end_position): ent.text 
            for ent in entities
        }
        
        # Apply general relationship patterns
        for pattern_info in self.relationship_patterns:
            pattern = pattern_info["pattern"]
            rel_type = pattern_info["type"]
            confidence = pattern_info["confidence"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                source_text = match.group(1).strip()
                target_text = match.group(2).strip()
                
                # Check if the matched text corresponds to extracted entities
                source_entity = self._find_matching_entity(source_text, entities)
                target_entity = self._find_matching_entity(target_text, entities)
                
                if source_entity and target_entity and source_entity != target_entity:
                    # Get context around the relationship
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end].strip()
                    
                    relationship = EntityRelationship(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        relationship_type=rel_type,
                        confidence=confidence,
                        context=context,
                        evidence_text=match.group(0)
                    )
                    relationships.append(relationship)
        
        # Also identify co-occurrence relationships
        relationships.extend(self._identify_cooccurrence_relationships(text, entities))
        
        # Remove duplicate relationships
        unique_relationships = self._deduplicate_relationships(relationships)
        
        return unique_relationships
    
    def _initialize_domain_manager(self):
        """Initialize domain manager lazily to avoid circular imports."""
        if self.domain_manager is None:
            try:
                from .domain_processor import get_domain_manager, DomainType
                self.domain_manager = get_domain_manager()
                
                # Set domain if specified
                if self.domain_type:
                    domain_enum = DomainType(self.domain_type)
                    self.domain_manager.set_domain(domain_enum)
            except ImportError:
                # Domain processing not available
                self.domain_manager = None
    
    def _find_matching_entity(self, text: str, entities: List[ExtractedEntity]) -> Optional[str]:
        """Find an entity that matches the given text."""
        text_lower = text.lower().strip()
        
        for entity in entities:
            entity_text_lower = entity.text.lower().strip()
            
            # Exact match
            if text_lower == entity_text_lower:
                return entity.text
            
            # Partial match (entity text contains the search text or vice versa)
            if text_lower in entity_text_lower or entity_text_lower in text_lower:
                return entity.text
        
        return None
    
    def _identify_cooccurrence_relationships(
        self, 
        text: str, 
        entities: List[ExtractedEntity]
    ) -> List[EntityRelationship]:
        """Identify relationships based on entity co-occurrence in sentences."""
        relationships = []
        
        if self.nlp:
            # Use spaCy for sentence segmentation
            doc = self.nlp(text)
            
            for sent in doc.sents:
                sent_entities = []
                
                # Find entities in this sentence
                for entity in entities:
                    if (entity.start_position >= sent.start_char and 
                        entity.end_position <= sent.end_char):
                        sent_entities.append(entity)
                
                # Create co-occurrence relationships for entities in the same sentence
                for i, entity1 in enumerate(sent_entities):
                    for entity2 in sent_entities[i+1:]:
                        if entity1.text != entity2.text:
                            relationship = EntityRelationship(
                                source_entity=entity1.text,
                                target_entity=entity2.text,
                                relationship_type="CO_OCCURS",
                                confidence=0.5,
                                context=sent.text.strip(),
                                evidence_text=sent.text.strip()
                            )
                            relationships.append(relationship)
        else:
            # Fallback: use simple sentence splitting
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
            for sentence in sentences:
                sent_entities = []
                
                # Find entities in this sentence
                for entity in entities:
                    if entity.text.lower() in sentence.lower():
                        sent_entities.append(entity)
                
                # Create co-occurrence relationships
                for i, entity1 in enumerate(sent_entities):
                    for entity2 in sent_entities[i+1:]:
                        if entity1.text != entity2.text:
                            relationship = EntityRelationship(
                                source_entity=entity1.text,
                                target_entity=entity2.text,
                                relationship_type="CO_OCCURS",
                                confidence=0.4,  # Lower confidence for fallback
                                context=sentence,
                                evidence_text=sentence
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def process_document(
        self, 
        document: Document,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_BASED,
        chunk_size: int = 500
    ) -> Tuple[List[TextChunk], List[ExtractedEntity], List[EntityRelationship]]:
        """
        Process a complete document: extract text, chunk it, and identify entities and relationships.
        
        Args:
            document: Document to process
            chunking_strategy: Strategy for text chunking
            chunk_size: Target size for chunks
            
        Returns:
            Tuple of (chunks, entities, relationships)
        """
        # Extract plain text
        plain_text = self.extract_text(document.content, document.document_type)
        
        # Create chunks
        chunks = self.chunk_text(plain_text, chunking_strategy, chunk_size)
        
        # Set document ID for all chunks
        for chunk in chunks:
            chunk.document_id = document.id
        
        # Extract entities from the full text
        print(f"DEBUG: Processing text of length {len(plain_text)}")
        print(f"DEBUG: Text preview: {plain_text[:200]}...")
        entities = self.extract_entities(plain_text)
        print(f"DEBUG: Extracted {len(entities)} entities")
        
        # Identify relationships
        relationships = self.identify_relationships(plain_text, entities)
        print(f"DEBUG: Identified {len(relationships)} relationships")
        
        return chunks, entities, relationships
    
    def create_entities_from_extracted(
        self, 
        extracted_entities: List[ExtractedEntity],
        document_id: str
    ) -> List[Entity]:
        """
        Convert extracted entities to Entity model instances.
        
        Args:
            extracted_entities: List of extracted entities
            document_id: ID of the source document
            
        Returns:
            List of Entity instances
        """
        entities = []
        seen_entities = set()  # To avoid duplicates
        
        for extracted in extracted_entities:
            # Create a normalized key to check for duplicates
            entity_key = (extracted.text.lower().strip(), extracted.label)
            
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                
                entity = Entity(
                    name=extracted.text.strip(),
                    type=EntityType(extracted.label),
                    description=f"Entity extracted from document {document_id}",
                    properties={
                        "extraction_confidence": extracted.confidence,
                        "source_document": document_id,
                        "context": extracted.context,
                        "spacy_properties": extracted.properties
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities, preferring domain-specific ones."""
        seen_entities = {}
        unique_entities = []
        
        # Sort entities to prioritize domain-specific ones
        sorted_entities = sorted(entities, key=lambda e: (
            e.properties.get("extraction_method") != "domain_specific",  # Domain-specific first
            -e.confidence,  # Higher confidence first
            e.text.lower()  # Alphabetical for consistency
        ))
        
        for entity in sorted_entities:
            # Use only text for deduplication key to handle same entity with different labels
            entity_key = entity.text.lower().strip()
            
            if entity_key not in seen_entities:
                seen_entities[entity_key] = entity
                unique_entities.append(entity)
            else:
                # Keep the one with higher confidence or domain-specific extraction
                existing = seen_entities[entity_key]
                should_replace = False
                
                # Prefer domain-specific extraction
                if (entity.properties.get("extraction_method") == "domain_specific" and 
                    existing.properties.get("extraction_method") != "domain_specific"):
                    should_replace = True
                # If both are domain-specific or both are not, prefer higher confidence
                elif (entity.properties.get("extraction_method") == existing.properties.get("extraction_method") and
                      entity.confidence > existing.confidence):
                    should_replace = True
                
                if should_replace:
                    # Replace with better entity
                    unique_entities = [e for e in unique_entities if e != existing]
                    unique_entities.append(entity)
                    seen_entities[entity_key] = entity
        
        return unique_entities
    
    def _deduplicate_relationships(self, relationships: List[EntityRelationship]) -> List[EntityRelationship]:
        """Remove duplicate relationships."""
        seen_relationships = set()
        unique_relationships = []
        
        for relationship in relationships:
            # Create a normalized key for the relationship
            rel_key = (
                relationship.source_entity.lower(),
                relationship.target_entity.lower(),
                relationship.relationship_type
            )
            
            if rel_key not in seen_relationships:
                seen_relationships.add(rel_key)
                unique_relationships.append(relationship)
        
        return unique_relationships
    
    def set_domain(self, domain_type: str):
        """Set the domain type for domain-specific processing."""
        self.domain_type = domain_type
        self._initialize_domain_manager()
        if self.domain_manager:
            from .domain_processor import DomainType
            domain_enum = DomainType(domain_type)
            self.domain_manager.set_domain(domain_enum)
    
    def get_current_domain(self) -> Optional[str]:
        """Get the current domain type."""
        self._initialize_domain_manager()
        if self.domain_manager and self.domain_manager.current_domain:
            return self.domain_manager.current_domain.value
        return None
    
    def list_available_domains(self) -> List[str]:
        """List all available domains."""
        self._initialize_domain_manager()
        if self.domain_manager:
            return [domain.value for domain in self.domain_manager.list_available_domains()]
        return []


class DocumentProcessingError(Exception):
    """Exception raised during document processing."""
    pass


class EntityExtractionError(DocumentProcessingError):
    """Exception raised during entity extraction."""
    pass


class RelationshipExtractionError(DocumentProcessingError):
    """Exception raised during relationship extraction."""
    pass