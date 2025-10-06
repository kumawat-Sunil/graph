"""
Domain configuration system for the Graph-Enhanced Agentic RAG system.

This module provides configuration management for domain-specific schemas,
allowing users to configure and switch between different knowledge domains.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import logging

from .domain_processor import (
    DomainType, 
    DomainSchema, 
    DomainProcessorManager,
    BaseDomainExtractor,
    get_domain_manager
)
from .models import EntityType

logger = logging.getLogger(__name__)


class DomainConfigurationManager:
    """Manager for domain configuration and schema persistence."""
    
    def __init__(self, config_dir: str = "config/domains"):
        """
        Initialize the domain configuration manager.
        
        Args:
            config_dir: Directory to store domain configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.domain_manager = get_domain_manager()
        self.active_domain: Optional[DomainType] = None
        
        # Load existing configurations
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all domain configurations from the config directory."""
        try:
            for config_file in self.config_dir.glob("*.json"):
                try:
                    schema = self._load_schema_from_file(config_file)
                    logger.info(f"Loaded domain schema: {schema.name} ({schema.domain_type})")
                except Exception as e:
                    logger.warning(f"Failed to load schema from {config_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to load domain configurations: {e}")
    
    def _load_schema_from_file(self, file_path: Path) -> DomainSchema:
        """Load a domain schema from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        
        return self._dict_to_schema(schema_data)
    
    def _dict_to_schema(self, schema_data: Dict[str, Any]) -> DomainSchema:
        """Convert dictionary data to DomainSchema object."""
        from .domain_processor import EntityPattern, RelationshipPattern
        
        # Convert entity patterns
        entity_patterns = []
        for pattern_data in schema_data.get('entity_patterns', []):
            entity_patterns.append(EntityPattern(
                name=pattern_data['name'],
                pattern=pattern_data['pattern'],
                entity_type=EntityType(pattern_data['entity_type']),
                confidence=pattern_data['confidence'],
                flags=pattern_data.get('flags', 0),
                description=pattern_data.get('description', ''),
                examples=pattern_data.get('examples', [])
            ))
        
        # Convert relationship patterns
        relationship_patterns = []
        for pattern_data in schema_data.get('relationship_patterns', []):
            relationship_patterns.append(RelationshipPattern(
                name=pattern_data['name'],
                pattern=pattern_data['pattern'],
                relationship_type=pattern_data['relationship_type'],
                confidence=pattern_data['confidence'],
                source_group=pattern_data.get('source_group', 1),
                target_group=pattern_data.get('target_group', 2),
                flags=pattern_data.get('flags', 0),
                description=pattern_data.get('description', ''),
                examples=pattern_data.get('examples', [])
            ))
        
        return DomainSchema(
            domain_type=DomainType(schema_data['domain_type']),
            name=schema_data['name'],
            description=schema_data['description'],
            entity_types=[EntityType(et) for et in schema_data.get('entity_types', [])],
            entity_patterns=entity_patterns,
            relationship_patterns=relationship_patterns,
            graph_constraints=schema_data.get('graph_constraints', []),
            metadata=schema_data.get('metadata', {})
        )
    
    def _schema_to_dict(self, schema: DomainSchema) -> Dict[str, Any]:
        """Convert DomainSchema object to dictionary."""
        return {
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
    
    def save_schema(self, schema: DomainSchema) -> str:
        """
        Save a domain schema to the configuration directory.
        
        Args:
            schema: Domain schema to save
            
        Returns:
            Path to the saved configuration file
        """
        filename = f"{schema.domain_type.value}.json"
        file_path = self.config_dir / filename
        
        schema_dict = self._schema_to_dict(schema)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved domain schema to {file_path}")
        return str(file_path)
    
    def load_schema(self, domain_type: DomainType) -> Optional[DomainSchema]:
        """
        Load a domain schema by type.
        
        Args:
            domain_type: Type of domain to load
            
        Returns:
            Domain schema if found, None otherwise
        """
        filename = f"{domain_type.value}.json"
        file_path = self.config_dir / filename
        
        if file_path.exists():
            return self._load_schema_from_file(file_path)
        
        return None
    
    def configure_domain(self, domain_type: DomainType) -> bool:
        """
        Configure the system for a specific domain.
        
        Args:
            domain_type: Domain type to configure
            
        Returns:
            True if configuration was successful, False otherwise
        """
        try:
            # Check if domain is already registered
            if domain_type in self.domain_manager.extractors:
                self.domain_manager.set_domain(domain_type)
                self.active_domain = domain_type
                logger.info(f"Configured domain: {domain_type}")
                return True
            
            # Try to load schema from configuration
            schema = self.load_schema(domain_type)
            if schema:
                # Create a generic extractor with the loaded schema
                from .domain_processor import BaseDomainExtractor
                import re
                
                class ConfigurableExtractor(BaseDomainExtractor):
                    def extract_entities(self, text: str):
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
                                    
                                    from .document_processor import ExtractedEntity
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
                    
                    def identify_relationships(self, text: str, entities):
                        relationships = []
                        
                        for pattern_name, pattern_info in self.relationship_patterns.items():
                            matches = re.finditer(pattern_info.pattern, text, pattern_info.flags)
                            
                            for match in matches:
                                source_text = match.group(pattern_info.source_group).strip()
                                target_text = match.group(pattern_info.target_group).strip()
                                
                                source_entity = self._find_matching_entity(source_text, entities)
                                target_entity = self._find_matching_entity(target_text, entities)
                                
                                if source_entity and target_entity and source_entity != target_entity:
                                    from .document_processor import EntityRelationship
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
                    
                    def _find_matching_entity(self, text: str, entities) -> Optional[str]:
                        text_lower = text.lower().strip()
                        
                        for entity in entities:
                            if entity.text.lower() == text_lower:
                                return entity.text
                        
                        for entity in entities:
                            entity_text_lower = entity.text.lower()
                            if text_lower in entity_text_lower or entity_text_lower in text_lower:
                                return entity.text
                        
                        return None
                
                extractor = ConfigurableExtractor(schema)
                self.domain_manager.register_extractor(domain_type, extractor)
                self.domain_manager.set_domain(domain_type)
                self.active_domain = domain_type
                
                logger.info(f"Configured domain from schema: {domain_type}")
                return True
            
            logger.warning(f"No configuration found for domain: {domain_type}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to configure domain {domain_type}: {e}")
            return False
    
    def list_configured_domains(self) -> List[DomainType]:
        """List all configured domains."""
        return self.domain_manager.list_available_domains()
    
    def get_active_domain(self) -> Optional[DomainType]:
        """Get the currently active domain."""
        return self.active_domain
    
    def create_custom_domain(
        self,
        domain_name: str,
        description: str,
        entity_patterns: List[Dict[str, Any]],
        relationship_patterns: List[Dict[str, Any]],
        graph_constraints: Optional[List[str]] = None
    ) -> DomainSchema:
        """
        Create a custom domain schema.
        
        Args:
            domain_name: Name for the custom domain
            description: Description of the domain
            entity_patterns: List of entity pattern definitions
            relationship_patterns: List of relationship pattern definitions
            graph_constraints: Optional list of Neo4j constraints
            
        Returns:
            Created domain schema
        """
        from .domain_processor import EntityPattern, RelationshipPattern
        
        # Convert pattern dictionaries to objects
        converted_entity_patterns = []
        for pattern_data in entity_patterns:
            converted_entity_patterns.append(EntityPattern(
                name=pattern_data['name'],
                pattern=pattern_data['pattern'],
                entity_type=EntityType(pattern_data['entity_type']),
                confidence=pattern_data.get('confidence', 0.8),
                flags=pattern_data.get('flags', 0),
                description=pattern_data.get('description', ''),
                examples=pattern_data.get('examples', [])
            ))
        
        converted_relationship_patterns = []
        for pattern_data in relationship_patterns:
            converted_relationship_patterns.append(RelationshipPattern(
                name=pattern_data['name'],
                pattern=pattern_data['pattern'],
                relationship_type=pattern_data['relationship_type'],
                confidence=pattern_data.get('confidence', 0.8),
                source_group=pattern_data.get('source_group', 1),
                target_group=pattern_data.get('target_group', 2),
                flags=pattern_data.get('flags', 0),
                description=pattern_data.get('description', ''),
                examples=pattern_data.get('examples', [])
            ))
        
        # Create custom domain type
        custom_domain_type = DomainType.GENERAL  # Use GENERAL as base for custom domains
        
        # Extract entity types from patterns
        entity_types = list(set(pattern.entity_type for pattern in converted_entity_patterns))
        
        schema = DomainSchema(
            domain_type=custom_domain_type,
            name=domain_name,
            description=description,
            entity_types=entity_types,
            entity_patterns=converted_entity_patterns,
            relationship_patterns=converted_relationship_patterns,
            graph_constraints=graph_constraints or [],
            metadata={
                "custom": True,
                "created_at": "2024-01-01",  # Would use actual timestamp
                "version": "1.0"
            }
        )
        
        return schema
    
    def export_domain_config(self, domain_type: DomainType, output_path: str):
        """
        Export a domain configuration to a file.
        
        Args:
            domain_type: Domain type to export
            output_path: Path to save the exported configuration
        """
        schema = self.domain_manager.get_domain_schema(domain_type)
        if not schema:
            raise ValueError(f"No schema found for domain {domain_type}")
        
        schema_dict = self._schema_to_dict(schema)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported domain configuration to {output_path}")
    
    def import_domain_config(self, config_path: str) -> DomainType:
        """
        Import a domain configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Domain type of the imported configuration
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        schema = self._load_schema_from_file(config_file)
        
        # Save to local configuration directory
        self.save_schema(schema)
        
        # Configure the domain
        self.configure_domain(schema.domain_type)
        
        logger.info(f"Imported and configured domain: {schema.domain_type}")
        return schema.domain_type


# Global configuration manager instance
_config_manager = None


def get_domain_config_manager() -> DomainConfigurationManager:
    """Get the global domain configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = DomainConfigurationManager()
    return _config_manager


def configure_system_domain(domain_type: DomainType) -> bool:
    """Configure the system for a specific domain."""
    manager = get_domain_config_manager()
    return manager.configure_domain(domain_type)


def get_active_domain() -> Optional[DomainType]:
    """Get the currently active domain."""
    manager = get_domain_config_manager()
    return manager.get_active_domain()


def list_available_domains() -> List[DomainType]:
    """List all available domains."""
    manager = get_domain_config_manager()
    return manager.list_configured_domains()