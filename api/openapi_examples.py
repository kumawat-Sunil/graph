"""
OpenAPI Examples and Schema Enhancements for Graph-Enhanced Agentic RAG API

This module provides comprehensive examples and schema definitions for
enhanced API documentation.
"""

from typing import Dict, Any

# Query Examples
QUERY_EXAMPLES = {
    "simple_factual": {
        "summary": "Simple factual query",
        "description": "Basic question about a concept that can be answered with vector search",
        "value": {
            "query": "What is machine learning?",
            "max_results": 10,
            "include_reasoning": True
        }
    },
    "relationship_query": {
        "summary": "Relationship exploration query",
        "description": "Question about relationships between entities that benefits from graph traversal",
        "value": {
            "query": "How are neural networks related to deep learning and artificial intelligence?",
            "max_results": 15,
            "strategy": "graph_focused",
            "include_reasoning": True
        }
    },
    "complex_multi_hop": {
        "summary": "Complex multi-hop query",
        "description": "Complex question requiring multiple reasoning steps and hybrid search",
        "value": {
            "query": "What are the applications of reinforcement learning in robotics and how do they relate to computer vision techniques?",
            "max_results": 20,
            "strategy": "hybrid",
            "include_reasoning": True
        }
    },
    "domain_specific": {
        "summary": "Domain-specific query",
        "description": "Query targeting specific knowledge domain",
        "value": {
            "query": "Explain the transformer architecture in natural language processing",
            "max_results": 12,
            "strategy": "vector_focused"
        }
    }
}

# Query Response Examples
QUERY_RESPONSE_EXAMPLES = {
    "successful_response": {
        "summary": "Successful query response",
        "description": "Complete response with sources, citations, and reasoning",
        "value": {
            "query_id": "123e4567-e89b-12d3-a456-426614174000",
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns. Key types include supervised learning (learning from labeled examples), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment).",
            "sources": [
                {
                    "id": "doc_ml_intro_001",
                    "title": "Introduction to Machine Learning",
                    "content_preview": "Machine learning algorithms build mathematical models based on training data...",
                    "source_type": "document",
                    "relevance_score": 0.92,
                    "domain": "technical"
                },
                {
                    "id": "entity_ml_concept",
                    "name": "Machine Learning",
                    "type": "concept",
                    "source_type": "graph_entity",
                    "relevance_score": 0.89,
                    "relationships": ["part_of:Artificial Intelligence", "includes:Supervised Learning"]
                }
            ],
            "citations": [
                {
                    "id": "1",
                    "source": "Introduction to Machine Learning",
                    "page": 15,
                    "relevance": 0.92,
                    "citation_text": "[1] Introduction to Machine Learning, p. 15"
                },
                {
                    "id": "2",
                    "source": "Knowledge Graph Entity: Machine Learning",
                    "relevance": 0.89,
                    "citation_text": "[2] Knowledge Graph: Machine Learning concept"
                }
            ],
            "reasoning_path": "Query Analysis: Identified 'machine learning' as primary concept → Strategy Selection: Chose vector search for factual query → Vector Search: Found 8 relevant documents → Graph Navigation: Explored ML concept relationships → Synthesis: Combined vector results with graph context → Response Generation: Created comprehensive answer with citations",
            "confidence_score": 0.89,
            "processing_time": 1.23,
            "strategy_used": "vector_focused",
            "entities_found": ["machine learning", "artificial intelligence", "supervised learning", "unsupervised learning"]
        }
    },
    "hybrid_response": {
        "summary": "Hybrid strategy response",
        "description": "Response using both graph and vector search",
        "value": {
            "query_id": "456e7890-f12b-34c5-d678-901234567890",
            "response": "Neural networks and deep learning are closely related concepts within artificial intelligence. Neural networks are computational models inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information. Deep learning is a subset of machine learning that uses neural networks with multiple hidden layers (hence 'deep') to model complex patterns in data. The relationship extends to computer vision, where convolutional neural networks (CNNs) have revolutionized image recognition tasks.",
            "sources": [
                {
                    "id": "doc_nn_guide_002",
                    "title": "Neural Networks Fundamentals",
                    "source_type": "document",
                    "relevance_score": 0.94
                },
                {
                    "id": "entity_neural_network",
                    "name": "Neural Network",
                    "source_type": "graph_entity",
                    "relevance_score": 0.91
                }
            ],
            "citations": [
                {
                    "id": "1",
                    "source": "Neural Networks Fundamentals",
                    "citation_text": "[1] Neural Networks Fundamentals, Chapter 3"
                }
            ],
            "reasoning_path": "Query Analysis: Identified multiple related concepts → Strategy Selection: Chose hybrid approach → Graph Navigation: Explored relationships between neural networks, deep learning, and AI → Vector Search: Retrieved relevant documents → Synthesis: Integrated graph relationships with document content",
            "confidence_score": 0.91,
            "processing_time": 2.15,
            "strategy_used": "hybrid",
            "entities_found": ["neural networks", "deep learning", "artificial intelligence", "computer vision"]
        }
    }
}

# Document Upload Examples
DOCUMENT_UPLOAD_EXAMPLES = {
    "technical_document": {
        "summary": "Technical documentation upload",
        "description": "Upload technical documentation with metadata",
        "value": {
            "title": "Transformer Architecture in NLP",
            "content": "The Transformer architecture, introduced in 'Attention Is All You Need', revolutionized natural language processing. It relies entirely on attention mechanisms, dispensing with recurrence and convolutions. The architecture consists of an encoder and decoder, each composed of multiple identical layers. Key components include multi-head self-attention, position-wise feed-forward networks, and positional encoding.",
            "source": "https://arxiv.org/abs/1706.03762",
            "domain": "technical",
            "metadata": {
                "author": "Vaswani et al.",
                "publication_year": 2017,
                "paper_type": "research",
                "keywords": ["transformer", "attention", "NLP", "neural networks"],
                "venue": "NIPS 2017"
            }
        }
    },
    "business_document": {
        "summary": "Business documentation upload",
        "description": "Upload business-focused content",
        "value": {
            "title": "AI Implementation Strategy",
            "content": "Implementing artificial intelligence in enterprise environments requires careful planning and consideration of multiple factors. Key considerations include data quality and availability, infrastructure requirements, talent acquisition, change management, and ROI measurement. Organizations should start with pilot projects to demonstrate value before scaling AI initiatives across the enterprise.",
            "domain": "business",
            "metadata": {
                "document_type": "strategy_guide",
                "target_audience": "executives",
                "industry": "technology"
            }
        }
    },
    "research_document": {
        "summary": "Research paper upload",
        "description": "Upload academic research content",
        "value": {
            "title": "Advances in Reinforcement Learning for Robotics",
            "content": "Abstract: This paper presents recent advances in applying reinforcement learning techniques to robotic control tasks. We explore deep reinforcement learning methods for continuous control, sim-to-real transfer, and multi-agent coordination in robotic systems. Our experiments demonstrate significant improvements in sample efficiency and task performance across various robotic manipulation and navigation scenarios.",
            "source": "https://example.com/rl-robotics-paper",
            "domain": "research",
            "metadata": {
                "paper_type": "conference",
                "venue": "ICRA 2024",
                "authors": ["Smith, J.", "Johnson, A.", "Williams, B."],
                "keywords": ["reinforcement learning", "robotics", "deep learning", "control"],
                "abstract": "Recent advances in RL for robotics..."
            }
        }
    }
}

# Document Upload Response Examples
DOCUMENT_UPLOAD_RESPONSE_EXAMPLES = {
    "successful_upload": {
        "summary": "Successful document upload",
        "description": "Document processed successfully with entity extraction",
        "value": {
            "document_id": "doc_789abc12-def3-4567-8901-234567890abc",
            "status": "success",
            "message": "Document 'Transformer Architecture in NLP' uploaded and processed successfully",
            "entities_extracted": 25,
            "relationships_created": 18,
            "processing_time": 3.45,
            "processing_details": {
                "text_chunks_created": 12,
                "embeddings_generated": 12,
                "graph_nodes_created": 25,
                "graph_relationships_created": 18,
                "domain_entities_identified": ["transformer", "attention mechanism", "encoder", "decoder", "NLP"]
            }
        }
    }
}

# Error Response Examples
ERROR_RESPONSE_EXAMPLES = {
    "validation_error": {
        "summary": "Validation error example",
        "description": "Request validation failed",
        "value": {
            "error": "Validation Error",
            "message": "Request validation failed. Please check your input data.",
            "details": {
                "validation_errors": [
                    {
                        "loc": ["body", "query"],
                        "msg": "field required",
                        "type": "value_error.missing"
                    }
                ],
                "suggestions": [
                    "Field 'query' is required but was not provided"
                ]
            },
            "timestamp": "2024-01-01T12:00:00Z",
            "request_id": "req_123456789"
        }
    },
    "not_found_error": {
        "summary": "Not found error example",
        "description": "Endpoint not found",
        "value": {
            "error": "Not Found",
            "message": "The requested endpoint '/invalid-endpoint' was not found",
            "details": {
                "method": "GET",
                "path": "/invalid-endpoint",
                "available_endpoints": [
                    "/docs - API documentation",
                    "/health - Health check",
                    "/query - Process queries",
                    "/documents/upload - Upload documents"
                ],
                "documentation": "/docs"
            },
            "timestamp": "2024-01-01T12:00:00Z",
            "request_id": "req_987654321"
        }
    },
    "rate_limit_error": {
        "summary": "Rate limit error example",
        "description": "Rate limit exceeded",
        "value": {
            "error": "Rate Limit Exceeded",
            "message": "Too many requests. Please slow down.",
            "details": {
                "rate_limits": {
                    "query_endpoint": "100 requests per minute",
                    "document_upload": "50 requests per minute"
                },
                "retry_after": "60 seconds",
                "suggestions": [
                    "Wait before making additional requests",
                    "Implement exponential backoff in your client"
                ]
            },
            "timestamp": "2024-01-01T12:00:00Z",
            "request_id": "req_555666777"
        }
    },
    "internal_server_error": {
        "summary": "Internal server error example",
        "description": "Unexpected server error",
        "value": {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred while processing your request",
            "details": {
                "request_id": "req_111222333",
                "support_message": "Please contact support with this request ID if the problem persists",
                "troubleshooting": [
                    "Check system status at /health",
                    "Verify your request format matches the API documentation",
                    "Try again in a few moments"
                ]
            },
            "timestamp": "2024-01-01T12:00:00Z",
            "request_id": "req_111222333"
        }
    }
}

# Health Response Examples
HEALTH_RESPONSE_EXAMPLES = {
    "healthy_system": {
        "summary": "Healthy system status",
        "description": "All components are functioning normally",
        "value": {
            "status": "healthy",
            "version": "1.0.0",
            "environment": "production",
            "timestamp": "2024-01-01T12:00:00Z",
            "components": {
                "api": "healthy",
                "neo4j": "healthy",
                "chroma": "healthy",
                "coordinator_agent": "healthy",
                "graph_navigator": "healthy",
                "vector_retrieval": "healthy",
                "synthesis_agent": "healthy"
            },
            "uptime": 86400.5
        }
    },
    "degraded_system": {
        "summary": "Degraded system status",
        "description": "Some components are experiencing issues",
        "value": {
            "status": "degraded",
            "version": "1.0.0",
            "environment": "production",
            "timestamp": "2024-01-01T12:00:00Z",
            "components": {
                "api": "healthy",
                "neo4j": "degraded",
                "chroma": "healthy",
                "coordinator_agent": "healthy",
                "graph_navigator": "degraded",
                "vector_retrieval": "healthy",
                "synthesis_agent": "healthy"
            },
            "uptime": 86400.5,
            "issues": [
                "Neo4j connection experiencing intermittent timeouts",
                "Graph Navigator Agent performance degraded due to database issues"
            ]
        }
    }
}

# Agent Status Examples
AGENT_STATUS_EXAMPLES = {
    "all_agents_healthy": {
        "summary": "All agents healthy",
        "description": "All agents are functioning normally",
        "value": {
            "agents": {
                "coordinator": {
                    "status": "healthy",
                    "last_activity": "2024-01-01T11:59:30Z",
                    "queries_processed": 1247,
                    "average_response_time": 0.15,
                    "error_count": 2,
                    "description": "Orchestrates query processing and agent coordination"
                },
                "graph_navigator": {
                    "status": "healthy",
                    "last_activity": "2024-01-01T11:59:45Z",
                    "queries_processed": 892,
                    "average_response_time": 0.45,
                    "error_count": 1,
                    "description": "Handles graph traversal and Cypher query execution"
                },
                "vector_retrieval": {
                    "status": "healthy",
                    "last_activity": "2024-01-01T11:59:50Z",
                    "queries_processed": 1156,
                    "average_response_time": 0.32,
                    "error_count": 0,
                    "description": "Performs semantic similarity search in vector database"
                },
                "synthesis": {
                    "status": "healthy",
                    "last_activity": "2024-01-01T11:59:55Z",
                    "responses_generated": 1247,
                    "average_response_time": 0.78,
                    "error_count": 3,
                    "description": "Synthesizes results and generates final responses"
                }
            },
            "total_agents": 4,
            "healthy_agents": 4,
            "last_updated": "2024-01-01T12:00:00Z"
        }
    }
}

def get_openapi_examples() -> Dict[str, Any]:
    """Get all OpenAPI examples organized by endpoint."""
    return {
        "query": {
            "request_examples": QUERY_EXAMPLES,
            "response_examples": QUERY_RESPONSE_EXAMPLES
        },
        "document_upload": {
            "request_examples": DOCUMENT_UPLOAD_EXAMPLES,
            "response_examples": DOCUMENT_UPLOAD_RESPONSE_EXAMPLES
        },
        "health": {
            "response_examples": HEALTH_RESPONSE_EXAMPLES
        },
        "agents": {
            "response_examples": AGENT_STATUS_EXAMPLES
        },
        "errors": {
            "response_examples": ERROR_RESPONSE_EXAMPLES
        }
    }