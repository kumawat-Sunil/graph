# 🧠 Graph-Enhanced Agentic RAG System

A sophisticated **Retrieval-Augmented Generation (RAG)** system that combines **multi-agent architecture** with **graph databases** and **vector embeddings** for intelligent document processing and query answering.

## 🌟 Features

### 🤖 Multi-Agent Architecture
- **Coordinator Agent**: Orchestrates the entire workflow and decision-making
- **Graph Navigator Agent**: Explores entity relationships in Neo4j graph database
- **Vector Retrieval Agent**: Performs semantic similarity search using embeddings
- **Synthesis Agent**: Combines results from multiple sources using Gemini AI

### 📊 Dual Storage System
- **Neo4j Graph Database**: Stores entities, relationships, and knowledge graphs
- **Pinecone Vector Database**: Stores document embeddings for semantic search
- **Bidirectional Mapping**: Links graph entities with vector embeddings

### ⚡ Smart Query Processing
- **Intelligent Strategy Selection**: Automatically chooses optimal retrieval approach
- **Hybrid Search**: Combines graph traversal with vector similarity
- **Entity Recognition**: Extracts and processes entities from queries and documents
- **Context-Aware Responses**: Generates comprehensive answers with citations

### 🚀 Performance Optimizations
- **Lazy Initialization**: Components load on-demand for fast startup
- **Connection Pooling**: Efficient database connection management
- **Caching**: Optimized for repeated queries and operations
- **Error Resilience**: Automatic retry logic and graceful degradation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   FastAPI App   │    │  Multi-Agents   │
│                 │◄──►│                 │◄──►│                 │
│  • Query Input  │    │  • REST APIs    │    │ • Coordinator   │
│  • File Upload  │    │  • Health Check │    │ • Graph Nav     │
│  • Admin Panel  │    │  • Error Handle │    │ • Vector Search │
└─────────────────┘    └─────────────────┘    │ • Synthesis     │
                                              └─────────────────┘
                                                       │
                              ┌────────────────────────┼────────────────────────┐
                              │                        │                        │
                    ┌─────────▼─────────┐    ┌────────▼────────┐    ┌─────────▼─────────┐
                    │   Neo4j Graph DB  │    │ Pinecone Vector │    │   Gemini AI API   │
                    │                   │    │                 │    │                   │
                    │ • Entities        │    │ • Embeddings    │    │ • Text Generation │
                    │ • Relationships   │    │ • Similarity    │    │ • Response Synth  │
                    │ • Knowledge Graph │    │ • Semantic      │    │ • Quality Control │
                    └───────────────────┘    └─────────────────┘    └───────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance web framework
- **Python 3.11+**: Core programming language
- **Neo4j**: Graph database for entity relationships
- **Pinecone**: Vector database for embeddings
- **Sentence Transformers**: Text embedding generation
- **Google Gemini**: AI text generation and synthesis

### Frontend
- **HTML5/CSS3/JavaScript**: Modern web interface
- **Responsive Design**: Works on desktop and mobile
- **Real-time Status**: Live system health monitoring
- **File Upload**: Drag-and-drop document processing

### Infrastructure
- **Docker Ready**: Containerized deployment
- **Cloud Compatible**: Works with any hosting platform
- **Environment Variables**: Secure configuration management
- **Health Monitoring**: Comprehensive system diagnostics

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Neo4j database (local or cloud)
- Pinecone account and API key
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sunilk240/graph-enhanced-rag.git
   cd graph-enhanced-rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URLs
   ```

4. **Configure databases**
   ```bash
   # Neo4j (local installation or cloud)
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   
   # Pinecone
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=rag-documents
   
   # Gemini AI
   GEMINI_API_KEY=your_gemini_api_key
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

6. **Access the interface**
   - Web Interface: `http://localhost:8000/interface`
   - Admin Panel: `http://localhost:8000/admin`
   - API Documentation: `http://localhost:8000/docs`

## 📖 Usage

### Document Upload
1. Navigate to the web interface
2. Upload documents (PDF, TXT, MD supported)
3. System automatically:
   - Extracts entities and relationships
   - Generates embeddings
   - Stores in both graph and vector databases
   - Creates bidirectional mappings

### Query Processing
1. Enter your question in the query interface
2. System intelligently:
   - Analyzes query complexity and type
   - Selects optimal retrieval strategy
   - Searches both graph and vector databases
   - Synthesizes comprehensive response with citations

### Admin Operations
- Monitor system health and performance
- View database statistics
- Manage documents and entities
- Configure system parameters

## 🔧 API Endpoints

### Core Endpoints
- `POST /query` - Process user queries
- `POST /documents/upload` - Upload documents
- `POST /documents/upload-file` - Upload files
- `GET /health` - System health check
- `GET /system/status` - Detailed system status

### Admin Endpoints
- `GET /admin/stats` - Database statistics
- `GET /admin/database/neo4j` - Neo4j details
- `GET /admin/database/pinecone` - Pinecone details
- `GET /agents/status` - Agent system status

### Utility Endpoints
- `GET /ping` - Quick health check
- `GET /ready` - Readiness probe
- `GET /keepalive` - Prevent server sleep
- `GET /warmup` - System warmup

## 🏗️ Deployment

### Local Development
```bash
python main.py
```

### Production Deployment
The system is optimized for cloud deployment with:
- **Lazy Initialization**: Fast startup times
- **Health Checks**: Built-in monitoring endpoints
- **Error Resilience**: Automatic retry and recovery
- **Resource Efficiency**: On-demand component loading

### Supported Platforms
- Railway
- Fly.io
- Vercel
- Heroku
- PythonAnywhere
- Any Docker-compatible platform

## 📊 Performance

### Startup Performance
- **Server Start**: ~2-3 seconds
- **First Query**: ~8-10 seconds (includes agent initialization)
- **First Upload**: ~5-8 seconds (includes database initialization)
- **Subsequent Operations**: ~1-2 seconds (cached components)

### Scalability
- **Concurrent Users**: Supports multiple simultaneous queries
- **Document Processing**: Batch processing for large uploads
- **Database Optimization**: Indexed queries and connection pooling
- **Memory Management**: Efficient resource utilization

## 🔒 Security

- **Environment Variables**: Secure API key management
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Secure error responses
- **CORS Configuration**: Configurable cross-origin policies
- **Rate Limiting**: Built-in request throttling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Neo4j** for graph database technology
- **Pinecone** for vector database services
- **Google** for Gemini AI API
- **Hugging Face** for Sentence Transformers
- **FastAPI** for the excellent web framework

---

**Built with ❤️ for intelligent document processing and knowledge retrieval**
