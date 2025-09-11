# 🌳 RAPTOR RAG Service - Deep Research Flow & Smart Citations

> **Production-ready RAG system** with **RAPTOR hierarchical trees**, **LangFlow MCP integration**, and **intelligent deep research capabilities** for complex document analysis and multi-agent workflows.

---

## 🚀 Overview

**RAPTOR RAG Service** provides intelligent document processing with deep research capabilities:

- 📚 **RAGFlow Processing**: Advanced document chunking with parallel upload support
- 🌳 **RAPTOR Trees**: Hierarchical clustering with GMM+BIC for optimal retrieval  
- 🔗 **MCP Integration**: Model Context Protocol server for LangFlow and external AI tools
- 🧠 **Deep Research Flow**: Multi-agent architecture for complex research queries
- 🤖 **Dual LLM Setup**: DeepSeek-R1 (primary) + Gemini (fallback) for reliable responses
- 💡 **Smart Citations**: Model-driven citation extraction with thinking analysis
- 🏢 **Multi-tenant**: Isolated data per tenant/knowledge base

### ⭐ Key Features

- ✅ **LangFlow MCP Server** - Official MCP protocol with SSE transport for external orchestration
- ✅ **Deep Research Architecture** - Planning Agent → Research Executor → Writer Agent workflow
- ✅ **Parallel MCP Calls** - True parallel retrieval for comprehensive research coverage
- ✅ **DeepSeek-R1 Integration** - Primary LLM with thinking extraction for citations
- ✅ **Intelligent Citations** - Model decides what to cite based on internal reasoning  
- ✅ **VoyageAI Multi-Key** - Production-grade embedding with parallel processing
- ✅ **RAPTOR Tree Building** - Hierarchical clustering for enhanced context retrieval
- ✅ **Configurable Chunking** - Adjustable chunk sizes and overlap settings

### 🧠 **Deep Research Intelligence**

**Multi-Agent Research System:**
- 🎯 **Planning Agent** - Breaks complex queries into 3 focused research areas
- 🔍 **Research Executor** - Performs parallel MCP calls with quality self-assessment
- ✍️ **Writer Agent** - Synthesizes findings into comprehensive reports
- ⚡ **True Parallelism** - Simultaneous retrieval calls for comprehensive coverage
- 🧹 **Quality Control** - Built-in completeness assessment and gap detection

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph "LangFlow Integration"
        A[Chat Input] --> B[Planning Agent]
        B --> C[Research Executor]
        C --> D[Writer Agent]
        D --> E[Chat Output]
    end
    
    subgraph "MCP Server Layer"
        F[Official MCP Server] --> G[SSE Transport]
        G --> H[JSON-RPC Protocol]
        H --> I[RAPTOR Retrieve Tool]
    end
    
    subgraph "Deep Research Core"
        J[Query Planning] --> K[3 Parallel MCP Calls]
        K --> L[Quality Assessment]
        L --> M[Research Synthesis]
    end
    
    subgraph "LLM Strategy" 
        N[DeepSeek-R1 Primary] --> O{Success?}
        O -->|Yes| P[Extract Thinking]
        O -->|No| Q[Gemini Fallback]
        Q --> P
    end
    
    subgraph "Storage"
        R[(Supabase DB)]
        S[(Vector Index)]
        T[(RAPTOR Trees)]
    end
    
    C --> F
    I --> J
    M --> N
    P --> D
    
    style A fill:#e1f5fe
    style C fill:#fff3e0
    style F fill:#f3e5f5
    style N fill:#e8f5e8
```

### Tech Stack

- **Backend**: FastAPI + Uvicorn (async)
- **MCP Server**: Official MCP library with SSE transport
- **Database**: Supabase (PostgreSQL + pgvector) 
- **Embeddings**: VoyageAI (multi-key) or BGE-M3 (local)
- **Primary LLM**: DeepSeek-R1 via FPT Cloud API 
- **Fallback LLM**: Google Gemini 1.5 Flash
- **Clustering**: Gaussian Mixture Models + BIC optimization
- **Vector Search**: FAISS with similarity ranking
- **External Integration**: LangFlow deep research workflows

---

## 📖 API Endpoints

### 💬 Chat & Retrieval

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/assistant` | POST | Chat with AI assistant (DeepSeek-R1 → Gemini fallback) |
| `/v1/ragflow/ragflow_retrieve` | POST | RAPTOR retrieval with enhanced scoring |

### 🤖 AI Assistant Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/ai/assistants` | GET | List all AI assistants |
| `/v1/ai/assistants` | POST | Create new AI assistant |
| `/v1/ai/assistants/{id}` | DELETE | Delete AI assistant (cascade) |

### 📚 Document Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/ragflow/process` | POST | Upload & process documents with RAPTOR trees |
| `/v1/kb/create` | POST | Create knowledge base |
| `/v1/kb/list` | GET | List knowledge bases |

### 🔗 MCP Integration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/mcp/sse` | GET | SSE transport for LangFlow MCP connection |
| `/api/v1/mcp/rpc` | POST | JSON-RPC endpoint for MCP tool calls |
| `/health` | GET | MCP server health check |

---

## ⚡ Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **uv** (Python package manager)
- **FPT Cloud API Key** (for DeepSeek-R1)
- **Google Gemini API Key** (for fallback)
- **VoyageAI API Key** (recommended) or **Ollama** (for local embeddings)
- **Supabase** account (for database)
- **LangFlow** (optional, for deep research workflows)

### 2. Backend Setup

```bash
git clone <your-repo>
cd raptor_service

# Install dependencies with uv
uv pip install -r requirements.txt

# Setup environment
cp env.template .env
# Edit .env with your API keys (see Configuration section)

# Setup database
python setup_database.py

# Start backend
uvicorn main:app --reload --host 0.0.0.0 --port 8081
```

### 3. MCP Server Setup

```bash
# Start MCP server for LangFlow integration
python run_mcp_server.py --mode http --host 127.0.0.1 --port 3333
```

### 4. Access Applications

- **Backend API**: http://127.0.0.1:8081
- **API Documentation**: http://127.0.0.1:8081/docs
- **MCP Server**: http://127.0.0.1:3333

---

## 🔧 Configuration

### Embedding Options

Choose between VoyageAI (production) or BGE-M3 (local):

**VoyageAI:** Multi-key support, zero rate limits
**BGE-M3:** Cost-effective, local processing

Setup details in Complete Environment Variables section below.

### Complete Environment Variables

```env
# === DUAL LLM CONFIGURATION ===
# Primary LLM: DeepSeek-R1 (via FPT Cloud)
LLM_BASE_URL=https://mkp-api.fptcloud.com/v1
LLM_API_KEY=your_fpt_cloud_api_key
PRIMARY_CHAT_MODEL=DeepSeek-R1
PRIMARY_CHAT_TEMPERATURE=0.1
PRIMARY_CHAT_MAX_TOKENS=2048

# Fallback LLM: Google Gemini  
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_OUTPUT_TOKENS=2048

# === EMBEDDING CONFIGURATION ===
# VoyageAI (RECOMMENDED - multi-key support)
EMBED_BASE_URL=https://api.voyageai.com/v1
EMBED_API_KEY=pa-key1,pa-key2,pa-key3,pa-key4
EMBED_MODEL=voyage-context-3
EMBED_VECTOR_DIM=1024

# === DATABASE CONFIGURATION ===
DATABASE_URL=postgresql+psycopg://postgres.PROJECT_ID:PASSWORD@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require
DB_ENABLE_SSL=true
SUPABASE_SSLROOTCERT=./database/prod-ca-2021.crt

# === RAPTOR & CHUNKING CONFIGURATION ===
# Raptor tree building
RAPTOR_MAX_CLUSTERS=64
RAPTOR_THRESHOLD=0.1
RAPTOR_RANDOM_SEED=42

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP_PERCENT=10

# API settings
API_HOST=0.0.0.0
API_PORT=8081
API_RELOAD=true
```

---

## 🔗 LangFlow Integration

**MCP Server Setup:**
- URL: `http://127.0.0.1:3333/api/v1/mcp/sse`
- Tool: `raptor_retrieve` with parameters:
  - tenant_id: "test_tenant"
  - kb_id: "test_tenant::kb::liên_minh_huyền_thoại"
  - top_k: 8

**Deep Research Flow:**
```
Chat Input → Planning Agent → Research Executor → Writer Agent → Chat Output
```

Recommended models: Groq Llama 3.3 70B (planning/writing), Llama 4 Scout (research)

---

## 🧪 Usage Examples

### Deep Research Query (via LangFlow)

```
Input: "Analyze T1's World Championship performance focusing on:
1) Their championship victories and years they won titles
2) Key players and MVP performances across their campaigns  
3) Their rivalry matchups and notable defeats in finals"

Expected Flow:
- Planning Agent: Breaks into 3 research areas
- Research Executor: 3 parallel MCP calls + quality check
- Writer Agent: Comprehensive synthesis with all findings
```

### Direct Chat with Smart Citations

```bash
curl -X POST "http://127.0.0.1:8081/v1/chat/assistant" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What team won the 2019 League of Legends World Championship?",
    "session_id": "demo::session::12345"
  }'
```

**Response with Smart Citations:**
```json
{
  "answer": "FunPlus Phoenix (FPX) won the 2019 League of Legends World Championship, defeating G2 Esports 3-0 in the finals.",
  "context_passages": [
    {
      "source_id": "doc1_chunk_25",
      "relevant_excerpt": "2019 saw FunPlus Phoenix secure another Chinese victory over G2 Esports (3-0) with Tian earning MVP honors.",
      "similarity_score": 0.89
    }
  ]
}
```

### MCP Tool Call (Direct)

```bash
curl -X POST "http://127.0.0.1:3333/api/v1/mcp/rpc" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tools/call",
    "params": {
      "name": "raptor_retrieve",
      "arguments": {
        "query": "T1 championship victories",
        "tenant_id": "test_tenant",
        "kb_id": "test_tenant::kb::liên_minh_huyền_thoại",
        "top_k": 8
      }
    }
  }'
```

### Upload Documents with RAPTOR Processing

```bash
curl -X POST "http://127.0.0.1:8081/v1/ragflow/process" \
  -F "file=@technical_guide.md" \
  -F "tenant_id=demo" \
  -F "kb_id=my_documents" \
  -F "enable_raptor=true" \
  -F "max_clusters=64"
```

---

## 📊 Configuration & Features

### 🔧 **Document Processing**

**RAPTOR Tree Features:**
- Hierarchical clustering with GMM+BIC optimization
- Multi-level document summarization  
- Vector-based similarity search
- Configurable clustering parameters

### 🏆 MCP Integration Features

**LangFlow Deep Research:**
```bash
✅ Parallel Execution: 3 simultaneous MCP calls
✅ Quality Self-Assessment: Built-in completeness scoring  
✅ Token Management: Compatible with Groq rate limits
✅ Research Workflow: Comprehensive multi-agent flow
```

**VoyageAI Multi-Key Features:**
```bash
✅ RAPTOR Tree Building: Hierarchical clustering support
✅ Parallel Processing: Multiple keys for embedding generation
✅ Rate Limit Management: Load balancing across keys
✅ Production-Ready: Multi-key embedding support
```

### Database Features
- **Multi-tenant Isolation**: Data separated by `tenant_id`
- **Vector Search**: pgvector with HNSW indexing
- **Cascade Deletion**: Automatic cleanup when deleting assistants
- **Bulk Operations**: Embedding storage operations

---

## 🔧 Troubleshooting

### Common LangFlow Issues

**❌ "Timeout waiting for SSE session"**
```bash
# Verify both servers running (see Quick Start section)
# Check LangFlow MCP URL: http://127.0.0.1:3333/api/v1/mcp/sse
```

**❌ "Tool parameter validation failed"**
```bash
# Ensure correct parameter types:
top_k: 8 (integer, not "8" string)
tenant_id: "test_tenant" (string)
kb_id: "test_tenant::kb::liên_minh_huyền_thoại" (string)
```

**❌ Incomplete Research Results**
```bash
# Check document processing status
# Verify knowledge base contains relevant documents
# Consider adjusting retrieval parameters (top_k, etc.)
```

### Common API Issues

**❌ DeepSeek-R1 API Issues**
```bash
# Check FPT Cloud API key and credits
# Verify model name: "DeepSeek-R1" (exact case)
# System will fallback to Gemini automatically
```

**❌ Database Connection Failed**
```bash
# Test connection
python setup_database.py

# Check Supabase project status and credentials
```

### Debug Mode

Enable detailed logging:
```env
LOG_LEVEL=INFO
```

Monitor for:
- `🚀 Starting Official RAPTOR MCP server`
- `🤖 Trying DeepSeek-R1 (Primary LLM)`
- `🔍 3 parallel MCP calls executed`
- `📊 Citation decision: True/False`

---

## 🚀 Production Deployment

### MCP Server Production Setup
```bash
# Use process manager for MCP server
pm2 start run_mcp_server.py --name "raptor-mcp" -- --mode http --port 3333

# Configure reverse proxy for MCP endpoints
# Nginx: proxy_pass http://127.0.0.1:3333/api/v1/mcp/;
```

### Security Checklist
- [ ] Use environment variables for all API keys
- [ ] Enable HTTPS for all connections
- [ ] Configure CORS for production domains
- [ ] Set up proper database user permissions
- [ ] Regular backup of database and files
- [ ] Rotate API keys regularly (FPT Cloud, Gemini, VoyageAI)

### Production Recommendations
- **🏆 Embedding**: Use VoyageAI multi-key for production
- **📊 Monitoring**: Track API usage and costs
- **🔄 Infrastructure**: Load balancing and persistent storage
- **📈 Scaling**: Add more embedding keys for throughput

### LangFlow Production Integration
- **🔗 Stable MCP Connection**: Dedicated MCP server instance
- **🧠 Model Selection**: Use appropriate Groq models for token limits
- **⚙️ Result Caching**: Cache planning results for similar queries
- **📊 Quality Monitoring**: Track research completeness scores

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/deep-research-improvements`
3. Follow code style: `black . && isort .`
4. Test MCP integration and deep research flows
5. Commit changes: `git commit -m 'Enhance deep research capabilities'`
6. Push to branch: `git push origin feature/deep-research-improvements`
7. Open a Pull Request

### Development Setup

```bash
# Backend development (using uv)
uv pip install -r requirements.txt
python setup_database.py

# Development setup (see Quick Start for commands)
pytest tests/ -v  # Run tests
```

### MCP Development
- Test MCP tool calls with curl commands
- Verify SSE transport compatibility with LangFlow
- Monitor parallel call functionality
- Validate research flow completeness

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **RAPTOR**: Based on the RAPTOR paper for hierarchical retrieval (arXiv:2401.18059)
- **Model Context Protocol**: Anthropic's MCP for AI tool integration
- **DeepSeek-R1**: Advanced reasoning LLM with thinking capabilities
- **VoyageAI**: High-quality embedding APIs with multi-key support
- **LangFlow**: Visual workflow builder for LLM applications
- **Google Gemini**: Reliable fallback LLM
- **Supabase**: Modern PostgreSQL with vector extensions

---

*Built with ❤️ for intelligent deep research and multi-agent document analysis*

*🧠 **Deep Research Architecture** - LangFlow-ready with MCP integration*

*🎯 **Production-ready** - VoyageAI multi-key embedding and RAPTOR hierarchical trees*