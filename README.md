# 🌳 RAPTOR RAG Service - AI Assistants & Smart Document Q&A

> **Production-ready RAG system** with **RAPTOR hierarchical trees**, **AI Assistants**, and **React Frontend** for intelligent document processing and conversation management.

---

## 🚀 Overview

**RAPTOR RAG Service** transforms documents into intelligent AI assistants with conversation capabilities:

- 📚 **RAGFlow Processing**: Advanced document chunking with parallel upload support
- 🌳 **RAPTOR Trees**: Hierarchical clustering with GMM+BIC for optimal retrieval  
- 🤖 **AI Assistants**: Create and manage AI assistants linked to knowledge bases
- 💬 **Chat Sessions**: Persistent conversation history with message management
- 🎨 **React Frontend**: Complete UI for document upload, assistant creation, and chat
- 🏢 **Multi-tenant**: Isolated data per tenant/knowledge base with cascade deletion

### ⭐ Key Features

- ✅ **AI Assistant Management** - Create, configure, and delete AI assistants
- ✅ **Chat Sessions & History** - Persistent conversations with message tracking
- ✅ **Parallel Document Upload** - Upload multiple .md files simultaneously with progress tracking
- ✅ **RAPTOR Tree Building** - Hierarchical clustering for enhanced context retrieval
- ✅ **Smart Retrieval** - Context-aware responses using knowledge base content
- ✅ **React Frontend** - Modern UI with real-time progress and error handling
- ✅ **Multi-language Support** - Auto-detect and respond in Vietnamese, English
- ✅ **Production Ready** - Async FastAPI, database migrations, comprehensive error handling

### 🚀 **Performance Optimizations**

**Retrieval Performance Improvements:**
- ⚡ **Significantly faster chat responses** - Optimized retrieval pipeline
- 🔥 **Cold start acceleration** - Persistent vector indexing for faster startup
- 💯 **Advanced scoring optimization** - Multi-layer scoring with early termination
- 🗃️ **Database bulk loading** - N+1 queries eliminated for consistent performance
- ⚡ **Smart caching strategy** - Improved cache hit rates with intelligent normalization
- 🚀 **Embedding bottleneck resolved** - Multi-key parallel processing

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph "Frontend (React)"
        A[Document Upload UI] --> B[AI Assistant Creation]
        B --> C[Chat Interface]
    end
    
    subgraph "Backend (FastAPI)"
        D[RAGFlow Processing] --> E[RAPTOR Tree Building]
        E --> F[AI Assistant Management]
        F --> G[Chat Service]
    end
    
    subgraph "Optimizations"
        H[Persistent Vector Index]
        I[Smart Cache Layer]
        J[Bulk DB Loading]
    end
    
    subgraph "Embeddings"
        K[VoyageAI Multi-Key]
        L[BGE-M3 Local]
    end
    
    subgraph "Storage"
        M[(Supabase DB)]
        N[(Redis Cache)]
    end
    
    A -->|Parallel Upload| D
    D --> M
    E --> K
    E --> L
    G --> M
    C -->|Real-time Chat| G
    
    G --> H
    G --> I
    G --> J
    I --> N
    
    style A fill:#e1f5fe
    style D fill:#fff3e0
    style G fill:#f3e5f5
    style H fill:#e8f5e8
    style I fill:#ffeb3b
    style M fill:#f1f8e9
```

### Tech Stack

- **Backend**: FastAPI + Uvicorn (async)
- **Frontend**: React + TypeScript + TanStack Router + Hero UI
- **Database**: Supabase (PostgreSQL + pgvector) with Alembic migrations
- **Embeddings**: 
  - **🏆 Recommended**: VoyageAI (multi-key, production-ready)
  - **Alternative**: BGE-M3 via Ollama (local, cost-effective)
- **LLM**: Gemini 1.5 Flash for chat + DeepSeek-V3 for summarization
- **Clustering**: Gaussian Mixture Models + BIC optimization
- **Caching**: Smart hash normalization + Redis persistence
- **Vector Index**: FAISS with disk persistence

---

## 📖 API Endpoints

### 🤖 AI Assistant Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/ai/assistants` | GET | List all AI assistants |
| `/v1/ai/assistants` | POST | Create new AI assistant |
| `/v1/ai/assistants/{id}` | PUT | Update AI assistant |
| `/v1/ai/assistants/{id}` | DELETE | Delete AI assistant (cascade) |

### 💬 Chat & Sessions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/sessions` | POST | Create new chat session |
| `/v1/chat/sessions/{id}/messages` | GET | Get session messages |
| `/v1/chat/sessions/{id}/chat` | POST | Send message & get AI response |
| `/v1/chat/assistants/{id}/sessions` | GET | Get assistant's chat sessions |

### 📚 Document Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/ragflow/process` | POST | Upload & process documents (supports .md/.markdown) |
| `/v1/ragflow/retrieve` | POST | ⚡ **Optimized retrieval** with advanced scoring |

### 🗂️ Knowledge Base

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/kb/create` | POST | Create knowledge base |
| `/v1/kb/list` | GET | List knowledge bases |
| `/v1/kb/{id}/documents` | GET | Get documents in KB |

---

## ⚡ Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for frontend)
- **uv** (fast Python package manager)
- **pnpm** (efficient Node.js package manager)
- **VoyageAI API Key** (recommended) or **Ollama** (for local BGE-M3)
- **Supabase** account (for database)

### 2. Backend Setup

```bash
git clone <your-repo>
cd raptor_service

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
uv pip install -r requirements.txt

# Setup environment
cp env.template .env
# Edit .env with your credentials (see Configuration section)

# Setup database
python setup_database.py

# Start backend
uvicorn main:app --reload --host 0.0.0.0 --port 8081
```

### 3. Frontend Setup

```bash
cd frontend

# Install pnpm if not already installed
npm install -g pnpm

# Install dependencies with pnpm
pnpm install

# Start development server  
pnpm dev
```

### 4. Access Applications

- **Backend API**: http://localhost:8081
- **API Documentation**: http://localhost:8081/docs
- **Frontend UI**: http://localhost:5173

---

## 🔧 Configuration

### 🏆 Recommended: VoyageAI Multi-Key Setup

```env
# VoyageAI Embeddings (RECOMMENDED for production)
EMBED_BASE_URL=https://api.voyageai.com/v1
EMBED_API_KEY=pa-key1,pa-key2,pa-key3,pa-key4,pa-key5,pa-key6
EMBED_MODEL=voyage-context-3
EMBED_VECTOR_DIM=1024

# Benefits:
# ✅ Fastest performance with parallel processing
# ✅ Multi-key parallel processing (4-6 keys recommended)
# ✅ Professional-grade reliability
# ✅ Zero rate limits with proper key distribution
```

### 🔄 Alternative: BGE-M3 Local Setup

```env
# BGE-M3 Local Embeddings (cost-effective)
EMBED_BASE_URL=http://localhost:11434/api/embeddings
EMBED_API_KEY=
EMBED_MODEL=bge-m3:latest
EMBED_VECTOR_DIM=1024
```

**Setup Ollama for BGE-M3:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull BGE-M3 model
ollama pull bge-m3:latest

# Start Ollama service
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

### Complete Environment Variables

```env
# === EMBEDDING CONFIGURATION ===
# VoyageAI (RECOMMENDED - fastest performance)
EMBED_BASE_URL=https://api.voyageai.com/v1
EMBED_API_KEY=pa-key1,pa-key2,pa-key3,pa-key4,pa-key5,pa-key6
EMBED_MODEL=voyage-context-3
EMBED_VECTOR_DIM=1024

# Alternative: BGE-M3 Local (cost-effective)
# EMBED_BASE_URL=http://localhost:11434/api/embeddings
# EMBED_API_KEY=
# EMBED_MODEL=bge-m3:latest
# EMBED_VECTOR_DIM=1024

# === LLM CONFIGURATION ===
# FPT Cloud for document summarization
LLM_BASE_URL=https://mkp-api.fptcloud.com/v1
LLM_API_KEY=your_fpt_cloud_api_key
LLM_MODEL=DeepSeek-V3

# Google Gemini for smart chat
GEMINI_API_KEY=your_gemini_api_key

# === DATABASE CONFIGURATION ===
# Supabase PostgreSQL with pgvector
DATABASE_URL=postgresql+psycopg://postgres.PROJECT_ID:PASSWORD@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require
DB_ENABLE_SSL=true
DB_SSL_CERT_PATH=prod-ca-2021.crt
SUPABASE_SSLROOTCERT=./database/prod-ca-2021.crt

# === CACHING CONFIGURATION ===
# Redis Cache (optional, for better performance)
REDIS_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_TTL=300

# Cache settings
RETRIEVAL_CACHE_TTL_SECONDS=300
RETRIEVAL_CACHE_MAX_ENTRIES=100
EMBED_CACHE_TTL_SECONDS=86400

# === OPTIONAL CONFIGURATION ===
# Raptor tree building
RAPTOR_MAX_CLUSTERS=64
RAPTOR_THRESHOLD=0.1
RAPTOR_RANDOM_SEED=42

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# API settings
API_HOST=0.0.0.0
API_PORT=8081
API_RELOAD=true
```

### 🔑 VoyageAI Multi-Key Best Practices

**✅ Recommended Key Count:** 4-6 keys
```env
# Example with multiple keys for optimal performance
EMBED_API_KEY=pa-xxxxxxxxxx1,pa-xxxxxxxxxx2,pa-xxxxxxxxxx3,pa-xxxxxxxxxx4,pa-xxxxxxxxxx5,pa-xxxxxxxxxx6
```

**Benefits of Multiple Keys:**
- 🚀 **Parallel Processing**: True parallel embedding across all keys
- ⚡ **Zero Rate Limits**: Smart load balancing prevents rate limit encounters
- 🎯 **Optimal Performance**: Significantly faster than sequential processing
- 🛡️ **Automatic Failover**: Rapid failover if any key encounters issues
- 📊 **Token-aware Selection**: Prioritizes keys with lower current usage

---

## 💻 Frontend Features

### 📁 Document Management
- **Parallel Upload**: Upload multiple .md files simultaneously
- **Progress Tracking**: Real-time upload progress with percentage
- **Validation**: Client-side validation for file types and sizes
- **Error Handling**: Detailed error messages and retry capability

### 🤖 AI Assistant Creation
- **Knowledge Base Integration**: Link assistants to specific knowledge bases
- **Custom Configuration**: Set system prompts and model parameters
- **Management**: Full CRUD operations with cascade deletion

### 💬 Chat Interface
- **Real-time Messaging**: Send messages and receive AI responses
- **Session Management**: Create and switch between chat sessions
- **Message History**: Persistent conversation storage
- **Context-aware Responses**: AI responses based on knowledge base content

---

## 🧪 Usage Examples

### Create AI Assistant

```bash
curl -X POST "http://localhost:8081/v1/ai/assistants" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "kb_id": "my_documents",
    "name": "Technical Support Bot",
    "description": "AI assistant for technical documentation",
    "system_prompt": "You are a helpful technical support assistant."
  }'
```

### Upload Documents

```bash
curl -X POST "http://localhost:8081/v1/ragflow/process" \
  -F "file=@technical_guide.md" \
  -F "tenant_id=demo" \
  -F "kb_id=my_documents" \
  -F "enable_raptor=true"
```

### Start Chat Session

```bash
curl -X POST "http://localhost:8081/v1/chat/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "assistant_id": "demo::assistant::12345",
    "name": "Support Chat"
  }'
```

### Send Message

```bash
curl -X POST "http://localhost:8081/v1/chat/sessions/{session_id}/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I configure the database connection?",
    "stream": false
  }'
```

---

## 📊 Performance & Scalability

### 🚀 **Performance Improvements**

**Chat Response Experience:**
```bash
✅ Popular Questions:    Lightning fast responses with cache hits
✅ Regular Chat:         Smooth, consistent conversation experience  
✅ Cold Start:           Significantly faster server startup
✅ Database Fallback:    Reliable performance without timeouts

🎯 Overall: Much faster chat responses across all scenarios
```

**Retrieval Optimizations:**
```bash
✅ Embedding Bottleneck: RESOLVED   (Multi-key parallel processing)
✅ Vector Index Rebuild: OPTIMIZED  (Persistent indexing for faster startup)
✅ Scoring Complexity:   ENHANCED   (Early termination + preprocessing)
✅ Database N+1 Queries: ELIMINATED (Bulk loading eliminates slow fallback)
✅ Cache Strategy:       IMPROVED   (Smart normalization increases hit rates)
```

### 🏆 RAPTOR Processing Performance

**VoyageAI Multi-Key (Recommended):**
```bash
✅ RAPTOR Tree Building: Fast and reliable
✅ Embedding Generation: Near-instant with parallel processing
✅ Total Processing: Optimized for production workloads
✅ Parallel Keys: Multiple keys simultaneous processing
✅ Rate Limits: Zero encounters with proper load balancing
```

**BGE-M3 Local (Optimized):**
```bash
✅ RAPTOR Tree Building: Competitive performance
✅ Embedding Generation: Fast with optimized parallel processing
✅ Total Processing: Efficient for cost-conscious deployments
✅ Parallel Processing: High concurrent connections
✅ Cost: Zero API costs
```

### Performance Comparison

| Feature | VoyageAI Multi-Key | BGE-M3 Local |
|---------|-------------------|--------------|
| **RAPTOR Speed** | Fastest ⚡ | Fast ✅ |
| **Embedding Speed** | Excellent 🚀 | Good ✅ |
| **Reliability** | Production-grade 🛡️ | Reliable 🛡️ |
| **Cost** | API usage 💳 | Free 🆓 |
| **Privacy** | Cloud ☁️ | Local 🔒 |

### Database Features
- **Multi-tenant Isolation**: Data separated by `tenant_id`
- **Vector Search**: pgvector with HNSW indexing for fast similarity search
- **Optimized Writes**: Bulk operations with skip-refresh for embeddings
- **Cascade Deletion**: Automatic cleanup when deleting assistants
- **Migration System**: Alembic for database schema management

### Frontend Performance  
- **Parallel Processing**: Multiple file uploads with individual progress tracking
- **Real-time Updates**: Immediate UI updates without page refresh
- **Error Recovery**: Graceful handling of upload failures and retries
- **Responsive Design**: Works on desktop and mobile devices

---

## 🔧 Troubleshooting

### Common Issues

**❌ Database Connection Failed**
```bash
# Test connection
python setup_database.py

# Check Supabase project status and credentials in .env
```

**❌ VoyageAI API Issues**
```bash
# Check API keys format
# Ensure keys start with "pa-" and are comma-separated
# Verify API key permissions and rate limits
```

**❌ Ollama/BGE-M3 Not Working**
```bash
# Check Ollama service
curl http://localhost:11434/api/tags

# Restart and pull model
ollama pull bge-m3:latest
```

**❌ Frontend Not Loading**
```bash
cd frontend
pnpm install
pnpm dev

# Check if backend is running on port 8081
```

**❌ File Upload Fails**
- Ensure files are .md or .markdown format
- Check file size limits (10MB default)
- Verify knowledge base exists before upload

### Debug Mode

Enable detailed logging:
```env
LOG_LEVEL=DEBUG
```

Check browser console for frontend issues and backend logs for API problems.

---

## 🚀 Production Deployment

### Backend Deployment
1. Set production environment variables (use VoyageAI for best performance)
2. Run database migrations: `alembic upgrade head`
3. Deploy FastAPI with proper ASGI server (Gunicorn + Uvicorn)
4. Configure reverse proxy (Nginx) for static files and API
5. Setup Redis for production caching

### Frontend Deployment
```bash
cd frontend
pnpm build
# Deploy dist/ folder to static hosting (Vercel, Netlify, etc.)
```

### Security Checklist
- [ ] Use environment variables for all secrets
- [ ] Enable HTTPS for all connections
- [ ] Configure CORS for production domains
- [ ] Set up proper database user permissions
- [ ] Regular backup of database and uploaded files
- [ ] Rotate VoyageAI API keys regularly
- [ ] Secure Redis instance with authentication

### Production Recommendations
- **🏆 Use VoyageAI**: Best performance and reliability
- **📊 Monitor Usage**: Track API usage and costs
- **🔄 Load Balancing**: Consider multiple backend instances
- **📈 Scaling**: Add more VoyageAI keys for higher throughput
- **⚡ Redis Cache**: Enable for significantly improved cache hit rates

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`  
3. Follow code style: `black . && isort .`
4. Test both backend and frontend changes
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Setup

```bash
# Backend development (using uv)
uv pip install -r requirements.txt
python setup_database.py
uvicorn main:app --reload

# Frontend development (using pnpm)
cd frontend
pnpm install
pnpm dev

# Run tests
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **RAPTOR**: Based on the RAPTOR paper for hierarchical retrieval (arXiv:2401.18059)
- **RAGFlow**: Inspired by RAGFlow's hybrid retrieval methodology
- **VoyageAI**: Advanced embedding APIs with excellent performance
- **BGE-M3**: Beijing Academy of AI's multilingual embedding model
- **Supabase**: Modern PostgreSQL with vector extensions
- **Ollama**: Local embedding model serving
- **React + Hero UI**: Modern frontend framework and component library

---

*Built with ❤️ for intelligent document processing and AI assistant management*

*🏆 Optimized for production with VoyageAI multi-key embedding and RAPTOR hierarchical trees*

*⚡ **Significantly faster chat responses** with advanced retrieval optimizations*