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
    
    subgraph "Storage"
        H[(Supabase DB)]
        I[BGE-M3 Embeddings]
    end
    
    A -->|Parallel Upload| D
    D --> H
    E --> I
    G --> H
    C -->|Real-time Chat| G
    
    style A fill:#e1f5fe
    style D fill:#fff3e0
    style G fill:#f3e5f5
    style H fill:#e8f5e8
```

### Tech Stack

- **Backend**: FastAPI + Uvicorn (async)
- **Frontend**: React + TypeScript + TanStack Router + Hero UI
- **Database**: Supabase (PostgreSQL + pgvector) with Alembic migrations
- **Embeddings**: BGE-M3 via Ollama (1024-dimensional vectors)
- **LLM**: Gemini 1.5 Flash for chat + DeepSeek-V3 for summarization
- **Clustering**: Gaussian Mixture Models + BIC optimization

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
| `/v1/ragflow/retrieve` | POST | Raw retrieval with chunks and scores |

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
- **Ollama** (for BGE-M3 embeddings)
- **Supabase** account (for database)

### 2. Backend Setup

```bash
git clone <your-repo>
cd raptor_service

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp env.template .env
# Edit .env with your Supabase and API credentials

# Setup database
python setup_database.py

# Start backend
uvicorn main:app --reload --host 0.0.0.0 --port 8081
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server  
npm run dev
```

### 4. Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull BGE-M3 model
ollama pull bge-m3:latest

# Verify
curl http://localhost:11434/api/tags
```

### 5. Access Applications

- **Backend API**: http://localhost:8081
- **API Documentation**: http://localhost:8081/docs
- **Frontend UI**: http://localhost:5173

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

## 🔧 Configuration

### Environment Variables

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres.PROJECT_ID:PASSWORD@...supabase.com:6543/postgres?sslmode=require

# API Keys  
GEMINI_API_KEY=your_gemini_api_key
LLM_API_KEY=your_fpt_cloud_api_key

# BGE-M3 Embeddings
EMBED_BASE_URL=http://localhost:11434/api/embeddings
EMBED_MODEL=bge-m3:latest
EMBED_VECTOR_DIM=1024

# Optional
RAPTOR_MAX_CLUSTERS=64
CHUNK_SIZE=1000
```

### Key Configuration Files

- `config/database.py` - Database connection and pooling settings
- `config/embedding.py` - BGE-M3 embedding configuration  
- `config/raptor.py` - RAPTOR tree building parameters
- `config/chat.py` - Gemini LLM settings

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

### RAPTOR Processing Performance
- **Upload Speed**: Multiple files processed in parallel
- **Tree Building**: GMM clustering with BIC optimization (typically 1-3 minutes)
- **Embedding Generation**: BGE-M3 1024-dimensional vectors via Ollama
- **Query Response**: ~1-2 seconds for retrieval + generation

### Database Features
- **Multi-tenant Isolation**: Data separated by `tenant_id`
- **Vector Search**: pgvector with HNSW indexing for fast similarity search
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
npm install
npm run dev

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
1. Set production environment variables
2. Run database migrations: `alembic upgrade head`
3. Deploy FastAPI with proper ASGI server (Gunicorn + Uvicorn)
4. Configure reverse proxy (Nginx) for static files and API

### Frontend Deployment
```bash
cd frontend
npm run build
# Deploy dist/ folder to static hosting (Vercel, Netlify, etc.)
```

### Security Checklist
- [ ] Use environment variables for all secrets
- [ ] Enable HTTPS for all connections
- [ ] Configure CORS for production domains
- [ ] Set up proper database user permissions
- [ ] Regular backup of database and uploaded files

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
# Backend development
pip install -r requirements.txt
python setup_database.py
uvicorn main:app --reload

# Frontend development  
cd frontend
npm install
npm run dev

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
- **BGE-M3**: Beijing Academy of AI's multilingual embedding model
- **Supabase**: Modern PostgreSQL with vector extensions
- **Ollama**: Local embedding model serving
- **React + Hero UI**: Modern frontend framework and component library

---

*Built with ❤️ for intelligent document processing and AI assistant management*