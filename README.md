# 🌳 RAG Service - Intelligent Document Q&A with RAPTOR Trees

> **Modern RAG system** combining **RAGFlow retrieval**, **RAPTOR hierarchical trees**, and **Smart LLM chat** for intelligent document question-answering with multi-language support.

---

## 🚀 Overview

**RAG Service** is a production-ready system that transforms documents into intelligent Q&A capabilities:

- 📚 **RAGFlow Processing**: Advanced document chunking and embedding with BGE-M3
- 🌳 **RAPTOR Trees**: Hierarchical clustering with GMM+BIC for optimal retrieval  
- 🤖 **Smart Chat**: LLM-powered answers using Gemini with context awareness
- 🏢 **Multi-tenant**: Isolated data per tenant/knowledge base
- ⚡ **High Performance**: 1-second query time with production optimizations

### Key Features

- ✅ **Document Upload** - Process .md, .txt, .pdf files with intelligent chunking
- ✅ **RAPTOR Tree Building** - Hierarchical clustering for enhanced context retrieval
- ✅ **Smart Retrieval** - Hybrid search with vector + text + keyword matching  
- ✅ **Intelligent Chat** - Context-aware LLM responses with safety checks
- ✅ **Multi-language** - Auto-detect and respond in user's language (Vietnamese, English, Chinese)
- ✅ **Production Ready** - Docker, async processing, comprehensive error handling

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Document Upload] --> B[RAGFlow Processing]
    B --> C[Chunking & Cleaning]
    C --> D[BGE-M3 Embeddings<br/>via Ollama]
    D --> E[RAPTOR Tree Building]
    E --> F[GMM Clustering + BIC]
    F --> G[DeepSeek-V3 Summary]
    G --> H[Supabase Storage]
    
    I[User Query] --> J[Smart Retrieval]
    J --> K[Vector Search + Ranking]
    K --> L[Context Preparation]
    L --> M[Gemini LLM]
    M --> N[Intelligent Answer]
    
    style D fill:#e1f5fe
    style G fill:#fff3e0
    style M fill:#f3e5f5
```

### Tech Stack

- **API**: FastAPI + Uvicorn (async)
- **Database**: Supabase (PostgreSQL + pgvector)  
- **Embeddings**: BGE-M3 via Ollama (local) or Voyage AI (cloud)
- **LLM Summary**: DeepSeek-V3 via FPT Cloud
- **LLM Chat**: Google Gemini 1.5 Flash
- **Clustering**: Gaussian Mixture Models + BIC optimization
- **Deployment**: Docker + Docker Compose

---

## 📖 API Endpoints

### Core APIs

| Endpoint | Method | Description | Use Case |
|----------|--------|-------------|----------|
| `/v1/ragflow/process` | POST | Upload & process documents into RAPTOR trees | Document ingestion |
| `/v1/ragflow/retrieve` | POST | Raw retrieval with chunks and similarity scores | Advanced retrieval |
| `/v1/chat/smart` | POST | **🌟 Intelligent Q&A** - Auto retrieval + LLM generation | **Recommended for end users** |

### 🌟 Smart Chat API (Main Feature)

**Simplest way to get intelligent answers from your documents:**

```bash
curl --location 'http://localhost:8081/v1/chat/smart' \
--header 'Content-Type: application/json' \
--data '{
  "query": "Ngành Thiết kế vi mạch bán dẫn có cơ hội nghề nghiệp như thế nào?",
  "tenant_id": "demo", 
  "kb_id": "docs"
}'
```

**Response:**
```json
{
  "answer": "Ngành Thiết kế vi mạch bán dẫn có nhiều cơ hội nghề nghiệp hấp dẫn:\n\n**Cơ hội nghề nghiệp:**\n- Kỹ sư thiết kế vi mạch (vị trí cốt lõi)\n- Thiết kế mạch số, mạch tương tự, mạch hỗn hợp\n- Hệ thống trên chip (SoC)\n- Kiểm thử vi mạch\n\n**Mức lương:**\n- Khởi điểm từ 12-20 triệu đồng/tháng\n- Có thể cao hơn tùy theo năng lực và kinh nghiệm"
}
```

---

## ⚡ Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Ollama** (for BGE-M3 embeddings)
- **Supabase** account (for database)

### 2. Install Dependencies

```bash
git clone <your-repo>
cd raptor_service

# Install with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### 3. Setup Ollama & BGE-M3

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull BGE-M3 model
ollama pull bge-m3:latest

# Verify
curl http://localhost:11434/api/tags
```

### 4. Configure Environment

Copy and configure environment file:
```bash
cp env.template .env
```

Edit `.env` with your credentials:
```env
# Supabase Database
DATABASE_URL=postgresql+psycopg://postgres.PROJECT_ID:PASSWORD@...supabase.com:6543/postgres?sslmode=require

# API Keys
GEMINI_API_KEY=your_gemini_api_key
LLM_API_KEY=your_fpt_cloud_api_key

# BGE-M3 via Ollama (default)
EMBED_BASE_URL=http://localhost:11434/api/embeddings
EMBED_MODEL=bge-m3:latest
```

### 5. Setup Database

```bash
python setup_database.py
```

**Expected output:**
```
🚀 RAPTOR Service Database Setup
✅ Database connection successful!
✅ pgvector extension enabled
✅ All required tables created
🎉 Database setup completed successfully!
```

### 6. Start Service

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8081
```

### 7. Upload Documents & Ask Questions

**Upload document:**
```bash
curl -X POST "http://localhost:8081/v1/ragflow/process" \
  -F "file=@your_document.md" \
  -F "tenant_id=demo" \
  -F "kb_id=docs"
```

**Ask questions:**
```bash
curl -X POST "http://localhost:8081/v1/chat/smart" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "tenant_id": "demo",
    "kb_id": "docs"
  }'
```

---

## 🛠️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| **Database** |
| `DATABASE_URL` | ✅ | - | Supabase PostgreSQL connection string |
| `DB_ENABLE_SSL` | ❌ | `true` | Enable SSL for database connection |
| **Embeddings** |
| `EMBED_BASE_URL` | ✅ | `http://localhost:11434/api/embeddings` | BGE-M3 via Ollama |
| `EMBED_MODEL` | ✅ | `bge-m3:latest` | Embedding model name |
| `EMBED_VECTOR_DIM` | ✅ | `1024` | Vector dimension |
| **LLM Services** |
| `GEMINI_API_KEY` | ✅ | - | Google Gemini for smart chat |
| `LLM_API_KEY` | ✅ | - | FPT Cloud for document summarization |
| `LLM_MODEL` | ❌ | `DeepSeek-V3` | Summary model |
| **Optional** |
| `RAPTOR_MAX_CLUSTERS` | ❌ | `64` | Max clusters per RAPTOR level |
| `CHUNK_SIZE` | ❌ | `1000` | Default chunk size |

### Alternative Embedding Services

**Switch to Voyage AI** (cloud-based):
```env
EMBED_BASE_URL=https://api.voyageai.com/v1
EMBED_API_KEY=your_voyage_api_keys_comma_separated
EMBED_MODEL=voyage-context-3
```

### Advanced Configuration

Detailed settings in `config/` directory:
- `config/database.py` - Database and connection pooling
- `config/embedding.py` - Embedding model configuration  
- `config/raptor.py` - RAPTOR tree building parameters
- `config/chat.py` - Gemini LLM settings

---

## 🧪 Testing

### Run Test Suite
```bash
pytest tests/ -v
```

### Manual API Testing
```bash
# Test smart chat
python test_chat_simple.py

# Test document processing  
python test_knowledge_base_api.py

# Test with cURL
bash test_chat_curl.sh
```

### Health Check
```bash
curl http://localhost:8081/v1/chat/health
```

---

## 🐳 Docker Deployment

### Development
```bash
docker-compose -f docker-compose.dev.yml up -d
```

### Production
```bash
docker-compose up -d
```

**Service URLs:**
- API: `http://localhost:8081`
- Docs: `http://localhost:8081/docs`

---

## 📊 Performance Metrics

### Retrieval Performance
- **Query Speed**: ~1 second for retrieval + generation
- **Embedding**: BGE-M3 1024-dimensional vectors
- **Tree Building**: GMM clustering with BIC optimization
- **Concurrency**: Async processing with connection pooling

### Scalability
- **Multi-tenant**: Isolated data per `tenant_id`/`kb_id`
- **Vector Search**: pgvector with HNSW indexing
- **Batch Processing**: Optimized embedding generation
- **Memory**: Efficient vector storage and retrieval

---

## 🔧 Troubleshooting

### Common Issues

**❌ Database Connection Failed**
```bash
# Test connection
python -c "from database.connection import test_connection; test_connection()"

# Check Supabase status
curl https://your-project.supabase.co/rest/v1/
```

**❌ Ollama/BGE-M3 Not Working**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
sudo systemctl restart ollama

# Re-pull model
ollama pull bge-m3:latest
```

**❌ Gemini API Errors**  
```bash
# Verify API key
python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); print('OK')"

# Check quota
curl -H "Authorization: Bearer $GEMINI_API_KEY" https://generativelanguage.googleapis.com/v1/models
```

**❌ No Context Retrieved**
```bash
# Check if documents are processed
curl -X POST "http://localhost:8081/v1/ragflow/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "tenant_id": "demo", "kb_id": "docs"}'
```

### Debug Mode

Enable detailed logging:
```env
LOG_LEVEL=DEBUG
DB_DATABASE_ECHO=true
```

Check logs in `logs/` directory for detailed debugging information.

---

## 🛡️ Production Deployment

### Security Checklist
- [ ] Change default database credentials
- [ ] Enable SSL/TLS for all connections
- [ ] Rotate API keys regularly
- [ ] Set up proper firewall rules
- [ ] Enable request rate limiting

### Monitoring
- [ ] Set up health check endpoints
- [ ] Monitor database connection pool
- [ ] Track API response times
- [ ] Monitor embedding model performance

### Backup Strategy
- [ ] Regular database backups
- [ ] Document storage backup
- [ ] Environment configuration backup

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Follow code style: `black . && isort .`
4. Add tests for new features
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ --cov=raptor_service
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
- **Ollama**: Local LLM and embedding model serving
- **FPT Cloud**: Vietnamese LLM services
- **Google Gemini**: Advanced language model for chat

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: See `/docs` folder for detailed guides

---

*Built with ❤️ for intelligent document Q&A - Supporting Vietnamese, English, and Chinese languages*