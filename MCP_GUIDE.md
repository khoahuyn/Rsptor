# 🚀 RAPTOR MCP Server Guide

## Overview

The MCP (Model Context Protocol) server enables LangFlow and other AI tools to access the RAPTOR retrieval system through a standardized protocol.

## 🔧 Dependencies

The MCP server uses the official MCP library with SSE transport support:

```bash
cd raptor_service
uv add "mcp[server]>=1.0.0"
```

## 🏃‍♂️ Running the MCP Server

### 1. Start Main FastAPI Server (Required)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8081
```

### 2. Start MCP Server

#### HTTP Mode (for LangFlow & testing):
```bash
python run_mcp_server.py --mode http --host 127.0.0.1 --port 3333
```

#### STDIO Mode (for MCP clients):
```bash
python run_mcp_server.py --mode stdio
```

**Note**: HTTP mode provides SSE (Server-Sent Events) transport at `/mcp/sse` for LangFlow compatibility.

## 🧪 Testing the MCP Server

### Health Check:
```bash
curl http://localhost:3333/health
```

### List Available Tools:
```bash
curl -X POST http://localhost:3333/api/v1/mcp/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tools/list",
    "params": {}
  }'
```

### Test RAPTOR Retrieval:
```bash
curl -X POST http://localhost:3333/api/v1/mcp/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "2", 
    "method": "tools/call",
    "params": {
      "name": "raptor_retrieve",
      "arguments": {
        "query": "T1 World Championship victories",
        "tenant_id": "test_tenant",
        "kb_id": "test_tenant::kb::liên_minh_huyền_thoại",
        "top_k": 8
      }
    }
  }'
```

## 📝 Expected Response Formats

### Tools List Response:
```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    "tools": [
      {
        "name": "raptor_retrieve",
        "description": "Retrieve relevant documents using RAPTOR hierarchical retrieval system",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string", 
              "description": "Search query for document retrieval"
            },
            "tenant_id": {
              "type": "string", 
              "description": "Tenant identifier",
              "default": "test_tenant"
            },
            "kb_id": {
              "type": "string", 
              "description": "Knowledge base identifier",
              "default": "test_tenant::kb::liên_minh_huyền_thoại"
            },
            "top_k": {
              "type": "integer", 
              "description": "Number of top results to return",
              "default": 8
            }
          },
          "required": ["query"]
        }
      }
    ]
  }
}
```

### Retrieval Response:
```json
{
  "jsonrpc": "2.0",
  "id": "2",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "🎯 Retrieved 8 relevant chunks for query: 'T1 World Championship victories'\n📊 Knowledge Base: test_tenant::kb::liên_minh_huyền_thoại | Tenant: test_tenant\n\n📌 **Chunk 1** (ID: 27188baf-86a6-49b0-9309-7471ca00c489_chunk_28)\n📈 Similarity Score: 0.7261\n🌲 Level: 1\n🔗 URI: https://test_tenant::test_tenant::kb::liên_minh_huyền_thoại/chunks/27188baf-86a6-49b0-9309-7471ca00c489_chunk_28\n📝 **Content:**\nT1's five wins (2013, 2015, 2016, 2023, 2024) demonstrate their dominance...\n──────────────────────────────────────────────────"
      }
    ],
    "isError": false
  }
}
```

## 🔗 LangFlow Integration

### MCP Agent Configuration:

1. **Add MCP Server to LangFlow:**
   - Server URL: `http://127.0.0.1:3333/api/v1/mcp/sse`
   - Name: `raptor-retrieval`
   - Type: `HTTP`

2. **Agent Setup:**
   ```
   Agent Component → Settings → Tools → raptor_retrieve
   
   Required Parameters:
   - tenant_id: "test_tenant" 
   - kb_id: "test_tenant::kb::liên_minh_huyền_thoại"
   - top_k: 8 (integer, no quotes)
   ```

3. **Deep Research Flow Architecture:**
   ```
   Chat Input → Planning Agent → Research Executor → Writer Agent → Chat Output
                                      ↓
                               (3 parallel MCP calls)
   ```

### Alternative HTTP API Method:
```python
# Direct API call (bypass MCP)
import requests

def call_raptor_retrieval(query: str):
    response = requests.post(
        "http://localhost:8081/v1/ragflow/ragflow_retrieve",
        json={
            "query": query, 
            "tenant_id": "test_tenant", 
            "kb_id": "test_tenant::kb::liên_minh_huyền_thoại",
            "top_k": 8
        }
    )
    return response.json()
```

## 🔍 Troubleshooting

### Common Issues:

1. **"Timeout waiting for SSE session"**
   - Check FastAPI server is running on port 8081
   - Verify MCP server is running on port 3333
   - Ensure correct URL: `http://127.0.0.1:3333/api/v1/mcp/sse`

2. **"Tool parameter validation failed"**
   - Ensure `top_k` is integer (8) not string ("8")
   - Verify correct tenant_id and kb_id format

3. **"No chunks retrieved"**
   - Check if documents are processed in the knowledge base
   - Verify chunk size configuration (recommended: 1200 tokens)

### Debugging Steps:
- **MCP Server Logs**: Check console output for connection issues
- **FastAPI Logs**: Monitor uvicorn console for retrieval errors
- **Health Check**: `curl http://localhost:3333/health`
- **Direct API Test**: Use curl commands above to isolate MCP vs API issues

## ⚙️ Configuration Optimization

### Chunk Size Optimization:
```bash
# Edit config/chunking.py or set environment variable:
CHUNK_SIZE=1200  # Recommended (default: 512 too small)
CHUNK_OVERLAP_PERCENT=15  # Better context preservation
```

### Performance Notes:
- **MCP Server Role**: Acts as a protocol bridge, no caching
- **Retrieval Performance**: Dependent on FastAPI server and vector index
- **Parallel Calls**: Research Executor supports 3 simultaneous MCP calls
- **Index Persistence**: Vector indices cached for faster subsequent calls

## 📋 Production Checklist

- [ ] FastAPI server running and accessible
- [ ] MCP server configured with correct endpoints  
- [ ] Documents processed with optimal chunk size (1200+ tokens)
- [ ] LangFlow agent configured with correct parameters
- [ ] Health checks passing
- [ ] Test queries returning expected results

