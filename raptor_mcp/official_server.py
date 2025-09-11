import asyncio
import logging
from typing import Any, Dict, List, Sequence

from fastapi import FastAPI
from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
from starlette.routing import BaseRoute, Route
import uvicorn

# Import existing raptor service
from api.ragflow_raptor import ragflow_retrieve
from models import RetrievalRequest

logger = logging.getLogger("raptor.mcp.official")


class RaptorMCPService:
    """Official MCP Service for RAPTOR functionality using standard MCP library"""

    def __init__(self):
        """Initialize MCP service with official MCP library."""
        self.sse_transport = SseServerTransport("/mcp")
        self.mcp_server = self._create_mcp_server()
        self._connection_count = 0

    def _create_mcp_server(self) -> Server:
        """Create a standard MCP server with RAPTOR tools"""
        server = Server("RAPTOR Retrieval Service")

        # Register tools with the server
        @server.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="raptor_retrieve",
                    description="Retrieve relevant content using RAPTOR algorithm from knowledge base",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant content"
                            },
                            "tenant_id": {
                                "type": "string",
                                "description": "Tenant ID for data isolation",
                                "default": "test_tenant"
                            },
                            "kb_id": {
                                "type": "string",
                                "description": "Knowledge base ID",
                                "default": "main"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top results to return",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    },
                ),
            ]

        @server.call_tool()
        async def call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> Sequence[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            try:
                logger.info(f"🔧 Tool called: {name}")
                logger.info(f"📝 Arguments: {arguments}")

                if name == "raptor_retrieve":
                    # Validate and extract parameters
                    query = arguments.get("query", "").strip()
                    if not query:
                        raise ValueError("Query parameter is required and cannot be empty")
                        
                    tenant_id = arguments.get("tenant_id", "test_tenant")
                    kb_id = arguments.get("kb_id", "main")
                    top_k = min(max(arguments.get("top_k", 5), 1), 20)

                    logger.info(f"🔍 Retrieving: query='{query}', tenant={tenant_id}, kb={kb_id}, top_k={top_k}")

                    # Create and execute request
                    request = RetrievalRequest(
                        query=query,
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        top_k=top_k
                    )

                    result = await ragflow_retrieve(request)

                    # Parse response
                    if hasattr(result, 'dict'):
                        result_dict = result.dict()
                    else:
                        result_dict = result

                    chunks = result_dict.get("retrieved_nodes", [])
                    logger.info(f"📄 Retrieved {len(chunks)} chunks")

                    # Format response for MCP
                    if not chunks:
                        return [types.TextContent(type="text", text="❌ No relevant content found for your query.")]

                    # Create summary
                    summary = f"🎯 Retrieved {len(chunks)} relevant chunks for query: '{query}'\n"
                    summary += f"📊 Knowledge Base: {kb_id} | Tenant: {tenant_id}\n\n"

                    # Format chunks
                    content_text = summary
                    for i, node in enumerate(chunks, 1):
                        node_id = node.get('node_id', f'chunk_{i}')
                        similarity_score = node.get('similarity_score', 0.0)
                        content = node.get('content', '').strip()
                        level = node.get('level', 0)
                        
                        resource_uri = f"https://{tenant_id}::{kb_id}/chunks/{node_id}"
                        
                        chunk_text = f"📌 **Chunk {i}** (ID: {node_id})\n"
                        chunk_text += f"📈 Similarity Score: {similarity_score:.4f}\n"
                        chunk_text += f"🌲 Level: {level}\n"
                        chunk_text += f"🔗 URI: {resource_uri}\n"
                        chunk_text += f"📝 **Content:**\n{content}\n"
                        chunk_text += "─" * 50 + "\n"
                        
                        content_text += chunk_text

                    return [types.TextContent(type="text", text=content_text)]

                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                error_msg = f"❌ Tool execution failed: {str(e)}"
                logger.error(error_msg)
                import traceback
                traceback.print_exc()
                return [types.TextContent(type="text", text=error_msg)]

        return server

    async def handle_sse_connection(self, scope, receive, send):
        """Handle SSE connection for the MCP server"""
        logger.info(f"🔌 Handling SSE connection: {scope.get('method')} {scope.get('path')}")
        logger.info(f"📋 Headers: {dict(scope.get('headers', []))}")

        async with self.sse_transport.connect_sse(scope, receive, send) as (
            read_stream,
            write_stream,
        ):
            await self.mcp_server.run(
                read_stream, write_stream, self.mcp_server.create_initialization_options()
            )

    async def handle_post_message(self, scope, receive, send):
        """Handle POST messages for the MCP server"""
        logger.info(f"📨 Handling POST message: {scope.get('method')} {scope.get('path')}")
        logger.info(f"📋 Headers: {dict(scope.get('headers', []))}")

        await self.sse_transport.handle_post_message(scope, receive, send)

    async def handle_mcp_endpoint(self, scope, receive, send):
        """Handle the main MCP endpoint for both GET and POST according to MCP spec"""
        method = scope.get("method")
        path = scope.get('path')
        
        # Track connections and only log periodically  
        self._connection_count += 1
        if self._connection_count == 1:
            logger.info(f"🎯 First MCP connection established: {method} {path}")
        elif self._connection_count % 5 == 0:  # Log every 5th connection
            logger.info(f"📊 MCP connection #{self._connection_count} (LangFlow maintaining connections)")
        
        if method == "GET":
            # Handle SSE connection (reduced logging)
            async with self.sse_transport.connect_sse(scope, receive, send) as (
                read_stream,
                write_stream,
            ):
                await self.mcp_server.run(
                    read_stream, write_stream, self.mcp_server.create_initialization_options()
                )
        elif method == "POST":
            # Handle JSON-RPC message (reduced logging)
            if not hasattr(self, '_logged_first_post'):
                logger.info("✅ First POST request for JSON-RPC communication")
                self._logged_first_post = True
            await self.sse_transport.handle_post_message(scope, receive, send)
        else:
            # Return 405 Method Not Allowed
            logger.warning(f"❓ Unsupported method: {method}")
            response_body = b"Method Not Allowed"
            await send(
                {
                    "type": "http.response.start",
                    "status": 405,
                    "headers": [
                        [b"content-type", b"text/plain"],
                        [b"content-length", str(len(response_body)).encode()],
                    ],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": response_body,
                }
            )

    async def start_server(self, host: str = "127.0.0.1", port: int = 3333):
        """Start the MCP server with proper endpoints"""

        # Health check endpoint
        async def health_check(request):
            return {"status": "healthy", "service": "RAPTOR MCP Server (Official)"}

        # Debug endpoint to show available routes
        async def show_routes(request):
            return {
                "available_endpoints": [
                    "GET  /health",
                    "GET  /routes", 
                    "GET/POST /mcp (Main MCP endpoint)",
                    "GET  /api/v1/mcp/sse (LangFlow SSE)",
                    "POST /api/v1/mcp/rpc (LangFlow RPC)",
                ]
            }

        # Create handler class for the main MCP endpoint
        class HandleMCP:
            def __init__(self, service):
                self.service = service

            async def __call__(self, scope, receive, send):
                await self.service.handle_mcp_endpoint(scope, receive, send)

        # Create routes
        routes: List[BaseRoute] = [
            Route("/health", endpoint=health_check, methods=["GET"]),
            Route("/routes", endpoint=show_routes, methods=["GET"]),
            Route("/mcp", endpoint=HandleMCP(self), methods=["GET", "POST"]),  # Main MCP endpoint
            Route("/api/v1/mcp/sse", endpoint=HandleMCP(self), methods=["GET"]),  # LangFlow SSE
            Route("/api/v1/mcp/rpc", endpoint=HandleMCP(self), methods=["POST"]),  # LangFlow RPC
        ]

        app = FastAPI(routes=routes)
        
        logger.info("🚀 Starting Official RAPTOR MCP Server")
        logger.info(f"🌐 Server: http://{host}:{port}")
        logger.info("📍 Available endpoints:")
        logger.info("   • GET  /health")
        logger.info("   • GET  /routes") 
        logger.info("   • GET/POST /mcp (Main MCP endpoint)")
        logger.info("   • GET  /api/v1/mcp/sse (LangFlow SSE)")
        logger.info("   • POST /api/v1/mcp/rpc (LangFlow RPC)")
        logger.info("✅ Ready to accept MCP connections!")
        
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()


def create_raptor_mcp_service():
    """Create a MCP service for RAPTOR retrieval."""
    return RaptorMCPService()


if __name__ == "__main__":
    import argparse
    import sys

    # Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3333)
    args = parser.parse_args()

    print(f"Starting Official RAPTOR MCP server on {args.host}:{args.port}...")
    service = RaptorMCPService()
    asyncio.run(service.start_server(args.host, args.port))
