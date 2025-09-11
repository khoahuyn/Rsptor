import asyncio
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Setup logging for MCP server
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("raptor.mcp.official").setLevel(logging.INFO)

# Reduce noise from internal MCP library, uvicorn, and faiss
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

from raptor_mcp.official_server import create_raptor_mcp_service


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pure RAPTOR MCP Server")
    parser.add_argument("--mode", choices=["stdio", "http"], default="http",
                       help="Server mode: stdio for Cursor, http for LangFlow/testing")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host")
    parser.add_argument("--port", type=int, default=3333, help="HTTP port")
    
    args = parser.parse_args()
    
    try:
        service = create_raptor_mcp_service()
        
        if args.mode == "stdio":
            print("🚀 Starting Official RAPTOR MCP server in stdio mode...", file=sys.stderr)
            print("⚠️  stdio mode not implemented yet, use http mode", file=sys.stderr)
        else:
            asyncio.run(service.start_server(args.host, args.port))
            
    except KeyboardInterrupt:
        print("\n⛔ Server stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"❌ Server failed to start: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

