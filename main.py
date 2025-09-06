import asyncio
import sys

# Fix Windows event loop for psycopg
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import warnings
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, using system environment")

from api.knowledge_base import router as kb_router
from api.ragflow_raptor import router as ragflow_raptor_router
from api.chat_completion import router as chat_router
from api.assistant import router as assistant_router

warnings.filterwarnings("ignore", module="umap")
warnings.filterwarnings("ignore", message=".*n_jobs.*overridden.*random_state.*")
logging.basicConfig(level=logging.INFO)  # Show INFO logs for debugging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)

# Enable RAPTOR timing logs specifically
logging.getLogger("services.build_tree").setLevel(logging.INFO)
logging.getLogger("services.document.raptor_builder").setLevel(logging.INFO)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(kb_router)
app.include_router(ragflow_raptor_router)
app.include_router(chat_router)
app.include_router(assistant_router)

