import logging
import warnings
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed, using system environment")

from api.knowledge_base import router as kb_router
from api.ragflow_raptor import router as ragflow_raptor_router
from api.chat_completion import router as chat_router

warnings.filterwarnings("ignore", module="umap")
warnings.filterwarnings("ignore", message=".*n_jobs.*overridden.*random_state.*")
logging.basicConfig(level=logging.WARNING)  
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(kb_router)
app.include_router(ragflow_raptor_router)
app.include_router(chat_router)

