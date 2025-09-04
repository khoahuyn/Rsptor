import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from services.document.optimized_processor import get_optimized_processor
from services.retrieval import enhanced_ragflow_retrieval
from models import DocumentProcessSummary, RetrievalRequest, RetrievalResponse
from utils.progress import raptor_progress_context

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/ragflow", tags=["RAGFlow"])


@router.post("/process", response_model=DocumentProcessSummary)
async def process_document_optimized(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    kb_id: str = Form(...),
    enable_raptor: bool = Form(True),
    max_clusters: int = Form(64),
    threshold: float = Form(0.1),
    random_seed: int = Form(42)
):

    try:
        logger.info(f"🚀 OPTIMIZED RAGFlow processing START: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file content")
        
        # Validate file type
        if not file.filename.endswith(('.md', '.markdown')):
            raise HTTPException(
                status_code=400, 
                detail="Only Markdown files (.md, .markdown) are supported"
            )
        
        # Use optimized processor with progress tracking
        with raptor_progress_context(f"OPTIMIZED Processing {file.filename}", show_console=True) as progress:
            progress.update(completed=5, description="🚀 Starting optimized pipeline...")
            
            def progress_callback(msg: str):
                progress.update(advance=15, description=f"🚀 {msg}")
            
            optimized_processor = get_optimized_processor()
            
            # 🚀 SINGLE OPTIMIZED CALL: Document + RAPTOR in memory
            result = await optimized_processor.process_document_with_raptor(
                file_content=file_content,
                filename=file.filename,
                tenant_id=tenant_id,
                kb_id=kb_id,
                enable_raptor=enable_raptor,
                max_clusters=max_clusters,
                threshold=threshold,
                random_seed=random_seed,
                progress_callback=progress_callback
            )
            
            progress.update(completed=100, description="✅ Optimized processing completed!")
        
        logger.info(f"✅ OPTIMIZED RAGFlow processing complete: {file.filename} → {result.doc_id}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ OPTIMIZED RAGFlow processing failed: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Optimized RAGFlow processing failed", 
                "message": str(e),
                "type": type(e).__name__
            }
        )


# Wrapper function for chat service
async def ragflow_retrieve(req: RetrievalRequest) -> RetrievalResponse:
    """Wrapper function for enhanced_ragflow_retrieval"""
    return await enhanced_ragflow_retrieval.retrieve(req)


@router.post("/retrieve", response_model=RetrievalResponse, response_model_exclude_none=True)
async def retrieve_endpoint(req: RetrievalRequest):

    if not (req.tenant_id and req.kb_id):
        raise HTTPException(
            status_code=400, 
            detail="Both 'tenant_id' and 'kb_id' are required for RAGFlow retrieval"
        )
    
    logger.info(f"Starting RAGFlow retrieval for {req.tenant_id}::{req.kb_id}")
    
    try:
        # Enhanced RAGFlow approach: query enhancement + fast vector search + advanced ranking
        result = await enhanced_ragflow_retrieval.retrieve(req)
        logger.info(f"Retrieval successful: {result.retrieval_stats.search_method}")
        return result
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Retrieval failed",
                "message": str(e),
                "solution": "Please ensure documents are processed with /v1/ragflow/process first"
            }
        )
