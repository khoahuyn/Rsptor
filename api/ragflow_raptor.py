import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from services.document import DocumentProcessor, RaptorBuilder
from services.retrieval import enhanced_ragflow_retrieval
from models import DocumentProcessSummary, RetrievalRequest, RetrievalResponse
from utils.progress import raptor_progress_context

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/ragflow", tags=["RAGFlow"])


@router.post("/process", response_model=DocumentProcessSummary)
async def process_document_ragflow(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    kb_id: str = Form(...),
    enable_raptor: bool = Form(True),
    max_clusters: int = Form(64),
    threshold: float = Form(0.1),
    random_seed: int = Form(42)
):

    try:
        logger.info(f"🚀 RAGFlow processing START: {file.filename} (tenant: {tenant_id}, kb: {kb_id})")
        
        # Validation
        if not file.filename:
            logger.error(f"❌ Filename validation failed for upload")
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # File extension validation - only allow .md and .markdown
        filename_lower = file.filename.lower()
        if not (filename_lower.endswith('.md') or filename_lower.endswith('.markdown')):
            raise HTTPException(
                status_code=400, 
                detail="Only .md and .markdown files are supported"
            )
        
        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # RAGFlow single-pass processing: Parse → Chunk → Embed → Save ALL
        document_processor = DocumentProcessor()
        result = await document_processor.process_document(
            file_content=file_content,
            filename=file.filename,
            tenant_id=tenant_id,
            kb_id=kb_id
        )
        
        # Optional RAPTOR stage (separate for performance)  
        # Note: If result is returned, processing was successful (otherwise exception would be raised)
        if enable_raptor:
            logger.info(f"🌳 Starting RAPTOR tree building for {file.filename} → {result.doc_id}")
            
            try:
                # Create progress tracking context (non-blocking, server-side only)
                with raptor_progress_context(f"Building RAPTOR Tree for {file.filename} ({result.doc_id[:8]}...)", show_console=True) as progress:
                    progress.update(completed=10, description="🌳 Initializing RAPTOR builder...")
                    
                    raptor_builder = RaptorBuilder()
                    progress.update(completed=20, description="🌳 Loading chunks and embeddings...")
                    
                    raptor_result = await raptor_builder.build_raptor_tree(
                        doc_id=result.doc_id,
                        tenant_id=tenant_id,
                        kb_id=kb_id,
                        max_clusters=max_clusters,
                        threshold=threshold,
                        random_seed=random_seed,
                        progress_callback=lambda msg: progress.update(advance=5, description=f"🌳 {msg}")
                    )
                    
                    progress.update(completed=90, description="🌳 Finalizing RAPTOR results...")
                    
                    # Update result with RAPTOR success (create new immutable object)
                    from datetime import datetime
                    final_result = DocumentProcessSummary(
                        doc_id=result.doc_id,
                        filename=result.filename,  # ✅ Required field
                        total_chunks=raptor_result.get('total_chunks', result.total_chunks),  # ✅ Updated total
                        total_embeddings=result.total_embeddings,  # ✅ Required field
                        processing_time=result.processing_time + raptor_result.get('processing_time', 0),  # ✅ Required field
                        status=result.status,  # ✅ Required field
                        tenant_id=result.tenant_id,
                        kb_id=result.kb_id,
                        original_chunks=result.original_chunks,
                        summary_chunks=raptor_result.get('summary_chunks', 0),  # ✅ RAPTOR summaries only
                        raptor_enabled=True,  # ✅ RAPTOR was successfully enabled
                        raptor_summary_count=raptor_result.get('summary_chunks', 0),  # ✅ Same as summary_chunks
                        raptor_tree_levels=raptor_result.get('tree_levels', 0),  # ✅ Tree depth info
                        created_at=datetime.now().isoformat()  # ✅ Set timestamp
                    )
                    
                    progress.update(completed=100, description="✅ RAPTOR tree building completed!")
                    
                    logger.info(f"✅ RAPTOR completed: {raptor_result.get('summary_chunks', 0)} summary chunks created")
                    logger.info(f"🌳 RAPTOR tree info: {raptor_result}")
                
                return final_result
                
            except Exception as raptor_error:
                logger.warning(f"⚠️ RAPTOR tree building failed (document processing still successful): {raptor_error}")
                # Note: Document processing succeeded, RAPTOR is optional enhancement
        
        logger.info(f"🎉 RAGFlow processing complete: {file.filename} → {result.doc_id}")
        return result  # Return original result if RAPTOR failed or not enabled
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ RAGFlow processing failed: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RAGFlow processing failed", 
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
