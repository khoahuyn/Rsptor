import hashlib
import time
from typing import Dict, Any
from models import DocumentProcessSummary


def calculate_content_hash(content: bytes) -> str:
    """Calculate SHA256 hash for document content deduplication (RAGFlow approach)"""
    return hashlib.sha256(content).hexdigest()


def create_document_summary(
    doc_id: str, 
    tenant_id: str, 
    kb_id: str, 
    filename: str,
    total_chunks: int, 
    total_embeddings: int,
    start_time: float,
    raptor_enabled: bool = False
) -> DocumentProcessSummary:
    """Create DocumentProcessSummary with consistent structure (RAGFlow helper)"""
    from datetime import datetime
    processing_time_value = time.time() - start_time
    return DocumentProcessSummary(
        doc_id=doc_id,
        filename=filename,
        total_chunks=total_chunks,
        total_embeddings=total_embeddings,
        processing_time=processing_time_value,
        status="completed",
        # RAPTOR compatibility fields
        original_chunks=total_chunks,  # Set original chunks = total_chunks initially
        summary_chunks=0,  # No RAPTOR summaries in document processing stage
        raptor_enabled=raptor_enabled,
        tenant_id=tenant_id,
        kb_id=kb_id,
        created_at=datetime.now().isoformat()  # ✅ Set timestamp
    )


def prepare_document_data(
    doc_id: str,
    tenant_id: str, 
    kb_id: str,
    filename: str,
    content_hash: str,
    total_tokens: int,
    chunk_count: int,
    chunking_settings
) -> Dict[str, Any]:
    """Prepare document data for database storage (RAGFlow minimalist approach)"""
    return {
        "doc_id": doc_id,
        "tenant_id": tenant_id,
        "kb_id": kb_id,
        "filename": filename,
        "checksum": content_hash,
        "processing_stats": {
            # Essential stats only
            "chunk_count": chunk_count,
            "token_count": total_tokens,
            # Common chunking settings (moved from chunk metadata)
            "chunk_method": "hierarchical",
            "chunk_size": chunking_settings.chunk_size,
            "overlap_percent": chunking_settings.chunk_overlap_percent,
            "pattern_set": getattr(chunking_settings, 'hierarchical_pattern_set', 2)
        }
    }


def prepare_chunk_data_list(chunks, token_count_fn):
    """Prepare chunk data for document creation (RAGFlow approach)"""
    return [
        {
            "chunk_id": chunk.chunk_id,  # Changed from "id"
            "content": chunk.content,
            "chunk_index": i,
            "token_count": token_count_fn(chunk.content),
            "meta": chunk.metadata  # Changed from "metadata"
        }
        for i, chunk in enumerate(chunks)
    ]


