import logging
import re
from typing import List, Tuple, Optional, Callable
from models.document import DocumentChunk
from utils import token_count
from .chunk_clean import clean_content, force_split_large_text
from .chunk_helpers import ChunkOptimizer, ChunkStatistics, TokenCacheManager

logger = logging.getLogger("hierarchical_chunker")

# RAGFlow-style bullet patterns for different hierarchy levels
BULLET_PATTERNS = [
    [  # Pattern set 0: Roman numerals and basic
        r"^[IVX]+[\.\)]\s*",
        r"^[0-9]+[\.\)]\s*",  
        r"^[a-z][\.\)]\s*"
    ],
    [  # Pattern set 1: Chinese/numbered
        r"^第[一二三四五六七八九十]+[章节条]\s*",
        r"^[0-9]+[\.\)]\s*",
        r"^[一二三四五六七八九十]+[\.\)]\s*"
    ],
    [  # Pattern set 2: Markdown-style
        r"^#{1,3}\s+",  # H1-H3 headers
        r"^[-\*\+]\s+",  # Bullet points
        r"^[0-9]+\.\s+"  # Numbered lists
    ]
]

class HierarchicalChunker:
    """RAGFlow-optimized hierarchical chunker for structured documents"""
    
    def __init__(self, chunk_size: int = None, delimiter: str = None, 
                 overlap_percent: int = None, pattern_set: int = None):
        # Get defaults from config
        from config.chunking import get_chunking_settings
        from config.cache import get_cache_settings
        
        config = get_chunking_settings()
        cache_config = get_cache_settings()
        
        # Basic settings
        self.chunk_size = chunk_size or config.chunk_size
        self.delimiter = delimiter or config.chunk_delimiter
        self.overlap_percent = max(0, min(50, overlap_percent or config.chunk_overlap_percent))
        self.pattern_set = pattern_set or config.hierarchical_pattern_set
        self.min_chunk_tokens = config.min_chunk_tokens
        
        # Performance optimizations flags
        self.token_aware_overlap = config.token_aware_overlap
        self.pattern_cache_enabled = config.pattern_cache_enabled
        self.batch_token_counting = config.batch_token_counting
        
        # Pattern setup with caching (using helpers)
        self.bullet_patterns = BULLET_PATTERNS[self.pattern_set] if self.pattern_set < len(BULLET_PATTERNS) else BULLET_PATTERNS[2]
        self._compiled_patterns = ChunkOptimizer.setup_pattern_cache(self.bullet_patterns, self.pattern_cache_enabled)
        
        # Token counting cache (using helpers)
        self.token_cache_enabled = cache_config.token_cache_enabled
        self._token_count = TokenCacheManager.create_cached_token_counter(
            self.token_cache_enabled, 
            cache_config.token_cache_max_size
        )
        
        logger.info(f"HierarchicalChunker optimized: size={self.chunk_size}, patterns={len(self._compiled_patterns)}, token_cache={self.token_cache_enabled}, overlap_mode={'token-aware' if self.token_aware_overlap else 'char-based'}")
    
    async def chunk_only(self, text: str, doc_id: str, max_chunk_tokens: Optional[int] = None, 
                         progress_callback: Optional[Callable[[str], None]] = None) -> List[DocumentChunk]:
        """
        Hierarchical chunking - structure-aware splitting
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            max_chunk_tokens: Override chunk size
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Use configured max_tokens if not provided
        if max_chunk_tokens is None:
            max_chunk_tokens = self.chunk_size
        
        # Step 1: Clean content
        if progress_callback:
            progress_callback(f"Starting content cleaning for document {doc_id}")
        cleaned_text = clean_content(text)
        
        # Step 2: Detect structure and create sections
        if progress_callback:
            progress_callback("Analyzing document structure and hierarchy")
        sections = self._detect_sections(cleaned_text)
        
        if progress_callback:
            progress_callback(f"Detected {len(sections)} structured sections")
        
        # Step 3: Apply hierarchical merge
        if progress_callback:
            progress_callback(f"Starting hierarchical chunking with {max_chunk_tokens} token limit")
        chunk_texts = self._hierarchical_merge(sections, max_chunk_tokens)
        
        # Step 4: Convert to DocumentChunk objects with optimized token counting
        if progress_callback:
            progress_callback(f"Converting {len(chunk_texts)} text chunks to DocumentChunk objects")
        chunks = []
        if self.batch_token_counting and chunk_texts:
            # Batch token counting for performance (using helpers)
            stripped_texts = [text.strip() for text in chunk_texts if text.strip()]
            if progress_callback:
                progress_callback("Performing batch token counting optimization")
            token_counts = ChunkOptimizer.batch_token_count(stripped_texts, self.token_cache_enabled, self._token_count)
            
            text_index = 0
            for i, chunk_text in enumerate(chunk_texts):
                if chunk_text.strip():
                    actual_token_count = token_counts[text_index]
                    chunk = DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{i}",
                        doc_id=doc_id,
                        content=chunk_text.strip(),
                        chunk_index=i,
                        metadata={
                            "chunking_method": "hierarchical_structure_aware_optimized",
                            "chunk_size": self.chunk_size,
                            "actual_tokens": actual_token_count,
                            "overlap_percent": self.overlap_percent,
                            "overlap_mode": "token-aware" if self.token_aware_overlap else "char-based",
                            "pattern_set": self.pattern_set,
                            "pattern_cache_enabled": self.pattern_cache_enabled,
                            "structure_detected": True
                        }
                    )
                    chunks.append(chunk)
                    text_index += 1
        else:
            # Individual processing (fallback)
            for i, chunk_text in enumerate(chunk_texts):
                if chunk_text.strip():
                    actual_token_count = self._token_count(chunk_text.strip())
                    chunk = DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{i}",
                        doc_id=doc_id,
                        content=chunk_text.strip(),
                        chunk_index=i,
                        metadata={
                            "chunking_method": "hierarchical_structure_aware_optimized",
                            "chunk_size": self.chunk_size,
                            "actual_tokens": actual_token_count,
                            "overlap_percent": self.overlap_percent,
                            "overlap_mode": "token-aware" if self.token_aware_overlap else "char-based",
                            "pattern_set": self.pattern_set,
                            "pattern_cache_enabled": self.pattern_cache_enabled,
                            "structure_detected": True
                        }
                    )
                    chunks.append(chunk)
        
        # Step 5: Finalization
        if progress_callback:
            progress_callback(f"✅ Hierarchical chunking complete: Generated {len(chunks)} chunks from {len(text)} characters")
        
        logger.info(f"✅ Hierarchical chunking complete: {len(chunks)} chunks from {len(text)} chars")
        return chunks
    
    def _detect_sections(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Detect sections with hierarchy levels
        
        Returns:
            List of (text, layout_type, level) tuples
        """
        lines = text.split('\n')
        sections = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect hierarchy level
            level, layout_type = self._detect_hierarchy_level(line)
            sections.append((line, layout_type, level))
        
        return sections
    
    def _detect_hierarchy_level(self, line: str) -> Tuple[int, str]:
        """
        Detect hierarchy level of a line (OPTIMIZED with pattern caching)
        
        Returns:
            (level, layout_type) tuple
        """
        # PERFORMANCE BOOST: Use compiled patterns if enabled
        if self.pattern_cache_enabled and self._compiled_patterns:
            for level, compiled_pattern in enumerate(self._compiled_patterns):
                if compiled_pattern.match(line):
                    return level, "bullet"
        else:
            # Fallback to original method
            for level, pattern in enumerate(self.bullet_patterns):
                if re.match(pattern, line):
                    return level, "bullet"
        
        # Check markdown headers
        if line.startswith('#'):
            header_level = len(line) - len(line.lstrip('#'))
            return min(header_level - 1, 2), "header"
        
        # Check if looks like title (short, capitalized)
        if (len(line) < 100 and 
            (line.isupper() or line.istitle()) and 
            not line.endswith('.') and
            not re.search(r'\d+', line)):
            return 1, "title"
        
        # Regular content
        return 3, "content"
    
    def _hierarchical_merge(self, sections: List[Tuple[str, str, int]], max_tokens: int) -> List[str]:
        """
        Merge sections hierarchically based on structure
        
        Args:
            sections: List of (text, layout_type, level) tuples
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of chunk texts
        """
        if not sections:
            return []
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        min_chunk_tokens = 50  # Minimum viable chunk size
        
        for text, layout_type, level in sections:
            # Use cached token counting for performance
            text_tokens = self._token_count(text)
            
            # Skip very small content (unless it's a header)
            if text_tokens < 5 and layout_type != "header":
                continue
            
            # If this is a high-level heading/title, consider starting new chunk (with overlap)
            if level <= 1 and current_chunk and current_tokens >= min_chunk_tokens:
                chunks.append(current_chunk.strip())
                
                # Apply RAGFlow-style token-aware overlap
                if self.overlap_percent > 0:
                    if self.token_aware_overlap:
                        # RAGFlow approach: token-based overlap calculation (using helpers)
                        overlap_threshold = int(current_tokens * (100 - self.overlap_percent) / 100)
                        overlap_text = ChunkOptimizer.get_text_by_token_count(current_chunk, current_tokens - overlap_threshold)
                        current_chunk = overlap_text
                        current_tokens = self._token_count(current_chunk)
                    else:
                        # Original character-based approach (fallback)
                        overlap_size = int(len(current_chunk) * self.overlap_percent / 100)
                        overlap_text = current_chunk[-overlap_size:] if overlap_size > 0 else ""
                        current_chunk = overlap_text
                        current_tokens = self._token_count(current_chunk)
                else:
                    current_chunk = ""
                    current_tokens = 0
            
            # Check if adding this text would exceed limit (RAGFlow-style logic)
            if current_tokens + text_tokens > max_tokens and current_tokens >= min_chunk_tokens:
                # Save current chunk and start new one with overlap
                chunks.append(current_chunk.strip())
                
                # Apply optimized overlap logic
                if self.overlap_percent > 0 and current_chunk:
                    if self.token_aware_overlap:
                        # RAGFlow token-aware overlap (using helpers)
                        overlap_threshold = int(current_tokens * (100 - self.overlap_percent) / 100)
                        overlap_text = ChunkOptimizer.get_text_by_token_count(current_chunk, current_tokens - overlap_threshold)
                        current_chunk = overlap_text + "\n" + text
                    else:
                        # Character-based fallback
                        overlap_size = int(len(current_chunk) * self.overlap_percent / 100)
                        overlap_text = current_chunk[-overlap_size:] if overlap_size > 0 else ""
                        current_chunk = overlap_text + "\n" + text
                    
                    current_tokens = self._token_count(current_chunk)  # Recalculate total tokens
                else:
                    current_chunk = text
                    current_tokens = text_tokens
                
                # If new chunk is still too large, force split it
                if current_tokens > max_tokens:
                    force_split_parts = force_split_large_text(current_chunk, max_tokens)
                    if len(force_split_parts) > 1:
                        # Add all but last part to chunks
                        chunks.extend(force_split_parts[:-1])
                        current_chunk = force_split_parts[-1]
                        current_tokens = self._token_count(current_chunk)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n" + text
                else:
                    current_chunk = text
                current_tokens += text_tokens
        
        # Add final chunk (with force split if needed)
        if current_chunk:
            if current_tokens > max_tokens:
                # Force split final chunk if too large
                force_split_parts = force_split_large_text(current_chunk, max_tokens)
                chunks.extend(force_split_parts)
            elif current_tokens >= min_chunk_tokens:
                chunks.append(current_chunk.strip())
            elif chunks:
                # Merge small final chunk with last chunk - but check size limit
                test_merge = chunks[-1] + "\n" + current_chunk
                if token_count(test_merge) <= max_tokens:
                    chunks[-1] = test_merge
                else:
                    # Can't merge safely, force split the combination
                    force_split_parts = force_split_large_text(test_merge, max_tokens)
                    chunks[-1] = force_split_parts[0]
                    chunks.extend(force_split_parts[1:])
            else:
                # If it's the only chunk, keep it anyway
                chunks.append(current_chunk.strip())
        
        # FINAL SAFETY CHECK: Force split any remaining oversized chunks
        final_chunks = []
        if self.batch_token_counting:
            # Batch token counting for performance (using helpers)
            chunk_tokens = ChunkOptimizer.batch_token_count(chunks, self.token_cache_enabled, self._token_count)
            for chunk, tokens in zip(chunks, chunk_tokens):
                if tokens > max_tokens:
                    split_parts = force_split_large_text(chunk, max_tokens)
                    final_chunks.extend(split_parts)
                else:
                    final_chunks.append(chunk)
        else:
            # Individual token counting (fallback)
            for chunk in chunks:
                chunk_tokens = self._token_count(chunk)
                if chunk_tokens > max_tokens:
                    split_parts = force_split_large_text(chunk, max_tokens)
                    final_chunks.extend(split_parts)
                else:
                    final_chunks.append(chunk)
        
        return final_chunks

    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> dict:
        """Get hierarchical chunking statistics (using helpers)"""
        return ChunkStatistics.get_chunking_stats(
            chunks, self.chunk_size, self.pattern_set, self.overlap_percent
        )
