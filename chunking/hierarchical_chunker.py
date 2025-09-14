import logging
import re
from typing import List, Tuple, Optional, Callable
from models.document import DocumentChunk
from utils import token_count
from .chunk_clean import clean_content, force_split_large_text
from .chunk_helpers import ChunkOptimizer, ChunkStatistics, TokenCacheManager

logger = logging.getLogger("hierarchical_chunker")

def ragflow_get_delimiters(delimiters: str):
    """
    RAGFlow-inspired dynamic delimiter parsing
    Parse delimiter string like "\n。；！？" into regex pattern
    Support backtick patterns for complex delimiters
    """
    dels = []
    s = 0
    # Handle backtick patterns like `\n\n`
    for m in re.finditer(r"`([^`]+)`", delimiters, re.I):
        f, t = m.span()
        dels.append(m.group(1))
        dels.extend(list(delimiters[s: f]))
        s = t
    if s < len(delimiters):
        dels.extend(list(delimiters[s:]))

    # Sort by length (longest first) for proper matching
    dels.sort(key=lambda x: -len(x))
    dels = [re.escape(d) for d in dels if d]
    dels = [d for d in dels if d]
    dels_pattern = "|".join(dels)
    return dels_pattern

def ragflow_smart_split(text: str, delimiter_pattern: str, max_tokens: int, token_counter):
    """
    RAGFlow-inspired smart text splitting that respects semantic boundaries
    """
    if token_counter(text) <= max_tokens:
        return [text]
    
    # Split by delimiters while preserving them
    parts = re.split(f"({delimiter_pattern})", text, flags=re.DOTALL)
    result = []
    current = ""
    
    for part in parts:
        if not part:
            continue
            
        # Skip if part is just a delimiter
        if re.match(f"^{delimiter_pattern}$", part):
            continue
            
        # Check if adding this part would exceed token limit
        test_text = current + part
        if current and token_counter(test_text) > max_tokens:
            # Save current chunk and start new one
            if current.strip():
                result.append(current.strip())
            current = part
        else:
            current += part
    
    # Add final chunk
    if current.strip():
        result.append(current.strip())
    
    return result

def get_text_by_token_count(text: str, target_tokens: int, token_counter) -> str:
    """
    Extract text up to target token count, word-boundary aware
    """
    if not text or target_tokens <= 0:
        return ""
    
    words = text.split()
    if not words:
        return text[:target_tokens * 4]  # Rough character estimate
    
    # Binary search for optimal word count
    left, right = 0, len(words)
    result = ""
    
    while left <= right:
        mid = (left + right) // 2
        candidate = " ".join(words[:mid])
        tokens = token_counter(candidate)
        
        if tokens <= target_tokens:
            result = candidate
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def get_token_overlap(text: str, overlap_tokens: int, token_counter, from_end: bool = True) -> str:
    """
    Extract token-based overlap from text, maintaining word boundaries
    """
    if not text or overlap_tokens <= 0:
        return ""
    
    words = text.split()
    if not words:
        return ""
    
    if from_end:
        # Get overlap from end of text
        for i in range(len(words)):
            candidate = " ".join(words[i:])
            if token_counter(candidate) <= overlap_tokens:
                return candidate
        return " ".join(words[-1:])  # At least last word
    else:
        # Get overlap from beginning of text  
        return get_text_by_token_count(text, overlap_tokens, token_counter)

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
        
        # RAGFlow-inspired settings
        self.ragflow_delimiters = "\n。；！？.!?\n\n"  # Smart semantic boundaries
        self.ragflow_delimiter_pattern = ragflow_get_delimiters(self.ragflow_delimiters)
        
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
        Optimized RAGFlow-inspired hierarchical merge for better token efficiency
        
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
        
        # Optimized thresholds for better efficiency
        target_utilization = 0.82  # Aim for 82% of max_tokens (420 for 512)
        target_tokens = int(max_tokens * target_utilization)
        overlap_tokens = int(max_tokens * self.overlap_percent / 100)
        
        def finalize_current_chunk():
            """Finalize and add current chunk with token-based overlap"""
            nonlocal chunks, current_chunk, current_tokens
            
            if not current_chunk.strip():
                return
            
            # Add current chunk to results
            chunks.append(current_chunk.strip())
            
            # Generate token-based overlap for next chunk
            if overlap_tokens > 0:
                overlap_text = get_token_overlap(current_chunk, overlap_tokens, self._token_count, from_end=True)
                current_chunk = overlap_text
                current_tokens = self._token_count(current_chunk)
            else:
                current_chunk = ""
                current_tokens = 0
        
        def should_start_new_chunk(text_tokens: int, level: int) -> bool:
            """Determine if we should start a new chunk"""
            # Force new chunk for major headers
            if level <= 1 and current_tokens >= self.min_chunk_tokens:
                return True
            
            # Start new chunk if adding would exceed target utilization
            would_exceed_target = current_tokens + text_tokens > target_tokens
            meets_minimum = current_tokens >= self.min_chunk_tokens
            
            return would_exceed_target and meets_minimum
        
        # Process sections with optimized chunking strategy
        for text, layout_type, level in sections:
            text_tokens = self._token_count(text)
            
            # Skip tiny content (unless it's important structure)
            if text_tokens < 5 and layout_type not in ["header", "title"]:
                continue
            
            # Handle oversized sections first
            if text_tokens > max_tokens:
                # Finalize current chunk before handling large section
                if current_chunk:
                    finalize_current_chunk()
                
                # Split oversized section using RAGFlow method
                split_parts = ragflow_smart_split(text, self.ragflow_delimiter_pattern, max_tokens, self._token_count)
                
                # Process split parts
                for part in split_parts:
                    part_tokens = self._token_count(part)
                    
                    if should_start_new_chunk(part_tokens, level):
                        finalize_current_chunk()
                    
                    # Add part to current chunk
                    separator = "\n" if current_chunk and part else ""
                    current_chunk += separator + part
                    current_tokens += part_tokens
                
                continue
            
            # Regular section processing
            if should_start_new_chunk(text_tokens, level):
                finalize_current_chunk()
            
            # Add section to current chunk
            separator = "\n" if current_chunk and text else ""
            current_chunk += separator + text
            current_tokens += text_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Post-processing: optimize small chunks
        final_chunks = []
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            chunk_tokens = self._token_count(chunk)
            
            # Handle oversized chunks (rare but possible)
            if chunk_tokens > max_tokens:
                split_parts = force_split_large_text(chunk, max_tokens)
                final_chunks.extend(split_parts)
                continue
            
            # Handle small chunks - try to merge with previous
            if (chunk_tokens < self.min_chunk_tokens and final_chunks and 
                chunk_tokens > 5):  # Don't merge extremely tiny chunks
                
                test_merge = final_chunks[-1] + "\n" + chunk
                merge_tokens = self._token_count(test_merge)
                
                if merge_tokens <= max_tokens:
                    final_chunks[-1] = test_merge
                    continue
            
            # Add chunk as-is
            final_chunks.append(chunk)
        
        # Final filter: remove any remaining tiny chunks
        filtered_chunks = []
        for chunk in final_chunks:
            chunk_tokens = self._token_count(chunk)
            
            # Keep chunks that meet minimum or are the only chunk
            if chunk_tokens >= self.min_chunk_tokens or len(final_chunks) == 1:
                filtered_chunks.append(chunk)
            elif filtered_chunks:  # Try one more merge attempt
                test_merge = filtered_chunks[-1] + "\n" + chunk
                if self._token_count(test_merge) <= max_tokens:
                    filtered_chunks[-1] = test_merge
                # Otherwise discard tiny chunk
        
        return filtered_chunks if filtered_chunks else final_chunks

    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> dict:
        """Get hierarchical chunking statistics (using helpers)"""
        return ChunkStatistics.get_chunking_stats(
            chunks, self.chunk_size, self.pattern_set, self.overlap_percent
        )
