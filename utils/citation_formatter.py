from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger("citation_formatter")

def format_context_passages_for_frontend(retrieved_nodes: List = None, user_query: str = None, thinking_content: str = None, model_answer: str = None) -> List[Dict[str, Any]]:
    """
    Generate context_passages format for frontend display
    Clean, simple approach - let LLM handle citation intelligence via prompt
    """
    if not retrieved_nodes:
        logger.info("📋 No retrieved nodes provided")
        return []
        
    logger.info(f"📋 Processing {len(retrieved_nodes)} retrieved nodes")
    logger.info(f"📋 First node preview: {getattr(retrieved_nodes[0], 'content', '')[:100]}..." if retrieved_nodes else "No nodes")
    if thinking_content:
        logger.info(f"🧠 Thinking content available: {len(thinking_content)} chars")
    
    context_passages = []
    
    # Extract chunks that model actually referenced in thinking
    referenced_chunks = _extract_thinking_referenced_chunks(thinking_content) if thinking_content else []
    logger.info(f"🧠 Model referenced chunks in thinking: {referenced_chunks}")
    
    # Prioritize chunks mentioned in thinking, then top retrieval results
    prioritized_chunks = []
    
    # First, add chunks specifically mentioned in thinking
    for chunk in retrieved_nodes:
        meta = getattr(chunk, 'metadata', getattr(chunk, 'meta', {}))
        chunk_index = meta.get('chunk_index', 0)
        if str(chunk_index) in referenced_chunks:
            prioritized_chunks.append(chunk)
            logger.info(f"✅ Prioritizing chunk_{chunk_index} (mentioned in thinking)")
    
    # Add other top chunks dynamically (no fixed limit)
    max_chunks = min(len(retrieved_nodes), len(referenced_chunks) + 1) if referenced_chunks else 3
    for chunk in retrieved_nodes:
        if chunk not in prioritized_chunks and len(prioritized_chunks) < max_chunks:
            prioritized_chunks.append(chunk)
    
    # Process all prioritized chunks
    for chunk in prioritized_chunks:
        content = getattr(chunk, 'content', '')
        if not content:
            continue
            
        # Extract metadata - try both meta and metadata attributes
        meta = getattr(chunk, 'metadata', getattr(chunk, 'meta', {}))
        doc_id = meta.get('doc_id', 'unknown')
        chunk_index = meta.get('chunk_index', 0)
        source_id = f"{doc_id}_chunk_{chunk_index}"
        
        # Generate relevant excerpt
        excerpt = _extract_best_excerpt(content, thinking_content, user_query, 150, model_answer)
        
        passage = {
            "content": content,
            "relevant_excerpt": excerpt,
            "similarity_score": getattr(chunk, 'similarity_score', 0.0),
            "source_id": source_id,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "owner_type": "standard_retrieval"
        }
        
        context_passages.append(passage)
        logger.info(f"✅ Added passage {source_id}: {excerpt[:60]}...")
    
    logger.info(f"🎯 Generated {len(context_passages)} context passages")
    return context_passages

def _extract_best_excerpt(content: str, thinking_content: str = None, user_query: str = None, max_length: int = 150, model_answer: str = None) -> str:
    """
    Extract the most relevant excerpt from content
    Priority: thinking-guided > query-aware > simple
    """
    if not content:
        return ""
    
    # If content is short, return as-is
    if len(content) <= max_length:
        return content
    
    # Try thinking-guided approach first
    if thinking_content:
        from .thinking_helper import extract_thinking_guided_excerpt
        thinking_excerpt = extract_thinking_guided_excerpt(content, thinking_content, 150, model_answer)
        if thinking_excerpt:
            return thinking_excerpt
    
    # Fallback to simple sentence extraction
    return _extract_simple_excerpt(content, max_length)

def _extract_simple_excerpt(content: str, max_length: int = 150) -> str:
    """
    Simple but effective excerpt extraction
    Returns first meaningful sentences, avoiding generic intros
    """
    if not content:
        return ""
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return _truncate_at_word_boundary(content, max_length)
    
    # Generic patterns to avoid
    generic_patterns = [
        'commonly abbreviated', 'also known as', 'is a type of', 
        'is an annual', 'is the premier', 'hosted by'
    ]
    
    selected_sentences = []
    total_length = 0
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Skip generic introductory sentences
        is_generic = any(pattern in sentence_lower for pattern in generic_patterns)
        
        if not is_generic and total_length + len(sentence) + 2 <= max_length:
            selected_sentences.append(sentence)
            total_length += len(sentence) + 2
            
            # Stop after 2 sentences for readability
            if len(selected_sentences) >= 2:
                break
    
    # Return selected sentences or fallback
    if selected_sentences:
        result = '. '.join(selected_sentences)
        if not result.endswith(('.', '!', '?')):
            result += '.'
        return result
    
    # No good sentences found, use first sentence anyway
    if sentences:
        first_sentence = sentences[0]
        if len(first_sentence) <= max_length:
            return first_sentence + ('.' if not first_sentence.endswith(('.', '!', '?')) else '')
    
    # Last resort: truncate
    return _truncate_at_word_boundary(content, max_length)

def _extract_thinking_referenced_chunks(thinking_content: str) -> list:
    """Extract chunk numbers that model referenced in thinking"""
    if not thinking_content:
        return []
    
    import re
    # Extract chunk numbers from patterns like "chunk_24", "chunk 21", etc.
    chunk_matches = re.findall(r'chunk[_\s]*(\d+)', thinking_content.lower())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chunks = []
    for chunk_num in chunk_matches:
        if chunk_num not in seen:
            seen.add(chunk_num)
            unique_chunks.append(chunk_num)
    
    return unique_chunks

def _truncate_at_word_boundary(text: str, max_length: int) -> str:
    """Truncate text at word boundary"""
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.7:  # Don't cut too much
        truncated = truncated[:last_space]
    
    return truncated + "..."