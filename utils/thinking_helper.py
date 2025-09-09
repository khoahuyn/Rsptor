import re
import logging

logger = logging.getLogger("thinking_helper")

def extract_thinking_guided_excerpt(content: str, thinking_content: str = None, max_length: int = 150, model_answer: str = None) -> str:
    """
    Extract excerpt - let model drive the selection completely
    """
    if not thinking_content or not content:
        return ""
    
    # Model-driven approach: Find what model specifically described
    model_described = _extract_model_described_content(thinking_content, content, model_answer)
    if model_described:
        logger.info(f"🎯 Using model description: {model_described[:60]}...")
        return model_described
    
    # Simple fallback: first meaningful sentence  
    sentences = re.split(r'(?<=[.!?])\s+', content)
    meaningful_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 20]
    
    if meaningful_sentences:
        logger.info(f"📝 Using first meaningful sentence: {meaningful_sentences[0][:60]}...")
        return meaningful_sentences[0]
    
    logger.info("❌ No suitable excerpt found")
    return ""

def _extract_model_described_content(thinking_content: str, content: str, model_answer: str = None) -> str:
    """Find content that model specifically described in its thinking"""
    if not thinking_content or not content:
        return ""
    
    # Split into sentences
    content_sentences = re.split(r'(?<=[.!?])\s+', content)  
    content_sentences = [s.strip() for s in content_sentences if s.strip() and len(s.strip()) >= 15]
    
    thinking_sentences = re.split(r'(?<=[.!?])\s+', thinking_content)
    thinking_sentences = [s.strip() for s in thinking_sentences if s.strip()]
    
    # Collect all candidate sentences with scores
    candidates = []
    
    # Find thinking sentences that describe specific content
    for thinking_sent in thinking_sentences:
        thinking_lower = thinking_sent.lower()
        
        # Look for descriptive language
        descriptive_words = ['mentions', 'states', 'says', 'shows', 'indicates', 'contains', 'has']
        if not any(word in thinking_lower for word in descriptive_words):
            continue
        
        # Extract meaningful words from this thinking sentence
        thinking_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', thinking_sent.lower()))
        
        # Remove common words dynamically
        common = {'that', 'this', 'they', 'have', 'been', 'were', 'will', 'with', 'from'}
        thinking_words = thinking_words - common - set(descriptive_words)
        
        if len(thinking_words) < 2:
            continue
        
        # Find content sentences that match this thinking
        for content_sent in content_sentences:
            content_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', content_sent.lower()))
            thinking_overlap = len(thinking_words.intersection(content_words))
            
            if thinking_overlap >= 2:
                # Additional scoring: how well does content match model's answer?
                answer_score = 0
                if model_answer:
                    answer_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', model_answer.lower()))
                    answer_overlap = len(answer_words.intersection(content_words))
                    answer_score = answer_overlap
                
                total_score = thinking_overlap * 2 + answer_score  # Weight thinking higher
                candidates.append((content_sent, total_score, thinking_sent[:40]))
    
    if candidates:
        # Sort by score and return best
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sent, best_score, thinking_desc = candidates[0]
        
        logger.info(f"🎯 Model described: '{thinking_desc}...' → best match (score: {best_score})")
        return best_sent
    
    return ""