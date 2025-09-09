import re
import logging
from typing import List, Tuple

logger = logging.getLogger("universal_query_enhancer")


class MinimalQueryEnhancer:

    
    def __init__(self):
        # Minimal approach - no hardcoded stop words or regex patterns
        pass
    
    def normalize_text(self, text: str) -> str:
        """Universal text normalization - like RAGFlow"""
        # Convert full-width to half-width characters (for Chinese/Japanese)
        text = text.replace('　', ' ')  # Full-width space
        
        # Normalize common punctuation
        text = re.sub(r'[，。？！；：''""【】（）、]', ' ', text)  # Chinese punctuation
        text = re.sub(r'[,\.\?!;:\'""\(\)\[\]{}]', ' ', text)  # English punctuation
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase but preserve non-ASCII
        text = text.lower()
        
        return text
    
    def remove_question_words(self, text: str) -> str:
        """Minimal processing - just basic cleanup"""
        # Only basic whitespace normalization
        text = ' '.join(text.split())
        return text.strip()
    
    
    def enhance_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Minimal query enhancement - Raptor-service style
        Returns: (enhanced_query, simple_keywords)
        """
        logger.info(f"🔍 Minimal query processing: '{query}'")
        
        # Simple normalization only
        normalized_query = self.normalize_text(query)
        
        # Simple keyword extraction - just meaningful words
        words = normalized_query.split()
        keywords = [word for word in words if len(word) > 2 and not word.isdigit()]
        
        logger.info(f"✅ Processing complete: '{normalized_query}'")
        logger.debug(f"   Keywords: {keywords[:5]}")  # Max 5 keywords
        
        return normalized_query, keywords[:5]
    
    def calculate_text_similarity(self, query_tokens: List[str], content_tokens: List[str]) -> float:
        """Universal text similarity - no hardcoded terms"""
        if not query_tokens or not content_tokens:
            return 0.0
        
        # Convert to sets for intersection
        query_set = set(query_tokens)
        content_set = set(content_tokens)
        
        # Calculate weighted intersection
        matches = len(query_set.intersection(content_set))
        
        # Normalize by query length (RAGFlow style)
        similarity = matches / len(query_tokens) if query_tokens else 0.0
        
        return min(similarity, 1.0)


# Global instance  
universal_query_enhancer = MinimalQueryEnhancer()

