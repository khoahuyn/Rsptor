import re
import logging
import math
from typing import List, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger("universal_query_enhancer")


class UniversalQueryEnhancer:

    
    def __init__(self):
        # Universal stop words (language-agnostic patterns)
        self.stop_words = {
            # Vietnamese question words
            "là", "gì", "như", "thế", "nào", "ở", "đâu", "khi", "nào", 
            "bao", "nhiêu", "tại", "sao", "vì", "có", "không", "được",
            "rồi", "chưa", "thì", "mà", "nhỉ", "vậy", "à", "nhé",
            "của", "trong", "với", "từ", "về", "cho", "và", "hay",
            
            # English question words  
            "what", "who", "how", "which", "where", "why", "when",
            "is", "are", "were", "was", "do", "does", "did", "can", "could",
            "should", "would", "will", "the", "a", "an", "in", "on", "at",
            "for", "with", "by", "from", "to", "of", "and", "or", "but",
            
            # Chinese question words
            "什么", "怎么", "哪个", "哪些", "啥", "为什么", "怎样", "如何",
            "是", "的", "了", "在", "有", "与", "及", "即", "为", "从", "以"
        }
        
        # Universal question patterns (regex-based, no hardcoding)
        self.question_patterns = [
            # Vietnamese patterns
            r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
            # English patterns  
            r"(^| )(what|who|how|which|where|why)('re|'s)? ",
            r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) "
        ]
    
    def detect_language(self, text: str) -> str:
        """Detect primary language of text - RAGFlow approach"""
        # Count different character types
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        vietnamese_chars = len(re.findall(r'[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]', text))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
        
        total_chars = len(text)
        if total_chars == 0:
            return "unknown"
        
        # Calculate percentages
        chinese_ratio = chinese_chars / total_chars
        vietnamese_ratio = vietnamese_chars / total_chars
        english_ratio = english_words / len(text.split()) if text.split() else 0
        
        # Determine primary language
        if chinese_ratio > 0.3:
            return "chinese"
        elif vietnamese_ratio > 0.1:
            return "vietnamese"
        elif english_ratio > 0.6:
            return "english"
        else:
            return "mixed"
    
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
        """Remove question words using universal patterns"""
        original_text = text
        
        # Apply universal question patterns
        for pattern in self.question_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Remove individual stop words
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in self.stop_words]
        
        # Ensure we don't remove too much content
        if len(filtered_words) >= 2:
            return ' '.join(filtered_words)
        else:
            return original_text  # Return original if too much was removed
    
    def calculate_term_weights(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """Calculate TF-IDF style weights - RAGFlow approach"""
        if not tokens:
            return []
        
        # Calculate term frequency
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        
        # Simple IDF approximation (in real system, this would use corpus statistics)
        total_tokens = len(tokens)
        weights = []
        
        for token, freq in tf.items():
            # Basic TF-IDF calculation
            tf_score = freq / total_tokens
            
            # Boost longer terms (more likely to be important)
            length_boost = min(2.0, 1.0 + len(token) / 10.0)
            
            # Penalize very short terms
            if len(token) <= 2:
                length_boost *= 0.3
            
            # Boost terms with mixed case or numbers (likely technical terms)
            if re.search(r'[A-Z]', token) or re.search(r'\d', token):
                length_boost *= 1.5
            
            final_weight = tf_score * length_boost
            weights.append((token, final_weight))
        
        # Normalize weights
        total_weight = sum(w for _, w in weights)
        if total_weight > 0:
            weights = [(token, w / total_weight) for token, w in weights]
        
        return sorted(weights, key=lambda x: x[1], reverse=True)
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract weighted keywords - universal approach"""
        # Normalize and clean text
        clean_text = self.normalize_text(text)
        
        # Remove question words
        content_text = self.remove_question_words(clean_text)
        
        # Tokenize (simple approach, could be enhanced with language-specific tokenizers)
        tokens = []
        for word in content_text.split():
            # Filter meaningful tokens
            if len(word) >= 2 and not word.isdigit():
                tokens.append(word)
        
        # Calculate weights
        weighted_terms = self.calculate_term_weights(tokens)
        
        # Extract top keywords
        keywords = [term for term, weight in weighted_terms[:max_keywords]]
        
        return keywords
    
    def enhance_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Universal query enhancement
        Returns: (enhanced_query, important_keywords)
        """
        logger.info(f"🌐 Universal query processing: '{query}'")
        
        # Step 1: Detect language for context
        language = self.detect_language(query)
        logger.debug(f"Detected language: {language}")
        
        # Step 2: Normalize text
        normalized_query = self.normalize_text(query)
        
        # Step 3: Extract keywords (statistical, not hardcoded)
        keywords = self.extract_keywords(normalized_query, max_keywords=8)
        
        # Step 4: Clean query for search (minimal processing)
        enhanced_query = self.remove_question_words(normalized_query)
        
        # Fallback to original if too much was removed
        if len(enhanced_query.split()) < 2:
            enhanced_query = normalized_query
        
        logger.info(f"✅ Universal processing complete:")
        logger.info(f"   Enhanced query: '{enhanced_query}'")
        logger.info(f"   Keywords: {keywords}")
        logger.info(f"   Language: {language}")
        
        return enhanced_query, keywords
    
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
universal_query_enhancer = UniversalQueryEnhancer()

