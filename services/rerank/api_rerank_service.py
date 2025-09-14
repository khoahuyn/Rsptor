import logging
import time
import requests
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("api_rerank")

@dataclass
class RerankResult:
    scores: List[float]
    processing_time: float
    model_name: str

class JinaReranker:
    """Jina API Reranker - Very fast, 300ms vs 60s local"""
    
    def __init__(self, api_key: str, model_name: str = "jina-reranker-v2-base-multilingual"):
        self.api_key = api_key
        
        # 🚀 PRODUCTION-OPTIMIZED MODEL SELECTION
        # v2-base-multilingual: 9-12s, score ~0.62 (FAST for production)
        # jina-reranker-m0: 28-33s, score ~0.93 (SLOW but high quality)
        
        self.model_name = model_name
        self.base_url = "https://api.jina.ai/v1/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json"  # RAGFlow best practice
        }
        
        # Model-specific configurations
        if "m0" in model_name:
            self.max_tokens = 10240  # m0 supports 10k
            self.timeout = 45  # m0 needs more time (28-33s)
        else:
            self.max_tokens = 8196   # v2-base-multilingual  
            self.timeout = 20  # v2-base is faster (9-12s)
            
        logger.info(f"🚀 Initialized Jina reranker: {model_name}")
        logger.info(f"   📏 Max tokens: {self.max_tokens}")
        logger.info(f"   ⏱️ Timeout: {self.timeout}s")
    
    def _smart_truncate(self, text: str) -> str:
        """Smart truncation with sentence boundary awareness (RAGFlow style)"""
        if len(text) <= self.max_tokens:
            return text
        
        # Find last sentence ending within limit
        truncated = text[:self.max_tokens]
        sentence_endings = ['.', '!', '?', '。', '！', '？']
        
        for pos in range(len(truncated) - 1, max(0, len(truncated) - 300), -1):
            if truncated[pos] in sentence_endings and pos < len(truncated) - 1:
                # Keep if we retain 80%+ of content
                if pos > self.max_tokens * 0.8:
                    return truncated[:pos + 1].strip()
                break
        
        return truncated  # Fallback to character truncation
    
    def rerank(self, query: str, documents: List[str], normalize: bool = True) -> RerankResult:
        if not query or not documents:
            return RerankResult(scores=[0.0] * len(documents), processing_time=0.0, model_name=self.model_name)
        
        # Smart truncation (RAGFlow style)
        truncated_docs = [self._smart_truncate(doc) for doc in documents]
        
        start_time = time.time()
        max_retries = 3  # RAGFlow best practice
        
        for attempt in range(max_retries):
            try:
                data = {
                    "model": self.model_name,
                    "query": query,
                    "documents": truncated_docs,
                    "top_n": len(truncated_docs),
                    "return_documents": False  # Save bandwidth
                }
                
                response = requests.post(
                    self.base_url, 
                    headers=self.headers, 
                    json=data,
                    timeout=self.timeout  # Dynamic timeout based on model
                )
                response.raise_for_status()
                
                result = response.json()
                processing_time = time.time() - start_time
                
                # Initialize scores array
                scores = np.zeros(len(documents), dtype=float)
                
                # Fill scores from API response
                for item in result.get("results", []):
                    index = item.get("index", 0)
                    score = item.get("relevance_score", 0.0)
                    if 0 <= index < len(scores):
                        scores[index] = score
                
                logger.info(f"⚡ Jina reranked {len(documents)} docs in {processing_time:.3f}s (attempt {attempt + 1})")
                logger.debug(f"📊 Score range: {scores.min():.3f} - {scores.max():.3f}")
                
                return RerankResult(
                    scores=scores.tolist(),
                    processing_time=processing_time,
                    model_name=self.model_name
                )
                
            except requests.exceptions.Timeout:
                logger.warning(f"⏱️ Jina timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    logger.warning(f"⏳ Rate limited on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                logger.error(f"❌ HTTP error: {e}")
                break
                    
            except Exception as e:
                logger.error(f"❌ Jina error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.2)
        
        # Fallback result
        processing_time = time.time() - start_time
        logger.error(f"❌ All Jina attempts failed after {processing_time:.3f}s")
        return RerankResult(
            scores=[0.0] * len(documents),
            processing_time=processing_time,
            model_name=self.model_name
        )
    
    def rerank_with_chunk_data(self, query: str, chunks: List[dict]) -> List[Tuple[dict, float]]:
        if not chunks:
            return []
        
        documents = [chunk.get('content', '') for chunk in chunks]
        result = self.rerank(query, documents)
        
        chunk_scores = list(zip(chunks, result.scores))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        return chunk_scores




def get_fast_reranker(provider: str = "jina", **kwargs) -> Optional[object]:
    """Factory function to get fast API rerankers
    
    Supported providers:
    - jina: Fast, multilingual, good quality
    """
    
    if provider == "jina":
        api_key = kwargs.get("api_key")
        # Hardcode model name as requested by user
        model_name = "jina-reranker-v2-base-multilingual"
        if api_key:
            return JinaReranker(api_key=api_key, model_name=model_name)
        else:
            logger.warning("⚠️ Jina API key required")
            return None
    else:
        logger.warning(f"⚠️ Unknown provider: {provider}")
        return None
