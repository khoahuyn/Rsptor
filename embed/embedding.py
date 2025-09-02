import logging
from typing import List, Dict, Any
from config.embedding import EmbeddingSettings
from exceptions import EmbeddingError
from .voyage_config import VoyageConfig
from .bge_config import BGEConfig
from .embedding_constants import EmbeddingConstants
from utils import get_embed_cache, set_embed_cache

logger = logging.getLogger(__name__)


EmbedError = EmbeddingError


def get_provider_config(cfg: EmbeddingSettings) -> Dict[str, Any]:
    """Get provider-specific configuration"""
    provider = cfg.active_provider
    
    if provider == "voyage":
        # Check if we have a valid API key for production tier
        has_valid_key = bool(cfg.embed_api_key and cfg.embed_api_key.strip())
        is_voyage_key = has_valid_key and cfg.embed_api_key.strip().startswith("pa-")
        
        # Use Voyage optimized config
        base_config = VoyageConfig.get_optimized_config(
            is_production=is_voyage_key
        )
        return {
            **base_config,
            "base_url": cfg.embed_base_url,
            "api_key": cfg.embed_api_key,
            "api_keys": cfg.api_keys_list,  # Multi-key support
            "model": cfg.embed_model
        }
    else:
        # Use BGE local config
        base_config = BGEConfig.get_optimized_config()
        return {
            **base_config,
            "base_url": cfg.embed_base_url,
            "api_key": cfg.embed_api_key,
            "model": cfg.embed_model
        }


async def embed_texts(texts: List[str], vector_dim: int, cfg: EmbeddingSettings) -> List[List[float]]:
    """Main embedding function with provider-specific optimizations + retry logic"""
    if not cfg.embed_base_url or not cfg.embed_model:
        raise EmbedError("embedding config missing: require EMBED_BASE_URL (endpoint) and EMBED_MODEL")

    # Validate input
    if not texts:
        return []
    
    if vector_dim not in EmbeddingConstants.STANDARD_DIMENSIONS:
        print(f"Warning: Non-standard vector dimension {vector_dim}")

    # Retry logic with exponential backoff (RAGFlow pattern)
    import time
    import random
    
    retry_max = 3
    base_delay = 1.0  # Start with 1 second
    
    for attempt in range(retry_max):
        try:
            return await _embed_texts_internal(texts, vector_dim, cfg)
            
        except Exception as e:
            if attempt == retry_max - 1:  # Last attempt
                logger.error(f"❌ Embedding failed after {retry_max} attempts: {e}")
                raise EmbedError(f"Embedding failed after {retry_max} retries: {str(e)}")
            
            # Exponential backoff: 1s, 2s, 4s
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)  # Add jitter
            logger.warning(f"⚠️ Embedding attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)


async def _embed_texts_internal(texts: List[str], vector_dim: int, cfg: EmbeddingSettings) -> List[List[float]]:
    """Internal function that handles actual embedding logic"""
    # Get provider-specific config
    provider_config = get_provider_config(cfg)
    
    # Special handling for voyage-context-3 which uses contextualized_embed API
    if cfg.embed_model == "voyage-context-3":
        # Always use multi-key embedder (works with 1+ keys)
        from .voyage_multi_key import create_voyage_multi_key_embedder
        
        api_keys = provider_config.get("api_keys", [])
        
        # Ensure we have at least one key
        if not api_keys and cfg.embed_api_key:
            api_keys = [cfg.embed_api_key]
        
        if not api_keys:
            raise EmbedError("No Voyage API keys provided")
        
        # Use unified multi-key embedder (handles 1 or more keys)
        embedder = create_voyage_multi_key_embedder(
            api_keys=api_keys,
            model=cfg.embed_model,
            dimension=cfg.embed_dimension
        )
        
        # Adaptive batching based on key count and performance
        from .adaptive_batch import get_adaptive_batch_size, record_batch_performance
        
        # Calculate base batch size considering multi-key setup
        key_multiplier = min(len(api_keys), 8)  # Support up to 8 keys
        base_batch_size = 4 * key_multiplier  # 4 texts per key
        
        # Update provider config for adaptive sizing
        voyage_adaptive_config = provider_config.copy()
        voyage_adaptive_config.update({
            "batch_size": base_batch_size,
            "max_batch_size": min(32, key_multiplier * 8),  # Scale with key count
            "target_response_time": 3.0,  # Voyage is typically slower than BGE
        })
        
        # Get optimal batch size
        optimal_batch_size = get_adaptive_batch_size("Voyage", voyage_adaptive_config, len(texts))
        
        logger.info(f"Using {len(api_keys)} Voyage key(s), adaptive batch size: {optimal_batch_size}")
        
        # Measure performance
        import time
        start_time = time.time()
        try:
            vectors = embedder.embed_texts_batch_balanced(texts, max_batch_size=optimal_batch_size, smart_batching=True)
            
            # Record successful performance
            response_time = time.time() - start_time
            record_batch_performance("Voyage", optimal_batch_size, response_time, True)
            logger.debug(f"🎯 Voyage batch {optimal_batch_size} texts in {response_time:.1f}s")
            
        except Exception as e:
            # Record failed performance and re-raise
            response_time = time.time() - start_time
            record_batch_performance("Voyage", optimal_batch_size, response_time, False)
            raise e
    else:
        # Use direct requests for BGE/Ollama embedding
        vectors = await _embed_texts_bge(
            base_url=provider_config["base_url"],
            model=provider_config["model"], 
            texts=texts,
            provider_config=provider_config  # ← Pass full config to use batch_size, timeout, etc.
        )

    # Validation
    if len(vectors) != len(texts):
        raise EmbedError(f"result size mismatch: {len(vectors)} != {len(texts)}")
    
    for i, v in enumerate(vectors):
        if vector_dim and len(v) != vector_dim:
            raise EmbedError(f"vector dimension mismatch at index {i}: {len(v)} != {vector_dim}")

    return vectors


async def embed_text_cached(text: str, vector_dim: int, cfg: EmbeddingSettings) -> List[float]:
    """
    Single text embedding with RAGFlow-style caching + retry logic
    """
    # Check cache first (RAGFlow approach)
    cached_embedding = get_embed_cache(model=cfg.embed_model, text=text)
    if cached_embedding is not None:
        return cached_embedding
    
    # Retry logic for cache miss
    import time
    import random
    
    retry_max = 3
    base_delay = 1.0  # Start with 1 second
    
    for attempt in range(retry_max):
        try:
            # Cache miss - get embedding
            embeddings = await _embed_texts_internal([text], vector_dim, cfg)
            if not embeddings:
                raise EmbedError("No embedding returned for text")
            
            embedding = embeddings[0]
            
            # Cache the result (RAGFlow approach)
            set_embed_cache(model=cfg.embed_model, text=text, embedding=embedding)
            
            return embedding
            
        except Exception as e:
            if attempt == retry_max - 1:  # Last attempt
                logger.error(f"❌ Single text embedding failed after {retry_max} attempts: {e}")
                # Return zero vector as fallback
                return [0.0] * vector_dim
            
            # Exponential backoff: 1s, 2s, 4s  
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)  # Add jitter
            logger.warning(f"⚠️ Single embedding attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    
    # Fallback (should not reach here)
    return [0.0] * vector_dim


async def _embed_texts_bge(base_url: str, model: str, texts: List[str], provider_config: Dict[str, Any] = None) -> List[List[float]]:
    """
    Optimized BGE/Ollama embedding with batch processing and config integration
    """
    import asyncio
    import aiohttp
    import time
    
    # Use provider config or fallback to defaults
    if provider_config is None:
        provider_config = BGEConfig.get_optimized_config()
    
    # Get adaptive batch sizer
    from .adaptive_batch import get_adaptive_batch_size, record_batch_performance
    
    timeout = provider_config.get("timeout", 30) 
    rpm_limit = provider_config.get("rpm_limit", 60)     # Updated to match BGEConfig.DEFAULT_RPM_LIMIT
    max_retries = provider_config.get("max_retries", 3)
    
    # Calculate delay between requests based on RPM limit
    min_delay = 60.0 / rpm_limit if rpm_limit > 0 else 0.1  # seconds between requests
    
    headers = {"Content-Type": "application/json"}
    vectors = []
    
    # Process texts in batches for better performance
    async def embed_batch_async(batch_texts: List[str]) -> List[List[float]]:
        """Async batch processing for better concurrency"""
        batch_vectors = []
        
        # Strengthen async with better retry handling
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i, text in enumerate(batch_texts):
                # Add delay between requests to respect RPM limit
                delay = i * min_delay
                tasks.append(embed_single_text_async(session, text, delay))
            
            # Execute all requests concurrently with better error handling
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results with detailed error reporting
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Let top-level retry handle this instead of fallback
                    raise EmbeddingError(f"BGE async request failed for text {i}: {result}", "BGE_ASYNC_FAILED")
                batch_vectors.append(result)
        
        return batch_vectors
    
    async def embed_single_text_async(session: aiohttp.ClientSession, text: str, delay: float = 0) -> List[float]:
        """Single text embedding with retry logic"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        payload = {"model": model, "prompt": text}
        
        for attempt in range(max_retries):
            try:
                async with session.post(base_url, headers=headers, json=payload, timeout=timeout) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise EmbeddingError(f"HTTP {response.status}: {error_text}", f"HTTP_ERROR_{response.status}")
                    
                    data = await response.json()
                    embedding = data.get("embedding")
                    
                    if not isinstance(embedding, list):
                        raise EmbeddingError("Invalid embedding format from BGE/Ollama", "BGE_INVALID_FORMAT")
                    
                    return embedding
                    
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter (consistent với top-level retry)
                    import random
                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
                    continue
                raise EmbeddingError(f"BGE request timeout after {max_retries} attempts", "BGE_TIMEOUT")
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter (consistent với top-level retry)
                    import random
                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
                    continue
                raise EmbeddingError(f"BGE request failed: {e}", "BGE_REQUEST_FAILED")
    
    
    # Process all texts in adaptive batches
    try:
        i = 0
        while i < len(texts):
            # Get optimal batch size for current conditions
            optimal_batch_size = get_adaptive_batch_size("BGE", provider_config, len(texts) - i)
            batch_texts = texts[i:i + optimal_batch_size]
            
            # Measure performance for adaptive sizing
            start_time = time.time()
            success = False
            
            try:
                # ASYNC ONLY - No fallback, better performance + top-level retry handles errors
                batch_vectors = await embed_batch_async(batch_texts)
                vectors.extend(batch_vectors)
                success = True
                
            except Exception as e:
                # Record failed performance and re-raise
                response_time = time.time() - start_time
                record_batch_performance("BGE", optimal_batch_size, response_time, False)
                raise e
            
            # Record successful performance
            response_time = time.time() - start_time
            record_batch_performance("BGE", optimal_batch_size, response_time, True)
            
            logger.debug(f"🎯 BGE batch {optimal_batch_size} texts in {response_time:.1f}s")
            i += optimal_batch_size
    
    except Exception as e:
        raise EmbeddingError(f"BGE batch processing failed: {e}", "BGE_BATCH_FAILED")

    return vectors




