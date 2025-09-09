import logging
from typing import List, Dict, Any
from config.embedding import EmbeddingSettings
from exceptions import EmbeddingError
from .voyage_config import VoyageConfig
from .bge_config import BGEConfig
from .embedding_constants import EmbeddingConstants
from utils import get_embed_cache, set_embed_cache

logger = logging.getLogger(__name__)

# Global embedder cache to persist round-robin counter
_voyage_embedder_cache = {}

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

    if not cfg.embed_base_url or not cfg.embed_model:
        raise EmbedError("embedding config missing: require EMBED_BASE_URL (endpoint) and EMBED_MODEL")

    # Validate input
    if not texts:
        return []
    
    if vector_dim not in EmbeddingConstants.STANDARD_DIMENSIONS:
        logger.warning(f"⚠️ Non-standard vector dimension {vector_dim}")

    logger.info(f"🚀 EMBED_TEXTS: Processing {len(texts)} texts with intelligent batching")
    
    # 🔥 USE BATCH PROCESSING for all scenarios (single + multiple texts)
    try:
        return await embed_texts_batch(texts, vector_dim, cfg)
    except Exception as e:
        logger.error(f"❌ Batch embedding failed: {e}")
        raise EmbedError(f"Batch embedding failed: {str(e)}")


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
        
        # Create or get cached smart multi-key embedder (preserves round-robin counter)
        cache_key = f"{cfg.embed_model}_{len(api_keys)}_{hash(tuple(api_keys))}"
        
        if cache_key not in _voyage_embedder_cache:
            logger.info(f"🔄 Creating new Voyage embedder for cache key: {cache_key[:20]}...")
            try:
                _voyage_embedder_cache[cache_key] = create_voyage_multi_key_embedder(
                    api_keys=api_keys,
                    model=cfg.embed_model,
                    dimension=cfg.embed_dimension
                )
            except ValueError as e:
                if "MULTIPLE KEYS REQUIRED" in str(e):
                    raise EmbedError(f"❌ Configuration Error: {str(e)}")
                raise e
        else:
            logger.info(f"♻️ Using cached Voyage embedder: {cache_key[:20]}...")
        
        embedder = _voyage_embedder_cache[cache_key]
        
        # Calculate base batch size considering multi-key setup
        key_multiplier = min(len(api_keys), 8)  # Support up to 8 keys
        base_batch_size = 4 * key_multiplier  # 4 texts per key
        
        # Update provider config for batch sizing
        voyage_adaptive_config = provider_config.copy()
        voyage_adaptive_config.update({
            "batch_size": base_batch_size,
            "max_batch_size": min(32, key_multiplier * 8),  # Scale with key count
            "target_response_time": 3.0,  # Voyage is typically slower than BGE
        })
        
        # Use fixed batch size from config
        batch_size = voyage_adaptive_config["batch_size"]
        
        logger.info(f"Using {len(api_keys)} Voyage key(s), batch size: {batch_size}")
        
        # 🎯 SMART PARALLEL EXECUTION - Optimal resource allocation
        logger.info(f"🎯 Using SMART PARALLEL execution with resource allocation for {len(texts)} texts")
        vectors = await embedder.embed_texts_smart_parallel(texts)
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


async def embed_texts_batch(texts: List[str], vector_dim: int, cfg: EmbeddingSettings) -> List[List[float]]:

    if not texts:
        return []
    
    logger.info(f"🔥 BATCH EMBEDDING: Processing {len(texts)} texts")
    
    # Phase 1: Check cache for all texts
    cached_results = {}
    uncached_texts = []
    uncached_indices = []
    
    for i, text in enumerate(texts):
        cached = get_embed_cache(model=cfg.embed_model, text=text)
        if cached is not None:
            cached_results[i] = cached
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    logger.info(f"📊 Cache stats: {len(cached_results)} hits, {len(uncached_texts)} misses")
    
    # Phase 2: Batch embed uncached texts (MAXIMUM BATCH SIZE)
    if uncached_texts:
        logger.info(f"🚀 Batch embedding {len(uncached_texts)} uncached texts")
        embeddings = await _embed_texts_internal(uncached_texts, vector_dim, cfg)
        
        # Cache new embeddings
        for text, embedding in zip(uncached_texts, embeddings):
            set_embed_cache(model=cfg.embed_model, text=text, embedding=embedding)
    else:
        embeddings = []
    
    # Phase 3: Combine results in original order
    results = [None] * len(texts)
    
    # Fill cached results
    for i, cached_emb in cached_results.items():
        results[i] = cached_emb
    
    # Fill new embeddings  
    for uncached_idx, embedding in zip(uncached_indices, embeddings):
        results[uncached_idx] = embedding
    
    logger.info(f"✅ Batch completed: {len(results)} embeddings returned")
    return results


async def embed_text_cached(text: str, vector_dim: int, cfg: EmbeddingSettings) -> List[float]:
    """
    Single text embedding - now uses batch function for consistency and parallel optimization
    """
    # Use batch function for consistency (will hit batch buffer for true parallel)
    results = await embed_texts_batch([text], vector_dim, cfg)
    return results[0] if results else [0.0] * vector_dim


async def _embed_texts_bge(base_url: str, model: str, texts: List[str], provider_config: Dict[str, Any] = None) -> List[List[float]]:
    """
    Optimized BGE/Ollama embedding with batch processing and config integration
    """
    import asyncio
    import aiohttp
    
    # Use provider config or fallback to defaults
    if provider_config is None:
        provider_config = BGEConfig.get_optimized_config()
    
    # Use optimized config for LOCAL BGE - NO RATE LIMITING
    timeout = provider_config.get("timeout", 60)      
    max_retries = provider_config.get("max_retries", 3)
    max_concurrent = provider_config.get("max_concurrent", 8)  
    
    
    headers = {"Content-Type": "application/json"}
    vectors = []
    
    logger.info(f"🔥 BGE LOCAL: Processing {len(texts)} texts with aggressive parallel batching")
    
    async def embed_batch_async(batch_texts: List[str]) -> List[List[float]]:
        """OPTIMIZED async batch processing - LOCAL BGE performance mode"""
        batch_vectors = []
        
        semaphore = asyncio.Semaphore(max_concurrent)  # Control concurrency
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for text in batch_texts:
                tasks.append(embed_single_with_semaphore(session, semaphore, text))
            
            # Execute all requests concurrently with better error handling
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results with detailed error reporting
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Let top-level retry handle this instead of fallback
                    raise EmbeddingError(f"BGE async request failed for text {i}: {result}", "BGE_ASYNC_FAILED")
                batch_vectors.append(result)
        
        return batch_vectors
    
    async def embed_single_with_semaphore(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, text: str) -> List[float]:
        """Semaphore-controlled embedding for optimal local BGE performance"""
        async with semaphore:
            return await embed_single_text_async(session, text)
    
    async def embed_single_text_async(session: aiohttp.ClientSession, text: str, delay: float = 0) -> List[float]:
        """Single text embedding with retry logic - OPTIMIZED for local BGE"""
        
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
                    # Exponential backoff with jitter 
                    import random
                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
                    continue
                raise EmbeddingError(f"BGE request timeout after {max_retries} attempts", "BGE_TIMEOUT")
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter 
                    import random
                    delay = (2 ** attempt) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
                    continue
                raise EmbeddingError(f"BGE request failed: {e}", "BGE_REQUEST_FAILED")
    
    
    # Process all texts in LARGE batches for optimal local performance
    batch_size = provider_config.get("batch_size", 32)  # ✅ Large batches like VoyageAI
    
    try:
        i = 0
        while i < len(texts):
            # Use fixed batch size from config
            current_batch_size = min(batch_size, len(texts) - i)
            batch_texts = texts[i:i + current_batch_size]
            
            try:
                # ASYNC ONLY - No fallback, better performance + top-level retry handles errors
                batch_vectors = await embed_batch_async(batch_texts)
                vectors.extend(batch_vectors)
                
            except Exception as e:
                raise e
            
            logger.info(f"✅ BGE LOCAL batch {current_batch_size} texts completed - PARALLEL PROCESSING")
            i += current_batch_size
    
    except Exception as e:
        raise EmbeddingError(f"BGE batch processing failed: {e}", "BGE_BATCH_FAILED")

    return vectors




