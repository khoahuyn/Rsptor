import hashlib
import json
import logging
from typing import Optional, List, Dict, Any

from config.cache import get_cache_settings
from .ttl_cache import TTLCache, cache_manager

logger = logging.getLogger(__name__)

# Global TTL caches (Enhanced RAGFlow approach)
_llm_cache: Optional[TTLCache[str]] = None
_embed_cache: Optional[TTLCache[List[float]]] = None


def _initialize_caches():
    """Initialize TTL caches with current settings"""
    global _llm_cache, _embed_cache
    
    settings = get_cache_settings()
    
    if _llm_cache is None:
        _llm_cache = TTLCache[str](
            max_size=settings.llm_cache_max_size,
            ttl_seconds=settings.llm_cache_ttl_seconds,
            name="LLM"
        )
        cache_manager.register_cache("llm", _llm_cache)
    
    if _embed_cache is None:
        _embed_cache = TTLCache[List[float]](
            max_size=settings.embed_cache_max_size,
            ttl_seconds=settings.embed_cache_ttl_seconds,
            name="Embedding"
        )
        cache_manager.register_cache("embedding", _embed_cache)
    
    # Start cleanup thread if enabled  
    if settings.cache_auto_cleanup:
        cache_manager.start_cleanup_thread(settings.retrieval_cache_ttl_seconds)


def _make_cache_key(data: str) -> str:
    """Generate consistent cache key from data"""
    return hashlib.md5(data.encode('utf-8')).hexdigest()


# LLM Cache Functions (Enhanced TTL approach)
def get_llm_cache(model: str, system: str, history: List[Dict[str, Any]], gen_conf: Dict[str, Any]) -> Optional[str]:
    """Get cached LLM response if available with TTL"""
    settings = get_cache_settings()
    
    if not settings.llm_cache_enabled:
        return None
    
    _initialize_caches()
    
    # Create cache key from all parameters
    cache_data = f"{model}:{system}:{json.dumps(history, sort_keys=True)}:{json.dumps(gen_conf, sort_keys=True)}"
    cache_key = _make_cache_key(cache_data)
    
    response = _llm_cache.get(cache_key)
    if response and settings.cache_hit_log_enabled:
        logger.debug(f"🎯 LLM cache HIT: {cache_key[:8]}...")
    
    return response


def set_llm_cache(model: str, response: str, system: str, history: List[Dict[str, Any]], gen_conf: Dict[str, Any]) -> None:
    """Cache LLM response with TTL"""
    settings = get_cache_settings()
    
    if not settings.llm_cache_enabled:
        return
    
    _initialize_caches()
    
    # Create cache key from all parameters  
    cache_data = f"{model}:{system}:{json.dumps(history, sort_keys=True)}:{json.dumps(gen_conf, sort_keys=True)}"
    cache_key = _make_cache_key(cache_data)
    
    _llm_cache.set(cache_key, response)


# Embedding Cache Functions (Enhanced TTL approach)
def get_embed_cache(model: str, text: str) -> Optional[List[float]]:
    """Get cached embedding if available with TTL"""
    settings = get_cache_settings()
    
    if not settings.embed_cache_enabled:
        return None
    
    _initialize_caches()
    
    cache_data = f"{model}:{text}"
    cache_key = _make_cache_key(cache_data)
    
    embedding = _embed_cache.get(cache_key)
    if embedding and settings.cache_hit_log_enabled:
        logger.debug(f"🎯 Embedding cache HIT: {cache_key[:8]}...")
    
    return embedding


def set_embed_cache(model: str, text: str, embedding: List[float]) -> None:
    """Cache embedding with TTL"""
    settings = get_cache_settings()
    
    if not settings.embed_cache_enabled:
        return
    
    _initialize_caches()
    
    cache_data = f"{model}:{text}"
    cache_key = _make_cache_key(cache_data)
    
    _embed_cache.set(cache_key, embedding)


# Cache Statistics & Management (for performance monitoring/optimization)
def get_cache_stats() -> Dict[str, Any]:
    """Get enhanced cache statistics with TTL info for performance monitoring"""
    _initialize_caches()
    
    return {
        "llm_cache": _llm_cache.get_stats() if _llm_cache else {"enabled": False},
        "embed_cache": _embed_cache.get_stats() if _embed_cache else {"enabled": False},
        "global_stats": cache_manager.get_all_stats()
    }



