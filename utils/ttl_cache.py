import time
import threading
import logging
from typing import Optional, Dict, Any, TypeVar, Generic
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL support"""
    value: T
    timestamp: float
    access_count: int = 0
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry is expired"""
        return time.time() - self.timestamp > ttl_seconds
    
    def touch(self) -> None:
        """Update access count for LRU"""
        self.access_count += 1


class TTLCache(Generic[T]):
    """Thread-safe TTL cache with LRU eviction"""
    
    def __init__(self, max_size: int, ttl_seconds: int, name: str = "TTLCache"):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.name = name
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "expired": 0,
            "evicted": 0,
            "sets": 0
        }
        
        logger.info(f"🎯 {name} TTL cache: max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache with TTL check"""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            # Check if expired
            if entry.is_expired(self.ttl_seconds):
                del self._cache[key]
                self._stats["expired"] += 1
                self._stats["misses"] += 1
                logger.debug(f"⏰ {self.name} cache EXPIRED: {key[:8]}...")
                return None
            
            # Move to end for LRU and update access
            entry.touch()
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            
            logger.debug(f"🎯 {self.name} cache HIT: {key[:8]}...")
            return entry.value
    
    def set(self, key: str, value: T) -> None:
        """Set value in cache with TTL"""
        with self._lock:
            # Remove expired entries first
            self._cleanup_expired()
            
            # Ensure cache size limit
            while len(self._cache) >= self.max_size:
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._stats["evicted"] += 1
                logger.debug(f"🗑️ {self.name} cache EVICTED: {oldest_key[:8]}...")
            
            # Add new entry
            entry = CacheEntry(value=value, timestamp=time.time())
            self._cache[key] = entry
            self._stats["sets"] += 1
            
            logger.debug(f"💾 {self.name} cache SET: {key[:8]}... (size: {len(self._cache)})")
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"🗑️ {self.name} cache DELETE: {key[:8]}...")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"🧹 {self.name} cache CLEARED: {count} entries")
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries (called internally)"""
        expired_keys = []
        current_time = time.time()
        
        for key, entry in self._cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self._stats["expired"] += 1
        
        if expired_keys:
            logger.debug(f"⏰ {self.name} cleanup: removed {len(expired_keys)} expired entries")
        
        return len(expired_keys)
    
    def cleanup_expired(self) -> int:
        """Public method to cleanup expired entries"""
        with self._lock:
            return self._cleanup_expired()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "name": self.name,
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hit_rate": f"{hit_rate:.1f}%",
                "stats": self._stats.copy()
            }
    
    def get_size(self) -> int:
        """Get current cache size"""
        with self._lock:
            return len(self._cache)


class CacheManager:
    """Global cache manager with automatic cleanup"""
    
    def __init__(self):
        self._caches: Dict[str, TTLCache] = {}
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_interval = 300  # 5 minutes
        self._running = False
        self._lock = threading.Lock()
    
    def register_cache(self, name: str, cache: TTLCache) -> None:
        """Register a cache for automatic management"""
        with self._lock:
            self._caches[name] = cache
            logger.info(f"📝 Registered cache: {name}")
    
    def start_cleanup_thread(self, interval: int = None) -> None:
        """Start automatic cleanup thread"""
        with self._lock:
            if self._running:
                return
            
            if interval:
                self._cleanup_interval = interval
                
            self._running = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="CacheCleanup"
            )
            self._cleanup_thread.start()
            logger.info(f"🧹 Cache cleanup thread started (interval: {self._cleanup_interval}s)")
    
    def stop_cleanup_thread(self) -> None:
        """Stop automatic cleanup thread"""
        with self._lock:
            self._running = False
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=1.0)
                logger.info("🛑 Cache cleanup thread stopped")
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self._running:
            try:
                total_cleaned = 0
                for name, cache in self._caches.items():
                    cleaned = cache.cleanup_expired()
                    total_cleaned += cleaned
                
                if total_cleaned > 0:
                    logger.info(f"🧹 Cache cleanup: removed {total_cleaned} expired entries across {len(self._caches)} caches")
                
                # Sleep in small chunks to allow quick shutdown
                for _ in range(self._cleanup_interval):
                    if not self._running:
                        break
                    time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"❌ Cache cleanup error: {e}")
                time.sleep(5.0)  # Brief pause before retrying
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all managed caches"""
        with self._lock:
            return {name: cache.get_stats() for name, cache in self._caches.items()}
    
    def clear_all_caches(self) -> None:
        """Clear all managed caches"""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            logger.info(f"🧹 All {len(self._caches)} caches cleared")


# Global cache manager instance
cache_manager = CacheManager()
