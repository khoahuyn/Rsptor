import time
import logging
from typing import Dict, Any
from collections import deque

logger = logging.getLogger(__name__)


class AdaptiveBatchSizer:
    """Manages adaptive batch sizing based on performance metrics"""
    
    def __init__(self, provider: str, initial_config: Dict[str, Any]):
        self.provider = provider
        self.current_batch_size = initial_config.get("batch_size", 12)
        self.min_batch_size = initial_config.get("min_batch_size", 4)
        self.max_batch_size = initial_config.get("max_batch_size", 24)
        self.adaptive_enabled = initial_config.get("adaptive_enabled", True)
        
        # Performance thresholds
        self.target_response_time = initial_config.get("target_response_time", 2.0)
        self.slow_threshold = initial_config.get("slow_threshold", 4.0)
        self.fast_threshold = initial_config.get("fast_threshold", 1.0)
        
        # Performance tracking
        self.response_times = deque(maxlen=10)  # Keep last 10 measurements
        self.last_adjustment_time = time.time()
        self.adjustment_cooldown = 30.0  # 30 seconds between adjustments
        
        logger.info(f"🎯 {provider} adaptive batch: {self.current_batch_size} "
                   f"(range: {self.min_batch_size}-{self.max_batch_size})")
    
    def record_performance(self, batch_size: int, response_time: float, success: bool):
        """Record performance metrics for a batch operation"""
        if not success:
            # Failed batches indicate size might be too large
            logger.warning(f"⚠️ {self.provider} batch failed: size={batch_size}, time={response_time:.1f}s")
            self.response_times.append(response_time * 2)  # Penalize failures
        else:
            self.response_times.append(response_time)
            logger.debug(f"📊 {self.provider} batch success: size={batch_size}, time={response_time:.1f}s")
    
    def get_optimal_batch_size(self, num_texts: int) -> int:
        """Get optimal batch size for current conditions"""
        if not self.adaptive_enabled:
            return min(self.current_batch_size, num_texts)
        
        # Check if we have enough data and cooldown has passed
        if len(self.response_times) < 3:
            return min(self.current_batch_size, num_texts)
        
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.adjustment_cooldown:
            return min(self.current_batch_size, num_texts)
        
        # Calculate average response time
        avg_response_time = sum(self.response_times) / len(self.response_times)
        
        # Determine if adjustment is needed
        adjustment = self._calculate_adjustment(avg_response_time)
        
        if adjustment != 0:
            old_size = self.current_batch_size
            self.current_batch_size = max(
                self.min_batch_size,
                min(self.max_batch_size, self.current_batch_size + adjustment)
            )
            
            if self.current_batch_size != old_size:
                logger.info(f"🔧 {self.provider} batch size adjusted: {old_size} → {self.current_batch_size} "
                           f"(avg response: {avg_response_time:.1f}s)")
                self.last_adjustment_time = current_time
                self.response_times.clear()  # Reset measurements after adjustment
        
        return min(self.current_batch_size, num_texts)
    
    def _calculate_adjustment(self, avg_response_time: float) -> int:
        """Calculate batch size adjustment based on performance"""
        if avg_response_time > self.slow_threshold:
            # Too slow, reduce batch size significantly
            return -4
        elif avg_response_time > self.target_response_time * 1.5:
            # Somewhat slow, reduce batch size moderately
            return -2
        elif avg_response_time < self.fast_threshold:
            # Very fast, increase batch size significantly
            return +4
        elif avg_response_time < self.target_response_time * 0.7:
            # Fast, increase batch size moderately
            return +2
        else:
            # Optimal range, no adjustment
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.response_times:
            return {
                "provider": self.provider,
                "current_batch_size": self.current_batch_size,
                "avg_response_time": 0.0,
                "measurements": 0
            }
        
        return {
            "provider": self.provider,
            "current_batch_size": self.current_batch_size,
            "avg_response_time": sum(self.response_times) / len(self.response_times),
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times),
            "measurements": len(self.response_times),
            "adaptive_enabled": self.adaptive_enabled
        }


# Global instances for each provider
_batch_sizers: Dict[str, AdaptiveBatchSizer] = {}


def get_batch_sizer(provider: str, config: Dict[str, Any]) -> AdaptiveBatchSizer:
    """Get or create adaptive batch sizer for provider"""
    if provider not in _batch_sizers:
        _batch_sizers[provider] = AdaptiveBatchSizer(provider, config)
    return _batch_sizers[provider]


def get_adaptive_batch_size(provider: str, config: Dict[str, Any], num_texts: int) -> int:
    """Get optimal batch size for current request"""
    sizer = get_batch_sizer(provider, config)
    return sizer.get_optimal_batch_size(num_texts)


def record_batch_performance(provider: str, batch_size: int, response_time: float, success: bool):
    """Record performance metrics for a batch operation"""
    if provider in _batch_sizers:
        _batch_sizers[provider].record_performance(batch_size, response_time, success)


def get_performance_stats() -> Dict[str, Dict[str, Any]]:
    """Get performance statistics for all providers"""
    return {provider: sizer.get_stats() for provider, sizer in _batch_sizers.items()}
