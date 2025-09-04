import asyncio
import threading
import time
import logging
from typing import List, Optional
from models import RaptorParams
from config.llm import get_llm_settings
from .fpt_client import create_fpt_client
from utils import llm_errors, get_llm_cache, set_llm_cache

logger = logging.getLogger(__name__)


class AsyncLLMSummarizer:
    
    def __init__(
        self,
        max_concurrent: int = 20,  # Increased from 8 to 20
        rpm_limit: Optional[int] = None,
        min_interval: Optional[float] = None,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        backoff_jitter: float = 0.25,
    ):
        self.cfg = get_llm_settings()
        
        # Concurrency control (like Raptor-service)
        self.max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        self._thread_sem = threading.BoundedSemaphore(max_concurrent)
        
        # Rate limiting (like Raptor-service)
        if rpm_limit is None:
            rpm_limit = getattr(self.cfg, 'rpm_limit', 60)  # Default 60 RPM
        
        if rpm_limit and rpm_limit > 0:
            self.min_interval = max(min_interval or 0.0, 60.0 / float(rpm_limit))
        else:
            self.min_interval = float(min_interval or 0.0)
            
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0
        
        # Retry configuration
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.backoff_jitter = backoff_jitter
        
        logger.info(f"🚀 AsyncLLMSummarizer initialized: max_concurrent={max_concurrent}, rpm={rpm_limit}, min_interval={self.min_interval:.2f}s")
    
    def _apply_rate_limiting(self):
        """Apply rate limiting (blocking call in thread)"""
        if self.min_interval <= 0:
            return
            
        with self._rate_lock:
            now = time.time()
            if now < self._next_request_at:
                sleep_time = self._next_request_at - now
                logger.debug(f"⏳ Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                now = self._next_request_at
            self._next_request_at = now + self.min_interval
    
    def _compute_backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay with jitter"""
        import random
        delay = min(self.backoff_base * (2 ** max(0, attempt - 1)), self.backoff_max)
        if self.backoff_jitter:
            delay += random.uniform(0, self.backoff_jitter)
        return delay
    
    def _summarize_sync(self, text: str, system_prompt: str, max_tokens: int, temperature: float) -> str:
        """Synchronous summarization with rate limiting and retries (runs in thread)"""
        
        # Apply rate limiting
        self._apply_rate_limiting()
        
        # Prepare messages for caching
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        gen_conf = {
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Check cache first
        cached_response = get_llm_cache(
            model=self.cfg.model,
            system=system_prompt,
            history=messages,
            gen_conf=gen_conf
        )
        
        if cached_response:
            return cached_response
        
        # Cache miss - make API call with retries
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                with self._thread_sem:  # Thread-level concurrency control
                    fpt_client = create_fpt_client(self.cfg.base_url, self.cfg.api_key)
                    
                    response = fpt_client.chat_completions_create(
                        model=self.cfg.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_retries=1
                    )
                    
                    content = response.choices[0].message.content.strip()
                    content = content.replace('```', '').strip()
                    
                    # Cache the response
                    set_llm_cache(
                        model=self.cfg.model,
                        response=content,
                        system=system_prompt,
                        history=messages,
                        gen_conf=gen_conf
                    )
                    
                    if attempt > 0:
                        logger.info(f"✅ LLM succeeded on attempt {attempt + 1}")
                    
                    return content
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"⚠️ LLM attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    backoff_delay = self._compute_backoff_delay(attempt)
                    logger.info(f"🔄 Retrying in {backoff_delay:.1f}s...")
                    time.sleep(backoff_delay)
                    continue
        
        # All retries failed
        logger.error(f"❌ LLM failed after {self.max_retries} attempts")
        raise last_exception or RuntimeError("LLM summarization failed")
    
    async def summarize_cluster_async(
        self, 
        cluster_texts: List[str],
        cluster_id: int = 0,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Async cluster summarization with full optimization
        """
        if not cluster_texts:
            return "Empty cluster"
        
        # Prepare content and prompt
        member_ids = [f"chunk_{i}" for i in range(len(cluster_texts))]
        lines = [f"[{mid}] {content}" for mid, content in zip(member_ids, cluster_texts)]
        user_content = "\n".join(lines)
        
        # Use config defaults
        max_tokens = max_tokens or self.cfg.summary_max_tokens
        temperature = self.cfg.summary_temperature
        sys_prompt = self.cfg.summary_prompt.replace("{max_tokens}", str(max_tokens))
        
        # Async execution with semaphore
        async with self._sem:
            try:
                # Run blocking operation in thread
                content = await asyncio.to_thread(
                    self._summarize_sync,
                    user_content,
                    sys_prompt, 
                    max_tokens,
                    temperature
                )
                
                logger.debug(f"📝 Cluster {cluster_id}: {len(cluster_texts)} texts → summary ({len(content)} chars)")
                return content
                
            except Exception as e:
                logger.error(f"❌ Cluster {cluster_id} summarization failed: {e}")
                # Fallback: concatenate first sentences
                fallback = ". ".join([text.split('.')[0] for text in cluster_texts[:3]])
                return fallback[:max_tokens] if fallback else f"Cluster {cluster_id} summary"


# Global instance for easy access
_global_summarizer: Optional[AsyncLLMSummarizer] = None

def get_async_summarizer() -> AsyncLLMSummarizer:
    """Get global async summarizer instance"""
    global _global_summarizer
    if _global_summarizer is None:
        _global_summarizer = AsyncLLMSummarizer()
    return _global_summarizer


@llm_errors  
async def summarize_cluster_from_contents_async(
    member_contents: List[str], 
    member_ids: List[str], 
    p: RaptorParams,
    cluster_id: int = 0
) -> str:
    """
    Async version of summarize_cluster_from_contents with optimizations
    """
    # Check if summarization is enabled
    if not bool(getattr(p, "enable_summary", False)):
        return f"Cluster summary ({len(member_contents)} nodes)"
    
    summarizer = get_async_summarizer()
    max_tokens = int(getattr(p, "summary_max_tokens", None) or summarizer.cfg.summary_max_tokens)
    
    return await summarizer.summarize_cluster_async(
        cluster_texts=member_contents,
        cluster_id=cluster_id, 
        max_tokens=max_tokens
    )
