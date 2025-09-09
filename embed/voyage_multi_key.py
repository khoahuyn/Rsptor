import asyncio
import itertools
import logging
import time
from collections import deque
from typing import List
import voyageai
from .voyage_config import VoyageConfig
from utils.token_packing import pack_texts_by_token_budget

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-aware rate limiter with RPM and TPM sliding window"""
    
    def __init__(self, rpm: int, tpm: int, window_seconds: int = 60):
        self.rpm = rpm
        self.tpm = tpm
        self.window = window_seconds
        self._req_times = deque()
        self._tok_times = deque()  # (timestamp, tokens)
        self._lock = asyncio.Lock()
    
    def _prune(self, now: float):
        """Remove old entries outside the window"""
        cutoff = now - self.window
        while self._req_times and self._req_times[0] <= cutoff:
            self._req_times.popleft()
        while self._tok_times and self._tok_times[0][0] <= cutoff:
            self._tok_times.popleft()
    
    def _tokens_used(self) -> int:
        """Count tokens used in current window"""
        return sum(tokens for _, tokens in self._tok_times)
    
    async def acquire(self, tokens_for_next_req: int):
        """Wait until rate limits allow the request"""
        async with self._lock:
            while True:
                now = time.time()
                self._prune(now)
                
                req_ok = len(self._req_times) < self.rpm
                tpm_ok = (self._tokens_used() + tokens_for_next_req) <= self.tpm
                
                if req_ok and tpm_ok:
                    self._req_times.append(now)
                    self._tok_times.append((now, tokens_for_next_req))
                    return
                
                # Calculate wait time
                wait_req = (
                    max(0.0, self.window - (now - self._req_times[0])) if self._req_times else 0.0
                )
                wait_tok = (
                    max(0.0, self.window - (now - self._tok_times[0][0]))
                    if self._tok_times else 0.0
                )
                wait_time = max(wait_req, wait_tok, 0.1)
                logger.info(f"🛡️ Rate limit wait: {wait_time:.1f}s (RPM: {req_ok}, TPM: {tpm_ok})")
                await asyncio.sleep(wait_time)


class VoyageMultiKeyEmbedder:
    """
    Multi-key Voyage AI embedder with round-robin load balancing
    Allows using multiple API keys to bypass rate limits and achieve higher throughput
    """
    
    def __init__(self, api_keys: List[str], model: str = "voyage-context-3", dimension: int = 1024):
        
        if not api_keys:
            raise ValueError("At least one API key is required")
        
        # REQUIRE MULTIPLE KEYS - No single key support
        if len(api_keys) < 2:
            raise ValueError(
                f"❌ MULTIPLE KEYS REQUIRED: Found {len(api_keys)} key(s), need at least 2 keys for optimal performance. "
                "Please add more Voyage API keys to your EMBED_API_KEY environment variable (comma-separated)."
            )
        
        self.api_keys = api_keys
        self.model = model
        self.dimension = dimension
    
        
        # Create async clients for each key  
        self.clients = [voyageai.AsyncClient(api_key=key) for key in api_keys]
        
        # Round-robin key selector
        self.key_cycle = itertools.cycle(range(len(api_keys)))
        
        # Config for each key (create FIRST before using)
        self.config = VoyageConfig.get_optimized_config(is_production=False)
        
        # Conservative rate limiting per key (like Raptor-service)
        rpm_limit = self.config.get("rpm_limit", VoyageConfig.DEFAULT_RPM_LIMIT)  # 3 RPM
        tpm_limit = self.config.get("tpm_limit", VoyageConfig.DEFAULT_TPM_LIMIT)  # 10K TPM
        self.rate_limiters = [RateLimiter(rpm_limit, tpm_limit) for _ in api_keys]
        
        # Concurrency control per key (from config)
        concurrent_per_key = self.config.get("concurrent_per_key", VoyageConfig.DEFAULT_CONCURRENT_PER_KEY)
        self.semaphores = [asyncio.Semaphore(concurrent_per_key) for _ in api_keys]
        
        # Error tracking for failover
        self.failed_keys = set()  # Track failed keys
        self.key_failure_counts = [0] * len(api_keys)  # Track failure counts
        
        # Key usage tracking for better load balancing
        self.key_usage_counts = [0] * len(api_keys)  # Track how many times each key is used
        self.key_last_used = [0.0] * len(api_keys)  # Track last use time for each key
        

        
        logger.info(f"🚀 Initialized smart parallel Voyage embedder with {len(api_keys)} keys")
    

    
    async def _apply_rate_limiting(self, key_index: int, token_count: int):
        """Apply conservative rate limiting for specific key (like Raptor-service)"""
        rate_limiter = self.rate_limiters[key_index]
        
        # Use token-aware rate limiting (3 RPM, 10K TPM)
        await rate_limiter.acquire(token_count)
        
        print(f" 🛡️ KEY {key_index}: Rate limited (3 RPM, 10K TPM) - tokens: {token_count}")
    

    
    def _pick_least_used_keys(self, count: int) -> List[int]:
        """Pick keys with lowest token usage for optimal rate limit avoidance"""
        current_time = time.time()
        
        # Create list of (key_index, token_score) where lower score = better
        key_scores = []
        for i in range(len(self.api_keys)):
            if i in self.failed_keys:
                continue  # Skip permanently failed keys
                
            # Get EXACT token and request usage from rate limiter
            rate_limiter = self.rate_limiters[i]
            rate_limiter._prune(current_time)  # Remove old entries
            
            current_tokens = rate_limiter._tokens_used()
            current_requests = len(rate_limiter._req_times)
            
            # Score based on actual API usage (lower = better)
            token_penalty = current_tokens / 10000.0    # 0-1 scale (10K TPM limit)
            request_penalty = current_requests / 3.0    # 0-1 scale (3 RPM limit)
            
            # Token usage is more important for Voyage API
            total_score = token_penalty * 0.8 + request_penalty * 0.2
            
            key_scores.append((i, total_score, current_tokens, current_requests))
        
        # Sort by score (lowest token usage first) and take top N
        key_scores.sort(key=lambda x: x[1])
        selected_keys = [key_index for key_index, _, _, _ in key_scores[:count]]
        
        logger.info(f"🎯 TOKEN-BASED SELECTION: {count} keys with lowest token usage")
        for i, (key_idx, score, tokens, reqs) in enumerate(key_scores[:count]):
            logger.info(f"   Key {key_idx}: {tokens}/10K tokens, {reqs}/3 requests (score: {score:.3f})")
        
        return selected_keys
    
    def _update_key_usage(self, key_index: int):
        """Update usage statistics for a key"""
        self.key_usage_counts[key_index] += 1
        self.key_last_used[key_index] = time.time()
    
    async def _embed_with_key(self, client: voyageai.AsyncClient, api_key: str, texts: List[str], key_index: int) -> List[List[float]]:
        """Embed texts using specific key with concurrency control and rapid failover"""
        from utils.math_utils import token_count
        
        estimated_tokens = sum(token_count(text) for text in texts)
        logger.info(f"Key {key_index}: {len(texts)} texts, {estimated_tokens} tokens")
        
        # Use semaphore for concurrency control
        async with self.semaphores[key_index]:
            # Apply conservative rate limiting for this key
            await self._apply_rate_limiting(key_index, estimated_tokens)
            
            # Update usage stats
            self._update_key_usage(key_index)
        
            max_retries = self.config.get("max_retries", 3)
            
            for attempt in range(max_retries):
                try:
                    # Format input as List[List[str]] where each text is its own document
                    inputs = [[text] for text in texts]
                    
                    # Make API call (now async)
                    result = await client.contextualized_embed(
                        inputs=inputs,
                        model=self.model,
                        input_type="document",
                        output_dimension=self.dimension
                    )
                    
                    # Extract embeddings
                    embeddings = []
                    for doc_result in result.results:
                        if doc_result.embeddings:
                            embeddings.append(doc_result.embeddings[0])
                        else:
                            logger.warning(f"Key {key_index}: No embedding returned for one document")
                            embeddings.append([0.0] * self.dimension)
                    
                    if attempt > 0:
                        logger.info(f"✅ Key {key_index} succeeded on attempt {attempt + 1}")
                    
                    logger.info(f"Key {key_index}: Successfully embedded {len(embeddings)} texts")
                    return embeddings
                    
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = any(keyword in error_str for keyword in ["rate limit", "reduced rate limits", "429", "too many requests"])
                    
                    if is_rate_limit:
                        # RAPID FAILOVER: Don't wait, raise immediately for failover
                        logger.warning(f"🚦 Key {key_index} rate limited, immediate failover requested")
                        raise Exception(f"Rate limited on key {key_index}: {e}")
                    
                    if attempt < max_retries - 1:
                        # Special handling for rate limit: longer backoff
                        backoff_delay = min(25.0 + (attempt * 15.0), 60.0)  # 25s, 40s, 55s max
                        logger.warning(f"🚦 Key {key_index} rate limited, retry {attempt + 1}/{max_retries} in {backoff_delay:.1f}s")
                        await asyncio.sleep(backoff_delay)
                        continue
                    elif not is_rate_limit and attempt < max_retries - 1:
                        # Standard exponential backoff for other errors
                        backoff_delay = (2 ** attempt) + 1.0  # 1s, 3s, 7s
                        logger.warning(f"⚠️ Key {key_index} error, retry {attempt + 1}/{max_retries} in {backoff_delay:.1f}s: {e}")
                        await asyncio.sleep(backoff_delay)
                        continue
                    else:
                        # Final attempt or non-retryable error
                        logger.error(f"❌ Key {key_index} failed after {attempt + 1} attempts: {e}")
            self._handle_key_failure(key_index, e)
            raise
    
    async def embed_texts_smart_parallel(self, texts: List[str]) -> List[List[float]]:

        print(f"🔍 SMART PARALLEL ENTRY: Called with {len(texts) if texts else 0} texts")
        logger.info(f"🔍 DEBUG: embed_texts_smart_parallel called with {len(texts) if texts else 0} texts")
        
        if not texts:
            logger.info("🔍 DEBUG: Empty texts, returning empty list")
            return []
        
        # 🚀 SINGLE TEXT → ALSO USE MULTI-KEY (avoid Key 0 overuse)
        if len(texts) == 1:
            print(f"🚀 SINGLE TEXT: Using rapid failover across ALL available keys")
            # Try multiple keys rapidly if rate limited (no delay between attempts)
            available_keys = [i for i in range(len(self.api_keys)) if i not in self.failed_keys]
            if not available_keys:
                available_keys = list(range(len(self.api_keys)))
            
            # Start with token-based selection, but failover rapidly if needed
            best_keys = self._pick_least_used_keys(1)
            start_key = best_keys[0] if best_keys else available_keys[0]
            print(f"🎯 SINGLE TEXT: Starting with Key {start_key} (lowest tokens), {len(available_keys)} keys available for failover")
            
            # Try keys in sequence until success (rapid failover)
            start_idx = available_keys.index(start_key)
            for attempt in range(len(available_keys)):
                key_index = available_keys[(start_idx + attempt) % len(available_keys)]
                
                try:
                    print(f"🔄 ATTEMPT {attempt + 1}: Trying Key {key_index}")
                    result = await self._embed_with_key(
                        client=self.clients[key_index],
                        api_key=self.api_keys[key_index],
                        texts=texts,
                        key_index=key_index
                    )
                    print(f"✅ SINGLE TEXT: Key {key_index} succeeded on attempt {attempt + 1}")
                    return result
                    
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = any(keyword in error_str for keyword in ["rate limit", "reduced rate limits", "429", "too many requests"])
                    
                    if is_rate_limit and attempt < len(available_keys) - 1:
                        print(f"🚦 Key {key_index} rate limited, trying next key immediately...")
                        continue  # No delay - rapid failover!
                    else:
                        print(f"❌ Key {key_index} failed: {e}")
                        if attempt >= len(available_keys) - 1:
                            print(f"💥 ALL {len(available_keys)} KEYS FAILED!")
                            raise
                        continue
        
        print(f"📦 TOKEN PACKING: Analyzing {len(texts)} texts for optimal grouping")
        
        # 🔢 STEP 1: Pack texts by token budget to minimize API calls
        try:
            api_key = self.api_keys[0]  # Use first key for token counting
            groups = pack_texts_by_token_budget(
                texts=texts,
                token_budget=12000,  # More aggressive packing
                model=self.model,
                api_key=api_key
            )
        except Exception as e:
            logger.warning(f"Token packing failed: {e}, fallback to single group")
            groups = [texts]  # Fallback: single group
        
        print(f"📦 PACKED: {len(texts)} texts → {len(groups)} groups")
        
        # 🎯 STEP 2: ALWAYS MULTI-KEY - No single key exceptions!
        # Even small batches benefit from key rotation to avoid rate limiting
        print(f"🔑 MULTI-KEY: ALWAYS use ALL {len(self.api_keys)} keys for {len(texts)} texts (avoid rate limits)")
        
        # 🚀 STEP 3: Multi-key execution for multiple large groups
        return await self._embed_multi_key_groups(groups)
    
    
    async def _embed_multi_key_groups(self, groups: List[List[str]]) -> List[List[float]]:
        """Multi-key approach - PRESERVE TOKEN PACKING"""
        print(f"🔑 MULTI-KEY PROCESSING: {len(groups)} groups across {len(self.api_keys)} keys")
        
        # 🎯 KEEP GROUPS INTACT: Preserve token packing optimization!
        # Each group = 1 API call (much more efficient than flattening)
        print(f"🎯 PRESERVING TOKEN PACKING: {len(groups)} API calls instead of flattening")
        
        # SMART GROUP DISTRIBUTION: Assign groups to keys (preserve token packing)
        available_keys = [i for i in range(len(self.api_keys)) if i not in self.failed_keys]
        if not available_keys:
            available_keys = list(range(len(self.api_keys)))  # Reset if all failed
        
        # Use least-used keys for optimal load balancing
        selected_keys = self._pick_least_used_keys(min(len(groups), len(available_keys)))
        print(f"🎯 GROUP ASSIGNMENT: {len(groups)} groups → {len(selected_keys)} keys")
        
        # Assign each GROUP to a key (1 group = 1 API call)
        group_assignments = {}
        for i, group in enumerate(groups):
            key_index = selected_keys[i % len(selected_keys)]
            if key_index not in group_assignments:
                group_assignments[key_index] = []
            group_assignments[key_index].extend(group)  # All texts in group go to same key
        
        # Log distribution
        for key_idx, texts in group_assignments.items():
            print(f"📊 KEY {key_idx}: {len(texts)} texts (from assigned groups)")
        
        # Execute ALL keys in parallel
        tasks = []
        for key_index, texts in group_assignments.items():
            task = self._embed_with_key(
                client=self.clients[key_index],
                api_key=self.api_keys[key_index],
                texts=texts,
                key_index=key_index
            )
            tasks.append((key_index, texts, task))
        
        start_time = time.time()
        print(f"🔥 EXECUTING {len(tasks)} PARALLEL KEYS (ROUND-ROBIN)...")
        
        # Simple parallel execution without stagger delays (like Raptor-service)
        just_tasks = [task for _, _, task in tasks]
        results = await asyncio.gather(*just_tasks, return_exceptions=True)
        elapsed = time.time() - start_time
        
        # RAPID FAILOVER: Retry failed tasks with different keys
        failed_tasks = []
        for i, (key_index, texts, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                error_str = str(result).lower()
                if "rate limit" in error_str:
                    logger.warning(f"🔄 FAILOVER: Key {key_index} rate limited, trying other keys...")
                    failed_tasks.append((key_index, texts, i))
        
        # Retry failed tasks with least-used keys
        if failed_tasks:
            print(f"🚀 RAPID FAILOVER: Retrying {len(failed_tasks)} failed tasks...")
            retry_results = []
            
            for original_key, texts, original_index in failed_tasks:
                retry_success = False
                
                # Try other least-used keys
                available_keys = [k for k in range(len(self.api_keys)) if k != original_key and k not in self.failed_keys]
                if available_keys:
                    # Sort by usage (least used first)
                    available_keys.sort(key=lambda k: (self.key_usage_counts[k], self.key_last_used[k]))
                    
                    for retry_key in available_keys[:3]:  # Try up to 3 alternative keys
                        try:
                            print(f"🔄 FAILOVER: Trying key {retry_key} for {len(texts)} texts...")
                            retry_result = await self._embed_with_key(
                                client=self.clients[retry_key],
                                api_key=self.api_keys[retry_key],
                                texts=texts,
                                key_index=retry_key
                            )
                            results[original_index] = retry_result  # Replace failed result
                            retry_success = True
                            print(f"✅ FAILOVER SUCCESS: Key {retry_key} succeeded!")
                            break
                        except Exception as e:
                            logger.warning(f"🚦 Failover key {retry_key} also failed: {e}")
                            continue
                
                if not retry_success:
                    logger.error(f"❌ FAILOVER FAILED: All keys exhausted for {len(texts)} texts")
                    # Keep the original exception
        
        print(f"✅ MULTI-KEY COMPLETED: ALL {len(tasks)} keys in {elapsed:.1f}s")
        
        # Reconstruct in original order (flatten groups for mapping)
        all_texts = []
        for group in groups:
            all_texts.extend(group)
        
        final_embeddings = [None] * len(all_texts)
        
        for i, (key_index, texts, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Key {key_index} failed: {result}")
                # Add zero vectors for failed key
                for j in range(len(texts)):
                    text_pos = all_texts.index(texts[j])
                    final_embeddings[text_pos] = [0.0] * self.dimension
            else:
                # Map results back to original positions
                for j, embedding in enumerate(result):
                    text_pos = all_texts.index(texts[j])
                    final_embeddings[text_pos] = embedding
        
        return final_embeddings
    
    
    def _handle_key_failure(self, key_index: int, error: Exception):
        """Handle key failure and track for failover"""
        self.key_failure_counts[key_index] += 1
        
        # Mark key as failed after 3 consecutive failures (more tolerant for free tier)
        if self.key_failure_counts[key_index] >= 3:
            self.failed_keys.add(key_index)
            logger.warning(f"Key {key_index} marked as failed after {self.key_failure_counts[key_index]} failures")
        
        # Log failure statistics
        active_keys = len(self.api_keys) - len(self.failed_keys)
        logger.info(f"Active keys: {active_keys}/{len(self.api_keys)}")
    
    

def create_voyage_multi_key_embedder(api_keys: List[str], model: str = None, dimension: int = None) -> VoyageMultiKeyEmbedder:

    if model is None or dimension is None:
        from config.embedding import get_embedding_settings
        config = get_embedding_settings()
        model = model or config.embed_model
        dimension = dimension or config.embed_dimension
    
    return VoyageMultiKeyEmbedder(api_keys=api_keys, model=model, dimension=dimension)
