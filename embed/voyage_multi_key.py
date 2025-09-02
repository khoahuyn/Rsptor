import asyncio
import itertools
import logging
import time
from typing import List, Dict, Any
import voyageai
from .voyage_config import VoyageConfig

logger = logging.getLogger(__name__)


class VoyageMultiKeyEmbedder:
    """
    Multi-key Voyage AI embedder with round-robin load balancing
    Allows using multiple API keys to bypass rate limits and achieve higher throughput
    """
    
    def __init__(self, api_keys: List[str], model: str = "voyage-context-3", dimension: int = 1024):
        """
        Initialize multi-key embedder
        
        Args:
            api_keys: List of Voyage AI API keys
            model: Voyage model name
            dimension: Output dimension
        """
        if not api_keys:
            raise ValueError("At least one API key is required")
        
        self.api_keys = api_keys
        self.model = model
        self.dimension = dimension
        
        # Create clients for each key
        self.clients = [voyageai.Client(api_key=key) for key in api_keys]
        
        # Round-robin key selector
        self.key_cycle = itertools.cycle(range(len(api_keys)))
        
        # Rate limiting per key (independent tracking)
        self.last_request_times = [0] * len(api_keys)
        
        # Error tracking for failover
        self.failed_keys = set()  # Track failed keys
        self.key_failure_counts = [0] * len(api_keys)  # Track failure counts
        
        # Config for each key
        self.config = VoyageConfig.get_optimized_config(is_production=False)
        
        logger.info(f"Initialized Voyage multi-key embedder with {len(api_keys)} keys")
    
    def _get_next_key_index(self) -> int:
        """Get next available key using round-robin, skipping failed keys"""
        max_attempts = len(self.api_keys) * 2  # Prevent infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            key_index = next(self.key_cycle)
            
            # Skip failed keys (unless all are failed)
            if key_index not in self.failed_keys or len(self.failed_keys) >= len(self.api_keys):
                return key_index
                
            attempts += 1
        
        # Fallback: reset failed keys and try again
        logger.warning("All keys failed, resetting failure tracking")
        self.failed_keys.clear()
        self.key_failure_counts = [0] * len(self.api_keys)
        return next(self.key_cycle)
    
    def _apply_rate_limiting(self, key_index: int):
        """Apply rate limiting for specific key"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_times[key_index]
        
        # With multiple keys, we can be more aggressive
        # Single key: 45s delay for 2 RPM
        # Multi-key: Proportionally less delay per key
        required_delay = VoyageConfig.BATCH_DELAY_SECONDS / len(self.api_keys)
        
        if self.last_request_times[key_index] > 0 and time_since_last < required_delay:
            wait_time = required_delay - time_since_last
            logger.debug(f"Key {key_index}: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        self.last_request_times[key_index] = time.time()
    
    async def embed_texts_parallel(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts using multiple keys in parallel
        Splits texts across available keys for maximum throughput
        """
        if not texts:
            return []
        
        # Split texts across available keys
        texts_per_key = len(texts) // len(self.api_keys)
        remainder = len(texts) % len(self.api_keys)
        
        tasks = []
        start_idx = 0
        
        for i, (client, api_key) in enumerate(zip(self.clients, self.api_keys)):
            # Calculate texts for this key
            end_idx = start_idx + texts_per_key + (1 if i < remainder else 0)
            key_texts = texts[start_idx:end_idx]
            
            if key_texts:
                task = self._embed_with_key(client, api_key, key_texts, i)
                tasks.append(task)
            
            start_idx = end_idx
        
        # Execute all keys in parallel
        results = await asyncio.gather(*tasks)
        
        # Combine results in order
        all_embeddings = []
        for embeddings in results:
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    async def _embed_with_key(self, client: voyageai.Client, api_key: str, texts: List[str], key_index: int) -> List[List[float]]:
        """Embed texts using specific key"""
        from utils import token_count
        
        estimated_tokens = sum(token_count(text) for text in texts)
        logger.info(f"Key {key_index}: {len(texts)} texts, {estimated_tokens} tokens")
        
        # Apply rate limiting for this key
        self._apply_rate_limiting(key_index)
        
        try:
            # Use single batch approach for simplicity
            # Format input as List[List[str]] where each text is its own document
            inputs = [[text] for text in texts]
            
            # Make API call
            result = client.contextualized_embed(
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
            
            logger.info(f"Key {key_index}: Successfully embedded {len(embeddings)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Key {key_index} failed: {e}")
            self._handle_key_failure(key_index, e)
            raise
    
    def embed_texts_batch_balanced(self, texts: List[str], max_batch_size: int = 64, smart_batching: bool = True) -> List[List[float]]:
        """
        Embed texts using load-balanced batching across keys
        Smart batching optimizes for large documents and rate limits
        """
        if not texts:
            return []
        
        # Smart batching for large documents
        inter_batch_delay = 0.0  # Default no delay
        if smart_batching and len(texts) > 200:
            logger.info(f"Large document detected ({len(texts)} chunks), using smart batching")
            # Use smaller batches to stay within TPM limits
            max_batch_size = min(max_batch_size, 32)
            
            # Add inter-batch delay for large documents to respect TPM limits
            inter_batch_delay = 0.1  # 100ms delay between batches
        
        all_embeddings = []
        
        # Process in batches, rotating keys
        for i in range(0, len(texts), max_batch_size):
            batch_texts = texts[i:i + max_batch_size]
            
            # Get next key for this batch
            key_index = self._get_next_key_index()
            client = self.clients[key_index]
            
            # Apply rate limiting
            self._apply_rate_limiting(key_index)
            
            try:
                # Format input
                inputs = [[text] for text in batch_texts]
                
                # Make API call
                result = client.contextualized_embed(
                    inputs=inputs,
                    model=self.model,
                    input_type="document",
                    output_dimension=self.dimension
                )
                
                # Extract embeddings
                batch_embeddings = []
                for doc_result in result.results:
                    if doc_result.embeddings:
                        batch_embeddings.append(doc_result.embeddings[0])
                    else:
                        batch_embeddings.append([0.0] * self.dimension)
                
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Key {key_index}: Batch {i//max_batch_size + 1} completed ({len(batch_embeddings)} embeddings)")
                
                # Smart delay for large documents
                if smart_batching and len(texts) > 200 and inter_batch_delay > 0:
                    time.sleep(inter_batch_delay)
                
            except Exception as e:
                logger.error(f"Key {key_index} batch failed: {e}")
                self._handle_key_failure(key_index, e)
                
                # Try failover to another key
                if len(self.failed_keys) < len(self.api_keys):
                    logger.info(f"Attempting failover for batch {i//max_batch_size + 1}")
                    try:
                        # Get alternative key
                        alt_key_index = self._get_next_key_index()
                        alt_client = self.clients[alt_key_index]
                        
                        # Retry with alternative key
                        self._apply_rate_limiting(alt_key_index)
                        inputs = [[text] for text in batch_texts]
                        result = alt_client.contextualized_embed(
                            inputs=inputs,
                            model=self.model,
                            input_type="document",
                            output_dimension=self.dimension
                        )
                        
                        # Process successful result
                        batch_embeddings = []
                        for doc_result in result.results:
                            if doc_result.embeddings:
                                batch_embeddings.append(doc_result.embeddings[0])
                            else:
                                batch_embeddings.append([0.0] * self.dimension)
                        
                        all_embeddings.extend(batch_embeddings)
                        logger.info(f"Failover successful: Key {alt_key_index} completed batch {i//max_batch_size + 1}")
                        continue  # Continue to next batch
                        
                    except Exception as failover_error:
                        logger.error(f"Failover also failed: {failover_error}")
                
                # If all keys failed or no failover possible
                raise
        
        return all_embeddings
    
    def _handle_key_failure(self, key_index: int, error: Exception):
        """Handle key failure and track for failover"""
        self.key_failure_counts[key_index] += 1
        
        # Mark key as failed after 2 consecutive failures
        if self.key_failure_counts[key_index] >= 2:
            self.failed_keys.add(key_index)
            logger.warning(f"Key {key_index} marked as failed after {self.key_failure_counts[key_index]} failures")
        
        # Log failure statistics
        active_keys = len(self.api_keys) - len(self.failed_keys)
        logger.info(f"Active keys: {active_keys}/{len(self.api_keys)}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all keys"""
        return {
            "total_keys": len(self.api_keys),
            "active_keys": len(self.api_keys) - len(self.failed_keys),
            "failed_keys": list(self.failed_keys),
            "failure_counts": self.key_failure_counts,
            "health_percentage": (len(self.api_keys) - len(self.failed_keys)) / len(self.api_keys) * 100
        }


def create_voyage_multi_key_embedder(api_keys: List[str], model: str = None, dimension: int = None) -> VoyageMultiKeyEmbedder:

    if model is None or dimension is None:
        from config.embedding import get_embedding_settings
        config = get_embedding_settings()
        model = model or config.embed_model
        dimension = dimension or config.embed_dimension
    
    return VoyageMultiKeyEmbedder(api_keys=api_keys, model=model, dimension=dimension)
