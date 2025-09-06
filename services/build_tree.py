import asyncio
import logging
import numpy as np
import time
import umap
from sklearn.mixture import GaussianMixture
from typing import List, Tuple, Optional, Callable

from llm.async_summary import summarize_cluster_from_contents_async
from models import RaptorParams



logger = logging.getLogger(__name__)


class BuildTree:

    
    def __init__(
        self, 
        max_clusters: int = None,           
        similarity_threshold: float = None, # Will use config default (renamed from threshold)
        max_token: int = None,              
        random_seed: int = None,            
        max_levels: int = None,             
        llm_concurrency: int = None,        # Will use config default (RAGFlow inspired)
    ):

        from config.raptor import get_raptor_settings
        from config.llm import get_llm_settings
        from config.embedding import get_embedding_settings
        from embed.embedding import get_provider_config
        
        self.raptor_config = get_raptor_settings()  # Store as instance variable
        llm_config = get_llm_settings()
        self.embed_config = get_embedding_settings()
        
        # Use config defaults unless overridden
        self.max_clusters = max_clusters or self.raptor_config.max_clusters
        self.threshold = similarity_threshold or self.raptor_config.similarity_threshold  # Renamed for consistency
        self.max_token = max_token or llm_config.summary_max_tokens  # Use LLM summary token limit
        self.random_seed = random_seed or self.raptor_config.random_seed
        self.max_levels = max_levels or self.raptor_config.max_levels
        self.llm_concurrency = llm_concurrency or llm_config.llm_concurrency
        self.layers_built = 0
        
        
        # Log configuration being used
        logger.info(f"🎯 RAPTOR Config: clusters={self.max_clusters}, threshold={self.threshold}, tokens={self.max_token}")
        
    def _get_optimal_clusters(self, embeddings: np.ndarray, random_state: int):
        """RAGFlow Line 78-87: BIC optimal cluster selection"""
        max_clusters = min(self.max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        optimal_clusters = n_clusters[np.argmin(bics)]
        return optimal_clusters
        
    async def _summarize_cluster(self, cluster_texts: List[str], cluster_id: int = 0) -> str:
        """
        Summarize a cluster of texts using optimized async LLM
        """
        # Use async optimized summarization
        try:
            # Generate dummy member IDs for the function
            member_ids = [f"chunk_{i}" for i in range(len(cluster_texts))]
            
            # Create minimal RaptorParams for the function
            raptor_params = RaptorParams(
                max_clusters=self.max_clusters,
                similarity_threshold=self.threshold,
                max_levels=self.max_levels,  # Use config value, not hardcoded!
                random_seed=self.random_seed,
                enable_summary=True,  # Enable summarization
                summary_max_tokens=self.max_token  # Use class max_token
            )
            
            # 🚀 USE ASYNC OPTIMIZED VERSION
            summary = await summarize_cluster_from_contents_async(
                member_contents=cluster_texts,
                member_ids=member_ids,
                p=raptor_params,
                cluster_id=cluster_id
            )
            logger.debug(f"📝 Generated summary: {summary[:100]}...")
            return summary
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            # Fallback: concatenate first sentences  
            fallback = ". ".join([text.split('.')[0] for text in cluster_texts[:3]])
            return fallback + "..."
    
    async def _embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text with RAGFlow-style caching  + timeout protection
        """
        import asyncio
        
        # RAGFlow-style timeout protection (direct async call)
        try:
            # Timeout protection (using RAPTOR-specific longer timeout)
            from embed.embedding_constants import EmbeddingConstants
            from embed.embedding import embed_text_cached
            embedding = await asyncio.wait_for(
                embed_text_cached(text, self.embed_config.embed_dimension, cfg=self.embed_config),
                timeout=EmbeddingConstants.RAPTOR_TIMEOUT
            )
            return np.array(embedding) if len(embedding) > 0 else np.zeros(self.embed_config.embed_dimension)
            
        except asyncio.TimeoutError:
            logger.error(f"Embedding timeout for text: {text[:50]}...")
            raise Exception(f"Embedding timeout after {EmbeddingConstants.RAPTOR_TIMEOUT} seconds")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
    
    async def __call__(
        self, 
        chunks: List[Tuple[str, np.ndarray]], 
        callback: Optional[Callable] = None
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Build RAPTOR tree and return augmented chunks
        """
        # 🕐 START OVERALL TIMER
        overall_start_time = time.time()
        
        if len(chunks) <= 1:
            logger.info("🛑 Only 1 chunk, no tree building needed")
            return chunks
            
        # Filter valid chunks
        preprocessing_start = time.time()
        valid_chunks = [(content, emb) for content, emb in chunks 
                       if content and emb is not None and len(emb) > 0]
        
        if len(valid_chunks) != len(chunks):
            logger.warning(f"⚠️ Filtered {len(chunks) - len(valid_chunks)} invalid chunks")
            
        preprocessing_time = time.time() - preprocessing_start
        logger.info(f"🌱 RAPTOR START: {len(valid_chunks)} chunks (preprocessing: {preprocessing_time:.2f}s)")
        
        # Track all chunks (original + summaries) with level information
        # Original chunks are level 0, summaries start from level 1
        all_chunks = [(content, emb, 0) for content, emb in valid_chunks]
        layers = [(0, len(all_chunks))]  # (start_idx, end_idx) for each layer
        
        start, end = 0, len(all_chunks)
        layer_num = 0
        
        # Build tree layer by layer (RAGFlow approach)
        while end - start > 1 and layer_num < self.max_levels:
            
            # 🕐 LAYER TIMER
            layer_start_time = time.time()
            layer_num += 1
            current_chunks = all_chunks[start:end]
            embeddings = np.array([emb for _, emb, _ in current_chunks])
            
            logger.info(f"🔧 Building Layer {layer_num}: {len(current_chunks)} chunks")
            
            # Special case: only 2 chunks
            if len(embeddings) == 2:
                # 🚀 SPECIAL CASE: 2 chunks direct processing
                two_chunk_start = time.time()
                texts = [content for content, _, _ in current_chunks]
                
                summary = await self._summarize_cluster(texts, cluster_id=0)
                summary_embedding = await self._embed_text(summary)  # Single call OK here
                # Include level information for raptor_builder
                all_chunks.append((summary, summary_embedding, layer_num))
                
                two_chunk_time = time.time() - two_chunk_start
                layer_time = time.time() - layer_start_time
                logger.info(f"✅ LAYER {layer_num} (2-chunk): {layer_time:.2f}s total, processing: {two_chunk_time:.2f}s")
                
                if callback:
                    callback(msg=f"Layer {layer_num}: {len(current_chunks)} → 1 clusters")
                    
                layers.append((end, len(all_chunks)))
                start = end
                end = len(all_chunks)
                continue
            
            # UMAP dimensionality reduction (RAGFlow formula - INLINE for performance)
            n_neighbors = int((len(embeddings) - 1) ** 0.8)
            n_components = min(self.raptor_config.umap_n_components, len(embeddings) - 2)  # Use config value
            
            logger.debug(f"📐 UMAP: {len(embeddings[0])}→{n_components} dims (n_neighbors={n_neighbors})")
            
            reduced_embeddings = umap.UMAP(
                n_neighbors=max(2, n_neighbors),
                n_components=n_components,
                metric=self.raptor_config.umap_metric,  # Use config value
                random_state=self.random_seed
            ).fit_transform(embeddings)
            
            # Find optimal clusters using BIC (RAGFlow exact approach)
            n_clusters = self._get_optimal_clusters(reduced_embeddings, self.random_seed)
            logger.debug(f"🎯 BIC optimal clusters: {n_clusters}")
            
            # GMM clustering with probabilistic assignment (RAGFlow exact Line 152-160)
            if n_clusters == 1:
                lbls = [0 for _ in range(len(reduced_embeddings))]
            else:
                gm = GaussianMixture(n_components=n_clusters, random_state=self.random_seed)
                gm.fit(reduced_embeddings)
                probs = gm.predict_proba(reduced_embeddings)
                lbls = [np.where(prob > self.threshold)[0] for prob in probs]
                lbls = [lbl[0] if isinstance(lbl, np.ndarray) and len(lbl) > 0 else 
                       np.argmax(prob) for lbl, prob in zip(lbls, probs)]
            
            # Use lbls as labels for compatibility
            labels = lbls
            logger.info(f"🎯 GMM: {len(reduced_embeddings)} chunks → {n_clusters} clusters")
            
            # 📝 STEP 2: LLM SUMMARIZATION PHASE
            logger.info(f"📝 LLM SUMMARIZATION: {n_clusters} clusters with {self.llm_concurrency} max concurrent")
            llm_start_time = time.time()
            
            # Create semaphore for concurrency control
            sem = asyncio.Semaphore(max(1, self.llm_concurrency))
            
            async def summarize_single_cluster_asyncio(cluster_id):
                """Asyncio-compatible cluster summarizer with concurrency control - SUMMARIES ONLY"""
                async with sem:  # Raptor-service pattern: limit concurrent LLM calls
                    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                    cluster_texts = [current_chunks[i][0] for i in cluster_indices]  # [0] is content, same for 3-tuples
                    
                    if not cluster_texts:
                        return None
                    
                    summary = await self._summarize_cluster(cluster_texts, cluster_id=cluster_id)
                    logger.debug(f"📝 Cluster {cluster_id}: {len(cluster_texts)} chunks → summary")
                    return summary  # ✅ Return summary only, embed later in parallel
            
            # 🚀 PHASE 1: Generate all summaries in parallel
            logger.info(f"🚀 Phase 1: Generating {n_clusters} summaries in parallel")
            cluster_tasks = [summarize_single_cluster_asyncio(cluster_id) for cluster_id in range(n_clusters)]
            summary_results = await asyncio.gather(*cluster_tasks)
            summaries = [result for result in summary_results if result is not None]
            
            # 🚀 STEP 3: EMBEDDING PHASE
            embed_start_time = time.time()
            logger.info(f"🚀 EMBEDDING: {len(summaries)} summaries in TRUE PARALLEL")
            if summaries:
                # 🚀 BATCH EMBEDDING: All summaries at once for maximum parallelism
                from embed.embedding import embed_texts_batch
                embeddings = await embed_texts_batch(summaries, self.embed_config.embed_dimension, self.embed_config)
                
                # Convert back to numpy arrays for consistency
                embeddings = [np.array(emb) for emb in embeddings]
                
                # Combine summaries with embeddings
                new_summaries = list(zip(summaries, embeddings))
            else:
                new_summaries = []
            
            embed_time = time.time() - embed_start_time
            logger.info(f"✅ EMBEDDING COMPLETED: {len(new_summaries)} summaries in {embed_time:.2f}s")
            
            llm_time = time.time() - llm_start_time
            logger.info(f"✅ LLM SUMMARIZATION COMPLETED: {len(new_summaries)} summaries in {llm_time:.2f}s")
            
            # Add summaries to all_chunks with level information
            summaries_with_level = [(summary, embedding, layer_num) for summary, embedding in new_summaries]
            all_chunks.extend(summaries_with_level)
            
            if callback:
                callback(msg=f"Layer {layer_num}: {end - start} → {len(new_summaries)} clusters")
                
            # Update for next layer
            layers.append((end, len(all_chunks)))
            start = end
            end = len(all_chunks)
            
            # 🕐 LAYER COMPLETION
            layer_time = time.time() - layer_start_time
            logger.info(f"✅ LAYER {layer_num} COMPLETED: {layer_time:.2f}s total")
            # Log breakdown if timing variables are available
            breakdown_parts = []
            try:
                clustering_total_var = locals().get('clustering_total')
                if clustering_total_var is not None:
                    breakdown_parts.append(f"Clustering: {clustering_total_var:.2f}s")
            except (NameError, TypeError):
                pass
            try:
                llm_time_var = locals().get('llm_time')
                if llm_time_var is not None:
                    breakdown_parts.append(f"LLM: {llm_time_var:.2f}s") 
            except (NameError, TypeError):
                pass
            try:
                embed_time_var = locals().get('embed_time')
                if embed_time_var is not None:
                    breakdown_parts.append(f"Embedding: {embed_time_var:.2f}s")
            except (NameError, TypeError):
                pass
            if breakdown_parts:
                logger.info(f"   📊 Layer breakdown - {', '.join(breakdown_parts)}")
            logger.info(f"   📈 Progress: {len(valid_chunks)} → {len(all_chunks)} total chunks")
            
            # 🚦 KEY ROTATION: Random key selection should prevent rate limits
            # Inter-layer cooldown removed - using smart key rotation instead
            
        # Check if we reached max levels (safety guard - not in RAGFlow but good practice)
        if layer_num >= self.max_levels:
            logger.warning(f"⚠️ Reached maximum tree levels ({self.max_levels}) - stopping for safety (not RAGFlow behavior)")
            
        # 🎉 OVERALL COMPLETION
        overall_time = time.time() - overall_start_time
        logger.info(f"🎉 RAPTOR TREE COMPLETED: {len(valid_chunks)} → {len(all_chunks)} total chunks ({layer_num} layers)")
        logger.info(f"⏱️  TOTAL TIME: {overall_time:.2f}s")
        
        # Calculate summary stats
        total_clusters_created = len(all_chunks) - len(valid_chunks)
        avg_layer_time = overall_time / max(layer_num, 1)
        logger.info(f"📊 SUMMARY: {total_clusters_created} summaries created, avg {avg_layer_time:.2f}s/layer")
        
        self.layers_built = layer_num
        return all_chunks


async def build_tree(
    chunks: List[Tuple[str, np.ndarray]], 
    max_clusters: int = None,       
    threshold: float = None,          
    random_seed: int = None,        
    max_levels: int = None,         
    llm_concurrency: int = None,    # Will use config default (RAGFlow inspired)
    callback: Optional[Callable] = None
) -> dict:

    tree_builder = BuildTree(
        max_clusters=max_clusters,           
        similarity_threshold=threshold,      
        max_token=None,                        
        random_seed=random_seed,             
        max_levels=max_levels,               
        llm_concurrency=llm_concurrency,     # User-specified parameter
    )
    
    augmented_chunks = await tree_builder(chunks, callback)
    
    # Return dict format expected by raptor_builder
    return {
        "augmented_chunks": augmented_chunks,
        "tree_levels": tree_builder.layers_built if hasattr(tree_builder, 'layers_built') else len(augmented_chunks) - len(chunks),
        "original_count": len(chunks),
        "summary_count": len(augmented_chunks) - len(chunks),
        "total_count": len(augmented_chunks)
    }




