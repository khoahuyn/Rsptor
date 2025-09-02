import logging
import numpy as np
from typing import List, Dict, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("vector_index")

try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("✅ FAISS available for vector indexing")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️ FAISS not available - falling back to brute force search")


class VectorIndex:
    """
    Fast vector similarity search with FAISS
    Falls back to numpy if FAISS not available
    """
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = None
        self.chunk_ids = []
        self.metadata = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        if FAISS_AVAILABLE:
            # Use FAISS IndexFlatIP for cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"🚀 FAISS index created with dimension {dimension}")
        else:
            # Fallback to numpy storage
            self.vectors = []
            logger.info(f"📦 Using numpy fallback for dimension {dimension}")
    
    def add_vectors(self, vectors: List[List[float]], chunk_ids: List[str], metadata: List[Dict]):
        """Add vectors to index"""
        try:
            if not vectors:
                return
            
            # Normalize vectors for cosine similarity
            vectors_array = np.array(vectors, dtype=np.float32)
            
            # L2 normalize for cosine similarity
            norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized_vectors = vectors_array / norms
            
            if FAISS_AVAILABLE and self.index is not None:
                # Add to FAISS index
                self.index.add(normalized_vectors)
            else:
                # Add to numpy storage
                if not self.vectors:
                    self.vectors = normalized_vectors
                else:
                    self.vectors = np.vstack([self.vectors, normalized_vectors])
            
            # Store metadata
            self.chunk_ids.extend(chunk_ids)
            self.metadata.extend(metadata)
            
            logger.info(f"📦 Added {len(vectors)} vectors to index (total: {len(self.chunk_ids)})")
            
        except Exception as e:
            logger.error(f"Error adding vectors to index: {e}")
    
    async def search_async(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Async vector similarity search"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor, 
                self.search, 
                query_vector, 
                top_k
            )
        except Exception as e:
            logger.error(f"Async search failed: {e}")
            return []
    
    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors"""
        try:
            if not self.chunk_ids:
                return []
            
            # Normalize query vector
            query_array = np.array([query_vector], dtype=np.float32)
            norm = np.linalg.norm(query_array)
            if norm > 0:
                query_array = query_array / norm
            
            if FAISS_AVAILABLE and self.index is not None:
                # FAISS search
                similarities, indices = self.index.search(query_array, min(top_k, len(self.chunk_ids)))
                
                results = []
                for sim, idx in zip(similarities[0], indices[0]):
                    if idx >= 0 and idx < len(self.chunk_ids):  # Valid index
                        results.append((
                            self.chunk_ids[idx],
                            float(sim),  # Convert numpy float to Python float
                            self.metadata[idx]
                        ))
                
            else:
                # Numpy fallback
                if len(self.vectors) == 0:
                    return []
                
                # Compute cosine similarities
                similarities = np.dot(self.vectors, query_array.T).flatten()
                
                # Get top_k indices
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    if idx < len(self.chunk_ids):
                        results.append((
                            self.chunk_ids[idx],
                            float(similarities[idx]),
                            self.metadata[idx]
                        ))
            
            logger.debug(f"🔍 Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def clear(self):
        """Clear the index"""
        try:
            if FAISS_AVAILABLE and self.index is not None:
                self.index.reset()
            else:
                self.vectors = []
            
            self.chunk_ids = []
            self.metadata = []
            logger.info("🗑️ Vector index cleared")
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
    
    def size(self) -> int:
        """Get number of vectors in index"""
        return len(self.chunk_ids)
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_vectors": len(self.chunk_ids),
            "dimension": self.dimension,
            "backend": "faiss" if FAISS_AVAILABLE and self.index is not None else "numpy",
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        try:
            vector_size = len(self.chunk_ids) * self.dimension * 4  # 4 bytes per float32
            metadata_size = len(self.metadata) * 500  # Rough estimate
            return (vector_size + metadata_size) / (1024 * 1024)
        except:
            return 0.0


# Global index instances (can be multiple for different KBs)
vector_indexes: Dict[str, VectorIndex] = {}


def get_vector_index(kb_id: str, dimension: int = 1024) -> VectorIndex:
    """Get or create vector index for KB"""
    key = f"{kb_id}_{dimension}"
    if key not in vector_indexes:
        vector_indexes[key] = VectorIndex(dimension)
    return vector_indexes[key]


def clear_vector_index(kb_id: str, dimension: int = 1024):
    """Clear vector index for KB"""
    key = f"{kb_id}_{dimension}"
    if key in vector_indexes:
        vector_indexes[key].clear()
        del vector_indexes[key]

