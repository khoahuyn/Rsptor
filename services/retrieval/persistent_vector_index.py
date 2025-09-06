import os
import pickle
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger("persistent_vector_index")


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


class PersistentVectorIndex(VectorIndex):
    """FAISS Vector Index with disk persistence"""
    
    def __init__(self, kb_id: str, dimension: int = 1024):
        super().__init__(dimension)
        self.kb_id = kb_id
        
        # Create indexes directory
        self.index_dir = Path("indexes")
        self.index_dir.mkdir(exist_ok=True)
        
        # File paths
        safe_kb_id = kb_id.replace("::", "_").replace(":", "_")  # Safe filename
        self.faiss_path = self.index_dir / f"{safe_kb_id}_{dimension}.faiss"
        self.metadata_path = self.index_dir / f"{safe_kb_id}_{dimension}_meta.pkl"
        
        logger.info(f"🗂️ Persistent index: {safe_kb_id} (dim={dimension})")
        logger.info(f"📁 Index files: {self.faiss_path.name}")
    
    def save_to_disk(self) -> bool:
        """Save FAISS index and metadata to disk"""
        try:
            if not FAISS_AVAILABLE or self.index is None:
                logger.warning("❌ Cannot save: FAISS not available or no index")
                return False
                
            if len(self.chunk_ids) == 0:
                logger.warning("❌ Cannot save: Empty index")
                return False
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.faiss_path))
            logger.info(f"💾 Saved FAISS index: {self.faiss_path}")
            
            # Save metadata
            metadata = {
                'chunk_ids': self.chunk_ids,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'kb_id': self.kb_id,
                'size': len(self.chunk_ids),
                'version': '1.0'  # For future compatibility
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"💾 Saved metadata: {len(self.chunk_ids)} vectors to {self.metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
            return False
    
    def load_from_disk(self) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            # Check if files exist
            if not (self.faiss_path.exists() and self.metadata_path.exists()):
                logger.info(f"📁 No existing index files found")
                return False
            
            if not FAISS_AVAILABLE:
                logger.warning("❌ Cannot load: FAISS not available")
                return False
                
            # Load metadata first
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
            # Validate metadata
            if metadata.get('dimension') != self.dimension:
                logger.warning(f"❌ Dimension mismatch: {metadata.get('dimension')} vs {self.dimension}")
                return False
                
            if metadata.get('kb_id') != self.kb_id:
                logger.warning(f"❌ KB ID mismatch: {metadata.get('kb_id')} vs {self.kb_id}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.faiss_path))
            
            # Restore metadata
            self.chunk_ids = metadata['chunk_ids']
            self.metadata = metadata['metadata']
            
            logger.info(f"📁 Loaded index: {len(self.chunk_ids)} vectors from disk")
            logger.info(f"✅ Index ready for search")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Load failed: {e}")
            # Clean up partial state
            self.chunk_ids = []
            self.metadata = []
            self.index = faiss.IndexFlatIP(self.dimension) if FAISS_AVAILABLE else None
            return False
    
    def is_index_stale(self, current_embedding_count: int, tolerance: int = 5) -> bool:
        """Check if index needs rebuild based on embedding count"""
        try:
            if not self.metadata_path.exists():
                logger.info("🔄 Index stale: No metadata file")
                return True
                
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
            saved_count = metadata.get('size', 0)
            diff = abs(current_embedding_count - saved_count)
            
            if diff > tolerance:
                logger.info(f"🔄 Index stale: DB={current_embedding_count}, Index={saved_count} (diff={diff})")
                return True
            else:
                logger.info(f"✅ Index fresh: DB={current_embedding_count}, Index={saved_count} (diff={diff})")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Cannot check staleness: {e}")
            return True
    
    def get_stats(self) -> Dict:
        """Enhanced stats including persistence info"""
        stats = super().get_stats()
        
        # Add persistence info
        stats.update({
            "kb_id": self.kb_id,
            "faiss_file_exists": self.faiss_path.exists(),
            "metadata_file_exists": self.metadata_path.exists(),
            "files_size_mb": self._get_files_size_mb()
        })
        
        return stats
    
    def _get_files_size_mb(self) -> float:
        """Get total size of index files in MB"""
        total_size = 0
        
        if self.faiss_path.exists():
            total_size += self.faiss_path.stat().st_size
            
        if self.metadata_path.exists():
            total_size += self.metadata_path.stat().st_size
            
        return round(total_size / (1024 * 1024), 2)
    
    def clear_disk_files(self) -> bool:
        """Remove index files from disk"""
        try:
            removed = []
            
            if self.faiss_path.exists():
                self.faiss_path.unlink()
                removed.append(self.faiss_path.name)
                
            if self.metadata_path.exists():
                self.metadata_path.unlink()
                removed.append(self.metadata_path.name)
            
            if removed:
                logger.info(f"🗑️ Removed index files: {', '.join(removed)}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear disk files: {e}")
            return False


# Global persistent index instances (replaces old vector_indexes)
persistent_indexes: Dict[str, PersistentVectorIndex] = {}


def get_persistent_vector_index(kb_id: str, dimension: int = 1024) -> PersistentVectorIndex:
    """Get or create persistent vector index for KB (replaces get_vector_index)"""
    key = f"{kb_id}_{dimension}"
    if key not in persistent_indexes:
        persistent_indexes[key] = PersistentVectorIndex(kb_id, dimension)
    return persistent_indexes[key]


def clear_persistent_vector_index(kb_id: str, dimension: int = 1024):
    """Clear persistent vector index for KB (replaces clear_vector_index)"""
    key = f"{kb_id}_{dimension}"
    if key in persistent_indexes:
        persistent_indexes[key].clear()
        persistent_indexes[key].clear_disk_files()  # Also remove disk files
        del persistent_indexes[key]


# Factory function for easy usage (backward compatibility)
def create_persistent_index(kb_id: str, dimension: int = 1024) -> PersistentVectorIndex:
    """Create persistent vector index for KB"""
    return PersistentVectorIndex(kb_id, dimension)
