import logging
import numpy as np
from typing import List

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger("math_utils")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:

    try:
        np_vec1 = np.array(vec1)
        np_vec2 = np.array(vec2)
        
        # Validate dimensions
        if len(vec1) != len(vec2):
            logger.warning(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
            return 0.0
        
        # Calculate dot product and norms
        dot_product = np.dot(np_vec1, np_vec2)
        norms = np.linalg.norm(np_vec1) * np.linalg.norm(np_vec2)
        
        # Handle zero norm case
        if norms == 0:
            return 0.0
            
        return float(dot_product / norms)
        
    except Exception as e:
        logger.warning(f"Cosine similarity calculation failed: {e}")
        return 0.0


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors
    
    Args:
        vec1: First vector as list of floats
        vec2: Second vector as list of floats
        
    Returns:
        Non-negative float representing Euclidean distance
        Returns float('inf') if calculation fails
    """
    try:
        np_vec1 = np.array(vec1)
        np_vec2 = np.array(vec2)
        
        if len(vec1) != len(vec2):
            logger.warning(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
            return float('inf')
            
        return float(np.linalg.norm(np_vec1 - np_vec2))
        
    except Exception as e:
        logger.warning(f"Euclidean distance calculation failed: {e}")
        return float('inf')


def normalize_vector(vec: List[float]) -> List[float]:
    """
    L2 normalize a vector
    
    Args:
        vec: Input vector as list of floats
        
    Returns:
        Normalized vector as list of floats
        Returns original vector if normalization fails
    """
    try:
        np_vec = np.array(vec)
        norm = np.linalg.norm(np_vec)
        
        if norm == 0:
            return vec  # Return original if zero vector
            
        normalized = np_vec / norm
        return normalized.tolist()
        
    except Exception as e:
        logger.warning(f"Vector normalization failed: {e}")
        return vec


def token_count(text: str) -> int:

    if not text:
        return 0
    
    if tiktoken is None:
        logger.warning("tiktoken not available, using word count estimation")
        return len(text.split())
    
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}")
        return len(text.split())  # Fallback to word count
