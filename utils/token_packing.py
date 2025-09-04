import logging
from typing import List
import voyageai

logger = logging.getLogger(__name__)


def count_tokens_for_texts(texts: List[str], model: str = "voyage-context-3", api_key: str = None) -> List[int]:
    """Count tokens for each text using Voyage API"""
    if not texts:
        return []
    
    if not api_key:
        raise ValueError("API key required for token counting")
    
    try:
        client = voyageai.Client(api_key=api_key)
        tokenizations = client.tokenize(texts=texts, model=model)
        return [len(enc.ids) for enc in tokenizations]
    except Exception as e:
        logger.warning(f"Token counting failed, using approximation: {e}")
        # Fallback: approximate 4 chars = 1 token
        return [len(text) // 4 + 1 for text in texts]



def count_total_tokens(texts: List[str], model: str = "voyage-context-3", api_key: str = None) -> int:
    """Count total tokens for all texts"""
    if not texts:
        return 0
        
    if not api_key:
        raise ValueError("API key required for token counting")
    
    try:
        client = voyageai.Client(api_key=api_key)
        return client.count_tokens(texts=texts, model=model)
    except Exception as e:
        logger.warning(f"Token counting failed, using approximation: {e}")
        # Fallback: approximate 4 chars = 1 token
        return sum(len(text) // 4 + 1 for text in texts)


def pack_texts_by_token_budget(
    texts: List[str], 
    token_budget: int = 12000,  # More aggressive - closer to 10K limit
    model: str = "voyage-context-3", 
    api_key: str = None
) -> List[List[str]]:

    if not texts:
        return []
    
    if len(texts) == 1:
        return [texts]
    
    logger.info(f"🔢 TOKEN PACKING: {len(texts)} texts with budget {token_budget}")
    
    # Count tokens for each text
    # FAST FALLBACK: Skip token counting overhead for speed
    if len(texts) <= 20:  # Small batches - single group
        logger.info(f"🚀 FAST PACK: {len(texts)} texts → single group (no token counting)")
        return [texts]
    
    try:
        token_counts = count_tokens_for_texts(texts, model, api_key)
    except Exception as e:
        logger.warning(f"Token counting failed: {e}, using fast approximation")
        # Fast approximation: 4 chars = 1 token, pack more aggressively
        token_counts = [max(1, len(text) // 3) for text in texts]  # More aggressive estimate
    
    # Pack into groups
    groups = []
    current_group = []
    current_tokens = 0
    
    for text, token_count in zip(texts, token_counts):
        # Handle oversized single text
        if token_count > token_budget:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0
            groups.append([text])  # Solo group for oversized text
            logger.warning(f"⚠️ Oversized text: {token_count} tokens (budget: {token_budget})")
            continue
        
        # Check if adding this text exceeds budget
        if current_tokens + token_count > token_budget and current_group:
            groups.append(current_group)
            current_group = []
            current_tokens = 0
        
        # Add text to current group
        current_group.append(text)
        current_tokens += token_count
    
    # Add final group if not empty
    if current_group:
        groups.append(current_group)
    
    # Log packing results
    group_sizes = [len(group) for group in groups]
    group_tokens = []
    for group in groups:
        try:
            tokens = count_total_tokens(group, model, api_key)
            group_tokens.append(tokens)
        except Exception:
            group_tokens.append(sum(len(text) // 4 + 1 for text in group))
    
    logger.info(f"📦 PACKED: {len(groups)} groups, sizes={group_sizes}, tokens={group_tokens} (budget={token_budget})")
    
    return groups


