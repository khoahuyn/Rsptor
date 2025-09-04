class EmbeddingConstants:
    """Common embedding constants"""
    
    # Standard dimensions
    STANDARD_DIMENSIONS = [512, 768, 1024, 1536, 2048]
    
    DEFAULT_TIMEOUT = 30         
    RAPTOR_TIMEOUT = 60          # 1min reasonable for single text embedding with retry buffer
    DEFAULT_RETRY_ATTEMPTS = 3   
    
   


