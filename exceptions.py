class RaptorServiceError(Exception):
    """Base exception for RAPTOR service"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class EmbeddingError(RaptorServiceError):
    """Embedding-related errors"""
    pass


class ClusteringError(RaptorServiceError):  
    """Clustering-related errors"""
    pass


class LLMError(RaptorServiceError):
    """LLM-related errors"""
    pass


class ValidationError(RaptorServiceError):
    """Validation-related errors"""
    pass


class ConfigurationError(RaptorServiceError):
    """Configuration-related errors"""
    pass


# Keep backward compatibility with existing EmbedError
class EmbedError(EmbeddingError):
    """Backward compatibility alias for EmbeddingError"""
    pass




