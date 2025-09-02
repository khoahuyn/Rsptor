import logging
import functools
from typing import Callable
from fastapi import HTTPException


def handle_service_errors(
    error_type: str = "Service", 
    status_code: int = 500,
    log_level: str = "error"
) -> Callable:
    """
    Generic error handler decorator - consolidates all specific error handlers
    
    Args:
        error_type: Type of error for logging/display (e.g., "Embedding", "LLM")
        status_code: HTTP status code to return (default: 500)
        log_level: Logging level ("error", "warning", "info")
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{error_type} error in {func.__name__}: {str(e)}"
                
                # Log with specified level
                log_func = getattr(logging, log_level.lower(), logging.error)
                log_func(error_msg)
                
                # Return appropriate HTTP exception
                raise HTTPException(
                    status_code=status_code, 
                    detail=f"{error_type} error: {str(e)}"
                )
        return wrapper
    return decorator


# Backward compatible aliases using the generic decorator
def handle_embedding_errors(func: Callable) -> Callable:
    """Handle embedding-related errors - uses generic handler"""
    return handle_service_errors("Embedding", 500)(func)

def handle_llm_errors(func: Callable) -> Callable:
    """Handle LLM-related errors - uses generic handler"""
    return handle_service_errors("LLM", 500)(func)

def handle_clustering_errors(func: Callable) -> Callable:
    """Handle clustering-related errors - uses generic handler"""
    return handle_service_errors("Clustering", 500)(func)

def handle_validation_errors(func: Callable) -> Callable:
    """Handle validation-related errors - uses generic handler"""
    return handle_service_errors("Validation", 400)(func)

def handle_raptor_tree_errors(func: Callable) -> Callable:
    """Handle RAPTOR tree building errors - uses generic handler"""
    return handle_service_errors("RAPTOR tree", 500)(func)


# Decorator aliases for specific error types
embed_errors = handle_embedding_errors
llm_errors = handle_llm_errors
cluster_errors = handle_clustering_errors
validate_errors = handle_validation_errors
tree_errors = handle_raptor_tree_errors


# Common error utility functions
def create_embedding_error(operation: str, error: Exception) -> RuntimeError:
    """Create standardized embedding error - consolidates duplicate error messages"""
    return RuntimeError(f"Failed to {operation}: {str(error)}")

def create_query_embedding_error(error: Exception) -> RuntimeError:
    """Create standardized query embedding error - most common case"""
    return create_embedding_error("embed query", error)

