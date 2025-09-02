from sqlalchemy.orm import DeclarativeBase

# Standard vector dimension for embeddings
# VECTOR_DIM constant moved to config/embedding.py - use get_embedding_settings().embed_dimension

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models"""
    pass




