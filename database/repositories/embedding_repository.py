from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, text
from sqlalchemy.dialects.postgresql import insert

from .base import BaseRepository
from models.database.embedding import EmbeddingORM, EmbeddingOwnerType


class EmbeddingRepository(BaseRepository[EmbeddingORM]):
    """Repository for vector embeddings with pgvector support"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, EmbeddingORM)
    
    async def store_embeddings(
        self,
        tenant_id: str,
        kb_id: str,
        owner_type: EmbeddingOwnerType,
        owner_ids: List[str],
        vectors: List[List[float]],
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EmbeddingORM]:
        """Store multiple embeddings for chunks or tree nodes"""
        try:
            # Get defaults from current embedding config if not provided
            if model is None or dimension is None:
                from config.embedding import get_embedding_settings
                embed_config = get_embedding_settings()
                if model is None:
                    model = embed_config.embed_model
                if dimension is None:
                    dimension = embed_config.embed_dimension
            
            if len(owner_ids) != len(vectors):
                raise ValueError("owner_ids and vectors must have same length")
            
            # Validate vector dimensions
            for i, vector in enumerate(vectors):
                if len(vector) != dimension:
                    raise ValueError(f"Vector {i} has dimension {len(vector)}, expected {dimension}")
            
            # Prepare embedding data
            embeddings_data = []
            for owner_id, vector in zip(owner_ids, vectors):
                embedding_data = {
                    "id": f"{owner_type.value}::{owner_id}",
                    "tenant_id": tenant_id,
                    "kb_id": kb_id,
                    "owner_type": owner_type,
                    "owner_id": owner_id,
                    "model": model,
                    "dimension": dimension,
                    "vector": vector,
                    "meta": metadata or {}
                }
                embeddings_data.append(embedding_data)
            
            # Bulk upsert
            return await self.bulk_upsert_embeddings(embeddings_data)
        except Exception as e:
            await self.session.rollback()
            raise ValueError(f"Failed to store embeddings: {str(e)}")
    
    async def bulk_upsert_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> List[EmbeddingORM]:
        """Bulk upsert embeddings with conflict resolution"""
        try:
            if not embeddings_data:
                return []
            
            stmt = insert(EmbeddingORM).values(embeddings_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=[EmbeddingORM.id],
                set_={
                    "vector": stmt.excluded.vector,
                    "model": stmt.excluded.model,
                    "dimension": stmt.excluded.dimension,
                    "meta": stmt.excluded.meta,
                }
            )
            await self.session.execute(stmt)
            
            # Return created/updated embeddings
            embedding_ids = [emb["id"] for emb in embeddings_data]
            stmt = select(EmbeddingORM).where(EmbeddingORM.id.in_(embedding_ids))
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            await self.session.rollback()
            raise ValueError(f"Failed to bulk upsert embeddings: {str(e)}")
    
    async def similarity_search(
        self,
        query_vector: List[float],
        tenant_id: str,
        kb_id: str,
        owner_type: Optional[EmbeddingOwnerType] = None,
        limit: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[EmbeddingORM, float]]:
        """Perform cosine similarity search using pgvector"""
        try:
            from sqlalchemy.sql import select as sql_select
            
            # pgvector expects the vector as a list/array directly, not a string
            
            # Build SQLAlchemy query 
            stmt = sql_select(
                EmbeddingORM,
                (1 - EmbeddingORM.vector.cosine_distance(query_vector)).label('similarity')
            ).where(
                EmbeddingORM.tenant_id == tenant_id,
                EmbeddingORM.kb_id == kb_id
            )
            
            # Add owner type filter if specified
            if owner_type:
                stmt = stmt.where(EmbeddingORM.owner_type == owner_type)
            
            # Add similarity threshold
            if similarity_threshold > 0:
                stmt = stmt.where(
                    (1 - EmbeddingORM.vector.cosine_distance(query_vector)) >= similarity_threshold
                )
            
            # Add ordering and limit
            stmt = stmt.order_by(EmbeddingORM.vector.cosine_distance(query_vector)).limit(limit)
            
            # Execute query
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            
            # Convert to results format
            results = []
            for row in rows:
                embedding = row[0]  # EmbeddingORM object
                similarity = float(row[1])  # similarity score
                results.append((embedding, similarity))
            
            return results
        except Exception as e:
            raise ValueError(f"Similarity search failed: {str(e)}")
    
    async def get_embeddings_by_owners(
        self,
        tenant_id: str,
        kb_id: str,
        owner_type: EmbeddingOwnerType,
        owner_ids: List[str]
    ) -> Dict[str, EmbeddingORM]:
        """Get embeddings for specific owners, returned as dict[owner_id -> embedding]"""
        try:
            stmt = select(EmbeddingORM).where(
                and_(
                    EmbeddingORM.tenant_id == tenant_id,
                    EmbeddingORM.kb_id == kb_id,
                    EmbeddingORM.owner_type == owner_type,
                    EmbeddingORM.owner_id.in_(owner_ids)
                )
            )
            result = await self.session.execute(stmt)
            embeddings = result.scalars().all()
            
            # Return as dict for easy lookup
            return {emb.owner_id: emb for emb in embeddings}
        except Exception as e:
            raise ValueError(f"Failed to get embeddings by owners: {str(e)}")
    
    async def delete_embeddings_by_owners(
        self,
        tenant_id: str,
        kb_id: str,
        owner_type: EmbeddingOwnerType,
        owner_ids: List[str]
    ) -> int:
        """Delete embeddings for specific owners"""
        try:
            from sqlalchemy import delete
            stmt = delete(EmbeddingORM).where(
                and_(
                    EmbeddingORM.tenant_id == tenant_id,
                    EmbeddingORM.kb_id == kb_id,
                    EmbeddingORM.owner_type == owner_type,
                    EmbeddingORM.owner_id.in_(owner_ids)
                )
            )
            result = await self.session.execute(stmt)
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            raise ValueError(f"Failed to delete embeddings: {str(e)}")
    
    async def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        tenant_id: str,
        kb_id: str,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        limit: int = 10
    ) -> List[Tuple[EmbeddingORM, float]]:
        """Hybrid search combining vector similarity and text matching"""
        try:
            from sqlalchemy import case
            from sqlalchemy.sql import select as sql_select
            
            # Calculate vector similarity (pass query_vector as list directly)
            vector_sim = 1 - EmbeddingORM.vector.cosine_distance(query_vector)
            
            # Calculate text match (simplified - can be enhanced with full-text search)
            text_match = case(
                (EmbeddingORM.meta['content'].astext.ilike(f"%{query_text}%"), 1.0),
                else_=0.0
            )
            
            # Calculate hybrid score
            hybrid_score = (vector_weight * vector_sim + text_weight * text_match).label('hybrid_score')
            
            # Build query
            stmt = sql_select(
                EmbeddingORM,
                hybrid_score
            ).where(
                EmbeddingORM.tenant_id == tenant_id,
                EmbeddingORM.kb_id == kb_id
            ).order_by(hybrid_score.desc()).limit(limit)
            
            # Execute query
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            
            # Convert to results format
            results = []
            for row in rows:
                embedding = row[0]  # EmbeddingORM object
                score = float(row[1])  # hybrid score
                results.append((embedding, score))
            
            return results
        except Exception as e:
            raise ValueError(f"Hybrid search failed: {str(e)}")
    
    async def get_embedding_stats(
        self,
        tenant_id: str,
        kb_id: str
    ) -> Dict[str, Any]:
        """Get statistics about embeddings in a knowledge base"""
        try:
            from sqlalchemy import func
            from sqlalchemy.sql import select as sql_select
            
            # Build SQLAlchemy query
            stmt = sql_select(
                EmbeddingORM.owner_type,
                func.count().label('count'),
                func.avg(EmbeddingORM.dimension).label('avg_dimension')
            ).where(
                EmbeddingORM.tenant_id == tenant_id,
                EmbeddingORM.kb_id == kb_id
            ).group_by(EmbeddingORM.owner_type)
            
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            
            stats = {
                "by_owner_type": {row.owner_type.value: {
                    "count": row.count,
                    "avg_dimension": float(row.avg_dimension or 0)
                } for row in rows},
                "total_embeddings": sum(row.count for row in rows)
            }
            
            return stats
        except Exception as e:
            raise ValueError(f"Failed to get embedding stats: {str(e)}")
