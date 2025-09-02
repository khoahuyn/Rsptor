from typing import TypeVar, Generic, Type, Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from models.database.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations"""
    
    def __init__(self, session: AsyncSession, model: Type[ModelType]):
        self.session = session
        self.model = model
    
    async def create(self, **kwargs) -> ModelType:
        """Create a new record"""
        try:
            obj = self.model(**kwargs)
            self.session.add(obj)
            await self.session.flush()
            await self.session.refresh(obj)
            return obj
        except IntegrityError as e:
            await self.session.rollback()
            raise ValueError(f"Record creation failed: {str(e)}")
    
    async def get_by_id(self, id_value: Any) -> Optional[ModelType]:
        """Get record by primary key"""
        try:
            stmt = select(self.model).where(
                list(self.model.__table__.primary_key.columns)[0] == id_value
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            raise ValueError(f"Failed to get record: {str(e)}")
    
    async def get_many(self, limit: int = 100, offset: int = 0, **filters) -> List[ModelType]:
        """Get multiple records with optional filters"""
        try:
            stmt = select(self.model)
            
            # Apply filters
            for key, value in filters.items():
                if hasattr(self.model, key):
                    stmt = stmt.where(getattr(self.model, key) == value)
            
            stmt = stmt.limit(limit).offset(offset)
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as e:
            raise ValueError(f"Failed to get records: {str(e)}")
    
    async def update_by_id(self, id_value: Any, **kwargs) -> Optional[ModelType]:
        """Update record by primary key"""
        try:
            primary_key = list(self.model.__table__.primary_key.columns)[0]
            stmt = update(self.model).where(primary_key == id_value).values(**kwargs)
            await self.session.execute(stmt)
            
            # Return updated record
            return await self.get_by_id(id_value)
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise ValueError(f"Failed to update record: {str(e)}")
    
    async def delete_by_id(self, id_value: Any) -> bool:
        """Delete record by primary key"""
        try:
            primary_key = list(self.model.__table__.primary_key.columns)[0]
            stmt = delete(self.model).where(primary_key == id_value)
            result = await self.session.execute(stmt)
            return result.rowcount > 0
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise ValueError(f"Failed to delete record: {str(e)}")
    
    async def count(self, **filters) -> int:
        """Count records with optional filters"""
        try:
            from sqlalchemy import func
            stmt = select(func.count()).select_from(self.model)
            
            # Apply filters
            for key, value in filters.items():
                if hasattr(self.model, key):
                    stmt = stmt.where(getattr(self.model, key) == value)
            
            result = await self.session.execute(stmt)
            return result.scalar() or 0
        except SQLAlchemyError as e:
            raise ValueError(f"Failed to count records: {str(e)}")
    
    async def exists(self, **filters) -> bool:
        """Check if record exists with filters"""
        try:
            stmt = select(self.model)
            for key, value in filters.items():
                if hasattr(self.model, key):
                    stmt = stmt.where(getattr(self.model, key) == value)
            
            stmt = stmt.limit(1)
            result = await self.session.execute(stmt)
            return result.first() is not None
        except SQLAlchemyError as e:
            raise ValueError(f"Failed to check existence: {str(e)}")
    
    async def bulk_create(self, records: List[Dict[str, Any]]) -> List[ModelType]:
        """Bulk create records"""
        try:
            objects = [self.model(**record) for record in records]
            self.session.add_all(objects)
            await self.session.flush()
            
            # Refresh all objects to get generated IDs
            for obj in objects:
                await self.session.refresh(obj)
            
            return objects
        except IntegrityError as e:
            await self.session.rollback()
            raise ValueError(f"Bulk creation failed: {str(e)}")
    
    async def commit(self):
        """Commit current transaction"""
        await self.session.commit()
    
    async def rollback(self):
        """Rollback current transaction"""
        await self.session.rollback()



