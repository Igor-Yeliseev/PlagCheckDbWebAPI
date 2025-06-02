from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import relationship
from app.models.document import Base

class DocEmbedding(Base):
    """
    Модель для хранения эмбеддингов документов для быстрого поиска
    """
    __tablename__ = 'doc_embeddings'
    
    document_id = Column(UUID(as_uuid=True), ForeignKey('"Documents"."Id"'), primary_key=True, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    
    # Relationship с основной таблицей документов
    # document = relationship("Document")  # если есть модель Document
    
    def __repr__(self):
        return f"<DocEmbedding(document_id={self.document_id})>" 