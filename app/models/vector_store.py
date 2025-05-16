from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import relationship
from app.models.document import Base

class DocEmbedding(Base):
    """
    Model for storing document embeddings for faster search
    """
    __tablename__ = 'docs_embeddings'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey('"Documents"."Id"'), nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    
    # Relationship с основной таблицей документов
    # document = relationship("Document")  # если есть модель Document
    
    def __repr__(self):
        return f"<DocEmbedding(id={self.id}, document_id={self.document_id})>" 