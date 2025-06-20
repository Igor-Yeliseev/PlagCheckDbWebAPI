from sqlalchemy import Column, Float, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from app.models.doc_signature import Base, Document

class DocEmbedding(Base):
    """
    Модель для хранения эмбеддингов документов для быстрого поиска
    """
    __tablename__ = 'doc_embeddings'
    
    document_id = Column(UUID(as_uuid=True), ForeignKey('Documents.Id'), primary_key=True, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    
    def __repr__(self):
        return f"<DocEmbedding(document_id={self.document_id})>" 