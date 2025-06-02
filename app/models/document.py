from sqlalchemy import Column, Integer, ForeignKey, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

Base = declarative_base()

class DocSignature(Base):
    """
    Модель DocSignature для хранения сигнатур документов и связи с основным документом
    """
    __tablename__ = 'doc_signatures'
    
    document_id = Column(UUID, ForeignKey('Documents.Id'), primary_key=True, nullable=False)
    hashes = Column(ARRAY(Integer), nullable=False)
    
    # Связь с основной таблицей документов
    document = relationship("Document", backref="signatures")
    
    def __repr__(self):
        return f"<DocSignature(document_id={self.document_id})>" 