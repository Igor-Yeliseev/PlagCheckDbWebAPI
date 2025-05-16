from sqlalchemy import Column, Integer, ForeignKey, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

Base = declarative_base()

class DocSignature(Base):
    """
    DocSignature model for storing document signatures and link to main document
    """
    __tablename__ = 'doc_signatures'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(UUID, ForeignKey('Documents.Id'), nullable=False)
    hashes = Column(ARRAY(Integer), nullable=False)
    
    # Связь с основной таблицей документов
    document = relationship("Document", backref="signatures")
    
    def __repr__(self):
        return f"<DocSignature(id={self.id}, document_id={self.document_id})>" 