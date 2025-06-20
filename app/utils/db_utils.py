from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from app.models.doc_signature import DocSignature, Base
from app.models.doc_embedding import DocEmbedding
from dotenv import load_dotenv

load_dotenv()

# Create SQLAlchemy engine
DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://postgres:2004@localhost:5432/plag_search_db')
engine = create_engine(DATABASE_URI)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_signature_by_id(document_id):
    """Get doc signature by document_id (id = document_id)"""
    db = SessionLocal()
    try:
        return db.query(DocSignature).filter(DocSignature.document_id == document_id).first()
    finally:
        db.close()

def save_signature_to_db(document_id, hashes):
    """Save doc signature to database"""
    db = SessionLocal()
    try:
        signature = DocSignature(document_id=document_id, hashes=hashes)
        db.add(signature)
        db.commit()
        db.refresh(signature)
        return signature
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def get_all_signatures():
    """Get all doc signatures from database"""
    db = SessionLocal()
    try:
        return db.query(DocSignature).all()
    finally:
        db.close()

def get_signatures_by_ids(document_ids):
    """Get doc signatures by list of document_ids (id = document_id)"""
    db = SessionLocal()
    try:
        return db.query(DocSignature).filter(DocSignature.document_id.in_(document_ids)).all()
    finally:
        db.close()

def save_embedding_to_db(document_id, embedding):
    """Save document embedding to database"""
    db = SessionLocal()
    try:
        doc_embedding = DocEmbedding(document_id=document_id, embedding=embedding)
        db.add(doc_embedding)
        db.commit()
        return doc_embedding
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
