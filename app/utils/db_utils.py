from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from app.models.document import DocSignature, Base
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

def get_signature_by_id(sig_id):
    """Get doc signature by ID"""
    db = SessionLocal()
    try:
        return db.query(DocSignature).filter(DocSignature.id == sig_id).first()
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

def get_signatures_by_ids(sig_ids):
    """Get doc signatures by list of IDs"""
    db = SessionLocal()
    try:
        return db.query(DocSignature).filter(DocSignature.id.in_(sig_ids)).all()
    finally:
        db.close() 