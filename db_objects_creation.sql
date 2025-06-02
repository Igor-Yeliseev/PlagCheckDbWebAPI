-- Create database if it doesn't exist
-- Note: This command should be run from psql prompt separately
-- CREATE DATABASE plag_search_db;

-- Create extension for vector operations (pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create doc_signatures table
CREATE TABLE IF NOT EXISTS doc_signatures (
    document_id UUID PRIMARY KEY REFERENCES "Documents"("Id") ON DELETE CASCADE,
    hashes INTEGER[] NOT NULL
);

-- Create doc_embeddings table
CREATE TABLE IF NOT EXISTS doc_embeddings (
    document_id UUID PRIMARY KEY REFERENCES "Documents"("Id") ON DELETE CASCADE,
    embedding FLOAT[] NOT NULL
);

-- Create indices
CREATE INDEX IF NOT EXISTS idx_doc_id_embeddings ON doc_embeddings(document_id);

-- Create a function to find similar documents based on hash Jaccard similarity
CREATE OR REPLACE FUNCTION hash_similarity(a INTEGER[], b INTEGER[]) 
RETURNS FLOAT AS $$
DECLARE
    intersection INTEGER;
    union_size INTEGER;
BEGIN
    SELECT COUNT(*) INTO intersection FROM (
        SELECT UNNEST(a) AS elem
        INTERSECT
        SELECT UNNEST(b) AS elem
    ) AS intersect_result;
    
    SELECT COUNT(*) INTO union_size FROM (
        SELECT UNNEST(a) AS elem
        UNION
        SELECT UNNEST(b) AS elem
    ) AS union_result;
    
    IF union_size = 0 THEN
        RETURN 0;
    END IF;
    
    RETURN intersection::FLOAT / union_size::FLOAT;
END;
$$ LANGUAGE plpgsql;