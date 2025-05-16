import re
# import os
# import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pymorphy2
from datasketch import MinHash, LeanMinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
import dask.bag as db
from app.utils.db_utils import get_all_signatures, get_signatures_by_ids

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize tools
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = set(stopwords.words('russian'))

# Initialize transformer model
model_name = "DeepPavlov/rubert-base-cased-sentence"
transformer_model = None  # Lazy loading

def get_transformer_model():
    """Get or initialize the transformer model"""
    global transformer_model
    if transformer_model is None:
        transformer_model = SentenceTransformer(model_name)
    return transformer_model

def preprocess_text(text):
    """
    Preprocess text: remove stopwords, punctuation, lemmatize
    
    Args:
        text (str): Raw text from document
        
    Returns:
        tuple: (processed_text, shingles)
    """
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text, language='russian')
    
    # Remove stopwords and lemmatize
    lemmatized_tokens = []
    for token in tokens:
        if token not in russian_stopwords and len(token) > 2:
            lemma = morph.parse(token)[0].normal_form
            lemmatized_tokens.append(lemma)
    
    # Create shingles (n-grams)
    text_length = len(lemmatized_tokens)
    
    # Dynamically determine shingle size based on text length
    shingle_size = 5  # Default 
    if text_length > 30000:
        shingle_size = 6
    
    shingles = []
    for i in range(len(lemmatized_tokens) - shingle_size + 1):
        shingle = ' '.join(lemmatized_tokens[i:i+shingle_size])
        shingles.append(shingle)
    
    # Rebuilt processed text
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text, shingles

def create_minhash(shingles, num_perm=128):
    """
    Create MinHash signature for a set of shingles
    
    Args:
        shingles (list): List of shingles
        num_perm (int): Number of permutations
        
    Returns:
        MinHash: MinHash signature
    """
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode('utf8'))
    return m

def find_candidates_with_minhash(shingles, threshold=0.4):
    """
    Find candidate documents using MinHash LSH
    
    Args:
        shingles (list): List of shingles from the query document
        threshold (float): Similarity threshold
        
    Returns:
        list: List of document IDs that are candidates for plagiarism
    """
    # Create MinHash for query document
    query_minhash = create_minhash(shingles)
    
    # Get all documents from DB
    all_sigs = get_all_signatures()
    if not all_sigs:
        return []
    
    # Create LSH index with stored documents
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    # Add all documents to LSH index
    for sig in all_sigs:
        doc_minhash = LeanMinHash(seed=42)
        doc_minhash.hashvalues = np.array(sig.hashes, dtype=np.uint32)
        lsh.insert(str(sig.id), doc_minhash)
    
    # Query LSH index
    candidate_ids = lsh.query(query_minhash)
    
    # Convert string IDs back to integers
    candidate_ids = [int(doc_id) for doc_id in candidate_ids]
    
    return candidate_ids

def split_text_into_chunks(text, chunk_size=250, overlap=0.3):
    """
    Split text into overlapping chunks
    
    Args:
        text (str): The text to split
        chunk_size (int): Number of words per chunk
        overlap (float): Overlap ratio between chunks
        
    Returns:
        list: List of text chunks
    """
    words = text.split()
    overlap_size = int(chunk_size * overlap)
    step = chunk_size - overlap_size
    
    chunks = []
    for i in range(0, len(words), step):
        if i + chunk_size > len(words):
            # Handle the last chunk
            chunk = ' '.join(words[i:])
            if len(chunk.split()) >= chunk_size // 2:  # Only include if it's at least half the size
                chunks.append(chunk)
            break
        
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks

def compute_chunk_embeddings(chunks):
    """
    Compute embeddings for a list of text chunks using transformers
    
    Args:
        chunks (list): List of text chunks
        
    Returns:
        np.ndarray: Array of embeddings
    """
    model = get_transformer_model()
    
    # Use Dask for parallelization
    chunks_bag = db.from_sequence(chunks)
    embeddings = chunks_bag.map(lambda x: model.encode(x)).compute()
    
    return np.array(embeddings)

def compute_average_embedding(embeddings):
    """
    Compute average embedding from a list of embeddings
    
    Args:
        embeddings (np.ndarray): Array of embeddings
        
    Returns:
        np.ndarray: Average embedding
    """
    return np.mean(embeddings, axis=0)

def compute_similarity(emb1, emb2):
    """
    Compute cosine similarity between two embeddings
    
    Args:
        emb1 (np.ndarray): First embedding
        emb2 (np.ndarray): Second embedding
        
    Returns:
        float: Cosine similarity score
    """
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def check_plagiarism_with_transformers(text, candidate_ids, chunk_size=250, overlap=0.3, similarity_threshold=0.5):
    """
    Check plagiarism using transformer embeddings
    
    Args:
        text (str): Preprocessed query text
        candidate_ids (list): List of candidate document IDs
        chunk_size (int): Number of words per chunk
        overlap (float): Overlap ratio between chunks
        similarity_threshold (float): Similarity threshold
        
    Returns:
        dict: Result with similarity scores and details
    """
    if not candidate_ids:
        return {
            'max_similarity': 0,
            'similar_docs': [],
            'details': {}
        }
    
    # Get candidate documents from DB
    candidates = get_signatures_by_ids(candidate_ids)
    
    # Split query text into chunks
    query_chunks = split_text_into_chunks(text, chunk_size, overlap)
    
    # Compute embeddings for query chunks
    query_embeddings = compute_chunk_embeddings(query_chunks)
    
    # Compute average embedding for query
    query_avg_embedding = compute_average_embedding(query_embeddings)
    
    result = {
        'max_similarity': 0,
        'similar_docs': [],
        'details': {}
    }
    
    # Check similarity with each candidate
    for sig in candidates:
        # Получаем url через sig.document.url
        doc_url = sig.document.url if sig.document else None
        doc_text = download_and_preprocess_document(doc_url)
        
        # Split into chunks
        doc_chunks = split_text_into_chunks(doc_text, chunk_size, overlap)
        
        # Compute embeddings
        doc_embeddings = compute_chunk_embeddings(doc_chunks)
        
        # Compute average embedding
        doc_avg_embedding = compute_average_embedding(doc_embeddings)
        
        # Compute similarity
        similarity = compute_similarity(query_avg_embedding, doc_avg_embedding)
        
        # If similarity exceeds threshold, add to results
        if similarity > similarity_threshold:
            if similarity > result['max_similarity']:
                result['max_similarity'] = similarity
            
            result['similar_docs'].append(sig.id)
            
            # Store chunk-level details
            chunk_similarities = []
            for i, q_emb in enumerate(query_embeddings):
                for j, d_emb in enumerate(doc_embeddings):
                    chunk_sim = compute_similarity(q_emb, d_emb)
                    if chunk_sim > similarity_threshold:
                        chunk_similarities.append({
                            'query_chunk_idx': i,
                            'doc_chunk_idx': j,
                            'similarity': chunk_sim
                        })
            
            # Sort by similarity and take top 10
            chunk_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            result['details'][sig.id] = chunk_similarities[:10]
    
    return result

def download_and_preprocess_document(url):
    """
    Download document from URL and preprocess it
    
    Args:
        url (str): URL to the document
        
    Returns:
        str: Preprocessed text
    """
    from app.services.document_processor import download_doc_from_url
    
    # Download and extract text
    raw_text = download_doc_from_url(url)
    
    # Preprocess
    processed_text, _ = preprocess_text(raw_text)
    
    return processed_text 