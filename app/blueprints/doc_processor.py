from flask import Blueprint, request, jsonify
import uuid
from app.services.document_service import download_doc_from_url
from app.services.plag_check_service import (
    preprocess_text, 
    create_minhash,
    compute_chunk_embeddings,
    compute_average_embedding,
    split_text_into_chunks
)
from app.models.document import DocSignature
from app.models.vector_store import DocEmbedding
from app.utils.db_utils import save_signature_to_db
from app.utils.db_utils import save_embedding_to_db
import numpy as np

doc_processor_bp = Blueprint('doc_processor', __name__, url_prefix='/py-api')

@doc_processor_bp.route('/new-doc', methods=['POST'])
def process_document():
    """
    Эндпоинт для обработки нового документа и сохранения его сигнатур и эмбеддингов
    """
    # Проверяем наличие document_id в запросе
    data = request.get_json()
    if not data or 'document_id' not in data:
        return jsonify({'error': 'No document_id in request'}), 400
    
    if 'url' not in data:
        return jsonify({'error': 'No document URL in request'}), 400
    
    try:
        # Преобразуем document_id в UUID если он в строковом формате
        document_id = data['document_id']
        if isinstance(document_id, str):
            document_id = uuid.UUID(document_id)
        
        document_url = data['url']
        
        # Загружаем и извлекаем текст документа
        raw_text = download_doc_from_url(document_url)
        
        # Предобрабатываем текст и создаем шинглы
        processed_text, shingles = preprocess_text(raw_text)
        
        # Создаем MinHash
        minhash = create_minhash(shingles)
        hashes = minhash.hashvalues.tolist()
        
        # Сохраняем сигнатуру в базу данных
        save_signature_to_db(document_id, hashes)
        
        # Разбиваем текст на чанки и создаем эмбеддинги
        chunks = split_text_into_chunks(processed_text)
        
        # Вычисляем эмбеддинги для чанков
        embeddings = compute_chunk_embeddings(chunks)
        
        # Вычисляем средний эмбеддинг
        avg_embedding = compute_average_embedding(embeddings)
        
        # Сохраняем эмбеддинг в базу данных
        save_embedding_to_db(document_id, avg_embedding.tolist())
        
        return jsonify({
            'success': True,
            'message': 'Document processed successfully',
            'document_id': str(document_id)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500