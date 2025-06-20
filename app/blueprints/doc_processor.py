from flask import Blueprint, request, jsonify
import uuid
from app.services.document_service import download_doc_from_url, extract_text_from_doc
from app.services.plag_check_service import (
    preprocess_text, 
    create_minhash,
    compute_chunk_embeddings,
    compute_average_embedding,
    split_into_chunks
)
from app.utils.db_utils import save_signature_to_db
from app.utils.db_utils import save_embedding_to_db
# import numpy as np
import tempfile
import os

doc_processor_bp = Blueprint('doc_processor', __name__, url_prefix='/py-api')

@doc_processor_bp.route('/new-doc', methods=['POST'])
def process_new_doc():
    """
    Эндпоинт для обработки нового документа и сохранения его сигнатур и эмбеддингов через загрузку файла
    """
    # Проверяем наличие файла в запросе
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Проверяем наличие document_id в запросе
    if 'document_id' not in request.form:
        return jsonify({'error': 'No document_id in request'}), 400
    
    try:
        document_id = request.form['document_id']
        if isinstance(document_id, str):
            document_id = uuid.UUID(document_id)
        
        # Сохраняем файл во временную папку
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        raw_text = extract_text_from_doc(temp_path) # Извлекаем текст из документа
        
        processed_text, shingles = preprocess_text(raw_text) # Предобрабатываем текст и создаем шинглы
        
        # Создаем MinHash
        minhash = create_minhash(shingles)
        hashes = minhash.hashvalues.tolist()
        
        save_signature_to_db(document_id, hashes) # Сохраняем сигнатуру в базу данных
        
        chunks = split_into_chunks(processed_text) # Разбиваем текст на чанки и создаем эмбеддинги
        
        embeddings = compute_chunk_embeddings(chunks) # Вычисляем эмбеддинги для чанков
        
        avg_embedding = compute_average_embedding(embeddings) # Вычисляем средний эмбеддинг
        
        save_embedding_to_db(document_id, avg_embedding.tolist()) # Сохраняем эмбеддинг в базу данных
        
        # Удаляем временный файл
        os.remove(temp_path) 
        os.rmdir(temp_dir)
        return jsonify({
            'success': True,
            'message': 'Document processed successfully',
            'document_id': str(document_id)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

''' 
@doc_processor_bp.route('/new-doc-url', methods=['POST'])
def process_new_doc_url():
    """
    Эндпоинт для обработки нового документа и сохранения его сигнатур и эмбеддингов
    """
    data = request.get_json() # Проверяем наличие document_id в запросе
    
    if not data or 'document_id' not in data:
        return jsonify({'error': 'No document_id in request'}), 400
    if 'url' not in data:
        return jsonify({'error': 'No document URL in request'}), 400
    
    try:
        document_id = data['document_id'] # Преобразуем document_id в UUID если он в строковом формате
        if isinstance(document_id, str):
            document_id = uuid.UUID(document_id)
        document_url = data['url']
        
        raw_text = download_doc_from_url(document_url) # Загружаем и извлекаем текст документа
        
        processed_text, shingles = preprocess_text(raw_text) # Предобрабатываем текст и создаем шинглы
        
        minhash = create_minhash(shingles) # Создаем MinHash
        hashes = minhash.hashvalues.tolist()
        
        save_signature_to_db(document_id, hashes) # Сохраняем сигнатуру в базу данных
        
        chunks = split_into_chunks(processed_text) # Разбиваем текст на чанки и создаем эмбеддинги
        
        embeddings = compute_chunk_embeddings(chunks) # Вычисляем эмбеддинги для чанков
        
        avg_embedding = compute_average_embedding(embeddings) # Вычисляем средний эмбеддинг
        
        save_embedding_to_db(document_id, avg_embedding.tolist()) # Сохраняем эмбеддинг в базу данных
        
        return jsonify({
            'success': True,
            'message': 'Document processed successfully',
            'document_id': str(document_id)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
'''