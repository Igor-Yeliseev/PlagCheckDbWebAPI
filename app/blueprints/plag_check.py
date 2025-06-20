from flask import Blueprint, request, jsonify
import os
import tempfile
from app.services.document_service import extract_text_from_doc
from app.services.plag_check_service import (
    preprocess_text, 
    find_candidates_with_minhash, 
    check_plagiarism_with_transformers,
    check_plagiarism_with_all_docs
)
from app.models.doc_signature import DocSignature
from app.utils.db_utils import get_signature_by_id, save_signature_to_db

plag_check_bp = Blueprint('plag_check', __name__, url_prefix='/py-api')

@plag_check_bp.route('/check-with-db', methods=['POST'])
def check_document():
    """
    Эндпоинт для проверки документа на плагиат по базе данных
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.docx'):
        return jsonify({'error': 'Only .docx files are supported'}), 400
    
    # Сохраняем загруженный файл во временную папку
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    
    try:
        # Извлекаем текст из документа
        text = extract_text_from_doc(temp_path)
        
        # Предобрабатываем текст (очистка, лемматизация, создание шинглов)
        processed_text, shingles = preprocess_text(text)
        
        # Первый этап: MinHash + LSH для поиска кандидатов
        candidates = find_candidates_with_minhash(shingles, threshold=0.4)
        
        # Если кандидаты не найдены, документ оригинальный
        if not candidates:
            return jsonify({
                'message': 'Документ является оригинальным. Плагиат не обнаружен.',
                'similarity': 0
            })
        
        # Второй этап: проверка с помощью трансформеров
        result = check_plagiarism_with_transformers(processed_text, candidates)
        
        # Возвращаем результат на основе порога сходства
        if result['max_similarity'] > 0.7:
            return jsonify({
                'message': 'Обнаружен плагиат',
                'similarity': result['max_similarity']
            })
        else:
            return jsonify({
                'message': 'Документ является оригинальным. Найденные совпадения незначительны.',
                'similarity': result['max_similarity']
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Очистка временных файлов
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
@plag_check_bp.route('/check-with-db-neural', methods=['POST'])
def check_document_emb():
    """
    Эндпоинт для проверки документа на плагиат по базе данных с использованием
    только нейросетевого подхода (без MinHash LSH).
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.docx'):
        return jsonify({'error': 'Only .docx files are supported'}), 400

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        text = extract_text_from_doc(temp_path)
        processed_text, _ = preprocess_text(text)  # Шинглы не нужны

        # Проверка напрямую со всеми документами через трансформеры
        result = check_plagiarism_with_all_docs(processed_text, similarity_threshold=0.2)

        if result['max_similarity'] > 0.2:
            return jsonify({
                'message': 'Обнаружен плагиат',
                'similarity': result['max_similarity']
            })
        else:
            return jsonify({
                'message': 'Документ является оригинальным.',
                'similarity': result.get('max_similarity', 0)
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
@plag_check_bp.route('/download-candidate/<uuid:document_id>', methods=['GET'])
def download_candidate(document_id):
    """
    Эндпоинт для скачивания документа-кандидата на плагиат
    """
    try:
        signature = get_signature_by_id(document_id)
        if not signature:
            return jsonify({'error': 'Signature not found'}), 404
        # Получаем ссылку на файл через signature.document.url
        url = signature.document.url if signature.document else None
        return jsonify({
            'document_id': signature.document_id,
            'url': url,
            'message': 'Используйте URL для скачивания документа'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500