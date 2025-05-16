from flask import Blueprint, request, jsonify
import os
import tempfile
from app.services.document_processor import extract_text_from_doc
from app.services.plagiarism_checker import (
    preprocess_text, 
    find_candidates_with_minhash, 
    check_plagiarism_with_transformers
)
from app.models.document import DocSignature
from app.utils.db_utils import get_signature_by_id, save_signature_to_db

plag_check_bp = Blueprint('plag_check', __name__, url_prefix='/py-api')

@plag_check_bp.route('/check-with-db', methods=['POST'])
def check_document():
    """
    Endpoint for checking document plagiarism against the database
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.docx'):
        return jsonify({'error': 'Only .docx files are supported'}), 400
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)
    
    try:
        # Extract text from document
        text = extract_text_from_doc(temp_path)
        
        # Preprocess text (clean, lemmatize, create shingles)
        processed_text, shingles = preprocess_text(text)
        
        # First stage: MinHash + LSH to find candidates
        candidates = find_candidates_with_minhash(shingles, threshold=0.4)
        
        # If no candidates found, document is original
        if not candidates:
            return jsonify({
                'original': True,
                'message': 'Документ является оригинальным. Плагиат не обнаружен.',
                'similarity': 0
            })
        
        # Second stage: Check with transformers
        result = check_plagiarism_with_transformers(processed_text, candidates)
        
        # Return result based on similarity threshold
        if result['max_similarity'] > 0.6:
            return jsonify({
                'original': False,
                'message': 'Обнаружен плагиат',
                'similarity': result['max_similarity'],
                'similar_document_ids': result['similar_docs'],
                'details': result['details']
            })
        else:
            return jsonify({
                'original': True,
                'message': 'Документ является оригинальным. Найденные совпадения незначительны.',
                'similarity': result['max_similarity']
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
@plag_check_bp.route('/download-candidate/<int:sig_id>', methods=['GET'])
def download_candidate(sig_id):
    """
    Endpoint for downloading a plagiarism candidate document
    """
    try:
        signature = get_signature_by_id(sig_id)
        if not signature:
            return jsonify({'error': 'Signature not found'}), 404
        # Получаем ссылку на файл через signature.document.url
        url = signature.document.url if signature.document else None
        return jsonify({
            'id': signature.id,
            'document_id': signature.document_id,
            'url': url,
            'message': 'Используйте URL для скачивания документа'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500 