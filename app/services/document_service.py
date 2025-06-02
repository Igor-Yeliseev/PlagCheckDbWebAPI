import docx
import os
import requests
from io import BytesIO
import tempfile

def extract_text_from_doc(file_path):
    """
    Извлечь текст из файла .docx
    
    Аргументы:
        file_path (str): Путь к файлу .docx
        
    Возвращает:
        str: Извлечённый текст из документа
    """
    doc = docx.Document(file_path)
    full_text = []
    
    for para in doc.paragraphs:
        full_text.append(para.text)
        
    return '\n'.join(full_text)

def download_doc_from_url(url):
    """
    Скачать документ по URL и извлечь текст
    
    Аргументы:
        url (str): URL документа
        
    Возвращает:
        str: Извлечённый текст из документа
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download document: {response.status_code}")
    
    # Временное сохранение файла
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
    temp_file.write(response.content)
    temp_file.close()
    
    try:
        # Извлечение текста
        text = extract_text_from_doc(temp_file.name)
        return text
    finally:
        # Удаление временного файла
        os.unlink(temp_file.name) 