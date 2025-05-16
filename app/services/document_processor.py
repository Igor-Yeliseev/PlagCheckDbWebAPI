import docx
import os
import requests
from io import BytesIO
import tempfile

def extract_text_from_doc(file_path):
    """
    Extract text from a .docx file
    
    Args:
        file_path (str): Path to the .docx file
        
    Returns:
        str: Extracted text from the document
    """
    doc = docx.Document(file_path)
    full_text = []
    
    for para in doc.paragraphs:
        full_text.append(para.text)
        
    return '\n'.join(full_text)

def download_doc_from_url(url):
    """
    Download document from URL and extract text
    
    Args:
        url (str): URL to the document
        
    Returns:
        str: Extracted text from the document
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download document: {response.status_code}")
    
    # Save temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
    temp_file.write(response.content)
    temp_file.close()
    
    try:
        # Extract text
        text = extract_text_from_doc(temp_file.name)
        return text
    finally:
        # Clean up
        os.unlink(temp_file.name) 