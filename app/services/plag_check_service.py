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
import dask.bag as dask_bag
from app.utils.db_utils import get_all_signatures, get_signatures_by_ids
from app.services.document_service import download_doc_from_url

# Загрузка необходимых данных NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Инициализация инструментов
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = set(stopwords.words('russian'))

# Инициализация трансформерной модели
model_name = "DeepPavlov/rubert-base-cased-sentence"
transformer_model = None  # Ленивое создание

def get_transformer_model():
    """
    Получить или инициализировать трансформерную модель
    """
    global transformer_model
    if transformer_model is None:
        transformer_model = SentenceTransformer(model_name)
    return transformer_model

def preprocess_text(text):
    """
    Предобработка текста: удаление стоп-слов, знаков препинания, лемматизация
    
    Аргументы:
        text (str): Исходный текст документа
        
    Возвращает:
        tuple: (обработанный_текст, шинглы)
    """
    # Приведение к нижнему регистру и удаление знаков препинания
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Токенизация
    tokens = word_tokenize(text, language='russian')
    
    # Удаление стоп-слов и лемматизация
    lemmatized_tokens = []
    for token in tokens:
        if token not in russian_stopwords and len(token) > 2:
            lemma = morph.parse(token)[0].normal_form
            lemmatized_tokens.append(lemma)
    
    # Создание шинглов (n-грамм)
    text_length = len(lemmatized_tokens)
    
    # Динамический выбор размера шингла в зависимости от длины текста
    shingle_size = 5  # По умолчанию
    if text_length > 30000:
        shingle_size = 6
    
    shingles = []
    for i in range(len(lemmatized_tokens) - shingle_size + 1):
        shingle = ' '.join(lemmatized_tokens[i:i+shingle_size])
        shingles.append(shingle)
    
    # Восстановление обработанного текста
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text, shingles

def create_minhash(shingles, num_perm=128):
    """
    Создать MinHash-подпись для набора шинглов
    
    Аргументы:
        shingles (list): Список шинглов
        num_perm (int): Количество перестановок
        
    Возвращает:
        MinHash: MinHash-подпись
    """
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode('utf8'))
    return m

def find_candidates_with_minhash(shingles, threshold=0.4):
    """
    Найти кандидатов на плагиат с помощью MinHash LSH
    
    Аргументы:
        shingles (list): Список шинглов из проверяемого документа
        threshold (float): Порог сходства
        
    Возвращает:
        list: Список ID документов-кандидатов
    """
    # Создание MinHash для проверяемого документа
    query_minhash = create_minhash(shingles)
    
    # Получение всех документов из БД
    all_sigs = get_all_signatures()
    if not all_sigs:
        return []
    
    # Создание LSH-индекса для сохранённых документов
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    # Добавление всех документов в LSH-индекс
    for sig in all_sigs:
        doc_minhash = LeanMinHash(seed=42)
        doc_minhash.hashvalues = np.array(sig.hashes, dtype=np.uint32)
        lsh.insert(str(sig.id), doc_minhash)
    
    # Поиск кандидатов в индексе LSH
    candidate_ids = lsh.query(query_minhash)
    
    # Преобразование строковых ID обратно в числа
    candidate_ids = [int(doc_id) for doc_id in candidate_ids]
    
    return candidate_ids

def split_text_into_chunks(text, chunk_size=250, overlap=0.3):
    """
    Разбить текст на перекрывающиеся чанки
    
    Аргументы:
        text (str): Текст для разбиения
        chunk_size (int): Количество слов в чанке
        overlap (float): Доля перекрытия между чанками
        
    Возвращает:
        list: Список текстовых чанков
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
    Вычислить эмбеддинги для списка текстовых чанков с помощью трансформеров
    
    Аргументы:
        chunks (list): Список текстовых чанков
        
    Возвращает:
        np.ndarray: Массив эмбеддингов
    """
    model = get_transformer_model()
    
    # Используем Dask для параллелизации
    chunks_bag = dask_bag.from_sequence(chunks)
    embeddings = chunks_bag.map(lambda x: model.encode(x)).compute()
    
    return np.array(embeddings)

def compute_average_embedding(embeddings):
    """
    Вычислить средний эмбеддинг из списка эмбеддингов
    
    Аргументы:
        embeddings (np.ndarray): Массив эмбеддингов
        
    Возвращает:
        np.ndarray: Средний эмбеддинг
    """
    return np.mean(embeddings, axis=0)

def compute_similarity(emb1, emb2):
    """
    Вычислить косинусное сходство между двумя эмбеддингами
    
    Аргументы:
        emb1 (np.ndarray): Первый эмбеддинг
        emb2 (np.ndarray): Второй эмбеддинг
        
    Возвращает:
        float: Значение косинусного сходства
    """
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def check_plagiarism_with_transformers(text, candidate_ids, chunk_size=250, overlap=0.3, similarity_threshold=0.5):
    """
    Проверка плагиата с использованием эмбеддингов трансформеров
    
    Аргументы:
        text (str): Предобработанный текст запроса
        candidate_ids (list): Список ID документов-кандидатов
        chunk_size (int): Количество слов в чанке
        overlap (float): Доля перекрытия между чанками
        similarity_threshold (float): Порог сходства
        
    Возвращает:
        dict: Результат с оценками сходства и деталями
    """
    if not candidate_ids:
        return {
            'max_similarity': 0,
            'similar_docs': [],
            'details': {}
        }
    
    # Получение документов-кандидатов из БД
    candidates = get_signatures_by_ids(candidate_ids)
    
    # Разбиваем текст запроса на чанки
    query_chunks = split_text_into_chunks(text, chunk_size, overlap)
    
    # Вычисляем эмбеддинги для чанков запроса
    query_embeddings = compute_chunk_embeddings(query_chunks)
    
    # Вычисляем средний эмбеддинг для запроса
    query_avg_embedding = compute_average_embedding(query_embeddings)
    
    result = {
        'max_similarity': 0,
        'similar_docs': [],
        'details': {}
    }
    
    # Проверяем сходство с каждым кандидатом
    for sig in candidates:
        # Получаем url через sig.document.url
        doc_url = sig.document.url if sig.document else None
        doc_text = download_and_preprocess_document(doc_url)
        
        # Разбиваем на чанки
        doc_chunks = split_text_into_chunks(doc_text, chunk_size, overlap)
        
        # Вычисляем эмбеддинги
        doc_embeddings = compute_chunk_embeddings(doc_chunks)
        
        # Вычисляем средний эмбеддинг
        doc_avg_embedding = compute_average_embedding(doc_embeddings)
        
        # Вычисляем сходство
        similarity = compute_similarity(query_avg_embedding, doc_avg_embedding)
        
        # Если сходство превышает порог, добавляем в результат
        if similarity > similarity_threshold:
            if similarity > result['max_similarity']:
                result['max_similarity'] = similarity
            
            result['similar_docs'].append(sig.id)
            
            # Сохраняем детали по чанкам
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
            
            # Сортируем по сходству и берём топ-10
            chunk_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            result['details'][sig.id] = chunk_similarities[:10]
    
    return result

def download_and_preprocess_document(url):
    """
    Скачать документ по URL и предобработать его
    
    Аргументы:
        url (str): URL документа
        
    Возвращает:
        str: Предобработанный текст
    """
    # Скачиваем и извлекаем текст
    raw_text = download_doc_from_url(url)
    
    # Предобработка
    processed_text, _ = preprocess_text(raw_text)
    
    return processed_text 