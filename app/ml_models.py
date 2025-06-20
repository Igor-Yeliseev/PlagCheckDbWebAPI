from sentence_transformers import SentenceTransformer

def load_transformer_model():
    """
    Загружает и возвращает предварительно обученную модель Sentence Transformer.
    Модель загружается один раз и кэшируется в памяти.
    """
    model_name = "DeepPavlov/rubert-base-cased-sentence"
    print(f"Загружаю модель {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("Модель загружена успешно!")
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None

# Загружаем модель при инициализации модуля
transformer_model = load_transformer_model()

def get_transformer_model():
    """
    Возвращает экземпляр загруженной модели.
    """
    return transformer_model 