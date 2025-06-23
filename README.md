# PlagCheck WebAPI

API сервис на Flask для проверки документов на плагиат с использованием MinHash и трансформеров.

## Описание

Сервис принимает документы Word (.docx) через POST запросы и проверяет их на плагиат в сравнении с документами в базе данных. Процесс происходит в два этапа:

1. Быстрая проверка с использованием MinHash + LSH для выявления кандидатов (порог сходства: 40%)
2. Точная проверка с использованием трансформеров для определения семантического сходства (порог: 50%)

## Установка

1. Создайте и активируйте виртуальное окружение:

```
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Установите зависимости:

```
pip install -r requirements.txt
```

3. Создайте базу данных PostgreSQL и настройте подключение в файле `.env`:

```
DATABASE_URI=postgresql://postgres:2004@localhost:5432/plag_search_db
```

4. Примените SQL-скрипт для создания таблиц:

```
psql -U postgres -d plag_search_db -f create_tables.sql
```

## Запуск

```
python app.py
```

Сервер запустится на `http://localhost:5000`.

## API Эндпоинты

### Проверка документа на плагиат

```
POST /py-api/check-with-db
```

Параметры:
- `file`: Документ Word (.docx) для проверки (form-data)

Ответ:
```json
{
  ,,, Дописать
}
```
