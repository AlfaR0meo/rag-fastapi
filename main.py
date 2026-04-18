from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(
    title="Бэкенд микросервис для семантического поиска",
    description="API для функций семантического поиска RAG-системы"
)

# Модель (легкая для бесплатного тарифа Render), хуже понимает русский язык
model = SentenceTransformer('all-MiniLM-L6-v2')

# База знаний (пока в памяти)
knowledge_base = [
    "n8n — это инструмент для автоматизации рабочих процессов",
    "RAG — это подход, который сочетает retrieval и генерацию",
    "Sentence Transformers используются для создания эмбеддингов текста",
    "FastAPI — это фреймворк для создания API на Python",
    "Векторное сходство обычно считается через cosine similarity",
    "Рост Антона составляет 190 сантиметров",
    "Пароль от wi-fi: 1234####"
]

# 1. Создаем эмбеддинги базы
kb_embeddings = model.encode(knowledge_base)

# 2. Переводим в float32 (обязательно для FAISS)
kb_embeddings = np.array(kb_embeddings).astype("float32")

# 3. Нормализация (чтобы работал cosine)
faiss.normalize_L2(kb_embeddings)

# 4. Создаём FAISS индекс
dimension = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# 5. Добавляем векторы в индекс
index.add(kb_embeddings)

# endpoint: EMBEDDING
@app.post("/embed")
def embed(data: dict):
    text = data["text"]

    embedding = model.encode(text)
    embedding = embedding.tolist()

    return {"embedding": embedding}

# endpoint: SEARCH (FAISS)
@app.post("/search")
def search(data: dict):
    query = data["text"]

    # всегда 3 результата
    top_k = 3

    # query embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # нормализация
    faiss.normalize_L2(query_embedding)

    # поиск
    distances, indices = index.search(query_embedding, top_k)

    # сбор результатов
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        distance = float(distances[0][i])

        # перевод в "похожесть"
        similarity = 1 - distance

        results.append({
            "text": knowledge_base[idx],
            "score": similarity
        })

    return {
        "query": query,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
