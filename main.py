import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tiktoken
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
import time

# Подключаем все переменные из окружения
load_dotenv()
# Подключаем ключ для LLM-модели
LLM_API_KEY = os.getenv("LLM_API_KEY")
# Подключаем ключ для EMBEDDER-модели
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

# RAG Fusion: константа для RRF
RRF_K = 60

# Константы для оптимизации
EMBEDDING_BATCH_SIZE = 100  # Размер батча для эмбеддингов
MAX_WORKERS = 10  # Количество потоков для параллельной обработки
CHECKPOINT_INTERVAL = 10  # Сохранять результаты каждые N вопросов
PROGRESS_FILE = 'progress.json'  # Файл для отслеживания прогресса
MAX_RETRIES = 3  # Максимальное количество попыток при ошибке


# RAG Fusion: функция для разбиения текста на чанки
def split_text_into_chunks(text, max_tokens=8000):
    """Разбивает текст на части если он слишком длинный"""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))
    
    return chunks


# RAG Fusion: ОПТИМИЗИРОВАННАЯ функция для создания эмбеддингов с батчингом
def create_embeddings_batch(texts, api_key, batch_size=EMBEDDING_BATCH_SIZE):
    """Создает эмбеддинги с батчингом для ускорения"""
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=api_key,
    )
    
    all_embeddings = []
    
    # Обрабатываем батчами
    for i in tqdm(range(0, len(texts), batch_size), desc="Создание эмбеддингов (батчи)"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = []
        
        for text in batch_texts:
            # Разбиваем текст на чанки если он слишком длинный
            chunks = split_text_into_chunks(text, max_tokens=8000)
            
            if len(chunks) == 1:
                # Если один чанк, просто получаем эмбеддинг
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunks[0]
                )
                batch_embeddings.append(response.data[0].embedding)
            else:
                # Если несколько чанков, получаем эмбеддинги и усредняем
                chunk_embeddings = []
                for chunk in chunks:
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk
                    )
                    chunk_embeddings.append(response.data[0].embedding)
                avg_embedding = np.mean(chunk_embeddings, axis=0)
                batch_embeddings.append(avg_embedding)
        
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)


# ОПТИМИЗАЦИЯ: Кеширование эмбеддингов обучающих данных
def load_or_create_train_embeddings(train_texts, api_key, cache_file='train_embeddings.pkl'):
    """Загружает эмбеддинги из кеша или создает новые"""
    if os.path.exists(cache_file):
        print(f"Загрузка эмбеддингов из кеша: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        print("Создание новых эмбеддингов...")
        embeddings = create_embeddings_batch(train_texts, api_key)
        # Сохраняем в кеш
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Эмбеддинги сохранены в кеш: {cache_file}")
        return embeddings


# RAG Fusion: функция для создания TF-IDF индекса (замена BM25)
def create_tfidf_index(texts):
    """Создает TF-IDF векторизатор и матрицу для текстового поиска"""
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        use_idf=True
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


# RAG Fusion: Reciprocal Rank Fusion (векторизованная версия)
def reciprocal_rank_fusion(semantic_scores, tfidf_scores, k=RRF_K):
    n_docs = len(semantic_scores)
    
    # Получаем ранги
    semantic_ranks = np.argsort(-semantic_scores)
    tfidf_ranks = np.argsort(-tfidf_scores)
    
    # Создаем mapping: doc_id -> rank
    semantic_rank_map = np.zeros(n_docs, dtype=int)
    tfidf_rank_map = np.zeros(n_docs, dtype=int)
    
    semantic_rank_map[semantic_ranks] = np.arange(n_docs)
    tfidf_rank_map[tfidf_ranks] = np.arange(n_docs)
    
    # RRF score (векторизованная операция)
    rrf_scores = (
        1.0 / (k + semantic_rank_map) +
        1.0 / (k + tfidf_rank_map)
    )
    
    return rrf_scores


# ОПТИМИЗАЦИЯ: Батчинг эмбеддингов вопросов
def get_questions_embeddings_batch(questions, api_key, batch_size=EMBEDDING_BATCH_SIZE):
    """Получает эмбеддинги для всех вопросов батчами"""
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app/",
        api_key=api_key,
    )
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(questions), batch_size), desc="Создание эмбеддингов вопросов"):
        batch = questions[i:i + batch_size]
        
        for q in batch:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=q
            )
            all_embeddings.append(response.data[0].embedding)
    
    return np.array(all_embeddings)


# ОПТИМИЗАЦИЯ: поиск релевантных документов с предвычисленным эмбеддингом
def search_relevant_docs_optimized(q_emb, train_texts, train_embeddings, tfidf_vectorizer, tfidf_matrix, question, top_k=4):
    """Оптимизированная версия поиска с предвычисленным эмбеддингом вопроса (TF-IDF вместо BM25)"""
    
    # Semantic search (векторизованная операция)
    semantic_scores = cosine_similarity(q_emb.reshape(1, -1), train_embeddings)[0]
    
    # TF-IDF search (замена BM25)
    question_tfidf = tfidf_vectorizer.transform([question])
    tfidf_scores = cosine_similarity(question_tfidf, tfidf_matrix).flatten()
    
    # RRF Fusion
    rrf_scores = reciprocal_rank_fusion(semantic_scores, tfidf_scores, k=RRF_K)
    
    # Top-K документов
    top_indices = np.argsort(rrf_scores)[-top_k:][::-1]
    
    # Формируем контекст (ограничиваем длину каждого документа)
    retrieved_docs = []
    for i in top_indices:
        doc = train_texts[i]
        # Ограничиваем длину каждого документа
        if len(doc) > 3000:
            doc = doc[:3000] + "..."
        retrieved_docs.append(doc)
    
    context = "\n\n---\n\n".join(retrieved_docs)
    
    return context


# Функция для генерации ответа по заданному вопросу, вы можете изменять ее в процессе работы, однако
# просим оставить структуру обращения, т.к. при запуске на сервере, потребуется корректно указанный путь 
# для формирования ответов. Также не вставляйте ключ вручную, поскольку при запуске ключ подтянется автоматически
def answer_generation(question, context=""):
    # Подключаемся к модели
    client = OpenAI(
        # Базовый url - сохранять без изменения
        base_url="https://ai-for-finance-hack.up.railway.app/",
        # Указываем наш ключ, полученный ранее
        api_key=LLM_API_KEY,
    )
    
    # RAG Fusion: формируем промпт с контекстом
    if context:
        # Ограничиваем контекст для промпта
        context_preview = context[:5000] if len(context) > 5000 else context
        
        prompt = f"""Ты — эксперт по финансам и банковским продуктам. Твоя задача — дать исчерпывающий, структурированный и понятный ответ на вопрос клиента, используя предоставленную информацию из базы знаний.

ПРИМЕРЫ ИДЕАЛЬНЫХ ОТВЕТОВ:

ПРИМЕР 1:
Вопрос: Как часто выплачивается купон по облигации?

Ответ: Процентная выплата по облигации, которую получает держатель облигации, обычно устанавливается в процентах годовых от номинала облигации. Купон может выплачиваться один или несколько раз в год. Купон обычно выплачивается на брокерский счет клиента.

ПРИМЕР 2:
Вопрос: Как изменение ключевой ставки влияет на цену облигации с фиксированной ставкой?

Ответ: Облигации, купон по которым установлен фиксированным значением. Например, купон может быть 5%, 10%. Обычно цена таких облигаций сильно зависит от ключевой ставки. Если ключевая ставка выше купона — цена облигации ниже. Если ключевая ставка ниже купона — цена облигации выше. Если ключевая ставка растет, цена облигации падает. Если ключевая ставка падает, цена облигации растет. Обычно, чем ближе срок погашения облигации, тем ближе цена облигации к номиналу.

ПРИМЕР 3:
Вопрос: Как просрочка по «беспроцентному» займу скажется на переплате/ПСК?

Ответ: ### Влияние просрочки по "беспроцентному" займу на переплату и ПСК

"Беспроцентный" заём (например, рассрочка в магазине или онлайн-займ без процентов) на бумаге не предполагает начисления процентов, но просрочка платежа (невыплата в срок) обычно сильно сказывается на общей стоимости. Вот как это работает на общих принципах российского законодательства (ФЗ-353 "О потребительском кредите" и условия договоров). Обратите внимание: точные последствия зависят от конкретного договора — обязательно проверьте его или проконсультируйтесь с кредитором/юристом.

#### 1. **Влияние на переплату**
   - **Что это значит?** Переплата — это разница между суммой, которую вы вернули, и первоначальной суммой займа. В "беспроцентном" займе она изначально нулевая или минимальная (только если есть комиссии за оформление).
   - **Как просрочка влияет?**
     - **Штрафы и пени:** За просрочку кредитор (банк, МФО или магазин) обычно начисляет штрафы или пени. Это может быть фиксированная сумма (например, 500–1000 руб.) или процент от просроченной суммы (0,1–1% в день, но не выше 0,1% в день по закону для потребкредитов с 2021 года).
     - **Дополнительные расходы:** Могут добавиться комиссии за взыскание долга, услуги коллекторов или даже судебные издержки. В итоге переплата вырастет на тысячи рублей (или больше, в зависимости от суммы займа и срока просрочки).
     - **Пример:** Займ 10 000 руб. на 6 месяцев без процентов. Если просрочить на 30 дней, пени в 1% в день = 300 руб. + возможный штраф 500 руб. = переплата 800 руб. (вместо 0).
   - **Итог:** Просрочка превращает "бесплатный" заём в платный. Чем дольше задержка, тем выше переплата — она может превысить даже 20–50% от суммы займа.

#### 2. **Влияние на ПСК (Полную Стоимость Кредита)**
   - **Что это?** ПСК — это годовая ставка, которая отражает **все** расходы по займу (проценты, комиссии, страховки + штрафы). В "беспроцентном" займе ПСК изначально низкая (0–10%, если только фиксированные fees).
   - **Как просрочка влияет?**
     - **Перерасчёт ПСК:** По закону (п. 10 ст. 5 ФЗ-353) ПСК включает все платежи, в том числе штрафы за просрочку. Если вы доплатите пени, ПСК автоматически вырастет — она пересчитывается на основе фактических выплат.
     - **Рост ставки:** Если просрочка приводит к штрафам, ПСК может подскочить до 50–365% годовых или выше (в зависимости от условий). Для беспроцентных займов это особенно заметно, так как базовая ставка нулевая.
     - **Пример:** Изначальный ПСК = 0%. После просрочки на 1 месяц с пенями 1000 руб. на займ 10 000 руб. ПСК может стать 20–30% годовых (или больше, если просрочка хроническая).
     - **Важно:** Банки/МФО обязаны раскрывать ПСК в договоре, но если штрафы не были указаны как обязательные, их учёт в ПСК возможен только после фактической просрочки. В суде (если дойдёт) просрочка может быть учтена для снижения штрафа, но не отменит его полностью.
   - **Итог:** ПСК вырастет пропорционально штрафам, делая заём дороже. Это влияет на сравнение с другими кредитами — ваш "бесплатный" заём станет одним из самых дорогих.

#### Советы, чтобы избежать проблем
- **Проверьте договор:** Ищите разделы о штрафах (ст. 6 ФЗ-353 лимитирует их). В рассрочках (типа "0%") штрафы часто скрыты в условиях.
- **Если просрочили:** Свяжитесь с кредитором сразу — иногда можно реструктуризировать долг или списать часть пени.
- **Альтернативы:** Если заём уже просрочен, рассмотрите рефинансирование под реальные низкие проценты, чтобы избежать роста ПСК.
- **Правовая защита:** Если штрафы кажутся завышенными (>20% годовых или необоснованными), обратитесь в Роспотребнадзор или суд. По 214-ФЗ "О потребкредитах" вы имеете право на прозрачность.

Это общая информация, не юридическая консультация. Для вашего случая изучите договор или обратитесь к специалисту. Если нужны детали по конкретному займу, уточните!

---

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context_preview}

ВОПРОС КЛИЕНТА:
{question}

ИНСТРУКЦИИ ПО ФОРМИРОВАНИЮ ОТВЕТА:

1. **Формат и структура:**
   - Используй Markdown-форматирование (###, ####, **, -, списки)
   - Для сложных тем: начни с основного заголовка (###)
   - Используй подзаголовки (####) для разделения логических блоков
   - Выделяй **жирным** ключевые термины и важные моменты
   - Используй маркированные и нумерованные списки
   - Добавляй блоки "Важно:", "Обратите внимание:", "Пример:" где нужно

2. **Содержание:**
   - Отвечай СТРОГО на основе информации из контекста
   - Используй конкретные цифры, проценты, сроки из контекста
   - Приводи примеры и кейсы
   - Ссылайся на законы, регламенты, документы если они упомянуты
   - Объясняй термины простым языком
   - Давай практические советы

3. **Стиль:**
   - Пиши понятно и структурированно
   - Для простых вопросов: краткий ответ (2-4 абзаца)
   - Для сложных вопросов: детальный ответ с разделами и примерами
   - Логика: от общего к частному, пошагово
   - Добавляй предупреждения/оговорки где важно

4. **Завершение:**
   - Практические рекомендации если уместно
   - Для сложных финансовых/юридических тем: оговорка о консультации со специалистом

ОТВЕТ:"""
    else:
        prompt = f"""Ты — эксперт по финансам и банковским продуктам. Ответь на вопрос клиента.

ПРИМЕРЫ ОТВЕТОВ:

ПРИМЕР 1:
Вопрос: Как часто выплачивается купон по облигации?

Ответ: Процентная выплата по облигации, которую получает держатель облигации, обычно устанавливается в процентах годовых от номинала облигации. Купон может выплачиваться один или несколько раз в год. Купон обычно выплачивается на брокерский счет клиента.

---

ВОПРОС:
{question}

ТРЕБОВАНИЯ:
- Используй Markdown (###, **жирный**, списки)
- Структурируй логически
- Объясняй термины просто
- Приводи примеры
- Будь конкретным

ОТВЕТ:"""
    
    # Формируем запрос к клиенту
    response = client.chat.completions.create(
        # Выбираем любую допступную модель из предоставленного списка
        model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
        # Формируем сообщение
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        temperature=0.3,
    )
    # Формируем ответ на запрос и возвращаем его в результате работы функции
    return response.choices[0].message.content


# НОВОЕ: Загрузка прогресса из файла
def load_progress():
    """Загружает сохраненный прогресс"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


# НОВОЕ: Сохранение прогресса в файл
def save_progress(progress):
    """Сохраняет прогресс в файл"""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# НОВОЕ: Сохранение промежуточных результатов
def save_checkpoint(questions_df, answer_dict):
    """Сохраняет промежуточные результаты"""
    # Создаем копию DataFrame
    df_copy = questions_df.copy()
    
    # Заполняем ответы
    answer_list = []
    for idx in range(len(df_copy)):
        answer_list.append(answer_dict.get(idx, ""))
    
    df_copy['Ответы на вопрос'] = answer_list
    
    # Сохраняем
    df_copy.to_csv('submission.csv', index=False)


# НОВОЕ: Функция для обработки одного вопроса с retry и обработкой ошибок
def process_single_question_safe(idx, question, q_emb, train_texts, train_embeddings, tfidf_vectorizer, tfidf_matrix):
    """Обрабатывает один вопрос с обработкой ошибок и retry"""
    
    for attempt in range(MAX_RETRIES):
        try:
            # Ищем релевантные документы
            context = search_relevant_docs_optimized(
                q_emb, 
                train_texts, 
                train_embeddings, 
                tfidf_vectorizer,
                tfidf_matrix,
                question,
                top_k=4
            )
            
            # Генерируем ответ
            answer = answer_generation(question=question, context=context)
            
            return idx, answer, None  # успешно
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"\nОшибка при обработке вопроса {idx}, попытка {attempt + 1}/{MAX_RETRIES}: {str(e)}")
                time.sleep(2 ** attempt)  # экспоненциальная задержка
            else:
                error_msg = f"ОШИБКА после {MAX_RETRIES} попыток: {str(e)}"
                print(f"\n{error_msg} для вопроса {idx}")
                return idx, f"[Ошибка генерации: {str(e)[:100]}]", error_msg
    
    return idx, "[Ошибка: превышено количество попыток]", "Max retries exceeded"


# Блок кода для запуска. Пожалуйста оставляйте его в самом низу вашего скрипта,
# при необходимости добавить код - опишите функции выше и вставьте их вызов в блок после if
# в том порядке, в котором они нужны для запуска решения, пути к файлам оставьте неизменными.
if __name__ == "__main__":
    # RAG Fusion: загружаем обучающие данные
    print("Загрузка обучающих данных...")
    train_data = pd.read_csv('./train_data.csv')
    train_texts = train_data['text'].tolist()
    
    # RAG Fusion: ОПТИМИЗАЦИЯ - загружаем или создаем эмбеддинги с кешированием
    print("Подготовка эмбеддингов обучающих данных...")
    train_embeddings = load_or_create_train_embeddings(train_texts, EMBEDDER_API_KEY)
    
    # RAG Fusion: создаем TF-IDF индекс (замена BM25)
    print("Создание TF-IDF индекса...")
    tfidf_vectorizer, tfidf_matrix = create_tfidf_index(train_texts)
    
    # Считываем список вопросов
    questions = pd.read_csv('./questions.csv')
    # Выделяем список вопросов
    questions_list = questions['Вопрос'].tolist()
    
    # НОВОЕ: Загружаем сохраненный прогресс
    print("Проверка сохраненного прогресса...")
    progress = load_progress()
    
    # Определяем, какие вопросы уже обработаны
    processed_indices = set(progress.keys())
    total_questions = len(questions_list)
    
    if processed_indices:
        print(f"Найдено {len(processed_indices)} уже обработанных вопросов из {total_questions}")
    
    # ОПТИМИЗАЦИЯ: создаем эмбеддинги только для необработанных вопросов
    print("Создание эмбеддингов для вопросов...")
    questions_embeddings = get_questions_embeddings_batch(questions_list, EMBEDDER_API_KEY)
    
    # НОВОЕ: Словарь для хранения всех ответов (включая загруженные из прогресса)
    answer_dict = {int(k): v for k, v in progress.items()}
    
    # Определяем вопросы для обработки
    questions_to_process = [
        (idx, q, q_emb) 
        for idx, (q, q_emb) in enumerate(zip(questions_list, questions_embeddings))
        if idx not in processed_indices
    ]
    
    if not questions_to_process:
        print("Все вопросы уже обработаны!")
    else:
        print(f"Обработка {len(questions_to_process)} вопросов...")
        
        # ОПТИМИЗАЦИЯ: параллельная обработка вопросов с сохранением прогресса
        completed_count = len(processed_indices)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Создаем задачи только для необработанных вопросов
            futures = {}
            for idx, question, q_emb in questions_to_process:
                future = executor.submit(
                    process_single_question_safe,
                    idx,
                    question,
                    q_emb,
                    train_texts,
                    train_embeddings,
                    tfidf_vectorizer,
                    tfidf_matrix
                )
                futures[future] = idx
            
            # Собираем результаты с прогресс-баром и периодическим сохранением
            pbar = tqdm(total=len(questions_to_process), desc="Обработка вопросов")
            
            for future in as_completed(futures):
                try:
                    idx, answer, error = future.result()
                    answer_dict[idx] = answer
                    
                    # Сохраняем в прогресс
                    progress[str(idx)] = answer
                    
                    completed_count += 1
                    pbar.update(1)
                    
                    # НОВОЕ: Периодическое сохранение
                    if completed_count % CHECKPOINT_INTERVAL == 0:
                        save_progress(progress)
                        save_checkpoint(questions, answer_dict)
                        pbar.set_postfix({"Сохранено": completed_count})
                    
                    if error:
                        tqdm.write(f"Вопрос {idx} обработан с ошибкой")
                
                except Exception as e:
                    tqdm.write(f"Критическая ошибка при обработке: {str(e)}")
            
            pbar.close()
    
    # ФИНАЛЬНОЕ сохранение
    print("\nФинальное сохранение результатов...")
    save_progress(progress)
    
    # Формируем финальный список ответов в правильном порядке
    final_answer_list = []
    for idx in range(len(questions_list)):
        final_answer_list.append(answer_dict.get(idx, ""))
    
    # Добавляем в данные список ответов
    questions['Ответы на вопрос'] = final_answer_list
    
    # Сохраняем submission
    questions.to_csv('submission.csv', index=False)
    
    # НОВОЕ: Удаляем файл прогресса после успешного завершения
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    
    print("Готово! Результаты сохранены в submission.csv")
    print(f"Обработано вопросов: {len([a for a in final_answer_list if a])}/{len(questions_list)}")