# Recommender pipeline (Goodreads): product v1 (resource-limited)

## 1) Offline (training)

### 1.1 Inputs (prepared data)
- `Interactions`: `local_train.pq`
- `Validation`: `local_val.pq`, `local_val_warm.pq`, `local_val_cold.pq`, `eval_ground_truth.pq`
- `Books metadata`: `books.pq`
- (опционально) готовые `item_popularity.pq`, `user_histories.pq`

### 1.2 Data bundle build
- Валидация схемы данных
- Фиксированный `time-based split` (используем готовый split)
- Построение:
  - `item_text` (title/tags/authors/series/description)
  - `item_popularity`
  - `seen_items_by_user`
  - `warm_item_ids`, `cold_item_ids`
- Output: единый `bundle` для всех моделей

### 1.3 Train base components
- `TopPopular` (baseline + fallback)
- `Content TF-IDF` (cold-aware)
- `Item2Item CF` (co-occurrence / similarity; warm personalization)

### 1.4 Hybrid training (lightweight ranking)
- Candidate sources:
  - `CF candidates`
  - `Content candidates`
  - `Popular/New candidates`
- Merge candidates (union)
- Score normalization per source
- Weighted blend:
  - `score = w_cf * cf + w_content * content + w_pop * pop (+ w_fresh * fresh)`
- Небольшой tuning (3-5 конфигов)

### 1.5 Offline evaluation
- Metrics:
  - `NDCG@10`, `Recall@10`, `Coverage@10`
  - `warm_*`, `cold_*`
- Champion selection:
  - composite score
  - constraints на `cold` и `coverage`****

### 1.6 Save artifacts (production-ready)
- `top_popular` artifacts
- `content_tfidf` artifacts
- `item2item` artifacts
- `champion_config.json`
- `metrics_table.csv/json`
- `run_manifest.json`

### 1.7 Batch export for serving
- `top-N recommendations` per user (precomputed)
- `fallback_popular/new`
- Версия модели / дата расчета


## 2) Online (inference / serving)

### 2.1 Input request
- `user_id`
- (опционально) `constraints` (availability, exclude list, language)

### 2.2 Fast response (batch-first)
- Читаем precomputed `top-N` по `user_id`
- Если нет пользователя / нет выдачи:
  - fallback `popular/new`

### 2.3 Post-processing
- Фильтры: availability / exclude
- Refill из fallback, если список укоротился

### 2.4 Output
- `Top-K recommendations`: `[book_id, score]`
- `model_version` (для логирования и A/B)


## 3) Future upgrades (without redesign)
- Заменить `Item2Item` -> `ALS/BPR`
- Заменить TF-IDF -> `SBERT`
- Заменить blend -> `LTR ranker`
- Добавить ANN / online candidate retrieval
