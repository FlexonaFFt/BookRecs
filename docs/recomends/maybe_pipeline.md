# Recommender pipeline (Goodreads): structure

## 1) Offline (training)

### 1.1 Inputs (raw data)
- `User-Item interactions`: ratings / shelves / reviews / clicks (если есть)
- `Book metadata`: author, genres/tags, year, language, series
- `Text data`: description, reviews text (опционально)
- (опционально) `Context`: timestamp, device, locale

### 1.2 Preprocessing
- ID mapping: `user_id -> user_idx`, `book_id -> item_idx`
- Filtering: min interactions per user/item
- Split: time-based split или leave-last-out per user
- Labeling:
  - для top-K: positive interactions (например rating >= 4)
  - negative sampling (случайные / popularity-based)
- Feature normalization: log(popularity), recency features и т.п.

### 1.3 Train candidate model (retrieval)
- Model: `Implicit ALS` или `BPR-MF`
- Input: interaction matrix (user_idx, item_idx, weight)
- Output:
  - `user_emb` (матрица эмбеддингов пользователей)
  - `item_emb` (матрица эмбеддингов книг)
  - (опционально) ANN index по `item_emb` для быстрого поиска

### 1.4 Train text encoder (content) (опционально, но полезно)
- Model: `BERT/SBERT` (fine-tune или inference-only)
- Input: texts (description/reviews) + book_id
- Output:
  - `text_emb(book)` (вектор книги по тексту)

### 1.5 Build ranker training set (feature table)
- Entities: пары `(user, item)` из:
  - positives (из train)
  - negatives (sampled)
  - candidates (top-N из candidate model)
- Features (примеры):
  - `cf_score`: dot(user_emb, item_emb)
  - `cf_sim`: similarity metrics
  - `text_sim`: similarity(user_profile_text_emb, text_emb(book))
  - `genre_match`, `author_match`, `tag_overlap`
  - `popularity`, `avg_rating`, `recency`, `language_match`
  - user stats: `user_mean_rating`, `user_activity`
- Labels: relevance (binary или graded)

### 1.6 Train ranking model (re-ranking)
- Model: `LightGBM LambdaRank` / `CatBoost Ranking` (или neural ranker)
- Input: (user, item, features, label), grouping by user
- Output:
  - `ranker_model`: функция `score(user, item, features) -> float`

### 1.7 Store artifacts (model registry)
- `id_maps` (user/item)
- `candidate_model` artifacts: `user_emb`, `item_emb`, ANN index (если есть)
- `text_encoder` weights (если обучали) + `text_emb(book)`
- `ranker_model`
- `feature_schema` (список/версии фич)
- (опционально) offline caches: popularity tables, metadata dictionaries


## 2) Online (inference / serving)

### 2.1 Input request
- Required:
  - `user_id`
- Optional:
  - `context` (time/device/locale)
  - `constraints` (language, age rating, availability, exclude list)

### 2.2 Candidate retrieval (fast)
- Convert: `user_id -> user_idx` (через id_map)
- Retrieve top-N items:
  - `Top-N = ANN search` по `item_emb` с вектором `user_emb[user_idx]`
  - или direct dot-product (если матрицы небольшие)
- Output: list of candidate `item_idx` (например N=200..1000)

### 2.3 Feature generation for candidates
- Join required data:
  - metadata by item_idx
  - precomputed text_emb(book) (если есть)
  - user stats / user profile
- Compute features per (user, candidate item):
  - `cf_score`, `text_sim`, `genre_match`, `popularity`, etc.
- Output: feature matrix for candidates

### 2.4 Scoring and re-ranking
- Apply `ranker_model`:
  - `score_i = ranker(user, item_i, features_i)`
- Sort candidates by score
- Output: ranked list

### 2.5 Post-processing
- Filters:
  - constraints (language/availability/exclude)
- Business rules:
  - de-dup authors/series (если нужно)
- Diversity/novelty:
  - re-ranking с penalty за слишком похожие items (опционально)
- Output: final top-K

### 2.6 Output response
- `Top-K recommendations`: `[book_id, score]`
- (опционально) explanations:
  - top contributing features (для tree models)
  - nearest liked items (по embeddings)