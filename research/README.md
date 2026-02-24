# Research (RecSys, cold-start)

Небольшая research-зона для быстрых экспериментов по рекомендательной системе с фокусом на `cold items`.

## Что уже есть

- единый `precompute` для research-данных (чтобы не гонять предобработку перед каждым методом)
- общий `evaluator` с метриками:
  - `NDCG@10`
  - `Recall@10`
  - `Coverage@10`
  - `warm_*`
  - `cold_*`
- baseline:
  - `TopPopular`
  - `Random` (только как sanity-check, опционально)
- content-метод:
  - `TF-IDF` по `title + authors + series + tags + description`
- hybrid-метод:
  - `content + popularity`

## Текущий research-срез данных

Использовали облегченный режим (для скорости):
- `k_core = 5`
- `keep_recent_fraction = 0.05`
- `test_fraction = 0.25`

Итоговый размер (по `summary.json`):
- `train_rows = 1,023,917`
- `val_rows = 203,863`
- `val_users = 46,091`
- `val_cold_rows = 2,411`
- `val_cold_ratio ≈ 1.18%`

Это нормально для research-итерации: хватает, чтобы сравнивать методы и видеть поведение на cold.

## Результаты (текущие)

### 1) TopPopular (baseline)

Сильный `overall`, но cold не решает вообще:
- `ndcg@10 = 0.0329`
- `recall@10 = 0.0472`
- `coverage@10 = 0.0011`
- `cold_ndcg@10 = 0.0`
- `cold_recall@10 = 0.0`

Вывод: хороший warm-baseline, но не подходит как решение задачи cold-start.

### 2) Content TF-IDF

Слабее по `overall`, но уже умеет работать с cold:
- `ndcg@10 = 0.0067`
- `recall@10 = 0.0122`
- `coverage@10 = 0.3917`
- `cold_ndcg@10 = 0.00621`
- `cold_recall@10 = 0.00972`

Вывод: это хороший cold-aware компонент для гибрида.

### 3) Hybrid (content 0.8 / popular 0.2)

Заметно лучше content-only по `overall`, cold сохраняется:
- `ndcg@10 = 0.0101`
- `recall@10 = 0.0199`
- `coverage@10 = 0.3655`
- `cold_ndcg@10 = 0.00621`
- `cold_recall@10 = 0.00972`

### 4) Hybrid (content 0.7 / popular 0.3) — лучший из текущих hybrid

Лучший баланс на текущем прогоне:
- `ndcg@10 = 0.0131`
- `recall@10 = 0.0261`
- `coverage@10 = 0.3393`
- `cold_ndcg@10 = 0.00604`
- `cold_recall@10 = 0.00972`

Вывод: `overall` вырос, `cold` почти сохранился.

### 5) Увеличение кандидатов (`top500`)

Почти не помогло:
- `content_tfidf_top500` ≈ как `content_tfidf`
- `hybrid_70_30_top500` даже хуже `hybrid_70_30`

Вывод: bottleneck сейчас не в размере candidate list, а в качестве контентного сигнала / схеме смешивания.

## Что это значит (простыми словами)

- `TopPopular` хорошо угадывает "обычные" книги, но не умеет новые (`cold`)
- `Content` умеет новые, но хуже в целом
- `Hybrid` — правильное направление (компромисс)

Сейчас лучший кандидат для MVP в research:
- `hybrid_70_30`

## Запуск

```bash
cd /Users/flexonafft/BookRecs/research
docker compose build
```

Подготовка данных:
```bash
docker compose run --rm data_preprocess python data/creator.py --rebuild-processed --n-neg 0 --interactions-chunksize 200000 --keep-recent-fraction 0.05 --k-core 5
```

Precompute:
```bash
docker compose run --rm precompute
```

Методы:
```bash
docker compose run --rm baselines
docker compose run --rm content
docker compose run --rm hybrid
```

## Где смотреть результаты

- JSON-метрики: `research/results/**/*.json`
- сравнение в ноутбуке: `research/notebooks/metrics_comparison.ipynb`

---

Тонкий момент: это пока **research-режим** на урезанном датасете. Для финальных выводов надо будет прогнать лучшие методы на более полном срезе / полном датасете.
