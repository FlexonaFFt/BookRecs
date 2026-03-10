# BookRecs

## Описание проекта
BookRecs — это ML-система рекомендаций книг (Goodreads YA) с фокусом на гибридный pipeline и устойчивость к cold-start по айтемам. Проект объединяет подготовку данных, обучение моделей, офлайн-оценку и онлайн-инференс через FastAPI.

Система построена в стиле Clean Architecture: доменная модель и use-case логика отделены от инфраструктуры (PostgreSQL, S3/MinIO, docker-сервисы). На текущем этапе реализованы batch training pipeline и inference API, которые работают с сохраненными артефактами `stage1/stage2/stage3`.

## Ссылки на документацию проекта
- [ML System Design Doc](docs/ML_System_Design.md)
- [Research Results](docs/Research_Results.md)

<details>
<summary><h2>Как запустить проект</h2></summary>

### 1. Подготовка окружения
```bash
make init-env
```

Проверь в `.env` ключевые параметры:
- `BOOKRECS_API_MODEL_URI` (например `artifacts/baselines/testmodel/models`)
- `BOOKRECS_PG_DSN`
- `BOOKRECS_S3_ENDPOINT`, `BOOKRECS_S3_BUCKET`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

### 2. Запуск инфраструктуры
```bash
make infra-up
```
Поднимутся `postgres`, `minio`, `minio-init`.

### 3. Запуск batch pipeline (опционально)
```bash
make pipeline-up
```

### 3.1 Эмуляция batch за 5 дней (offline replay / пилот)
```bash
make batch-emulate DAYS=5 END_DATE=2026-03-10
```
Если `END_DATE` не задан, берется текущая дата; для каждого дня запускается отдельный batch run с `run_name=batch_YYYYMMDD`.

### 4. Запуск API инференса
```bash
make api-up
```

### 5. Проверка API
```bash
curl http://localhost:8000/healthz
```

### Полезные команды
```bash
make ps
make logs SERVICE=api
make logs SERVICE=pipeline
make demo-seed
make batch-emulate DAYS=5 END_DATE=2026-03-10
make test
make down
make down-volumes
```

`make demo-seed` загружает demo-таблицы для фронтенда в PostgreSQL из preprocessed датасета:
- источник по умолчанию: `artifacts/tmp_preprocessed/goodreads_ya`
- миграции: `source/infrastructure/storage/postgres/migrations`
- лимиты по умолчанию:
  - `BOOKRECS_DEMO_USERS_LIMIT=2000`
  - `BOOKRECS_DEMO_MAX_HISTORY_PER_USER=100`
  - `BOOKRECS_DEMO_RESET=true`

### Локальный запуск без docker compose
```bash
python -m source.interfaces.pipeline_entrypoint
python -m source.interfaces.api_entrypoint
python -m source.interfaces.batch_backfill_entrypoint
```

</details>

## Airflow Batch DAG
В проект добавлен DAG `bookrecs_daily_batch` (файл `deploy/airflow/dags/bookrecs_batch_dag.py`) с `DockerOperator` и `catchup=True`.

Пример backfill в Airflow за 5 дней:
```bash
airflow dags backfill bookrecs_daily_batch -s 2026-03-06 -e 2026-03-10
```

## Product Pipeline
![Recommendation pipeline](docs/images/pipeline.png)

Pipeline состоит из пяти шагов:
1. `Data & Split` — загрузка и предобработка interactions/metadata, time split, разметка warm/cold.
2. `Candidate Generation` — объединение кандидатов из `CF`, `Content`, `Popular`.
3. `Pre-ranking` — легкий Stage-2 отбор top-M кандидатов.
4. `Final Ranking` — Stage-3 финальное ранжирование и postprocessing.
5. `Evaluation & Publish` — метрики (`NDCG@K`, `Recall@K`, `Coverage@K`, cold-срезы) и сохранение артефактов.

<details>
<summary><h2>Architecture</h2></summary>

Текущая структура проекта:

```text
BookRecs/
├── source/
│   ├── domain/                 # сущности и доменные модели
│   ├── application/            # use-cases и порты
│   ├── infrastructure/         # storage, ranking, preprocessing, inference
│   ├── interfaces/             # entrypoints: pipeline + api
│   └── tests/
├── docs/                       # проектная документация
├── artifacts/                  # модели и артефакты запусков
├── data/                       # raw/data assets
├── docker-compose.yml
├── Dockerfile
└── README.md
```

Слои взаимодействуют так:

```mermaid
flowchart LR
UI["Interfaces (API/Batch)"] --> APP["Application (Use Cases + Ports)"]
APP --> DOM["Domain (Entities)"]
APP --> INFRA["Infrastructure (Storage/Models/Processing)"]
INFRA --> EXT["Postgres / S3 / Local Artifacts"]
```

</details>

<details>
<summary><h2>Artifacts</h2></summary>

### Training run artifacts
Каждый run обучения сохраняется в каталоге:

```text
artifacts/runs/<run_id>/
├── manifest.json
├── metrics.json
├── timings.json
├── train.log.jsonl
└── models/
    ├── stage1.pkl
    ├── stage2.pkl
    ├── stage3.pkl
    └── metrics_snapshot.json
```

### Baseline/test artifacts
Для быстрых запусков и smoke-тестов используется:

```text
artifacts/baselines/testmodel/
└── models/
    ├── stage1.pkl
    ├── stage2.pkl
    ├── stage3.pkl
    └── metrics_snapshot.json
```

### Prepared dataset artifacts
После `prepare` сохраняются:

```text
artifacts/tmp_preprocessed/goodreads_ya/
├── books.parquet
├── train.parquet
├── test.parquet
├── local_train.parquet
├── local_val.parquet
├── local_val_warm.parquet
├── local_val_cold.parquet
├── summary.json
└── manifest.json
```

</details>
