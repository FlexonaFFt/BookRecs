.PHONY: help init-env infra-up pipeline-up api-up demo-seed batch-emulate promote-run train-prepared lint lint-backend lint-frontend test up down down-volumes logs ps restart-pipeline restart-api

SERVICE ?= pipeline
DAYS ?= 5
END_DATE ?=
RUN_NAME ?=

help:
	@echo "Доступные команды:"
	@echo "  make init-env         # создать .env из .env.example (если .env отсутствует)"
	@echo "  make infra-up         # поднять postgres + minio + minio-init"
	@echo "  make pipeline-up      # собрать образ и запустить pipeline"
	@echo "  make api-up           # собрать образ и запустить inference API"
	@echo "  make demo-seed        # загрузить demo-таблицы в postgres из preprocessed parquet"
	@echo "  make batch-emulate    # эмуляция батч-запусков за N дней (DAYS=5 END_DATE=YYYY-MM-DD, с promote)"
	@echo "  make promote-run      # вручную промоутнуть run в active pointer (RUN_NAME=batch_YYYYMMDD)"
	@echo "  make train-prepared   # обучить модель на уже подготовленном датасете (BOOKRECS_TRAIN_DATASET_DIR=...)"
	@echo "  make lint             # запустить backend и frontend линтеры"
	@echo "  make lint-backend     # запустить ruff для backend-кода"
	@echo "  make lint-frontend    # запустить eslint для frontend-кода"
	@echo "  make test             # запустить unit-тесты"
	@echo "  make up               # infra-up + pipeline-up"
	@echo "  make down             # остановить все сервисы"
	@echo "  make down-volumes     # остановить сервисы и удалить volumes"
	@echo "  make logs SERVICE=... # смотреть логи сервиса (по умолчанию pipeline)"
	@echo "  make ps               # статус сервисов"
	@echo "  make restart-pipeline # пересобрать и перезапустить pipeline"

init-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Создан .env из .env.example"; \
	else \
		echo ".env уже существует"; \
	fi

infra-up: init-env
	docker compose up -d postgres minio minio-init

pipeline-up: init-env
	docker compose up --build pipeline

api-up: init-env
	docker compose up --build api

demo-seed: init-env
	docker compose run --build --rm api python -m source.interfaces.demo_seed_entrypoint

batch-emulate: init-env
	docker compose run --build --rm \
		-e BOOKRECS_BATCH_BACKFILL_DAYS=$(DAYS) \
		-e BOOKRECS_BATCH_END_DATE=$(END_DATE) \
		-e BOOKRECS_BATCH_BACKFILL_PROMOTE=true \
		pipeline python -m source.interfaces.batch_backfill_entrypoint

promote-run: init-env
	docker compose run --build --rm \
		-e BOOKRECS_PROMOTE_RUN_NAME=$(RUN_NAME) \
		pipeline python -m source.interfaces.promote_model_entrypoint

train-prepared: init-env
	docker compose run --build --rm \
		-v $(BOOKRECS_TRAIN_DATASET_DIR):/dataset:ro \
		-e BOOKRECS_TRAIN_DATASET_DIR=/dataset \
		pipeline python -m source.interfaces.train_entrypoint

lint: lint-backend lint-frontend

lint-backend:
	python3 -m ruff check source data

lint-frontend:
	cd frontend && npm run lint

test:
	python3 -m pytest

up: infra-up pipeline-up

down:
	docker compose down

down-volumes:
	docker compose down -v

logs: init-env
	docker compose logs -f $(SERVICE)

ps: init-env
	docker compose ps

restart-pipeline: init-env
	docker compose up --build --force-recreate pipeline

restart-api: init-env
	docker compose up --build --force-recreate api
