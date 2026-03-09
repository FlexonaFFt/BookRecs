.PHONY: help init-env infra-up pipeline-up api-up demo-seed test up down down-volumes logs ps restart-pipeline restart-api

SERVICE ?= pipeline

help:
	@echo "Доступные команды:"
	@echo "  make init-env         # создать .env из .env.example (если .env отсутствует)"
	@echo "  make infra-up         # поднять postgres + minio + minio-init"
	@echo "  make pipeline-up      # собрать образ и запустить pipeline"
	@echo "  make api-up           # собрать образ и запустить inference API"
	@echo "  make demo-seed        # загрузить demo-таблицы в postgres из preprocessed parquet"
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
