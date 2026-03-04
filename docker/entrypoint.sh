#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${BOOKRECS_DATASET_NAME:-goodreads_ya}"
RAW_DIR="${BOOKRECS_RAW_DIR:-data/raw_data}"
BOOKS_RAW_URI="${BOOKRECS_BOOKS_RAW_URI:-${RAW_DIR}/books.json.gz}"
INTERACTIONS_RAW_URI="${BOOKRECS_INTERACTIONS_RAW_URI:-${RAW_DIR}/interactions.json.gz}"

STORE_BACKEND="${BOOKRECS_STORE_BACKEND:-local}"             # local | s3
REGISTRY_BACKEND="${BOOKRECS_REGISTRY_BACKEND:-memory}"      # memory | postgres
S3_PREFIX="${BOOKRECS_S3_PREFIX:-s3://bookrecs/datasets/${DATASET_NAME}}"
PG_DSN="${BOOKRECS_PG_DSN:-}"
PG_MIGRATION_PATH="${BOOKRECS_PG_MIGRATION_PATH:-source/infrastructure/storage/postgres/migrations}"
S3_BUCKET="${BOOKRECS_S3_BUCKET:-}"
S3_REGION="${BOOKRECS_S3_REGION:-us-east-1}"
S3_ENDPOINT="${BOOKRECS_S3_ENDPOINT:-}"
DATASET_DIR="${BOOKRECS_TRAIN_DATASET_DIR:-artifacts/tmp_preprocessed/${DATASET_NAME}}"
OUTPUT_ROOT="${BOOKRECS_TRAIN_OUTPUT_ROOT:-artifacts/runs}"
RUN_NAME="${BOOKRECS_TRAIN_RUN_NAME:-}"
EVAL_USERS_LIMIT="${BOOKRECS_TRAIN_EVAL_USERS_LIMIT:-2000}"
CANDIDATE_POOL_SIZE="${BOOKRECS_TRAIN_CANDIDATE_POOL_SIZE:-1000}"
CANDIDATE_PER_SOURCE_LIMIT="${BOOKRECS_TRAIN_PER_SOURCE_LIMIT:-300}"
PRE_TOP_M="${BOOKRECS_TRAIN_PRE_TOP_M:-300}"
FINAL_TOP_K="${BOOKRECS_TRAIN_FINAL_TOP_K:-10}"
CF_MAX_NEIGHBORS="${BOOKRECS_TRAIN_CF_MAX_NEIGHBORS:-120}"
CONTENT_MAX_NEIGHBORS="${BOOKRECS_TRAIN_CONTENT_MAX_NEIGHBORS:-120}"
TRAIN_SEED="${BOOKRECS_TRAIN_SEED:-42}"
K_CORE="${BOOKRECS_K_CORE:-2}"
KEEP_RECENT_FRACTION="${BOOKRECS_KEEP_RECENT_FRACTION:-0.6}"
TEST_FRACTION="${BOOKRECS_TEST_FRACTION:-0.25}"
LOCAL_VAL_FRACTION="${BOOKRECS_LOCAL_VAL_FRACTION:-0.2}"
WARM_USERS_ONLY="${BOOKRECS_WARM_USERS_ONLY:-true}"
LANGUAGE_FILTER_ENABLED="${BOOKRECS_LANGUAGE_FILTER_ENABLED:-true}"
INTERACTIONS_CHUNKSIZE="${BOOKRECS_INTERACTIONS_CHUNKSIZE:-200000}"
SKIP_PREPARE="${BOOKRECS_SKIP_PREPARE:-false}"
SKIP_TRAIN="${BOOKRECS_SKIP_TRAIN:-false}"
RUN_MIGRATE="${BOOKRECS_RUN_MIGRATE:-true}"

echo "[entrypoint] dataset=${DATASET_NAME}"
echo "[entrypoint] raw books=${BOOKS_RAW_URI}"
echo "[entrypoint] raw interactions=${INTERACTIONS_RAW_URI}"
echo "[entrypoint] dataset_dir=${DATASET_DIR}"
echo "[entrypoint] store_backend=${STORE_BACKEND} registry_backend=${REGISTRY_BACKEND}"

CMD=(python -m source.interfaces.cli run \
  --dataset-name "${DATASET_NAME}" \
  --raw-dir "${RAW_DIR}" \
  --books-raw-uri "${BOOKS_RAW_URI}" \
  --interactions-raw-uri "${INTERACTIONS_RAW_URI}" \
  --dataset-dir "${DATASET_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --k-core "${K_CORE}" \
  --keep-recent-fraction "${KEEP_RECENT_FRACTION}" \
  --test-fraction "${TEST_FRACTION}" \
  --local-val-fraction "${LOCAL_VAL_FRACTION}" \
  --interactions-chunksize "${INTERACTIONS_CHUNKSIZE}" \
  --eval-users-limit "${EVAL_USERS_LIMIT}" \
  --candidate-pool-size "${CANDIDATE_POOL_SIZE}" \
  --candidate-per-source-limit "${CANDIDATE_PER_SOURCE_LIMIT}" \
  --pre-top-m "${PRE_TOP_M}" \
  --final-top-k "${FINAL_TOP_K}" \
  --cf-max-neighbors "${CF_MAX_NEIGHBORS}" \
  --content-max-neighbors "${CONTENT_MAX_NEIGHBORS}" \
  --seed "${TRAIN_SEED}" \
  --s3-prefix "${S3_PREFIX}" \
  --registry-backend "${REGISTRY_BACKEND}" \
  --pg-dsn "${PG_DSN}" \
  --store-backend "${STORE_BACKEND}" \
  --s3-bucket "${S3_BUCKET}" \
  --s3-region "${S3_REGION}" \
  --s3-endpoint "${S3_ENDPOINT}" \
  --migration-path "${PG_MIGRATION_PATH}")

if [[ "${WARM_USERS_ONLY}" == "true" ]]; then
  CMD+=(--warm-users-only)
else
  CMD+=(--no-warm-users-only)
fi

if [[ "${LANGUAGE_FILTER_ENABLED}" == "true" ]]; then
  CMD+=(--language-filter-enabled)
else
  CMD+=(--no-language-filter-enabled)
fi

if [[ "${SKIP_PREPARE}" == "true" ]]; then
  CMD+=(--skip-prepare)
else
  CMD+=(--no-skip-prepare)
fi

if [[ "${SKIP_TRAIN}" == "true" ]]; then
  CMD+=(--skip-train)
else
  CMD+=(--no-skip-train)
fi

if [[ "${RUN_MIGRATE}" == "true" ]]; then
  CMD+=(--migrate)
else
  CMD+=(--no-migrate)
fi

"${CMD[@]}"

echo "[entrypoint] completed"
