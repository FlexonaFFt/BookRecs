#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${BOOKRECS_DATASET_NAME:-goodreads_ya}"
RAW_DIR="${BOOKRECS_RAW_DIR:-data/raw_data}"
BOOKS_RAW_URI="${BOOKRECS_BOOKS_RAW_URI:-${RAW_DIR}/books.json.gz}"
INTERACTIONS_RAW_URI="${BOOKRECS_INTERACTIONS_RAW_URI:-${RAW_DIR}/interactions.json.gz}"
PREP_DIR="${BOOKRECS_PREP_DIR:-artifacts/tmp_preprocessed/${DATASET_NAME}}"
PREP_MANIFEST="${PREP_DIR}/manifest.json"

STORE_BACKEND="${BOOKRECS_STORE_BACKEND:-local}"             # local | s3
REGISTRY_BACKEND="${BOOKRECS_REGISTRY_BACKEND:-memory}"      # memory | postgres
S3_PREFIX="${BOOKRECS_S3_PREFIX:-s3://bookrecs/datasets/${DATASET_NAME}}"
PG_DSN="${BOOKRECS_PG_DSN:-}"
PG_MIGRATION_FILE="${BOOKRECS_PG_MIGRATION_FILE:-configs/sql/001_init.sql}"
S3_BUCKET="${BOOKRECS_S3_BUCKET:-}"
S3_REGION="${BOOKRECS_S3_REGION:-us-east-1}"
S3_ENDPOINT="${BOOKRECS_S3_ENDPOINT:-}"

K_CORE="${BOOKRECS_K_CORE:-2}"
KEEP_RECENT_FRACTION="${BOOKRECS_KEEP_RECENT_FRACTION:-0.6}"
TEST_FRACTION="${BOOKRECS_TEST_FRACTION:-0.25}"
LOCAL_VAL_FRACTION="${BOOKRECS_LOCAL_VAL_FRACTION:-0.2}"
INTERACTIONS_CHUNKSIZE="${BOOKRECS_INTERACTIONS_CHUNKSIZE:-200000}"

echo "[entrypoint] dataset=${DATASET_NAME}"
echo "[entrypoint] raw books=${BOOKS_RAW_URI}"
echo "[entrypoint] raw interactions=${INTERACTIONS_RAW_URI}"
echo "[entrypoint] prep manifest=${PREP_MANIFEST}"
echo "[entrypoint] store_backend=${STORE_BACKEND} registry_backend=${REGISTRY_BACKEND}"

if [[ "${REGISTRY_BACKEND}" == "postgres" ]]; then
  if [[ -z "${PG_DSN}" ]]; then
    echo "[entrypoint] BOOKRECS_PG_DSN is required for postgres backend"
    exit 1
  fi
  echo "[entrypoint] applying postgres migration: ${PG_MIGRATION_FILE}"
  python -m source.interfaces.migrate
fi

# 1) If raw dataset is missing -> download it.
if [[ ! -f "${BOOKS_RAW_URI}" || ! -f "${INTERACTIONS_RAW_URI}" ]]; then
  echo "[entrypoint] raw dataset not found, downloading..."
  python -m data.goodreads --raw-dir "${RAW_DIR}"
else
  echo "[entrypoint] raw dataset exists"
fi

# 2) If preprocessed dataset is missing -> preprocessing + publish.
# 3) If preprocessed dataset exists -> reuse local artifacts and publish to S3/PG.
if [[ -f "${PREP_MANIFEST}" ]]; then
  echo "[entrypoint] preprocessed dataset exists, reuse local artifacts and run registry/store flow"
else
  echo "[entrypoint] preprocessed dataset not found, run full preprocessing"
fi

python -m source.interfaces.cli prepare-data \
  --dataset-name "${DATASET_NAME}" \
  --books-raw-uri "${BOOKS_RAW_URI}" \
  --interactions-raw-uri "${INTERACTIONS_RAW_URI}" \
  --s3-prefix "${S3_PREFIX}" \
  --k-core "${K_CORE}" \
  --keep-recent-fraction "${KEEP_RECENT_FRACTION}" \
  --test-fraction "${TEST_FRACTION}" \
  --local-val-fraction "${LOCAL_VAL_FRACTION}" \
  --interactions-chunksize "${INTERACTIONS_CHUNKSIZE}" \
  --registry-backend "${REGISTRY_BACKEND}" \
  --pg-dsn "${PG_DSN}" \
  --store-backend "${STORE_BACKEND}" \
  --s3-bucket "${S3_BUCKET}" \
  --s3-region "${S3_REGION}" \
  --s3-endpoint "${S3_ENDPOINT}"

echo "[entrypoint] completed"
