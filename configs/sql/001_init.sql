CREATE TABLE IF NOT EXISTS dataset_registry (
    dataset_name TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    version_id TEXT NOT NULL,
    s3_prefix TEXT NOT NULL,
    params_json JSONB NOT NULL,
    stats_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (dataset_name, params_hash)
);

CREATE INDEX IF NOT EXISTS idx_dataset_registry_dataset_name
    ON dataset_registry (dataset_name);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NULL,
    message TEXT NOT NULL DEFAULT '',
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_pipeline_name
    ON pipeline_runs (pipeline_name);
