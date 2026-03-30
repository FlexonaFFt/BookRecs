CREATE TABLE IF NOT EXISTS offline_experiments (
    experiment_id TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    holdout_version_id TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_results (
    id BIGSERIAL PRIMARY KEY,
    experiment_id TEXT NOT NULL
        REFERENCES offline_experiments(experiment_id) ON DELETE CASCADE,
    model_tag TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'test',
    split TEXT NOT NULL DEFAULT 'overall',
    k INT NOT NULL DEFAULT 10,
    ndcg_at_k FLOAT,
    recall_at_k FLOAT,
    coverage_at_k FLOAT,
    extra_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (experiment_id, model_tag, split)
);

CREATE INDEX IF NOT EXISTS idx_experiment_results_exp
    ON experiment_results (experiment_id);
