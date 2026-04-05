CREATE TABLE IF NOT EXISTS training_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    interactions_count BIGINT NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_checkpoints_trained_at
    ON training_checkpoints (trained_at DESC);
