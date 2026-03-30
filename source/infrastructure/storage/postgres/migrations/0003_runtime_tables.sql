CREATE TABLE IF NOT EXISTS user_item_interactions (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    event_type TEXT NOT NULL DEFAULT 'implicit',
    interacted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_item_interactions_user_time
    ON user_item_interactions (user_id, interacted_at DESC);

CREATE TABLE IF NOT EXISTS inference_requests (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    request_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_dir TEXT NOT NULL DEFAULT '',
    latency_ms INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inference_requests_endpoint_time
    ON inference_requests (endpoint, created_at DESC);
