CREATE TABLE IF NOT EXISTS demo_books (
    item_id BIGINT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    image_url TEXT NOT NULL DEFAULT '',
    authors_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    tags_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    series_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_demo_books_title
    ON demo_books (title);

CREATE TABLE IF NOT EXISTS demo_users (
    user_id TEXT PRIMARY KEY,
    history_len INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS demo_user_seen (
    user_id TEXT NOT NULL REFERENCES demo_users(user_id) ON DELETE CASCADE,
    item_id BIGINT NOT NULL REFERENCES demo_books(item_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL DEFAULT 'seed',
    event_ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, item_id)
);

CREATE INDEX IF NOT EXISTS idx_demo_user_seen_user_time
    ON demo_user_seen (user_id, event_ts DESC);

CREATE INDEX IF NOT EXISTS idx_demo_user_seen_item
    ON demo_user_seen (item_id);
