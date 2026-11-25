CREATE TABLE IF NOT EXISTS vector_records (
    id BIGINT PRIMARY KEY,
    external_key TEXT UNIQUE,
    category TEXT NOT NULL DEFAULT 'general',
    price DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    tags TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vr_category_price ON vector_records (category, price);
CREATE INDEX IF NOT EXISTS idx_vr_tags ON vector_records USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_vr_metadata ON vector_records USING GIN (metadata);
