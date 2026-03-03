from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    pg_dsn: str
    s3_bucket: str
    s3_region: str
    s3_endpoint: str


def load_settings() -> Settings:
    return Settings(
        pg_dsn=os.getenv("BOOKRECS_PG_DSN", ""),
        s3_bucket=os.getenv("BOOKRECS_S3_BUCKET", ""),
        s3_region=os.getenv("BOOKRECS_S3_REGION", "us-east-1"),
        s3_endpoint=os.getenv("BOOKRECS_S3_ENDPOINT", ""),
    )
