from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from source.domain.entities.data.preprocessing_params import PreprocessingParams


@dataclass(frozen=True)
class DatasetVersion:
    dataset_name: str
    version_id: str
    params_hash: str
    s3_prefix: str
    params: PreprocessingParams
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stats: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name is required")
        if not self.version_id.strip():
            raise ValueError("version_id is required")
        if not self.params_hash.strip():
            raise ValueError("params_hash is required")
        if not self.s3_prefix.strip():
            raise ValueError("s3_prefix is required")
        self.params.validate()
