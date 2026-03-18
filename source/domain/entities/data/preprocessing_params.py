from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
# Описывает параметры препроцессинга.
class PreprocessingParams:
    k_core: int = 2
    keep_recent_fraction: float = 0.6
    test_fraction: float = 0.25
    local_val_fraction: float = 0.2
    cold_max_interactions: int = 5
    warm_users_only: bool = True
    language_filter_enabled: bool = True
    interactions_chunksize: int = 200_000

    def validate(self) -> None:
        if self.k_core < 0:
            raise ValueError("k_core must be >= 0")
        if not 0 < self.keep_recent_fraction <= 1:
            raise ValueError("keep_recent_fraction must be in (0, 1]")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1)")
        if not 0 < self.local_val_fraction < 1:
            raise ValueError("local_val_fraction must be in (0, 1)")
        if self.cold_max_interactions < 0:
            raise ValueError("cold_max_interactions must be >= 0")
        if self.interactions_chunksize <= 0:
            raise ValueError("interactions_chunksize must be > 0")
