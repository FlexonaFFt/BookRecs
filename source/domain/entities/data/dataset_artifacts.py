from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetArtifacts:
    books_uri: str
    train_uri: str
    test_uri: str
    local_train_uri: str
    local_val_uri: str
    local_val_warm_uri: str
    local_val_cold_uri: str
    summary_uri: str
    manifest_uri: str

    def uris(self) -> list[str]:
        return [
            self.books_uri,
            self.train_uri,
            self.test_uri,
            self.local_train_uri,
            self.local_val_uri,
            self.local_val_warm_uri,
            self.local_val_cold_uri,
            self.summary_uri,
            self.manifest_uri,
        ]
