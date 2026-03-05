from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
# Описывает источник датасета.
class DatasetSource:
    dataset_name: str
    books_raw_uri: str
    interactions_raw_uri: str

    def validate(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name is required")
        if not self.books_raw_uri.strip():
            raise ValueError("books_raw_uri is required")
        if not self.interactions_raw_uri.strip():
            raise ValueError("interactions_raw_uri is required")
