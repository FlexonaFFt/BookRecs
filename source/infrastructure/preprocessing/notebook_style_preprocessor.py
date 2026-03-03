from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from source.application.ports import PreprocessorPort
from source.domain.entities import DatasetArtifacts, DatasetSource, PreprocessingParams


class NotebookStylePreprocessorStub(PreprocessorPort):


    def __init__(self, work_dir: str = "artifacts/tmp_preprocessed") -> None:
        self._work_dir = Path(work_dir)

    def run(self, source: DatasetSource, params: PreprocessingParams) -> DatasetArtifacts:
        target = self._work_dir / source.dataset_name
        target.mkdir(parents=True, exist_ok=True)

        # Заглушки parquet-файлов. Здесь будут реальные данные на следующем этапе.
        books = target / "books.parquet"
        train = target / "train.parquet"
        test = target / "test.parquet"
        local_train = target / "local_train.parquet"
        local_val = target / "local_val.parquet"
        local_val_warm = target / "local_val_warm.parquet"
        local_val_cold = target / "local_val_cold.parquet"

        for path in [books, train, test, local_train, local_val, local_val_warm, local_val_cold]:
            path.write_text("stub\n", encoding="utf-8")

        summary = target / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "dataset_name": source.dataset_name,
                    "books_raw_uri": source.books_raw_uri,
                    "interactions_raw_uri": source.interactions_raw_uri,
                    "params": asdict(params),
                    "mode": "stub",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        manifest = target / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "CREATED_BY_STUB",
                    "files": [
                        "books.parquet",
                        "train.parquet",
                        "test.parquet",
                        "local_train.parquet",
                        "local_val.parquet",
                        "local_val_warm.parquet",
                        "local_val_cold.parquet",
                        "summary.json",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return DatasetArtifacts(
            books_uri=str(books),
            train_uri=str(train),
            test_uri=str(test),
            local_train_uri=str(local_train),
            local_val_uri=str(local_val),
            local_val_warm_uri=str(local_val_warm),
            local_val_cold_uri=str(local_val_cold),
            summary_uri=str(summary),
            manifest_uri=str(manifest),
        )
