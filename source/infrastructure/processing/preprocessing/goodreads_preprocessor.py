from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from source.application.ports import PreprocessorPort
from source.domain.entities import DatasetArtifacts, DatasetSource, PreprocessingParams
# Готовит очищенные датасеты и локальные обучающие/валидационные сплиты из сырых данных Goodreads.
class GoodreadsPreprocessor(PreprocessorPort):


    def __init__(self, work_dir: str = "artifacts/tmp_preprocessed") -> None:
        self._work_dir = Path(work_dir)

    def run(self, source: DatasetSource, params: PreprocessingParams) -> DatasetArtifacts:
        if pd is None:
            raise RuntimeError(
                "pandas is required for preprocessing. Install project dependencies first."
            )
        source.validate()
        params.validate()

        cached = self._cached_artifacts(source.dataset_name)
        if cached is not None:
            print("[prepare] Используются локальные кэшированные артефакты.", flush=True)
            return cached

        print(f"[prepare] Чтение books из {source.books_raw_uri}", flush=True)
        books_raw = self._read_books(source.books_raw_uri)
        print(f"[prepare] Книги загружены: {len(books_raw)} строк", flush=True)
        print("[prepare] Подготовка таблицы книг и item_id маппинга", flush=True)
        books, book_to_item = self._prepare_books(books_raw, language_filter=params.language_filter_enabled)
        print(f"[prepare] Подготовлено книг: {len(books)}", flush=True)

        print(f"[prepare] Чтение interactions из {source.interactions_raw_uri}", flush=True)
        interactions = self._prepare_interactions(
            interactions_uri=source.interactions_raw_uri,
            book_to_item=book_to_item,
            chunksize=params.interactions_chunksize,
        )
        print(f"[prepare] Interactions после merge: {len(interactions)}", flush=True)
        print(f"[prepare] Применение k-core: {params.k_core}", flush=True)
        interactions = self._apply_k_core(interactions, params.k_core)
        print(f"[prepare] После k-core: {len(interactions)}", flush=True)
        print(f"[prepare] Отбор хвоста по времени: {params.keep_recent_fraction}", flush=True)
        interactions = self._keep_recent_tail(interactions, params.keep_recent_fraction)
        print(f"[prepare] После keep_recent_tail: {len(interactions)}", flush=True)

        print("[prepare] Построение train/test split", flush=True)
        train, test, split_ts = self._build_train_test(
            interactions=interactions,
            test_fraction=params.test_fraction,
            warm_users_only=params.warm_users_only,
        )
        print(f"[prepare] Split train/test: {len(train)} / {len(test)}", flush=True)
        print("[prepare] Построение local_train/local_val split", flush=True)
        local_train, local_val, local_val_warm, local_val_cold, local_split_ts = self._build_local_split(
            train=train,
            val_fraction=params.local_val_fraction,
            warm_users_only=params.warm_users_only,
            cold_max_interactions=params.cold_max_interactions,
        )
        local_train = self._add_interaction_weight(local_train)
        print(
            f"[prepare] Local split train/val/warm/cold: {len(local_train)} / {len(local_val)} / "
            f"{len(local_val_warm)} / {len(local_val_cold)}",
            flush=True,
        )

        target = self._work_dir / source.dataset_name
        target.mkdir(parents=True, exist_ok=True)
        print(f"[prepare] Сохранение локальных parquet/json в {target}", flush=True)

        books_path = target / "books.parquet"
        train_path = target / "train.parquet"
        test_path = target / "test.parquet"
        local_train_path = target / "local_train.parquet"
        local_val_path = target / "local_val.parquet"
        local_val_warm_path = target / "local_val_warm.parquet"
        local_val_cold_path = target / "local_val_cold.parquet"
        summary_path = target / "summary.json"
        manifest_path = target / "manifest.json"

        books.to_parquet(books_path, index=False)
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)
        local_train.to_parquet(local_train_path, index=False)
        local_val.to_parquet(local_val_path, index=False)
        local_val_warm.to_parquet(local_val_warm_path, index=False)
        local_val_cold.to_parquet(local_val_cold_path, index=False)

        train_item_counts = train.groupby("item_id").size().to_dict()
        cold_test_items = {
            item_id for item_id in set(test["item_id"].tolist()) if int(train_item_counts.get(item_id, 0)) <= params.cold_max_interactions
        }

        summary = {
            "dataset_name": source.dataset_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "split_ts": str(split_ts),
            "local_split_ts": str(local_split_ts),
            "params": asdict(params),
            "stats": {
                "books_rows": int(len(books)),
                "interactions_rows": int(len(interactions)),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "local_train_rows": int(len(local_train)),
                "local_val_rows": int(len(local_val)),
                "local_val_warm_rows": int(len(local_val_warm)),
                "local_val_cold_rows": int(len(local_val_cold)),
                "train_users": int(train["user_id"].nunique()),
                "test_users": int(test["user_id"].nunique()),
                "train_items": int(train["item_id"].nunique()),
                "test_items": int(test["item_id"].nunique()),
                "cold_test_items": int(len(cold_test_items)),
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest = {
            "status": "SUCCESS",
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
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[prepare] Локальные артефакты успешно сохранены.", flush=True)

        return DatasetArtifacts(
            books_uri=str(books_path),
            train_uri=str(train_path),
            test_uri=str(test_path),
            local_train_uri=str(local_train_path),
            local_val_uri=str(local_val_path),
            local_val_warm_uri=str(local_val_warm_path),
            local_val_cold_uri=str(local_val_cold_path),
            summary_uri=str(summary_path),
            manifest_uri=str(manifest_path),
        )

    def _cached_artifacts(self, dataset_name: str) -> DatasetArtifacts | None:
        """
        If local preprocessed artifacts already exist, reuse them.
        This supports server restarts without repeating heavy preprocessing.
        """
        target = self._work_dir / dataset_name
        required = {
            "books": target / "books.parquet",
            "train": target / "train.parquet",
            "test": target / "test.parquet",
            "local_train": target / "local_train.parquet",
            "local_val": target / "local_val.parquet",
            "local_val_warm": target / "local_val_warm.parquet",
            "local_val_cold": target / "local_val_cold.parquet",
            "summary": target / "summary.json",
            "manifest": target / "manifest.json",
        }
        if not all(path.exists() for path in required.values()):
            return None

        return DatasetArtifacts(
            books_uri=str(required["books"]),
            train_uri=str(required["train"]),
            test_uri=str(required["test"]),
            local_train_uri=str(required["local_train"]),
            local_val_uri=str(required["local_val"]),
            local_val_warm_uri=str(required["local_val_warm"]),
            local_val_cold_uri=str(required["local_val_cold"]),
            summary_uri=str(required["summary"]),
            manifest_uri=str(required["manifest"]),
        )

    @staticmethod
    def _read_books(books_uri: str) -> pd.DataFrame:

        path = Path(books_uri)
        if not path.exists():
            raise FileNotFoundError(f"Books file not found: {path}")
        books = pd.read_json(path, lines=True, compression="gzip")
        required = ["book_id", "title", "description", "popular_shelves", "authors"]
        _require_columns(books, required, "books")
        return books

    def _prepare_books(self, books: pd.DataFrame, language_filter: bool) -> tuple[pd.DataFrame, pd.DataFrame]:

        data = books.copy()
        for col in ["title", "description"]:
            data[col] = data[col].fillna("").astype(str)

        data["tags"] = data["popular_shelves"].apply(_extract_shelf_names)
        data["series"] = data["series"].apply(_to_list) if "series" in data.columns else [[] for _ in range(len(data))]
        data["authors"] = data["authors"].apply(_extract_authors)

        if language_filter:
            if "language_code" in data.columns:
                language_code = data["language_code"].fillna("").astype(str).str.lower()
                data = data[(language_code.str.contains("eng")) | (language_code == "")]

        data["norm_title"] = data["title"].apply(_normalize_title)

        title_map = (
            data[["norm_title"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .assign(item_id=lambda x: x.index.astype(int))
        )
        data = data.merge(title_map, on="norm_title", how="left")

        book_to_item = data[["book_id", "item_id"]].drop_duplicates()

        cols = [
            c
            for c in ["item_id", "series", "tags", "title", "description", "url", "image_url", "authors"]
            if c in data.columns
        ]
        reduced = data[cols].copy()
        reduced["desc_len"] = reduced["description"].fillna("").astype(str).str.len()
        reduced = reduced.sort_values("desc_len", ascending=False)

        rows: list[dict[str, Any]] = []
        for item_id, part in reduced.groupby("item_id", sort=False):
            rows.append(
                {
                    "item_id": int(item_id),
                    "series": _merge_unique_lists(part["series"]) if "series" in part.columns else [],
                    "tags": _merge_unique_lists(part["tags"]) if "tags" in part.columns else [],
                    "title": _first_non_empty(part["title"]) if "title" in part.columns else "",
                    "description": _first_non_empty(part["description"]) if "description" in part.columns else "",
                    "url": _first_non_empty(part["url"]) if "url" in part.columns else "",
                    "image_url": _best_image(part["image_url"]) if "image_url" in part.columns else "",
                    "authors": _merge_unique_lists(part["authors"]) if "authors" in part.columns else [],
                }
            )
        out_books = pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)
        return out_books, book_to_item

    def _prepare_interactions(self, interactions_uri: str,

        book_to_item: pd.DataFrame, chunksize: int) -> pd.DataFrame:

        path = Path(interactions_uri)
        if not path.exists():
            raise FileNotFoundError(f"Interactions file not found: {path}")

        required = ["user_id", "book_id", "is_read", "rating", "date_added"]
        partials: list[pd.DataFrame] = []

        reader = pd.read_json(path, lines=True, compression="gzip", chunksize=chunksize)
        total_rows = 0
        for chunk_idx, chunk in enumerate(reader, start=1):
            total_rows += len(chunk)
            _require_columns(chunk, required, "interactions_chunk")
            chunk = chunk[required].copy()

            merged = chunk.merge(book_to_item, on="book_id", how="inner").drop(columns=["book_id"])
            merged["date_added"] = pd.to_datetime(
                merged["date_added"],
                format="%a %b %d %H:%M:%S %z %Y",
                errors="coerce",
                utc=True,
            )
            merged = merged.dropna(subset=["date_added"]).copy()
            merged["date_added"] = merged["date_added"].dt.tz_localize(None)

            agg = (
                merged.groupby(["user_id", "item_id"], as_index=False)
                .agg(
                    is_read=("is_read", "max"),
                    rating=("rating", "max"),
                    date_added=("date_added", "min"),
                )
            )
            partials.append(agg)
            if chunk_idx % 10 == 0:
                print(
                    f"[prepare] Обработано чанков interactions: {chunk_idx}, "
                    f"сырьевых строк: {total_rows}, агрегированных блоков: {len(partials)}",
                    flush=True,
                )

        if not partials:
            raise ValueError("No interactions after preprocessing")

        all_parts = pd.concat(partials, ignore_index=True)
        out = (
            all_parts.groupby(["user_id", "item_id"], as_index=False)
            .agg(
                is_read=("is_read", "max"),
                rating=("rating", "max"),
                date_added=("date_added", "min"),
            )
            .sort_values(["date_added", "user_id", "item_id"])
            .reset_index(drop=True)
        )
        return out

    @staticmethod
    def _apply_k_core(interactions: pd.DataFrame, k_core: int) -> pd.DataFrame:
        if k_core <= 0:
            return interactions
        user_counts = interactions.groupby("user_id")["item_id"].nunique()
        keep_users = set(user_counts[user_counts > k_core].index.tolist())
        return interactions[interactions["user_id"].isin(keep_users)].copy()

    @staticmethod
    def _keep_recent_tail(interactions: pd.DataFrame, keep_recent_fraction: float) -> pd.DataFrame:
        data = interactions.sort_values("date_added").reset_index(drop=True)
        if keep_recent_fraction >= 1:
            return data
        n_keep = max(1, int(len(data) * keep_recent_fraction))
        return data.tail(n_keep).reset_index(drop=True)

    @staticmethod
    def _build_train_test(interactions: pd.DataFrame, test_fraction: float,
        warm_users_only: bool) -> tuple[pd.DataFrame, pd.DataFrame, Any]:

        split_ts = interactions["date_added"].quantile(1 - test_fraction, interpolation="nearest")
        train = interactions[interactions["date_added"] < split_ts].copy()
        test = interactions[interactions["date_added"] >= split_ts].copy()

        if warm_users_only:
            train_users = set(train["user_id"].tolist())
            test = test[test["user_id"].isin(train_users)].copy()

        test = test.sort_values(["user_id", "rating", "is_read"], ascending=[True, False, False])
        train = train.sort_values(["date_added", "user_id", "item_id"])
        test = test[["user_id", "item_id"]].reset_index(drop=True)
        train = train.reset_index(drop=True)
        return train, test, split_ts

    @staticmethod
    def _build_local_split(train: pd.DataFrame, val_fraction: float,
        warm_users_only: bool,
        cold_max_interactions: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:

        if "date_added" not in train.columns:
            raise ValueError("Train must contain date_added for local split")
        if len(train) == 0:
            raise ValueError("Train is empty")

        split_ts = train["date_added"].quantile(1 - val_fraction, interpolation="nearest")
        local_train = train[train["date_added"] < split_ts].copy()
        local_val = train[train["date_added"] >= split_ts].copy()

        if warm_users_only:
            users = set(local_train["user_id"].tolist())
            local_val = local_val[local_val["user_id"].isin(users)].copy()

        val_items = set(local_val["item_id"].tolist())
        train_item_counts = local_train.groupby("item_id").size().to_dict()
        cold_items = {
            item_id for item_id in val_items if int(train_item_counts.get(item_id, 0)) <= int(cold_max_interactions)
        }
        warm_items = val_items - cold_items

        local_val_warm = local_val[local_val["item_id"].isin(warm_items)].copy().reset_index(drop=True)
        local_val_cold = local_val[local_val["item_id"].isin(cold_items)].copy().reset_index(drop=True)

        if len(local_val_warm) + len(local_val_cold) != len(local_val):
            raise ValueError("Warm/cold local split mismatch")

        local_train = local_train.reset_index(drop=True)
        local_val = local_val.reset_index(drop=True)
        return local_train, local_val, local_val_warm, local_val_cold, split_ts

    @staticmethod
    def _add_interaction_weight(train: pd.DataFrame) -> pd.DataFrame:
        out = train.copy()
        rating = pd.to_numeric(out["rating"], errors="coerce").fillna(0)
        is_read = pd.to_numeric(out["is_read"], errors="coerce").fillna(0)
        out["interaction_weight"] = 1.0 + 0.25 * rating.clip(lower=0) + 0.5 * is_read.clip(lower=0)
        return out


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _to_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _extract_shelf_names(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("name"):
            out.append(str(item["name"]))
    return out


def _extract_authors(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if item.get("name"):
            out.append(str(item["name"]))
        elif item.get("author_id") is not None:
            out.append(str(item["author_id"]))
    return out


def _normalize_title(title: object) -> str:
    text = "" if pd.isna(title) else str(title)
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _merge_unique_lists(series: pd.Series) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for value in series:
        if not isinstance(value, list):
            continue
        for item in value:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _first_non_empty(series: pd.Series) -> str:
    for value in series:
        if pd.isna(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return str(value)
    return ""


def _best_image(series: pd.Series) -> str:
    vals = [value for value in series if not pd.isna(value)]
    for value in vals:
        if "nophoto" not in str(value).lower():
            return str(value)
    if vals:
        return str(vals[0])
    return ""
