from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass
class DataBundle:
    books: pd.DataFrame
    local_train: pd.DataFrame
    local_val: pd.DataFrame
    local_val_warm: pd.DataFrame
    local_val_cold: pd.DataFrame
    eval_ground_truth: pd.DataFrame
    item_text: pd.DataFrame
    item_popularity: pd.DataFrame
    seen_items_by_user: dict[Any, set[Any]]
    eval_users: list[Any]
    warm_item_ids: set[Any]
    cold_item_ids: set[Any]
    candidate_item_ids: list[Any]
    catalog_size: int
    summary: dict[str, Any]
    data_root: str
    split_name: str


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; got {list(df.columns)}")


def _read_parquet(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_parquet(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return pd.read_json(path, typ="series").to_dict()


def build_item_text(books: pd.DataFrame) -> pd.DataFrame:
    _require_columns(books, ["item_id"], "books")

    data = books.copy()

    for col in ["title", "description"]:
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].fillna("").astype(str)

    for col in ["tags", "authors", "series"]:
        if col not in data.columns:
            data[col] = [[] for _ in range(len(data))]
        data[col] = data[col].apply(lambda x: x if isinstance(x, list) else [])

    data["tags_text"] = data["tags"].apply(lambda x: " ".join(map(str, x)))
    data["authors_text"] = data["authors"].apply(lambda x: " ".join(map(str, x)))
    data["series_text"] = data["series"].apply(lambda x: " ".join(map(str, x)))
    data["item_text"] = (
        data["title"] + " " + data["authors_text"]
        + " " + data["series_text"] + " " + data["tags_text"]
        + " " + data["description"]).str.strip()

    return data[["item_id", "item_text", "title", "tags", "authors", "series"]].copy()


def build_item_popularity(local_train: pd.DataFrame) -> pd.DataFrame:
    _require_columns(local_train, ["item_id"], "local_train")

    if "interaction_weight" in local_train.columns:
        popularity = (
            local_train.groupby("item_id", as_index=False)
            .agg(
                n_interactions=("item_id", "size"),
                popularity_weight=("interaction_weight", "sum"),
            )
            .sort_values(
                ["popularity_weight", "n_interactions", "item_id"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
    else:
        popularity = (
            local_train.groupby("item_id", as_index=False)
            .agg(n_interactions=("item_id", "size"))
            .assign(popularity_weight=lambda x: x["n_interactions"].astype(float))
            .sort_values(
                ["popularity_weight", "n_interactions", "item_id"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )

    return popularity


def build_seen_items_by_user(local_train: pd.DataFrame) -> dict[Any, set[Any]]:
    _require_columns(local_train, ["user_id", "item_id"], "local_train")
    return (
        local_train.groupby("user_id", sort=False)["item_id"]
        .agg(lambda x: set(x.tolist()))
        .to_dict()
    )


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_eval_users(eval_ground_truth: pd.DataFrame, sample_users_n: Optional[int], seed: int) -> list[Any]:
    users = eval_ground_truth["user_id"].drop_duplicates()
    if sample_users_n is None or sample_users_n <= 0 or sample_users_n >= len(users):
        return users.tolist()
    return users.sample(n=sample_users_n, random_state=seed).tolist()


def load_data_bundle(
    data_root: str = "data/data06", split_name: str = "local_v1",
    sample_users_n: Optional[int] = None, seed: int = 42) -> DataBundle:
    root = Path(data_root)
    split_dir = root / split_name

    books = _read_parquet(root / "books.pq", "books")
    local_train = _read_parquet(split_dir / "local_train.pq", "local_train")
    local_val = _read_parquet(split_dir / "local_val.pq", "local_val")
    local_val_warm = _read_parquet(split_dir / "local_val_warm.pq", "local_val_warm")
    local_val_cold = _read_parquet(split_dir / "local_val_cold.pq", "local_val_cold")
    eval_ground_truth = _read_parquet(split_dir / "eval_ground_truth.pq", "eval_ground_truth")

    _require_columns(books, ["item_id"], "books")
    _require_columns(local_train, ["user_id", "item_id"], "local_train")
    _require_columns(local_val, ["user_id", "item_id"], "local_val")
    _require_columns(local_val_warm, ["user_id", "item_id"], "local_val_warm")
    _require_columns(local_val_cold, ["user_id", "item_id"], "local_val_cold")
    _require_columns(eval_ground_truth, ["user_id", "item_id"], "eval_ground_truth")

    eval_users = _filter_eval_users(eval_ground_truth, sample_users_n=sample_users_n, seed=seed)
    eval_user_set = set(eval_users)
    eval_ground_truth = eval_ground_truth[eval_ground_truth["user_id"].isin(eval_user_set)].reset_index(drop=True)
    local_val = local_val[local_val["user_id"].isin(eval_user_set)].reset_index(drop=True)
    local_val_warm = local_val_warm[local_val_warm["user_id"].isin(eval_user_set)].reset_index(drop=True)
    local_val_cold = local_val_cold[local_val_cold["user_id"].isin(eval_user_set)].reset_index(drop=True)

    item_text = build_item_text(books)
    item_popularity_path = split_dir / "item_popularity.pq"
    if item_popularity_path.exists():
        item_popularity = _read_parquet(item_popularity_path, "item_popularity")
    else:
        item_popularity = build_item_popularity(local_train)

    seen_items_by_user = build_seen_items_by_user(local_train)
    warm_item_ids = set(local_val_warm["item_id"].dropna().tolist())
    cold_item_ids = set(local_val_cold["item_id"].dropna().tolist())
    candidate_item_ids = books["item_id"].drop_duplicates().tolist()
    summary = _load_summary(split_dir / "summary.json")

    return DataBundle(
        books=books,
        local_train=local_train,
        local_val=local_val,
        local_val_warm=local_val_warm,
        local_val_cold=local_val_cold,
        eval_ground_truth=eval_ground_truth,
        item_text=item_text,
        item_popularity=item_popularity,
        seen_items_by_user=seen_items_by_user,
        eval_users=eval_users,
        warm_item_ids=warm_item_ids,
        cold_item_ids=cold_item_ids,
        candidate_item_ids=candidate_item_ids,
        catalog_size=int(books["item_id"].nunique()),
        summary=summary,
        data_root=str(root),
        split_name=split_name,
    )