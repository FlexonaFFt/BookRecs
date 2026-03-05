from __future__ import annotations

from typing import Any


def load_dataset(pd: Any, dataset_dir: str) -> dict[str, Any]:
    from pathlib import Path

    root = Path(dataset_dir)
    required = {
        "books": root / "books.parquet",
        "local_train": root / "local_train.parquet",
        "local_val": root / "local_val.parquet",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
    return {
        "books": pd.read_parquet(required["books"]),
        "local_train": pd.read_parquet(required["local_train"]),
        "local_val": pd.read_parquet(required["local_val"]),
    }


def build_seen_map(train: Any) -> dict[Any, set[Any]]:
    return train.groupby("user_id", sort=False)["item_id"].agg(lambda x: set(x.tolist())).to_dict()


def cold_items(train: Any, val: Any) -> set[Any]:
    train_items = set(train["item_id"].dropna().tolist())
    val_items = set(val["item_id"].dropna().tolist())
    return val_items - train_items


def build_val_ground_truth(val: Any, limit: int) -> tuple[list[Any], dict[Any, list[Any]]]:
    grouped = val.groupby("user_id", sort=False)["item_id"].agg(list).reset_index()
    users = grouped["user_id"].tolist()[:limit]
    users_set = set(users)
    gt_map = {row.user_id: list(row.item_id) for row in grouped.itertuples(index=False) if row.user_id in users_set}
    return users, gt_map
