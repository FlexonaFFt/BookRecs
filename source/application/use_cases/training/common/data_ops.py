from __future__ import annotations

from typing import Any


def build_item_interaction_counts(train: Any) -> dict[Any, int]:
    counts = train.groupby("item_id", sort=False).size()
    if hasattr(counts, "to_dict"):
        return {item_id: int(count) for item_id, count in counts.to_dict().items()}
    return {item_id: int(count) for item_id, count in counts.items()}


def resolve_cold_item_ids(
    *,
    train: Any,
    candidate_item_ids: set[Any],
    max_train_interactions: int,
) -> set[Any]:
    counts = build_item_interaction_counts(train)
    threshold = max(0, int(max_train_interactions))
    return {
        item_id
        for item_id in candidate_item_ids
        if int(counts.get(item_id, 0)) <= threshold
    }


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
    return (
        train.groupby("user_id", sort=False)["item_id"]
        .agg(lambda x: set(x.tolist()))
        .to_dict()
    )


def cold_items(train: Any, val: Any, max_train_interactions: int = 0) -> set[Any]:
    val_items = set(val["item_id"].dropna().tolist())
    return resolve_cold_item_ids(
        train=train,
        candidate_item_ids=val_items,
        max_train_interactions=max_train_interactions,
    )


def build_val_ground_truth(
    val: Any, limit: int
) -> tuple[list[Any], dict[Any, list[Any]]]:
    grouped = val.groupby("user_id", sort=False)["item_id"].agg(list).reset_index()
    users = grouped["user_id"].tolist()[:limit]
    users_set = set(users)
    gt_map = {
        row.user_id: list(row.item_id)
        for row in grouped.itertuples(index=False)
        if row.user_id in users_set
    }
    return users, gt_map
