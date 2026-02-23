import numpy as np
import pandas as pd

from data.data import group_test_ground_truth


# Добавить веса взаимодействий
def add_interaction_weight(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()

    weight = np.ones(len(train), dtype=float)

    if "is_read" in train.columns:
        weight += (train["is_read"].fillna(0) > 0).astype(float).to_numpy()

    if "rating" in train.columns:
        rating_part = train["rating"].fillna(0)
        rating_part = np.where(rating_part > 0, rating_part / 5.0, 0.0)
        weight += rating_part

    train["interaction_weight"] = weight
    return train


# Собрать истории пользователей
def make_user_histories(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()

    sort_cols = [c for c in ["user_id", "date_added", "item_id"] if c in train.columns]
    if sort_cols:
        train = train.sort_values(sort_cols)

    rows = []
    for user_id, part in train.groupby("user_id", sort=False):
        row = {
            "user_id": user_id,
            "seen_item_ids": part["item_id"].tolist(),
            "n_interactions": int(len(part)),
            "n_unique_items": int(part["item_id"].nunique()),
        }
        if "date_added" in part.columns:
            row["last_interaction_at"] = part["date_added"].max()
        rows.append(row)

    return pd.DataFrame(rows)


# Посчитать популярность айтемов
def make_item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    if "interaction_weight" in train.columns:
        popularity = (
            train.groupby("item_id", as_index=False)
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
            train.groupby("item_id", as_index=False)
            .agg(n_interactions=("item_id", "size"))
            .assign(popularity_weight=lambda x: x["n_interactions"].astype(float))
            .sort_values(
                ["popularity_weight", "n_interactions", "item_id"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )

    return popularity


# Сформировать grouped ground truth для валидации
def make_eval_ground_truth(val_exploded: pd.DataFrame) -> pd.DataFrame:
    return group_test_ground_truth(val_exploded)


# Сэмплировать негативные пары user-item
def make_negative_samples(
    train: pd.DataFrame,
    books: pd.DataFrame | None = None,
    n_neg: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    if n_neg < 0:
        raise ValueError("n_neg must be >= 0")
    if n_neg == 0:
        return pd.DataFrame(columns=["user_id", "item_id"])

    if books is not None and "item_id" in books.columns:
        item_pool = books[["item_id"]].drop_duplicates().reset_index(drop=True)
    else:
        item_pool = train[["item_id"]].drop_duplicates().reset_index(drop=True)

    users = train[["user_id"]].drop_duplicates().reset_index(drop=True)
    positives = train[["user_id", "item_id"]].drop_duplicates()

    unseen = users.merge(item_pool, how="cross").merge(
        positives, on=["user_id", "item_id"], how="left", indicator=True
    )
    unseen = unseen[unseen["_merge"] == "left_only"][["user_id", "item_id"]].copy()

    rng = np.random.default_rng(seed)
    unseen["_rand"] = rng.random(len(unseen))
    unseen = unseen.sort_values(["user_id", "_rand"])
    negatives = unseen.groupby("user_id", sort=False).head(n_neg).drop(columns="_rand")

    return negatives.reset_index(drop=True)


# Собрать обучающие пары с меткой
def make_training_pairs(
    train: pd.DataFrame,
    negatives: pd.DataFrame,
) -> pd.DataFrame:
    pos_cols = [c for c in ["user_id", "item_id", "date_added", "interaction_weight"] if c in train.columns]
    positives = train[pos_cols].copy()
    positives["label"] = 1

    negatives = negatives[["user_id", "item_id"]].copy()
    negatives["label"] = 0

    if "date_added" in positives.columns and "date_added" not in negatives.columns:
        negatives["date_added"] = pd.NaT
    if "interaction_weight" in positives.columns and "interaction_weight" not in negatives.columns:
        negatives["interaction_weight"] = 0.0

    ordered_cols = [c for c in ["user_id", "item_id", "date_added", "interaction_weight", "label"] if c in positives.columns or c in negatives.columns]
    pairs = pd.concat(
        [positives.reindex(columns=ordered_cols), negatives.reindex(columns=ordered_cols)],
        ignore_index=True,
    )

    return pairs.sort_values(["user_id", "label"], ascending=[True, False]).reset_index(drop=True)


# Собрать краткую сводку по датасету
def make_dataset_summary(
    train: pd.DataFrame,
    val: pd.DataFrame | None = None,
    val_warm: pd.DataFrame | None = None,
    val_cold: pd.DataFrame | None = None,
) -> dict:
    summary = {
        "train_rows": int(len(train)),
        "train_users": int(train["user_id"].nunique()),
        "train_items": int(train["item_id"].nunique()),
    }

    if val is not None:
        summary["val_rows"] = int(len(val))
        summary["val_users"] = int(val["user_id"].nunique())
        summary["val_items"] = int(val["item_id"].nunique())

    if val_warm is not None:
        summary["val_warm_rows"] = int(len(val_warm))

    if val_cold is not None:
        summary["val_cold_rows"] = int(len(val_cold))
        if val is not None and len(val) > 0:
            summary["val_cold_ratio"] = float(len(val_cold) / len(val))

    return summary
