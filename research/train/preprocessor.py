import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Проверить обязательные колонки
def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: нет колонок {missing}. Есть: {list(df.columns)}")


# Загрузить parquet с логом
def _load_parquet(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл {name}: {path}")
    logger.info("Чтение %s: %s", name, path)
    df = pd.read_parquet(path)
    logger.info("%s shape=%s", name, df.shape)
    return df


# Собрать текст книги для content-моделей
def build_item_text(books: pd.DataFrame) -> pd.DataFrame:
    books = books.copy()
    _check_columns(books, ["item_id"], "books")

    for col in ["title", "description"]:
        if col not in books.columns:
            books[col] = ""
        books[col] = books[col].fillna("").astype(str)

    for col in ["tags", "authors", "series"]:
        if col not in books.columns:
            books[col] = [[] for _ in range(len(books))]
        books[col] = books[col].apply(lambda x: x if isinstance(x, list) else [])

    books["tags_text"] = books["tags"].apply(lambda x: " ".join(map(str, x)))
    books["authors_text"] = books["authors"].apply(lambda x: " ".join(map(str, x)))
    books["series_text"] = books["series"].apply(lambda x: " ".join(map(str, x)))

    books["item_text"] = (
        books["title"]
        + " "
        + books["authors_text"]
        + " "
        + books["series_text"]
        + " "
        + books["tags_text"]
        + " "
        + books["description"]
    ).str.strip()

    return books[["item_id", "item_text", "title", "tags", "authors", "series"]].copy()


# Посчитать популярность айтемов из train
def build_item_popularity(local_train: pd.DataFrame) -> pd.DataFrame:
    _check_columns(local_train, ["item_id"], "local_train")
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


# Собрать словарь просмотренных айтемов по пользователю
def build_seen_items_by_user(local_train: pd.DataFrame) -> dict:
    _check_columns(local_train, ["user_id", "item_id"], "local_train")
    return (
        local_train.groupby("user_id", sort=False)["item_id"]
        .agg(lambda x: set(x.tolist()))
        .to_dict()
    )


# Отобрать подмножество пользователей валидации для быстрых экспериментов
def sample_eval_users(
    eval_ground_truth: pd.DataFrame,
    sample_users_n: Optional[int] = None,
    seed: int = 42,
) -> list:
    _check_columns(eval_ground_truth, ["user_id"], "eval_ground_truth")
    users = eval_ground_truth["user_id"].drop_duplicates().tolist()
    if sample_users_n is None or sample_users_n <= 0 or sample_users_n >= len(users):
        return users
    sampled = (
        eval_ground_truth[["user_id"]]
        .drop_duplicates()
        .sample(n=sample_users_n, random_state=seed)["user_id"]
        .tolist()
    )
    return sampled


# Подготовить единый набор данных для всех моделей research
def prepare_research_bundle(
    data_dir: Union[str, Path] = "data",
    local_name: str = "local_v1",
    sample_users_n: Optional[int] = None,
    seed: int = 42,
) -> dict:
    data_dir = Path(data_dir)
    local_dir = data_dir / "processed" / local_name

    books = _load_parquet(data_dir / "books.pq", "books")
    local_train = _load_parquet(local_dir / "local_train.pq", "local_train")
    local_val = _load_parquet(local_dir / "local_val.pq", "local_val")
    local_val_warm = _load_parquet(local_dir / "local_val_warm.pq", "local_val_warm")
    local_val_cold = _load_parquet(local_dir / "local_val_cold.pq", "local_val_cold")
    eval_ground_truth = _load_parquet(local_dir / "eval_ground_truth.pq", "eval_ground_truth")

    _check_columns(books, ["item_id"], "books")
    _check_columns(local_train, ["user_id", "item_id"], "local_train")
    _check_columns(local_val, ["user_id", "item_id"], "local_val")
    _check_columns(local_val_warm, ["user_id", "item_id"], "local_val_warm")
    _check_columns(local_val_cold, ["user_id", "item_id"], "local_val_cold")
    _check_columns(eval_ground_truth, ["user_id", "item_id"], "eval_ground_truth")

    # Нормализуем datetime
    if "date_added" in local_train.columns and not pd.api.types.is_datetime64_any_dtype(local_train["date_added"]):
        local_train["date_added"] = pd.to_datetime(local_train["date_added"], errors="coerce")

    eval_users = sample_eval_users(eval_ground_truth, sample_users_n=sample_users_n, seed=seed)
    eval_user_set = set(eval_users)

    # Фильтруем все eval-срезы одними и теми же пользователями
    eval_ground_truth = eval_ground_truth[eval_ground_truth["user_id"].isin(eval_user_set)].reset_index(drop=True)
    local_val = local_val[local_val["user_id"].isin(eval_user_set)].reset_index(drop=True)
    local_val_warm = local_val_warm[local_val_warm["user_id"].isin(eval_user_set)].reset_index(drop=True)
    local_val_cold = local_val_cold[local_val_cold["user_id"].isin(eval_user_set)].reset_index(drop=True)

    item_text = build_item_text(books)
    item_popularity = build_item_popularity(local_train)
    seen_items_by_user = build_seen_items_by_user(local_train)

    warm_item_ids = set(local_val_warm["item_id"].dropna().tolist())
    cold_item_ids = set(local_val_cold["item_id"].dropna().tolist())
    candidate_item_ids = books["item_id"].drop_duplicates().tolist()

    logger.info(
        "Research bundle готов: users=%s, train=%s, val=%s, warm_items=%s, cold_items=%s",
        len(eval_users),
        local_train.shape,
        local_val.shape,
        len(warm_item_ids),
        len(cold_item_ids),
    )

    return {
        "books": books,
        "item_text": item_text,
        "local_train": local_train,
        "local_val": local_val,
        "local_val_warm": local_val_warm,
        "local_val_cold": local_val_cold,
        "eval_ground_truth": eval_ground_truth,
        "eval_users": eval_users,
        "seen_items_by_user": seen_items_by_user,
        "item_popularity": item_popularity,
        "candidate_item_ids": candidate_item_ids,
        "warm_item_ids": warm_item_ids,
        "cold_item_ids": cold_item_ids,
        "catalog_size": int(books["item_id"].nunique()),
    }
