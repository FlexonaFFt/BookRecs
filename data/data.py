from pathlib import Path

import pandas as pd


RAW_BOOKS_FILENAME = "books.json.gz"
RAW_INTERACTIONS_FILENAME = "interactions.json.gz"

PROCESSED_BOOKS_FILENAME = "books.pq"
PROCESSED_TRAIN_FILENAME = "train.pq"
PROCESSED_TEST_FILENAME = "test.pq"


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _check_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")


# Проверяем обязательные колоночки
def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Got: {df.columns}")


# Загрузим сырые данные Goodreads
def load_raw_goodreads(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = _as_path(raw_dir)
    books_path = raw_dir / RAW_BOOKS_FILENAME
    interactions_path = raw_dir / RAW_INTERACTIONS_FILENAME

    _check_file(books_path)
    _check_file(interactions_path)

    books = pd.read_json(books_path, lines=True, compression="gzip")
    interactions = pd.read_json(interactions_path, lines=True, compression="gzip")

    _check_columns(
        books,
        ["book_id", "title", "description", "popular_shelves", "authors", "language_code"],
        "raw books",
    )
    _check_columns(
        interactions,
        ["user_id", "book_id", "is_read", "rating", "date_added"],
        "raw interactions",
    )

    return books, interactions


# Загрузим подготовленные parquet-таблицы
def load_processed_goodreads(processed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = _as_path(processed_dir)
    books_path = processed_dir / PROCESSED_BOOKS_FILENAME
    train_path = processed_dir / PROCESSED_TRAIN_FILENAME
    test_path = processed_dir / PROCESSED_TEST_FILENAME

    _check_file(books_path)
    _check_file(train_path)
    _check_file(test_path)

    books = pd.read_parquet(books_path)
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    _check_columns(books, ["item_id", "title"], "books")
    _check_columns(train, ["user_id", "item_id"], "train")
    _check_columns(test, ["user_id", "item_id"], "test")

    return books, train, test


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def group_test_ground_truth(test_exploded: pd.DataFrame) -> pd.DataFrame:
    _check_columns(test_exploded, ["user_id", "item_id"], "test_exploded")
    return (
        test_exploded.groupby("user_id", sort=False)["item_id"]
        .apply(list)
        .reset_index()
    )
