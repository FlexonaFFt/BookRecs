import logging
import re
import time
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd


RAW_BOOKS_FILENAME = "books.json.gz"
RAW_INTERACTIONS_FILENAME = "interactions.json.gz"

PROCESSED_BOOKS_FILENAME = "books.pq"
PROCESSED_TRAIN_FILENAME = "train.pq"
PROCESSED_TEST_FILENAME = "test.pq"

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Привести путь к Path
def _as_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


# Проверить, что файл существует
def _check_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")


# Проверить обязательные колонки
def _check_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Got: {list(df.columns)}")


# Нормализовать заголовок для дедупликации
def _normalize_title(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Безопасно вытащить список тегов из Goodreads shelves
def _extract_shelf_names(value: object) -> list:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                result.append(str(name))
    return result


# Безопасно привести поле к списку
def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    return []


# Безопасно вытащить авторов
def _extract_authors(value: object) -> list:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("name")
            author_id = item.get("author_id")
            if name:
                result.append(str(name))
            elif author_id is not None:
                result.append(str(author_id))
    return result


# Объединить список списков в уникальный список
def _merge_unique_lists(series: pd.Series) -> list:
    seen = set()
    out = []
    for value in series:
        if not isinstance(value, list):
            continue
        for x in value:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


# Взять первую непустую строку
def _first_non_empty(series: pd.Series) -> object:
    for value in series:
        if pd.isna(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


# Взять картинку, избегая nophoto если возможно
def _best_image(series: pd.Series) -> object:
    values = [v for v in series if not pd.isna(v)]
    for v in values:
        if "nophoto" not in str(v).lower():
            return v
    return values[0] if values else None


# Отформатировать секунды в человекочитаемый вид
def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}ч {m}м {s}с"
    if m > 0:
        return f"{m}м {s}с"
    return f"{s}с"


# Логировать прогресс цикла с оценкой времени
def _log_progress(prefix: str, current: int, total: int, started_at: float, step: int = 10000) -> None:
    if total <= 0:
        return
    if current != total and current % step != 0:
        return
    elapsed = time.time() - started_at
    rate = current / elapsed if elapsed > 0 else 0.0
    eta = (total - current) / rate if rate > 0 else 0.0
    logger.info(
        "%s: %s/%s (%.1f%%), прошло=%s, осталось~%s",
        prefix,
        current,
        total,
        current * 100.0 / total,
        _fmt_seconds(elapsed),
        _fmt_seconds(eta),
    )


# Загрузить сырые данные Goodreads
def load_raw_goodreads(raw_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = _as_path(raw_dir)
    books_path = raw_dir / RAW_BOOKS_FILENAME
    interactions_path = raw_dir / RAW_INTERACTIONS_FILENAME

    _check_file(books_path)
    _check_file(interactions_path)

    logger.info("Чтение raw books: %s", books_path)
    books = pd.read_json(books_path, lines=True, compression="gzip")
    logger.info("Чтение raw interactions: %s", interactions_path)
    interactions = pd.read_json(interactions_path, lines=True, compression="gzip")
    logger.info("Raw загружены: books=%s, interactions=%s", books.shape, interactions.shape)

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


# Загрузить подготовленные parquet-таблицы
def load_processed_goodreads(
    processed_dir: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = _as_path(processed_dir)
    books_path = processed_dir / PROCESSED_BOOKS_FILENAME
    train_path = processed_dir / PROCESSED_TRAIN_FILENAME
    test_path = processed_dir / PROCESSED_TEST_FILENAME

    _check_file(books_path)
    _check_file(train_path)
    _check_file(test_path)

    logger.info("Чтение parquet: %s", books_path)
    books = pd.read_parquet(books_path)
    logger.info("Чтение parquet: %s", train_path)
    train = pd.read_parquet(train_path)
    logger.info("Чтение parquet: %s", test_path)
    test = pd.read_parquet(test_path)

    _check_columns(books, ["item_id", "title"], "books")
    _check_columns(train, ["user_id", "item_id"], "train")
    _check_columns(test, ["user_id", "item_id"], "test")

    return books, train, test


# Сохранить таблицу в parquet
def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Сохранение parquet: %s shape=%s", path, df.shape)
    df.to_parquet(path, index=False)


# Сгруппировать ground truth по пользователю
def group_test_ground_truth(test_exploded: pd.DataFrame) -> pd.DataFrame:
    _check_columns(test_exploded, ["user_id", "item_id"], "test_exploded")
    return test_exploded.groupby("user_id", sort=False)["item_id"].apply(list).reset_index()


# Предобработать raw Goodreads и сохранить books/train/test
def preprocess_goodreads_raw_to_parquet(
    raw_dir: Union[str, Path] = "data/raw_data",
    processed_dir: Union[str, Path] = "data",
    k_core: int = 2,
    keep_recent_fraction: float = 0.6,
    test_fraction: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started_total = time.time()
    if not 0 < keep_recent_fraction <= 1:
        raise ValueError("keep_recent_fraction must be in (0, 1]")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")

    logger.info(
        "Старт предобработки raw -> parquet (raw_dir=%s, processed_dir=%s)",
        raw_dir,
        processed_dir,
    )
    logger.info(
        "Параметры: k_core=%s, keep_recent_fraction=%s, test_fraction=%s",
        k_core,
        keep_recent_fraction,
        test_fraction,
    )
    books, interactions = load_raw_goodreads(raw_dir)

    stage_started = time.time()
    logger.info("Этап: подготовка книг")
    books = books.copy()
    books["title"] = books["title"].fillna("")
    books["description"] = books["description"].fillna("")
    books["tags"] = books["popular_shelves"].apply(_extract_shelf_names)
    books["series"] = books["series"].apply(_as_list) if "series" in books.columns else [[] for _ in range(len(books))]
    books["authors"] = books["authors"].apply(_extract_authors)
    books["preprocessed_title"] = books["title"].apply(_normalize_title)
    books["description_len"] = books["description"].astype(str).str.len()

    title_map = (
        books[["preprocessed_title"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(item_id=lambda x: range(len(x)))
    )
    books = books.merge(title_map, on="preprocessed_title", how="left")

    book_to_item = books[["book_id", "item_id"]].drop_duplicates()

    keep_book_cols = [c for c in ["item_id", "book_id", "series", "tags", "title", "description", "url", "image_url", "authors", "description_len"] if c in books.columns]
    books = books[keep_book_cols].copy()
    books = books.sort_values("description_len", ascending=False)

    grouped_rows = []
    grouped = books.groupby("item_id", sort=False)
    n_groups = int(books["item_id"].nunique())
    loop_started = time.time()
    for i, (item_id, part) in enumerate(grouped, start=1):
        grouped_rows.append(
            {
                "item_id": item_id,
                "title": _first_non_empty(part["title"]) if "title" in part.columns else None,
                "description": _first_non_empty(part["description"]) if "description" in part.columns else None,
                "url": _first_non_empty(part["url"]) if "url" in part.columns else None,
                "image_url": _best_image(part["image_url"]) if "image_url" in part.columns else None,
                "series": _merge_unique_lists(part["series"]) if "series" in part.columns else [],
                "tags": _merge_unique_lists(part["tags"]) if "tags" in part.columns else [],
                "authors": _merge_unique_lists(part["authors"]) if "authors" in part.columns else [],
            }
        )
        _log_progress("Агрегация книг", i, n_groups, loop_started, step=max(1000, n_groups // 20 or 1))
    books_processed = pd.DataFrame(grouped_rows).sort_values("item_id").reset_index(drop=True)
    logger.info(
        "Этап завершен: подготовка книг за %s, books_processed=%s",
        _fmt_seconds(time.time() - stage_started),
        books_processed.shape,
    )

    stage_started = time.time()
    logger.info("Этап: подготовка взаимодействий")
    interactions = interactions.merge(book_to_item, on="book_id", how="inner")
    interactions = interactions.drop(columns=["book_id"])

    interactions["date_added"] = pd.to_datetime(
        interactions["date_added"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
        utc=True,
    )
    interactions = interactions.dropna(subset=["date_added"]).copy()
    interactions["date_added"] = interactions["date_added"].dt.tz_localize(None)

    agg_dict = {"date_added": "min"}
    if "is_read" in interactions.columns:
        agg_dict["is_read"] = "max"
    if "rating" in interactions.columns:
        agg_dict["rating"] = "max"

    logger.info("Агрегация interactions по (user_id, item_id)")
    interactions = interactions.groupby(["user_id", "item_id"], as_index=False).agg(agg_dict).copy()
    logger.info("После агрегации interactions=%s", interactions.shape)

    user_item_counts = interactions.groupby("user_id")["item_id"].nunique().rename("unique_item_count")
    warm_users = user_item_counts[user_item_counts > k_core].index
    interactions = interactions[interactions["user_id"].isin(warm_users)].copy()

    interactions = interactions.sort_values("date_added").reset_index(drop=True)
    if keep_recent_fraction < 1:
        n_keep = max(1, int(len(interactions) * keep_recent_fraction))
        logger.info(
            "Оставляем хвост interactions: %s из %s (%.1f%%)",
            n_keep,
            len(interactions),
            keep_recent_fraction * 100,
        )
        interactions = interactions.tail(n_keep).reset_index(drop=True)

    q = 1 - test_fraction
    split_ts = interactions["date_added"].quantile(q, interpolation="nearest")

    train = interactions[interactions["date_added"] < split_ts].copy()
    train_users = set(train["user_id"].tolist())
    test = interactions[
        (interactions["date_added"] >= split_ts) & (interactions["user_id"].isin(train_users))
    ].copy()

    test_sort_cols = [c for c in ["user_id", "rating", "is_read"] if c in test.columns]
    if test_sort_cols:
        ascending = [True] + [False] * (len(test_sort_cols) - 1)
        test = test.sort_values(test_sort_cols, ascending=ascending)
    test = test[["user_id", "item_id"]].reset_index(drop=True)

    train_sort_cols = [c for c in ["date_added", "user_id", "item_id"] if c in train.columns]
    if train_sort_cols:
        train = train.sort_values(train_sort_cols).reset_index(drop=True)

    logger.info(
        "Split готов: train=%s, test=%s, split_ts=%s",
        train.shape,
        test.shape,
        split_ts,
    )

    processed_dir = _as_path(processed_dir)
    save_parquet(books_processed, processed_dir / PROCESSED_BOOKS_FILENAME)
    save_parquet(train, processed_dir / PROCESSED_TRAIN_FILENAME)
    save_parquet(test, processed_dir / PROCESSED_TEST_FILENAME)

    logger.info("Сохранены подготовленные данные в %s", processed_dir)
    logger.info(
        "Предобработка завершена за %s",
        _fmt_seconds(time.time() - started_total),
    )

    return books_processed, train, test
