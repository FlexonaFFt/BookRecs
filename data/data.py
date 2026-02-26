import gzip
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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


# Форматировать секунды
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


# Форматировать размер
def _fmt_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(0, num_bytes))
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"


# Нормализовать title для дедупликации
def _normalize_title(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Привести поле к списку
def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    return []


# Вытянуть теги из popular_shelves
def _extract_shelf_names(value: object) -> list:
    if not isinstance(value, list):
        return []
    out = []
    for x in value:
        if isinstance(x, dict) and x.get("name"):
            out.append(str(x["name"]))
    return out


# Вытянуть авторов
def _extract_authors(value: object) -> list:
    if not isinstance(value, list):
        return []
    out = []
    for x in value:
        if not isinstance(x, dict):
            continue
        if x.get("name"):
            out.append(str(x["name"]))
        elif x.get("author_id") is not None:
            out.append(str(x["author_id"]))
    return out


# Объединить списки в уникальный список
def _merge_unique_lists(series: pd.Series) -> list:
    seen = set()
    out = []
    for value in series:
        if not isinstance(value, list):
            continue
        for item in value:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


# Первая непустая строка
def _first_non_empty(series: pd.Series) -> object:
    for value in series:
        if pd.isna(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


# Лучшая картинка (не nophoto)
def _best_image(series: pd.Series) -> object:
    values = [v for v in series if not pd.isna(v)]
    for v in values:
        if "nophoto" not in str(v).lower():
            return v
    return values[0] if values else None


# Лог прогресса цикла с ETA
def _log_progress(prefix: str, current: int, total: int, started_at: float, step: int) -> None:
    if total <= 0:
        return
    if current != total and current % step != 0:
        return
    elapsed = time.time() - started_at
    speed = current / elapsed if elapsed > 0 else 0.0
    eta = (total - current) / speed if speed > 0 else 0.0
    logger.info(
        "%s: %s/%s (%.1f%%), прошло=%s, осталось~%s",
        prefix,
        current,
        total,
        current * 100.0 / total,
        _fmt_seconds(elapsed),
        _fmt_seconds(eta),
    )


# Загрузить raw books
def load_raw_books(raw_dir: Union[str, Path]) -> pd.DataFrame:
    raw_dir = _as_path(raw_dir)
    books_path = raw_dir / RAW_BOOKS_FILENAME
    _check_file(books_path)

    logger.info("Чтение raw books: %s", books_path)
    books = pd.read_json(books_path, lines=True, compression="gzip")
    logger.info("books raw shape=%s", books.shape)

    _check_columns(
        books,
        ["book_id", "title", "description", "popular_shelves", "authors"],
        "raw books",
    )
    return books


# Загрузить готовые parquet
def load_processed_goodreads(processed_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = _as_path(processed_dir)
    books_path = processed_dir / PROCESSED_BOOKS_FILENAME
    train_path = processed_dir / PROCESSED_TRAIN_FILENAME
    test_path = processed_dir / PROCESSED_TEST_FILENAME

    _check_file(books_path)
    _check_file(train_path)
    _check_file(test_path)

    logger.info("Чтение parquet books: %s", books_path)
    books = pd.read_parquet(books_path)
    logger.info("Чтение parquet train: %s", train_path)
    train = pd.read_parquet(train_path)
    logger.info("Чтение parquet test: %s", test_path)
    test = pd.read_parquet(test_path)

    _check_columns(books, ["item_id", "title"], "books")
    _check_columns(train, ["user_id", "item_id"], "train")
    _check_columns(test, ["user_id", "item_id"], "test")

    return books, train, test


# Сохранить parquet
def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Сохранение parquet: %s shape=%s", path, df.shape)
    df.to_parquet(path, index=False)


# Сгруппировать ground truth
def group_test_ground_truth(test_exploded: pd.DataFrame) -> pd.DataFrame:
    _check_columns(test_exploded, ["user_id", "item_id"], "test_exploded")
    return test_exploded.groupby("user_id", sort=False)["item_id"].apply(list).reset_index()


# Подготовить books как в ноутбуке (без language detection)
def _prepare_books_like_notebook(books: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    started = time.time()
    logger.info("Этап books: старт")

    books = books.copy()
    books["title"] = books["title"].fillna("")
    books["description"] = books["description"].fillna("")
    books["tags"] = books["popular_shelves"].apply(_extract_shelf_names)
    books["series"] = books["series"].apply(_as_list) if "series" in books.columns else [[] for _ in range(len(books))]
    books["authors"] = books["authors"].apply(_extract_authors)
    books["preprocessed_title"] = books["title"].apply(_normalize_title)

    # Дедуп как в ноутбуке: item_id по уникальному preprocessed_title
    title_map = (
        books[["preprocessed_title"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .assign(item_id=lambda x: range(len(x)))
    )
    books = books.merge(title_map, on="preprocessed_title", how="left")
    book_to_item = books[["book_id", "item_id"]].drop_duplicates()

    keep_cols = [
        c for c in
        ["item_id", "book_id", "series", "tags", "title", "description", "url", "image_url", "authors"]
        if c in books.columns
    ]
    books = books[keep_cols].copy()
    books["description_len"] = books["description"].fillna("").astype(str).str.len()

    # Агрегация дублей item_id
    books = books.sort_values("description_len", ascending=False)
    n_groups = int(books["item_id"].nunique())
    loop_started = time.time()
    step = max(1000, n_groups // 20 or 1)
    rows = []
    for i, (item_id, part) in enumerate(books.groupby("item_id", sort=False), start=1):
        rows.append(
            {
                "item_id": item_id,
                "series": _merge_unique_lists(part["series"]) if "series" in part.columns else [],
                "tags": _merge_unique_lists(part["tags"]) if "tags" in part.columns else [],
                "title": _first_non_empty(part["title"]) if "title" in part.columns else None,
                "description": _first_non_empty(part["description"]) if "description" in part.columns else None,
                "url": _first_non_empty(part["url"]) if "url" in part.columns else None,
                "image_url": _best_image(part["image_url"]) if "image_url" in part.columns else None,
                "authors": _merge_unique_lists(part["authors"]) if "authors" in part.columns else [],
            }
        )
        _log_progress("Агрегация books", i, n_groups, loop_started, step)

    books_processed = pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)
    logger.info("Этап books: готово за %s, shape=%s", _fmt_seconds(time.time() - started), books_processed.shape)
    return books_processed, book_to_item


# Прочитать и агрегировать interactions по чанкам (логика как в ноутбуке)
def _read_interactions_chunked_like_notebook(
    interactions_path: Path,
    book_to_item: pd.DataFrame,
    chunksize: int = 200000,
    max_interaction_chunks: Optional[int] = None,
    reduce_every_chunks: int = 20,
) -> pd.DataFrame:
    if chunksize <= 0:
        raise ValueError("chunksize must be > 0")
    if reduce_every_chunks <= 0:
        raise ValueError("reduce_every_chunks must be > 0")

    file_size = interactions_path.stat().st_size
    usecols = ["user_id", "book_id", "is_read", "rating", "date_added"]
    partials: list[pd.DataFrame] = []
    total_raw_rows = 0
    total_chunk_agg_rows = 0
    started = time.time()

    logger.info(
        "Этап interactions: чтение по чанкам (chunksize=%s, file_size=%s, max_chunks=%s)",
        chunksize,
        _fmt_size(file_size),
        max_interaction_chunks,
    )

    with gzip.open(interactions_path, mode="rt", encoding="utf-8") as gz:
        reader = pd.read_json(gz, lines=True, chunksize=chunksize)
        for chunk_idx, chunk in enumerate(reader, start=1):
            if max_interaction_chunks is not None and chunk_idx > max_interaction_chunks:
                logger.info("Достигнут лимит max_interaction_chunks=%s, останавливаю чтение", max_interaction_chunks)
                break

            chunk_started = time.time()
            _check_columns(chunk, usecols, f"interactions chunk {chunk_idx}")
            chunk = chunk[usecols].copy()
            total_raw_rows += len(chunk)

            chunk = chunk.merge(book_to_item, on="book_id", how="inner")
            chunk = chunk.drop(columns=["book_id"])

            chunk["date_added"] = pd.to_datetime(
                chunk["date_added"],
                format="%a %b %d %H:%M:%S %z %Y",
                errors="coerce",
                utc=True,
            )
            chunk = chunk.dropna(subset=["date_added"]).copy()
            chunk["date_added"] = chunk["date_added"].dt.tz_localize(None)

            chunk_agg = (
                chunk.groupby(["user_id", "item_id"], as_index=False)
                .agg({"is_read": "max", "rating": "max", "date_added": "min"})
            )
            total_chunk_agg_rows += len(chunk_agg)
            partials.append(chunk_agg)

            # Периодически сжимаем промежуточные результаты, чтобы не накапливать память
            if len(partials) >= reduce_every_chunks:
                logger.info("Промежуточная редукция partials (%s кусков)", len(partials))
                merged = pd.concat(partials, ignore_index=True)
                merged = (
                    merged.groupby(["user_id", "item_id"], as_index=False)
                    .agg({"is_read": "max", "rating": "max", "date_added": "min"})
                )
                partials = [merged]
                logger.info("После промежуточной редукции shape=%s", merged.shape)

            compressed_pos = 0
            try:
                compressed_pos = gz.fileobj.tell()  # type: ignore[attr-defined]
            except Exception:
                compressed_pos = 0

            elapsed = time.time() - started
            if compressed_pos > 0 and file_size > 0:
                ratio = min(1.0, compressed_pos / file_size)
                eta = (elapsed / ratio - elapsed) if ratio > 0 else 0.0
                logger.info(
                    "Chunk %s: raw=%s, agg=%s, file=%.1f%%, total_raw=%s, прошло=%s, осталось~%s, chunk_time=%s",
                    chunk_idx,
                    len(chunk),
                    len(chunk_agg),
                    ratio * 100.0,
                    total_raw_rows,
                    _fmt_seconds(elapsed),
                    _fmt_seconds(eta),
                    _fmt_seconds(time.time() - chunk_started),
                )
            else:
                logger.info(
                    "Chunk %s: raw=%s, agg=%s, total_raw=%s, chunk_time=%s",
                    chunk_idx,
                    len(chunk),
                    len(chunk_agg),
                    total_raw_rows,
                    _fmt_seconds(time.time() - chunk_started),
                )

    if not partials:
        raise ValueError("Не удалось прочитать interactions (partials пустой)")

    logger.info(
        "Финальная агрегация interactions: partials=%s, total_raw=%s, total_chunk_agg=%s",
        len(partials),
        total_raw_rows,
        total_chunk_agg_rows,
    )
    merged = pd.concat(partials, ignore_index=True)
    interactions = (
        merged.groupby(["user_id", "item_id"], as_index=False)
        .agg({"is_read": "max", "rating": "max", "date_added": "min"})
    )
    logger.info("Этап interactions: финальный shape=%s", interactions.shape)
    return interactions


# Предобработка raw Goodreads -> books/train/test как в data_splitting.ipynb (без language detection)
def preprocess_goodreads_raw_to_parquet(
    raw_dir: Union[str, Path] = "data/raw_data",
    processed_dir: Union[str, Path] = "data",
    k_core: int = 2,
    keep_recent_fraction: float = 0.6,
    test_fraction: float = 0.25,
    interactions_chunksize: int = 200000,
    max_interaction_chunks: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.time()
    if not 0 < keep_recent_fraction <= 1:
        raise ValueError("keep_recent_fraction must be in (0, 1]")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")

    raw_dir = _as_path(raw_dir)
    processed_dir = _as_path(processed_dir)
    _check_file(raw_dir / RAW_BOOKS_FILENAME)
    _check_file(raw_dir / RAW_INTERACTIONS_FILENAME)

    logger.info(
        "Старт preprocess (notebook-style, no langdetect): raw_dir=%s processed_dir=%s",
        raw_dir,
        processed_dir,
    )
    logger.info(
        "Параметры: k_core=%s, keep_recent_fraction=%s, test_fraction=%s, chunksize=%s, max_interaction_chunks=%s",
        k_core,
        keep_recent_fraction,
        test_fraction,
        interactions_chunksize,
        max_interaction_chunks,
    )

    books_raw = load_raw_books(raw_dir)
    books_processed, book_to_item = _prepare_books_like_notebook(books_raw)

    interactions = _read_interactions_chunked_like_notebook(
        interactions_path=raw_dir / RAW_INTERACTIONS_FILENAME,
        book_to_item=book_to_item,
        chunksize=interactions_chunksize,
        max_interaction_chunks=max_interaction_chunks,
    )

    # k-core фильтр как в ноутбуке: unique_item_count > k_core
    logger.info("Применяю k-core фильтр по пользователям (>%s)", k_core)
    user_item_counts = interactions.groupby("user_id")["item_id"].nunique().rename("unique_item_count")
    warm_users = user_item_counts[user_item_counts > k_core].index
    interactions = interactions[interactions["user_id"].isin(warm_users)].copy()
    logger.info("После k-core interactions=%s, users=%s", interactions.shape, interactions["user_id"].nunique())

    # Оставляем хвост датасета как в ноутбуке (по умолчанию 0.6)
    interactions = interactions.sort_values("date_added").reset_index(drop=True)
    if keep_recent_fraction < 1:
        n_keep = max(1, int(len(interactions) * keep_recent_fraction))
        logger.info("Оставляю хвост interactions: %s из %s (%.1f%%)", n_keep, len(interactions), keep_recent_fraction * 100.0)
        interactions = interactions.tail(n_keep).reset_index(drop=True)

    # Temporal split по квантилю (в ноутбуке ~75% в train / 25% в test)
    split_q = 1 - test_fraction
    split_ts = interactions["date_added"].quantile(split_q, interpolation="nearest")
    logger.info("Split timestamp (q=%.2f): %s", split_q, split_ts)

    train = interactions[interactions["date_added"] < split_ts].copy()
    train_users = set(train["user_id"].tolist())
    test = interactions[
        (interactions["date_added"] >= split_ts) &
        (interactions["user_id"].isin(train_users))
    ].copy()

    # test сортируем по релевантности как в ноутбуке, затем оставляем только user_id/item_id
    test_sort_cols = [c for c in ["user_id", "rating", "is_read"] if c in test.columns]
    if test_sort_cols:
        ascending = [True] + [False] * (len(test_sort_cols) - 1)
        test = test.sort_values(test_sort_cols, ascending=ascending)
    test = test[["user_id", "item_id"]].reset_index(drop=True)

    train_sort_cols = [c for c in ["date_added", "user_id", "item_id"] if c in train.columns]
    if train_sort_cols:
        train = train.sort_values(train_sort_cols).reset_index(drop=True)

    # Санити чеки как в ноутбуке
    all_books = set(books_processed["item_id"].tolist())
    train_books = set(train["item_id"].tolist())
    test_books = set(test["item_id"].tolist())
    logger.info(
        "Sanity: books(train/test/all) = %s / %s / %s; cold_test_items=%s",
        len(train_books),
        len(test_books),
        len(all_books),
        len(test_books - train_books),
    )
    cold_users_in_test = set(test["user_id"].tolist()) - set(train["user_id"].tolist())
    if len(cold_users_in_test) > 0:
        raise ValueError(f"В test попали cold users: {len(cold_users_in_test)}")

    save_parquet(books_processed, processed_dir / PROCESSED_BOOKS_FILENAME)
    save_parquet(train, processed_dir / PROCESSED_TRAIN_FILENAME)
    save_parquet(test, processed_dir / PROCESSED_TEST_FILENAME)

    logger.info("Preprocess завершен за %s", _fmt_seconds(time.time() - started))
    logger.info("Итог: books=%s train=%s test=%s", books_processed.shape, train.shape, test.shape)
    return books_processed, train, test
