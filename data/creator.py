import json
import logging
import time
import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from data import (
        PROCESSED_BOOKS_FILENAME,
        PROCESSED_TEST_FILENAME,
        PROCESSED_TRAIN_FILENAME,
        RAW_BOOKS_FILENAME,
        RAW_INTERACTIONS_FILENAME,
        group_test_ground_truth,
        load_processed_goodreads,
        preprocess_goodreads_raw_to_parquet,
        save_parquet,
    )
    from splitter import make_local_validation_split
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.data import (  # type: ignore
        PROCESSED_BOOKS_FILENAME,
        PROCESSED_TEST_FILENAME,
        PROCESSED_TRAIN_FILENAME,
        RAW_BOOKS_FILENAME,
        RAW_INTERACTIONS_FILENAME,
        group_test_ground_truth,
        load_processed_goodreads,
        preprocess_goodreads_raw_to_parquet,
        save_parquet,
    )
    from data.splitter import make_local_validation_split  # type: ignore


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Отформатировать секунды
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


# Лог прогресса с ETA
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


# Проверить наличие prepared parquet
def _has_processed_goodreads(data_dir: Path) -> bool:
    return (
        (data_dir / PROCESSED_BOOKS_FILENAME).exists()
        and (data_dir / PROCESSED_TRAIN_FILENAME).exists()
        and (data_dir / PROCESSED_TEST_FILENAME).exists()
    )


# Проверить наличие raw Goodreads файлов
def _check_raw_files(raw_dir: Path) -> None:
    books_path = raw_dir / RAW_BOOKS_FILENAME
    interactions_path = raw_dir / RAW_INTERACTIONS_FILENAME
    if not books_path.exists() or not interactions_path.exists():
        raise FileNotFoundError(
            "Не найдены raw файлы Goodreads. "
            f"Ожидаются: {books_path} и {interactions_path}. "
            "Скачайте их вручную через curl и запустите скрипт снова."
        )


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
    n_users = int(train["user_id"].nunique())
    started = time.time()
    step = max(1000, n_users // 20 or 1)
    for i, (user_id, part) in enumerate(train.groupby("user_id", sort=False), start=1):
        row = {
            "user_id": user_id,
            "seen_item_ids": part["item_id"].tolist(),
            "n_interactions": int(len(part)),
            "n_unique_items": int(part["item_id"].nunique()),
        }
        if "date_added" in part.columns:
            row["last_interaction_at"] = part["date_added"].max()
        rows.append(row)
        _log_progress("Сбор user_histories", i, n_users, started, step)

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
    books: Optional[pd.DataFrame] = None,
    n_neg: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    started = time.time()
    logger.info("Старт negative sampling (n_neg=%s, seed=%s)", n_neg, seed)

    if n_neg < 0:
        raise ValueError("n_neg must be >= 0")
    if n_neg == 0:
        return pd.DataFrame(columns=["user_id", "item_id"])

    if books is not None and "item_id" in books.columns:
        item_pool = books["item_id"].drop_duplicates().to_numpy()
    else:
        item_pool = train["item_id"].drop_duplicates().to_numpy()

    if len(item_pool) == 0:
        return pd.DataFrame(columns=["user_id", "item_id"])

    user_seen = (
        train.groupby("user_id", sort=False)["item_id"]
        .agg(lambda x: set(x.tolist()))
    )
    logger.info(
        "Negative sampling вход: users=%s, items=%s",
        len(user_seen),
        len(item_pool),
    )

    rng = np.random.default_rng(seed)
    rows = []
    total_users = len(user_seen)
    step = max(1000, total_users // 20 or 1)
    loop_started = time.time()

    for i, (user_id, seen_items) in enumerate(user_seen.items(), start=1):
        max_available = len(item_pool) - len(seen_items)
        k = min(n_neg, max_available)
        if k <= 0:
            _log_progress("Negative sampling", i, total_users, loop_started, step)
            continue

        chosen = []
        chosen_set = set()

        # Быстрый режим: у большинства пользователей seen_items мало, поэтому rejection sampling работает быстро
        target_batch = max(64, k * 4)
        attempts = 0
        max_attempts = max(200, k * 50)
        while len(chosen) < k and attempts < max_attempts:
            idx = rng.integers(0, len(item_pool), size=target_batch)
            for item in item_pool[idx]:
                if item in seen_items or item in chosen_set:
                    continue
                chosen.append(item)
                chosen_set.add(item)
                if len(chosen) == k:
                    break
            attempts += target_batch

        # Фолбэк: если rejection sampling не добрал нужное количество
        if len(chosen) < k:
            rest = [item for item in item_pool if item not in seen_items and item not in chosen_set]
            if rest:
                need = k - len(chosen)
                if len(rest) <= need:
                    chosen.extend(rest)
                else:
                    extra_idx = rng.choice(len(rest), size=need, replace=False)
                    chosen.extend([rest[j] for j in extra_idx])

        if chosen:
            rows.extend((user_id, item_id) for item_id in chosen[:k])

        _log_progress("Negative sampling", i, total_users, loop_started, step)

    negatives = pd.DataFrame(rows, columns=["user_id", "item_id"])

    logger.info(
        "Negative sampling завершен за %s, negatives=%s",
        _fmt_seconds(time.time() - started),
        negatives.shape,
    )
    return negatives


# Собрать обучающие пары с меткой
def make_training_pairs(train: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    pos_cols = [c for c in ["user_id", "item_id", "date_added", "interaction_weight"] if c in train.columns]
    positives = train[pos_cols].copy()
    positives["label"] = 1

    negatives = negatives[["user_id", "item_id"]].copy()
    negatives["label"] = 0

    if "date_added" in positives.columns and "date_added" not in negatives.columns:
        negatives["date_added"] = pd.NaT
    if "interaction_weight" in positives.columns and "interaction_weight" not in negatives.columns:
        negatives["interaction_weight"] = 0.0

    ordered_cols = [
        c
        for c in ["user_id", "item_id", "date_added", "interaction_weight", "label"]
        if c in positives.columns or c in negatives.columns
    ]
    pairs = pd.concat(
        [positives.reindex(columns=ordered_cols), negatives.reindex(columns=ordered_cols)],
        ignore_index=True,
    )
    return pairs.sort_values(["user_id", "label"], ascending=[True, False]).reset_index(drop=True)


# Собрать краткую сводку по датасету
def make_dataset_summary(
    train: pd.DataFrame,
    val: Optional[pd.DataFrame] = None,
    val_warm: Optional[pd.DataFrame] = None,
    val_cold: Optional[pd.DataFrame] = None,
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


# Собрать и сохранить локальный ML-датасет
def run_dataset_build(
    data_dir: Union[str, Path] = "data",
    output_name: str = "local_v1",
    raw_dir: Union[str, Path] = "data/raw_data",
    rebuild_processed: bool = False,
    k_core: int = 2,
    keep_recent_fraction: float = 0.6,
    test_fraction: float = 0.25,
    val_fraction: float = 0.2,
    warm_users_only: bool = True,
    n_neg: int = 20,
    seed: int = 42,
) -> None:
    started_total = time.time()
    root = Path(data_dir)
    raw_dir = Path(raw_dir)
    out_dir = root / "processed" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Старт сборки датасета (data_dir=%s, output=%s)", root, output_name)

    if rebuild_processed or not _has_processed_goodreads(root):
        _check_raw_files(raw_dir)
        logger.info("Запускаю предобработку raw -> parquet")
        preprocess_goodreads_raw_to_parquet(
            raw_dir=raw_dir,
            processed_dir=root,
            k_core=k_core,
            keep_recent_fraction=keep_recent_fraction,
            test_fraction=test_fraction,
        )
    else:
        logger.info("Использую существующие parquet в %s", root)

    books, train, _test = load_processed_goodreads(root)

    if "date_added" in train.columns and not pd.api.types.is_datetime64_any_dtype(train["date_added"]):
        logger.info("Преобразую train.date_added в datetime")
        train["date_added"] = pd.to_datetime(train["date_added"], errors="coerce")

    logger.info("Локальный split (val_fraction=%s, warm_users_only=%s)", val_fraction, warm_users_only)
    split = make_local_validation_split(
        train,
        val_fraction=val_fraction,
        warm_users_only=warm_users_only,
    )
    logger.info(
        "Split готов: train=%s, val=%s, warm=%s, cold=%s",
        split["train"].shape,
        split["val"].shape,
        split["val_warm"].shape,
        split["val_cold"].shape,
    )

    logger.info("Собираю таблицы для обучения")
    local_train = add_interaction_weight(split["train"])
    val = split["val"]
    val_warm = split["val_warm"]
    val_cold = split["val_cold"]

    user_histories = make_user_histories(local_train)
    logger.info("user_histories=%s", user_histories.shape)
    item_popularity = make_item_popularity(local_train)
    logger.info("item_popularity=%s", item_popularity.shape)
    eval_ground_truth = make_eval_ground_truth(val)
    logger.info("eval_ground_truth=%s", eval_ground_truth.shape)
    negatives = make_negative_samples(local_train, books=books, n_neg=n_neg, seed=seed)
    train_pairs = make_training_pairs(local_train, negatives)
    logger.info("train_pairs=%s", train_pairs.shape)

    summary = make_dataset_summary(
        train=local_train,
        val=val,
        val_warm=val_warm,
        val_cold=val_cold,
    )
    summary["split_ts"] = str(split["split_ts"])
    summary["n_neg"] = int(n_neg)
    summary["seed"] = int(seed)
    summary["raw_dir"] = str(raw_dir)
    summary["k_core"] = int(k_core)
    summary["keep_recent_fraction"] = float(keep_recent_fraction)
    summary["test_fraction"] = float(test_fraction)

    logger.info("Сохраняю артефакты в %s", out_dir)
    save_parquet(local_train, out_dir / "local_train.pq")
    save_parquet(val, out_dir / "local_val.pq")
    save_parquet(val_warm, out_dir / "local_val_warm.pq")
    save_parquet(val_cold, out_dir / "local_val_cold.pq")
    save_parquet(user_histories, out_dir / "user_histories.pq")
    save_parquet(item_popularity, out_dir / "item_popularity.pq")
    save_parquet(eval_ground_truth, out_dir / "eval_ground_truth.pq")
    save_parquet(negatives, out_dir / "negatives.pq")
    save_parquet(train_pairs, out_dir / "train_pairs.pq")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Сборка датасета завершена за %s", _fmt_seconds(time.time() - started_total))
    print("Готово")
    print(f"Папка: {out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# CLI для удобного запуска
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сборка ML-датасета для RecSys")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--raw-dir", default="data/raw_data")
    parser.add_argument("--output-name", default="local_v1")
    parser.add_argument("--rebuild-processed", action="store_true")
    parser.add_argument("--k-core", type=int, default=2)
    parser.add_argument("--keep-recent-fraction", type=float, default=0.6)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--warm-users-only", action="store_true", default=True)
    parser.add_argument("--all-val-users", action="store_true")
    parser.add_argument("--n-neg", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_dataset_build(
        data_dir=args.data_dir,
        output_name=args.output_name,
        raw_dir=args.raw_dir,
        rebuild_processed=args.rebuild_processed,
        k_core=args.k_core,
        keep_recent_fraction=args.keep_recent_fraction,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        warm_users_only=not args.all_val_users,
        n_neg=args.n_neg,
        seed=args.seed,
    )
