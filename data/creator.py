import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from data import (
        PROCESSED_BOOKS_FILENAME,
        PROCESSED_TEST_FILENAME,
        PROCESSED_TRAIN_FILENAME,
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


# Формат времени
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


# Проверить наличие processed parquet
def _processed_exists(data_dir: Path) -> bool:
    return all(
        (data_dir / name).exists()
        for name in [PROCESSED_BOOKS_FILENAME, PROCESSED_TRAIN_FILENAME, PROCESSED_TEST_FILENAME]
    )


# Добавить вес взаимодействия
def add_interaction_weight(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    rating = train["rating"] if "rating" in train.columns else 0
    is_read = train["is_read"] if "is_read" in train.columns else 0

    if not isinstance(rating, pd.Series):
        rating = pd.Series([0] * len(train), index=train.index)
    if not isinstance(is_read, pd.Series):
        is_read = pd.Series([0] * len(train), index=train.index)

    rating = pd.to_numeric(rating, errors="coerce").fillna(0)
    is_read = pd.to_numeric(is_read, errors="coerce").fillna(0)
    train["interaction_weight"] = 1.0 + 0.25 * rating.clip(lower=0) + 0.5 * is_read.clip(lower=0)
    return train


# Собрать истории пользователей
def make_user_histories(train: pd.DataFrame) -> pd.DataFrame:
    started = time.time()
    rows = []
    groups = list(train.groupby("user_id", sort=False))
    total = len(groups)
    log_step = max(1000, total // 20 or 1)

    for i, (user_id, part) in enumerate(groups, start=1):
        part = part.sort_values("date_added") if "date_added" in part.columns else part
        rows.append(
            {
                "user_id": user_id,
                "seen_item_ids": part["item_id"].tolist(),
                "n_interactions": int(len(part)),
                "last_interaction_at": part["date_added"].max() if "date_added" in part.columns else None,
            }
        )
        if i == total or i % log_step == 0:
            elapsed = time.time() - started
            speed = i / elapsed if elapsed > 0 else 0.0
            eta = (total - i) / speed if speed > 0 else 0.0
            logger.info(
                "User histories: %s/%s (%.1f%%), прошло=%s, осталось~%s",
                i, total, i * 100.0 / total, _fmt_seconds(elapsed), _fmt_seconds(eta)
            )

    return pd.DataFrame(rows)


# Посчитать популярность айтемов
def make_item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    if "interaction_weight" in train.columns:
        out = (
            train.groupby("item_id", as_index=False)
            .agg(
                n_interactions=("item_id", "size"),
                popularity_weight=("interaction_weight", "sum"),
            )
            .sort_values(["popularity_weight", "n_interactions", "item_id"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
    else:
        out = (
            train.groupby("item_id", as_index=False)
            .agg(n_interactions=("item_id", "size"))
            .assign(popularity_weight=lambda x: x["n_interactions"].astype(float))
            .sort_values(["popularity_weight", "n_interactions", "item_id"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
    return out


# Собрать ground truth для evaluator
def make_eval_ground_truth(val: pd.DataFrame) -> pd.DataFrame:
    return group_test_ground_truth(val[["user_id", "item_id"]].copy())


# Сэмплировать негативы без cross join
def make_negative_samples(
    train: pd.DataFrame,
    books: pd.DataFrame,
    n_neg: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    if n_neg <= 0:
        return pd.DataFrame(columns=["user_id", "item_id", "label"])

    started = time.time()
    rng = np.random.default_rng(seed)
    all_items = books["item_id"].drop_duplicates().tolist()
    if not all_items:
        raise ValueError("books пустой, не из чего сэмплировать негативы")

    seen_by_user = (
        train.groupby("user_id", sort=False)["item_id"]
        .agg(lambda x: set(x.tolist()))
        .to_dict()
    )

    users = list(seen_by_user.keys())
    rows = []
    log_step = max(500, len(users) // 20 or 1)

    for i, user_id in enumerate(users, start=1):
        seen = seen_by_user[user_id]
        need = n_neg
        picked = set()
        tries = 0
        max_tries = n_neg * 50 + 100

        while len(picked) < need and tries < max_tries:
            item_id = all_items[int(rng.integers(0, len(all_items)))]
            tries += 1
            if item_id in seen or item_id in picked:
                continue
            picked.add(item_id)

        if len(picked) < need:
            for item_id in all_items:
                if item_id in seen or item_id in picked:
                    continue
                picked.add(item_id)
                if len(picked) >= need:
                    break

        for item_id in picked:
            rows.append({"user_id": user_id, "item_id": item_id, "label": 0})

        if i == len(users) or i % log_step == 0:
            elapsed = time.time() - started
            speed = i / elapsed if elapsed > 0 else 0.0
            eta = (len(users) - i) / speed if speed > 0 else 0.0
            logger.info(
                "Negative sampling: %s/%s (%.1f%%), прошло=%s, осталось~%s",
                i, len(users), i * 100.0 / len(users), _fmt_seconds(elapsed), _fmt_seconds(eta)
            )

    negatives = pd.DataFrame(rows)
    logger.info("Negative sampling готов: shape=%s за %s", negatives.shape, _fmt_seconds(time.time() - started))
    return negatives


# Собрать train пары (positive + negative)
def make_training_pairs(train: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    pos_cols = ["user_id", "item_id"]
    positives = train[pos_cols].copy()
    positives["label"] = 1
    if "interaction_weight" in train.columns:
        pos_weights = train[pos_cols + ["interaction_weight"]].copy()
        positives = positives.merge(pos_weights, on=pos_cols, how="left")
    else:
        positives["interaction_weight"] = 1.0

    if negatives.empty:
        out = positives.copy()
        out["interaction_weight"] = out["interaction_weight"].fillna(1.0)
        return out.reset_index(drop=True)

    neg = negatives.copy()
    if "interaction_weight" not in neg.columns:
        neg["interaction_weight"] = 0.0
    out = pd.concat([positives, neg], ignore_index=True)
    return out.reset_index(drop=True)


# Собрать summary для local датасета
def make_dataset_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    val_warm: pd.DataFrame,
    val_cold: pd.DataFrame,
    split_ts: object,
    n_neg: int,
    seed: int,
    raw_dir: str,
    k_core: int,
    keep_recent_fraction: float,
    test_fraction: float,
    interactions_chunksize: int,
    max_interaction_chunks: Optional[int],
) -> dict:
    return {
        "train_rows": int(len(train)),
        "train_users": int(train["user_id"].nunique()) if "user_id" in train.columns else 0,
        "train_items": int(train["item_id"].nunique()) if "item_id" in train.columns else 0,
        "val_rows": int(len(val)),
        "val_users": int(val["user_id"].nunique()) if "user_id" in val.columns else 0,
        "val_items": int(val["item_id"].nunique()) if "item_id" in val.columns else 0,
        "val_warm_rows": int(len(val_warm)),
        "val_cold_rows": int(len(val_cold)),
        "val_cold_ratio": float(len(val_cold) / len(val)) if len(val) else 0.0,
        "split_ts": str(split_ts),
        "n_neg": int(n_neg),
        "seed": int(seed),
        "raw_dir": str(raw_dir),
        "k_core": int(k_core),
        "keep_recent_fraction": float(keep_recent_fraction),
        "test_fraction": float(test_fraction),
        "interactions_chunksize": int(interactions_chunksize),
        "max_interaction_chunks": None if max_interaction_chunks is None else int(max_interaction_chunks),
    }


# Собрать весь локальный ML датасет
def run_dataset_build(
    data_dir: str = "data",
    output_name: str = "local_v1",
    raw_dir: str = "data/raw_data",
    rebuild_processed: bool = False,
    val_fraction: float = 0.2,
    warm_users_only: bool = True,
    n_neg: int = 0,
    seed: int = 42,
    k_core: int = 2,
    keep_recent_fraction: float = 0.6,
    test_fraction: float = 0.25,
    interactions_chunksize: int = 200000,
    max_interaction_chunks: Optional[int] = None,
) -> None:
    started = time.time()
    data_dir_path = Path(data_dir)
    out_dir = data_dir_path / "processed" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Старт сборки датасета (data_dir=%s, output=%s)", data_dir, output_name)

    if rebuild_processed or not _processed_exists(data_dir_path):
        logger.info("Запускаю предобработку raw -> parquet")
        preprocess_goodreads_raw_to_parquet(
            raw_dir=raw_dir,
            processed_dir=data_dir,
            k_core=k_core,
            keep_recent_fraction=keep_recent_fraction,
            test_fraction=test_fraction,
            interactions_chunksize=interactions_chunksize,
            max_interaction_chunks=max_interaction_chunks,
        )
    else:
        logger.info("Готовые parquet найдены, пропускаю raw -> parquet")

    books, train, _test = load_processed_goodreads(data_dir)

    if "date_added" in train.columns and not pd.api.types.is_datetime64_any_dtype(train["date_added"]):
        train["date_added"] = pd.to_datetime(train["date_added"], errors="coerce")
        train = train.dropna(subset=["date_added"]).copy()

    split = make_local_validation_split(
        train,
        val_fraction=val_fraction,
        warm_users_only=warm_users_only,
    )

    local_train = add_interaction_weight(split["train"])
    local_val = split["val"]
    local_val_warm = split["val_warm"]
    local_val_cold = split["val_cold"]

    logger.info("Собираю user_histories")
    user_histories = make_user_histories(local_train)

    logger.info("Собираю item_popularity")
    item_popularity = make_item_popularity(local_train)

    logger.info("Собираю eval_ground_truth")
    eval_ground_truth = make_eval_ground_truth(local_val)

    negatives = pd.DataFrame(columns=["user_id", "item_id", "label"])
    train_pairs = pd.DataFrame(columns=["user_id", "item_id", "label", "interaction_weight"])
    if n_neg > 0:
        logger.info("Собираю negatives (n_neg=%s)", n_neg)
        negatives = make_negative_samples(local_train, books=books, n_neg=n_neg, seed=seed)
        logger.info("Собираю train_pairs")
        train_pairs = make_training_pairs(local_train, negatives)

    summary = make_dataset_summary(
        train=local_train,
        val=local_val,
        val_warm=local_val_warm,
        val_cold=local_val_cold,
        split_ts=split["split_ts"],
        n_neg=n_neg,
        seed=seed,
        raw_dir=raw_dir,
        k_core=k_core,
        keep_recent_fraction=keep_recent_fraction,
        test_fraction=test_fraction,
        interactions_chunksize=interactions_chunksize,
        max_interaction_chunks=max_interaction_chunks,
    )

    save_parquet(local_train, out_dir / "local_train.pq")
    save_parquet(local_val, out_dir / "local_val.pq")
    save_parquet(local_val_warm, out_dir / "local_val_warm.pq")
    save_parquet(local_val_cold, out_dir / "local_val_cold.pq")
    save_parquet(user_histories, out_dir / "user_histories.pq")
    save_parquet(item_popularity, out_dir / "item_popularity.pq")
    save_parquet(eval_ground_truth, out_dir / "eval_ground_truth.pq")
    if n_neg > 0:
        save_parquet(negatives, out_dir / "negatives.pq")
        save_parquet(train_pairs, out_dir / "train_pairs.pq")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Сборка датасета завершена за %s", _fmt_seconds(time.time() - started))
    logger.info("Папка: %s", out_dir)
    print("Готово")
    print(f"Папка: {out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# CLI параметры
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сборка датасета для research (goodreads YA)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-name", default="local_v1")
    parser.add_argument("--raw-dir", default="data/raw_data")
    parser.add_argument("--rebuild-processed", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--warm-users-only", action="store_true", default=True)
    parser.add_argument("--n-neg", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-core", type=int, default=2)
    parser.add_argument("--keep-recent-fraction", type=float, default=0.6)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--interactions-chunksize", type=int, default=200000)
    parser.add_argument("--max-interaction-chunks", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_dataset_build(
        data_dir=args.data_dir,
        output_name=args.output_name,
        raw_dir=args.raw_dir,
        rebuild_processed=args.rebuild_processed,
        val_fraction=args.val_fraction,
        warm_users_only=args.warm_users_only,
        n_neg=args.n_neg,
        seed=args.seed,
        k_core=args.k_core,
        keep_recent_fraction=args.keep_recent_fraction,
        test_fraction=args.test_fraction,
        interactions_chunksize=args.interactions_chunksize,
        max_interaction_chunks=args.max_interaction_chunks,
    )
