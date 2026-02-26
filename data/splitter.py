import logging
from typing import Dict

import pandas as pd


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# Разделить interactions по времени
def split_by_time(interactions: pd.DataFrame, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    if interactions.empty:
        raise ValueError("interactions пустой")
    if "date_added" not in interactions.columns:
        raise ValueError("В interactions нет date_added")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction должен быть в (0, 1)")

    q = 1 - val_fraction
    split_ts = interactions["date_added"].quantile(q, interpolation="nearest")
    train = interactions[interactions["date_added"] < split_ts].copy()
    val = interactions[interactions["date_added"] >= split_ts].copy()

    if train.empty or val.empty:
        raise ValueError("После split получился пустой train или val")

    train = train.sort_values(["date_added", "user_id", "item_id"]).reset_index(drop=True)
    val = val.sort_values(["user_id", "date_added", "item_id"]).reset_index(drop=True)
    return train, val, split_ts


# Оставить только warm пользователей в val
def keep_warm_users(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    train_users = set(train["user_id"].tolist())
    out = val[val["user_id"].isin(train_users)].copy()
    if out.empty:
        raise ValueError("val пустой после фильтра warm users")
    return out


# Разбить val на warm/cold items
def split_warm_cold_items(train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, set, set]:
    train_items = set(train["item_id"].tolist())
    val_items = set(val["item_id"].tolist())

    warm_items = val_items & train_items
    cold_items = val_items - train_items

    val_warm = val[val["item_id"].isin(warm_items)].copy().reset_index(drop=True)
    val_cold = val[val["item_id"].isin(cold_items)].copy().reset_index(drop=True)
    return val_warm, val_cold, warm_items, cold_items


# Проверить split
def check_split(train: pd.DataFrame, val: pd.DataFrame, val_warm: pd.DataFrame, val_cold: pd.DataFrame) -> None:
    if train["date_added"].max() >= val["date_added"].min():
        raise ValueError("Temporal leakage: train max >= val min")

    train_users = set(train["user_id"].tolist())
    val_users = set(val["user_id"].tolist())
    if not val_users.issubset(train_users):
        raise ValueError("В val есть cold users")

    if len(val_warm) + len(val_cold) != len(val):
        raise ValueError("Warm/cold split не совпадает с размером val")

    warm_items = set(val_warm["item_id"].tolist())
    cold_items = set(val_cold["item_id"].tolist())
    if warm_items & cold_items:
        raise ValueError("Warm и cold items пересекаются")


# Собрать локальный split для research
def make_local_validation_split(
    interactions: pd.DataFrame,
    val_fraction: float = 0.2,
    warm_users_only: bool = True,
) -> Dict[str, object]:
    logger.info("Локальный split: val_fraction=%s warm_users_only=%s", val_fraction, warm_users_only)
    train, val, split_ts = split_by_time(interactions, val_fraction=val_fraction)

    if warm_users_only:
        val = keep_warm_users(train, val)

    val_warm, val_cold, warm_items, cold_items = split_warm_cold_items(train, val)
    check_split(train, val, val_warm, val_cold)

    logger.info(
        "Локальный split готов: train=%s val=%s warm=%s cold=%s cold_ratio=%.4f",
        train.shape,
        val.shape,
        val_warm.shape,
        val_cold.shape,
        (len(val_cold) / len(val)) if len(val) else 0.0,
    )

    return {
        "train": train,
        "val": val,
        "val_warm": val_warm,
        "val_cold": val_cold,
        "warm_items": warm_items,
        "cold_items": cold_items,
        "split_ts": split_ts,
    }
