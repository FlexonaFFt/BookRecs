import pandas as pd


# Разделить данные по времени на train и val
def split_by_time(
    interactions: pd.DataFrame, val_fraction: float = 0.2,
    timestamp_col: str = "date_added",
) -> tuple[pd.DataFrame, pd.DataFrame, object]:

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if interactions.empty:
        raise ValueError("interactions is empty")
    if timestamp_col not in interactions.columns:
        raise ValueError(f"Column not found: {timestamp_col}")

    q = 1 - val_fraction
    split_ts = interactions[timestamp_col].quantile(q, interpolation="nearest")
    train = interactions[interactions[timestamp_col] < split_ts].copy()
    val = interactions[interactions[timestamp_col] >= split_ts].copy()

    if train.empty or val.empty:
        raise ValueError("Split produced empty train or val")

    train_sort = [c for c in ["date_added", "user_id", "item_id"] if c in train.columns]
    val_sort = [c for c in ["user_id", "date_added", "item_id"] if c in val.columns]

    if train_sort:
        train = train.sort_values(train_sort).reset_index(drop=True)
    if val_sort:
        val = val.sort_values(val_sort).reset_index(drop=True)

    return train, val, split_ts


# Оставим в валидации только теплых пользователей
def keep_warm_users(
    train: pd.DataFrame,
    val: pd.DataFrame,
    user_col: str = "user_id",
) -> pd.DataFrame:
    train_users = set(train[user_col].tolist())
    val_warm_users = val[val[user_col].isin(train_users)].copy()
    if val_warm_users.empty:
        raise ValueError("Validation is empty after warm-user filtering")
    return val_warm_users


# Разделим валидацию на warm и cold айтемы
def split_warm_cold_items(
    train: pd.DataFrame,
    val: pd.DataFrame,
    item_col: str = "item_id",
) -> tuple[pd.DataFrame, pd.DataFrame, set, set]:
    train_items = set(train[item_col].tolist())
    val_items = set(val[item_col].tolist())

    warm_items = val_items & train_items
    cold_items = val_items - train_items

    val_warm = val[val[item_col].isin(warm_items)].copy()
    val_cold = val[val[item_col].isin(cold_items)].copy()

    return val_warm, val_cold, warm_items, cold_items


def check_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    val_warm: pd.DataFrame,
    val_cold: pd.DataFrame,
    timestamp_col: str = "date_added",
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> None:
    if train.empty:
        raise ValueError("Train is empty")
    if val.empty:
        raise ValueError("Validation is empty")

    train_max_ts = train[timestamp_col].max()
    val_min_ts = val[timestamp_col].min()
    if train_max_ts >= val_min_ts:
        raise ValueError("Temporal leakage: train max timestamp >= val min timestamp")

    train_users = set(train[user_col].tolist())
    val_users = set(val[user_col].tolist())
    if not val_users.issubset(train_users):
        raise ValueError("Validation contains cold users")

    if len(val_warm) + len(val_cold) != len(val):
        raise ValueError("Warm/cold split does not match validation size")

    warm_items = set(val_warm[item_col].tolist())
    cold_items = set(val_cold[item_col].tolist())
    if warm_items & cold_items:
        raise ValueError("Warm/cold item sets overlap")


# Собрать локальный сплит для разработки моделей
def make_local_validation_split(
    interactions: pd.DataFrame,
    val_fraction: float = 0.2,
    warm_users_only: bool = True,
) -> dict:
    train, val, split_ts = split_by_time(interactions, val_fraction=val_fraction)

    if warm_users_only:
        val = keep_warm_users(train, val)

    val_warm, val_cold, warm_items, cold_items = split_warm_cold_items(train, val)
    check_split(train, val, val_warm, val_cold)

    return {
        "train": train,
        "val": val,
        "val_warm": val_warm,
        "val_cold": val_cold,
        "warm_items": warm_items,
        "cold_items": cold_items,
        "split_ts": split_ts,
    }
