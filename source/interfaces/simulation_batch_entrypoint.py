from __future__ import annotations

import os
from datetime import date, timedelta

from source.interfaces.batch_entrypoint import main as _run_batch

_DEFAULT_EXEC_WINDOW_START = "2026-03-01"
_DEFAULT_EXEC_WINDOW_END = "2026-04-01"
_DEFAULT_DATA_START = "2011-01-01"
_DEFAULT_DATA_END = "2017-01-01"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def _compute_cutoff_date(
    execution_date: date,
    exec_window_start: date,
    exec_window_end: date,
    data_start: date,
    data_end: date,
) -> date:
    total_exec_days = (exec_window_end - exec_window_start).days
    if total_exec_days <= 0:
        return data_end
    elapsed = (execution_date - exec_window_start).days
    progress = max(0.0, min(1.0, elapsed / total_exec_days))
    data_range_days = (data_end - data_start).days
    return data_start + timedelta(days=int(progress * data_range_days))


def main() -> None:
    execution_date_str = (os.getenv("BOOKRECS_BATCH_EXECUTION_DATE") or "").strip()
    exec_window_start = _parse_date(
        os.getenv("BOOKRECS_SIMULATION_WINDOW_START") or _DEFAULT_EXEC_WINDOW_START
    )
    exec_window_end = _parse_date(
        os.getenv("BOOKRECS_SIMULATION_WINDOW_END") or _DEFAULT_EXEC_WINDOW_END
    )
    data_start = _parse_date(
        os.getenv("BOOKRECS_SIMULATION_DATA_START") or _DEFAULT_DATA_START
    )
    data_end = _parse_date(
        os.getenv("BOOKRECS_SIMULATION_DATA_END") or _DEFAULT_DATA_END
    )

    if execution_date_str:
        execution_date = _parse_date(execution_date_str)
        cutoff = _compute_cutoff_date(
            execution_date, exec_window_start, exec_window_end, data_start, data_end
        )
    else:
        cutoff = data_end

    cutoff_str = cutoff.isoformat()
    cutoff_compact = cutoff.strftime("%Y%m%d")
    dataset_name = f"goodreads_ya_sim_{cutoff_compact}"
    dataset_dir = f"artifacts/tmp_preprocessed/{dataset_name}"
    execution_compact = (
        execution_date_str.replace("-", "") if execution_date_str else "manual"
    )

    os.environ["BOOKRECS_SIMULATION_CUTOFF_DATE"] = cutoff_str
    os.environ["BOOKRECS_DATASET_NAME"] = dataset_name
    os.environ["BOOKRECS_TRAIN_DATASET_DIR"] = dataset_dir
    os.environ["BOOKRECS_BATCH_RUN_NAME"] = f"simulation_{execution_compact}"

    print(
        f"[simulation] execution_date={execution_date_str or 'N/A'}"
        f" → data_cutoff={cutoff_str}",
        flush=True,
    )
    print(f"[simulation] dataset_name={dataset_name}", flush=True)

    _run_batch()


if __name__ == "__main__":
    main()
