from __future__ import annotations

import os
from datetime import date, timedelta

from source.interfaces.batch_entrypoint import main as run_batch_once


def _parse_days(raw: str | None) -> int:
    value = (raw or "").strip()
    if not value:
        return 5
    days = int(value)
    if days <= 0:
        raise ValueError("BOOKRECS_BATCH_BACKFILL_DAYS must be > 0")
    return days


def _parse_end_date(raw: str | None) -> date:
    value = (raw or "").strip()
    if not value:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("BOOKRECS_BATCH_END_DATE must be in YYYY-MM-DD format") from exc


def _build_dates(end_date: date, days: int) -> list[str]:
    start = end_date - timedelta(days=days - 1)
    return [(start + timedelta(days=offset)).isoformat() for offset in range(days)]


def main() -> None:
    days = _parse_days(os.getenv("BOOKRECS_BATCH_BACKFILL_DAYS"))
    end_date = _parse_end_date(os.getenv("BOOKRECS_BATCH_END_DATE"))
    execution_dates = _build_dates(end_date=end_date, days=days)

    # Для backfill нужен run_name, зависящий от execution date.
    os.environ.pop("BOOKRECS_BATCH_RUN_NAME", None)

    print(
        f"[batch-backfill] start: days={days}, end_date={end_date.isoformat()}, "
        f"range={execution_dates[0]}..{execution_dates[-1]}",
        flush=True,
    )

    for execution_date in execution_dates:
        os.environ["BOOKRECS_BATCH_EXECUTION_DATE"] = execution_date
        print(f"[batch-backfill] run execution_date={execution_date}", flush=True)
        run_batch_once()

    print("[batch-backfill] done", flush=True)


if __name__ == "__main__":
    main()
