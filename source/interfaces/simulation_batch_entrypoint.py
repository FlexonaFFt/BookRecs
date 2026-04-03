from __future__ import annotations

import os

from source.interfaces.batch_entrypoint import main as _run_batch


def main() -> None:
    execution_date_str = (os.getenv("BOOKRECS_BATCH_EXECUTION_DATE") or "").strip()
    execution_compact = (
        execution_date_str.replace("-", "") if execution_date_str else "manual"
    )

    os.environ["BOOKRECS_BATCH_RUN_NAME"] = f"simulation_{execution_compact}"

    print(f"[simulation] execution_date={execution_date_str or 'N/A'}", flush=True)
    print(f"[simulation] run_name=simulation_{execution_compact}", flush=True)

    _run_batch()


if __name__ == "__main__":
    main()
