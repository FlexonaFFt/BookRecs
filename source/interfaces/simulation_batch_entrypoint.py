from __future__ import annotations

import os

from source.interfaces.batch_entrypoint import main as _run_batch


def main() -> None:
    execution_date_str = (os.getenv("BOOKRECS_BATCH_EXECUTION_DATE") or "").strip()
    execution_compact = (
        execution_date_str.replace("-", "") if execution_date_str else "manual"
    )

    existing_run_name = (os.getenv("BOOKRECS_BATCH_RUN_NAME") or "").strip()
    if not existing_run_name:
        os.environ["BOOKRECS_BATCH_RUN_NAME"] = f"simulation_{execution_compact}"

    run_name = os.environ["BOOKRECS_BATCH_RUN_NAME"]
    print(f"[simulation] execution_date={execution_date_str or 'N/A'}", flush=True)
    print(f"[simulation] run_name={run_name}", flush=True)

    _run_batch()


if __name__ == "__main__":
    main()
