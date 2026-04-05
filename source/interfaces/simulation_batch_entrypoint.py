from __future__ import annotations

import os
import re

from source.infrastructure.config import load_pipeline_settings
from source.interfaces.batch_entrypoint import _success_manifest_exists
from source.interfaces.batch_entrypoint import main as _run_batch


def _resolve_run_name(base_name: str, output_root: str) -> str:
    """Return base_name if not yet SUCCESS.

    If already SUCCESS, increments the trailing number in the name:
      sim2_15pct -> sim3_15pct -> sim4_15pct ...
    Falls back to appending _v2, _v3 if no number is found.
    """
    if not _success_manifest_exists(output_root, base_name):
        return base_name

    match = re.match(r"^(.*?)(\d+)(_\S+)?$", base_name)
    if match:
        prefix, num, suffix = match.group(1), int(match.group(2)), match.group(3) or ""
        next_num = num + 1
        while True:
            candidate = f"{prefix}{next_num}{suffix}"
            if not _success_manifest_exists(output_root, candidate):
                return candidate
            next_num += 1

    # fallback: no numeric part found
    version = 2
    while True:
        candidate = f"{base_name}_v{version}"
        if not _success_manifest_exists(output_root, candidate):
            return candidate
        version += 1


def main() -> None:
    execution_date_str = (os.getenv("BOOKRECS_BATCH_EXECUTION_DATE") or "").strip()
    execution_compact = (
        execution_date_str.replace("-", "") if execution_date_str else "manual"
    )

    base_name = (os.getenv("BOOKRECS_BATCH_RUN_NAME") or "").strip()
    if not base_name:
        base_name = f"simulation_{execution_compact}"

    settings = load_pipeline_settings()
    run_name = _resolve_run_name(base_name, settings.output_root)

    os.environ["BOOKRECS_BATCH_RUN_NAME"] = run_name
    print(f"[simulation] execution_date={execution_date_str or 'N/A'}", flush=True)
    print(f"[simulation] run_name={run_name}", flush=True)

    _run_batch()


if __name__ == "__main__":
    main()
