from __future__ import annotations

import json
import os
from pathlib import Path


def _resolve_run_dir() -> Path:
    run_dir_raw = (os.getenv("BOOKRECS_REPORT_RUN_DIR") or "").strip()
    if run_dir_raw:
        return Path(run_dir_raw)

    run_id = (os.getenv("BOOKRECS_REPORT_RUN_ID") or "").strip()
    output_root = Path((os.getenv("BOOKRECS_TRAIN_OUTPUT_ROOT") or "artifacts/runs").strip())
    if run_id:
        return output_root / run_id

    candidates = [path for path in output_root.iterdir() if path.is_dir()] if output_root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No runs found in {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    run_dir = _resolve_run_dir()
    metrics_path = run_dir / "metrics.json"
    timings_path = run_dir / "timings.json"
    manifest_path = run_dir / "manifest.json"

    metrics = _load_json(metrics_path)
    timings = _load_json(timings_path) if timings_path.exists() else {}
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}

    print(f"[metrics] run_dir={run_dir}")
    print(f"[metrics] status={manifest.get('status', 'UNKNOWN')}")
    if manifest.get("duration_sec") is not None:
        print(f"[metrics] duration_sec={manifest['duration_sec']}")

    for key in sorted(metrics.keys()):
        print(f"[metrics] {key}={float(metrics[key]):.6f}")

    if timings:
        print("[metrics] timings")
        for key in sorted(timings.keys()):
            print(f"[metrics]   {key}={float(timings[key]):.3f}s")


if __name__ == "__main__":
    main()
