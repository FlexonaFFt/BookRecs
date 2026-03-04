from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TrainLogger:


    def __init__(self, run_id: str, log_file: Path) -> None:
        self._run_id = run_id
        self._log_file = log_file
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._step_started_at: dict[str, float] = {}

    def event(self, event: str, **payload: Any) -> None:
        row = {
            "ts_unix": time.time(),
            "run_id": self._run_id,
            "event": event,
            **payload,
        }
        line = json.dumps(row, ensure_ascii=False)
        print(line, flush=True)
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def start_step(self, step: str, total: int | None = None) -> None:
        self._step_started_at[step] = time.time()
        self.event("START", step=step, progress_total=total)

    def progress(self, step: str, done: int, total: int) -> None:
        started = self._step_started_at.get(step, time.time())
        elapsed = max(1e-9, time.time() - started)
        speed = done / elapsed if done > 0 else 0.0
        eta_sec = (total - done) / speed if speed > 0 else None
        self.event(
            "PROGRESS",
            step=step,
            progress_done=done,
            progress_total=total,
            elapsed_sec=round(elapsed, 3),
            eta_sec=(None if eta_sec is None else round(eta_sec, 3)),
        )

    def end_step(self, step: str, status: str = "SUCCESS", **payload: Any) -> None:
        started = self._step_started_at.get(step, time.time())
        duration_sec = time.time() - started
        self.event("END", step=step, status=status, duration_sec=round(duration_sec, 3), **payload)

