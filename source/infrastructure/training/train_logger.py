from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
# Описывает логгер обучения.
class TrainLogger:
    def __init__(self, run_id: str, log_file: Path) -> None:
        self._run_id = run_id
        self._log_file = log_file
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._step_started_at: dict[str, float] = {}
        self._run_started_at = time.time()
        self._event_seq = 0
        self._stdout_format = (os.getenv("BOOKRECS_TRAIN_STDOUT_FORMAT") or "pretty").strip().lower()

    def event(self, event: str, **payload: Any) -> None:
        self._event_seq += 1
        ts_unix = time.time()
        row = {
            "ts_unix": ts_unix,
            "ts_utc": datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat(),
            "event_seq": self._event_seq,
            "run_id": self._run_id,
            "event": event,
            "run_elapsed_sec": round(ts_unix - self._run_started_at, 3),
            **payload,
        }
        line = json.dumps(row, ensure_ascii=False)
        if self._stdout_format == "json":
            print(line, flush=True)
        else:
            pretty = self._format_pretty(event=event, payload=row)
            if pretty:
                print(pretty, flush=True)
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
        progress_ratio = float(done / total) if total > 0 else 0.0
        self.event(
            "PROGRESS",
            step=step,
            progress_done=done,
            progress_total=total,
            progress_ratio=round(progress_ratio, 6),
            progress_pct=round(progress_ratio * 100.0, 2),
            elapsed_sec=round(elapsed, 3),
            eta_sec=(None if eta_sec is None else round(eta_sec, 3)),
        )

    def end_step(self, step: str, status: str = "SUCCESS", **payload: Any) -> None:
        started = self._step_started_at.get(step, time.time())
        duration_sec = time.time() - started
        self.event("END", step=step, status=status, duration_sec=round(duration_sec, 3), **payload)

    def _format_pretty(self, event: str, payload: dict[str, Any]) -> str:
        if event == "RUN_START":
            config = payload.get("config", {})
            return (
                f"[train] run start id={payload.get('run_id')} "
                f"profile={config.get('train_profile')} "
                f"pool={config.get('candidate_pool_size')} "
                f"pre_top_m={config.get('pre_top_m')} "
                f"prerank={config.get('prerank_model')}"
            )
        if event == "DATA_LOADED":
            return (
                f"[train] data loaded train={payload.get('train_rows')} "
                f"val={payload.get('val_rows')} books={payload.get('books_rows')}"
            )
        if event == "START":
            total = payload.get("progress_total")
            total_part = f"/{total}" if total is not None else ""
            return f"[train] {payload.get('step')} start 0{total_part}"
        if event == "PROGRESS":
            step = payload.get("step")
            pct = payload.get("progress_pct")
            done = payload.get("progress_done")
            total = payload.get("progress_total")
            eta = _format_duration(payload.get("eta_sec"))
            elapsed = _format_duration(payload.get("elapsed_sec"))
            return f"[train] {step} {pct}% ({done}/{total}) elapsed={elapsed} eta={eta}"
        if event == "END":
            step = payload.get("step")
            duration = _format_duration(payload.get("duration_sec"))
            return f"[train] {step} {payload.get('status', 'DONE').lower()} in {duration}"
        if event == "RUN_END":
            duration = _format_duration(payload.get("duration_sec"))
            metrics = payload.get("metrics", {})
            metrics_summary = _format_metrics_summary(metrics)
            return f"[train] run complete status={payload.get('status')} duration={duration} {metrics_summary}".strip()
        return ""


def _format_duration(value: Any) -> str:
    if value is None:
        return "?"
    total = max(0, int(round(float(value))))
    minutes, seconds = divmod(total, 60)
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _format_metrics_summary(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    keys = [
        "ndcg@10",
        "recall@10",
        "cold_ndcg@10",
        "cold_recall@10",
        "candidate_recall@450",
        "prerank_recall@120",
    ]
    parts = []
    for key in keys:
        if key not in metrics:
            continue
        parts.append(f"{key}={float(metrics[key]):.4f}")
    return " ".join(parts)
