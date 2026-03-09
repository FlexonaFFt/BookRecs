from __future__ import annotations

import json
import os
import re
from pathlib import Path

from source.infrastructure.config import load_pipeline_settings
from source.interfaces.pipeline_entrypoint import run_pipeline_from_env


def _normalize_date_key(raw: str | None) -> str:
    value = (raw or '').strip()
    if not value:
        return 'manual'
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        return value.replace('-', '')
    compact = re.sub(r'[^0-9]', '', value)
    if len(compact) >= 8:
        return compact[:8]
    return 'manual'


def _build_run_name() -> str:
    explicit = (os.getenv('BOOKRECS_BATCH_RUN_NAME') or '').strip()
    if explicit:
        return explicit
    execution_ds = os.getenv('BOOKRECS_BATCH_EXECUTION_DATE')
    key = _normalize_date_key(execution_ds)
    return f'batch_{key}'


def _success_manifest_exists(output_root: str, run_name: str) -> bool:
    manifest = Path(output_root) / run_name / 'manifest.json'
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding='utf-8'))
    except Exception:
        return False
    return str(payload.get('status', '')).upper() == 'SUCCESS'


def main() -> None:
    settings = load_pipeline_settings()
    run_name = _build_run_name()

    os.environ['BOOKRECS_TRAIN_RUN_NAME'] = run_name

    if _success_manifest_exists(settings.output_root, run_name):
        print(f'[batch] skip: run_name={run_name} already SUCCESS', flush=True)
        return

    print(f'[batch] start: run_name={run_name}', flush=True)
    run_pipeline_from_env()


if __name__ == '__main__':
    main()
