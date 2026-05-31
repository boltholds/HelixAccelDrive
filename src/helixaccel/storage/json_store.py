from __future__ import annotations

import json
from pathlib import Path

from helixaccel.storage.run_record import RunRecord


def save_run_record(record: RunRecord, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record.run_id}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record.model_dump(), fh, indent=2, ensure_ascii=False)
    return path
