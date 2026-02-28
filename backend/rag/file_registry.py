from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List


@dataclass(frozen=True)
class FileRecord:
    file_name: str
    chunks_upserted: int
    ingested_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def registry_path(storage_dir: str) -> str:
    _ensure_dir(storage_dir)
    return os.path.join(storage_dir, "files.json")


def load_registry(storage_dir: str) -> Dict[str, FileRecord]:
    path = registry_path(storage_dir)
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f) or {}

    out: Dict[str, FileRecord] = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        file_name = v.get("file_name") or k
        out[file_name] = FileRecord(
            file_name=file_name,
            chunks_upserted=int(v.get("chunks_upserted") or 0),
            ingested_at=str(v.get("ingested_at") or ""),
        )

    return out


def save_registry(storage_dir: str, records: Dict[str, FileRecord]) -> None:
    path = registry_path(storage_dir)
    payload = {
        k: {
            "file_name": r.file_name,
            "chunks_upserted": r.chunks_upserted,
            "ingested_at": r.ingested_at,
        }
        for k, r in records.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def upsert_record(storage_dir: str, *, file_name: str, chunks_upserted: int) -> FileRecord:
    records = load_registry(storage_dir)
    rec = FileRecord(file_name=file_name, chunks_upserted=chunks_upserted, ingested_at=_utc_now_iso())
    records[file_name] = rec
    save_registry(storage_dir, records)
    return rec


def list_records(storage_dir: str) -> List[FileRecord]:
    records = load_registry(storage_dir)
    return sorted(records.values(), key=lambda r: r.ingested_at, reverse=True)
