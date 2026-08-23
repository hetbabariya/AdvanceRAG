"""In-process ingestion job registry.

Jobs live in memory (single-process deployment). The API layer creates a job,
runs the blocking ingestion in a worker thread, and clients poll
GET /ingest/status/{job_id} for progress.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.api.utils import get_logger

logger = get_logger(__name__)

_FINISHED_TTL_S = 30 * 60  # prune finished jobs after 30 minutes


@dataclass
class IngestJob:
    id: str
    user_id: int
    file_name: str
    status: str = "queued"          # queued | parsing | embedding | upserting | completed | failed
    progress_done: int = 0
    progress_total: int = 0
    error: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "file_name": self.file_name,
            "status": self.status,
            "chunks_embedded": self.progress_done,
            "total_chunks": self.progress_total,
            "error": self.error,
            "result": self.result if self.status == "completed" else {},
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, IngestJob] = {}
        self._lock = threading.Lock()

    def create(self, *, user_id: int, file_name: str) -> IngestJob:
        job = IngestJob(id=uuid.uuid4().hex[:16], user_id=user_id, file_name=file_name)
        with self._lock:
            self._prune_locked()
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str, *, user_id: int) -> Optional[IngestJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.user_id == user_id:
                return job
        return None

    def update(self, job_id: str, **fields: Any) -> None:
        fields.setdefault("updated_at", time.time())
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def _prune_locked(self) -> None:
        now = time.time()
        stale = [
            jid for jid, job in self._jobs.items()
            if job.status in {"completed", "failed"} and now - job.updated_at > _FINISHED_TTL_S
        ]
        for jid in stale:
            del self._jobs[jid]

    def active_for_user(self, user_id: int) -> List[IngestJob]:
        with self._lock:
            return [j for j in self._jobs.values() if j.user_id == user_id]


registry = JobRegistry()
