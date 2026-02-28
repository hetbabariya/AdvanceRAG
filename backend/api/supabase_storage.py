from __future__ import annotations

import os
from typing import Optional

from backend.api.utils import get_logger

logger = get_logger(__name__)


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def supabase_enabled() -> bool:
    return _truthy(os.getenv("SUPABASE_STORAGE_ENABLED", "1"))


def _get_required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _client():
    from supabase import create_client  # type: ignore

    url = _get_required_env("SUPABASE_URL")
    key = _get_required_env("SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)


def _bucket_name() -> str:
    return _get_required_env("SUPABASE_BUCKET")


def upload_bytes(*, path: str, content: bytes, content_type: str) -> None:
    if not supabase_enabled():
        return

    sb = _client()
    bucket = _bucket_name()

    resp = sb.storage.from_(bucket).upload(
        path=path,
        file=content,
        file_options={"content-type": str(content_type), "upsert": "true"},
    )

    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(str(resp.get("error")))


def upload_file(*, local_path: str, remote_path: str, content_type: str) -> None:
    if not supabase_enabled():
        return

    with open(local_path, "rb") as f:
        data = f.read()
    upload_bytes(path=remote_path, content=data, content_type=content_type)


def download_to_file(*, remote_path: str, local_path: str) -> bool:
    if not supabase_enabled():
        return False

    sb = _client()
    bucket = _bucket_name()

    data = sb.storage.from_(bucket).download(remote_path)
    if not data:
        return False

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
    return True
