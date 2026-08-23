"""Optional cloud object storage — Supabase Storage.

Design:
    * LOCAL-FIRST: everything is stored on local disk by default
      (UPLOAD_DIR for raw files, BM25_STORE_DIR for pickles).
    * Cloud mirroring is strictly opt-in via SUPABASE_STORAGE_ENABLED=1.
      When enabled, raw uploads and BM25 artifacts are mirrored to a
      Supabase bucket so the backend can rehydrate them after a restart
      on ephemeral hosts (e.g. Render free tier).

Supabase Storage REST API (no SDK needed, uses httpx which is already
a project dependency):
    Upload   POST   {url}/storage/v1/object/{bucket}/{path}
    Download GET    {url}/storage/v1/object/{bucket}/{path}
    Delete   DELETE {url}/storage/v1/object/{bucket}/{path}

All requests are authenticated with the service role key.
"""
from __future__ import annotations

import mimetypes
import os
from typing import Optional

import httpx

from backend.api.utils import get_logger

logger = get_logger(__name__)

_HTTP_TIMEOUT_S = float(os.getenv("STORAGE_HTTP_TIMEOUT", "60") or "60")


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def remote_storage_enabled() -> bool:
    """Cloud mirroring is opt-in — defaults to OFF (pure local mode)."""
    return _truthy(os.getenv("SUPABASE_STORAGE_ENABLED", "0"))


def _get_required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip().strip('"').strip("'")
    if not value:
        raise RuntimeError(
            f"SUPABASE_STORAGE_ENABLED=1 but required environment variable is missing: {name}"
        )
    return value


def _base_url() -> str:
    url = _get_required_env("SUPABASE_URL").rstrip("/")
    bucket = _get_required_env("SUPABASE_BUCKET")
    return f"{url}/storage/v1/object/{bucket}"


def _headers(content_type: str) -> dict[str, str]:
    key = _get_required_env("SUPABASE_SERVICE_ROLE_KEY")
    return {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": content_type,
    }


def upload_bytes(*, path: str, content: bytes, content_type: str) -> None:
    """Upload raw bytes. Raises on failure so callers can surface a 500."""
    if not remote_storage_enabled():
        return

    ct = (content_type or "").strip() or "application/octet-stream"
    url = f"{_base_url()}/{(path or '').lstrip('/')}"
    headers = _headers(ct)
    headers["x-upsert"] = "true"

    with httpx.Client(timeout=_HTTP_TIMEOUT_S) as client:
        resp = client.post(url, headers=headers, content=content)
    if resp.status_code >= 400:
        raise RuntimeError(f"Supabase upload failed ({resp.status_code}): {_safe_error(resp)}")


def upload_file(*, local_path: str, remote_path: str, content_type: str) -> None:
    if not remote_storage_enabled():
        return

    guessed, _ = mimetypes.guess_type(local_path)
    ct = (content_type or "").strip() or guessed or "application/octet-stream"

    with open(local_path, "rb") as f:
        data = f.read()
    upload_bytes(path=remote_path, content=data, content_type=ct)


def download_to_file(*, remote_path: str, local_path: str) -> bool:
    """Download an artifact into local_path. Returns False if missing/unreachable
    (callers treat this as 're-ingest needed', never as a hard error)."""
    if not remote_storage_enabled():
        return False

    try:
        url = f"{_base_url()}/{(remote_path or '').lstrip('/')}"
        with httpx.Client(timeout=_HTTP_TIMEOUT_S) as client:
            resp = client.get(url, headers=_headers("application/json"))
        if resp.status_code != 200 or not resp.content:
            logger.debug("Supabase download miss (%s): %s", resp.status_code, remote_path)
            return False

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception:
        logger.debug("Supabase download failed for %s", remote_path, exc_info=True)
        return False


def delete_remote(*, remote_path: str) -> None:
    """Best-effort deletion — never raises."""
    if not remote_storage_enabled():
        return

    try:
        url = f"{_base_url()}/{(remote_path or '').lstrip('/')}"
        with httpx.Client(timeout=_HTTP_TIMEOUT_S) as client:
            client.delete(url, headers=_headers("application/json"))
    except Exception:
        logger.debug("Supabase delete failed for %s", remote_path, exc_info=True)


def _safe_error(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        msg = body.get("message") or body.get("error") or ""
        if msg:
            return str(msg)
    except Exception:
        pass
    text = (resp.text or "").strip()
    return text[:200] if text else resp.reason_phrase or "unknown error"
