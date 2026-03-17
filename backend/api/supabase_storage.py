from __future__ import annotations

import os
import mimetypes
from typing import Optional

from backend.api.utils import get_logger

logger = get_logger(__name__)


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def s3_enabled() -> bool:
    return _truthy(os.getenv("AWS_S3_ENABLED", "1"))


def _get_required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _s3_client():
    import boto3  # type: ignore

    region = (os.getenv("AWS_REGION") or "").strip() or None
    return boto3.client("s3", region_name=region)


def _bucket_name() -> str:
    return _get_required_env("AWS_S3_BUCKET")


def _prefix() -> str:
    prefix = (os.getenv("AWS_S3_PREFIX") or "").strip().strip("/")
    return prefix


def _key(remote_path: str) -> str:
    remote_path = (remote_path or "").lstrip("/")
    pfx = _prefix()
    return f"{pfx}/{remote_path}" if pfx else remote_path


def upload_bytes(*, path: str, content: bytes, content_type: str) -> None:
    if not s3_enabled():
        return

    bucket = _bucket_name()
    key = _key(path)
    ct = (content_type or "").strip() or "application/octet-stream"
    s3 = _s3_client()
    s3.put_object(Bucket=bucket, Key=key, Body=content, ContentType=ct)


def upload_file(*, local_path: str, remote_path: str, content_type: str) -> None:
    if not s3_enabled():
        return

    guessed, _ = mimetypes.guess_type(local_path)
    ct = (content_type or "").strip() or guessed or "application/octet-stream"

    with open(local_path, "rb") as f:
        data = f.read()
    upload_bytes(path=remote_path, content=data, content_type=ct)


def download_to_file(*, remote_path: str, local_path: str) -> bool:
    if not s3_enabled():
        return False

    bucket = _bucket_name()
    key = _key(remote_path)
    s3 = _s3_client()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj.get("Body").read() if obj.get("Body") is not None else b""
    except Exception:
        return False

    if not data:
        return False

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
    return True


def delete_remote(*, remote_path: str) -> None:
    if not s3_enabled():
        return

    bucket = _bucket_name()
    key = _key(remote_path)
    s3 = _s3_client()
    try:
        s3.delete_object(Bucket=bucket, Key=key)
    except Exception:
        return
