from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import socket
import struct
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from backend.api.schemas import Citation


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Logging is configured once at application startup via ``configure_logging``.
    Call this function at module level::

        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a structured format.

    Should be called once at application startup before any other code runs.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "asyncio", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file (streaming, memory-safe)."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_file_size(file_path: str) -> int:
    """Return file size in bytes."""
    return os.path.getsize(file_path)


def ensure_directory(directory: str) -> None:
    """Ensure a directory exists, creating it (and parents) if needed."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def safe_upload_path(upload_dir: str, user_id: int, file_hash: str, suffix: str) -> str:
    """Return a collision-free upload path scoped to user_id and file content.

    Two different users uploading ``report.pdf`` will never overwrite each
    other because the path includes both ``user_id`` and the SHA-256 hash of
    the content.

    Example: ``uploads/42/a3f1…b9.pdf``
    """
    user_dir = os.path.join(upload_dir, str(user_id))
    ensure_directory(user_dir)
    filename = f"{file_hash[:16]}{suffix}"
    return os.path.join(user_dir, filename)


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

_PRIVATE_RANGES: list[tuple[int, int]] = [
    # Loopback
    (struct.unpack("!I", socket.inet_aton("127.0.0.0"))[0],
     struct.unpack("!I", socket.inet_aton("127.255.255.255"))[0]),
    # Private 10.x
    (struct.unpack("!I", socket.inet_aton("10.0.0.0"))[0],
     struct.unpack("!I", socket.inet_aton("10.255.255.255"))[0]),
    # Private 172.16–172.31
    (struct.unpack("!I", socket.inet_aton("172.16.0.0"))[0],
     struct.unpack("!I", socket.inet_aton("172.31.255.255"))[0]),
    # Private 192.168.x
    (struct.unpack("!I", socket.inet_aton("192.168.0.0"))[0],
     struct.unpack("!I", socket.inet_aton("192.168.255.255"))[0]),
    # Link-local / AWS metadata
    (struct.unpack("!I", socket.inet_aton("169.254.0.0"))[0],
     struct.unpack("!I", socket.inet_aton("169.254.255.255"))[0]),
]


def is_safe_url(url: str) -> tuple[bool, Optional[str]]:
    """Return ``(True, None)`` if the URL is safe to fetch, else ``(False, reason)``.

    Checks:
    * Scheme must be HTTPS (or HTTP — configurable via env ``ALLOW_HTTP_INGEST=1``)
    * Resolved IPv4 must not fall in any private/loopback/link-local range
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    allow_http = os.getenv("ALLOW_HTTP_INGEST", "0") == "1"
    allowed_schemes = {"https", "http"} if allow_http else {"https"}

    if parsed.scheme not in allowed_schemes:
        return False, f"Scheme '{parsed.scheme}' is not allowed; use https://"

    hostname = parsed.hostname
    if not hostname:
        return False, "Could not determine hostname from URL"

    try:
        ip_str = socket.gethostbyname(hostname)
        ip_int = struct.unpack("!I", socket.inet_aton(ip_str))[0]
    except (socket.gaierror, OSError) as exc:
        return False, f"DNS resolution failed: {exc}"

    for lo, hi in _PRIVATE_RANGES:
        if lo <= ip_int <= hi:
            return False, f"URL resolves to a private/reserved IP ({ip_str}); SSRF not allowed"

    return True, None


# ---------------------------------------------------------------------------
# Citation parsing (shared by /chat and /chat/stream)
# ---------------------------------------------------------------------------

def parse_citations(
    raw_answer: str,
    docs: "List[Document]",
    fetch_missing_fn=None,
) -> tuple[str, "List[Citation]"]:
    """Parse structured LLM output into an ``(answer_text, citations)`` tuple.

    The LLM is expected to return output in this format::

        Answer:
        <text with [1], [2] citations>

        Citations:
        1. chunk_id: <id>, source: <name>
        2. chunk_id: <id>, source: <name>

    Args:
        raw_answer: The full raw string returned by the LLM.
        docs: The retrieved documents used as context.
        fetch_missing_fn: Optional callable ``(ids: list[str]) -> list[Document]``
            used to hydrate citations whose chunks were not in ``docs``.

    Returns:
        ``(answer_text, citations)`` — both guaranteed to be non-empty on
        success; falls back gracefully to ``(raw_answer, [])`` on parse error.
    """
    # Avoid circular import — schemas only needed at runtime
    from backend.api.schemas import Citation  # noqa: PLC0415

    answer_text: str = raw_answer
    citations: List[Citation] = []

    try:
        # -------------------------------------------------------------------
        # Preferred format (JSON)
        # -------------------------------------------------------------------
        # The LLM is prompted via PydanticOutputParser to return ONLY JSON:
        # {
        #   "answer": "...",
        #   "citations": [{"number": 1, "chunk_id": "...", "source": "..."}]
        # }
        cleaned = (raw_answer or "").strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)
            cleaned = cleaned.strip()

        candidate_json: Optional[str] = None
        if cleaned.startswith("{"):
            candidate_json = cleaned
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate_json = cleaned[start : end + 1].strip()

        if candidate_json and "\"answer\"" in candidate_json and "\"citations\"" in candidate_json:
            try:
                payload = json.loads(candidate_json)
            except Exception:
                cleaned_sanitized = re.sub(
                    r"\\(?![\"\\/bfnrtu])",
                    r"\\\\",
                    candidate_json,
                )
                payload = json.loads(cleaned_sanitized)
            if isinstance(payload, dict) and "answer" in payload and "citations" in payload:
                answer_text = str(payload.get("answer") or "").strip() or raw_answer
                raw_citations = payload.get("citations")

                docs_map = {d.metadata.get("chunk_id"): d for d in docs if d.metadata}

                # Collect missing chunk IDs and hydrate if possible
                missing_ids: list[str] = []
                if isinstance(raw_citations, list):
                    for c in raw_citations:
                        if not isinstance(c, dict):
                            continue
                        cid = str(c.get("chunk_id") or "").strip()
                        if cid and cid not in docs_map:
                            missing_ids.append(cid)

                if missing_ids and fetch_missing_fn is not None:
                    try:
                        fetched = fetch_missing_fn(ids=missing_ids)
                        for d in fetched:
                            cid = d.metadata.get("chunk_id")
                            if cid:
                                docs_map[cid] = d
                    except Exception:
                        pass

                if isinstance(raw_citations, list):
                    for idx, c in enumerate(raw_citations):
                        if not isinstance(c, dict):
                            continue

                        num = c.get("number")
                        try:
                            number = int(num) if num is not None else (idx + 1)
                        except Exception:
                            number = idx + 1

                        cid = str(c.get("chunk_id") or "").strip()
                        src = str(c.get("source") or "").strip()

                        if cid and cid in docs_map:
                            d = docs_map[cid]
                            citations.append(
                                Citation(
                                    number=number,
                                    source=d.metadata.get("file_name") or src,
                                    chunk_id=cid,
                                    page_number=d.metadata.get("page_number"),
                                    text=d.page_content,
                                )
                            )
                        else:
                            citations.append(
                                Citation(
                                    number=number,
                                    source=src,
                                    chunk_id=cid,
                                    text="Source content not available.",
                                )
                            )

                if citations:
                    return answer_text, citations

        # -------------------------------------------------------------------
        # Legacy fallback format (plain text sections)
        # -------------------------------------------------------------------
        if "Answer:" in raw_answer and "Citations:" in raw_answer:
            parts = raw_answer.split("Citations:")
            answer_text = parts[0].replace("Answer:", "").strip()
            citation_lines = parts[1].strip().split("\n")

            docs_map = {d.metadata.get("chunk_id"): d for d in docs if d.metadata}

            # --- first pass: collect chunk IDs ---
            parsed_info: list[tuple[int, str, str]] = []
            missing_ids: list[str] = []
            for line in citation_lines:
                line = line.strip()
                if not line:
                    continue
                m = re.search(r"(\d+)\.\s+chunk_id:\s+([^,]+),\s+source:\s+(.+)", line)
                if m:
                    num = int(m.group(1))
                    cid = m.group(2).strip()
                    src = m.group(3).strip()
                    parsed_info.append((num, cid, src))
                    if cid not in docs_map:
                        missing_ids.append(cid)

            # --- hydrate missing chunks ---
            if missing_ids and fetch_missing_fn is not None:
                try:
                    fetched = fetch_missing_fn(ids=missing_ids)
                    for d in fetched:
                        cid = d.metadata.get("chunk_id")
                        if cid:
                            docs_map[cid] = d
                except Exception:
                    pass  # best-effort — fallback handled below

            # --- build Citation objects ---
            for num, cid, src in parsed_info:
                if cid in docs_map:
                    d = docs_map[cid]
                    citations.append(
                        Citation(
                            number=num,
                            source=d.metadata.get("file_name", src),
                            chunk_id=cid,
                            page_number=d.metadata.get("page_number"),
                            text=d.page_content,
                        )
                    )
                else:
                    citations.append(
                        Citation(number=num, source=src, chunk_id=cid, text="Source content not available.")
                    )
    except Exception:
        # Parsing failed entirely — return raw answer with no citations
        return raw_answer, []

    # Fallback: parser found nothing useful → expose all docs as citations
    if not citations:
        for d in docs:
            md = d.metadata or {}
            citations.append(
                Citation(
                    source=str(md.get("file_name") or md.get("source") or ""),
                    chunk_id=str(md.get("chunk_id") or ""),
                    page_number=md.get("page_number"),
                    text=d.page_content,
                )
            )

    return answer_text, citations
