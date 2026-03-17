from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field, HttpUrl


# ========== Health & Status ==========

class HealthResponse(BaseModel):
    status: str = "ok"
    database: Optional[str] = None
    cache: Optional[str] = None


# ========== Authentication ==========

class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)


class UserLogin(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=100)


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime


# ========== Files ==========

class FileItem(BaseModel):
    file_name: str
    chunks_upserted: int
    ingested_at: str
    file_size: Optional[int] = None
    file_hash: Optional[str] = None


class FilesResponse(BaseModel):
    files: list[FileItem]


class IngestResponse(BaseModel):
    file_name: str
    chunks_upserted: int
    file_hash: str
    cached: bool = False  # Whether file was loaded from cache


class IngestManyResponse(BaseModel):
    results: list[IngestResponse]


class URLIngestRequest(BaseModel):
    """Request body for /ingest-url endpoint."""
    url: HttpUrl = Field(description="Public URL to ingest (must be https://). Private IPs are blocked.")


# ========== Chat ==========

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    file_name: str = Field(min_length=1, max_length=260, description="The file to query against.")
    top_k: Optional[int] = Field(default=None, ge=1, le=200)
    use_reranker: bool = True
    rerank_top_k: Optional[int] = Field(default=None, ge=1, le=200)

    model_config = {"populate_by_name": True}


class Citation(BaseModel):
    source: str
    chunk_id: str
    page_number: str | int | None = None
    number: Optional[int] = None  # The citation number [1], [2], etc.
    text: Optional[str] = None    # The actual content of the chunk


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    used_context_chunks: int
    cached: bool = False  # Whether answer was from cache


class ChatHistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    citations: list[Citation]
    file_name: str
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    history: list[ChatHistoryItem]
    total: int


# ========== Errors ==========

class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None


# ========== Cache Stats ==========

class CacheStatsResponse(BaseModel):
    keyspace_hits: int
    keyspace_misses: int
    total_keys: int
    hit_rate: float
