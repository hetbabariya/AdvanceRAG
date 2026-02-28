from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    embedding_model: str
    upload_dir: str
    bm25_store_dir: str


def load_settings() -> Settings:
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip().strip('"').strip("'")

    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment")
    if not pinecone_api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment")

    project_root = Path(__file__).resolve().parents[2]

    def _resolve_dir(value: str) -> str:
        value = (value or "").strip()
        if not value:
            return str(project_root)
        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str(project_root / p)

    return Settings(
        groq_api_key=groq_api_key,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "langchain-pinecone-hybrid-search"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        upload_dir=_resolve_dir(os.getenv("UPLOAD_DIR", "backend_storage/uploads")),
        bm25_store_dir=_resolve_dir(os.getenv("BM25_STORE_DIR", "backend_storage/bm25")),
    )
