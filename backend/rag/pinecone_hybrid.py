from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import copy
import os
import pickle
import re
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from backend.api.utils import get_logger

logger = get_logger(__name__)


try:
    from langsmith import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator


_BM25_TEMPLATE: Optional[BM25Encoder] = None
_BM25_LOCK = threading.Lock()


def _get_bm25_template() -> BM25Encoder:
    global _BM25_TEMPLATE
    if _BM25_TEMPLATE is not None:
        return _BM25_TEMPLATE
    with _BM25_LOCK:
        if _BM25_TEMPLATE is None:
            _BM25_TEMPLATE = BM25Encoder().default()
    return _BM25_TEMPLATE


def init_bm25_template() -> BM25Encoder:
    """Pre-initialize BM25 template at startup to avoid first-request delay."""
    global _BM25_TEMPLATE
    if _BM25_TEMPLATE is not None:
        logger.info("BM25 template already initialized")
        return _BM25_TEMPLATE
    with _BM25_LOCK:
        if _BM25_TEMPLATE is None:
            logger.info("Initializing BM25 template at startup...")
            _BM25_TEMPLATE = BM25Encoder().default()
            logger.info("BM25 template initialized successfully")
    return _BM25_TEMPLATE


@dataclass
class HybridIndex:
    pc: Pinecone
    index_name: str
    index: Any
    embedding: HuggingFaceEmbeddings


def _safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name[:200] if name else "document"


def ensure_index(
    *,
    api_key: str,
    index_name: str,
    dimension: int,
    metric: str,
    cloud: str,
    region: str,
) -> Pinecone:
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    return pc


def build_hybrid_index(
    *, pinecone_api_key: str, index_name: str, embedding_model: str, cloud: str, region: str
) -> HybridIndex:
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    dim = len(embedding.embed_query("dimension probe"))
    _get_bm25_template()
    pc = ensure_index(
        api_key=pinecone_api_key,
        index_name=index_name,
        dimension=dim,
        metric="dotproduct",
        cloud=cloud,
        region=region,
    )
    index = pc.Index(index_name)
    return HybridIndex(pc=pc, index_name=index_name, index=index, embedding=embedding)


build_hybrid_index = traceable(name="pinecone.build_hybrid_index")(build_hybrid_index)


def bm25_path(store_dir: str, file_name: str) -> str:
    os.makedirs(store_dir, exist_ok=True)
    return os.path.join(store_dir, f"bm25_{_safe_filename(file_name)}.pkl")


def save_bm25(encoder: BM25Encoder, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(encoder, f)


save_bm25 = traceable(name="pinecone.save_bm25")(save_bm25)


def load_bm25(path: str) -> BM25Encoder:
    with open(path, "rb") as f:
        return pickle.load(f)


load_bm25 = traceable(name="pinecone.load_bm25")(load_bm25)


def upsert_documents(
    *,
    hybrid: HybridIndex,
    docs: List[Document],
    file_name: str,
    bm25_store_dir: str,
    batch_size: int = 200,
    max_workers: int = 4,
) -> Tuple[int, str]:
    """Embed and upsert documents into Pinecone with parallel batch upserts.

    Changes vs original:
    * Guards against empty ``docs`` (raises ValueError instead of ZeroDivisionError).
    * Upsert batches are dispatched concurrently via ThreadPoolExecutor.
    * Default ``batch_size`` raised from 100 → 200 to halve round-trips.
    """
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    # --- Guard: empty document list causes BM25 ZeroDivisionError ---
    if not texts:
        raise ValueError(
            f"No text content could be extracted from '{file_name}'. "
            "The document may be empty, password-protected, or the URL returned no readable content."
        )

    t0 = time.perf_counter()

    def _compute_sparse():
        bm25_local = copy.deepcopy(_get_bm25_template())
        bm25_local.fit(texts)
        sparse = bm25_local.encode_documents(texts)
        return bm25_local, sparse

    def _compute_dense():
        return hybrid.embedding.embed_documents(texts)

    logger.info("Ingest compute start: %d chunks (file=%s)", len(texts), file_name)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_sparse = pool.submit(_compute_sparse)
        fut_dense = pool.submit(_compute_dense)
        bm25, all_sparse_vectors = fut_sparse.result()
        all_dense_vectors = fut_dense.result()

    bm25_file = bm25_path(bm25_store_dir, file_name)
    save_bm25(bm25, bm25_file)
    t1 = time.perf_counter()
    logger.info(
        "Embeddings computed (file=%s): dense=%d sparse=%d in %.2fs",
        file_name,
        len(all_dense_vectors),
        len(all_sparse_vectors),
        t1 - t0,
    )

    vectors: List[Dict[str, Any]] = [
        {
            "id": metadata["chunk_id"],
            "values": dense_vector,
            "sparse_values": sparse_vector,
            "metadata": {**metadata, "file_name": file_name, "text": text},
        }
        for text, metadata, dense_vector, sparse_vector in zip(
            texts, metadatas, all_dense_vectors, all_sparse_vectors
        )
    ]

    # --- Parallel upsert batches ---
    batches = [vectors[i: i + batch_size] for i in range(0, len(vectors), batch_size)]
    logger.info("Upserting %d vectors in %d batches (workers=%d)...", len(vectors), len(batches), max_workers)

    def _upsert_batch(batch: List[Dict[str, Any]]) -> int:
        hybrid.index.upsert(vectors=batch)
        return len(batch)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as pool:
        futures = [pool.submit(_upsert_batch, batch) for batch in batches]
        for fut in as_completed(futures):
            fut.result()  # re-raises any exception from the thread

    t2 = time.perf_counter()
    logger.info("Upsert complete: %d vectors for '%s' in %.2fs", len(vectors), file_name, t2 - t1)
    return len(vectors), bm25_file


upsert_documents = traceable(name="pinecone.upsert_documents")(upsert_documents)


def query_hybrid(
    *,
    hybrid: HybridIndex,
    bm25: BM25Encoder,
    query: str,
    file_name: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Query a single file (local mode)."""
    dense_query = hybrid.embedding.embed_query(query)
    sparse_query = bm25.encode_queries([query])[0]

    results = hybrid.index.query(
        vector=dense_query,
        sparse_vector=sparse_query,
        top_k=top_k,
        include_metadata=True,
        include_values=True,
        filter={"file_name": {"$eq": file_name}},
    )
    return list(results.get("matches", []))


query_hybrid = traceable(name="pinecone.query_hybrid")(query_hybrid)


def query_hybrid_global(
    *,
    hybrid: HybridIndex,
    query: str,
    user_file_names: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Query across all of a user's files (global mode).

    Uses dense-only retrieval because there is no single BM25 model that spans
    all files. The ``$in`` filter ensures we never return results belonging to
    other users.
    """
    if not user_file_names:
        return []

    dense_query = hybrid.embedding.embed_query(query)

    results = hybrid.index.query(
        vector=dense_query,
        top_k=top_k,
        include_metadata=True,
        include_values=True,
        filter={"file_name": {"$in": user_file_names}},
    )
    return list(results.get("matches", []))


query_hybrid_global = traceable(name="pinecone.query_hybrid_global")(query_hybrid_global)


def fetch_by_ids(*, hybrid: HybridIndex, ids: List[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    results = hybrid.index.fetch(ids=ids)
    vectors = results.get("vectors", {})
    return list(vectors.values())


fetch_by_ids = traceable(name="pinecone.fetch_by_ids")(fetch_by_ids)
