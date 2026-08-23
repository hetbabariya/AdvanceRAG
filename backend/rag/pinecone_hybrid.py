from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import copy
import os
import pickle
import re
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document
import httpx
import numpy as np

from backend.api.utils import get_logger

logger = get_logger(__name__)


class BaseEmbeddings(Protocol):
    def embed_query(self, text: str) -> List[float]: ...
    def embed_documents(self, texts: List[str], progress_cb: Optional[Callable[[int, int], None]] = None) -> List[List[float]]: ...


# BGE retrieval models (bge-*-en-v1.5) are trained with a query-side instruction.
# Applying it ONLY to queries (never passages) measurably improves retrieval.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


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
            t0 = time.perf_counter()
            _BM25_TEMPLATE = BM25Encoder().default()
            dt = time.perf_counter() - t0
            logger.info("BM25 template initialized successfully (%.2fs)", dt)
    return _BM25_TEMPLATE


@dataclass
class HybridIndex:
    pc: Pinecone
    index_name: str
    index: Any
    embedding: BaseEmbeddings


def _mean_pool(token_embeddings) -> List[float]:
    """Mean-pool token-level embeddings [tokens][dim] → [dim] (vectorized)."""
    arr = np.asarray(token_embeddings, dtype=np.float32)
    if arr.size == 0 or arr.ndim != 2:
        return []
    return arr.mean(axis=0).tolist()


def _wants_bge_query_prefix(model_name: str) -> bool:
    if (os.getenv("EMBEDDING_BGE_QUERY_PREFIX", "") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return "bge" in (model_name or "").lower()


class LocalEmbeddings:
    """Sentence-transformers embeddings running locally.

    Uses L2-normalized vectors so Pinecone's dotproduct metric is equivalent
    to cosine similarity. Applies the BGE query instruction to queries only.
    """

    def __init__(self, *, model: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self._model_name = (model or "").strip()
        if not self._model_name:
            raise ValueError("Local embedding model name is empty")
        self._batch_size = max(1, int(batch_size))
        logger.info("Loading local embedding model '%s'...", self._model_name)
        t0 = time.perf_counter()
        # device=None → sentence-transformers auto-selects cuda if available
        self._model = SentenceTransformer(self._model_name)
        logger.info("Embedding model loaded in %.2fs (device=%s)", time.perf_counter() - t0, self._model.device)

    def embed_query(self, text: str) -> List[float]:
        if _wants_bge_query_prefix(self._model_name):
            text = f"{BGE_QUERY_INSTRUCTION}{text}"
        vec = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32).tolist()

    def embed_documents(self, texts: List[str], progress_cb: Optional[Callable[[int, int], None]] = None) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        total = len(texts)
        done = 0
        for i in range(0, total, self._batch_size):
            batch = texts[i : i + self._batch_size]
            emb = self._model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            out.extend(np.asarray(emb, dtype=np.float32).tolist())
            done += len(batch)
            if progress_cb:
                try:
                    progress_cb(done, total)
                except Exception:
                    pass
        return out


class HFInferenceEmbeddings:
    """Fallback provider — Hugging Face Inference API over HTTP."""
    def __init__(
        self,
        *,
        model: str,
        api_token: str,
        timeout_s: float = 60.0,
        batch_size: int = 16,
    ):
        self._model = (model or "").strip()
        if not self._model:
            raise ValueError("HF embedding model is empty")
        self._token = (api_token or "").strip()
        if not self._token:
            logger.warning("HF_API_TOKEN missing, embeddings will be skipped")
        self._timeout_s = timeout_s
        self._batch_size = max(1, int(batch_size))

    def _url(self) -> str:
        # Prefer HF router endpoint (recommended by HF for unified inference routing)
        return f"https://router.huggingface.co/hf-inference/models/{self._model}/pipeline/feature-extraction"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _embed_one_or_many(self, inputs: List[str]) -> List[List[float]]:
        if not self._token:
            return [[] for _ in inputs]

        payload: Dict[str, Any]
        if len(inputs) == 1:
            payload = {"inputs": inputs[0]}
        else:
            payload = {"inputs": inputs}

        with httpx.Client(timeout=self._timeout_s) as client:
            resp = client.post(self._url(), headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()

        def _to_vec(item: Any) -> List[float]:
            # HF feature-extraction returns either:
            # - pooled embedding: [dim]
            # - token embeddings: [tokens][dim]
            if not isinstance(item, list):
                return []
            if item and isinstance(item[0], list):
                return _mean_pool(item)  # token-level → mean pooled
            return [float(x) for x in item]

        if len(inputs) == 1:
            return [_to_vec(data)]
        if isinstance(data, list):
            return [_to_vec(x) for x in data]
        return [[] for _ in inputs]

    def embed_query(self, text: str) -> List[float]:
        if _wants_bge_query_prefix(self._model):
            text = f"{BGE_QUERY_INSTRUCTION}{text}"
        vecs = self._embed_one_or_many([text])
        return vecs[0] if vecs else []

    def embed_documents(self, texts: List[str], progress_cb: Optional[Callable[[int, int], None]] = None) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        total = len(texts)
        done = 0
        for i in range(0, total, self._batch_size):
            out.extend(self._embed_one_or_many(texts[i : i + self._batch_size]))
            done += min(self._batch_size, total - done)
            if progress_cb:
                try:
                    progress_cb(done, total)
                except Exception:
                    pass
        return out


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
    else:
        existing_dim = int(getattr(pc.describe_index(index_name), "dimension", 0) or 0)
        if existing_dim and existing_dim != int(dimension):
            raise RuntimeError(
                f"Pinecone index '{index_name}' has dimension {existing_dim} but embedding model "
                f"produces {dimension}. The embedding model changed — delete the old index in the "
                "Pinecone console (or point PINECONE_INDEX_NAME at a new one) and re-ingest your files."
            )
    return pc


def build_hybrid_index(
    *, pinecone_api_key: str, index_name: str, embedding_model: str, cloud: str, region: str
) -> HybridIndex:
    provider = (os.getenv("EMBEDDING_PROVIDER", "") or "").strip().lower() or "local"
    model = (embedding_model or "").strip()

    if provider == "hf":
        logger.info("Initializing HF Inference Embeddings (model=%s)...", model)
        embedding: BaseEmbeddings = HFInferenceEmbeddings(
            model=model,
            api_token=(os.getenv("HF_API_TOKEN", "") or "").strip(),
            timeout_s=float(os.getenv("HF_HTTP_TIMEOUT", "60") or "60"),
            batch_size=int(os.getenv("HF_EMBED_BATCH_SIZE", "16") or "16"),
        )
    else:
        if not model:
            model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
        embedding = LocalEmbeddings(
            model=model,
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64") or "64"),
        )

    logger.info("Probing embedding dimension...")
    dim = len(embedding.embed_query("dimension probe"))
    if dim <= 0:
        raise ValueError(
            "Embedding model returned an empty vector while probing dimension. "
            f"Check provider='{provider}' and model='{model}'."
        )
    logger.info("Embedding dimension: %d", dim)

    logger.info("Connecting to Pinecone...")
    pc = ensure_index(
        api_key=pinecone_api_key,
        index_name=index_name,
        dimension=dim,
        metric="dotproduct",
        cloud=cloud,
        region=region,
    )
    logger.info("Pinecone index ready: %s", index_name)

    index = pc.Index(index_name)
    return HybridIndex(pc=pc, index_name=index_name, index=index, embedding=embedding)


build_hybrid_index = traceable(name="pinecone.build_hybrid_index")(build_hybrid_index)


def bm25_path(store_dir: str, file_name: str) -> str:
    os.makedirs(store_dir, exist_ok=True)
    return os.path.join(store_dir, f"bm25_{_safe_filename(file_name)}.pkl")


def bm25_keyed_path(store_dir: str, file_name: str, content_hash: Optional[str]) -> str:
    """Collision-safe pickle name: two files with the same display name but
    different content (or different users) never overwrite each other."""
    os.makedirs(store_dir, exist_ok=True)
    h = (content_hash or "").strip()[:12]
    suffix = f"_{h}" if h else ""
    return os.path.join(store_dir, f"bm25_{_safe_filename(file_name)}{suffix}.pkl")


def resolve_bm25_path(store_dir: str, file_name: str, content_hash: Optional[str] = None) -> Optional[str]:
    """Find the BM25 pickle for a file.

    Resolution order:
      1. Exact hash-keyed pickle  bm25_{name}_{hash12}.pkl
      2. Legacy exact name        bm25_{name}.pkl
      3. Newest glob match        bm25_{name}_*.pkl
    """
    if not file_name:
        return None

    candidates: List[str] = []
    if content_hash:
        candidates.append(bm25_keyed_path(store_dir, file_name, content_hash))
    legacy = bm25_path(store_dir, file_name)
    if legacy not in candidates:
        candidates.append(legacy)

    for path in candidates:
        if os.path.exists(path):
            return path

    # Glob fallback — newest wins
    safe = _safe_filename(file_name)
    pattern = os.path.join(store_dir, f"bm25_{safe}_*.pkl")
    try:
        import glob as _glob

        matches = _glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    except Exception:
        pass
    return None


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
    bm25_key: Optional[str] = None,
    user_id: Optional[int] = None,
    batch_size: int = 200,
    max_workers: int = 4,
    stage_cb: Optional[Callable[[str], None]] = None,
    embed_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[int, str]:
    """Embed and upsert documents into Pinecone with parallel batch upserts.

    ``stage_cb`` receives coarse progress stages ("embedding", "upserting");
    ``bm25_key`` (usually the content hash) makes the BM25 pickle name
    collision-safe across users and re-uploads with the same display name;
    ``user_id`` stamps every vector so queries can be scoped per owner.
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
        return hybrid.embedding.embed_documents(texts, progress_cb=embed_progress)

    logger.info("Ingest compute start: %d chunks (file=%s)", len(texts), file_name)

    if stage_cb:
        try:
            stage_cb("embedding")
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_sparse = pool.submit(_compute_sparse)
        fut_dense = pool.submit(_compute_dense)
        bm25, all_sparse_vectors = fut_sparse.result()
        all_dense_vectors = fut_dense.result()

    bm25_file = bm25_keyed_path(bm25_store_dir, file_name, bm25_key) if bm25_key else bm25_path(bm25_store_dir, file_name)
    save_bm25(bm25, bm25_file)
    t1 = time.perf_counter()
    logger.info(
        "Embeddings computed (file=%s): dense=%d sparse=%d in %.2fs",
        file_name,
        len(all_dense_vectors),
        len(all_sparse_vectors),
        t1 - t0,
    )

    extra_meta: Dict[str, Any] = {"file_name": file_name}
    if user_id is not None:
        extra_meta["user_id"] = int(user_id)

    vectors: List[Dict[str, Any]] = []
    for text, metadata, dense_vector, sparse_vector in zip(
        texts, metadatas, all_dense_vectors, all_sparse_vectors
    ):
        meta = {**metadata, **extra_meta, "text": text}
        vectors.append(
            {
                "id": metadata["chunk_id"],
                "values": dense_vector,
                "sparse_values": sparse_vector,
                "metadata": meta,
            }
        )

    # --- Parallel upsert batches ---
    batches = [vectors[i: i + batch_size] for i in range(0, len(vectors), batch_size)]
    logger.info("Upserting %d vectors in %d batches (workers=%d)...", len(vectors), len(batches), max_workers)

    if stage_cb:
        try:
            stage_cb("upserting")
        except Exception:
            pass

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


def _scoped_filter(file_name: Optional[str], user_id: Optional[int]) -> Dict[str, Any]:
    """Build a Pinecone metadata filter scoped to one user (and optionally file).

    Vectors ingested before user-scoping lack the ``user_id`` key — run
    scripts/backfill_vector_user_ids.py once, otherwise legacy vectors become
    invisible to filtered queries.
    """
    f: Dict[str, Any] = {}
    if file_name:
        f["file_name"] = {"$eq": file_name}
    if user_id is not None:
        f["user_id"] = {"$eq": int(user_id)}
    return f


def query_hybrid(
    *,
    hybrid: HybridIndex,
    bm25: BM25Encoder,
    query: str,
    file_name: str,
    top_k: int,
    user_id: Optional[int] = None,
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
        filter=_scoped_filter(file_name, user_id),
    )
    return list(results.get("matches", []))


query_hybrid = traceable(name="pinecone.query_hybrid")(query_hybrid)


def query_hybrid_global(
    *,
    hybrid: HybridIndex,
    query: str,
    user_id: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Query across all of a user's files (global mode).

    Uses dense-only retrieval because there is no single BM25 model that spans
    all files. Filtering on ``user_id`` (rather than a list of file names)
    guarantees results can never cross users, even when two users ingest files
    with identical display names.
    """
    if user_id is None:
        return []

    dense_query = hybrid.embedding.embed_query(query)

    results = hybrid.index.query(
        vector=dense_query,
        top_k=top_k,
        include_metadata=True,
        include_values=True,
        filter=_scoped_filter(None, user_id),
    )
    return list(results.get("matches", []))


query_hybrid_global = traceable(name="pinecone.query_hybrid_global")(query_hybrid_global)


def delete_file_vectors(
    *,
    hybrid: HybridIndex,
    file_name: str,
    user_id: int,
) -> int:
    """Delete all vectors for a user's file. Best-effort; returns nothing useful.

    Pinecone's delete-by-filter does not report how many vectors were removed,
    so we read index stats before/after as an approximation.
    """
    flt = _scoped_filter(file_name, user_id)
    before = 0
    try:
        stats = hybrid.index.describe_index_stats()
        before = int(((stats or {}).get("namespaces", {}) or {}).get("", {}).get("vector_count", 0) or 0)
    except Exception:
        pass

    hybrid.index.delete(filter=flt)

    after = before
    try:
        time.sleep(0.5)
        stats = hybrid.index.describe_index_stats()
        after = int(((stats or {}).get("namespaces", {}) or {}).get("", {}).get("vector_count", 0) or 0)
    except Exception:
        pass
    deleted = max(0, before - after)
    logger.info("Deleted ~%d vectors for '%s' (user %d)", deleted, file_name, user_id)
    return deleted


def fetch_by_ids(*, hybrid: HybridIndex, ids: List[str]) -> List[Dict[str, Any]]:
    if not ids:
        return []
    results = hybrid.index.fetch(ids=ids)
    vectors = results.get("vectors", {})
    return list(vectors.values())


fetch_by_ids = traceable(name="pinecone.fetch_by_ids")(fetch_by_ids)
