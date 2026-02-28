from __future__ import annotations

import os
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from backend.rag.loader import OptimizedPreprocessedLoader
from backend.rag.pinecone_hybrid import (
    HybridIndex,
    bm25_path,
    load_bm25,
    query_hybrid,
    query_hybrid_global,
    upsert_documents,
)
from backend.rag.settings import Settings
from backend.api.utils import get_logger

logger = get_logger(__name__)


def _truthy_env(name: str) -> bool:
    return (os.getenv(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _dump_chunks(*, docs: List[Document], file_name: str) -> None:
    max_chars = _int_env("PRINT_CHUNKS_MAX_CHARS", 0)
    report_path = (os.getenv("CHUNK_REPORT_PATH", "") or "").strip()

    lines: List[str] = []
    if report_path:
        lines.append(f"# Chunk Report: {file_name}")
        lines.append("")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")
        lines.append(f"Total chunks: {len(docs)}")
        lines.append("")

    for i, d in enumerate(docs):
        md = d.metadata or {}
        header = (
            f"CHUNK {i+1}/{len(docs)} | chunk_id={md.get('chunk_id')} | "
            f"section={md.get('section_index')} | title={md.get('section_title')} | "
            f"level={md.get('section_level')} | page={md.get('page_number')}"
        )
        text = (d.page_content or "").strip()
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n[...truncated...]"

        logger.info("\n%s\n%s\n%s", "=" * 120, header, text)

        if report_path:
            lines.append("---")
            lines.append("")
            lines.append(f"## {i+1}. {md.get('section_title') or ''}")
            lines.append("")
            lines.append("```json")
            safe_md = {
                "chunk_index": md.get("chunk_index"),
                "chunk_id": md.get("chunk_id"),
                "section_index": md.get("section_index"),
                "section_title": md.get("section_title"),
                "section_level": md.get("section_level"),
                "section_id": md.get("section_id"),
                "page_number": md.get("page_number"),
                "source": md.get("source"),
                "file_type": md.get("file_type"),
            }
            import json

            lines.append(json.dumps(safe_md, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
            lines.append("```text")
            lines.append(text)
            lines.append("```")

    if report_path:
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info("Chunk report written: %s", report_path)
        except Exception:
            logger.exception("Failed writing chunk report: %s", report_path)


try:
    from langsmith import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator


try:
    from langchain.output_parsers import PydanticOutputParser  # type: ignore
except Exception:  # pragma: no cover
    from langchain_core.output_parsers import PydanticOutputParser  # type: ignore

try:
    from langchain.prompts import PromptTemplate  # type: ignore
except Exception:  # pragma: no cover
    from langchain_core.prompts import PromptTemplate  # type: ignore


class _LLMCitation(BaseModel):
    number: int = Field(ge=1)
    chunk_id: str
    source: str


class _LLMAnswer(BaseModel):
    answer: str
    citations: list[_LLMCitation]

# ---------------------------------------------------------------------------
# BM25 LRU in-memory cache (avoids re-loading from disk on every query)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _cached_load_bm25(bm25_file_path: str):
    """Load and cache a per-file BM25 model (LRU, max 32 slots)."""
    logger.debug("Loading BM25 from disk: %s", bm25_file_path)
    return load_bm25(bm25_file_path)


@dataclass
class RagService:
    settings: Settings
    hybrid: HybridIndex
    loader: OptimizedPreprocessedLoader
    llm: ChatGroq
    reranker: CrossEncoder

    @classmethod
    def create(cls, settings: Settings, hybrid: HybridIndex) -> "RagService":
        loader = OptimizedPreprocessedLoader()
        llm = ChatGroq(model=os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"))
        reranker = CrossEncoder(os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        return cls(settings=settings, hybrid=hybrid, loader=loader, llm=llm, reranker=reranker)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_file(self, *, saved_path: str, original_name: str) -> Tuple[int, str]:
        """Parse, embed, and upsert a file. Runs in a thread (CPU/IO-bound)."""
        docs = self.loader.load_and_split_file(saved_path, original_name)

        if _truthy_env("PRINT_CHUNKS"):
            _dump_chunks(docs=docs, file_name=original_name)

        count, bm25_file = upsert_documents(
            hybrid=self.hybrid,
            docs=docs,
            file_name=original_name,
            bm25_store_dir=self.settings.bm25_store_dir,
        )
        return count, bm25_file

    ingest_file = traceable(name="rag.ingest_file")(ingest_file)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _query_rewrite_enabled() -> bool:
        value = (os.getenv("QUERY_REWRITE_ENABLED", "true") or "").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def rewrite_query(self, *, question: str, file_name: str) -> str:
        if not question or not question.strip():
            return question
        if not self._query_rewrite_enabled():
            return question

        prompt = (
            "You are a search query optimizer for document retrieval.\n\n"

    "Your task is to rewrite the user's question into ONE optimized search query "
    "that will retrieve relevant information from a SINGLE document.\n\n"

    "### OBJECTIVE\n"
    "Create a concise, semantically rich query suitable for vector or hybrid search.\n\n"

    "### STRICT RULES\n"
    "1. Return ONLY the rewritten query text.\n"
    "2. Do NOT return JSON.\n"
    "3. Do NOT add explanations.\n"
    "4. Do NOT add quotes.\n"
    "5. Keep the original language unchanged.\n"
    "6. Remove filler words and conversational phrases.\n"
    "7. Expand obvious acronyms when helpful (e.g., 'ML' → 'machine learning').\n"
    "8. Preserve key technical terms exactly as written.\n"
    "9. Do NOT introduce new facts not implied by the question.\n"
    "10. If the question is already clear and optimized, return it unchanged.\n\n"

    "### OPTIMIZATION GUIDELINES\n"
    "- Focus on core entities, concepts, and constraints.\n"
    "- Convert questions into keyword-focused semantic queries.\n"
    "- Remove phrases like: 'can you explain', 'please tell me', 'I want to know'.\n"
    "- Keep important qualifiers (dates, versions, conditions, comparisons).\n"
    "- Prefer noun phrases over full sentences when possible.\n\n"

    f"Document Name: {file_name}\n"
    f"User Question: {question}\n\n"

    "Rewritten Search Query:"
        )

        try:
            resp = self.llm.invoke(prompt)
            text = (getattr(resp, "content", "") or str(resp)).strip()
            text = " ".join(text.split())
            if not text:
                return question
            if len(text) > 500:
                return question
            return text
        except Exception:
            return question

    rewrite_query = traceable(name="rag.rewrite_query")(rewrite_query)

    def _get_bm25_for_file(self, file_name: str):
        path = bm25_path(self.settings.bm25_store_dir, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BM25 model not found for file: {file_name}. Ingest the file first."
            )
        return _cached_load_bm25(path)

    def retrieve(
        self,
        *,
        query: str,
        file_name: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Local retrieval — scoped to a single file (hybrid dense+BM25)."""
        bm25 = self._get_bm25_for_file(file_name)
        return query_hybrid(
            hybrid=self.hybrid,
            bm25=bm25,
            query=query,
            file_name=file_name,
            top_k=top_k,
        )

    retrieve = traceable(name="rag.retrieve")(retrieve)

    def retrieve_global(
        self,
        *,
        query: str,
        user_file_names: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Global retrieval — searches across all of the user's ingested files.

        Uses dense-only retrieval (no BM25) because there is no single BM25
        model that spans multiple files. The ``$in`` filter ensures we never
        surface another user's documents.
        """
        return query_hybrid_global(
            hybrid=self.hybrid,
            query=query,
            user_file_names=user_file_names,
            top_k=top_k,
        )

    def fetch_chunks_by_ids(self, *, ids: List[str]) -> List[Document]:
        from backend.rag.pinecone_hybrid import fetch_by_ids
        matches = fetch_by_ids(hybrid=self.hybrid, ids=ids)
        return self._matches_to_documents(matches)

    def _matches_to_documents(self, matches: List[Dict[str, Any]]) -> List[Document]:
        docs: List[Document] = []
        for m in matches:
            md = m.get("metadata") or {}
            docs.append(Document(page_content=md.get("text", ""), metadata=md))
        return docs

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    def rerank(self, *, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [d for d, _ in reranked[:top_k]]

    rerank = traceable(name="rag.rerank")(rerank)

    # ------------------------------------------------------------------
    # Prompt builder (single source of truth for generate + generate_stream)
    # ------------------------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=1)
    def _output_parser() -> PydanticOutputParser:
        return PydanticOutputParser(pydantic_object=_LLMAnswer)

    @staticmethod
    def _build_prompt(question: str, context_docs: List[Document]) -> str:
        context = "\n\n".join(
            f"{d.page_content}\n\nMetadata: {d.metadata}"
            for d in context_docs
            if d.page_content and d.page_content.strip()
        )
        parser = RagService._output_parser()
        template = PromptTemplate(
            template=(
                "You are a document-grounded AI assistant.\n\n"
                "Answer the user's question using ONLY the information provided in <context>.\n\n"
                "Strict rules:\n"
                "1. Do NOT use outside knowledge.\n"
                "2. Do NOT hallucinate.\n"
                "3. If the answer is not present, respond: \"I don't know based on the provided documents.\"\n"
                "4. Every factual statement MUST have a citation.\n"
                "5. Citations in the answer must use [number] like [1], [2].\n"
                "6. The 'answer' field MUST be structured Markdown with Citations, not a single long paragraph.\n"
                "7. Use headings and bullets when helpful.\n"
                "8. Each citation object must include: number, chunk_id, source.\n\n"
                "Return ONLY a valid JSON object matching the schema.\n"
                "{format_instructions}\n\n"
                "<context>\n{context}\n</context>\n\n"
                "User Question:\n{question}\n"
            ),
            input_variables=["question", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return template.format(question=question, context=context)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, *, question: str, context_docs: List[Document]) -> str:
        prompt = self._build_prompt(question, context_docs)
        resp = self.llm.invoke(prompt)
        return getattr(resp, "content", str(resp))

    generate = traceable(name="rag.generate")(generate)

    async def generate_stream(self, *, question: str, context_docs: List[Document]):
        """Async generator — yields str tokens then a ``{"__done__": True}`` sentinel."""
        prompt = self._build_prompt(question, context_docs)
        full_text = ""
        async for chunk in self.llm.astream(prompt):
            token = getattr(chunk, "content", str(chunk))
            if token:
                full_text += token
                yield token
        yield {"__done__": True, "full_text": full_text}

    generate_stream = traceable(name="rag.generate_stream")(generate_stream)
