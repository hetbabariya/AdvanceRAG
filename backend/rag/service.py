from __future__ import annotations

import asyncio
import contextlib
import os
import re
import threading
from datetime import datetime
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Tuple

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from pydantic import BaseModel, Field
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

class _LLMResponse(Protocol):
    content: str

class _LLM(Protocol):
    def invoke(self, prompt: str) -> _LLMResponse: ...

    def astream(self, prompt: str) -> AsyncIterator[_LLMResponse]: ...

@dataclass
class _TextResponse:
    content: str

class OpenRouterLLM:
    def __init__(self, *, api_key: str, model: str, base_url: str, reasoning_enabled: bool):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._reasoning_enabled = reasoning_enabled

    def invoke(self, prompt: str) -> _TextResponse:
        kwargs: dict[str, Any] = {}
        if self._reasoning_enabled:
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        msg = resp.choices[0].message
        return _TextResponse(content=(getattr(msg, "content", None) or "").strip())

    async def astream(self, prompt: str) -> AsyncIterator[_TextResponse]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()

        def _run_streaming():
            try:
                kwargs: dict[str, Any] = {}
                if self._reasoning_enabled:
                    kwargs["extra_body"] = {"reasoning": {"enabled": True}}

                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    **kwargs,
                )

                for event in stream:
                    token: Optional[str] = None
                    try:
                        delta = event.choices[0].delta
                        token = getattr(delta, "content", None)
                    except Exception:
                        token = None

                    if token:
                        loop.call_soon_threadsafe(q.put_nowait, _TextResponse(content=str(token)))
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, sentinel)

        t = threading.Thread(target=_run_streaming, daemon=True)
        t.start()

        while True:
            item = await q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item  # type: ignore[misc]

class FallbackLLM:
    def __init__(self, *, primary: _LLM, fallback: _LLM):
        self._primary = primary
        self._fallback = fallback

    def invoke(self, prompt: str) -> _LLMResponse:
        try:
            return self._primary.invoke(prompt)
        except Exception:
            return self._fallback.invoke(prompt)

    async def astream(self, prompt: str) -> AsyncIterator[_LLMResponse]:
        try:
            async for chunk in self._primary.astream(prompt):
                yield chunk
            return
        except Exception:
            async for chunk in self._fallback.astream(prompt):
                yield chunk


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
    llm: _LLM
    # Stores per-doc relevance scores from the most recent rerank() call.
    # Consumed by AgenticRagGraph to compute confidence and gate re-retrieval.
    last_rerank_scores: List[float] = field(default_factory=list)

    @classmethod
    def create(cls, settings: Settings, hybrid: HybridIndex) -> "RagService":
        loader = OptimizedPreprocessedLoader()
        groq_llm = ChatGroq(model=os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"))

        openrouter_key = (os.getenv("OPENROUTER_API_KEY", "") or "").strip().strip('"').strip("'")
        openrouter_model = (os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free") or "").strip()
        openrouter_base_url = (os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") or "").strip()
        reasoning_raw = (os.getenv("OPENROUTER_REASONING_ENABLED", "true") or "").strip().lower()
        reasoning_enabled = reasoning_raw not in {"0", "false", "no", "off"}

        if openrouter_key:
            primary = OpenRouterLLM(
                api_key=openrouter_key,
                model=openrouter_model,
                base_url=openrouter_base_url,
                reasoning_enabled=reasoning_enabled,
            )
            llm: _LLM = FallbackLLM(primary=primary, fallback=groq_llm)
        else:
            llm = groq_llm
        return cls(settings=settings, hybrid=hybrid, loader=loader, llm=llm)

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

        file_name = (file_name or "").strip()
        path = bm25_path(self.settings.bm25_store_dir, file_name)
        if not os.path.exists(path):
            try:
                available = []
                if os.path.isdir(self.settings.bm25_store_dir):
                    available = sorted(os.listdir(self.settings.bm25_store_dir))[:40]
            except Exception:
                available = []
            raise FileNotFoundError(
                f"BM25 model not found for file: {file_name}. Ingest the file first. "
                f"Expected path: {path}. Available BM25 files (first 40): {available}"
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
        file_name = (file_name or "").strip()
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
            self.last_rerank_scores = []
            return []

        # NOTE: sentence-transformers removed. We rerank by cosine similarity
        # using the same embedding function that powers retrieval.
        import math

        def _l2_norm(v: List[float]) -> float:
            return math.sqrt(sum((float(x) * float(x)) for x in v))

        def _cosine(a: List[float], b: List[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            na = _l2_norm(a)
            nb = _l2_norm(b)
            if na == 0.0 or nb == 0.0:
                return 0.0
            dot = sum((float(x) * float(y)) for x, y in zip(a, b))
            return dot / (na * nb)

        query_vec = self.hybrid.embedding.embed_query(query)
        doc_vecs = self.hybrid.embedding.embed_documents([d.page_content for d in docs])

        # Map cosine [-1,1] → [0,1] so it can be treated as a confidence signal.
        sims = [(_cosine(query_vec, v) + 1.0) / 2.0 for v in doc_vecs]

        ranked = sorted(zip(docs, sims), key=lambda x: x[1], reverse=True)[:top_k]
        reranked_docs = [d for d, _ in ranked]
        scores = [float(s) for _, s in ranked]

        # Persist scores — read by AgenticRagGraph._tools_node for confidence gating
        self.last_rerank_scores = scores

        # Inject into metadata so callers / prompts can reference them
        for doc, score in zip(reranked_docs, scores):
            doc.metadata["rerank_score"] = round(score, 4)

        return reranked_docs

    rerank = traceable(name="rag.rerank")(rerank)

    # ------------------------------------------------------------------
    # Query decomposition
    # ------------------------------------------------------------------
    def decompose_query(self, *, question: str) -> List[str]:
        """Break a complex multi-part question into 2–4 focused sub-queries.

        Uses the existing self.llm (OpenRouter → Groq fallback already wired).
        Falls back to a rule-based splitter if the LLM call fails or returns
        fewer than 2 sub-queries.

        Simple, single-concept questions are returned unchanged as [question]
        so the caller can always safely iterate the result.

        Example
        -------
        decompose_query(question="What are the risks and benefits of X,
                                  and how does it compare to Y?")
        → ["What are the risks of X?",
           "What are the benefits of X?",
           "How does X compare to Y?"]
        """
        if not question or not question.strip():
            return [question]

        # Fast-path: skip decomposition for simple questions
        if not self._is_complex_question(question):
            return [question]

        try:
            sub_queries = self._decompose_via_llm(question)
            if len(sub_queries) >= 2:
                return sub_queries
        except Exception:
            logger.debug("decompose_query LLM call failed; using rule-based fallback")

        return self._decompose_rule_based(question)

    decompose_query = traceable(name="rag.decompose_query")(decompose_query)

    # ---- decompose helpers (private) -------------------------------------

    @staticmethod
    def _is_complex_question(question: str) -> bool:
        """Return True if the question likely contains multiple distinct
        information needs and would benefit from being split."""
        q = question.lower()

        # Multiple question marks → almost certainly multi-part
        if question.count("?") > 1:
            return True

        complexity_signals = [
            " and ",
            " as well as ",
            " additionally ",
            " furthermore ",
            " compare ",
            " versus ",
            " vs ",
            " vs. ",
            " difference between ",
            " both ",
            " also ",
        ]
        word_count = len(question.split())
        return word_count > 20 and any(s in q for s in complexity_signals)

    def _decompose_via_llm(self, question: str) -> List[str]:
        """Ask the LLM to produce a numbered list of sub-queries."""
        prompt = (
            "You are a search query optimizer.\n"
            "Break the following question into 2 to 4 concise, independent search queries.\n"
            "Each sub-query must target a single, distinct piece of information.\n\n"
            "STRICT RULES:\n"
            "1. Return ONLY a numbered list — one sub-query per line.\n"
            "2. Do NOT add preamble, explanation, or JSON.\n"
            "3. Do NOT repeat the original question.\n"
            "4. Keep sub-queries short (≤ 15 words each).\n\n"
            f"Question: {question}\n\n"
            "Sub-queries:"
        )

        resp = self.llm.invoke(prompt)
        raw = (getattr(resp, "content", "") or str(resp)).strip()
        sub_queries = self._parse_numbered_list(raw)

        # Sanitise: drop blanks, cap at 4
        sub_queries = [q.strip() for q in sub_queries if q.strip()][:4]

        if len(sub_queries) < 2:
            raise ValueError(f"LLM returned only {len(sub_queries)} sub-queries")

        return sub_queries

    @staticmethod
    def _parse_numbered_list(text: str) -> List[str]:
        """Parse '1. foo\\n2. bar' or '- foo\\n- bar' → ['foo', 'bar']."""
        results = []
        for line in text.splitlines():
            # Strip leading number+dot/paren or bullet
            cleaned = re.sub(r"^\s*\d+[.)]\s*", "", line)
            cleaned = re.sub(r"^\s*[-•*]\s*", "", cleaned).strip()
            if cleaned:
                results.append(cleaned)
        return results

    @staticmethod
    def _decompose_rule_based(question: str) -> List[str]:
        """Heuristic fallback — splits on common conjunctions and '?'."""
        parts = re.split(
            r"\s+and\s+|\s+as well as\s+|\s+additionally[,\s]+|\?\s+",
            question,
            flags=re.IGNORECASE,
        )
        sub_queries = []
        for p in parts:
            p = p.strip().rstrip("?").strip()
            if len(p.split()) >= 3:
                sub_queries.append(p + "?")

        return sub_queries if len(sub_queries) >= 2 else [question]

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
        try:
            async for chunk in self.llm.astream(prompt):
                token = getattr(chunk, "content", str(chunk))
                if token:
                    full_text += token
                    yield token
        except (asyncio.CancelledError, GeneratorExit):
            with contextlib.suppress(Exception):
                return
        except Exception:
            raise
        yield {"__done__": True, "full_text": full_text}

    generate_stream = traceable(name="rag.generate_stream")(generate_stream)