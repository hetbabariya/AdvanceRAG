"""
Agentic RAG Graph — Fixed & Optimized
======================================
Bugs fixed vs previous version (see logs):

  BUG 1 — decompose_query called twice (steps 1 & 2):
    Root cause: after decompose succeeded, the hint showed sub_queries as a
    truncated Python repr string. The LLM couldn't confirm decomposition worked
    and retried it. Fix: hint now explicitly confirms decomposition status with
    a numbered list and tells the LLM exactly what to do next.

  BUG 2 — step 3 stalls with tool_calls=[] (49 s timeout):
    Root cause: some providers (especially OpenRouter free-tier) respond with
    XML-style function calls:
        <function=retrieve><parameter=query>…</parameter></function>
    The old _parse_text_tool_calls only matched `name(key=val)` syntax.
    Fix: parser now handles both styles plus JSON block style.

  BUG 3 — file_name taken from LLM args (security / correctness):
    LLM was passing file_name in retrieve args, which could be wrong or
    hallucinated. Fix: always use state["file_name"] — never trust the model.

  BUG 4 — decompose fallback returned [question] (single item list):
    When decompose_query returned < 2 items the old code set sub_queries=[q]
    but still proceeded to _retrieve_parallel, which ran a single-item "parallel"
    loop. Fix: if decompose yields only 1 query, skip parallel path entirely.

Additional improvements:
  - Smarter _agent_node: after a validation error, explicitly injects a
    corrective directive so the LLM knows what to call instead.
  - _route: handles text-tool-call path directly (no double-parse).
  - Removed unused `lru_cache`, `Literal`, `asyncio` imports.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from backend.api.utils import get_logger
from backend.rag.service import RagService

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _int_env(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or "").strip())
    except (ValueError, AttributeError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float((os.getenv(name) or "").strip())
    except (ValueError, AttributeError):
        return default


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _pick_groq_model(raw: str) -> str:
    name = (raw or "").strip()
    if not name:
        return "llama-3.3-70b-versatile"
    low = name.lower()
    forbidden = ("openai", "anthropic", "google", "mistral/")
    if any(low.startswith(p) for p in forbidden):
        return "llama-3.3-70b-versatile"
    if "/" in low and not any(low.startswith(p) for p in ("llama", "gemma", "mixtral")):
        return "llama-3.3-70b-versatile"
    return name


def _openrouter_key() -> str:
    return (os.getenv("OPENROUTER_API_KEY") or "").strip().strip('"').strip("'")


def _truncate(value: object, max_len: int = 180) -> str:
    s = " ".join(str(value).split())
    return s if len(s) <= max_len else s[: max(0, max_len - 1)] + "…"


def _fmt_kv(pairs: Dict[str, object]) -> str:
    return " ".join(f"{k}={_truncate(v)}" for k, v in pairs.items() if v is not None)


def _doc_preview(d: Document) -> str:
    try:
        md = d.metadata or {}
        return _fmt_kv({
            "chunk_id": md.get("chunk_id") or md.get("id"),
            "source": md.get("source") or md.get("file_name"),
            "rerank_score": md.get("rerank_score"),
            "text": _truncate((d.page_content or "").strip(), 140),
        })
    except Exception:
        return _truncate(getattr(d, "page_content", ""), 140)


# ---------------------------------------------------------------------------
# FIX BUG 2 — robust text-tool-call parser
# Handles all three response styles seen in the wild:
#   Style A (function-call syntax):  retrieve(query="…", file_name="…", top_k=5)
#   Style B (XML parameter tags):    <function=retrieve><parameter=query>…</parameter>…</function>
#   Style C (JSON block):            {"name": "retrieve", "arguments": {…}}
# ---------------------------------------------------------------------------

def _parse_text_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Best-effort fallback when the model returns tool calls as plain text."""
    raw = (text or "").strip()
    if not raw:
        return []

    _TOOL_NAMES = {"retrieve", "rerank", "rewrite_query", "decompose_query", "answer"}

    # --- Style B: <function=NAME><parameter=KEY>VALUE</parameter>…</function> ---
    xml_match = re.search(
        r"<function=(\w+)>(.*?)</function>",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if xml_match:
        name = xml_match.group(1).strip()
        if name in _TOOL_NAMES:
            body = xml_match.group(2) or ""
            args: Dict[str, Any] = {}
            for pm in re.finditer(r"<parameter=(\w+)>(.*?)</parameter>", body, re.DOTALL):
                k = pm.group(1).strip()
                v = pm.group(2).strip()
                # Coerce numeric strings
                if re.fullmatch(r"\d+", v):
                    args[k] = int(v)
                else:
                    args[k] = v
            return [{"name": name, "args": args, "id": "xml_call", "type": "tool_call"}]

    # --- Style C: JSON block {"name": …, "arguments": {…}} ---
    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*\}', raw, re.DOTALL)
    if json_match:
        import json as _json
        try:
            obj = _json.loads(json_match.group(0))
            name = str(obj.get("name") or "").strip()
            if name in _TOOL_NAMES:
                args = obj.get("arguments") or obj.get("args") or {}
                return [{"name": name, "args": args, "id": "json_call", "type": "tool_call"}]
        except Exception:
            pass

    # --- Style A: function_name(key=value, …) ---
    fn_match = re.search(
        r"\b(retrieve|rerank|rewrite_query|decompose_query|answer)\s*\((.*?)\)",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if fn_match:
        name = fn_match.group(1).strip()
        args_blob = fn_match.group(2) or ""
        args = {}
        for kv in re.finditer(
            r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|\d+|true|false|null)',
            args_blob,
            re.IGNORECASE,
        ):
            k = kv.group(1)
            v = kv.group(2).strip()
            if v.lower() == "null":
                args[k] = None
            elif v.lower() in {"true", "false"}:
                args[k] = v.lower() == "true"
            elif re.fullmatch(r"\d+", v):
                args[k] = int(v)
            else:
                args[k] = v.strip('"\'')
        return [{"name": name, "args": args, "id": "text_call", "type": "tool_call"}]

    return []


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    question: str
    file_name: str
    query: str
    sub_queries: List[str]
    decompose_attempted: bool
    query_rewrites: int
    docs: List[Document]
    doc_scores: List[float]
    retrieved: bool
    retrieval_attempts: int
    steps: int
    done: bool
    phase: str                   # 'init' | 'retrieve' | 'rerank' | 'answer'
    last_tool_error: Optional[str]
    last_validation_directive: Optional[str]   # FIX BUG 1: corrective hint
    confidence: float


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

@tool("rewrite_query")
def tool_rewrite_query(question: str, file_name: str) -> str:
    """Rewrite the user question into a tighter search query.
    Call AT MOST ONCE, only if the question is genuinely ambiguous."""
    raise RuntimeError("Schema-only")


@tool("decompose_query")
def tool_decompose_query(question: str) -> str:
    """Break a complex multi-part question into 2–4 focused sub-queries.
    Call ONLY ONCE when the question has clearly distinct information needs."""
    raise RuntimeError("Schema-only")


@tool("retrieve")
def tool_retrieve(query: str, file_name: str, top_k: int) -> str:
    """Retrieve top-k passages via hybrid (dense+BM25) search.
    MUST be called before rerank or answer."""
    raise RuntimeError("Schema-only")


@tool("rerank")
def tool_rerank(query: str, top_k: int) -> str:
    """Rerank retrieved passages by relevance. Call after retrieve."""
    raise RuntimeError("Schema-only")


@tool("answer")
def tool_answer() -> str:
    """Finalise the run. Call ONLY after at least one successful retrieve."""
    raise RuntimeError("Schema-only")


_ALL_TOOLS = [tool_rewrite_query, tool_decompose_query, tool_retrieve, tool_rerank, tool_answer]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an agentic RAG controller. Your ONLY job is to call the next tool.
NEVER respond with plain text. ALWAYS call exactly one tool.

SEQUENCE (follow strictly):
  1. Complex/multi-part question?
       YES → call decompose_query ONCE, then go to step 2.
       NO  → go to step 2.
  2. Query ambiguous for search?
       YES → call rewrite_query ONCE, then call retrieve.
       NO  → call retrieve directly.
  3. retrieve returned > 0 chunks?
       YES → call rerank, then call answer.
       NO  → call rewrite_query (if not used), then retrieve again.
  4. After rerank, confidence reported?
       LOW  → call retrieve again (max 2 total attempts).
       OK   → call answer.

ABSOLUTE RULES — violations are blocked:
  • decompose_query : call at most ONCE per run
  • rewrite_query   : call at most ONCE per run
  • retrieve        : call at most TWICE per run
  • answer          : NEVER before retrieve succeeds
  • NEVER produce plain text — always call a tool
"""

_CONFIDENCE_THRESHOLD = _float_env("AGENT_CONFIDENCE_THRESHOLD", 0.35)
_ADAPTIVE_TOP_K_BOOST = _int_env("AGENT_ADAPTIVE_TOP_K_BOOST", 5)


# ---------------------------------------------------------------------------
# Main graph
# ---------------------------------------------------------------------------

class AgenticRagGraph:
    def __init__(
        self,
        *,
        service: RagService,
        retrieval_top_k: int,
        rerank_top_k: int,
        max_steps: Optional[int] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._service = service
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k
        self._max_steps = max_steps if max_steps is not None else _int_env("AGENT_MAX_STEPS", 8)
        self._on_event = on_event

        groq_model = _pick_groq_model(os.getenv("GROQ_AGENT_MODEL", "") or "")
        self._groq_llm = ChatGroq(model=groq_model).bind_tools(_ALL_TOOLS)

        openrouter_key = _openrouter_key()
        self._use_openrouter = bool(openrouter_key)
        if self._use_openrouter:
            self._openrouter_llm = ChatOpenAI(
                api_key=openrouter_key,
                base_url=(os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip(),
                model=(os.getenv("OPENROUTER_AGENT_MODEL") or "stepfun/step-3.5-flash:free").strip(),
            ).bind_tools(_ALL_TOOLS)

        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, *, question: str, file_name: str) -> AgentState:
        state: AgentState = {
            "messages": [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=f"Document: {file_name}\nQuestion: {question}\n\nCall the first tool now."),
            ],
            "question": question,
            "file_name": file_name,
            "query": question,
            "sub_queries": [],
            "decompose_attempted": False,
            "query_rewrites": 0,
            "docs": [],
            "doc_scores": [],
            "retrieved": False,
            "retrieval_attempts": 0,
            "steps": 0,
            "done": False,
            "phase": "init",
            "last_tool_error": None,
            "last_validation_directive": None,
            "confidence": 0.0,
        }
        return self._graph.invoke(state)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    def _emit(self, payload: Dict[str, Any]) -> None:
        payload.setdefault("ts", time.time())
        if self._on_event:
            try:
                self._on_event(payload)
            except Exception:
                pass
        if _truthy_env("AGENT_LOG_EVENTS", False):
            try:
                logger.info("AGENT %s", _fmt_kv({k: v for k, v in payload.items() if k != "ts"}))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # LLM invocation
    # ------------------------------------------------------------------

    def _invoke_llm(self, messages: List[BaseMessage]) -> AIMessage:
        if self._use_openrouter:
            try:
                resp = self._openrouter_llm.invoke(messages)
                if isinstance(resp, AIMessage):
                    return resp
            except Exception:
                logger.exception("OpenRouter failed; falling back to Groq")
        resp = self._groq_llm.invoke(messages)
        if isinstance(resp, AIMessage):
            return resp
        return AIMessage(content=str(getattr(resp, "content", resp)))

    # ------------------------------------------------------------------
    # FIX BUG 1 — context hint now clearly confirms what already happened
    # and what must happen next, so the LLM never second-guesses decompose.
    # ------------------------------------------------------------------

    def _build_hint(self, state: AgentState) -> str:
        lines: List[str] = []
        retrieved = bool(state.get("retrieved"))
        docs = state.get("docs") or []
        confidence = float(state.get("confidence") or 0.0)
        attempts = int(state.get("retrieval_attempts") or 0)
        rewrites = int(state.get("query_rewrites") or 0)
        sub_queries = state.get("sub_queries") or []
        decompose_attempted = bool(state.get("decompose_attempted"))
        phase = state.get("phase", "init")

        file_name = str(state.get("file_name") or "").strip()
        query = str(state.get("query") or state.get("question") or "").strip()

        lines.append(f"=== CURRENT STATE (phase={phase}) ===")

        lines.append(f"FILE_NAME_TO_USE: {file_name}")
        lines.append(f"QUERY_TO_USE: {query}")

        # Decompose status — show clearly as a numbered list so LLM knows it worked
        if sub_queries:
            lines.append(f"✓ decompose_query DONE — {len(sub_queries)} sub-queries:")
            for i, q in enumerate(sub_queries, 1):
                lines.append(f"  {i}. {q}")
            lines.append("  → DO NOT call decompose_query again.")
        else:
            if decompose_attempted:
                lines.append("  decompose_query: attempted but skipped (<2 sub-queries).")
            else:
                lines.append("  decompose_query: not called yet.")

        # Rewrite status
        if rewrites > 0:
            lines.append(f"✓ rewrite_query DONE (used {rewrites}/1 allowed).")
            lines.append("  → DO NOT call rewrite_query again.")
        else:
            lines.append("  rewrite_query: not called yet (0/1 used).")

        # Retrieval status
        lines.append(f"  retrieve: {attempts}/2 attempts | {len(docs)} chunks in memory | retrieved_flag={retrieved}")

        # Docs / confidence
        if retrieved and docs:
            if confidence > 0.0:
                conf_str = f"{confidence:.2f}"
                if confidence < _CONFIDENCE_THRESHOLD and attempts < 2:
                    lines.append(f"  confidence={conf_str} BELOW threshold {_CONFIDENCE_THRESHOLD:.2f} → call retrieve again with refined query.")
                else:
                    lines.append(f"  confidence={conf_str} OK → call answer NOW.")
            else:
                lines.append("  Docs retrieved, no rerank yet → call rerank or answer.")
        elif not retrieved:
            lines.append("  → NEXT ACTION: call retrieve(query=QUERY_TO_USE, file_name=FILE_NAME_TO_USE, top_k=...).")

        # Last error / directive
        directive = state.get("last_validation_directive")
        if directive:
            lines.append(f"  ⚠ CORRECTION NEEDED: {directive}")

        lines.append("=== END STATE ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Force recovery
    # ------------------------------------------------------------------

    def _force_recovery_message(self, state: AgentState, steps: int) -> AIMessage:
        if state.get("retrieved") and state.get("docs"):
            return AIMessage(
                content="Max steps — forcing answer.",
                tool_calls=[{"name": "answer", "args": {}, "id": f"force-answer-{steps}", "type": "tool_call"}],
            )
        return AIMessage(
            content="Max steps — forcing retrieve.",
            tool_calls=[{
                "name": "retrieve",
                "args": {
                    "query": str(state.get("query") or state.get("question") or ""),
                    "file_name": str(state.get("file_name") or ""),
                    "top_k": self._retrieval_top_k,
                },
                "id": f"force-retrieve-{steps}",
                "type": "tool_call",
            }],
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_tool_call(
        self, name: str, args: Dict[str, Any], state: AgentState
    ) -> Optional[str]:
        """Returns an error+directive string if the call is invalid, else None."""
        if name == "decompose_query":
            if state.get("sub_queries"):
                return "decompose_query already used. Call retrieve next."

        elif name == "rewrite_query":
            if int(state.get("query_rewrites") or 0) >= 1:
                return "rewrite_query already used. Call retrieve next."

        elif name == "retrieve":
            if int(state.get("retrieval_attempts") or 0) >= 2:
                return "Max 2 retrieve attempts reached. Call answer."
            q = str(args.get("query") or state.get("query") or "").strip()
            if not q:
                return "retrieve requires a non-empty query argument."

        elif name == "rerank":
            if not (state.get("docs") or []):
                return "rerank requires docs. Call retrieve first."

        elif name == "answer":
            if not state.get("retrieved"):
                return "answer requires a prior retrieve. Call retrieve first."

        return None

    # ------------------------------------------------------------------
    # Node: agent
    # ------------------------------------------------------------------

    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        steps = int(state.get("steps") or 0) + 1
        self._emit({"event": "step_start", "step": steps, "max_steps": self._max_steps})

        if steps > self._max_steps:
            msg = self._force_recovery_message(state, steps)
            self._emit({"event": "max_steps_forced", "step": steps})
            return {"steps": steps, "messages": [msg]}

        hint = self._build_hint(state)
        messages = list(state.get("messages") or []) + [HumanMessage(content=hint)]

        if _truthy_env("AGENT_LOG_EVENTS", False):
            logger.info("AGENT %s", _fmt_kv({"event": "controller_hint", "step": steps, "hint": _truncate(hint, 800)}))

        t0 = time.time()
        resp = self._invoke_llm(messages)
        elapsed = round(time.time() - t0, 3)

        tool_calls = getattr(resp, "tool_calls", None) or []

        # FIX BUG 2: if structured tool_calls is empty, try text parsing immediately
        if not tool_calls:
            parsed = _parse_text_tool_calls(getattr(resp, "content", "") or "")
            if parsed:
                tool_calls = parsed
                self._emit({"event": "tool_calls_text_parsed", "step": steps, "style": parsed[0].get("id", "")})
                # Patch the AIMessage so _route sees tool_calls on it
                resp = AIMessage(content=getattr(resp, "content", ""), tool_calls=tool_calls)

        # Guardrail: if we still have no tool calls but we need retrieval, force retrieve.
        if not tool_calls and state.get("phase") in {"init", "retrieve"} and not state.get("retrieved"):
            forced = [
                {
                    "name": "retrieve",
                    "args": {
                        "query": str(state.get("query") or state.get("question") or ""),
                        "file_name": str(state.get("file_name") or ""),
                        "top_k": self._retrieval_top_k,
                    },
                    "id": f"force-retrieve-{steps}",
                    "type": "tool_call",
                }
            ]
            tool_calls = forced
            resp = AIMessage(content=getattr(resp, "content", ""), tool_calls=tool_calls)
            self._emit({"event": "forced_tool_call", "step": steps, "tool_calls": ["retrieve"], "reason": "no_tool_calls_in_retrieve_phase"})

        # Guardrail: if we have docs and are in the answer phase, force the sentinel answer tool.
        if not tool_calls and state.get("phase") == "answer" and state.get("retrieved") and (state.get("docs") or []):
            forced = [{"name": "answer", "args": {}, "id": f"force-answer-{steps}", "type": "tool_call"}]
            tool_calls = forced
            resp = AIMessage(content=getattr(resp, "content", ""), tool_calls=tool_calls)
            self._emit({"event": "forced_tool_call", "step": steps, "tool_calls": ["answer"], "reason": "no_tool_calls_in_answer_phase"})

        self._emit({
            "event": "agent_decision",
            "step": steps,
            "tool_calls": [c.get("name") for c in tool_calls],
            "latency_s": elapsed,
        })

        if _truthy_env("AGENT_LOG_EVENTS", False):
            logger.info("AGENT %s", _fmt_kv({
                "event": "controller_raw", "step": steps,
                "content": _truncate(getattr(resp, "content", ""), 900),
                "tool_calls": tool_calls,
            }))

        return {"steps": steps, "messages": [resp]}

    # ------------------------------------------------------------------
    # Parallel sub-query retrieval
    # ------------------------------------------------------------------

    def _retrieve_parallel(
        self, sub_queries: List[str], file_name: str, top_k: int
    ) -> Tuple[List[Document], str]:
        all_docs: List[Document] = []
        seen: set[str] = set()
        for q in sub_queries:
            try:
                matches = self._service.retrieve(query=q, file_name=file_name, top_k=top_k)
                for d in self._service._matches_to_documents(matches):
                    key = (d.page_content or "")[:200]
                    if key and key not in seen:
                        seen.add(key)
                        all_docs.append(d)
            except Exception:
                logger.exception("Parallel retrieve failed for sub-query: %s", q)
        return all_docs, f"Parallel retrieval across {len(sub_queries)} sub-queries → {len(all_docs)} unique chunks."

    # ------------------------------------------------------------------
    # Node: tools
    # ------------------------------------------------------------------

    def _tools_node(self, state: AgentState) -> Dict[str, Any]:  # noqa: C901
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return {}

        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return {}

        out_messages: List[BaseMessage] = []
        updates: Dict[str, Any] = {"last_validation_directive": None}
        step = int(state.get("steps") or 0)

        for call in tool_calls:
            name = (call.get("name") or "").strip()
            args: Dict[str, Any] = call.get("args") or {}
            call_id: str = call.get("id") or name

            if _truthy_env("AGENT_LOG_EVENTS", False):
                logger.info("AGENT %s", _fmt_kv({
                    "event": "tool_start", "step": step, "tool": name,
                    "query": args.get("query") or state.get("query"),
                    "top_k": args.get("top_k"),
                }))

            error = self._validate_tool_call(name, args, state)
            if error:
                self._emit({"event": "tool_validation_error", "tool": name, "error": error, "step": step})
                # FIX BUG 1: store directive so _build_hint injects it into next LLM call
                updates["last_validation_directive"] = error
                out_messages.append(ToolMessage(
                    content=f"[Blocked] {error}",
                    tool_call_id=call_id,
                ))
                continue

            # -------------------------------------------------------
            if name == "rewrite_query":
                q = str(args.get("question") or state.get("query") or state.get("question") or "")
                file_name = str(state.get("file_name") or "")  # FIX BUG 3: always use state
                t0 = time.time()
                new_query = self._service.rewrite_query(question=q, file_name=file_name)
                elapsed = round(time.time() - t0, 3)
                updates["query"] = new_query
                updates["query_rewrites"] = int(state.get("query_rewrites") or 0) + 1
                updates["phase"] = "retrieve"
                self._emit({"event": "rewrite", "step": step, "query": new_query, "latency_s": elapsed})
                out_messages.append(ToolMessage(
                    content=f"Query rewritten to: {new_query!r}. Now call retrieve.",
                    tool_call_id=call_id,
                ))

            # -------------------------------------------------------
            elif name == "decompose_query":
                updates["decompose_attempted"] = True
                q = str(args.get("question") or state.get("question") or "")
                t0 = time.time()
                try:
                    sub = self._service.decompose_query(question=q)
                    if not isinstance(sub, list) or len(sub) < 2:
                        sub = []
                except Exception:
                    sub = []
                elapsed = round(time.time() - t0, 3)

                if sub:
                    updates["sub_queries"] = sub
                    updates["query"] = sub[0]
                    updates["phase"] = "retrieve"
                    self._emit({"event": "decompose", "step": step, "sub_queries": sub, "latency_s": elapsed})
                    # Clear numbered list in ToolMessage so LLM gets explicit confirmation
                    numbered = "\n".join(f"  {i}. {q}" for i, q in enumerate(sub, 1))
                    out_messages.append(ToolMessage(
                        content=(
                            f"Decomposed into {len(sub)} sub-queries:\n{numbered}\n\n"
                            "✓ decompose_query is DONE. DO NOT call it again.\n"
                            "NEXT: call retrieve now."
                        ),
                        tool_call_id=call_id,
                    ))
                else:
                    # Decompose returned nothing useful — skip to direct retrieve
                    updates["phase"] = "retrieve"
                    self._emit({"event": "decompose_skipped", "step": step, "reason": "insufficient_sub_queries"})
                    out_messages.append(ToolMessage(
                        content="Decomposition returned < 2 sub-queries. Call retrieve directly with the original query.",
                        tool_call_id=call_id,
                    ))

            # -------------------------------------------------------
            elif name == "retrieve":
                # FIX BUG 3: always use state file_name, never the LLM-provided one
                query = str(args.get("query") or state.get("query") or "")
                file_name = str(state.get("file_name") or "")
                attempts = int(state.get("retrieval_attempts") or 0)

                top_k = int(args.get("top_k") or self._retrieval_top_k)
                if attempts > 0:
                    top_k = min(top_k + _ADAPTIVE_TOP_K_BOOST, top_k * 2)
                    self._emit({"event": "adaptive_top_k", "step": step, "top_k": top_k})

                sub_queries = state.get("sub_queries") or []
                t0 = time.time()
                try:
                    # FIX BUG 4: only use parallel path when we have 2+ sub-queries
                    if len(sub_queries) >= 2:
                        docs, summary = self._retrieve_parallel(sub_queries, file_name, top_k)
                    else:
                        matches = self._service.retrieve(query=query, file_name=file_name, top_k=top_k)
                        docs = self._service._matches_to_documents(matches)
                        summary = f"Retrieved {len(docs)} chunks."
                except Exception as exc:
                    err = str(exc)
                    updates["last_tool_error"] = err
                    self._emit({"event": "tool_error", "tool": "retrieve", "step": step, "error": err})
                    out_messages.append(ToolMessage(content=f"[retrieve error] {err}", tool_call_id=call_id))
                    continue

                elapsed = round(time.time() - t0, 3)
                updates["docs"] = docs
                updates["retrieved"] = True
                updates["retrieval_attempts"] = attempts + 1
                updates["phase"] = "rerank" if docs else "retrieve"
                self._emit({"event": "retrieve", "step": step, "query": query, "top_k": top_k, "docs": len(docs), "latency_s": elapsed})

                if docs:
                    if _truthy_env("AGENT_LOG_EVENTS", False):
                        previews = [_doc_preview(d) for d in docs[:5]]
                        logger.info("AGENT %s", _fmt_kv({"event": "docs_preview", "step": step, "count": len(docs), "docs": previews}))
                    out_messages.append(ToolMessage(
                        content=f"{summary} Call rerank to improve quality, then call answer.",
                        tool_call_id=call_id,
                    ))
                else:
                    out_messages.append(ToolMessage(
                        content="No chunks retrieved. Call rewrite_query then retrieve again.",
                        tool_call_id=call_id,
                    ))

            # -------------------------------------------------------
            elif name == "rerank":
                query = str(args.get("query") or state.get("query") or "")
                docs = list(state.get("docs") or [])
                t0 = time.time()
                try:
                    reranked = self._service.rerank(
                        query=query, docs=docs, top_k=min(len(docs), self._rerank_top_k)
                    )
                except Exception as exc:
                    err = str(exc)
                    updates["last_tool_error"] = err
                    self._emit({"event": "tool_error", "tool": "rerank", "step": step, "error": err})
                    out_messages.append(ToolMessage(content=f"[rerank error] {err}", tool_call_id=call_id))
                    continue
                elapsed = round(time.time() - t0, 3)

                scores: List[float] = list(getattr(self._service, "last_rerank_scores", None) or [])
                avg_confidence = (sum(scores) / len(scores)) if scores else 0.5

                updates["docs"] = reranked
                updates["doc_scores"] = scores
                updates["confidence"] = avg_confidence
                updates["phase"] = "answer"

                self._emit({
                    "event": "rerank",
                    "step": step,
                    "top_k": min(len(docs), self._rerank_top_k),
                    "confidence": avg_confidence,
                    "latency_s": elapsed,
                })

                if reranked and _truthy_env("AGENT_LOG_EVENTS", False):
                    previews = [_doc_preview(d) for d in reranked[:5]]
                    logger.info(
                        "AGENT %s",
                        _fmt_kv({"event": "docs_preview", "step": step, "count": len(reranked), "docs": previews}),
                    )

                out_messages.append(
                    ToolMessage(
                        content="Reranked chunks. Call answer now.",
                        tool_call_id=call_id,
                    )
                )

            # -------------------------------------------------------
            elif name == "answer":
                updates["done"] = True
                updates["phase"] = "answer"
                self._emit({
                    "event": "stop", "step": step, "reason": "answer",
                    "docs": len(state.get("docs") or []),
                    "confidence": float(state.get("confidence") or 0.0),
                })
                out_messages.append(ToolMessage(content="answer", tool_call_id=call_id))

            else:
                out_messages.append(ToolMessage(content=f"Unknown tool: {name!r}", tool_call_id=call_id))

        if _truthy_env("AGENT_LOG_EVENTS", False):
            logger.info("AGENT %s", _fmt_kv({
                "event": "tools_done", "step": step,
                "phase": updates.get("phase") or state.get("phase"),
                "docs": len(updates.get("docs") or state.get("docs") or []),
                "retrieval_attempts": updates.get("retrieval_attempts") or state.get("retrieval_attempts"),
                "confidence": updates.get("confidence") or state.get("confidence"),
                "done": updates.get("done") or state.get("done"),
            }))

        return {"messages": out_messages, **updates}

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def _route(self, state: AgentState) -> str:
        if state.get("done"):
            return END

        messages = state.get("messages") or []
        if not messages:
            return END

        last = messages[-1]

        if isinstance(last, AIMessage):
            calls = getattr(last, "tool_calls", None) or []
            # tool_calls may have been patched in by _agent_node text parser
            if calls:
                return "tools"
            # Plain-text response with no parseable tool call → dead end
            if state.get("retrieved") and state.get("docs"):
                logger.warning("AGENT plain-text response with docs present; terminating.")
            return END

        if isinstance(last, ToolMessage):
            return END if state.get("done") else "agent"

        return END

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        graph: StateGraph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", self._route, {"tools": "tools", END: END})
        graph.add_conditional_edges("tools", self._route, {"agent": "agent", END: END})
        return graph.compile()