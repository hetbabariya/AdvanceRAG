"""Central LLM provider factory.

Providers: groq | openrouter | omniroute, selected via ``LLM_PROVIDER``:
  - auto (default): legacy behavior — OpenRouter when OPENROUTER_API_KEY is
    set, otherwise Groq. Zero breaking change for existing deployments.
  - explicit: always use that provider; Groq remains the runtime fallback
    (FallbackLLM / try-except) exactly as before.

OmniRoute is any OpenAI-compatible router (default http://localhost:20128/v1,
models named like "cx/gpt-5.5"). The OpenRouter-specific "reasoning"
extra_body is never sent to OmniRoute unless OMNIROUTE_REASONING_ENABLED=1.
"""
from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Protocol

from backend.api.utils import get_logger

logger = get_logger(__name__)

GROQ_DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_AGENT_DEFAULT_MODEL = "llama-3.3-70b-versatile"
OPENROUTER_DEFAULT_MODEL = "arcee-ai/trinity-large-preview:free"
OPENROUTER_AGENT_DEFAULT_MODEL = "stepfun/step-3.5-flash:free"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OMNIROUTE_DEFAULT_BASE_URL = "http://localhost:20128/v1"


class _LLMResponse(Protocol):
    content: str


class _LLM(Protocol):
    def invoke(self, prompt: str) -> _LLMResponse: ...

    def astream(self, prompt: str) -> AsyncIterator[_LLMResponse]: ...


@dataclass
class _TextResponse:
    content: str


@dataclass(frozen=True)
class ProviderConfig:
    name: str              # groq | openrouter | omniroute
    model: str             # simple-path model (rewrite/generate)
    agent_model: str       # tool-calling agent model
    base_url: str          # openai-compatible base url ("" for groq)
    api_key: str           # "" for groq
    extra_reasoning: bool  # OpenRouter reasoning extra_body


def _clean(value: object) -> str:
    return str(value or "").strip().strip('"').strip("'")


def _truthy(raw: object, default: bool = False) -> bool:
    v = _clean(raw).lower()
    if not v:
        return default
    return v not in {"0", "false", "no", "off"}


def pick_groq_model(raw: object) -> str:
    """Sanitize GROQ_AGENT_MODEL — reject non-Groq vendor prefixes."""
    name = _clean(raw)
    if not name:
        return GROQ_AGENT_DEFAULT_MODEL
    low = name.lower()
    forbidden = ("openai", "anthropic", "google", "mistral/")
    if any(low.startswith(p) for p in forbidden):
        return GROQ_AGENT_DEFAULT_MODEL
    if "/" in low and not any(low.startswith(p) for p in ("llama", "gemma", "mixtral")):
        return GROQ_AGENT_DEFAULT_MODEL
    return name


def resolve_provider_config() -> ProviderConfig:
    raw = _clean(os.getenv("LLM_PROVIDER")).lower()
    provider = raw if raw in {"groq", "openrouter", "omniroute"} else "auto"

    if provider == "auto":
        provider = "openrouter" if _clean(os.getenv("OPENROUTER_API_KEY")) else "groq"

    groq_model = _clean(os.getenv("GROQ_MODEL")) or GROQ_DEFAULT_MODEL
    groq_agent_model = pick_groq_model(os.getenv("GROQ_AGENT_MODEL"))

    if provider == "groq":
        return ProviderConfig("groq", groq_model, groq_agent_model, "", "", False)

    if provider == "omniroute":
        key = _clean(os.getenv("OMNIROUTE_API_KEY"))
        model = _clean(os.getenv("OMNIROUTE_MODEL"))
        base_url = _clean(os.getenv("OMNIROUTE_BASE_URL")) or OMNIROUTE_DEFAULT_BASE_URL
        if not key or not model:
            logger.warning(
                "LLM_PROVIDER=omniroute needs OMNIROUTE_API_KEY and OMNIROUTE_MODEL "
                "(base_url=%s) — falling back to Groq",
                base_url,
            )
            return ProviderConfig("groq", groq_model, groq_agent_model, "", "", False)
        return ProviderConfig(
            name="omniroute",
            model=model,
            agent_model=_clean(os.getenv("OMNIROUTE_AGENT_MODEL")) or model,
            base_url=base_url,
            api_key=key,
            extra_reasoning=_truthy(os.getenv("OMNIROUTE_REASONING_ENABLED"), False),
        )

    # openrouter
    key = _clean(os.getenv("OPENROUTER_API_KEY"))
    base_url = _clean(os.getenv("OPENROUTER_BASE_URL")) or OPENROUTER_DEFAULT_BASE_URL
    if not key:
        logger.warning(
            "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY missing — falling back to Groq"
        )
        return ProviderConfig("groq", groq_model, groq_agent_model, "", "", False)
    reasoning_raw = _clean(os.getenv("OPENROUTER_REASONING_ENABLED")) or "true"
    return ProviderConfig(
        name="openrouter",
        model=_clean(os.getenv("OPENROUTER_MODEL")) or OPENROUTER_DEFAULT_MODEL,
        agent_model=_clean(os.getenv("OPENROUTER_AGENT_MODEL")) or OPENROUTER_AGENT_DEFAULT_MODEL,
        base_url=base_url,
        api_key=key,
        extra_reasoning=_truthy(reasoning_raw, True),
    )


class OpenAICompatLLM:
    """invoke/astream against any OpenAI-compatible chat completions endpoint."""

    def __init__(self, *, api_key: str, model: str, base_url: str, extra_reasoning: bool = False):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._extra_reasoning = extra_reasoning

    def _kwargs(self) -> Dict[str, Any]:
        if self._extra_reasoning:
            return {"extra_body": {"reasoning": {"enabled": True}}}
        return {}

    def invoke(self, prompt: str) -> _TextResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **self._kwargs(),
        )
        msg = resp.choices[0].message
        return _TextResponse(content=(getattr(msg, "content", None) or "").strip())

    async def astream(self, prompt: str) -> AsyncIterator[_TextResponse]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()

        def _run_streaming():
            try:
                stream = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    **self._kwargs(),
                )
                for event in stream:
                    token: Any = None
                    try:
                        token = getattr(event.choices[0].delta, "content", None)
                    except Exception:
                        token = None
                    if token:
                        loop.call_soon_threadsafe(q.put_nowait, _TextResponse(content=str(token)))
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, sentinel)

        threading.Thread(target=_run_streaming, daemon=True).start()

        while True:
            item = await q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item  # type: ignore[misc]


class FallbackLLM:
    def __init__(self, *, primary: "_LLM", fallback: "_LLM"):
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


def _safe_groq(model: str):
    """Build a ChatGroq instance, or None when GROQ_API_KEY is unavailable."""
    from langchain_groq import ChatGroq

    try:
        return ChatGroq(model=model)
    except Exception as e:
        logger.warning("Groq unavailable as fallback (%s); continuing without it", e)
        return None


def build_simple_llm() -> "_LLM":
    """LLM for rewrite/generate/smalltalk paths. Falls back to Groq on error."""
    cfg = resolve_provider_config()
    if cfg.name == "groq":
        logger.info("LLM provider=groq model=%s", cfg.model)
        from langchain_groq import ChatGroq

        return ChatGroq(model=cfg.model)

    logger.info(
        "LLM provider=%s model=%s base_url=%s fallback=groq(%s)",
        cfg.name, cfg.model, cfg.base_url, cfg.model,
    )
    primary: "_LLM" = OpenAICompatLLM(
        api_key=cfg.api_key,
        model=cfg.model,
        base_url=cfg.base_url,
        extra_reasoning=cfg.extra_reasoning,
    )
    groq_llm = _safe_groq(cfg.model)
    if groq_llm is None:
        return primary
    return FallbackLLM(primary=primary, fallback=groq_llm)


def build_tool_llm(tools: List[Any]):
    """Tool-calling LLM for the agent graph.

    Returns (primary_bound, groq_fallback_bound_or_None, config). The caller
    wraps primary.invoke in try/except and retries with the fallback.
    """
    cfg = resolve_provider_config()
    groq_bound = None
    groq_fallback = _safe_groq(pick_groq_model(os.getenv("GROQ_AGENT_MODEL")))
    if groq_fallback is not None:
        groq_bound = groq_fallback.bind_tools(tools)
    if cfg.name == "groq":
        logger.info("Agent LLM provider=groq model=%s", cfg.agent_model)
        return groq_bound, groq_bound, cfg

    from langchain_openai import ChatOpenAI

    primary = ChatOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.agent_model,
    ).bind_tools(tools)
    logger.info(
        "Agent LLM provider=%s model=%s base_url=%s fallback=%s (tool support depends on upstream model)",
        cfg.name, cfg.agent_model, cfg.base_url, "groq" if groq_bound else "none",
    )
    return primary, groq_bound, cfg
