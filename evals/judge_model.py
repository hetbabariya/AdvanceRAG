"""DeepEval judge LLM backed by OmniRoute (OpenAI-compatible).

Used both as the metric judge and as the golden-dataset generator so scores
and data come from the same model family. Reuses the backend's OMNIROUTE_*
env vars (see evals/config.py).
"""

from __future__ import annotations

from typing import Optional, Union

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

from evals.config import JUDGE_MODEL, JUDGE_TEMPERATURE, OMNIROUTE_API_KEY, OMNIROUTE_BASE_URL


class OmniRouteJudge(DeepEvalBaseLLM):
    """Adapter that plugs an OpenAI-compatible endpoint into DeepEval."""

    def __init__(
        self,
        *,
        base_url: str = OMNIROUTE_BASE_URL,
        api_key: str = OMNIROUTE_API_KEY,
        model_name: str = JUDGE_MODEL,
        temperature: float = JUDGE_TEMPERATURE,
    ):
        # Set attributes before super().__init__, which calls load_model().
        self.model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        self._temperature = temperature
        super().__init__(model=model_name)

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model_name,
            openai_api_base=self._base_url,
            openai_api_key=self._api_key or "not-needed",
            temperature=self._temperature,
            timeout=120,
            max_retries=2,
        )

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Union[BaseModel, str]:
        llm = self.load_model()
        if schema is None:
            return str(llm.invoke(prompt).content)
        structured = llm.with_structured_output(schema)
        result = structured.invoke(prompt)
        if isinstance(result, BaseModel):
            return result
        # Some providers return dicts even in structured mode.
        return schema.model_validate(result)

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Union[BaseModel, str]:
        llm = self.load_model()
        if schema is None:
            resp = await llm.ainvoke(prompt)
            return str(resp.content)
        structured = llm.with_structured_output(schema)
        result = await structured.ainvoke(prompt)
        if isinstance(result, BaseModel):
            return result
        return schema.model_validate(result)


def _smoke_test() -> int:
    """Validate plain + structured calls against the endpoint."""
    from pydantic import Field

    print(f"endpoint : {OMNIROUTE_BASE_URL}")
    print(f"model    : {JUDGE_MODEL}")

    judge = OmniRouteJudge()

    plain = judge.generate("Reply with exactly: OK")
    print(f"plain    : {plain!r}")
    assert "OK" in str(plain), "plain generation failed"

    class Verdict(BaseModel):
        score: float = Field(description="score between 0 and 1")
        reason: str

    structured = judge.generate("Score how happy 'I love pizza' sounds (0-1).", schema=Verdict)
    print(f"structured: {structured}")
    assert isinstance(structured, Verdict), "structured generation failed"

    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
