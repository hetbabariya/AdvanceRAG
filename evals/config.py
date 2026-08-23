"""Central configuration for the local RAG evaluation harness."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env from the project root so backend-style env vars resolve here too.
load_dotenv(PROJECT_ROOT / ".env")

EVALS_DIR = PROJECT_ROOT / "evals"
CORPUS_DIR = EVALS_DIR / "corpus"
DATASETS_DIR = EVALS_DIR / "datasets"
RESULTS_DIR = EVALS_DIR / "results"

GOLDEN_PATH = DATASETS_DIR / "golden.json"
CANDIDATES_RAW_PATH = DATASETS_DIR / "candidates_raw.json"

# ---------------------------------------------------------------------------
# Judge / dataset-generator LLM — OmniRoute (OpenAI-compatible), same vars as
# backend/rag/llm_factory.py so one .env serves both.
# ---------------------------------------------------------------------------
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY", "not-needed")
JUDGE_MODEL = os.getenv("OMNIROUTE_MODEL", "cx/gpt-5.5")
JUDGE_TEMPERATURE = float(os.getenv("EVAL_JUDGE_TEMPERATURE", "0"))

# ---------------------------------------------------------------------------
# Retrieval pipeline settings used by run_eval.py (mirrors production defaults).
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = int(os.getenv("EVAL_TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("EVAL_RERANK_TOP_N", "4"))
REWRITE_QUERY = os.getenv("EVAL_REWRITE_QUERY", "").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# ---------------------------------------------------------------------------
# Dataset generation knobs.
# ---------------------------------------------------------------------------
GEN_CHUNKS_PER_DOC = int(os.getenv("EVAL_GEN_CHUNKS_PER_DOC", "12"))
GEN_QA_PER_CHUNK = int(os.getenv("EVAL_GEN_QA_PER_CHUNK", "1"))
GEN_TEMPERATURE = float(os.getenv("EVAL_GEN_TEMPERATURE", "0.7"))
GEN_MAX_WORKERS = int(os.getenv("EVAL_GEN_MAX_WORKERS", "4"))

# Question-type mix. Weights are relative; generator round-robins by weight.
QUESTION_TYPE_WEIGHTS = {
    "factual": 3,
    "multi_hop": 2,
    "summary": 2,
    "unanswerable": 1,
}
QUESTION_TYPES = list(QUESTION_TYPE_WEIGHTS.keys())

# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------
METRIC_THRESHOLD = 0.7

# Dedicated Postgres user that owns all eval-ingested documents.
EVAL_USERNAME = os.getenv("EVAL_USERNAME", "eval-bot")
EVAL_EMAIL = os.getenv("EVAL_EMAIL", "eval-bot@local.eval")


def ensure_dirs() -> None:
    for d in (CORPUS_DIR, DATASETS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
