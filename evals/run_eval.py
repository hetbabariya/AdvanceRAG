"""Phase 4 — score the RAG pipeline against the reviewed golden dataset.

Bootstraps RagService in-process (same as backend/api/main.py), ingests any
missing corpus files under a dedicated eval-bot user, replays the production
pipeline (rewrite -> hybrid retrieve -> rerank -> generate) for each approved
golden record, and scores results with DeepEval using the OmniRoute judge.

Run:  python -m evals.run_eval [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("TELEMETRY_OPT_OUT", "1")

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from sqlalchemy import select  # noqa: E402
from tqdm import tqdm  # noqa: E402

from evals.config import (  # noqa: E402
    CORPUS_DIR,
    EVAL_EMAIL,
    EVAL_USERNAME,
    GOLDEN_PATH,
    METRIC_THRESHOLD,
    RESULTS_DIR,
    RETRIEVAL_TOP_K,
    REWRITE_QUERY,
    RERANK_TOP_N,
    ensure_dirs,
)
from evals.judge_model import OmniRouteJudge  # noqa: E402

CONTEXT_METRICS = ("contextual_recall", "contextual_precision")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_golden(limit: int | None) -> list[dict]:
    if not GOLDEN_PATH.exists():
        raise SystemExit(
            f"{GOLDEN_PATH} not found. Run `streamlit run evals/review_app.py` "
            "and approve records first."
        )
    records = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    approved = [r for r in records if r.get("review_status") == "approved"]
    if limit:
        approved = approved[:limit]
    if not approved:
        raise SystemExit("no approved records in golden.json — review the dataset first")
    return approved


def bootstrap_service():
    from backend.rag.pinecone_hybrid import build_hybrid_index
    from backend.rag.service import RagService
    from backend.rag.settings import load_settings

    settings = load_settings()
    hybrid = build_hybrid_index(
        pinecone_api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        embedding_model=settings.embedding_model,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )
    return RagService.create(settings, hybrid)


async def prepare_data(service, wanted_docs: set[str]) -> tuple[int, dict[str, str]]:
    """All DB work in ONE event loop (asyncpg connections are loop-bound).

    Creates/returns the eval-bot user and ingests any missing corpus files.
    Returns (user_id, {file_name: file_hash}).
    """
    from backend.api.database import AsyncSessionLocal, engine, init_db
    from backend.api.models import FileMetadata, User

    await init_db()
    try:
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(User).where(User.username == EVAL_USERNAME))
            user = res.scalars().first()
            if user is None:
                user = User(username=EVAL_USERNAME, email=EVAL_EMAIL,
                            password_hash="!eval-no-login")
                db.add(user)
                await db.commit()
                await db.refresh(user)
                print(f"created eval user #{user.id} ({EVAL_USERNAME})")
        user_id = int(user.id)

        corpus = {p.name: p for p in sorted(CORPUS_DIR.glob("*.pdf"))}
        hashes: dict[str, str] = {}
        for name in sorted(wanted_docs):
            path = corpus.get(name)
            if path is None:
                raise SystemExit(f"corpus file '{name}' not found in {CORPUS_DIR}")
            hashes[name] = sha256_file(path)

        async with AsyncSessionLocal() as db:
            rows = (
                await db.execute(select(FileMetadata).where(FileMetadata.user_id == user_id))
            ).scalars().all()
        known = {(r.file_name, r.file_hash) for r in rows}

        for name, fh in hashes.items():
            if (name, fh) in known:
                print(f"already ingested: {name}")
                continue
            print(f"ingesting {name} ...", flush=True)
            count, _bm25 = await asyncio.to_thread(
                service.ingest_file,
                saved_path=str(corpus[name]),
                original_name=name,
                file_hash=fh,
                user_id=user_id,
            )
            async with AsyncSessionLocal() as db:
                db.add(
                    FileMetadata(
                        user_id=user_id,
                        file_name=name,
                        file_hash=fh,
                        file_size=corpus[name].stat().st_size,
                        chunks_count=count,
                    )
                )
                await db.commit()
            print(f"  done: {count} chunks")
        return user_id, hashes
    finally:
        # Close pooled connections bound to this (now-dying) event loop.
        await engine.dispose()


def extract_answer(raw) -> str:
    """service.generate returns a JSON string {'answer':..., 'citations':[...]}."""
    text = raw if isinstance(raw, str) else str(raw)
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            return data["answer"].strip()
    except (json.JSONDecodeError, ValueError):
        pass
    return text.strip()


def run_pipeline(service, rec: dict, hashes: dict[str, str], user_id: int):
    """Replay production: rewrite -> retrieve -> rerank -> generate."""
    question = rec["input"]
    fname = rec["metadata"]["source_doc"]

    query = service.rewrite_query(question=question, file_name=fname) if REWRITE_QUERY else question
    matches = service.retrieve(
        query=query,
        file_name=fname,
        top_k=RETRIEVAL_TOP_K,
        file_hash=hashes[fname],
        user_id=user_id,
    )
    docs = [
        Document(page_content=(m.get("metadata") or {}).get("text", ""),
                 metadata=dict(m.get("metadata") or {}))
        for m in matches
    ]
    reranked = service.rerank(query=query, docs=docs, top_k=RERANK_TOP_N)
    raw = service.generate(question=question, context_docs=reranked)
    return extract_answer(raw), [d.page_content for d in reranked]


def build_metrics(judge: OmniRouteJudge, contextual: bool) -> list:
    metrics = [
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
    ]
    if contextual:
        metrics += [
            ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
            ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=judge, include_reason=False),
        ]
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only first N records")
    args = parser.parse_args()

    ensure_dirs()
    records = load_golden(args.limit)
    print(f"evaluating {len(records)} approved goldens")

    service = bootstrap_service()
    user_id, hashes = asyncio.run(prepare_data(service, {r["metadata"]["source_doc"] for r in records}))

    judge = OmniRouteJudge()
    case_rows: list[dict] = []
    failed_cases: list[dict] = []

    for rec in tqdm(records, desc="pipeline+score", unit="case"):
        qtype = rec["metadata"]["question_type"]
        row = {
            "id": rec["id"],
            "source_doc": rec["metadata"]["source_doc"],
            "question_type": qtype,
            "scores": {},
            "error": None,
        }
        try:
            answer, retrieved = run_pipeline(service, rec, hashes, user_id)
            row["answer"] = answer
            tc = LLMTestCase(
                input=rec["input"],
                actual_output=answer,
                expected_output=rec["expected_output"],
                retrieval_context=retrieved or None,
            )
            use_contextual = qtype != "unanswerable" and bool(rec.get("context"))
            for metric in build_metrics(judge, use_contextual):
                try:
                    metric.measure(tc, _show_indicator=False)
                    row["scores"][metric.__class__.__name__] = round(float(metric.score), 4)
                except Exception as e:
                    row["scores"][metric.__class__.__name__] = None
                    row.setdefault("metric_errors", {})[metric.__class__.__name__] = str(e)[:300]
        except Exception as e:
            row["error"] = str(e)[:300]
            failed_cases.append(row)
        case_rows.append(row)

    # ----------------------------------------------------------- aggregation
    def agg(rows: list[dict], key_fn) -> dict:
        buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            k = key_fn(r)
            for m, s in r["scores"].items():
                if s is not None:
                    buckets[k][m].append(s)
        out: dict[str, dict] = {}
        for k, per_metric in buckets.items():
            out[k] = {m: round(sum(v) / len(v), 4) for m, v in per_metric.items()}
        return dict(sorted(out.items()))

    overall = agg(case_rows, lambda r: "overall")
    by_doc = agg([r for r in case_rows if not r["error"]], lambda r: r["source_doc"])
    by_type = agg([r for r in case_rows if not r["error"]], lambda r: r["question_type"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    detail_path = RESULTS_DIR / f"run_{stamp}.jsonl"
    detail_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in case_rows), encoding="utf-8"
    )
    summary = {
        "timestamp": stamp,
        "model": judge.model_name,
        "n_records": len(records),
        "n_failed": len(failed_cases),
        "retrieval": {"top_k": RETRIEVAL_TOP_K, "rerank_top_n": RERANK_TOP_N,
                      "rewrite_query": REWRITE_QUERY},
        "threshold": METRIC_THRESHOLD,
        "overall": overall.get("overall", {}),
        "by_source_doc": by_doc,
        "by_question_type": by_type,
    }
    summary_path = RESULTS_DIR / f"run_{stamp}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- report
    print("\n==================== SCORECARD ====================")
    print(json.dumps(summary, indent=2))
    print(f"\ndetails : {detail_path}")
    print(f"summary : {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
