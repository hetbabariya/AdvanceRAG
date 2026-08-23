"""Phase 2 — generate candidate QA pairs from the corpus.

Chunks every corpus file exactly like production (OptimizedPreprocessedLoader),
samples evenly-spaced chunks, and asks the judge LLM to write question/answer
pairs of four types. Output goes to evals/datasets/candidates_raw.json and MUST
pass through the human review gate (evals/review_app.py) before becoming the
committed golden dataset.

Run:  python -m evals.generate_dataset [--force]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from evals.config import (
    CANDIDATES_RAW_PATH,
    CORPUS_DIR,
    GEN_CHUNKS_PER_DOC,
    GEN_MAX_WORKERS,
    GEN_QA_PER_CHUNK,
    GEN_TEMPERATURE,
    QUESTION_TYPE_WEIGHTS,
    ensure_dirs,
)
from evals.judge_model import OmniRouteJudge

MAX_CONTEXT_CHARS = 6000

# Weighted round-robin sequence, e.g. [factual x3, multi_hop x2, summary x2, unanswerable x1]
_TYPE_CYCLE: list[str] = []
for _t, _w in QUESTION_TYPE_WEIGHTS.items():
    _TYPE_CYCLE.extend([_t] * _w)


class QAPair(BaseModel):
    question: str = Field(description="self-contained question a user could ask")
    answer: str = Field(description="ground-truth answer derived strictly from the context")


def _clip(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[...truncated...]"


def _prompt(qtype: str, chunks: list[str]) -> str:
    ctx = "\n\n---\n\n".join(f"CONTEXT {i + 1}:\n{_clip(c)}" for i, c in enumerate(chunks))
    common = (
        "You are creating ground-truth data for evaluating a document Q&A system. "
        "Use ONLY the context provided; never rely on outside knowledge.\n\n"
        f"{ctx}\n\n"
    )
    if qtype == "factual":
        task = (
            "Write ONE self-contained factual question whose answer is stated explicitly "
            "in the context (a specific value, definition, name, or claim). Then write its "
            "ground-truth answer in one or two complete sentences."
        )
    elif qtype == "summary":
        task = (
            "Pick ONE concept explained in the context and write ONE question asking to "
            "explain or summarize it (e.g. 'What is X and how does it work?'). The answer "
            "must be a faithful 3-5 sentence summary of what the context says about X."
        )
    elif qtype == "multi_hop":
        task = (
            "The two contexts above are related excerpts from the same document. Write ONE "
            "question whose answer requires combining information from BOTH contexts "
            "(e.g. comparing, relating, or building on both). The answer must cite facts "
            "from each context."
        )
    else:  # unanswerable
        task = (
            "Write ONE plausible-sounding question about details that are NOT present in "
            "the context (a number, date, name, or capability that never appears). Then the "
            "ground-truth answer must politely refuse in 1-2 sentences, stating that the "
            "document does not contain this information. Do NOT invent an answer."
        )
    return common + task


def _sample_indices(n_docs: int, n_samples: int) -> list[int]:
    """Evenly-spaced indices covering the whole chunk list."""
    if n_docs <= n_samples:
        return list(range(n_docs))
    step = n_docs / n_samples
    return sorted({min(n_docs - 1, int(i * step)) for i in range(n_samples)})


def _build_jobs() -> list[dict]:
    from backend.rag.loader import OptimizedPreprocessedLoader

    loader = OptimizedPreprocessedLoader()
    jobs: list[dict] = []

    pdfs = sorted(p for p in CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no corpus PDFs found in {CORPUS_DIR}; run python -m evals.corpus_prep")

    rng = random.Random(42)
    for pdf in pdfs:
        docs = loader.load_and_split_file(str(pdf), pdf.name)
        idxs = _sample_indices(len(docs), GEN_CHUNKS_PER_DOC)
        print(f"{pdf.name}: {len(docs)} chunks, sampled {len(idxs)}")

        for k, ci in enumerate(idxs):
            qtype = _TYPE_CYCLE[k % len(_TYPE_CYCLE)]
            doc = docs[ci]
            chunk_ids = [doc.metadata.get("chunk_id", f"chunk_{ci}")]

            if qtype == "multi_hop":
                partner = ci + 1 if ci + 1 < len(docs) else ci - 1
                partner_doc = docs[max(0, partner)]
                chunks = [doc.page_content, partner_doc.page_content]
                chunk_ids.append(partner_doc.metadata.get("chunk_id", f"chunk_{partner}"))
            else:
                chunks = [doc.page_content]

            jobs.append(
                {
                    "id": f"{pdf.stem}-{ci}-{qtype}",
                    "source_doc": pdf.name,
                    "question_type": qtype,
                    "chunks": chunks,
                    "chunk_ids": chunk_ids,
                }
            )

            # Extra QA pairs per chunk when configured (factual only).
            for extra in range(GEN_QA_PER_CHUNK - 1):
                alt = rng.randrange(len(docs))
                alt_doc = docs[alt]
                jobs.append(
                    {
                        "id": f"{pdf.stem}-{alt}-factual-x{extra}",
                        "source_doc": pdf.name,
                        "question_type": "factual",
                        "chunks": [alt_doc.page_content],
                        "chunk_ids": [alt_doc.metadata.get("chunk_id", f"chunk_{alt}")],
                    }
                )
    return jobs


def _generate_one(judge: OmniRouteJudge, job: dict, attempts: int = 3) -> dict | None:
    import time

    for attempt in range(attempts):
        try:
            pair = judge.generate(
                _prompt(job["question_type"], job["chunks"]),
                schema=QAPair,
            )
            if not isinstance(pair, QAPair):
                return None
            q = pair.question.strip()
            a = pair.answer.strip()
            if not q or not a:
                return None
            return {
                "id": job["id"],
                "input": q,
                "expected_output": a,
                "context": list(job["chunks"]),
                "metadata": {
                    "source_doc": job["source_doc"],
                    "chunk_ids": job["chunk_ids"],
                    "question_type": job["question_type"],
                },
            }
        except Exception as e:
            if attempt == attempts - 1:
                print(f"warn: {job['id']} failed after {attempts} attempts: {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing candidates file")
    args = parser.parse_args()

    ensure_dirs()

    records: list[dict] = []
    if CANDIDATES_RAW_PATH.exists() and not args.force:
        try:
            records = json.loads(CANDIDATES_RAW_PATH.read_text(encoding="utf-8"))
            print(f"merging into {len(records)} existing records (use --force to regenerate)")
        except json.JSONDecodeError:
            print(f"warn: {CANDIDATES_RAW_PATH.name} unreadable; starting fresh")
    elif CANDIDATES_RAW_PATH.exists() and args.force:
        CANDIDATES_RAW_PATH.unlink()

    jobs = _build_jobs()
    done_ids = {r["id"] for r in records}
    pending = [j for j in jobs if j["id"] not in done_ids]
    if not pending:
        print("nothing to generate; all jobs already present")
        return 0
    print(f"\ngenerating {len(pending)}/{len(jobs)} candidate QA pairs via {GEN_MAX_WORKERS} workers...")

    judge = OmniRouteJudge(temperature=GEN_TEMPERATURE)
    with ThreadPoolExecutor(max_workers=GEN_MAX_WORKERS) as pool:
        futures = {pool.submit(_generate_one, judge, job): job for job in pending}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="qa"):
            rec = fut.result()
            if rec:
                records.append(rec)

    records.sort(key=lambda r: r["id"])
    CANDIDATES_RAW_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

    by_type: dict[str, int] = {}
    by_doc: dict[str, int] = {}
    for r in records:
        by_type[r["metadata"]["question_type"]] = by_type.get(r["metadata"]["question_type"], 0) + 1
        by_doc[r["metadata"]["source_doc"]] = by_doc.get(r["metadata"]["source_doc"], 0) + 1

    print(f"\nwrote {len(records)} records -> {CANDIDATES_RAW_PATH}")
    print("by type:", json.dumps(by_type, indent=None))
    print("by doc :", json.dumps(by_doc, indent=None))
    print("\nnext: streamlit run evals/review_app.py  (review -> datasets/golden.json)")
    return 0 if records else 2


if __name__ == "__main__":
    sys.exit(main())
