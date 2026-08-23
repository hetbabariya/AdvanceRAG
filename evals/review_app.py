"""Phase 3 — human review gate for candidate QA pairs.

Streamlit app to approve / edit / reject LLM-generated candidates before they
become the committed golden dataset. Every action saves immediately to
evals/datasets/golden.json (all records, each tagged review_status); run_eval.py
scores only records marked "approved".

Run:  streamlit run evals/review_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from evals.config import CANDIDATES_RAW_PATH, GOLDEN_PATH  # noqa: E402

STATUS_ICONS = {"approved": "✅", "rejected": "❌", "pending": "⏳"}


def _load() -> list[dict]:
    """Resume from golden.json when present, else bootstrap from candidates."""
    source = GOLDEN_PATH if GOLDEN_PATH.exists() else CANDIDATES_RAW_PATH
    if not source.exists():
        st.error(
            f"No dataset found. Run `python -m evals.generate_dataset` first "
            f"(expected {CANDIDATES_RAW_PATH})."
        )
        st.stop()
    records = json.loads(source.read_text(encoding="utf-8"))
    for r in records:
        r.setdefault("review_status", "pending")
    return records


def _save(records: list[dict]) -> None:
    GOLDEN_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    st.set_page_config(page_title="Golden Dataset Review", page_icon="🧪", layout="wide")
    st.title("Golden Dataset Review")
    st.caption(f"source: {CANDIDATES_RAW_PATH.name} → output: {GOLDEN_PATH}")

    if "records" not in st.session_state:
        st.session_state.records = _load()
    records: list[dict] = st.session_state.records

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Filters")
        statuses = [s for s in ("pending", "approved", "rejected") if any(
            r["review_status"] == s for r in records)]
        f_status = st.multiselect("Status", ["pending", "approved", "rejected"], default=statuses)
        docs = sorted({r["metadata"]["source_doc"] for r in records})
        f_doc = st.multiselect("Source doc", docs)
        types = sorted({r["metadata"]["question_type"] for r in records})
        f_type = st.multiselect("Question type", types)

        st.divider()
        n_ok = sum(r["review_status"] == "approved" for r in records)
        st.metric("Approved / total", f"{n_ok}/{len(records)}")

        st.divider()
        st.subheader("Counts by doc × type")
        grid: dict[str, dict[str, int]] = {}
        for r in records:
            d = r["metadata"]["source_doc"]
            t = r["metadata"]["question_type"]
            row = grid.setdefault(d, {})
            row[t] = row.get(t, 0) + 1
        st.dataframe(
            {
                "doc": [d.replace(".pdf", "") for d in grid],
                **{
                    t: [grid[d].get(t, 0) for d in grid]
                    for t in sorted({t for row in grid.values() for t in row})
                },
            },
            hide_index=True,
            use_container_width=True,
        )

        st.divider()
        if st.button("Reset all reviews", use_container_width=True):
            confirm = st.checkbox("Yes, set every record back to pending")
            if confirm:
                for r in records:
                    r["review_status"] = "pending"
                _save(records)
                st.rerun()

    # ------------------------------------------------------------------- cards
    visible = [
        r
        for r in records
        if r["review_status"] in f_status
        and (not f_doc or r["metadata"]["source_doc"] in f_doc)
        and (not f_type or r["metadata"]["question_type"] in f_type)
    ]
    if not visible:
        st.info("No records match the current filters.")
        return

    for i, rec in enumerate(visible):
        meta = rec["metadata"]
        icon = STATUS_ICONS.get(rec["review_status"], "?")
        header = (
            f"{icon} [{rec['review_status']}] {rec['id']}  ·  "
            f"{meta['question_type']} · {meta['source_doc']}"
        )
        with st.container(border=True):
            st.markdown(f"**{header}**")

            new_q = st.text_area(
                "Question", value=rec["input"], key=f"q_{rec['id']}", height=68
            )
            new_a = st.text_area(
                "Expected answer", value=rec["expected_output"], key=f"a_{rec['id']}", height=110
            )
            if new_q != rec["input"] or new_a != rec["expected_output"]:
                rec["input"], rec["expected_output"] = new_q.strip(), new_a.strip()

            with st.expander(f"Golden context ({len(rec['context'])} chunk(s))"):
                for j, ctx in enumerate(rec["context"]):
                    st.markdown(f"chunk {j + 1}: `{meta['chunk_ids'][j]}`")
                    st.text(ctx)

            c1, c2, c3, _ = st.columns([1, 1, 1, 5])
            if c1.button("Approve", key=f"ok_{rec['id']}", type="primary"):
                rec["review_status"] = "approved"
                _save(records)
                st.rerun()
            if c2.button("Reject", key=f"no_{rec['id']}"):
                rec["review_status"] = "rejected"
                _save(records)
                st.rerun()
            if c3.button("Reset", key=f"rs_{rec['id']}"):
                rec["review_status"] = "pending"
                _save(records)
                st.rerun()

        if i < len(visible) - 1:
            st.write("")


if __name__ == "__main__":
    main()
