"""Phase 0 — build the frozen local benchmark corpus.

One-shot script:
  1. Moves the downloaded PDFs from data/ into evals/corpus/ under stable names.
  2. Extracts a chapter subset of Dive into Deep Learning so the 42 MB book
     becomes a focused ~200-page benchmark file.
  3. Writes evals/corpus/manifest.json with provenance + checksums.

Run:  python -m evals.corpus_prep [--force]
The original data/ files are left untouched (d2l source stays in place).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import fitz  # pymupdf

from evals.config import CORPUS_DIR, PROJECT_ROOT, ensure_dirs

DATA_DIR = PROJECT_ROOT / "data"

# data/<downloaded name> -> evals/corpus/<stable name>
RENAME_MAP = {
    "0000950170-24-087843.pdf": "report_msft_10k_fy2024.pdf",
    "1706.03762v7.pdf": "paper_attention_is_all_you_need.pdf",
    "2005.11401v4.pdf": "paper_rag_lewis_et_al.pdf",
    "Retrieval-augmented_generation.pdf": "wiki_retrieval_augmented_generation.pdf",
}

D2L_SOURCE = "d2l-en.pdf"
D2L_TARGET = "book_d2l_subset.pdf"

# d2l level-1 TOC titles to keep.
D2L_CHAPTERS = [
    "Preliminaries",
    "Multilayer Perceptrons",
    "Builders' Guide",
    "Attention Mechanisms and Transformers",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def move_renames(force: bool) -> list[Path]:
    """Move + rename the plain downloads. Returns the corpus files produced."""
    produced: list[Path] = []
    for src_name, dst_name in RENAME_MAP.items():
        src, dst = DATA_DIR / src_name, CORPUS_DIR / dst_name
        if dst.exists() and not force:
            print(f"skip (exists): {dst.name}")
            produced.append(dst)
            continue
        if not src.exists():
            if dst.exists():
                print(f"warn: {src_name} missing in data/, keeping existing {dst.name}")
                produced.append(dst)
            else:
                print(f"ERROR: neither {src} nor {dst} exists")
            continue
        shutil.move(str(src), str(dst))
        print(f"moved: {src_name} -> corpus/{dst_name}")
        produced.append(dst)
    return produced


def _chapter_ranges(doc: fitz.Document) -> dict[str, tuple[int, int]]:
    """Map level-1 TOC title -> (start_page_0based, end_page_inclusive)."""
    toc = doc.get_toc(simple=True)
    level1 = [(title, page - 1) for level, title, page in toc if level == 1]
    ranges: dict[str, tuple[int, int]] = {}
    for i, (title, start) in enumerate(level1):
        end = (level1[i + 1][1] - 1) if i + 1 < len(level1) else doc.page_count - 1
        ranges.setdefault(title, (start, max(start, end)))
    return ranges


def build_d2l_subset(force: bool) -> Path | None:
    src, dst = DATA_DIR / D2L_SOURCE, CORPUS_DIR / D2L_TARGET
    if dst.exists() and not force:
        print(f"skip (exists): {dst.name}")
        return dst
    if not src.exists():
        print(f"ERROR: d2l source missing: {src}")
        return None

    with fitz.open(src) as doc:
        ranges = _chapter_ranges(doc)
        missing = [c for c in D2L_CHAPTERS if c not in ranges]
        if missing:
            print(f"warn: chapters not found in TOC: {missing}")

        subset = fitz.open()
        total_pages = 0
        for title in D2L_CHAPTERS:
            if title not in ranges:
                continue
            start, end = ranges[title]
            subset.insert_pdf(doc, from_page=start, to_page=end)
            total_pages += end - start + 1
            print(f"d2l '{title}': pages {start + 1}-{end + 1}")
        if total_pages == 0:
            raise SystemExit("ERROR: no d2l chapters extracted; aborting")
        subset.save(str(dst), garbage=3, deflate=True)
        subset.close()

    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"built: corpus/{dst.name} ({total_pages} pages, {size_mb:.1f} MB)")
    return dst


def write_manifest(files: list[Path]) -> None:
    entries = []
    for f in files:
        with fitz.open(f) as d:
            pages = d.page_count
        entries.append(
            {"name": f.name, "pages": pages, "bytes": f.stat().st_size, "sha256": sha256_of(f)}
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "Frozen benchmark corpus. Regenerate only via python -m evals.corpus_prep --force.",
        "files": entries,
    }
    out = CORPUS_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = parser.parse_args()

    ensure_dirs()
    if not DATA_DIR.exists():
        print(f"ERROR: data dir not found: {DATA_DIR}")
        return 1

    moved = move_renames(args.force)
    d2l_subset = build_d2l_subset(args.force)
    files = [p for p in [*moved, d2l_subset] if p is not None and p.exists()]
    write_manifest(files)

    expected = len(RENAME_MAP) + 1
    print(f"\ncorpus ready: {len(files)}/{expected} files in {CORPUS_DIR}")
    return 0 if len(files) == expected else 2


if __name__ == "__main__":
    sys.exit(main())
