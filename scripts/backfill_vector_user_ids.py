"""Backfill `user_id` metadata on legacy Pinecone vectors.

Vectors ingested before user-scoping was added carry only `file_name` in
metadata. Strict per-user filters (see `_scoped_filter` in
backend/rag/pinecone_hybrid.py) make those vectors invisible to queries.
Run this ONCE after deploying the user-scoped code to stamp ownership onto
existing vectors without re-ingesting.

How it works:
  1. Read every (file_name, user_id) pair from the Postgres file_metadata table.
  2. For each file name, list its vectors from Pinecone via a metadata query.
  3. If ALL rows for that file_name belong to ONE user, stamp `user_id` on
     every vector of that file (`index.update` merges set_metadata).
  4. If MULTIPLE users own the same file_name, ownership of pre-existing
     vectors cannot be inferred — the script stamps them for every owner
     (vectors are content-identical only if the uploads were identical;
     otherwise they were already cross-contaminated). Review those warnings.

Usage:
    venv\\Scripts\\python.exe scripts\\backfill_vector_user_ids.py

Requires PINECONE_API_KEY, PINECONE_INDEX_NAME and DATABASE_URL in .env,
plus a reachable Postgres with the file_metadata table populated.
"""
from __future__ import annotations

import asyncio
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import asyncpg  # noqa: E402


async def fetch_owners() -> Dict[str, List[int]]:
    dsn = os.getenv("DATABASE_URL", "")
    if dsn.startswith("postgresql+asyncpg://"):
        dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch("SELECT DISTINCT file_name, user_id FROM file_metadata ORDER BY file_name")
    finally:
        await conn.close()
    owners: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        owners[r["file_name"]].append(int(r["user_id"]))
    return dict(owners)


def list_vector_ids(index, file_name: str, dimension: int) -> List[str]:
    """Collect all vector ids whose metadata matches this file_name.

    The SDK's ``index.list()`` only supports id-prefix listing, so we issue a
    dense query with a dummy vector plus a strict metadata filter; top_k=10000
    returns every vector of that file regardless of similarity.
    """
    ids: List[str] = []
    seen = set()
    dummy = [0.0] * dimension
    resp = index.query(
        vector=dummy,
        top_k=10000,
        include_metadata=False,
        include_values=False,
        filter={"file_name": {"$eq": file_name}},
    )
    for m in (resp or {}).get("matches", []):
        vid = m.get("id")
        if vid and vid not in seen:
            seen.add(vid)
            ids.append(vid)
    return ids


def main() -> None:
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY", "")
    index_name = os.getenv("PINECONE_INDEX_NAME", "")
    if not api_key or not index_name:
        sys.exit("Set PINECONE_API_KEY and PINECONE_INDEX_NAME in .env first.")

    owners = asyncio.run(fetch_owners())
    if not owners:
        print("No files in file_metadata — nothing to backfill.")
        return

    pc = Pinecone(api_key=api_key)
    description = pc.describe_index(index_name)
    dimension = int(getattr(description, "dimension", 0) or 0)
    if dimension <= 0:
        sys.exit(f"Could not determine dimension of index '{index_name}'.")
    index = pc.Index(index_name)

    total_stamped = 0
    ambiguous: List[Tuple[str, List[int]]] = []

    for i, (file_name, user_ids) in enumerate(sorted(owners.items()), 1):
        print(f"[{i}/{len(owners)}] '{file_name}' ...")
        ids = list_vector_ids(index, file_name, dimension)
        if not ids:
            print(f"  [skip] no vectors found (already deleted or different index)")
            continue
        if len(user_ids) == 1:
            uid = user_ids[0]
            done = 0
            for vid in ids:
                index.update(id=vid, set_metadata={"user_id": uid})
                done += 1
                if done % 200 == 0:
                    print(f"         stamped {done}/{len(ids)}")
            total_stamped += len(ids)
            print(f"  [ok]   stamped user_id={uid} on {len(ids)} vectors")
        else:
            ambiguous.append((file_name, user_ids))
            print(f"  [warn] owned by users {user_ids} — cannot infer ownership; skipped")

    print(f"\nDone. Stamped {total_stamped} vectors.")
    if ambiguous:
        print(f"{len(ambiguous)} ambiguous file names skipped. Re-ingest those files per user, "
              "or delete their vectors manually:")
        for name, uids in ambiguous:
            print(f"  - {name} (users {uids})")


if __name__ == "__main__":
    main()
