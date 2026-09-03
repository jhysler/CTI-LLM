#!/usr/bin/env python3
"""
Indexer: embed the JSONL shards into the LanceDB store.

Single incremental pass over $GEEDGE_WORK/text/shard_*.jsonl. Each record is
classified against the docs manifest by a sha1 of its text:

  unchanged (same sha1)  → skip embedding; refresh (shard_file, byte_offset) if
                           the line moved (03 --reextract-low-text rewrites in
                           place; 09 appends shards)
  changed   (sha1 differs, e.g. an OCR re-extract) → delete old chunks, re-embed
  new                    → embed
  empty text             → manifest row status="empty", no chunks

Doc atomicity for crash-resumability (same spirit as Stage 3's done-set): a
document's `docs` manifest row is upserted only *after* its chunks are written,
so an interrupted run re-stages the doc cleanly on restart.

Usage:
  python -m retrieval.index --dry-run                 # counts only, no GPU
  python -m retrieval.index                           # incremental build
  python -m retrieval.index --rebuild                 # drop tables, full build
  python -m retrieval.index --limit 500 --device cpu  # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from tqdm import tqdm

from retrieval.lib import paths
from retrieval.lib.chunking import chunk_document, get_tokenizer
from retrieval.lib.embedder import Embedder
from retrieval.lib.shards import iter_records
from retrieval.lib.store import Store, now_utc

# Embed + write in blocks of roughly this many chunks so the GPU sees full
# batches and we don't hold the whole corpus in RAM. Flushed at doc boundaries
# (never mid-document) to keep manifest upserts atomic.
FLUSH_CHUNKS = 2000

# Path substrings for ingested reference/fixture data with no organizational
# content — bulk noise, not knowledge:
#  - MaxMind GeoLite2-City database snapshots: third-party geolocation CSVs,
#    hundreds of chunks each, wholesale-attached to JIRA tickets.
#  - Raw IP/port list dumps: tens of thousands of "IP-IP#port-port" rows,
#    zero prose — meaningless to embed, not usefully full-text-searchable.
#  - ThunderVPN test fixtures: literal null-byte filler files used to test
#    upload/throughput, not real data.
EXCLUDE_PATH_SUBSTRINGS = [
    "GeoLite2-City-CSV",
    "geedge_jira/attachment/31580/App+Client+IPs.txt",
    "geedge_jira/attachment/31581/top+Server+IPs.txt",
    "geedge_jira/attachment/39559/穿透时间段服务器IP.txt",
    "CapturePacketByProces_Android/data/",  # whole dir: 1/2/5/10MB null-byte fixtures
]


def is_excluded(path: str) -> bool:
    return any(s in path for s in EXCLUDE_PATH_SUBSTRINGS)


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Index shards into LanceDB.")
    ap.add_argument("--rebuild", action="store_true",
                    help="drop both tables and index from scratch")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap number of new/changed docs staged (0 = all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print new/changed/unchanged/empty counts and exit before GPU")
    ap.add_argument("--no-fts", action="store_true", help="skip FTS index build")
    ap.add_argument("--target-tokens", type=int, default=800)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--overlap-tokens", type=int, default=150)
    ap.add_argument("--max-chunks-per-doc", type=int, default=0,
                    help="cap chunks per document (0 = unlimited)")
    args = ap.parse_args()

    if not paths.TEXT_DIR.exists():
        print(f"No text dir: {paths.TEXT_DIR}. Run scripts/03_extract_text.py first.",
              file=sys.stderr)
        return 1

    store = Store.open(paths.VECTORS_DIR, rebuild=args.rebuild)
    manifest = store.load_manifest()
    print(f"Store: {paths.VECTORS_DIR}")
    print(f"Manifest: {len(manifest):,} docs already indexed")

    # --- classification pass --------------------------------------------
    n_new = n_changed = n_unchanged = n_empty = n_excluded = 0
    seen: set[str] = set()
    staged: list[tuple[str, str, int, dict, str]] = []  # (id, shard_file, offset, rec, sha1)
    limit_hit = False

    for shard, offset, rec in tqdm(iter_records(paths.TEXT_DIR), unit="rec", desc="scan"):
        doc_id = rec.get("id")
        if not doc_id or doc_id in seen:
            continue  # first occurrence wins (real corpus ids are unique)
        seen.add(doc_id)

        text = rec.get("text") or ""
        sha1 = text_sha1(text)
        is_empty = not text.strip()
        path = rec.get("path") or ""
        prev = manifest.get(doc_id)

        # Checked unconditionally, ahead of the unchanged-doc shortcut below:
        # a doc already indexed under an older exclusion list must still be
        # caught and purged here, not skipped as "unchanged".
        if is_excluded(path):
            n_excluded += 1
            if not args.dry_run and prev and prev.get("status") != "excluded":
                store.delete_docs([doc_id])  # drop previously-embedded chunks
            if not args.dry_run and (not prev or prev.get("status") != "excluded"
                                      or prev.get("text_sha1") != sha1):
                store.upsert_doc({
                    "doc_id": doc_id,
                    "path": path,
                    "ext": rec.get("ext") or "",
                    "lang": rec.get("lang") or "",
                    "char_count": int(rec.get("char_count") or 0),
                    "text_sha1": sha1,
                    "n_chunks": 0,
                    "shard_file": shard.name,
                    "byte_offset": offset,
                    "status": "excluded",
                    "indexed_at": now_utc(),
                })
            continue

        if prev and prev.get("text_sha1") == sha1:
            n_unchanged += 1
            if (prev.get("shard_file") != shard.name
                    or prev.get("byte_offset") != offset):
                if not args.dry_run:
                    store.refresh_doc_location(doc_id, shard.name, offset)
            continue

        if is_empty:
            n_empty += 1
            if not args.dry_run:
                store.upsert_doc({
                    "doc_id": doc_id,
                    "path": path,
                    "ext": rec.get("ext") or "",
                    "lang": rec.get("lang") or "",
                    "char_count": int(rec.get("char_count") or 0),
                    "text_sha1": sha1,
                    "n_chunks": 0,
                    "shard_file": shard.name,
                    "byte_offset": offset,
                    "status": "empty",
                    "indexed_at": now_utc(),
                })
            continue

        if prev:
            n_changed += 1
        else:
            n_new += 1

        if args.limit and len(staged) >= args.limit:
            limit_hit = True
            continue
        staged.append((doc_id, shard.name, offset, rec, sha1))

    print(f"\nClassified: {n_new:,} new, {n_changed:,} changed, "
          f"{n_unchanged:,} unchanged, {n_empty:,} empty, {n_excluded:,} excluded")
    if limit_hit:
        print(f"--limit {args.limit}: staged first {len(staged):,} new/changed docs")

    if args.dry_run:
        print("Dry run — no embedding performed.")
        return 0

    if not staged:
        print("Nothing to embed.")
        if not args.no_fts and store.count_chunks() > 0:
            print("Rebuilding FTS index...")
            store.build_fts()
        store.optimize()
        return 0

    # --- embed + write ---------------------------------------------------
    print(f"\nLoading tokenizer + model (device={args.device})...")
    tokenizer = get_tokenizer()
    embedder = Embedder(device=args.device, max_length=args.max_length,
                        batch_size=args.batch_size)

    buf_docs: list[dict] = []   # {doc_id, meta, chunks:[Chunk]}
    buf_chunk_count = 0
    changed_ids = {sid for sid, *_ in staged
                   if manifest.get(sid)}  # need chunk delete before re-add
    total_chunks = 0

    def flush() -> None:
        nonlocal buf_docs, buf_chunk_count, total_chunks
        if not buf_docs:
            return
        # Delete stale chunks for any changed docs in this batch first.
        to_delete = [d["doc_id"] for d in buf_docs if d["doc_id"] in changed_ids]
        store.delete_docs(to_delete)

        texts = [c.text for d in buf_docs for c in d["chunks"]]
        vecs = embedder.encode_passages(texts, batch_size=args.batch_size)

        rows = []
        vi = 0
        for d in buf_docs:
            meta = d["meta"]
            for c in d["chunks"]:
                rows.append({
                    "chunk_id": f"{d['doc_id']}:{c.chunk_index:04d}",
                    "doc_id": d["doc_id"],
                    "chunk_index": int(c.chunk_index),
                    "char_start": int(c.char_start),
                    "char_end": int(c.char_end),
                    "text": c.text,
                    "path": meta["path"],
                    "ext": meta["ext"],
                    "lang": meta["lang"],
                    "ocr": meta["ocr"],
                    "vector": vecs[vi].tolist(),
                })
                vi += 1
        store.add_chunks(rows)

        # Manifest rows only after chunks land (crash-resumable). Batched into
        # one commit for the whole flush rather than one merge_insert per doc.
        indexed_at = now_utc()
        manifest_rows = [
            {
                "doc_id": d["doc_id"],
                "path": d["meta"]["path"],
                "ext": d["meta"]["ext"],
                "lang": d["meta"]["lang"],
                "char_count": d["meta"]["char_count"],
                "text_sha1": d["sha1"],
                "n_chunks": len(d["chunks"]),
                "shard_file": d["meta"]["shard_file"],
                "byte_offset": d["meta"]["byte_offset"],
                "status": "indexed",
                "indexed_at": indexed_at,
            }
            for d in buf_docs
        ]
        store.upsert_docs(manifest_rows)

        total_chunks += len(rows)
        buf_docs = []
        buf_chunk_count = 0

    for doc_id, shard_file, offset, rec, sha1 in tqdm(staged, unit="doc", desc="embed"):
        text = rec.get("text") or ""
        chunks = chunk_document(
            text, tokenizer=tokenizer,
            target_tokens=args.target_tokens, max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            max_chunks_per_doc=args.max_chunks_per_doc,
        )
        if not chunks:
            # Degenerate (all-whitespace after paragraph split) — record as empty.
            store.upsert_doc({
                "doc_id": doc_id, "path": rec.get("path") or "",
                "ext": rec.get("ext") or "", "lang": rec.get("lang") or "",
                "char_count": int(rec.get("char_count") or 0), "text_sha1": sha1,
                "n_chunks": 0, "shard_file": shard_file, "byte_offset": offset,
                "status": "empty", "indexed_at": now_utc(),
            })
            continue

        buf_docs.append({
            "doc_id": doc_id,
            "sha1": sha1,
            "chunks": chunks,
            "meta": {
                "path": rec.get("path") or "",
                "ext": rec.get("ext") or "",
                "lang": rec.get("lang") or "",
                "char_count": int(rec.get("char_count") or 0),
                "ocr": bool(rec.get("ocr", False)),
                "shard_file": shard_file,
                "byte_offset": offset,
            },
        })
        buf_chunk_count += len(chunks)
        if buf_chunk_count >= FLUSH_CHUNKS:
            flush()

    flush()

    # --- indexes ---------------------------------------------------------
    if not args.no_fts:
        print("Building FTS index...")
        store.build_fts()
    if store.build_ann_if_large():
        print("Built IVF_PQ vector index.")
    store.optimize()

    print(f"\nDone. Embedded {total_chunks:,} chunks across {len(staged):,} docs.")
    print(f"Store now: {store.count_chunks():,} chunks, {store.count_docs():,} docs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
