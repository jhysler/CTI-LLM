#!/usr/bin/env python3
"""
Query CLI: hybrid semantic search over the indexed corpus.

Thin wrapper over retrieval.lib.search.search() — all retrieval logic lives
there (so a future MCP server shares it). This file owns argument parsing and
terminal/JSON rendering only.

Usage:
  # exact identifier, no model load
  python -m retrieval.query "XDC02030000" --mode fts

  # Chinese semantic query, JIRA-scoped
  python -m retrieval.query "DNS 劫持 检测" --jira

  # full record dump for one doc
  python -m retrieval.query --show-doc <doc_id>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import orjson

from retrieval.lib import paths
from retrieval.lib.embedder import Embedder
from retrieval.lib.search import build_where, search
from retrieval.lib.shards import fetch_record
from retrieval.lib.store import Store


def collapse(text: str, limit: int = 200) -> str:
    """Whitespace-collapse a snippet, like find_snippet() in 04_triage.py."""
    s = re.sub(r"\s+", " ", text.replace("\n", " ").replace("\r", " ")).strip()
    return s[:limit] + ("…" if len(s) > limit else "")


def _doc_location(manifest: dict, doc_id: str):
    """(shard_path, byte_offset) for a doc from an already-loaded manifest, or None."""
    man = manifest.get(doc_id)
    if not man:
        return None
    return paths.TEXT_DIR / man["shard_file"], int(man["byte_offset"])


def cmd_show_doc(store: Store, doc_id: str) -> int:
    loc = _doc_location(store.load_manifest(), doc_id)
    if not loc:
        print(f"No such doc in manifest: {doc_id}", file=sys.stderr)
        return 1
    rec = fetch_record(loc[0], loc[1], doc_id)
    print(f"# {doc_id}")
    print(f"path: {rec.get('path')}")
    print(f"lang: {rec.get('lang')}  ext: {rec.get('ext')}  "
          f"chars: {rec.get('char_count')}  ocr: {rec.get('ocr')}")
    print("-" * 72)
    print(rec.get("text") or "(empty)")
    return 0


def render_table(docs) -> None:
    try:
        from tabulate import tabulate
        rows = [[i + 1, f"{d.score:.4f}", d.lang or "-", d.path] for i, d in enumerate(docs)]
        print(tabulate(rows, headers=["#", "score", "lang", "path"], tablefmt="simple"))
    except Exception:
        for i, d in enumerate(docs):
            print(f"{i+1:>3}  {d.score:.4f}  {d.lang or '-':<4}  {d.path}")


def render_snippets(docs, store: Store, full_context: bool) -> None:
    # load_manifest() is a full-table scan; load once here rather than once
    # per doc/chunk (the latter turned a handful of hits into a multi-minute
    # render for a corpus this size — see ask.py's gather_context for the
    # same fix).
    manifest = store.load_manifest() if full_context else {}
    for i, d in enumerate(docs):
        print(f"\n[{i+1}] {d.path}  (score {d.score:.4f}, {d.lang or '-'}, {d.doc_id})")
        for c in d.chunks:
            head = f"    [chunk {c.chunk_index} @ {c.char_start}..{c.char_end}] "
            if full_context:
                loc = _doc_location(manifest, d.doc_id)
                if loc:
                    rec = fetch_record(loc[0], loc[1], d.doc_id)
                    text = rec.get("text") or ""
                    s = max(0, c.char_start - 1500)
                    e = min(len(text), c.char_end + 1500)
                    print(head + "(±1500 chars)")
                    print("    " + collapse(text[s:e], limit=3200))
                    continue
            print(head + collapse(c.text, limit=200))


def main() -> int:
    ap = argparse.ArgumentParser(description="Hybrid semantic search over the corpus.")
    ap.add_argument("query", nargs="?", help="query string")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "fts"])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--per-doc", type=int, default=2, help="max chunks shown per doc")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--path-prefix", action="append", default=[],
                    help="restrict to paths starting with this (repeatable, OR-joined)")
    ap.add_argument("--jira", action="store_true", help="sugar for --path-prefix geedge_jira/")
    ap.add_argument("--lang", help="restrict to a detected language (e.g. zh, en)")
    ap.add_argument("--filter", dest="raw_filter", help="raw SQL predicate, AND-ed on")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--full-context", action="store_true",
                    help="print ±1500 chars around each hit from the source record")
    ap.add_argument("--show-doc", metavar="DOC_ID", help="dump one full record and exit")
    ap.add_argument("--json", action="store_true", help="emit DocHit list as JSON")
    args = ap.parse_args()

    if not paths.VECTORS_DIR.exists():
        print(f"No index at {paths.VECTORS_DIR}. Run `python -m retrieval.index` first.",
              file=sys.stderr)
        return 1
    store = Store.open(paths.VECTORS_DIR)

    if args.show_doc:
        return cmd_show_doc(store, args.show_doc)

    if not args.query:
        print("A query is required (or use --show-doc).", file=sys.stderr)
        return 2

    where = build_where(args.path_prefix, args.jira, args.lang, args.raw_filter)

    embedder = None
    if args.mode != "fts":
        print(f"Loading embedding model (device={args.device})... "
              f"(use --mode fts to skip)", file=sys.stderr)
        embedder = Embedder(device=args.device)

    docs = search(store, embedder, args.query, mode=args.mode, where=where,
                  top_k=args.top_k, per_doc=args.per_doc, rrf_k=args.rrf_k)

    if args.json:
        sys.stdout.buffer.write(
            orjson.dumps([d.to_dict() for d in docs], option=orjson.OPT_INDENT_2))
        sys.stdout.write("\n")
        return 0

    if not docs:
        print("No results.")
        return 0

    render_table(docs)
    render_snippets(docs, store, args.full_context)
    return 0


if __name__ == "__main__":
    sys.exit(main())
