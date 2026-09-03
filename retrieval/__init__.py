"""Hybrid semantic retrieval subsystem for the geedge-rag corpus.

Parallel to scripts/ (the batch keyword triage pipeline). Embeds the JSONL
shards written by scripts/03_extract_text.py once, stores dense vectors + a
BM25 FTS index in LanceDB, and exposes a query-time hybrid search CLI.

The JSONL shards under $GEEDGE_WORK/text/ are the only contract between this
package and scripts/ — there are no imports in either direction.
"""
