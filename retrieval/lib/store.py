"""LanceDB store: schemas, open/create, upsert, and index building.

Two tables under $GEEDGE_WORK/vectors/ (a disposable LanceDB database,
rebuildable from the shards):

  chunks — one row per chunk, dense vector + FTS-indexed text. The searchable
           table.
  docs   — one row per source document: a manifest for incremental indexing
           (text_sha1 change detection, shard_file/byte_offset back-reference)
           and for --full-context fetches.

The store never invents a hybrid/vector API surface; search.py owns query
construction. This module only creates tables, upserts, and builds indexes.

LanceDB's create_fts_index / hybrid signatures have shifted across releases, so
the FTS builder here is defensive: it degrades kwargs on TypeError rather than
hard-failing (docs/retrieval-plan.md flags this explicitly).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, List, Optional

import pyarrow as pa

VECTOR_DIM = 1024

CHUNKS_TABLE = "chunks"
DOCS_TABLE = "docs"

# ANN is only worth building past this many rows; below it LanceDB does an exact
# (flat) search, which is both faster to build and more accurate. Fixture tests
# stay well under this and never build an index.
ANN_MIN_ROWS = 50_000


def chunks_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("char_start", pa.int32()),
            pa.field("char_end", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("path", pa.string()),
            pa.field("ext", pa.string()),
            pa.field("lang", pa.string()),
            pa.field("ocr", pa.bool_()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )


def docs_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("ext", pa.string()),
            pa.field("lang", pa.string()),
            pa.field("char_count", pa.int64()),
            pa.field("text_sha1", pa.string()),
            pa.field("n_chunks", pa.int32()),
            pa.field("shard_file", pa.string()),  # filename only — portable
            pa.field("byte_offset", pa.int64()),
            pa.field("status", pa.string()),       # "indexed" | "empty"
            pa.field("indexed_at", pa.timestamp("us")),
        ]
    )


class Store:
    def __init__(self, db, chunks, docs):
        self.db = db
        self.chunks = chunks
        self.docs = docs

    # ---- lifecycle -------------------------------------------------------

    @classmethod
    def open(cls, vectors_dir: Path, *, rebuild: bool = False) -> "Store":
        import lancedb

        vectors_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(vectors_dir))
        existing = set(db.table_names())

        if rebuild:
            for name in (CHUNKS_TABLE, DOCS_TABLE):
                if name in existing:
                    db.drop_table(name)
            existing -= {CHUNKS_TABLE, DOCS_TABLE}

        chunks = (
            db.open_table(CHUNKS_TABLE)
            if CHUNKS_TABLE in existing
            else db.create_table(CHUNKS_TABLE, schema=chunks_schema())
        )
        docs = (
            db.open_table(DOCS_TABLE)
            if DOCS_TABLE in existing
            else db.create_table(DOCS_TABLE, schema=docs_schema())
        )
        return cls(db, chunks, docs)

    # ---- manifest --------------------------------------------------------

    def load_manifest(self) -> dict:
        """{doc_id: {"text_sha1", "status", "shard_file", "byte_offset"}}."""
        try:
            tbl = self.docs.to_arrow()
        except Exception:
            return {}
        cols = tbl.to_pydict()
        out = {}
        for i, doc_id in enumerate(cols.get("doc_id", [])):
            out[doc_id] = {
                "text_sha1": cols["text_sha1"][i],
                "status": cols["status"][i],
                "shard_file": cols["shard_file"][i],
                "byte_offset": cols["byte_offset"][i],
            }
        return out

    # ---- writes ----------------------------------------------------------

    def delete_docs(self, doc_ids: Iterable[str]) -> None:
        """Remove all chunk rows for these doc_ids (delete-then-add upsert).

        merge_insert can't drop a shrunk document's surplus chunks, so changed
        docs are always deleted first.
        """
        ids = [d.replace("'", "''") for d in doc_ids]
        if not ids:
            return
        quoted = ", ".join(f"'{d}'" for d in ids)
        self.chunks.delete(f"doc_id IN ({quoted})")

    def add_chunks(self, rows: List[dict]) -> None:
        if not rows:
            return
        for r in rows:
            v = r.get("vector")
            if v is not None and not isinstance(v, list):
                r["vector"] = list(v)
        self.chunks.add(rows)

    def upsert_doc(self, row: dict) -> None:
        self.upsert_docs([row])

    def upsert_docs(self, rows: List[dict]) -> None:
        """Batched upsert — one LanceDB commit for the whole list, not one per row.

        A per-row merge_insert (the original shape of this method) writes a
        fresh transaction/manifest file per document; on a flush full of many
        tiny docs (e.g. a folder of one-paragraph reports) that serialized
        loop can dominate wall-clock time with zero progress output. Batching
        the whole flush's rows into a single execute() cuts commits from
        O(docs) to O(flushes).
        """
        if not rows:
            return
        (
            self.docs.merge_insert("doc_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def refresh_doc_location(self, doc_id: str, shard_file: str, byte_offset: int) -> None:
        """Update only the shard_file/byte_offset of an unchanged doc whose line
        moved (in-place rewrite / re-shard) — no re-embed."""
        (
            self.docs.merge_insert("doc_id")
            .when_matched_update_all()
            .execute([{"doc_id": doc_id, "shard_file": shard_file, "byte_offset": byte_offset}])
        )

    # ---- indexes ---------------------------------------------------------

    def build_fts(self) -> None:
        """(Re)build the BM25 FTS index on chunks.text.

        lowercase on, stemming OFF (literal identifiers like XDC02030000 must
        survive), ascii-folding on, positions on (phrase queries recover
        `doescan.xyz` → `doescan xyz`). CJK isn't segmented by this tokenizer —
        CJK recall rides the dense leg in v1 (TODO: evaluate ngram/jieba).
        """
        attempts = [
            dict(field_names="text", replace=True, with_position=True,
                 use_tantivy=False, lower_case=True, stem=False, ascii_folding=True),
            dict(field_names="text", replace=True, with_position=True),
            dict(field_names="text", replace=True),
        ]
        last_err: Optional[Exception] = None
        for kwargs in attempts:
            field = kwargs.pop("field_names")
            try:
                self.chunks.create_fts_index(field, **kwargs)
                return
            except TypeError as e:
                last_err = e
            except Exception as e:
                last_err = e
                break
        if last_err:
            raise last_err

    def build_ann_if_large(self) -> bool:
        """IVF_PQ cosine index once chunks exceed ANN_MIN_ROWS. Returns whether
        it built."""
        n = self.chunks.count_rows()
        if n <= ANN_MIN_ROWS:
            return False
        import math

        num_partitions = max(1, int(math.sqrt(n)))
        self.chunks.create_index(
            metric="cosine",
            vector_column_name="vector",
            num_partitions=num_partitions,
            num_sub_vectors=64,  # 1024 / 64 = 16 dims per sub-vector
            index_type="IVF_PQ",
            replace=True,
        )
        return True

    def optimize(self) -> None:
        try:
            self.chunks.optimize()
            self.docs.optimize()
        except Exception:
            pass  # optimize() is a housekeeping nicety, not correctness-critical

    def count_chunks(self) -> int:
        return self.chunks.count_rows()

    def count_docs(self) -> int:
        return self.docs.count_rows()


def now_utc() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)
