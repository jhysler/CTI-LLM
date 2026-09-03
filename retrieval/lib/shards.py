"""Read the JSONL shards written by scripts/03_extract_text.py.

Ported from build_doc_id_index()/fetch_doc() in scripts/07_analyze.py:136 —
the byte-offset index idiom. Two extra concerns the indexer needs that 07
doesn't:

  * shards get rewritten in place (03 --reextract-low-text) and appended to
    (09_ingest_manual.py), so a stored (shard, offset) can point at a *different*
    record after the fact. fetch_record() verifies rec["id"] and raises a clear
    "re-run the indexer" error rather than returning silently-wrong text.
  * records from 09 omit the `ocr` key (and, in its error form, `ext` too), so
    every field access uses rec.get(...) with a default. iter_records() does not
    normalize — callers use .get(); the store layer supplies the defaults.

Record schema (03_extract_text.py:18):
    {id, path, ext, lang, char_count, n_elements, text, error, ocr}
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import orjson


class StaleIndexError(RuntimeError):
    """A stored (shard_file, byte_offset) no longer points at its expected id.

    Raised when shards were rewritten in place or re-sharded after indexing.
    Recovery is always the same: re-run `python -m retrieval.index`.
    """


def shard_paths(text_dir: Path) -> list[Path]:
    """Sorted shard_*.jsonl under text_dir (sorted = stable offset ordering)."""
    return sorted(text_dir.glob("shard_*.jsonl"))


def iter_records(text_dir: Path) -> Iterator[Tuple[Path, int, dict]]:
    """Yield (shard_path, byte_offset, record) for every parseable line.

    byte_offset is the offset of the line within its shard (raw bytes, so it
    survives seek()), matching build_doc_id_index() in scripts/07. Unparseable
    lines are skipped, exactly as the pipeline scripts do.
    """
    for shard in shard_paths(text_dir):
        with shard.open("rb") as f:
            offset = 0
            for line in f:
                try:
                    rec = orjson.loads(line)
                except Exception:
                    offset += len(line)
                    continue
                yield shard, offset, rec
                offset += len(line)


def fetch_record(shard_path: Path, byte_offset: int, expected_id: str) -> dict:
    """Seek to a stored offset and return the record, verifying its id.

    Raises StaleIndexError if the line there belongs to a different id (shards
    mutated since indexing) so callers surface a re-index instruction instead of
    serving the wrong document's text.
    """
    with shard_path.open("rb") as f:
        f.seek(byte_offset)
        line = f.readline()
    try:
        rec = orjson.loads(line)
    except Exception as e:
        raise StaleIndexError(
            f"Could not parse record at {shard_path.name}:{byte_offset} "
            f"(expected id {expected_id}). Re-run the indexer."
        ) from e
    if rec.get("id") != expected_id:
        raise StaleIndexError(
            f"Index stale: {shard_path.name}:{byte_offset} now holds id "
            f"{rec.get('id')!r}, expected {expected_id!r}. "
            f"Shards were rewritten — re-run `python -m retrieval.index`."
        )
    return rec
