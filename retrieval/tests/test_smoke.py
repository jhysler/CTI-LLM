"""Smoke tests.

Two tiers:
  * chunking unit tests — no model, always run (use the WhitespaceTokenizer stub)
  * @pytest.mark.slow end-to-end — index the fixture on CPU and query it; needs
    torch + FlagEmbedding + lancedb installed and a one-time ~2.3GB bge-m3
    download. Run with:  pytest retrieval/tests/ -m slow

Run everything fast (chunking only):  pytest retrieval/tests/ -m "not slow"
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import orjson
import pytest

from retrieval.lib.chunking import chunk_document
from retrieval.lib.search import build_where, rrf_fuse
from retrieval.tests.fixtures import (
    IDENT_DOMAIN, IDENT_HARDWARE, WhitespaceTokenizer, append_shard,
    fixture_records, make_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOK = WhitespaceTokenizer()


# ---------------------------------------------------------------------------
# Chunking unit tests (no model)
# ---------------------------------------------------------------------------

def test_empty_text_zero_chunks():
    assert chunk_document("", tokenizer=TOK) == []
    assert chunk_document("   \n\n   ", tokenizer=TOK) == []


def test_offset_roundtrip():
    text = "\n\n".join(f"Paragraph number {i} has several words in it here." for i in range(30))
    chunks = chunk_document(text, tokenizer=TOK, target_tokens=20, max_tokens=30,
                            overlap_tokens=5)
    assert len(chunks) > 1
    for c in chunks:
        assert text[c.char_start:c.char_end] == c.text, "offset round-trip broken"
    # chunk_index is contiguous from 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_overlap_present():
    text = "\n\n".join(f"Para {i} word word word word word word word." for i in range(20))
    chunks = chunk_document(text, tokenizer=TOK, target_tokens=16, max_tokens=24,
                            overlap_tokens=8)
    assert len(chunks) >= 2
    # Consecutive chunks should overlap in char space (carried paragraphs).
    overlaps = [chunks[i + 1].char_start < chunks[i].char_end for i in range(len(chunks) - 1)]
    assert any(overlaps), "expected at least one overlapping chunk boundary"


def test_hard_split_oversized_paragraph():
    # One paragraph far larger than max_tokens must split into several chunks.
    big = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_document(big, tokenizer=TOK, target_tokens=40, max_tokens=50,
                            overlap_tokens=10)
    assert len(chunks) >= 4
    for c in chunks:
        assert big[c.char_start:c.char_end] == c.text


def test_max_chunks_per_doc():
    text = "\n\n".join(f"Para {i} some words here for tokens." for i in range(40))
    chunks = chunk_document(text, tokenizer=TOK, target_tokens=10, max_tokens=15,
                            overlap_tokens=3, max_chunks_per_doc=3)
    assert len(chunks) == 3


# ---------------------------------------------------------------------------
# WHERE + RRF unit tests (no model, no DB)
# ---------------------------------------------------------------------------

def test_build_where():
    assert build_where() is None
    assert build_where(jira=True) == "(path LIKE 'geedge_jira/%')"
    w = build_where(["a/", "b/"], lang="zh", raw_filter="ext = '.pdf'")
    assert "path LIKE 'a/%'" in w and "path LIKE 'b/%'" in w
    assert "lang = 'zh'" in w and "(ext = '.pdf')" in w


def test_rrf_fuse_orders_by_agreement():
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}, {"chunk_id": "z"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "x"}, {"chunk_id": "w"}]
    fused = rrf_fuse([a, b], rrf_k=60)
    ids = [row["chunk_id"] for row, _ in fused]
    # x and y appear in both lists near the top → they outrank singletons z, w.
    assert set(ids[:2]) == {"x", "y"}
    assert set(ids) == {"x", "y", "z", "w"}


# ---------------------------------------------------------------------------
# End-to-end (slow, CPU, needs model + lancedb)
# ---------------------------------------------------------------------------

def _run(args, work_dir):
    env = dict(os.environ, GEEDGE_WORK=str(work_dir))
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True)


def _query_json(args, work_dir):
    r = _run(["retrieval.query", *args, "--json"], work_dir)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.slow
def test_end_to_end(tmp_path):
    make_fixture(tmp_path)

    # Build index on CPU.
    r = _run(["retrieval.index", "--device", "cpu"], tmp_path)
    assert r.returncode == 0, r.stderr

    # Exact identifier via FTS → top hit is the hardware doc.
    docs = _query_json([IDENT_HARDWARE, "--mode", "fts", "--top-k", "5"], tmp_path)
    assert docs, "no FTS results for identifier"
    assert docs[0]["doc_id"] == "doc_hw"

    # Phrase identifier with a dot is recoverable.
    docs = _query_json([IDENT_DOMAIN, "--mode", "fts", "--top-k", "5"], tmp_path)
    assert any(d["doc_id"] in ("doc_domain", "doc_manual") for d in docs)

    # Chinese semantic query (hybrid) hits the DNS-hijack doc, not the cafeteria.
    docs = _query_json(["域名劫持 检测", "--mode", "hybrid", "--top-k", "5"], tmp_path)
    ids = [d["doc_id"] for d in docs]
    assert "doc_zh_dns" in ids
    assert ids[0] != "doc_zh_other"

    # --jira filter excludes non-jira paths.
    docs = _query_json(["DNS interception", "--mode", "fts", "--jira", "--top-k", "10"],
                       tmp_path)
    assert all(d["path"].startswith("geedge_jira/") for d in docs)

    # --show-doc returns full text.
    r = _run(["retrieval.query", "--show-doc", "doc_hw"], tmp_path)
    assert r.returncode == 0 and IDENT_HARDWARE in r.stdout


@pytest.mark.slow
def test_incremental_refresh(tmp_path):
    make_fixture(tmp_path)
    assert _run(["retrieval.index", "--device", "cpu"], tmp_path).returncode == 0

    # Rewrite one record in place with new text (new sha1).
    recs = fixture_records()
    for rec in recs:
        if rec["id"] == "doc_unrelated":
            rec["text"] = "Now this document mentions ZEBRAFISH telemetry uniquely."
            rec["char_count"] = len(rec["text"])
    shard = tmp_path / "text" / "shard_00000.jsonl"
    with shard.open("wb") as f:
        for rec in recs:
            f.write(orjson.dumps(rec) + b"\n")

    # Re-run without --rebuild: changed doc re-embedded, old term gone.
    assert _run(["retrieval.index", "--device", "cpu"], tmp_path).returncode == 0
    docs = _query_json(["ZEBRAFISH", "--mode", "fts", "--top-k", "5"], tmp_path)
    assert any(d["doc_id"] == "doc_unrelated" for d in docs)
    docs = _query_json(["vacation policy reimbursement", "--mode", "fts", "--top-k", "5"],
                       tmp_path)
    assert all(d["doc_id"] != "doc_unrelated" for d in docs)

    # Append a second shard → top-up only.
    append_shard(tmp_path, [{
        "id": "doc_added", "path": "geedge_docs/added.txt", "ext": ".txt",
        "lang": "en", "char_count": 40, "n_elements": 1,
        "text": "A newly appended MONGOOSE reference doc.", "error": "", "ocr": False,
    }])
    assert _run(["retrieval.index", "--device", "cpu"], tmp_path).returncode == 0
    docs = _query_json(["MONGOOSE", "--mode", "fts", "--top-k", "5"], tmp_path)
    assert any(d["doc_id"] == "doc_added" for d in docs)
