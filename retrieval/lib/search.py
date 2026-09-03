"""Hybrid search core — the future MCP surface too, so it stays free of CLI/print
concerns (query.py is the thin CLI, mcp_server.py would be another caller).

Three modes:
  fts    — LanceDB BM25 only. No model load; the fast path for exact identifiers
           (XDC02030000, doescan.xyz).
  dense  — bge-m3 cosine only.
  hybrid — both, Reciprocal-Rank-Fusion'd. Tries LanceDB's native hybrid first;
           on any failure (its signature has drifted across releases) falls back
           to running the two legs separately and fusing with the local rrf_fuse()
           below — which is kept regardless because it's deterministic and
           unit-testable.

Results are chunk hits grouped into DocHit rows: a document's score is its best
chunk's fused score; up to per_doc chunks are shown per doc.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional

# Path prefix that identifies JIRA-derived docs. Verify against real paths on the
# WSL box (`head` a shard); kept as a single constant so it's trivial to change.
JIRA_PATH_PREFIX = "geedge_jira/"


@dataclass
class ChunkHit:
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    score: float


@dataclass
class DocHit:
    doc_id: str
    path: str
    lang: str
    ext: str
    score: float
    chunks: List[ChunkHit] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# WHERE construction
# ---------------------------------------------------------------------------

def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_where(
    path_prefixes: Optional[List[str]] = None,
    jira: bool = False,
    lang: Optional[str] = None,
    raw_filter: Optional[str] = None,
) -> Optional[str]:
    """Assemble a LanceDB SQL WHERE from CLI flags. Prefixes OR together; lang
    and the raw filter AND on. Returns None when nothing constrains."""
    prefixes = list(path_prefixes or [])
    if jira and JIRA_PATH_PREFIX not in prefixes:
        prefixes.append(JIRA_PATH_PREFIX)

    clauses: List[str] = []
    if prefixes:
        ors = [f"path LIKE {_sql_str(p + '%')}" for p in prefixes]
        clauses.append("(" + " OR ".join(ors) + ")")
    if lang:
        clauses.append(f"lang = {_sql_str(lang)}")
    if raw_filter:
        clauses.append(f"({raw_filter})")
    return " AND ".join(clauses) if clauses else None


# ---------------------------------------------------------------------------
# Fusion + scoring
# ---------------------------------------------------------------------------

def rrf_fuse(result_lists: List[List[dict]], rrf_k: int = 60) -> List[tuple]:
    """Reciprocal Rank Fusion over several rank-ordered result lists.

    Each list is chunk-row dicts in descending relevance. Returns
    [(row, fused_score)] sorted by fused score. Unit-tested independently of any
    LanceDB call.
    """
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for results in result_lists:
        for rank, row in enumerate(results):
            cid = row["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            rows.setdefault(cid, row)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(rows[cid], sc) for cid, sc in fused]


def _row_score(row: dict) -> float:
    if "_relevance_score" in row and row["_relevance_score"] is not None:
        return float(row["_relevance_score"])
    if "_distance" in row and row["_distance"] is not None:
        return 1.0 - float(row["_distance"])  # cosine distance → similarity
    if "_score" in row and row["_score"] is not None:
        return float(row["_score"])
    return 0.0


# ---------------------------------------------------------------------------
# Leg queries
# ---------------------------------------------------------------------------

def _apply_where(q, where: Optional[str]):
    return q.where(where, prefilter=True) if where else q


def _fts_hits(store, query: str, where: Optional[str], fetch_k: int) -> List[dict]:
    q = _apply_where(store.chunks.search(query, query_type="fts"), where)
    return q.limit(fetch_k).to_list()


def _dense_hits(store, qv, where: Optional[str], fetch_k: int) -> List[dict]:
    q = _apply_where(store.chunks.search(qv).metric("cosine"), where)
    return q.limit(fetch_k).to_list()


def _hybrid_native(store, query, qv, where, fetch_k, rrf_k) -> List[dict]:
    from lancedb.rerankers import RRFReranker

    q = store.chunks.search(query_type="hybrid").text(query).vector(list(qv))
    q = _apply_where(q, where)
    try:
        reranker = RRFReranker(rrf_k)
    except TypeError:
        reranker = RRFReranker()
    return q.rerank(reranker).limit(fetch_k).to_list()


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _group(scored: List[tuple], top_k: int, per_doc: int) -> List[DocHit]:
    by_doc: dict[str, DocHit] = {}
    for row, score in scored:
        doc_id = row["doc_id"]
        hit = by_doc.get(doc_id)
        if hit is None:
            hit = DocHit(
                doc_id=doc_id,
                path=row.get("path") or "",
                lang=row.get("lang") or "",
                ext=row.get("ext") or "",
                score=score,
                chunks=[],
            )
            by_doc[doc_id] = hit
        hit.score = max(hit.score, score)
        hit.chunks.append(ChunkHit(
            chunk_index=int(row.get("chunk_index", 0)),
            char_start=int(row.get("char_start", 0)),
            char_end=int(row.get("char_end", 0)),
            text=row.get("text") or "",
            score=score,
        ))
    docs = list(by_doc.values())
    for d in docs:
        d.chunks.sort(key=lambda c: -c.score)
        d.chunks = d.chunks[:per_doc]
    docs.sort(key=lambda d: -d.score)
    return docs[:top_k]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def search(
    store,
    embedder,
    query: str,
    *,
    mode: str = "hybrid",
    where: Optional[str] = None,
    top_k: int = 10,
    per_doc: int = 2,
    rrf_k: int = 60,
    fetch_k: Optional[int] = None,
) -> List[DocHit]:
    fetch_k = fetch_k or max(50, top_k * 5)  # survive doc-grouping collapse

    if mode == "fts":
        scored = [(r, _row_score(r)) for r in _fts_hits(store, query, where, fetch_k)]
    elif mode == "dense":
        qv = embedder.encode_query(query)
        scored = [(r, _row_score(r)) for r in _dense_hits(store, qv, where, fetch_k)]
    elif mode == "hybrid":
        qv = embedder.encode_query(query)
        try:
            rows = _hybrid_native(store, query, qv, where, fetch_k, rrf_k)
            scored = [(r, _row_score(r)) for r in rows]
        except Exception as e:  # native hybrid signature drift → deterministic fallback
            print(f"[hybrid] native path unavailable ({type(e).__name__}: {e}); "
                  f"falling back to two-leg RRF", file=sys.stderr)
            fts = _fts_hits(store, query, where, fetch_k)
            dense = _dense_hits(store, qv, where, fetch_k)
            scored = rrf_fuse([fts, dense], rrf_k)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return _group(scored, top_k, per_doc)
