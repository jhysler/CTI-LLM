# Plan: `retrieval/` — Hybrid Semantic Retrieval Subsystem

## Context

The geedge-rag pipeline (scripts/01–09) does batch keyword triage per named topic
(config YAML → Stage 4 scan → tiering → LLM analysis). It cannot answer ad-hoc
semantic questions like "what JIRA issues reference X topic?" when the corpus
never uses the query's exact vocabulary. This plan adds a parallel, query-time
retrieval subsystem: embed the ~54k-doc corpus once locally (BGE-M3 on the
RTX 4080 / WSL2 box), store in LanceDB, and expose a hybrid (dense + BM25) query
CLI. The user explicitly wants this **outside `scripts/`** — new top-level
`retrieval/` directory; the JSONL shards are the only contract between the two.

Decisions already made with the user (fixed):
- Location: `retrieval/` in this repo; no imports in either direction with `scripts/`
- Model: BGE-M3 dense (FlagEmbedding), local, CUDA-focused; CPU fallback exists only for fixture tests
- Store: LanceDB at `$GEEDGE_WORK/vectors/` (disposable, rebuildable from shards)
- Search: **hybrid** — dense + LanceDB native FTS (BM25), RRF-fused; exact
  identifiers (`XDC02030000`, `doescan.xyz`, `YYDNS`) must match literally
- v1 interface: CLI query tool only (no MCP, no triage-CSV export — but don't design them out)

## Shard contract (verified against code)

`$GEEDGE_WORK/text/shard_*.jsonl` (GEEDGE_WORK defaults to `~/geedge-data`), one
JSON object per line, written by `scripts/03_extract_text.py::extract_one()`:
`{id, path, ext, lang, char_count, n_elements, text, error, ocr}`.
- `text` capped at `MAX_TEXT_CHARS = 2_000_000` (03_extract_text.py:73)
- Empty `text` records exist (extraction failures)
- **Quirk**: `09_ingest_manual.py` records omit `ocr`, and its error-record form
  omits `ext` too → all readers use `rec.get(...)` with defaults
- Shards mutate after the fact: `09_ingest_manual.py` appends new shards;
  `03_extract_text.py --reextract-low-text` rewrites lines **in place** →
  byte offsets can go stale; indexer must support incremental refresh
- Data lives only on the WSL box; this Mac has no shards → synthetic fixture
  tests are the primary local verification path

## Directory layout

```
retrieval/
  __init__.py
  README.md                # usage + WSL smoke checklist
  requirements.txt         # heavy deps, separate from root (torch, FlagEmbedding,
                           # lancedb, transformers, tabulate; orjson/tqdm/pyyaml matching root pins)
  index.py                 # indexer CLI: python -m retrieval.index
  query.py                 # query CLI:   python -m retrieval.query
  lib/
    __init__.py
    paths.py               # WORK/TEXT_DIR/VECTORS_DIR via GEEDGE_WORK idiom (copy from 04_triage.py)
    shards.py              # iter_records() + fetch_record() (port from 07_analyze.py)
    chunking.py            # tokenizer-based chunker
    embedder.py            # BGE-M3 wrapper, device auto-selection
    store.py               # LanceDB schemas, open/create, delete+add upsert, FTS index
    search.py              # hybrid search core (query.py is a thin CLI; future MCP surface)
  tests/
    fixtures.py            # synthetic shard generator
    test_smoke.py          # chunking unit tests + @pytest.mark.slow end-to-end
```

Run as `python -m retrieval.index` / `python -m retrieval.query` (package has
internal imports; `scripts/` files stay standalone). Root `requirements.txt`
gets one pointer comment only — its header already defers "Stage 5+ deps".

## Work items (implementation order)

### 1. Scaffolding
`retrieval/` tree, `lib/paths.py` (`WORK = Path(os.environ.get("GEEDGE_WORK", Path.home() / "geedge-data"))`,
`TEXT_DIR = WORK / "text"`, `VECTORS_DIR = WORK / "vectors"`), `retrieval/requirements.txt`.

### 2. `lib/shards.py` (no model — fully testable on Mac)
- `iter_records(text_dir) -> Iterator[(shard_path, byte_offset, record)]` —
  sorted shards, offset tracked on raw bytes; same approach as
  `build_doc_id_index()` in scripts/07_analyze.py:136
- `fetch_record(shard_path, byte_offset, expected_id) -> dict` — seek/readline/
  orjson; **verify `rec["id"] == expected_id`** and raise a clear
  "index stale — re-run indexer" error on mismatch (guards against in-place rewrites)

### 3. `lib/chunking.py` (no model needed for tests)
Tokenizer-based via `AutoTokenizer.from_pretrained("BAAI/bge-m3")` (fast XLM-R
tokenizer; same artifact FlagEmbedding downloads). Char heuristics are wrong for
this corpus: Chinese ≈1 token/char vs English ≈4 chars/token. Fast tokenizer's
`return_offsets_mapping=True` gives exact char back-references for free.

`chunk_document(text, target_tokens=800, max_tokens=1000, overlap_tokens=150) -> list[Chunk]`,
`Chunk = (chunk_index, char_start, char_end, text)`:
1. Split on `"\n\n"` (Stage 3's element joiner), keep char offsets per paragraph
2. Greedily pack paragraphs until next would exceed `target_tokens`
3. Single paragraph > `max_tokens` → hard-split into token windows with overlap
   (offsets mapping guarantees valid multibyte boundaries)
4. Paragraph-granular overlap ≥ `overlap_tokens` carried into next chunk
5. Invariant (assert in tests): `text[char_start:char_end] == chunk.text`

Empty-text records → zero chunks but still get a `docs` manifest row
(`status="empty"`) so incremental runs skip them. Stub `--max-chunks-per-doc`
(0 = unlimited).

### 4. `lib/embedder.py` + `lib/store.py`
**Embedder**: `Embedder(model="BAAI/bge-m3", device="auto", fp16=None, max_length=1024)`.
Device auto: cuda → cpu. fp16 default True on cuda only.
`encode_passages(texts, batch_size)` / `encode_query(text)` via
`FlagEmbedding.BGEM3FlagModel(return_dense=True, return_sparse=False, return_colbert_vecs=False)`;
1024-d normalized output. VRAM: ~570M params ≈ 1.2GB fp16; batch 32 × seq 1024
fits 16GB with big headroom; default `--batch-size 32`, expose flag.

**Store** — LanceDB db at `VECTORS_DIR`, two tables, explicit PyArrow schemas:

`chunks`: `chunk_id` (str, `f"{doc_id}:{chunk_index:04d}"`), `doc_id` (str),
`chunk_index` (int32), `char_start`/`char_end` (int32), `text` (str, FTS-indexed),
`path`/`ext`/`lang` (str, denormalized for filtering), `ocr` (bool,
`rec.get("ocr", False)`), `vector` (fixed_size_list<float32, 1024>).

`docs` (indexed manifest): `doc_id`, `path`, `ext`, `lang`, `char_count` (int64),
`text_sha1` (change detector for in-place OCR rewrites), `n_chunks` (int32),
`shard_file` (filename only — portable), `byte_offset` (int64), `status`
(`indexed`/`empty`), `indexed_at` (timestamp).

Indexes (built at end of an index run):
- FTS on `chunks.text`: `create_fts_index("text", replace=True, with_position=True)`,
  lowercase on, **stemming off** (literal identifiers), ascii-folding on,
  positions on (phrase match recovers `doescan.xyz` → `doescan xyz`).
  CJK caveat: default tokenizer doesn't segment Chinese — CJK recall rides on
  the dense leg in v1; leave a TODO for ngram/jieba evaluation, don't build it.
- Vector ANN: IVF_PQ, cosine, `num_partitions ≈ sqrt(n_rows)`, only when
  rows > 50k (fixture tests never build it; flat search is fine below that).

### 5. `retrieval/index.py`
```
python -m retrieval.index [--rebuild] [--device auto|cuda|cpu]
    [--batch-size 32] [--max-length 1024] [--limit N] [--dry-run]
    [--no-fts] [--target-tokens 800] [--overlap-tokens 150]
```
1. `--rebuild`: drop both tables, start clean (index is declared disposable)
2. Load manifest `{doc_id: (text_sha1, status)}`
3. Single pass over `iter_records()` (tqdm per shard); classify each record:
   **unchanged** (same sha1) → refresh `(shard_file, byte_offset)` if moved, skip;
   **changed** (sha1 differs — OCR re-extract case) → stage for re-index;
   **new** → stage; **empty text** → manifest row `status="empty"`
4. Staged docs in batches: `chunks_tbl.delete(f"doc_id IN (...)")` for changed ids
   (delete-then-add; `merge_insert` can't remove a shrunk doc's surplus chunks),
   then chunk → embed (buffer passages across docs to fill GPU batches) → add;
   upsert `docs` row via `merge_insert` on `doc_id` **after** chunks land
   (crash-resumable, same spirit as 03's done-set). Flush every ~2k chunks.
5. End: rebuild FTS (`replace=True`), conditional ANN build, `optimize()`
6. `--dry-run` prints new/changed/unchanged/empty counts, exits before GPU;
   `--limit N` caps staged docs; summary line in the pipeline's print style

### 6. `lib/search.py` + `retrieval/query.py`
```
python -m retrieval.query "what JIRA issues reference doescan.xyz" \
    [--path-prefix geedge_jira/]... [--jira] [--lang zh|en] [--filter "SQL"]
    [--top-k 10] [--per-doc 2] [--mode hybrid|dense|fts] [--rrf-k 60]
    [--device ...] [--show-doc DOC_ID] [--full-context] [--json]
```
Core (future MCP surface):
```python
def search(db, embedder, query, *, mode="hybrid", where=None,
           top_k=10, per_doc=2, rrf_k=60, fetch_k=50) -> list[DocHit]
# DocHit: doc_id, path, lang, ext, score, chunks: [ChunkHit(chunk_index, char_start, char_end, text, score)]
```
- `where` built from flags: `--path-prefix p` → `path LIKE 'p%'` (OR-joined);
  `--jira` = sugar for `--path-prefix geedge_jira/` (verify actual prefix
  against real paths on the WSL box; keep the constant easy to change);
  `--lang` → equality; `--filter` raw-SQL AND-ed on
- Hybrid: `chunks_tbl.search(query_type="hybrid")` + `RRFReranker(K=rrf_k)` +
  `.where(where, prefilter=True)` + `.limit(fetch_k)`, `fetch_k = max(50, top_k*5)`
  to survive doc-grouping. **Verify installed lancedb's hybrid API signature**
  (it has shifted across releases); if hybrid+prefilter misbehaves, run the two
  legs separately with the same `where` and fuse with a local RRF — keep that
  fusion function in `search.py` regardless (unit-testable)
- `--mode fts` needs no model load (fast path for exact-identifier lookups);
  dense/hybrid pay ~5–15s model load per invocation — print stderr notice
- Group chunks by `doc_id`; doc score = best fused chunk score; top `--per-doc`
  chunks per doc, top `--top-k` docs
- Output: terminal table (rank, score, lang, path) + snippets beneath —
  whitespace-collapse like `find_snippet()` in 04_triage.py, ~200 chars,
  prefixed `[chunk 3 @ 14200..15800]`
- `--full-context`: fetch full record via manifest `(shard_file, byte_offset)`
  + `fetch_record()`, print ±1500 chars around `char_start`
- `--show-doc DOC_ID`: no search, dump one full record
- `--json`: DocHit list to stdout (wire shape for future MCP/export)

### 7. Tests + README
`tests/fixtures.py::make_fixture(work_dir)` writes `text/shard_00000.jsonl` with
~10 handcrafted records matching the real schema: doc containing `XDC02030000`;
doc with `doescan.xyz` + `YYDNS`; two Chinese docs (one DNS-hijack-themed, one
unrelated); mixed CJK/EN doc; long doc (>6k tokens → multi-chunk); empty-text
failure record; a 09-style record missing the `ocr` key; paths under both
`geedge_jira/...` and other prefixes.

`tests/test_smoke.py` (pytest, `GEEDGE_WORK` → tmp dir):
- Chunking unit tests (no model, ungated): offset round-trip, overlap, hard
  split, empty → 0 chunks
- `@pytest.mark.slow` end-to-end on `--device cpu`: index fixture; assert row
  counts; `--mode fts` `XDC02030000` → rank 1; phrase `doescan.xyz` found;
  hybrid Chinese query hits the right doc; `--path-prefix geedge_jira/`
  excludes non-jira; `--show-doc` returns full text
- Incremental: rewrite one record in place (new sha1) → re-run without
  `--rebuild` → old chunks gone, new present, others untouched; append a second
  shard → re-run → top-up only

`retrieval/README.md`: usage examples + WSL operator checklist —
`pip install -r retrieval/requirements.txt` (CUDA torch),
`python -m retrieval.index --dry-run` (counts sanity vs ~54k), full build, then
three known-answer queries (literal identifier, Chinese semantic, `--jira`-filtered)
cross-checked against existing `triage/dns4cn.csv` hits.

## Verification

1. **Mac (this machine, no corpus)**: `pytest retrieval/tests/` — chunking unit
   tests always; `-m slow` end-to-end on CPU (one-time ~2.3GB model download)
2. **WSL box (real data)**: README checklist above — dry-run counts, full index
   build (well under an hour on the 4080), three known-answer queries validated
   against Stage 4 triage output

## Platform scope

CUDA on the WSL2 / RTX 4080 box is the sole performance target — do not spend
effort on MPS tuning or Apple Silicon support. The cheap hygiene that keeps a
future port possible stays (it costs nothing): device selection lives only in
`Embedder(device="auto")` → cuda → cpu, fp16 decided per device, all paths
derive from `GEEDGE_WORK`, and `shard_file` is stored filename-only so the
`vectors/` dir is copyable. The `--device cpu` path exists for the fixture
tests, not as a supported runtime.

## Future hooks (design notes only — no code in v1)

- **Triage-CSV export**: later `retrieval/export.py` calls `search(top_k=large)`,
  writes 04_triage.py's CSV schema to `$GEEDGE_WORK/triage/semantic_<name>.csv`
  → consumable by 06/07 unchanged (why `DocHit` carries everything that schema needs)
- **MCP server**: later `retrieval/mcp_server.py` holds Embedder + DB warm,
  exposes `search()`/`fetch_record()` as tools; `--json` already defines the shape

## Reference files
- scripts/03_extract_text.py — record schema, MAX_TEXT_CHARS, done-set resumability, in-place rewrite mode
- scripts/07_analyze.py:136 — byte-offset index idiom to port into lib/shards.py
- scripts/04_triage.py — GEEDGE_WORK idiom, CLI style, find_snippet(), triage CSV schema
- scripts/09_ingest_manual.py — shard-append path; records missing `ocr`/`ext` keys
- requirements.txt — add pointer comment only
