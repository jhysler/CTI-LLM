# `retrieval/` — Hybrid Semantic Retrieval

A query-time semantic search layer over the geedge-rag corpus, parallel to the
`scripts/` keyword-triage pipeline. It embeds the JSONL shards
(`$GEEDGE_WORK/text/shard_*.jsonl`) once with **BGE-M3**, stores dense vectors +
a **BM25 FTS** index in **LanceDB** (`$GEEDGE_WORK/vectors/`), and answers ad-hoc
queries — including ones whose vocabulary never appears literally in the corpus —
via **hybrid** (dense + lexical, RRF-fused) search.

The JSONL shards are the only contract with `scripts/`: no imports run in either
direction. The `vectors/` directory is disposable and fully rebuildable from the
shards.

## Install (WSL2 / RTX 4080 box)

First-time setup on the box? See the step-by-step runbook:
[`docs/retrieval-porting.md`](../docs/retrieval-porting.md).
Setting this up on a *different* machine (Mac, or after a rebuild)? See
[`docs/portability.md`](../docs/portability.md) — note that `resolve_device()`
has no MPS branch, so Apple Silicon falls back to CPU for index builds.

CUDA is the only performance target. Install the matching CUDA torch wheel
**first**, then the rest:

```bash
# example for CUDA 12.1 — match your driver
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r retrieval/requirements.txt
```

## Build the index

```bash
# 1. Sanity-check the classification counts against ~54k docs — no GPU touched
python -m retrieval.index --dry-run

# 2. Full build (well under an hour on the 4080)
python -m retrieval.index

# Incremental re-runs are automatic: unchanged docs are skipped by text sha1,
# OCR re-extracts (03 --reextract-low-text) and appended shards
# (09_ingest_manual.py) are picked up. Force a clean rebuild with --rebuild.
python -m retrieval.index --rebuild
```

Useful flags: `--limit N` (cap staged docs, for a smoke build), `--batch-size`,
`--device cuda|cpu`, `--no-fts`, `--target-tokens` / `--overlap-tokens`.

## Query

```bash
# Exact identifier — FTS only, no model load (fast)
python -m retrieval.query "XDC02030000" --mode fts

# Chinese semantic query, hybrid, restricted to JIRA-derived docs
python -m retrieval.query "域名劫持 检测" --jira

# Filters: --path-prefix (repeatable), --lang zh, --filter "ext = '.pdf'"
python -m retrieval.query "resolver telemetry" --path-prefix geedge_docs/ --top-k 20

# Inspect: ±1500 chars of source around each hit, or a full record
python -m retrieval.query "盾立方" --full-context
python -m retrieval.query --show-doc <doc_id>

# Machine-readable (wire shape for a future MCP server / CSV export)
python -m retrieval.query "DNS injection" --json
```

Modes: `hybrid` (default), `dense`, `fts`. `--mode fts` needs no model; `dense`
and `hybrid` pay a ~5–15s model load per invocation (printed to stderr).

## Ask (RAG answers via a local LLM)

`query.py` returns ranked hits. `ask.py` goes one step further: it retrieves,
expands each hit with ±1500 chars of surrounding source text, and asks a local
**Ollama** model to answer from that context alone, citing source paths. No
corpus text leaves the box, and there's no per-query API cost.

```bash
# Ollama must be running; default model is gemma4:12b
python -m retrieval.ask "What is XDC02030000 about?"
python -m retrieval.ask "域名劫持的解决方案是什么" --jira
python -m retrieval.ask "..." --model qwen2.5:14b --top-k 8 --show-context
```

It takes all of `query.py`'s retrieval flags (`--jira`, `--lang`, `--path-prefix`,
`--filter`, `--mode`, `--top-k`, `--per-doc`) plus `--model`, `--num-ctx`
(default 24576 — Ollama's own default silently truncates multi-doc CJK context),
and `--answer-lang` (default English).

Answers are forced English by default with a *literal-translation* rigor mode:
the prompt requires the original Chinese term in parentheses after each
translated technical term, so a reader who doesn't read Chinese can still
spot-check a rendering against the source. Temperature is pinned to 0.1 —
at Ollama's default of 1.0, models drifted into general training knowledge
instead of the provided excerpts.

## Seed Ask (batched grounded topic battery)

`seed_ask.py` runs a fixed battery of topic prompts from
[`seed_prompts/`](seed_prompts/) instead of a free-form question. Each topic
fills a hard-grounding template — `NOT FOUND` escape hatch, quote-then-translate,
`PROPOSAL`/`DEPLOYED`/`UNCLEAR` tagging, and a fixed output schema — tuned to
keep small local models from confabulating.

```bash
python -m retrieval.seed_ask                          # full battery
python -m retrieval.seed_ask --topic dns4cn --topic shield_cube
python -m retrieval.seed_ask --section A -o report.md
```

Two retrieval behaviors here differ from `ask.py`, both the result of debugging
real false negatives (see [`docs/known-issues.md`](../docs/known-issues.md)):

- **One search leg per target term, not one blended query.** Blending terms into
  a single embedding averages them into a vector that matches neither; adding
  even one extra term was enough to lose a doc that keys on a single proper noun.
- **`rrf_rescore()` fuses those legs by rank, not raw score.** A broad
  natural-language question produces structurally higher hybrid scores than a
  short proper-noun query, so raw-score merging let generic hits crowd out
  correct ones.

A topic marked `use_union_context: true` skips its own search and reuses the
deduplicated union of every prior topic's retrieved chunks — that's how the
attribution sweep cross-checks names against systems surfaced elsewhere.

Reports are written to `$GEEDGE_WORK/rag/<battery>_<timestamp>.md`.

> **Read the caveat before trusting a `no`.** `docs/known-issues.md` documents
> an open question: an unknown share of `FOUND: no` verdicts may be undiagnosed
> retrieval misses rather than genuine absences. Probe individual-term rank
> directly before believing one — re-running generation cannot surface a doc
> that never entered the context.

## WSL operator smoke checklist

1. `python -m retrieval.index --dry-run` — new/changed/unchanged/empty counts
   should sum to roughly the shard doc count (~54k).
2. `python -m retrieval.index` — full build; note the chunk/doc totals.
3. Three known-answer queries, cross-checked against `triage/dns4cn.csv`:
   - literal identifier: `python -m retrieval.query "XDC02030000" --mode fts`
   - Chinese semantic: `python -m retrieval.query "域名劫持" --lang zh`
   - JIRA-scoped: `python -m retrieval.query "DNS interception" --jira`

If a query errors with an "index stale" message, a shard was rewritten after
indexing — just re-run `python -m retrieval.index`.

## Tests (this repo, no corpus required)

```bash
# Fast: chunking + WHERE + RRF unit tests, no model
pytest retrieval/tests/ -m "not slow"

# Full end-to-end on CPU (one-time ~2.3GB bge-m3 download, builds a tmp index)
pytest retrieval/tests/ -m slow
```

## Layout

```text
retrieval/
  index.py            # indexer CLI    (python -m retrieval.index)
  query.py            # query CLI      (python -m retrieval.query)
  ask.py              # RAG answer CLI (python -m retrieval.ask)     — Ollama
  seed_ask.py         # topic battery  (python -m retrieval.seed_ask) — Ollama
  seed_prompts/
    geedge_mesa.yaml  # the topic battery: sections, questions, target terms
  lib/
    paths.py          # GEEDGE_WORK-derived paths
    shards.py         # JSONL shard reader (byte-offset index; stale-index guard)
    chunking.py       # tokenizer-based chunker
    embedder.py       # BGE-M3 dense wrapper, device auto-select (CUDA → CPU)
    store.py          # LanceDB schemas, upsert, FTS/ANN index build
    search.py         # hybrid search core (shared by all four CLIs)
  tests/              # fixtures + smoke tests
```

`search.py` is the single retrieval core; `query.py`, `ask.py`, and
`seed_ask.py` are all thin layers over it that differ only in what they do with
the hits. `ask.py` additionally exposes `answer_one()` and `add_common_args()`,
which `seed_ask.py` reuses so a battery run pays the Store/Embedder load once.

## Notes & known limits

- **CJK lexical recall** rides the dense leg in v1: LanceDB's FTS tokenizer does
  not segment Chinese. Evaluating an ngram/jieba tokenizer is a TODO, not built.
- **Vector ANN** (IVF_PQ) is only built past 50k chunks; below that LanceDB does
  an exact flat search, which is both faster to build and more accurate.
- The `--device cpu` path exists for the fixture tests, not as a supported
  runtime.
- **Cross-lingual query/doc mismatch**: querying in English for a topic that's
  predominantly documented in Chinese (e.g. a place name like "Xinjiang" vs.
  "新疆") can look inconsistent as you vary `--top-k`/`--per-doc` — FTS can't
  match across languages at all, and bge-m3's dense leg often doesn't bridge a
  proper noun strongly enough to survive hybrid RRF fusion, so the relevant
  doc's fused rank can sit right at the `--top-k` cutoff. Query in the source
  material's language when possible, and prefer `--mode fts` for named
  entities/identifiers over hybrid.
