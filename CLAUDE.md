# CLAUDE.md

Guidance for Claude Code working in this repo.

## What this is

A local-first pipeline for analyzing the Geedge Networks / MESA Lab document
leak (~600GB raw, 54,761 extracted docs). Solo research project. The operator
**does not read Chinese**, and the load-bearing source material is Chinese —
treat translation fidelity as a standing correctness risk, not a nicety.

**Status: paused as of 2026-07-18.** Docs were brought into sync with the code
at that point. Prefer keeping them in sync over adding features.

## Two workflows — don't conflate them

| `scripts/` (Stages 1–9) | `retrieval/` |
| --- | --- |
| Batch keyword triage per named topic | Ad-hoc semantic search over the whole corpus |
| Driven by `config/<topic>.yaml` | Driven by a query string |
| Anthropic API (costs money) | Local Ollama (free) |

They share **exactly one contract**: the Stage 3 JSONL shards at
`$GEEDGE_WORK/text/shard_*.jsonl`. **No imports run in either direction — keep
it that way.** `retrieval/` is not a numbered stage and shouldn't become one.

## Invariants worth not breaking

- **All paths derive from `$GEEDGE_WORK`** (`retrieval/lib/paths.py`, and the
  same idiom in `scripts/04_triage.py`). Never hardcode an absolute path.
- **`vectors/` is disposable** — always rebuildable from the shards. It stores
  `shard_file` as a *filename only* so the directory stays copyable between
  machines. Don't store absolute paths in it.
- **Stages 1–3 are topic-agnostic.** Verified by grep — no topic terms appear in
  them. A new research topic should need only a new `config/<topic>.yaml` plus a
  re-run of Stages 4 → 5 → 6 → 7.
- **`scripts/` files are standalone** (run directly); `retrieval/` is a package
  (run via `python -m retrieval.<mod>`).
- **`pdfminer.six==20221105` is pinned deliberately** — newer releases changed
  layout behavior that `unstructured` depends on.

## Before trusting retrieval output

Read [`docs/known-issues.md`](docs/known-issues.md). Two things bite repeatedly:

1. **A `FOUND: no` from `seed_ask.py` may be a retrieval miss, not an absence.**
   When a `no` contradicts known domain facts, probe individual-term search rank
   directly. **Do not** re-run generation N times and take a majority vote — that
   cannot surface a doc which never entered the context window.
2. **Blending terms into one embedding query dilutes it.** Search one term per
   leg and fuse by rank (`rrf_rescore()`), not by raw score — raw scores aren't
   comparable across legs of different query lengths.

Also: CJK lexical recall rides entirely on the dense leg, because LanceDB's FTS
tokenizer doesn't segment Chinese. For named entities and identifiers, prefer
`--mode fts`; query in the source material's language where possible.

## Verification reality

Most of this cannot be tested without the corpus, which is only on the WSL box.

```bash
pytest retrieval/tests/ -m "not slow"   # chunking/WHERE/RRF units — no model, no corpus
pytest retrieval/tests/ -m slow         # CPU end-to-end on synthetic fixtures (~2.3GB download)
```

Anything touching real retrieval quality needs the box and the known-answer
queries in [`retrieval/README.md`](retrieval/README.md). Don't claim a retrieval
behavior is verified on the strength of the fixture tests alone.

## Cost discipline

Stage 7 spends real money (defaults to `claude-opus-4-7` on both passes). The
DNS4CN run came to ~$15 against a $100 ceiling. **Always `--dry-run` first** and
report the estimate before spending.

## Doc map

| File | Covers |
| --- | --- |
| [`README.md`](README.md) | Pipeline, both architecture diagrams, stage details, run order |
| [`retrieval/README.md`](retrieval/README.md) | `retrieval/` CLI surface: index, query, ask, seed_ask |
| [`docs/portability.md`](docs/portability.md) | Moving machines, backups, Mac setup, the MPS gap |
| [`docs/new-topic.md`](docs/new-topic.md) | Repurposing Workflow A for a new research target |
| [`docs/known-issues.md`](docs/known-issues.md) | Open retrieval-correctness questions |
| [`docs/retrieval-plan.md`](docs/retrieval-plan.md) | Original design plan — historical, keep as built-record |
| [`docs/retrieval-porting.md`](docs/retrieval-porting.md) | First-time `retrieval/` setup on the WSL/4080 box |

## Publishing

Findings are published at <https://www.hysler.net/DNS4CN/> under the operator's
own byline. Public writeups lead with **CTI tradecraft**, treat LLM use as a tool
under analyst control rather than the story, keep the stated limitations intact
(they're a credibility signal for this audience), and **contain no code** — no
scoring constants, paths, or repo internals.
