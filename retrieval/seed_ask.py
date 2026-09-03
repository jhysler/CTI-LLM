#!/usr/bin/env python3
"""
Seed Ask CLI: run a fixed battery of grounded, schema-output topic prompts
(retrieval/seed_prompts/*.yaml) against the indexed corpus.

Unlike ask.py's free-form questions, each topic here fills a shared
hard-grounding template (NOT FOUND escape hatch, quote-then-translate,
PROPOSAL/DEPLOYED/UNCLEAR tagging, fixed OUTPUT schema) tuned for small local
models — see the template docstring below for why. The template fully
replaces ask.py's build_system_prompt(); it isn't appended to it.

The retrieval query per topic is QUESTION plus its TARGET TERMS, not QUESTION
alone: TERMS carry the Chinese-language anchors, and a Chinese-only chunk
often won't surface from an English-only query on a small embedder.

A topic flagged `use_union_context: true` (the attribution sweep, by
convention run last) skips its own search and instead reuses the deduplicated
union of doc chunks retrieved by every prior topic in the run — that's how it
cross-checks names against systems already surfaced elsewhere in the corpus.

Usage:
  python -m retrieval.seed_ask
  python -m retrieval.seed_ask --topic dns4cn --topic shield_cube
  python -m retrieval.seed_ask --section A --section B -o report.md
  python -m retrieval.seed_ask --topics-file retrieval/seed_prompts/geedge_mesa.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
import yaml

from retrieval import ask
from retrieval.lib import paths
from retrieval.lib.chunking import get_tokenizer
from retrieval.lib.embedder import Embedder
from retrieval.lib.search import DocHit, build_where, search
from retrieval.lib.store import Store

DEFAULT_TOPICS_FILE = Path(__file__).parent / "seed_prompts" / "geedge_mesa.yaml"

# Reserve for TEMPLATE/question/fields text (~500-1000 tok) plus the model's
# generated answer (~600 tok target) and slack, since get_tokenizer() (bge-m3)
# is a proxy for gemma4's own tokenizer, not an exact count.
PROMPT_OVERHEAD_TOKENS = 800
OUTPUT_RESERVE_TOKENS = 1200

TEMPLATE = """You are a retrieval analyst. Answer ONLY from the CONTEXT below. The context is
excerpts from a leaked Chinese document corpus; some text is Chinese.

TOPIC: {topic}
QUESTION: {question}
TARGET TERMS (evidence usually near these): {terms}

RULES:
- Use only facts stated in CONTEXT. Do not use outside knowledge. Do not guess.
- If CONTEXT does not answer the question, output exactly: NOT FOUND
- For every claim, quote the exact source snippet (Chinese if Chinese) and give a
  short English translation.
- Mark each claim PROPOSAL (planned/designed) or DEPLOYED (built/in use). If the
  context does not say, mark UNCLEAR.
- Do not merge claims from unrelated snippets.
- Write everything except the source quotes themselves in {answer_lang} — field
  labels' values, CONFIDENCE, and GAPS must all be in {answer_lang} even though
  the CONTEXT is Chinese.

OUTPUT (this schema only):
FOUND: yes | no
{fields}
EVIDENCE:
- "<source quote>" -> "<english>" [PROPOSAL|DEPLOYED|UNCLEAR]
CONFIDENCE: high | medium | low
GAPS: <what the context does not say>

CONTEXT:
{context}"""


def load_topics(path: Path, section_ids: list[str], topic_ids: list[str]) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    topics = []
    for section in data["sections"]:
        if section_ids and section["id"] not in section_ids:
            continue
        for t in section["topics"]:
            if topic_ids and t["id"] not in topic_ids:
                continue
            topics.append({**t, "section_id": section["id"], "section_name": section["name"]})
    return topics


def render_prompt(topic: dict, context: str, answer_lang: str) -> str:
    fields = "\n".join(f"{f}:" for f in topic["fields"])
    return TEMPLATE.format(
        topic=topic["topic"],
        question=topic["question"],
        terms=", ".join(topic["terms"]),
        fields=fields,
        context=context,
        answer_lang=answer_lang,
    )


def merge_docs(union: dict[str, DocHit], docs: list[DocHit], per_doc: int) -> None:
    """Fold newly retrieved docs into the running union, deduping chunks by
    chunk_index and keeping the best `per_doc` per doc — used to build topic
    F's context from every prior topic's hits without re-searching."""
    for d in docs:
        existing = union.get(d.doc_id)
        if existing is None:
            union[d.doc_id] = DocHit(
                doc_id=d.doc_id, path=d.path, lang=d.lang, ext=d.ext,
                score=d.score, chunks=list(d.chunks),
            )
            continue
        existing.score = max(existing.score, d.score)
        by_index = {c.chunk_index: c for c in existing.chunks}
        for c in d.chunks:
            prior = by_index.get(c.chunk_index)
            if prior is None or c.score > prior.score:
                by_index[c.chunk_index] = c
        existing.chunks = sorted(by_index.values(), key=lambda c: -c.score)[:per_doc]


def rrf_rescore(legs: list[list[DocHit]], topic_hits: dict[str, DocHit], rrf_k: int = 60) -> None:
    """Overwrite topic_hits[*].score with a rank-based score fused across the given
    ranked leg lists (one list per search call), instead of comparing each leg's raw
    hybrid scores directly. A broad natural-language question search produces
    structurally higher raw scores than a short 2-6 character proper-noun term
    search, so raw-score merging lets the question leg's generic hits crowd out
    exactly the precise term-leg hits the per-term split (see run_topic) exists to
    surface — confirmed on Shield-Cube: the one doc literally titled combining
    DNS4CN and 盾立方 ranked #45 by raw score, entirely on account of the scale
    mismatch, despite ranking well within its own term-leg searches."""
    rrf_scores: dict[str, float] = {}
    for leg in legs:
        for rank, d in enumerate(leg):
            rrf_scores[d.doc_id] = rrf_scores.get(d.doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for doc_id, hit in topic_hits.items():
        hit.score = rrf_scores.get(doc_id, 0.0)


def ask_ollama_raw(model: str, user_content: str, num_ctx: int, temperature: float, timeout: int) -> str:
    resp = requests.post(
        ask.OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": temperature},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def run_topic(store: Store, embedder, topic: dict, union: dict[str, DocHit], args) -> tuple[str, list[DocHit]]:
    if topic.get("use_union_context"):
        docs = sorted(union.values(), key=lambda d: -d.score)[:args.union_top_k]
        if not docs:
            print(f"  [warn] {topic['id']}: no accumulated context from prior topics, "
                  f"falling back to a fresh search", file=sys.stderr)
            docs = None  # fall through to search below
    else:
        docs = None

    if docs is None:
        # One search per individual term, plus one on the question, all merged: this
        # embedder doesn't treat a multi-term query as "the union of these signals" —
        # it produces one averaged vector, and blending in even a single extra term
        # (any language) is enough to dilute out a doc that keys almost entirely on
        # one proper noun. Confirmed directly: "912" alone surfaces the one doc
        # literally named after this project at #1; "912 新疆" already loses it, let
        # alone the full question+terms blend. Splitting to one term = one query is
        # the only version of this that reliably surfaces single-anchor docs.
        where = build_where(args.path_prefix, args.jira, args.lang, args.raw_filter)
        topic_hits: dict[str, DocHit] = {}
        legs: list[list[DocHit]] = []
        for query in [topic["question"], *topic["terms"]]:
            leg_docs = search(store, embedder, query, mode=args.mode, where=where,
                               top_k=args.top_k, per_doc=args.per_doc, fetch_k=args.fetch_k)
            legs.append(leg_docs)
            merge_docs(topic_hits, leg_docs, args.per_doc)
        # Raw scores aren't comparable across legs (see rrf_rescore docstring) —
        # rescore by rank before picking the final top_k.
        rrf_rescore(legs, topic_hits)
        docs = sorted(topic_hits.values(), key=lambda d: -d.score)[:args.top_k]
        if not topic.get("use_union_context"):
            merge_docs(union, docs, args.per_doc)

    if not docs:
        return "FOUND: no\nGAPS: no documents retrieved for this topic", []

    # Topic density varies too much across this corpus for any fixed top_k/per_doc/
    # fetch_k combo to reliably fit num_ctx (measured 17k-71k tokens across topics
    # at identical settings) — an oversized prompt doesn't error, it silently loses
    # its front (the instructions) to llama.cpp's --context-shift, which produced
    # both blank answers and one case of the model answering an unrelated question
    # found deep in the surviving tail. Trim the lowest-scored docs until the real
    # tokenizer count fits, instead of trusting the retrieval knobs to self-limit.
    tok = get_tokenizer()
    budget = args.num_ctx - PROMPT_OVERHEAD_TOKENS - OUTPUT_RESERVE_TOKENS
    context = ask.gather_context(store, docs)
    n_tok = len(tok(context, add_special_tokens=False)["input_ids"])
    while n_tok > budget and len(docs) > 1:
        docs = docs[:-1]  # docs is sorted by score descending; drop the weakest
        context = ask.gather_context(store, docs)
        n_tok = len(tok(context, add_special_tokens=False)["input_ids"])
    if n_tok > budget:
        print(f"  [warn] {topic['id']}: context still {n_tok} tokens after trimming to "
              f"1 doc (budget {budget}) — consider raising --num-ctx", file=sys.stderr)

    if args.show_context:
        print(f"--- retrieved context: {topic['id']} ---", file=sys.stderr)
        print(context, file=sys.stderr)
        print("--- end context ---\n", file=sys.stderr)

    prompt = render_prompt(topic, context, args.answer_lang)
    print(f"Asking {args.model} about {topic['id']}...", file=sys.stderr)
    answer = ""
    for attempt in range(2):  # empty completions happen occasionally with no exception raised
        try:
            answer = ask_ollama_raw(args.model, prompt, args.num_ctx, args.temperature, args.timeout)
        except requests.exceptions.RequestException as e:
            print(f"  [error] {topic['id']}: Ollama call failed: {e}", file=sys.stderr)
            answer = f"FOUND: no\nGAPS: Ollama call failed ({type(e).__name__}: {e})"
            break
        if answer.strip():
            break
        print(f"  [warn] {topic['id']}: empty completion (attempt {attempt + 1}/2)", file=sys.stderr)
    if not answer.strip():
        answer = "FOUND: no\nGAPS: model returned an empty response after retrying"
    return answer, docs


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the Geedge/MESA seed-prompt battery against the indexed corpus.")
    ap.add_argument("--topics-file", type=Path, default=DEFAULT_TOPICS_FILE,
                    help="YAML file of sections/topics (default: bundled geedge_mesa.yaml)")
    ap.add_argument("--section", action="append", default=[],
                    help="restrict to section id(s) (A-F, repeatable)")
    ap.add_argument("--topic", action="append", default=[],
                    help="restrict to topic id(s) (repeatable)")
    ap.add_argument("--model", default=ask.DEFAULT_MODEL, help="Ollama model name")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "fts"])
    ap.add_argument("--top-k", type=int, default=12, help="docs to retrieve as context, per topic")
    ap.add_argument("--per-doc", type=int, default=6,
                    help="chunks per doc to include as context (more = thorougher, slower)")
    ap.add_argument("--fetch-k", type=int, default=100,
                    help="raw candidate chunks pulled before per-doc grouping (default 100, up "
                         "from search()'s own max(50, top_k*5)=60 default — raise this, not "
                         "--per-doc, to sample more deeply within already-retrieved docs; "
                         "--per-doc only caps what this pool already contains. 150+ starts "
                         "running out of --num-ctx headroom on dense topics)")
    ap.add_argument("--union-top-k", type=int, default=40,
                    help="max docs from the A-E union to feed the attribution-sweep topic")
    ap.add_argument("--answer-lang", default="English",
                    help="language the model must answer in, regardless of source-excerpt language")
    ap.add_argument("--path-prefix", action="append", default=[],
                    help="restrict to paths starting with this (repeatable, OR-joined)")
    ap.add_argument("--jira", action="store_true", help="sugar for --path-prefix geedge_jira/")
    ap.add_argument("--lang", help="restrict retrieval to a detected language (e.g. zh, en)")
    ap.add_argument("--filter", dest="raw_filter", help="raw SQL predicate, AND-ed on")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--num-ctx", type=int, default=65536,
                    help="Ollama context window in tokens (raise further for large --top-k/--per-doc)")
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="keep <= 0.2 per the template's grounding design")
    ap.add_argument("--timeout", type=int, default=900,
                    help="Ollama request timeout in seconds (large --num-ctx needs more headroom)")
    ap.add_argument("--show-context", action="store_true",
                    help="print each topic's retrieved context (to stderr) before the answer")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="write results here as Markdown (default: print to stdout only)")
    args = ap.parse_args()

    if not paths.VECTORS_DIR.exists():
        print(f"No index at {paths.VECTORS_DIR}. Run `python -m retrieval.index` first.",
              file=sys.stderr)
        return 1
    if not args.topics_file.exists():
        print(f"No such file: {args.topics_file}", file=sys.stderr)
        return 1

    topics = load_topics(args.topics_file, args.section, args.topic)
    if not topics:
        print("No topics matched --section/--topic filters.", file=sys.stderr)
        return 1

    store = Store.open(paths.VECTORS_DIR)
    embedder = None if args.mode == "fts" else Embedder(device=args.device)

    # Written incrementally (flushed after every topic) rather than assembled in
    # memory and saved once at the end: a run this long-lived is bound to hit
    # a slow/timed-out topic eventually, and losing every prior topic's answer
    # to one bad topic is worse than a same-topic retry.
    out_fh = args.output.open("w", encoding="utf-8") if args.output else None

    union: dict[str, DocHit] = {}
    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] {topic['section_id']}: {topic['topic']}", file=sys.stderr)
        answer, docs = run_topic(store, embedder, topic, union, args)

        block = [f"## {topic['section_id']}. {topic['topic']}\n", f"{answer}\n",
                 "\n**Sources:**\n"]
        block.extend(f"- {d.path}\n" for d in docs)
        block.append("\n---\n\n")
        block_text = "".join(block)

        print(block_text)
        if out_fh:
            out_fh.write(block_text)
            out_fh.flush()

    if out_fh:
        out_fh.close()
        print(f"Wrote {len(topics)} topic answers to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
