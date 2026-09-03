#!/usr/bin/env python3
"""
Ask CLI: retrieval-augmented question answering over the indexed corpus.

Retrieves top-k relevant docs via retrieval.lib.search.search(), expands each
hit's best chunk with surrounding source text (same ±context idea as query.py's
--full-context), and asks a local Ollama model to answer using only that
context, citing source paths. Thin wrapper — all retrieval logic still lives
in lib/search.py; this file owns prompt assembly and the Ollama call.

Usage:
  python -m retrieval.ask "What is XDC02030000 about?"
  python -m retrieval.ask "域名劫持的解决方案是什么" --jira
  python -m retrieval.ask "..." --model qwen2.5:14b --top-k 8 --show-context
"""
from __future__ import annotations

import argparse
import sys

import requests

from retrieval.lib import paths
from retrieval.lib.embedder import Embedder
from retrieval.lib.search import DocHit, build_where, search
from retrieval.lib.shards import fetch_record
from retrieval.lib.store import Store

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma4:12b"
CONTEXT_CHARS = 1500  # chars of source text on either side of the cited chunk


def build_system_prompt(answer_lang: str) -> str:
    return (
        "You are a research assistant answering questions using ONLY the excerpts "
        "provided below, drawn from an internal document corpus. Cite the source "
        "path in brackets after any claim you draw from it, e.g. [geedge_jira/attachment/...]. "
        "If the excerpts don't contain enough information to answer, say so plainly "
        f"instead of guessing. Always answer in {answer_lang}, regardless of what "
        "language the excerpts are written in.\n\n"
        "The excerpts may be in a language the reader cannot read (e.g. Chinese). "
        "Because the reader cannot independently verify a translation, prioritize "
        "precise, literal rendering of technical content over natural-sounding prose: "
        "when translating a specific term, field name, or claim, put the original-"
        "language term in parentheses right after your translation the first time it "
        "appears, e.g. 'sinkhole (安全沉洞/黑洞)', so it can be independently checked "
        "against the source or a dictionary/translator. If a passage is ambiguous or "
        "you are not confident in a translation, say so explicitly instead of "
        "smoothing it over."
    )


def _doc_location(manifest: dict, doc_id: str):
    man = manifest.get(doc_id)
    if not man:
        return None
    return paths.TEXT_DIR / man["shard_file"], int(man["byte_offset"])


def gather_context(store: Store, docs: list[DocHit]) -> str:
    """One block per doc per retrieved chunk (all of d.chunks, not just the
    top one — favors thoroughness over a terser prompt), each expanded with
    ±CONTEXT_CHARS of surrounding source text. Falls back to the bare chunk
    text if the source record can't be re-fetched (e.g. a stale manifest).

    load_manifest() does a full-table scan of every doc in the store, so it
    must be called once here and reused — calling it per-doc (as this used
    to) turned an 8-doc context assembly into 8 redundant full scans, ~140s
    of dead time for a corpus this size."""
    manifest = store.load_manifest()
    blocks = []
    for d in docs:
        loc = _doc_location(manifest, d.doc_id)
        full = None
        if loc:
            rec = fetch_record(loc[0], loc[1], d.doc_id)
            full = rec.get("text") or ""
        for c in d.chunks:
            text = c.text
            if full is not None:
                s = max(0, c.char_start - CONTEXT_CHARS)
                e = min(len(full), c.char_end + CONTEXT_CHARS)
                text = full[s:e]
            blocks.append(f"### Source: {d.path}\n{text.strip()}")
    return "\n\n".join(blocks)


def ask_ollama(model: str, system_prompt: str, question: str, context: str, num_ctx: int) -> str:
    # gemma3's chat template mapped a "system" message to its own separate
    # <start_of_turn>user block instead of merging it into the next user
    # turn — two consecutive user turns with no model turn between them was
    # a structure it wasn't trained on, and it visibly ignored the
    # instructions when sent that way. A single combined user turn sidesteps
    # any model's system-role quirks, so kept even after moving to gemma4.
    user_content = (
        f"{system_prompt}\n\n"
        f"Context excerpts:\n\n{context}\n\n"
        f"Question: {question}"
    )
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "stream": False,
            "options": {
                # Ollama's runtime default num_ctx (2048-4096) silently
                # truncates multi-doc CJK context; ask for the real size.
                "num_ctx": num_ctx,
                # Default temperature (1.0) let gemma3 drift into general
                # training knowledge on broad questions instead of the
                # provided excerpts; low temperature keeps answers literal.
                "temperature": 0.1,
            },
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def answer_one(store: Store, embedder, question: str, args) -> tuple[str | None, list[DocHit]]:
    """Run one question through search + Ollama, reusing the caller's store/embedder.
    Returns (answer, docs); answer is None when nothing relevant was retrieved.
    Shared by main() and seed_ask.py so a multi-question run only pays the
    Store/Embedder load cost once."""
    where = build_where(args.path_prefix, args.jira, args.lang, args.raw_filter)
    docs = search(store, embedder, question, mode=args.mode, where=where,
                  top_k=args.top_k, per_doc=args.per_doc)
    if not docs:
        return None, []

    context = gather_context(store, docs)
    if args.show_context:
        print("--- retrieved context ---", file=sys.stderr)
        print(context, file=sys.stderr)
        print("--- end context ---\n", file=sys.stderr)

    print(f"Asking {args.model}...", file=sys.stderr)
    system_prompt = build_system_prompt(args.answer_lang)
    answer = ask_ollama(args.model, system_prompt, question, context, args.num_ctx)
    return answer, docs


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Flags shared between ask.py and seed_ask.py."""
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "fts"])
    ap.add_argument("--top-k", type=int, default=8, help="docs to retrieve as context")
    ap.add_argument("--per-doc", type=int, default=2,
                    help="chunks per doc to include as context (more = thorougher, slower)")
    ap.add_argument("--answer-lang", default="English",
                    help="language the model must answer in, regardless of source-excerpt language")
    ap.add_argument("--path-prefix", action="append", default=[],
                    help="restrict to paths starting with this (repeatable, OR-joined)")
    ap.add_argument("--jira", action="store_true", help="sugar for --path-prefix geedge_jira/")
    ap.add_argument("--lang", help="restrict retrieval to a detected language (e.g. zh, en)")
    ap.add_argument("--filter", dest="raw_filter", help="raw SQL predicate, AND-ed on")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--num-ctx", type=int, default=24576,
                    help="Ollama context window in tokens (raise further for large --top-k/--per-doc)")
    ap.add_argument("--show-context", action="store_true",
                    help="print the retrieved context (to stderr) before the answer")


def main() -> int:
    ap = argparse.ArgumentParser(description="Answer a question from the indexed corpus via a local LLM.")
    ap.add_argument("question", help="your question")
    add_common_args(ap)
    args = ap.parse_args()

    if not paths.VECTORS_DIR.exists():
        print(f"No index at {paths.VECTORS_DIR}. Run `python -m retrieval.index` first.",
              file=sys.stderr)
        return 1
    store = Store.open(paths.VECTORS_DIR)
    embedder = None if args.mode == "fts" else Embedder(device=args.device)

    answer, docs = answer_one(store, embedder, args.question, args)
    if answer is None:
        print("No relevant documents found.")
        return 0

    print(answer)
    print("\nSources:")
    for d in docs:
        print(f"  - {d.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
