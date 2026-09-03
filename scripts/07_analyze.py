#!/usr/bin/env python3
"""
Stage 7: Analyze top-tier triage hits with the Claude API.

Reads a tier-1 CSV (subset of triage output), pulls each document's extracted
text from the JSONL shards, and sends batches to Claude with prompt caching.
For each document produces:
  - structured English summary
  - extracted technical claims about DNS4CN / Shield-Cube
  - inline citations back to the source path
  - a translated abstract if the source is Chinese

If Stage 6 has rasterized a document's slides into $GEEDGE_WORK/rasterized/<id>/,
those PNG images are attached as vision blocks alongside the extracted text.

Then a final synthesis pass across all per-doc findings.

Usage:
  export ANTHROPIC_API_KEY=sk-...
  python scripts/07_analyze.py \
      --tier-csv ~/geedge-data/triage/tier1_dns4cn.csv \
      --topic dns4cn \
      --out ~/geedge-data/analysis/dns4cn_findings.md

Cost target: ~$5-15 for ~12 docs (text-only). Add ~$0.30 per 40-slide deck
when images are attached.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import orjson
import pandas as pd
import yaml
from anthropic import Anthropic
from tqdm import tqdm

WORK = Path(os.environ.get("GEEDGE_WORK", Path.home() / "geedge-data"))
TEXT_DIR = WORK / "text"
RASTER_DIR = WORK / "rasterized"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Truncate per-doc payload so a single huge JIRA export doesn't blow the budget.
# Most docs are well under this. Smoking-gun PPTX/PDF will be 5-50k chars.
MAX_DOC_CHARS = 80_000

# Cap on slide images per doc — API allows more but past 30 you usually pay
# for redundancy rather than added signal.
MAX_IMAGES_PER_DOC = 30

PER_DOC_SYSTEM = """You are analyzing leaked internal documents from Geedge Networks
and the MESA Lab (Chinese Academy of Sciences) — a Chinese network surveillance
and censorship infrastructure vendor. The operator is investigating a specific
topic: {topic_description}

For each document you receive, produce a structured analysis in this exact format:

## Document: {{filename}}
**Language:** {{detected language}}
**Document type:** {{slide deck | research paper | meeting notes | technical doc | other}}
**Relevance to topic:** {{high | medium | low}} — {{one sentence justifying this}}

When slide images are included alongside the extracted text, prioritize them
for architecture diagrams, flowcharts, system topology, and any content that
appears to be visual rather than textual. The extracted text from slide decks
is incomplete by design — the diagrams are the substance.

### Key technical claims
- Bullet list of specific, citable claims relevant to the topic.
- Each claim must be defensible from the source text.
- Prefer specifics (names, dates, mechanisms, numbers) over generalities.
- If the document is Chinese, give the original Chinese phrase in parentheses for any
  loaded or technical term whose English rendering loses nuance (e.g. "stability
  maintenance (维稳)", "deep packet inspection deployment (纵深部署)").

### People, orgs, projects named
- Bullet list. Include both English and Chinese forms when present.

### Open questions / things worth checking
- What this document hints at but doesn't fully explain.
- What follow-up search terms might surface related docs.

### Translation quality flag
- If you suspect the extracted text was sanitized, machine-translated, or
  garbled, say so. If the source is Chinese and you have any concern about
  word-choice fidelity, flag the specific phrases.

Be precise. Do not editorialize. Do not add disclaimers. The operator is doing
documented research on publicly leaked materials."""

SYNTHESIS_SYSTEM = """You have just analyzed {n_docs} documents from the Geedge /
MESA Lab leak related to: {topic_description}

Synthesize a coherent picture across all documents. Structure:

# {topic_title} — Synthesis

## Executive summary
3-5 sentences. What is this system, who built it, what stage is it at, what's
the most important finding.

## Architecture and mechanism
What the system actually does, technically. Cite source documents by filename.

## Timeline and provenance
When was this work done, by whom, in what organizational context. Cite.

## Key personnel and organizations
With Chinese names where available.

## Dependencies and open questions
What other systems this connects to. What's still unclear from the corpus.

## Document inventory
A table: filename | doc type | relevance | one-line description.

Be precise. Cite source filenames inline. Do not editorialize."""


def load_topic(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def build_doc_id_index() -> dict[str, tuple[Path, int]]:
    """Build {id: (shard_path, byte_offset)} so we don't rescan shards per doc."""
    idx = {}
    for shard in sorted(TEXT_DIR.glob("shard_*.jsonl")):
        with shard.open("rb") as f:
            offset = 0
            for line in f:
                try:
                    rec = orjson.loads(line)
                    idx[rec["id"]] = (shard, offset)
                except Exception:
                    pass
                offset += len(line)
    return idx


def fetch_doc(doc_id: str, idx: dict) -> Optional[dict]:
    loc = idx.get(doc_id)
    if not loc:
        return None
    shard, offset = loc
    with shard.open("rb") as f:
        f.seek(offset)
        line = f.readline()
        return orjson.loads(line)


def load_slide_images(doc_id: str, max_images: int = MAX_IMAGES_PER_DOC) -> list[dict]:
    """Return image content blocks for any rasterized slides for this doc."""
    img_dir = RASTER_DIR / doc_id
    if not img_dir.exists():
        return []

    blocks = []
    slides = sorted(img_dir.glob("slide-*.png"))[:max_images]
    for slide_path in slides:
        with slide_path.open("rb") as f:
            data = base64.standard_b64encode(f.read()).decode("ascii")
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": data,
            },
        })
    return blocks


def analyze_one(
    client: Anthropic,
    rec: dict,
    triage_row: dict,
    system_prompt: str,
    model: str,
    max_images: int = MAX_IMAGES_PER_DOC,
    max_doc_chars: int = MAX_DOC_CHARS,
) -> tuple[str, dict]:
    """Send one doc to Claude. Returns (analysis_markdown, usage_dict)."""
    text = (rec.get("text") or "")[:max_doc_chars]
    truncated = len(rec.get("text") or "") > max_doc_chars

    image_blocks = load_slide_images(rec["id"], max_images=max_images)
    has_images = len(image_blocks) > 0

    text_block = {
        "type": "text",
        "text": (
            f"Path: {rec['path']}\n"
            f"Detected language: {rec.get('lang') or 'unknown'}\n"
            f"Triage score: {triage_row['score']} "
            f"(high_signal_hits: {triage_row['high_hits']})\n"
            f"Matched terms: {triage_row['hits']}\n"
            f"{'[TRUNCATED to first %d chars]' % max_doc_chars if truncated else ''}\n"
            f"{'[%d slide images attached below]' % len(image_blocks) if has_images else ''}\n\n"
            f"--- DOCUMENT TEXT ---\n{text}"
        ),
    }

    # Order matters for the model: text context first, then images in slide order
    user_content = [text_block] + image_blocks

    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},  # cache the long system
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    analysis = "".join(b.text for b in resp.content if b.type == "text")
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
        "n_images": len(image_blocks),
    }
    return analysis, usage


def synthesize(
    client: Anthropic,
    per_doc_outputs: list[str],
    topic: dict,
    model: str,
) -> tuple[str, dict]:
    system = SYNTHESIS_SYSTEM.format(
        n_docs=len(per_doc_outputs),
        topic_description=topic["description"].strip(),
        topic_title=topic["topic"].upper(),
    )
    joined = "\n\n---\n\n".join(per_doc_outputs)

    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": joined}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    return text, usage


def estimate_cost(usage: dict, model: str) -> float:
    """Rough cost in USD. Update rates as needed."""
    rates = {
        # per 1M tokens
        "sonnet": {"in": 3.00, "out": 15.00, "cache_write": 3.75, "cache_read": 0.30},
        "opus":   {"in": 5.00, "out": 25.00, "cache_write": 6.25, "cache_read": 0.50},
    }
    family = "opus" if "opus" in model else "sonnet"
    r = rates[family]
    c = (
        usage.get("input_tokens", 0) * r["in"]
        + usage.get("output_tokens", 0) * r["out"]
        + usage.get("cache_creation_input_tokens", 0) * r["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * r["cache_read"]
    ) / 1_000_000
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-csv", required=True, help="CSV of docs to analyze")
    ap.add_argument("--topic", required=True, help="topic config name")
    ap.add_argument("--out", required=True, help="output markdown path")
    ap.add_argument("--per-doc-model", default="claude-opus-4-7")
    ap.add_argument("--synthesis-model", default="claude-opus-4-7")
    ap.add_argument("--max-docs", type=int, default=0, help="cap for testing")
    ap.add_argument("--dry-run", action="store_true", help="print plan and exit")
    ap.add_argument("--max-images", type=int, default=MAX_IMAGES_PER_DOC,
                    help=f"max slide images to send per doc (default {MAX_IMAGES_PER_DOC}; "
                         f"raise to handle decks longer than {MAX_IMAGES_PER_DOC} slides)")
    ap.add_argument("--max-doc-chars", type=int, default=MAX_DOC_CHARS,
                    help=f"max extracted-text characters per doc (default {MAX_DOC_CHARS})")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        env_file = Path.home() / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY=") and "=" in line:
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                    break

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in environment or ~/.env", file=sys.stderr)
        return 1

    topic = load_topic(args.topic)
    df = pd.read_csv(args.tier_csv)
    if args.max_docs:
        df = df.head(args.max_docs)

    print(f"Topic: {topic['topic']}")
    print(f"Docs to analyze: {len(df)}")
    print(f"Per-doc model: {args.per_doc_model}")
    print(f"Synthesis model: {args.synthesis_model}")

    print("\nBuilding shard index...")
    idx = build_doc_id_index()
    print(f"Indexed {len(idx):,} extracted docs")

    if args.dry_run:
        print("\nDRY RUN — would analyze:")
        for _, row in df.iterrows():
            present = "OK" if row["id"] in idx else "MISSING"
            img_dir = RASTER_DIR / row["id"]
            n_imgs = len(list(img_dir.glob("slide-*.png"))) if img_dir.exists() else 0
            img_note = f" +{n_imgs} imgs" if n_imgs else ""
            print(f"  [{present}] score={row['score']:>4} {row['path']}{img_note}")
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = Anthropic()
    per_doc_outputs = []
    total_cost = 0.0

    system_prompt = PER_DOC_SYSTEM.format(
        topic_description=topic["description"].strip()
    )

    print("\nAnalyzing documents...")
    for _, row in tqdm(df.iterrows(), total=len(df), unit="doc"):
        rec = fetch_doc(row["id"], idx)
        if not rec:
            print(f"  MISSING in shards: {row['path']}", file=sys.stderr)
            continue
        if not (rec.get("text") or "").strip():
            print(f"  EMPTY extracted text: {row['path']} (needs OCR pass)", file=sys.stderr)
            continue

        try:
            analysis, usage = analyze_one(
                client, rec, row.to_dict(), system_prompt, args.per_doc_model,
                max_images=args.max_images,
                max_doc_chars=args.max_doc_chars,
            )
        except Exception as e:
            print(f"  API error on {row['path']}: {e}", file=sys.stderr)
            continue

        cost = estimate_cost(usage, args.per_doc_model)
        total_cost += cost
        per_doc_outputs.append(analysis)
        img_note = f" +{usage['n_images']} imgs" if usage.get('n_images') else ""
        tqdm.write(f"  ✓ {row['path'][:60]} — ${cost:.3f}{img_note}")

    if not per_doc_outputs:
        print("No successful analyses. Aborting synthesis.", file=sys.stderr)
        return 1

    # Save per-doc outputs as we go (already accumulated; write before synthesis)
    interim = out_path.with_suffix(".per_doc.md")
    interim.write_text("\n\n---\n\n".join(per_doc_outputs))
    print(f"\nPer-doc analyses saved: {interim}")

    print("\nRunning synthesis pass (Opus)...")
    try:
        synth, synth_usage = synthesize(
            client, per_doc_outputs, topic, args.synthesis_model
        )
        synth_cost = estimate_cost(synth_usage, args.synthesis_model)
        total_cost += synth_cost
        print(f"  Synthesis cost: ${synth_cost:.3f}")
    except Exception as e:
        print(f"Synthesis failed: {e}", file=sys.stderr)
        synth = "(synthesis failed — see per-doc outputs)"

    out_path.write_text(
        f"# {topic['topic'].upper()} — Analysis\n\n"
        f"Total cost: ${total_cost:.2f}\n"
        f"Documents analyzed: {len(per_doc_outputs)}\n\n"
        f"---\n\n"
        f"{synth}\n\n"
        f"---\n\n"
        f"# Per-document analyses\n\n"
        f"{chr(10).join(['---' + chr(10) + d for d in per_doc_outputs])}"
    )
    print(f"\n✓ Written: {out_path}")
    print(f"✓ Total cost: ${total_cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
