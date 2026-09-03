#!/usr/bin/env python3
"""
Stage 4: Weighted keyword triage for a topic, with noise-reduction filters.

Reads extracted JSONL shards, scans each document's text against a topic
config (YAML in config/), and produces a ranked CSV of candidate documents
with hit counts per term and short context snippets.

Earlier versions of this stage used raw weighted scoring (10/3/1 points per
high/medium/low-signal term occurrence, uncapped). That let bulk data dumps
drown out real analysis — a spreadsheet with "DNS"×13,000 would outscore an
actual document that thoughtfully discussed DNS4CN a dozen times. This
version adds three adjustments that push high-signal analytic documents
above bulk data files:

  1. Per-tier occurrence caps — low-signal terms (e.g. bare "DNS") contribute
     at most --low-cap hits; medium-signal terms at most --mid-cap hits.
     A spreadsheet with DNS×13,000 scores the same as one with DNS×5.

  2. Data-file penalty — CSV / XLS / XLSX files with zero high-signal hits
     are multiplied by --data-penalty (default 0.25).  They stay in the list
     but sink far below analytical documents.

  3. Diversity bonus — +--div-bonus points per unique term hit.  A document
     matching DNS4CN + 盾立方 + DoH + resolver scores higher than one that
     hits only "DNS" five hundred times.

Usage:
  python scripts/04_triage.py --topic dns4cn
  python scripts/04_triage.py --topic dns4cn --low-cap 10 --mid-cap 50 \
      --data-penalty 0.1 --div-bonus 10 --top 200
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import orjson
import pandas as pd
import yaml
from tqdm import tqdm

WORK = Path(os.environ.get("GEEDGE_WORK", Path.home() / "geedge-data"))
TEXT_DIR = WORK / "text"
OUT_DIR = WORK / "triage"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

WEIGHTS = {"high_signal": 10, "medium_signal": 3, "low_signal": 1}
DATA_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def load_topic(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No topic config: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def build_patterns(topic: dict) -> list[tuple[str, str, int, re.Pattern]]:
    """Return list of (tier, term, weight, compiled_pattern)."""
    patterns = []
    for tier, weight in WEIGHTS.items():
        for term in topic.get(tier, []) or []:
            is_cjk = any("一" <= ch <= "鿿" for ch in term)
            if is_cjk:
                pat = re.compile(re.escape(term))
            else:
                pat = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            patterns.append((tier, term, weight, pat))
    return patterns


def find_snippet(text: str, pat: re.Pattern, span: int) -> str:
    m = pat.search(text)
    if not m:
        return ""
    s = max(0, m.start() - span)
    e = min(len(text), m.end() + span)
    snippet = text[s:e].replace("\n", " ").replace("\r", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return ("…" if s > 0 else "") + snippet + ("…" if e < len(text) else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="topic config name (without .yaml)")
    ap.add_argument("--top", type=int, default=300, help="rows to write")
    ap.add_argument("--snippet-chars", type=int, default=160, help="chars on each side of match")
    ap.add_argument("--min-score", type=int, default=1)
    ap.add_argument("--low-cap", type=int, default=5,
                    help="max occurrences counted for low_signal terms (default 5)")
    ap.add_argument("--mid-cap", type=int, default=30,
                    help="max occurrences counted for medium_signal terms (default 30)")
    ap.add_argument("--data-penalty", type=float, default=0.25,
                    help="score multiplier for data files (CSV/XLS/XLSX) with 0 high-signal hits")
    ap.add_argument("--div-bonus", type=int, default=5,
                    help="bonus points per unique term matched (default 5)")
    args = ap.parse_args()

    caps = {
        "high_signal": None,          # uncapped — every occurrence matters
        "medium_signal": args.mid_cap,
        "low_signal": args.low_cap,
    }

    topic = load_topic(args.topic)
    patterns = build_patterns(topic)
    if not patterns:
        print("No terms in topic config", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shards = sorted(TEXT_DIR.glob("shard_*.jsonl"))
    if not shards:
        print(f"No shards in {TEXT_DIR}. Run 03_extract_text.py first.", file=sys.stderr)
        return 1

    results = []
    n_docs = 0

    for shard in tqdm(shards, unit="shard"):
        with shard.open("rb") as f:
            for line in f:
                try:
                    rec = orjson.loads(line)
                except Exception:
                    continue
                n_docs += 1
                text = rec.get("text") or ""
                if not text:
                    continue

                raw_score = 0
                hits: dict[str, int] = {}
                tiers_hit: set[str] = set()
                first_snippets: dict[str, str] = {}
                high_hit_count = 0

                for tier, term, weight, pat in patterns:
                    matches = pat.findall(text)
                    n = len(matches)
                    if not n:
                        continue

                    cap = caps[tier]
                    effective_n = min(n, cap) if cap is not None else n

                    hits[term] = n          # record real count for display
                    tiers_hit.add(tier)
                    raw_score += weight * effective_n
                    if tier == "high_signal":
                        high_hit_count += 1
                    if tier in ("high_signal", "medium_signal") and term not in first_snippets:
                        first_snippets[term] = find_snippet(text, pat, args.snippet_chars)

                if raw_score < args.min_score:
                    continue

                # Tier bonus (same as 04_triage.py)
                tier_bonus = (10 if "high_signal" in tiers_hit else 0) + \
                             (3 if "medium_signal" in tiers_hit else 0)

                # Diversity bonus — reward breadth of term coverage
                diversity_bonus = args.div_bonus * len(hits)

                score = raw_score + tier_bonus + diversity_bonus

                # Data-file penalty: bulk tables with no high-signal hits
                path_str = rec.get("path") or ""
                suffix = Path(path_str).suffix.lower()
                if suffix in DATA_EXTENSIONS and high_hit_count == 0:
                    score = score * args.data_penalty

                results.append({
                    "id": rec.get("id"),
                    "path": path_str,
                    "lang": rec.get("lang"),
                    "char_count": rec.get("char_count"),
                    "score": round(score, 1),
                    "raw_score": raw_score,
                    "n_unique_terms": len(hits),
                    "high_hits": high_hit_count,
                    "hits": ", ".join(f"{k}×{v}" for k, v in
                                      sorted(hits.items(), key=lambda kv: -kv[1])),
                    "snippet": " || ".join(f"[{k}] {v}" for k, v in
                                           list(first_snippets.items())[:3]),
                })

    df = pd.DataFrame(results).sort_values("score", ascending=False)
    out_csv = OUT_DIR / f"{args.topic}.csv"
    df.head(args.top).to_csv(out_csv, index=False)

    print(f"\nScanned {n_docs:,} docs, {len(df):,} matched.")
    print(f"Top {min(args.top, len(df))} written to: {out_csv}\n")
    if not df.empty:
        print("Top 10:")
        print(df.head(10)[["score", "high_hits", "lang", "path", "hits"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
