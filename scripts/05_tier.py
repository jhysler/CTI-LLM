#!/usr/bin/env python3
"""
Stage 5: Filter triage output into analysis tiers.

Reads the raw triage CSV from Stage 4 and separates documents into:
  - tier1: internal Geedge/MESA docs — send directly to Claude API (Stage 7)
  - tier2: external research papers in internal spaces — context only, skip for now

The scoring from Stage 4 doesn't distinguish "Geedge built this" from
"Geedge was reading this." Both score high on DNS/DoH/DoT terms. This
script uses path heuristics and filename character sets to make that cut
before any API spend.

Tier 1 criteria (OR):
  - high_hits > 0: explicit DNS4CN / Shield-Cube / national-DNS terminology
  - CJK characters in filename AND path is an internal Confluence space
    (study/, MeshTrust/, YYDNS/ internal pages — not external attachments)

Usage:
  python scripts/05_tier.py --topic dns4cn
  python scripts/05_tier.py --topic dns4cn --print-tier2
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

WORK = Path(os.environ.get("GEEDGE_WORK", Path.home() / "geedge-data"))

# Confluence spaces that are internal Geedge/MESA working areas
# (not the /attachments/ subfolder that holds downloaded external PDFs)
INTERNAL_SPACES = re.compile(
    r"(study/|MeshTrust/|YYDNS/(?!attachments))",
    re.IGNORECASE,
)

HAS_CJK = re.compile(r"[\u4e00-\u9fff]")

# Paths that are clearly external research papers even if in an internal space
EXTERNAL_SIGNALS = re.compile(
    r"attachments/\d+_attachments_[A-Z]",  # attachment ID + English-titled PDF
)


def classify(row: pd.Series) -> str:
    path = str(row["path"])
    high_hits = int(row.get("high_hits", 0) or 0)

    # Explicit high-signal term match — always tier 1
    if high_hits > 0:
        return "tier1"

    # Internal space + CJK filename = likely internal doc
    if INTERNAL_SPACES.search(path) and HAS_CJK.search(path):
        # But not if it looks like an externally-sourced attachment
        if not EXTERNAL_SIGNALS.search(path):
            return "tier1"
        return "tier2"

    return "tier2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="topic name (matches triage CSV basename)")
    ap.add_argument("--print-tier2", action="store_true", help="also print tier-2 summary")
    args = ap.parse_args()

    triage_csv = WORK / "triage" / f"{args.topic}.csv"
    if not triage_csv.exists():
        print(f"ERROR: {triage_csv} not found. Run 04_triage.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(triage_csv)
    df["tier"] = df.apply(classify, axis=1)

    tier1 = df[df["tier"] == "tier1"].sort_values("score", ascending=False)
    tier2 = df[df["tier"] == "tier2"].sort_values("score", ascending=False)

    out_dir = WORK / "triage"
    tier1_path = out_dir / f"tier1_{args.topic}.csv"
    tier2_path = out_dir / f"tier2_{args.topic}.csv"

    tier1.to_csv(tier1_path, index=False)
    tier2.to_csv(tier2_path, index=False)

    print(f"\nTier 1 ({len(tier1)} docs) — internal, send to Claude API:")
    print(tier1[["score", "high_hits", "lang", "path"]].to_string(index=False))

    if args.print_tier2:
        print(f"\nTier 2 ({len(tier2)} docs) — external research, skip for now:")
        print(tier2[["score", "high_hits", "lang", "path"]].head(20).to_string(index=False))

    print(f"\n✓ tier1 → {tier1_path}")
    print(f"✓ tier2 → {tier2_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
