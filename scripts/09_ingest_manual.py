#!/usr/bin/env python3
"""
Stage 9: Ingest manually-added files into the pipeline.

Files placed in extracted/_manual_additions/ are already verified as relevant
and bypass Stages 2–5 (inventory, text extraction, triage, tier). This script:

  1. Computes stable IDs using the same SHA1-of-relative-path algorithm as Stage 2
  2. Extracts text via unstructured (same as Stage 3)
  3. Appends a new JSONL shard so Stage 6 and Stage 7 can find the records
  4. Writes triage/manual_<topic>.csv so Stage 6 and Stage 7 accept these docs

Zone.Identifier files (Windows alternate data streams) are skipped automatically.

Usage:
  python scripts/09_ingest_manual.py --topic dns4cn
  python scripts/09_ingest_manual.py --topic dns4cn --force   # re-extract already-done
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

WORK = Path(os.environ.get("GEEDGE_WORK", Path.home() / "geedge-data"))
MANUAL_DIR = WORK / "extracted" / "_manual_additions"
TEXT_DIR = WORK / "text"
TRIAGE_DIR = WORK / "triage"

EXTRACTABLE_EXTS = {
    ".docx", ".doc", ".pdf", ".pptx", ".ppt", ".xlsx", ".xls",
    ".odt", ".rtf", ".txt", ".md", ".html", ".htm", ".eml", ".msg",
    ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml",
}

MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000


def file_id(path: Path, root: Path) -> str:
    """Same algorithm as Stage 2."""
    rel = str(path.relative_to(root))
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]


def detect_lang(text: str) -> str:
    if not text or len(text) < 40:
        return ""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text[:4000])
    except Exception:
        return ""


def extract_text(path: Path) -> tuple[str, int, str]:
    """Returns (text, n_elements, error)."""
    try:
        from unstructured.partition.auto import partition
        elements = partition(filename=str(path), strategy="fast")
        parts = [str(el) for el in elements if str(el).strip()]
        text = "\n\n".join(parts)[:MAX_TEXT_CHARS]
        return text, len(parts), ""
    except Exception as e:
        return "", 0, f"{type(e).__name__}: {e}"


def already_done_ids() -> set[str]:
    done = set()
    for shard in TEXT_DIR.glob("shard_*.jsonl"):
        with shard.open("rb") as f:
            for line in f:
                try:
                    done.add(orjson.loads(line)["id"])
                except Exception:
                    pass
    return done


def next_shard_path() -> Path:
    existing = sorted(TEXT_DIR.glob("shard_*.jsonl"))
    idx = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0
    return TEXT_DIR / f"shard_{idx:05d}.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True, help="topic name (e.g. dns4cn)")
    ap.add_argument("--force", action="store_true", help="re-extract already-done files")
    args = ap.parse_args()

    if not MANUAL_DIR.exists():
        print(f"ERROR: {MANUAL_DIR} does not exist.", file=sys.stderr)
        return 1

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    TRIAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Enumerate files — skip Zone.Identifier and unsupported extensions
    extracted_root = WORK / "extracted"
    candidates = []
    for p in sorted(MANUAL_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.suffix == ".Identifier" or ":Zone.Identifier" in p.name or p.name.endswith(":Zone.Identifier"):
            continue
        # Windows ADS show up as "filename:Zone.Identifier" on some WSL versions
        if ":" in p.name:
            continue
        if p.suffix.lower() not in EXTRACTABLE_EXTS:
            print(f"  skip (unsupported ext): {p.name}")
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            print(f"  skip (>{MAX_FILE_BYTES // 1024 // 1024}MB): {p.name}")
            continue
        candidates.append(p)

    print(f"Manual additions dir: {MANUAL_DIR}")
    print(f"Candidates to ingest: {len(candidates)}")

    done_ids = set() if args.force else already_done_ids()
    print(f"Already in shards: {len(done_ids)} docs (use --force to re-extract)")

    to_process = [p for p in candidates if file_id(p, extracted_root) not in done_ids]
    already_present = [p for p in candidates if file_id(p, extracted_root) in done_ids]

    for p in already_present:
        print(f"  ⊙ already extracted: {p.name}")

    if not to_process:
        print("Nothing new to extract.")
    else:
        print(f"\nExtracting {len(to_process)} files...")
        new_records = []
        for p in tqdm(to_process, unit="file"):
            doc_id = file_id(p, extracted_root)
            rel_path = str(p.relative_to(extracted_root))
            text, n_elements, error = extract_text(p)
            lang = detect_lang(text) if text else ""
            rec = {
                "id": doc_id,
                "path": rel_path,
                "ext": p.suffix.lower(),
                "lang": lang,
                "char_count": len(text),
                "n_elements": n_elements,
                "text": text,
                "error": error,
            }
            new_records.append(rec)
            status = f"✓ {len(text):,} chars" if text else f"✗ {error[:60]}"
            tqdm.write(f"  {status}: {p.name}")

        shard_path = next_shard_path()
        with shard_path.open("wb") as f:
            for rec in new_records:
                f.write(orjson.dumps(rec) + b"\n")
        print(f"\nWrote {len(new_records)} records → {shard_path}")

    # Build tier CSV covering all candidates (including already-extracted ones)
    print(f"\nBuilding tier CSV for {len(candidates)} docs...")
    tier_rows = []
    for p in candidates:
        doc_id = file_id(p, extracted_root)
        tier_rows.append({
            "id": doc_id,
            "path": str(p.relative_to(extracted_root)),
            "lang": "",
            "char_count": 0,
            "score": 9999,
            "raw_score": 9999,
            "n_unique_terms": 0,
            "high_hits": 1,
            "hits": "manual",
            "snippet": "",
            "tier": "tier1",
        })

    out_csv = TRIAGE_DIR / f"manual_{args.topic}.csv"
    pd.DataFrame(tier_rows).to_csv(out_csv, index=False)
    print(f"Wrote tier CSV → {out_csv}")
    print(f"\nNext steps:")
    print(f"  python scripts/06_rasterize.py --tier-csv {out_csv}")
    print(f"  python scripts/07_analyze.py --tier-csv {out_csv} --topic {args.topic} \\")
    print(f"      --out ~/geedge-data/analysis/{args.topic}_findings_manual.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
