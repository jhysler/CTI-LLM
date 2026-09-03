#!/usr/bin/env python3
"""
Audit *_en.md translation files for slide truncation vs source page counts.

Two independent checks per document:

  FAIL_SHORT        images_attached (or max ## Slide N) < true source page count
  FAIL_MERGED_TAIL  last ## Slide block is >{MERGED_TAIL_FACTOR}x the median block
                    length — symptom of tail content concatenated into last block
  UNKNOWN           source file not found on disk; page count unverifiable
  SKIP              no images attached (text-only doc; slide count irrelevant)
  PASS              everything looks right

Usage:
  pip install pypdf python-pptx
  python3 audit_truncation.py \\
      --src ~/geedge-data/extracted \\
      --md  ~/geedge-data/translations/dns4cn

  # show all rows, not just failures:
  python3 audit_truncation.py --src ... --md ... --all

Exit code: 1 if any FAIL rows; 0 otherwise.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

SOURCE_PATH_RE = re.compile(r"\*\*Source path:\*\*\s*`([^`]+)`")
IMAGES_RE = re.compile(r"\*\*Images attached:\*\*\s*(\d+)")
SLIDE_HEADER_RE = re.compile(r"^##\s+Slide\s+(\d+)", re.MULTILINE)
SLIDE_SPLIT_RE = re.compile(r"(?m)^(?=##\s+Slide\s+\d+)")

MERGED_TAIL_FACTOR = 3.0  # last block must be this many × median to flag


def true_page_count(path: Path) -> tuple[int, str]:
    """Return (n_pages, error_message). n_pages=0 on error."""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            return len(PdfReader(str(path)).pages), ""
        elif ext in {".pptx", ".ppt"}:
            from pptx import Presentation
            return len(Presentation(str(path)).slides), ""
        else:
            return 0, f"unsupported ext {ext!r}"
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {str(exc)[:60]}"


def audit_one(md_path: Path, src_root: Path) -> dict:
    text = md_path.read_text(encoding="utf-8", errors="replace")

    m_src = SOURCE_PATH_RE.search(text)
    m_img = IMAGES_RE.search(text)

    src_rel = m_src.group(1) if m_src else None
    images_attached = int(m_img.group(1)) if m_img else 0
    slide_nums = [int(n) for n in SLIDE_HEADER_RE.findall(text)]
    max_slide = max(slide_nums) if slide_nums else 0

    # doc_id is the first underscore-delimited token of the filename
    doc_id = md_path.name.split("_")[0]

    row: dict = {
        "file": md_path.name,
        "doc_id": doc_id,
        "src": src_rel or "?",
        "images_attached": images_attached,
        "max_slide": max_slide,
        "true_pages": "?",
        "status": "UNKNOWN",
        "note": "",
    }

    # Text-only docs have nothing to check
    if images_attached == 0 and max_slide == 0:
        row["status"] = "SKIP"
        row["note"] = "no images, no slide headers"
        return row

    if not src_rel:
        row["note"] = "no Source path header"
        return row

    src_path = src_root / src_rel
    if not src_path.exists():
        row["note"] = f"not found: {src_path.name}"
        return row

    true_count, err = true_page_count(src_path)
    if err:
        row["note"] = f"count error: {err}"
        return row

    row["true_pages"] = true_count

    # Check 1: short — the highest slide number we documented vs true count
    # Use max of images_attached and max_slide (both should match; take the
    # higher to avoid double-flagging when Claude chose not to emit a slide header)
    documented = max(images_attached, max_slide)
    short = documented < true_count

    # Check 2: merged tail — last slide block suspiciously longer than median
    merged_tail = False
    tail_note = ""
    if len(slide_nums) >= 3:
        parts = SLIDE_SPLIT_RE.split(text)
        slide_blocks = [p for p in parts if SLIDE_HEADER_RE.match(p)]
        if len(slide_blocks) >= 3:
            word_counts = [len(b.split()) for b in slide_blocks]
            # Exclude the last block when computing median so it can't inflate itself
            median_wc = statistics.median(word_counts[:-1])
            last_wc = word_counts[-1]
            if median_wc > 0 and last_wc > MERGED_TAIL_FACTOR * median_wc:
                merged_tail = True
                tail_note = (
                    f"last={last_wc}w  median(others)={median_wc:.0f}w  "
                    f"ratio={last_wc / median_wc:.1f}x"
                )

    flags: list[str] = []
    notes: list[str] = []

    if short:
        flags.append("FAIL_SHORT")
        notes.append(f"documented={documented}  true={true_count}")
    if merged_tail:
        flags.append("FAIL_MERGED_TAIL")
        notes.append(tail_note)

    row["status"] = "+".join(flags) if flags else "PASS"
    row["note"] = "  |  ".join(notes)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--src", required=True,
        help="root of extracted source files (e.g. ~/geedge-data/extracted)",
    )
    ap.add_argument(
        "--md", required=True,
        help="directory containing *_en.md translation files",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="show PASS and SKIP rows in addition to failures",
    )
    args = ap.parse_args()

    src_root = Path(args.src).expanduser()
    md_root = Path(args.md).expanduser()

    if not src_root.is_dir():
        print(f"ERROR: --src {src_root} is not a directory", file=sys.stderr)
        return 1
    if not md_root.is_dir():
        print(f"ERROR: --md {md_root} is not a directory", file=sys.stderr)
        return 1

    md_files = sorted(md_root.glob("*.en.md"))
    if not md_files:
        print(f"No *.en.md files found in {md_root}", file=sys.stderr)
        return 1

    rows = [audit_one(p, src_root) for p in md_files]

    fail_rows = [r for r in rows if r["status"].startswith("FAIL")]

    # ── Table ────────────────────────────────────────────────────────────────
    col_status = 30
    print(f"{'STATUS':<{col_status}} {'IMG':>4} {'SLD':>4} {'TRUE':>5}  FILE")
    print("─" * (col_status + 4 + 4 + 5 + 2 + 60))

    for r in rows:
        if not args.all and not r["status"].startswith("FAIL"):
            continue
        note_str = f"  # {r['note']}" if r["note"] else ""
        print(
            f"{r['status']:<{col_status}} "
            f"{r['images_attached']:>4} "
            f"{r['max_slide']:>4} "
            f"{str(r['true_pages']):>5}  "
            f"{r['file']}"
            f"{note_str}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    counts = {
        "FAIL": len(fail_rows),
        "PASS": sum(1 for r in rows if r["status"] == "PASS"),
        "SKIP": sum(1 for r in rows if r["status"] == "SKIP"),
        "UNKNOWN": sum(1 for r in rows if r["status"] == "UNKNOWN"),
    }
    print(
        f"\nTotal: {len(rows)}  "
        + "  ".join(f"{k}: {v}" for k, v in counts.items())
    )

    if fail_rows:
        print("\nFiles to delete and re-translate:")
        for r in fail_rows:
            print(f"  rm {md_root / r['file']}")
        print(
            f"\nThen re-run 08_translate.py with --max-images <N> where N >= "
            f"{max(r['true_pages'] for r in fail_rows if isinstance(r['true_pages'], int))}; "
            f"--skip-existing (default) will process only the deleted files."
        )

    return 1 if fail_rows else 0


if __name__ == "__main__":
    sys.exit(main())
