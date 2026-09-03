#!/usr/bin/env python3
"""
Stage 6: Rasterize PPTX documents to PNG images for vision-aware analysis.

Slide decks carry critical content in architecture diagrams, flowcharts, and
image-embedded text that text-only extraction (Stage 3) misses entirely. For
tier-1 PPTX documents, this stage:

  1. Converts PPTX → PDF via LibreOffice headless
  2. Renders each PDF page to PNG via pdftoppm
  3. Stores images under $GEEDGE_WORK/rasterized/<doc_id>/slide_NN.png

Stage 7 detects these images and includes them as vision blocks alongside
the extracted text. Re-running this stage is a no-op for already-rasterized
docs (cached by presence of the output directory).

Requires: libreoffice, poppler-utils (pdftoppm). Both already installed per
the project README.

Usage:
  python scripts/06_rasterize.py --tier-csv ~/geedge-data/triage/tier1_dns4cn.csv
  python scripts/06_rasterize.py --tier-csv ... --dpi 150 --max-slides 40
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

WORK = Path(os.environ.get("GEEDGE_WORK", Path.home() / "geedge-data"))
TEXT_DIR = WORK / "text"
RASTER_DIR = WORK / "rasterized"

PPTX_EXTS = {".pptx", ".ppt"}
PDF_EXTS = {".pdf"}
LIBREOFFICE_TIMEOUT = 180  # seconds — kill hung LibreOffice processes
PDFTOPPM_TIMEOUT = 120


def build_doc_id_index() -> dict[str, tuple[Path, int]]:
    """{id: (shard_path, byte_offset)} — same approach as Stage 7."""
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


def fetch_doc(doc_id: str, idx: dict) -> dict | None:
    loc = idx.get(doc_id)
    if not loc:
        return None
    shard, offset = loc
    with shard.open("rb") as f:
        f.seek(offset)
        return orjson.loads(f.readline())


def find_source_file(rec: dict) -> Path | None:
    """The extracted path is relative; the absolute path lives in inventory."""
    extracted_root = WORK / "extracted"
    candidate = extracted_root / rec["path"]
    if candidate.exists():
        return candidate
    return None


def rasterize_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    max_slides: int,
) -> tuple[int, str]:
    """Rasterize a PDF directly (no LibreOffice conversion needed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "slide"
    try:
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r", str(dpi),
                "-l", str(max_slides),
                str(pdf_path),
                str(prefix),
            ],
            check=True,
            capture_output=True,
            timeout=PDFTOPPM_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return 0, "pdftoppm timeout"
    except subprocess.CalledProcessError as e:
        return 0, f"pdftoppm failed: {e.stderr.decode('utf-8', errors='replace')[:200]}"

    written = sorted(out_dir.glob("slide-*.png"))
    return len(written), ""


def rasterize_one(
    pptx_path: Path,
    out_dir: Path,
    dpi: int,
    max_slides: int,
) -> tuple[int, str]:
    """Returns (n_slides_written, error_message)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # LibreOffice needs a unique user profile to allow concurrent invocations
    # AND to avoid touching the user's real LO config.
    with tempfile.TemporaryDirectory(prefix="lo_profile_") as profile_dir, \
         tempfile.TemporaryDirectory(prefix="lo_out_") as lo_out:

        try:
            subprocess.run(
                [
                    "libreoffice",
                    f"-env:UserInstallation=file://{profile_dir}",
                    "--headless",
                    "--invisible",
                    "--nologo",
                    "--nofirststartwizard",
                    "--convert-to", "pdf",
                    "--outdir", lo_out,
                    str(pptx_path),
                ],
                check=True,
                capture_output=True,
                timeout=LIBREOFFICE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return 0, "LibreOffice timeout"
        except subprocess.CalledProcessError as e:
            return 0, f"LibreOffice failed: {e.stderr.decode('utf-8', errors='replace')[:200]}"

        pdfs = list(Path(lo_out).glob("*.pdf"))
        if not pdfs:
            return 0, "No PDF produced by LibreOffice"
        pdf_path = pdfs[0]

        # pdftoppm: -png, -r <dpi>, output prefix gets -NN.png appended
        prefix = out_dir / "slide"
        try:
            subprocess.run(
                [
                    "pdftoppm",
                    "-png",
                    "-r", str(dpi),
                    "-l", str(max_slides),  # last page to convert
                    str(pdf_path),
                    str(prefix),
                ],
                check=True,
                capture_output=True,
                timeout=PDFTOPPM_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return 0, "pdftoppm timeout"
        except subprocess.CalledProcessError as e:
            return 0, f"pdftoppm failed: {e.stderr.decode('utf-8', errors='replace')[:200]}"

    written = sorted(out_dir.glob("slide-*.png"))
    return len(written), ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-csv", required=True, help="CSV from Stage 5")
    ap.add_argument("--dpi", type=int, default=120,
                    help="rasterization DPI (default 120 ≈ 1600px wide for 16:9 slides)")
    ap.add_argument("--max-slides", type=int, default=50,
                    help="max slides to render per doc (default 50)")
    ap.add_argument("--force", action="store_true",
                    help="re-rasterize even if output dir already exists")
    ap.add_argument("--include-pdf", action="store_true",
                    help="also rasterize PDF docs (useful for scanned PDFs with no text)")
    args = ap.parse_args()

    if not shutil.which("libreoffice"):
        print("ERROR: libreoffice not found in PATH", file=sys.stderr)
        return 1
    if not shutil.which("pdftoppm"):
        print("ERROR: pdftoppm not found (install poppler-utils)", file=sys.stderr)
        return 1

    df = pd.read_csv(args.tier_csv)
    df["ext"] = df["path"].str.lower().str.extract(r"(\.[^.]+)$")
    target_exts = PPTX_EXTS | (PDF_EXTS if args.include_pdf else set())
    raster_df = df[df["ext"].isin(target_exts)].copy()

    print(f"Tier CSV: {args.tier_csv}")
    print(f"Docs to rasterize: {len(raster_df)} of {len(df)} (PPTX={len(df[df['ext'].isin(PPTX_EXTS)])}, PDF={len(df[df['ext'].isin(PDF_EXTS)])} {'included' if args.include_pdf else 'skipped'})")
    if raster_df.empty:
        print("Nothing to do.")
        return 0

    print("Building shard index...")
    idx = build_doc_id_index()

    RASTER_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"rasterized": 0, "cached": 0, "failed": 0, "missing_source": 0}

    for _, row in tqdm(raster_df.iterrows(), total=len(raster_df), unit="doc"):
        doc_id = row["id"]
        out_dir = RASTER_DIR / doc_id

        if out_dir.exists() and any(out_dir.glob("slide-*.png")) and not args.force:
            stats["cached"] += 1
            tqdm.write(f"  ⊙ cached: {row['path'][:70]}")
            continue

        rec = fetch_doc(doc_id, idx)
        if not rec:
            stats["missing_source"] += 1
            tqdm.write(f"  ✗ no shard record: {row['path']}", file=sys.stderr)
            continue

        src = find_source_file(rec)
        if not src:
            stats["missing_source"] += 1
            tqdm.write(f"  ✗ source missing on disk: {rec['path']}", file=sys.stderr)
            continue

        ext = row["ext"]
        if ext in PDF_EXTS:
            n, err = rasterize_pdf(src, out_dir, args.dpi, args.max_slides)
        else:
            n, err = rasterize_one(src, out_dir, args.dpi, args.max_slides)

        if err:
            stats["failed"] += 1
            tqdm.write(f"  ✗ {row['path'][:60]} — {err}", file=sys.stderr)
        else:
            stats["rasterized"] += 1
            tqdm.write(f"  ✓ {row['path'][:60]} — {n} slides")

    print(f"\nDone. {stats}")
    print(f"Output: {RASTER_DIR}/<doc_id>/slide-NN.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
