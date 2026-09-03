#!/usr/bin/env python3
"""
Stage 3: Extract text from inventoried documents.

- Reads inventory.parquet, filters to extractable types
- Uses unstructured.partition.auto on each file
- Falls back to OCR (pdftoppm/LibreOffice + Tesseract) for PDF/PPT/PPTX
  that come back under --ocr-min-chars — catches scanned decks and
  screenshot-pasted slides, which "fast" strategy returns near-empty for
  (found via a deck literally titled after its own topic that had only
  108 chars extracted and a title-slide's worth of real content, with 11
  more pages of screenshotted slides "fast" never touched)
- Writes one JSONL shard per 1000 files for resumability
- Multiprocessing pool sized to --workers
- Detects language with langdetect

Output rows:
  {id, path, ext, lang, char_count, n_elements, text, error, ocr}

Use --reextract-low-text to also re-run OCR fallback over PDF/PPT/PPTX
already sitting in shards from an earlier (pre-OCR-fallback) run of this
script — it rewrites their shard lines in place rather than appending
duplicates. Re-run Stage 4 afterwards to re-score with the new text.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

WORK = Path(os.environ.get("GEEDGE_WORK", Path.home() / "geedge-data"))
INVENTORY = WORK / "inventory.parquet"
OUT_DIR = WORK / "text"
SHARD_SIZE = 1000

# Only attempt extraction on these
EXTRACTABLE_EXTS = {
    ".docx", ".doc", ".pdf", ".pptx", ".ppt", ".xlsx", ".xls",
    ".odt", ".rtf", ".txt", ".md", ".html", ".htm", ".eml", ".msg",
    ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml",
}

# Extensions eligible for OCR fallback (scanned/screenshot-based content)
OCR_EXTS = {".pdf", ".ppt", ".pptx"}
PPTX_EXTS = {".ppt", ".pptx"}
OCR_MIN_CHARS = 300           # below this, treat "fast" extraction as a miss
OCR_LANGS = "chi_sim+eng"
OCR_MAX_DIM = 2000            # cap longest side in px — a fixed DPI blows up on
                               # decks with oversized declared page sizes (a
                               # screenshot-based deck seen in the wild had a
                               # 4032x2268pt page, i.e. 1:1 pixel-as-point; at a
                               # fixed 220 DPI that renders to a 12320x6930px,
                               # ~85MP PNG taking 60-70s/page and blowing every
                               # timeout below)
OCR_MAX_PAGES = 60
LIBREOFFICE_TIMEOUT = 180
PDFTOPPM_TIMEOUT = 300        # a normal-size 60-page deck alone measured ~122s
TESSERACT_TIMEOUT = 60

# Skip files larger than this (likely media or DB dumps)
MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_TEXT_CHARS = 2_000_000          # truncate insanely long extractions

# Worker count (override with --workers). Each worker independently imports
# the full unstructured/torch/onnxruntime stack (~500MB-1GB+ RSS before
# processing any file), and legacy .doc/.ppt files spawn a LibreOffice
# subprocess per conversion on top of that. Ran out of memory on WSL at
# --workers 12 — if that happens again, lower this before raising it back up.
DEFAULT_WORKERS = max(1, os.cpu_count() // 2)

# sosreport archives attached to JIRA tickets unpack (Stage 1) into tens of
# thousands of /proc, /etc, /var system-state files per ticket. A handful of
# those carry extractable extensions (sos.html summary, json/txt configs) but
# none of it is prose — exclude by path rather than extension.
EXCLUDE_PATH_SUBSTR = "sosreport"


def detect_lang(text: str) -> str:
    if not text or len(text) < 40:
        return ""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text[:4000])
    except Exception:
        return ""


def _pdf_to_pngs(pdf_path: Path, out_prefix: Path) -> list[Path]:
    subprocess.run(
        ["pdftoppm", "-png", "-scale-to", str(OCR_MAX_DIM), "-l", str(OCR_MAX_PAGES),
         str(pdf_path), str(out_prefix)],
        check=True, capture_output=True, timeout=PDFTOPPM_TIMEOUT,
    )
    return sorted(out_prefix.parent.glob(out_prefix.name + "-*.png"))


def _pptx_to_pngs(src: Path, out_prefix: Path) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="lo_profile_") as profile_dir, \
         tempfile.TemporaryDirectory(prefix="lo_out_") as lo_out:
        subprocess.run(
            ["libreoffice", f"-env:UserInstallation=file://{profile_dir}",
             "--headless", "--invisible", "--nologo", "--nofirststartwizard",
             "--convert-to", "pdf", "--outdir", lo_out, str(src)],
            check=True, capture_output=True, timeout=LIBREOFFICE_TIMEOUT,
        )
        pdfs = list(Path(lo_out).glob("*.pdf"))
        if not pdfs:
            return []
        return _pdf_to_pngs(pdfs[0], out_prefix)


def ocr_fallback(abs_path: str, ext: str) -> tuple[str, str]:
    """Rasterize + Tesseract OCR. Returns (text, error)."""
    src = Path(abs_path)
    with tempfile.TemporaryDirectory(prefix="ocr_raster_") as tmp:
        prefix = Path(tmp) / "page"
        try:
            pngs = _pptx_to_pngs(src, prefix) if ext in PPTX_EXTS else _pdf_to_pngs(src, prefix)
        except subprocess.TimeoutExpired:
            return "", "rasterize timeout"
        except subprocess.CalledProcessError as e:
            return "", f"rasterize failed: {e.stderr.decode('utf-8', errors='replace')[:200]}"

        if not pngs:
            return "", "no pages rendered"

        parts = []
        for png in pngs:
            try:
                out = subprocess.run(
                    ["tesseract", str(png), "-", "-l", OCR_LANGS],
                    check=True, capture_output=True, timeout=TESSERACT_TIMEOUT,
                )
                text = out.stdout.decode("utf-8", errors="replace").strip()
                if text:
                    parts.append(text)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        return "\n\n".join(parts), ""


def extract_one(row: dict) -> dict:
    """Worker: extract text from a single file, with OCR fallback for
    scanned/screenshot PDF/PPT/PPTX that "fast" extraction returns near-empty."""
    out = {
        "id": row["id"],
        "path": row["path"],
        "ext": row["ext"],
        "lang": "",
        "char_count": 0,
        "n_elements": 0,
        "text": "",
        "error": "",
        "ocr": False,
    }
    try:
        from unstructured.partition.auto import partition
        elements = partition(
            filename=row["abs_path"],
            strategy="fast",  # OCR handled explicitly below, only when needed
        )
        parts = [str(el) for el in elements if str(el).strip()]
        text = "\n\n".join(parts)[:MAX_TEXT_CHARS]
        out["text"] = text
        out["n_elements"] = len(parts)
        out["char_count"] = len(text)
        out["lang"] = detect_lang(text)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    if out["ext"] in OCR_EXTS and out["char_count"] < OCR_MIN_CHARS:
        ocr_text, ocr_err = ocr_fallback(row["abs_path"], out["ext"])
        if ocr_text and len(ocr_text) > out["char_count"]:
            out["text"] = ocr_text[:MAX_TEXT_CHARS]
            out["char_count"] = len(out["text"])
            out["n_elements"] = ocr_text.count("\n\n") + 1
            out["lang"] = detect_lang(out["text"])
            out["ocr"] = True
        elif ocr_err and not out["error"]:
            out["error"] = f"ocr_fallback: {ocr_err}"

    return out


def already_done_ids(out_dir: Path) -> set[str]:
    done = set()
    for shard in out_dir.glob("shard_*.jsonl"):
        with shard.open("rb") as f:
            for line in f:
                try:
                    rec = orjson.loads(line)
                    done.add(rec["id"])
                except Exception:
                    pass
    return done


def strip_stale_records(out_dir: Path, stale_ids: set[str]) -> None:
    """Remove shard lines for stale_ids in place, so re-extraction (with OCR
    fallback now enabled) can append fresh records without duplicating ids."""
    if not stale_ids:
        return
    for shard in out_dir.glob("shard_*.jsonl"):
        lines = shard.read_bytes().splitlines()
        kept = []
        removed = 0
        for line in lines:
            try:
                rec = orjson.loads(line)
            except Exception:
                kept.append(line)
                continue
            if rec.get("id") in stale_ids:
                removed += 1
            else:
                kept.append(line)
        if removed:
            with shard.open("wb") as f:
                for l in kept:
                    f.write(l + b"\n")


def find_low_text_ocr_candidates(out_dir: Path) -> set[str]:
    """ids of PDF/PPT/PPTX records already extracted below OCR_MIN_CHARS
    and not yet OCR'd — candidates for --reextract-low-text."""
    ids = set()
    for shard in out_dir.glob("shard_*.jsonl"):
        with shard.open("rb") as f:
            for line in f:
                try:
                    rec = orjson.loads(line)
                except Exception:
                    continue
                if (
                    rec.get("ext") in OCR_EXTS
                    and not rec.get("ocr")
                    and (rec.get("char_count") or 0) < OCR_MIN_CHARS
                ):
                    ids.add(rec["id"])
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=0, help="cap files for testing")
    parser.add_argument("--reextract-low-text", action="store_true",
                        help="also re-run OCR fallback over PDF/PPT/PPTX already in "
                             "shards from before OCR fallback existed (rewrites their "
                             "shard lines in place, no duplicate ids)")
    args = parser.parse_args()

    if not INVENTORY.exists():
        print(f"ERROR: {INVENTORY} missing. Run 02_inventory.py first.", file=sys.stderr)
        return 1

    missing_tools = [t for t in ("pdftoppm", "tesseract", "libreoffice") if not shutil.which(t)]
    if missing_tools:
        print(f"WARNING: {', '.join(missing_tools)} not found — OCR fallback for "
              f"low-text PDF/PPT/PPTX will be skipped (errors recorded per-doc).",
              file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INVENTORY)
    df = df[df["ext"].isin(EXTRACTABLE_EXTS)]
    df = df[df["size"] <= MAX_FILE_BYTES]
    df = df[df["size"] > 0]
    df = df[~df["path"].str.contains(EXCLUDE_PATH_SUBSTR, case=False)]

    done = already_done_ids(OUT_DIR)
    print(f"Already extracted: {len(done):,}")

    if args.reextract_low_text:
        stale = find_low_text_ocr_candidates(OUT_DIR)
        print(f"Low-text OCR candidates to re-extract: {len(stale):,}")
        strip_stale_records(OUT_DIR, stale)
        done -= stale

    df = df[~df["id"].isin(done)]
    if args.limit:
        df = df.head(args.limit)

    print(f"To extract: {len(df):,} files ({df['size'].sum() / 1e9:.1f} GB)")
    if df.empty:
        return 0

    # Determine next shard number
    existing = sorted(OUT_DIR.glob("shard_*.jsonl"))
    next_idx = (int(existing[-1].stem.split("_")[1]) + 1) if existing else 0

    rows = df.to_dict("records")
    buf: list[dict] = []
    shard_idx = next_idx

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, r): r["id"] for r in rows}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="file"):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"id": futures[fut], "error": f"WORKER_CRASH: {e}\n{traceback.format_exc()}"}
            buf.append(rec)

            if len(buf) >= SHARD_SIZE:
                shard_path = OUT_DIR / f"shard_{shard_idx:05d}.jsonl"
                with shard_path.open("wb") as f:
                    for r in buf:
                        f.write(orjson.dumps(r) + b"\n")
                shard_idx += 1
                buf.clear()

    if buf:
        shard_path = OUT_DIR / f"shard_{shard_idx:05d}.jsonl"
        with shard_path.open("wb") as f:
            for r in buf:
                f.write(orjson.dumps(r) + b"\n")

    print(f"\nDone. Shards in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
