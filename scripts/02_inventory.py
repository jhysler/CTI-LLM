#!/usr/bin/env python3
"""
Stage 2: Build a file inventory of the extracted corpus.

Walks $GEEDGE_WORK/extracted, records path/size/mtime/mime for every file,
and writes a parquet manifest. Fast (metadata only — does not read content).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import magic
import pandas as pd
from tqdm import tqdm

WORK = Path(os.environ.get("GEEDGE_WORK", Path.home() / "geedge-data"))
SRC = WORK / "extracted"
OUT = WORK / "inventory.parquet"

# Extension hints — used as a fallback when libmagic can't classify
EXT_GROUPS = {
    "doc":   {".docx", ".doc", ".pdf", ".pptx", ".ppt", ".xlsx", ".xls",
              ".odt", ".rtf", ".txt", ".md", ".html", ".htm", ".eml", ".msg"},
    "data":  {".json", ".jsonl", ".xml", ".csv", ".tsv", ".yaml", ".yml"},
    "code":  {".py", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".js", ".ts",
              ".java", ".sh", ".sql", ".tf"},
    "image": {".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp", ".webp", ".tiff"},
    "binary":{".tar", ".zip", ".gz", ".zst", ".7z", ".rpm", ".deb", ".so", ".o",
              ".pyc", ".jar", ".war", ".class", ".rar"},
}
EXT_GROUP = {ext: g for g, exts in EXT_GROUPS.items() for ext in exts}

# Hard skips: don't even inventory these
SKIP_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv"}


def file_id(path: Path, root: Path) -> str:
    """Stable short ID derived from relative path."""
    rel = str(path.relative_to(root))
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found. Run 01_extract_archives.sh first.", file=sys.stderr)
        return 1

    detector = magic.Magic(mime=True)
    rows = []

    print(f"Walking {SRC}...")
    for dirpath, dirnames, filenames in tqdm(os.walk(SRC), unit="dir"):
        # Prune skipped dirs in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]

        for fname in filenames:
            p = Path(dirpath) / fname
            try:
                st = p.lstat()
            except OSError:
                continue

            # Symlinks (we created some in stage 1) — resolve target size
            if p.is_symlink():
                try:
                    st = p.stat()
                except OSError:
                    continue

            ext = p.suffix.lower()
            group = EXT_GROUP.get(ext, "other")

            # libmagic on every file would be slow on 100GB+
            # Run it only when extension is ambiguous
            mime = ""
            if group in ("other", "data") and st.st_size < 50_000_000:
                try:
                    mime = detector.from_file(str(p))
                except Exception:
                    mime = ""

            rows.append({
                "id": file_id(p, SRC),
                "path": str(p.relative_to(SRC)),
                "abs_path": str(p),
                "ext": ext,
                "group": group,
                "mime": mime,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
            })

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)

    print(f"\nInventoried {len(df):,} files ({df['size'].sum() / 1e9:.1f} GB)")
    print("\nBy group:")
    print(df.groupby("group").agg(count=("id", "size"), gb=("size", lambda s: s.sum() / 1e9)))
    print(f"\nWrote: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
