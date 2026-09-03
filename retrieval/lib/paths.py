"""Filesystem paths, derived from GEEDGE_WORK (same idiom as scripts/04_triage.py).

Everything the retrieval subsystem reads or writes hangs off GEEDGE_WORK so the
whole tree stays relocatable and the vectors/ dir is disposable/rebuildable.
"""
from __future__ import annotations

import os
from pathlib import Path

WORK = Path(os.environ.get("GEEDGE_WORK", Path.home() / "geedge-data"))
TEXT_DIR = WORK / "text"        # input: shard_*.jsonl from scripts/03
VECTORS_DIR = WORK / "vectors"  # output: LanceDB database (disposable)
