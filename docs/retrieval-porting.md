# Porting `retrieval/` to the Windows / RTX 4080 box

The retrieval subsystem is CUDA-targeted, and the corpus + `scripts/` pipeline
already live in **WSL2** on the RTX 4080 machine (`~/geedge-data`, `GEEDGE_WORK`).
Run it there — you inherit the existing shards, the same paths, and a clean CUDA
torch install. Native Windows Python works too, but you'd be re-pathing
`GEEDGE_WORK` and fighting torch/CUDA wheels for no benefit; prefer WSL2.

For day-to-day usage after setup, see [`retrieval/README.md`](../retrieval/README.md).

## 1. Get the code onto the box

Inside WSL2, in your existing repo clone:

```bash
cd ~/Projects/geedge-rag        # wherever the repo lives on WSL
git fetch && git checkout dev && git pull
```

## 2. Install the heavy deps into your venv

```bash
source .venv/bin/activate       # the same venv scripts/ already uses

# CUDA torch FIRST — match your driver (check `nvidia-smi`, top-right CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or cu124

# then the rest
pip install -r retrieval/requirements.txt
```

Verify CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True  NVIDIA GeForce RTX 4080`. If it prints `False`, the CLI still runs
(CPU fallback) but slowly — fix the torch/driver match before building.

## 3. Dry-run first (no GPU, sanity-check counts)

```bash
python -m retrieval.index --dry-run
```

The new/changed/unchanged/empty counts should sum to roughly the shard doc count
(~54k). "No text dir" means `GEEDGE_WORK` isn't set the same way as for
`scripts/` — export it identically.

## 4. Build the index

```bash
python -m retrieval.index          # first bge-m3 use downloads ~2.3GB to ~/.cache/huggingface
```

Well under an hour on the 4080. Writes to `$GEEDGE_WORK/vectors/` (disposable —
rebuild anytime with `--rebuild`).

## 5. Validate with three known-answer queries

```bash
python -m retrieval.query "XDC02030000" --mode fts        # literal id, no model load
python -m retrieval.query "域名劫持" --lang zh              # Chinese semantic
python -m retrieval.query "DNS interception" --jira        # JIRA-scoped
```

Cross-check hits against the existing `triage/dns4cn.csv`. This is the primary
verification path that can't be run off the box (needs the real model + data).

## 6. Optional: full test suite on the box

```bash
pytest retrieval/tests/ -m slow    # CPU end-to-end on synthetic fixtures
```

## Two things to confirm against real data (both easy to change)

- **`--jira` prefix**: `JIRA_PATH_PREFIX = "geedge_jira/"` in
  [`retrieval/lib/search.py`](../retrieval/lib/search.py). Run
  `head -c 300 $GEEDGE_WORK/text/shard_00000.jsonl` and check the real `path`
  prefix; if it differs, edit that one constant.
- **LanceDB hybrid API**: worked on lancedb 0.34. If your installed version's
  hybrid signature differs, `search.py` auto-falls-back to the two-leg RRF path
  and prints a one-line stderr notice — functional either way.

Re-running `python -m retrieval.index` after any `scripts/03 --reextract-low-text`
or `09_ingest_manual.py` run is incremental (sha1-diffed) — no `--rebuild` needed.
