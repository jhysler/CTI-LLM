# Portability: moving this project to another machine

Written for the case "I want to work on this from the MacBook Pro instead of the
WSL2 / RTX 4080 box" — but the reasoning generalizes to any move, including a
Windows reinstall on the current machine.

For the original WSL-side setup, see [`retrieval-porting.md`](retrieval-porting.md).
That doc is about getting `retrieval/` running *on the 4080 box*. This one is
about getting the project off it.

## The one idea

**Sort every artifact by what it costs to recreate, not by its size.** The
expensive things here are small and the huge things are cheap. Once you see
that, the move plans write themselves.

| Tier | Artifact | Size | Cost to recreate | Move it? |
| --- | --- | --- | --- | --- |
| **Irreplaceable** | `geedge-rag/` repo | ~1MB | n/a — it's the work | Yes (it's in git) |
| **Irreplaceable** | Source archives (`*.tar.zst`) | ~52GB | Re-download the torrent | Only if you must |
| **Expensive** | `text/shard_*.jsonl` | **617MB** | **8–24 hr** of Stage 3 | **Yes — always** |
| **Expensive** | `triage/*.csv` | 880KB | Minutes (needs shards) | Yes, trivially |
| Moderate | `vectors/` (LanceDB) | 14GB | ~1 hr on the 4080; much worse elsewhere | Judgment call — see below |
| Moderate | `rasterized/` | 180MB | Minutes (needs `extracted/`) | Only with `extracted/` |
| Cheap | `inventory.parquet` | 28MB | 20–40 min (needs `extracted/`) | No |
| Regenerable | `extracted/` | **65GB** | 1–2 hr from archives | **No** |
| Junk | `text.bak-pre-ocr/` | 608MB | n/a — stale backup | No |

The headline: `extracted/` is 65GB of the 129GB working directory and is pure
derived data. `text/` — the thing that actually took a day of compute — is
617MB. **The entire expensive output of this pipeline fits on a thumb drive.**

## Move plans

### A. "I just want to query the corpus from the Mac" — ~15GB

```text
geedge-rag/          git clone (or push dev first, then clone)
geedge-data/
├── text/            617MB  ← the one thing you cannot regenerate cheaply
├── vectors/         14GB   ← copy rather than rebuild (see "Should I copy vectors/")
└── triage/          880KB  ← cheap, and lets you cross-check retrieval hits
```

This gets you the full `retrieval/` surface: `query.py`, `ask.py`, `seed_ask.py`.
It does **not** get you Stages 1–9, which need `extracted/`.

### B. "Everything except the raw corpus" — ~15GB, plus re-extract on demand

Same as A. If you later need Stages 6–7 (rasterize/analyze), you'd need
`extracted/` back — that means the source archives and a 1–2 hr Stage 1 re-run.
Fine to defer; it's a known, bounded cost.

### C. "Archival snapshot before a Windows reinstall" — ~53GB

The archives (52GB) + `text/` (617MB) + `triage/` + the repo. Skip `vectors/`
(rebuildable) and `extracted/` (rebuildable). This is the set from which
*everything else in the project can be reconstructed*, and nothing in it can be
regenerated from anything else you'd still have.

> **Also back up, separately:** the DNS4CN analysis outputs. `analysis/` and
> `translations/` do **not** currently exist under `~/geedge-data` on this box —
> the canonical findings live at <https://www.hysler.net/DNS4CN/> and in the
> `rag/` reports (`~/geedge-data/rag/*.md`, ~500KB). Those `rag/` reports are
> `seed_ask.py` output and are *not* in git. Grab them.

## The single knob: `GEEDGE_WORK`

Every path in both workflows derives from one environment variable
([`retrieval/lib/paths.py`](../retrieval/lib/paths.py), and the same idiom in
`scripts/04_triage.py`):

```python
WORK = Path(os.environ.get("GEEDGE_WORK", Path.home() / "geedge-data"))
```

So on the Mac:

```bash
export GEEDGE_WORK=~/geedge-data          # or an external SSD: /Volumes/T7/geedge-data
export GEEDGE_RAW=~/projects/geedge-docs  # only needed for Stages 1–3
```

There is no other path configuration to change. This was deliberate, and it's
the reason a move is a copy rather than a port.

## Should I copy `vectors/` or rebuild it?

**Copy it.** 14GB over USB-C beats any rebuild you can do on a Mac today.

The index was designed to be copyable: `store.py` records `shard_file` as a
*filename only*, not an absolute path, so the LanceDB directory has no
machine-specific state in it. Drop it at `$GEEDGE_WORK/vectors/` on the new box
and it works, provided `text/` came along too (the `--full-context` and `ask.py`
paths re-read the shards by byte offset).

Rebuilding instead means paying for embedding 290,073 chunks:

- **RTX 4080, fp16 CUDA:** under an hour. Verified.
- **Apple Silicon, MPS:** *not currently supported* — see the gap below.
- **Apple Silicon, CPU:** many hours. Don't.

If you do rebuild, don't guess at the runtime — measure it:

```bash
python -m retrieval.index --rebuild --limit 2000    # then extrapolate ×145
```

And note that a rebuild re-downloads bge-m3 (~2.3GB) to `~/.cache/huggingface`
unless you copy that cache over too.

## Known gap: no MPS support

[`retrieval/lib/embedder.py`](../retrieval/lib/embedder.py)'s `resolve_device()`
resolves `auto` → `cuda` → `cpu`. There is no `mps` branch, so on a MacBook Pro
`--device auto` silently lands on **CPU**, which is why the rebuild numbers above
are so bad. This was an explicit scoping decision, not an oversight — see
"Platform scope" in [`retrieval-plan.md`](retrieval-plan.md), which committed to
CUDA as the sole performance target and deliberately kept device selection
confined to this one function so a port would stay cheap.

Closing it is a ~5-line change to that one function:

```python
def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():   # add
        return "mps"                        # add
    return "cpu"
```

Two caveats if you do it:

- **Leave fp16 off on MPS.** `Embedder.__init__` already does the right thing —
  `fp16 = (self.device == "cuda")` — so an added `mps` branch inherits fp32
  automatically. Don't "fix" that; half-precision on MPS has a history of
  silently producing NaNs in transformer stacks.
- **Verify FlagEmbedding actually honors it.** `BGEM3FlagModel(devices=...)` is
  FlagEmbedding's own abstraction, not raw torch. Confirm with a 50-chunk
  `--limit` run and check Activity Monitor's GPU history before trusting a full
  build to it.

Since the index is copyable, this is genuinely optional. Only worth doing if you
plan to *re-index* on the Mac — new topics, re-extractions, corpus additions.

## Mac-specific setup notes

### System dependencies

The WSL `apt` list in the root README maps to Homebrew like this:

```bash
brew install ripgrep zstd p7zip unar poppler tesseract libmagic
brew install --cask libreoffice
```

Three things to know:

- **`libmagic`**: `python-magic==0.4.27` needs the dylib on the library path.
  If `import magic` fails, `export DYLD_LIBRARY_PATH=$(brew --prefix)/lib`.
- **LibreOffice**: Stage 3 (legacy `.doc`/`.ppt`) and Stage 6 (rasterize) shell
  out to `soffice`. The cask puts it at
  `/Applications/LibreOffice.app/Contents/MacOS/soffice` — not on `PATH` by
  default. Symlink it or the conversions fail silently-ish.
- **`pdfminer.six==20221105`** is pinned deliberately (newer releases changed
  layout behavior `unstructured` depends on). Don't let a resolver bump it.

### Python and torch

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # Stages 1–4
pip install -r retrieval/requirements.txt    # retrieval/ — plain torch is fine here
```

Unlike the CUDA box, there's no index-URL dance: the default PyPI `torch` wheel
on macOS is the one you want. `lancedb` and `tantivy` both ship arm64 wheels.

### Stage 3 memory

Already measured on a 24GB Mac and documented in the root README's known issues:
`--workers 7` held ~5GB baseline with a ~7–8GB peak once concurrent LibreOffice
conversions were running. Size `--workers` to RAM, not core count — this is what
OOM-killed the WSL box at `--workers 12`.

### Ollama for `ask.py` / `seed_ask.py`

Both hit `http://localhost:11434` and default to `gemma4:12b`. Ollama runs
natively on Apple Silicon and uses unified memory, so a 12B model needs roughly
9–10GB free. On a 16GB machine, expect to close things or drop to a smaller
model via `--model`; on 32GB+ it's comfortable.

Nothing about the Ollama path is CUDA-specific — it's the *embedding* step, not
the *answering* step, that wants a GPU. Even with the MPS gap unclosed, `ask.py`
and `seed_ask.py` are fully usable on a Mac against a copied index: a query
embeds exactly one string, so the CPU fallback costs ~a second, not hours.

## Verifying the move

Run these on the new box in order; each isolates one layer.

```bash
# 1. Paths resolve, shards are visible (no model load)
python -m retrieval.index --dry-run
#    → new/changed/unchanged/empty should sum to ~54,761
#    → all-"new" means it can't see vectors/; "no text dir" means GEEDGE_WORK is wrong

# 2. Index survived the copy — FTS leg only, still no model load
python -m retrieval.query "XDC02030000" --mode fts

# 3. Dense leg — first run downloads bge-m3 (~2.3GB) unless you copied the cache
python -m retrieval.query "域名劫持" --lang zh

# 4. Shards and index still agree (this is the one that catches a partial copy)
python -m retrieval.query "盾立方" --full-context
#    → an "index stale" error here means text/ and vectors/ are out of sync;
#      re-run `python -m retrieval.index` to reconcile

# 5. Answer layer, needs Ollama running
python -m retrieval.ask "What is DNS4CN?"

# 6. Repo-only sanity, no corpus needed
pytest retrieval/tests/ -m "not slow"
```

Step 4 is the one worth not skipping. Steps 1–3 can all pass on a `vectors/`
copy whose `text/` didn't finish transferring; the byte-offset re-read in step 4
is what actually proves both halves arrived intact.

## If you're reinstalling Windows on this box

The corpus lives on the **WSL2 ext4 filesystem**, inside the distro's virtual
disk — a Windows reinstall destroys it unless you export first.

```powershell
wsl --export <distro-name> D:\geedge-distro-backup.tar
```

That captures everything including the 129GB working directory, so check free
space first. The lighter alternative is to copy plan C's file set out to a
Windows drive or external disk and rebuild the distro from the root README's
setup steps afterward — you'd re-run Stage 1 (1–2 hr) but skip Stage 3 (8–24 hr),
which is the whole point of treating `text/` as the thing you protect.

Either way, confirm `git push` is current first — the repo is the only part of
this that has an off-box copy by default.
