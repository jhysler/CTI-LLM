# Repurposing the pipeline for a new topic

The corpus build (Stages 1–3) is topic-agnostic and runs once. Everything
downstream is parameterized by a single YAML file in `config/`. To research a
new topic, you usually only write a YAML and re-run Stages 4 → 5 → 6 → 7.

## What's topic-agnostic vs. per-topic

| Stage | Script | Topic-aware? | What's hardcoded |
| ----- | ------ | ------------ | ---------------- |
| 1 | `01_extract_archives.sh` | No | Archive list (corpus-wide) |
| 2 | `02_inventory.py` | No | Extension groups, dir skips |
| 3 | `03_extract_text.py` | No | Extractable extensions, 100MB cap |
| 4 | `04_triage.py` | **Yes** | Tier weights (10/3/1), caps, penalties, bonus; terms from YAML |
| 5 | `05_tier.py` | Partially | YAML name only; Confluence-space heuristics are hardcoded |
| 6 | `06_rasterize.py` | No | Just needs a tier CSV; not topic-specific itself |
| 7 | `07_analyze.py` | **Yes** | Topic description injected from YAML; framing prose hardcoded |

Verified by grep: stages 1–3 contain no references to DNS, signal tiers,
triage, or any topic term. The only filtering they apply is by extension /
MIME type / size — they would behave identically for any research target.

## Workflow for a new topic

### 1. Write `config/<topic>.yaml`

Three tiers control how strongly a match contributes to a document's score
(high ×10, medium ×3, low ×1). Choosing the right tier per term is the most
important decision and the main thing you'll iterate on.

```yaml
topic: <short-name>
description: |
  One paragraph describing the research target. This gets injected verbatim
  into the Stage 7 LLM prompt, so write it as if briefing an analyst.

high_signal:
  # Almost certainly relevant if matched. Project names, codewords,
  # internal acronyms. Aim for terms with near-zero false-positive rate.
  - "ProjectCodename"
  - "内部代号"

medium_signal:
  # Strong topical relevance but might appear in unrelated docs.
  # Technical terms specific to the topic area.
  - "specific-protocol-name"
  - "技术术语"

low_signal:
  # Generic terms — noisy alone but useful when co-occurring with above.
  - "broader-category-term"
  - "general acronym"
```

Tier-placement guide:

- A term that returns ≤ 50 corpus-wide hits and is uniquely yours → `high_signal`
- A term that's clearly on-topic but has > 50 hits → `medium_signal`
- A term that would match a textbook chapter on the general field → `low_signal`

Bare `DNS` in the DNS4CN config is the canonical low-signal example: relevant,
but on its own it matches every networking document in the corpus.

### 2. Run Stage 4

```bash
.venv/bin/python scripts/04_triage.py --topic <name>
```

Look at the top 20 in `~/geedge-data/triage/<name>.csv`. If you see:

- **Bulk data files dominating** (CSV, XLSX with huge raw counts): lower
  `--low-cap` (default 5) or `--data-penalty` (default 0.25).
- **All high scores are unrelated to your topic**: your high-signal terms
  aren't specific enough — move some to medium, add more distinctive ones.
- **Nothing matches**: your terms are too narrow or use the wrong language
  / transliteration. Check the corpus with `grep -ri "term" ~/geedge-data/text/`
  on one shard to see how the term actually appears.

Iterate the YAML and re-run. Each pass is ~1 minute.

### 3. Adapt Stage 5 (if needed)

`05_tier.py` uses path heuristics to separate **internal Geedge/MESA docs**
from **external papers Geedge happened to be reading**. The heuristic is
hardcoded:

```python
INTERNAL_SPACES = re.compile(r"(study/|MeshTrust/|YYDNS/(?!attachments))", re.IGNORECASE)
```

These are the Confluence spaces that hold the DNS4CN-adjacent working docs.
For a different topic, the relevant spaces will be different. Quick way to
find them:

```bash
# What Confluence spaces show up in your high-scoring docs?
head -50 ~/geedge-data/triage/<name>.csv | \
  awk -F, '{print $2}' | \
  grep -oE '(mesalab_docs|geedge_docs|geedge_jira)/[^/]+/' | \
  sort | uniq -c | sort -rn
```

If the top spaces are different from `study/MeshTrust/YYDNS`, edit
`INTERNAL_SPACES` in `05_tier.py` (or fork it to `05b_tier.py`). The
`EXTERNAL_SIGNALS` regex (catches `attachments/<id>_attachments_<EnglishTitle>`)
is fairly generic and usually doesn't need changes.

For some topics the tier-1/tier-2 split is meaningless because all candidates
are internal — in that case, just feed `<name>.csv` directly to Stage 7 and
skip 5.

### 4. (Optional) Rasterize slide decks, then run Stage 7

If tier1 contains PPTX/PPT (or scanned PDFs you want vision on), rasterize
first so Stage 7 picks up the images automatically:

```bash
.venv/bin/python scripts/06_rasterize.py \
    --tier-csv ~/geedge-data/triage/tier1_<name>.csv
```

```bash
export ANTHROPIC_API_KEY=sk-...
.venv/bin/python scripts/07_analyze.py \
    --tier-csv ~/geedge-data/triage/tier1_<name>.csv \
    --topic <name> \
    --out ~/geedge-data/analysis/<name>_findings.md \
    --max-docs 5 --dry-run     # always do a dry run first
```

The `--topic` flag pulls `description:` from your YAML and injects it into
the per-doc and synthesis system prompts. The surrounding framing prose
("leaked internal documents from Geedge Networks and the MESA Lab") is
hardcoded in `07_analyze.py` — if your topic is from a different corpus,
edit `PER_DOC_SYSTEM` and `SYNTHESIS_SYSTEM` accordingly.

Cost scales roughly linearly with doc count and char count. Budget ~$0.50–1.50
per doc with the default Sonnet 4.6 / Opus 4.7 split.

## When you'd need to touch Stages 1–3

Rare, but possible:

- **You need source-code archives** (e.g. researching the actual DPI binaries
  rather than policy docs). Re-add archives to `01_extract_archives.sh` and
  remove `.py / .c / .cpp / etc.` from the `code` group skip in Stage 2's
  inventory, then add them to `EXTRACTABLE_EXTS` in Stage 3.
- **You need image OCR** (Stage 3 currently skips images). Add Tesseract OCR
  as a fallback extractor in `03_extract_text.py` — non-trivial but additive.
- **You need files > 100 MB** (some DB dumps and video files were excluded).
  Raise `MAX_FILE_BYTES` in Stage 3.

Any of these invalidates the existing `text/shard_*.jsonl` outputs and
requires a full Stage 3 re-run (~8–24 hr). For most research questions, the
existing extraction is sufficient and you only touch Stages 4–8.

## Topic ideas in this corpus

Things visible in the existing triage results and directory structure that
look topic-shaped:

| Topic | Starter high-signal terms |
| ----- | ------------------------- |
| **Tiangou / TSGEN** | `Tiangou`, `天狗`, `TSGEN`, `境外攻击` |
| **MeshTrust** | `MeshTrust`, `网状信任` |
| **Pakistan deployment** | `Pakistan`, `PTA`, `巴基斯坦` |
| **DPI / traffic classification** | `DPI`, `deep packet inspection`, `流量识别`, `流量分类` |
| **Circumvention blocking** | `VPN`, `Shadowsocks`, `Trojan`, `翻墙`, `代理` |
| **Specific researchers** | Names from `mesalab_docs/student/` filenames |

For each, the workflow is the same: write a YAML, run Stage 4, inspect, tune,
then 5 → 8.
