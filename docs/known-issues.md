# Known issues

## seed_ask.py: are "not found" verdicts real misses, or undiagnosed retrieval bugs?

During the initial build-out of `retrieval/seed_ask.py`'s topic battery, two topics
(`xinjiang_912_firewall`, `shield_cube`) came back as false negatives across several
full runs — not because the content was absent from the corpus, but because of two
systematic retrieval bugs:

1. **Query dilution**: blending multiple terms (or a question + terms) into one
   embedding query doesn't behave like "the union of these signals" — bge-m3
   produces one averaged vector, and blending in even a single extra term is enough
   to dilute out a doc that keys almost entirely on one proper noun. Confirmed
   directly: `"912"` alone surfaced the doc literally named after this project at
   rank #1; `"912 新疆"` already lost it. Fixed by searching one term at a time and
   merging results (see `run_topic()`).
2. **Cross-leg score-scale mismatch**: after (1), merging each per-term search leg
   by raw score still failed for Shield-Cube — a broad natural-language question
   search produces structurally higher raw hybrid scores than a short 2-6 character
   proper-noun term search, so the question leg's generic hits crowded out the
   correct term-leg hits by score comparison alone (the single best doc ranked #45
   by raw score, #3 after the fix). Fixed with `rrf_rescore()` — rank-based fusion
   across search legs instead of raw-score comparison, mirroring how BM25/dense are
   already RRF-fused within a single `search()` call.

Both are fixed as of the commit that added this doc. The open question: **how many
of the other `FOUND: no` / bare `NOT FOUND` results in a given run are similarly
undiagnosed retrieval misses, not genuine absences?** No controlled test (same code,
same prompt, N reruns) has isolated pure generation-sampling noise
(`--temperature 0.1` isn't fully deterministic) from retrieval-driven variance —
every run so far also changed a retrieval fix between it and the last, so verdict
flips observed between runs (e.g. `ether_shield` yes→no) can't cleanly be attributed
to one cause or the other.

**Recommendation for picking this up**: don't default to multi-run majority voting
as a fix for a suspicious `no` — re-running the same retrieval pipeline N times just
resamples wording, it can't surface a doc that never made it into context (that's
exactly what happened here; the fix was diagnostic, not statistical). When a `no`
result looks wrong given known domain facts about the corpus, probe individual-term
retrieval rank directly (search each topic term alone via `retrieval.lib.search`,
check whether an expected doc appears near the top) rather than rerunning
generation. Reserve multi-run consistency checks (`--topic <id>`, cheap since it's
one topic not the full battery) for topics with `CONFIDENCE: low` or hedgy `GAPS`
text specifically. A worthwhile future improvement: have the tool report, per topic,
whether any individual term's search independently ranked a doc highly, as a
built-in guardrail flagging "possible retrieval miss" before a `no` verdict is
trusted.
