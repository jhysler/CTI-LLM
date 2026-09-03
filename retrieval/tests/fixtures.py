"""Synthetic shard generator + a model-free tokenizer stub for tests.

The real corpus lives only on the WSL box, so these handcrafted records — matching
the exact schema scripts/03_extract_text.py writes — are the primary local
verification path. They deliberately cover the awkward cases: literal identifiers,
mixed CJK/EN, an oversized doc, an empty-text failure record, and a 09-style
record missing the `ocr` key.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import orjson

# Known identifiers the tests assert on.
IDENT_HARDWARE = "XDC02030000"
IDENT_DOMAIN = "doescan.xyz"
IDENT_PROJECT = "YYDNS"


class WhitespaceTokenizer:
    """Minimal stand-in for the bge-m3 fast tokenizer: one token per whitespace-
    delimited run, plus one token per CJK character (so Chinese counts like the
    real ~1 token/char). Implements only the call surface chunking.py uses:
    tokenizer(text, add_special_tokens=..., return_offsets_mapping=True) ->
    {"offset_mapping": [(start, end), ...]}."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue
            if "一" <= ch <= "鿿":  # CJK ideograph → its own token
                offsets.append((i, i + 1))
                i += 1
                continue
            j = i
            while j < n and not text[j].isspace() and not ("一" <= text[j] <= "鿿"):
                j += 1
            offsets.append((i, j))
            i = j
        return {"offset_mapping": offsets}


def _rec(doc_id, path, ext, lang, text, *, ocr=False, error="", include_ocr=True):
    rec = {
        "id": doc_id,
        "path": path,
        "ext": ext,
        "lang": lang,
        "char_count": len(text),
        "n_elements": text.count("\n\n") + 1 if text else 0,
        "text": text,
        "error": error,
    }
    if include_ocr:
        rec["ocr"] = ocr
    return rec


def fixture_records() -> List[dict]:
    """~10 handcrafted records spanning the tricky cases."""
    long_text = "\n\n".join(
        f"Paragraph {i} discusses the resolver pipeline and DNS injection "
        f"behaviour in detail with many english words to accrue token count "
        f"across a span wide enough to force multiple chunks number {i}."
        for i in range(120)
    )
    return [
        _rec("doc_hw", "geedge_jira/ISSUE-1/spec.txt", ".txt", "en",
             f"The hardware SKU {IDENT_HARDWARE} ships in the Shield-Cube rack.\n\n"
             f"It is a probe appliance for DNS interception."),
        _rec("doc_domain", "geedge_jira/ISSUE-2/notes.txt", ".txt", "en",
             f"Scanning tool at {IDENT_DOMAIN} feeds the {IDENT_PROJECT} project.\n\n"
             f"Operators query {IDENT_DOMAIN} for coverage stats."),
        _rec("doc_zh_dns", "geedge_docs/dns_report.txt", ".txt", "zh",
             "本报告分析域名劫持与深度包检测的部署方式。\n\n"
             "该系统在骨干网络中实现 DNS 污染与流量识别。"),
        _rec("doc_zh_other", "geedge_docs/cafeteria.txt", ".txt", "zh",
             "本文档介绍公司食堂的菜单与营业时间安排。\n\n"
             "午餐提供多种选择，包括面食与米饭。"),
        _rec("doc_mixed", "geedge_docs/mixed.txt", ".txt", "zh",
             "The 盾立方 (Shield-Cube) platform performs DNS 劫持 detection.\n\n"
             "It integrates resolver telemetry with 流量分析 modules."),
        _rec("doc_long", "geedge_docs/long_resolver.txt", ".txt", "en", long_text),
        _rec("doc_empty", "geedge_jira/ISSUE-9/broken.pdf", ".pdf", "", "",
             error="PdfReadError: could not parse"),
        # 09-style manual record: no `ocr` key at all.
        _rec("doc_manual", "_manual_additions/manual_dns.txt", ".txt", "en",
             f"Manually added file referencing {IDENT_DOMAIN} and DNS4CN.",
             include_ocr=False),
        _rec("doc_unrelated", "geedge_docs/hr_policy.txt", ".txt", "en",
             "This document covers vacation policy and expense reimbursement.\n\n"
             "Nothing here relates to networking."),
    ]


def make_fixture(work_dir: Path) -> Path:
    """Write text/shard_00000.jsonl under work_dir. Returns the shard path."""
    text_dir = work_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    shard = text_dir / "shard_00000.jsonl"
    with shard.open("wb") as f:
        for rec in fixture_records():
            f.write(orjson.dumps(rec) + b"\n")
    return shard


def append_shard(work_dir: Path, records: List[dict]) -> Path:
    """Append a new shard_00001.jsonl (incremental top-up test)."""
    text_dir = work_dir / "text"
    shard = text_dir / "shard_00001.jsonl"
    with shard.open("wb") as f:
        for rec in records:
            f.write(orjson.dumps(rec) + b"\n")
    return shard
