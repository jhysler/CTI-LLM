"""Tokenizer-based document chunking.

Char-length heuristics are wrong for this corpus: Chinese is ~1 token/char while
English is ~4 chars/token, so a fixed char window would produce wildly uneven
token counts (and blow past the model's max_length on CJK text). We chunk on the
bge-m3 tokenizer instead — the same fast XLM-R tokenizer FlagEmbedding downloads.
Its `return_offsets_mapping=True` gives exact char back-references for free, so
every chunk carries a valid `text[char_start:char_end] == chunk.text` slice
(hard splits land on real multibyte boundaries).

The chunker takes an injected `tokenizer` so the unit tests run with no model
download (a whitespace stub is enough); production passes the real bge-m3
tokenizer via get_tokenizer().

Paragraph boundary is "\n\n" — Stage 3 (03_extract_text.py) joins its extracted
elements with exactly that separator, so paragraphs line up with document
structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

_DEFAULT_TOKENIZER = None


@dataclass
class Chunk:
    chunk_index: int
    char_start: int
    char_end: int
    text: str


def get_tokenizer():
    """Lazily load the bge-m3 fast tokenizer (cached process-wide)."""
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        from transformers import AutoTokenizer

        _DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return _DEFAULT_TOKENIZER


def _token_offsets(tokenizer, text: str) -> List[Tuple[int, int]]:
    """Char (start, end) per token, relative to `text`. Empty for empty text."""
    if not text:
        return []
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(o) for o in enc["offset_mapping"]]


def _split_paragraphs(text: str) -> List[Tuple[int, int]]:
    """(start, end) of each non-blank paragraph; text[start:end] == paragraph.

    Splits on the literal "\n\n" separator Stage 3 uses. Blank paragraphs are
    dropped (they'd add empty chunks); their char span is still spanned by any
    surrounding chunk's slice, so the round-trip invariant is unaffected.
    """
    spans: List[Tuple[int, int]] = []
    pos = 0
    for part in text.split("\n\n"):
        start = pos
        end = pos + len(part)
        if part.strip():
            spans.append((start, end))
        pos = end + 2  # skip the 2-char "\n\n" separator
    return spans


def _overlap_tail(
    paras: List[Tuple[int, int, int]], overlap_tokens: int
) -> List[Tuple[int, int, int]]:
    """Trailing paragraphs of a just-emitted chunk to carry into the next one.

    Takes paragraphs from the end until their token sum reaches overlap_tokens.
    Never carries the entire chunk (drops the first) so the next chunk is
    guaranteed to advance — otherwise a chunk fully containing its predecessor
    could repeat forever.
    """
    if overlap_tokens <= 0:
        return []
    tail: List[Tuple[int, int, int]] = []
    acc = 0
    for p in reversed(paras):
        if acc >= overlap_tokens:
            break
        tail.insert(0, p)
        acc += p[2]
    if len(tail) == len(paras) and len(paras) > 1:
        tail = tail[1:]
    return tail


def chunk_document(
    text: str,
    *,
    tokenizer: Optional[Callable] = None,
    target_tokens: int = 800,
    max_tokens: int = 1000,
    overlap_tokens: int = 150,
    max_chunks_per_doc: int = 0,
) -> List[Chunk]:
    """Split `text` into overlapping token-bounded chunks.

    1. Split on "\n\n" into paragraphs (char offsets tracked).
    2. Greedily pack paragraphs until the next would exceed target_tokens.
    3. A single paragraph over max_tokens is hard-split into token windows with
       overlap (offset mapping keeps boundaries on real char positions).
    4. Paragraph-granular overlap (>= overlap_tokens) is carried into the next
       chunk.

    Invariant (asserted in tests): text[c.char_start:c.char_end] == c.text.
    Empty/blank text yields zero chunks. max_chunks_per_doc > 0 truncates
    (0 = unlimited).
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    step = max(1, max_tokens - overlap_tokens)
    out: List[Chunk] = []

    def emit(paras: List[Tuple[int, int, int]]) -> None:
        if not paras:
            return
        s = paras[0][0]
        e = paras[-1][1]
        out.append(Chunk(len(out), s, e, text[s:e]))

    current: List[Tuple[int, int, int]] = []
    current_tokens = 0

    for p_start, p_end in _split_paragraphs(text):
        offsets = _token_offsets(tokenizer, text[p_start:p_end])
        n_tok = len(offsets)
        if n_tok == 0:
            continue

        if n_tok > max_tokens:
            # Flush whatever is buffered, then hard-split this oversized
            # paragraph into standalone token-window chunks.
            emit(current)
            current, current_tokens = [], 0
            i = 0
            while i < n_tok:
                j = min(i + max_tokens, n_tok)
                s = p_start + offsets[i][0]
                e = p_start + offsets[j - 1][1]
                emit([(s, e, j - i)])
                if j == n_tok:
                    break
                i += step
            continue

        if current and current_tokens + n_tok > target_tokens:
            emit(current)
            current = _overlap_tail(current, overlap_tokens)
            current_tokens = sum(p[2] for p in current)

        current.append((p_start, p_end, n_tok))
        current_tokens += n_tok

    emit(current)

    if max_chunks_per_doc and len(out) > max_chunks_per_doc:
        out = out[:max_chunks_per_doc]
    return out
