"""Chunk EPDK regulation text on article (MADDE) boundaries.

Strategy, in order of preference:

1. `article`      - one chunk per MADDE, the natural citation unit for legal text.
2. `article-sub`  - an oversized article split on its own (1)/(2)/a)/b) sub-items.
3. `article-window` - a sub-item that is still oversized, cut by token window.
4. `token-window` - documents with no article structure at all.

MADDE, GEÇİCİ MADDE and EK MADDE are *separate numbering namespaces*: a document
can contain both "MADDE 5" and "GEÇİCİ MADDE 5" meaning entirely different
provisions. They are therefore kept distinct in `ArticleRef.kind`, and a
citation must carry the kind as well as the number.

Separator characters were surveyed across the real corpus: en-dash (–) is the
most common by far (895 occurrences in a 120-document sample), then hyphen (263),
then '.' — all three are accepted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from .extract import ExtractedDoc, Page, tr_lower
from .titles import TitleInfo

# MADDE headings. The article number may carry a letter suffix ("MADDE 5/A"),
# a real construct for inserted articles. Anchored to line start so inline
# cross-references ("... 5 inci maddesinde") are not mistaken for headings.
_ARTICLE_RE = re.compile(
    r"^[ \t]*(?P<kind>geçici\s+madde|ek\s+madde|madde)\s*"
    r"(?P<num>\d+(?:\s*/\s*[A-ZÇĞİÖŞÜa-zçğıöşü])?)\s*"
    r"(?P<sep>[–\-—.:]|\s)\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Sub-item starts: "(1)", "(2)", "a)", "b)", "1)" - used to split long articles.
_SUBITEM_RE = re.compile(
    r"^[ \t]*(?:\((?P<paren>\d+)\)|(?P<letter>[a-zçğıöşü])\)|(?P<num>\d+)\))\s+",
    re.MULTILINE,
)

_KIND_CANON = {"madde": "MADDE", "geçici madde": "GEÇİCİ MADDE", "ek madde": "EK MADDE"}


@dataclass(frozen=True)
class ArticleRef:
    kind: str  # MADDE | GEÇİCİ MADDE | EK MADDE - distinct namespaces
    number: str

    def __str__(self) -> str:
        return f"{self.kind} {self.number}"


@dataclass
class Chunk:
    doc_id: str
    text: str
    strategy: str
    index: int
    article: ArticleRef | None = None
    document_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    # Why pages are absent, when they are. Never silently omitted.
    page_note: str | None = None
    token_estimate: int = 0
    flags: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        base = self.document_title or self.doc_id
        if self.article:
            base = f"{base} - {self.article}"
        if self.page_start:
            span = (
                f"s. {self.page_start}"
                if self.page_start == self.page_end
                else f"s. {self.page_start}-{self.page_end}"
            )
            base = f"{base} ({span})"
        return base


def estimate_tokens(text: str) -> int:
    """Approximate token count. See config.TOKENS_PER_WORD for why this is an estimate."""
    words = len(text.split())
    return int(words * config.TOKENS_PER_WORD)


def _canon_kind(raw: str) -> str:
    return _KIND_CANON.get(re.sub(r"\s+", " ", tr_lower(raw)).strip(), "MADDE")


def find_articles(text: str) -> list[tuple[ArticleRef, int, int]]:
    """Locate article headings. Returns [(ref, start, end)] spanning the whole text."""
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return []
    spans: list[tuple[ArticleRef, int, int]] = []
    for i, m in enumerate(matches):
        ref = ArticleRef(
            kind=_canon_kind(m.group("kind")),
            number=re.sub(r"\s*/\s*", "/", m.group("num").strip()),
        )
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((ref, m.start(), end))
    return spans


def _split_subitems(text: str) -> list[str]:
    """Split an article on its internal sub-items, keeping each marker with its text."""
    marks = [m.start() for m in _SUBITEM_RE.finditer(text)]
    if len(marks) < 2:
        return [text]
    # Text before the first sub-item is the article's own lead-in.
    bounds = ([0] if marks[0] > 0 else []) + marks
    parts = []
    for i, start in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else len(text)
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
    return parts


def _token_windows(text: str, size: int, overlap: int) -> list[str]:
    """Cut text into overlapping windows of approximately `size` tokens."""
    words = text.split()
    if not words:
        return []
    words_per_chunk = max(1, int(size / config.TOKENS_PER_WORD))
    step = max(1, words_per_chunk - int(overlap / config.TOKENS_PER_WORD))
    windows = []
    start = 0
    while start < len(words):
        windows.append(" ".join(words[start : start + words_per_chunk]))
        if start + words_per_chunk >= len(words):
            break
        start += step
    return windows


def _pack(parts: list[str], size: int) -> list[str]:
    """Greedily merge small consecutive parts up to the target size.

    Keeps sub-items whole while avoiding a chunk per one-line clause.
    """
    packed: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}\n{part}" if current else part
        if current and estimate_tokens(candidate) > size:
            packed.append(current)
            current = part
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def _page_span_for(doc: ExtractedDoc, start: int, end: int) -> tuple[int | None, int | None]:
    """Map a character range in doc.text back to the page numbers covering it."""
    if not doc.has_pages:
        return None, None
    cursor = 0
    first = last = None
    for page in doc.pages:
        if not page.text.strip():
            continue
        page_start = cursor
        page_end = cursor + len(page.text)
        if page_start < end and page_end > start:
            if first is None:
                first = page.number
            last = page.number
        cursor = page_end + 2  # the "\n\n" join in ExtractedDoc.text
    return first, last


def chunk_document(
    doc: ExtractedDoc,
    info: TitleInfo | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Chunk a document, preferring article boundaries. Sizes come from config."""
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    ov = overlap if overlap is not None else config.CHUNK_OVERLAP
    title = info.title if info else None

    text = doc.text
    if not text.strip():
        return []

    page_note = None if doc.has_pages else (doc.page_number_note or "no page model for this format")

    def build(body: str, strategy: str, ref: ArticleRef | None, start: int, end: int) -> Chunk:
        ps, pe = _page_span_for(doc, start, end)
        return Chunk(
            doc_id=doc.doc_id,
            text=body.strip(),
            strategy=strategy,
            index=0,  # assigned after assembly
            article=ref,
            document_title=title,
            page_start=ps,
            page_end=pe,
            page_note=page_note if ps is None else None,
            token_estimate=estimate_tokens(body),
        )

    chunks: list[Chunk] = []
    articles = find_articles(text)

    if not articles:
        # No article structure: fall back to plain token windows.
        for window in _token_windows(text, size, ov):
            chunks.append(build(window, "token-window", None, 0, len(text)))
    else:
        # Preamble before the first article (title block, purpose, etc.).
        lead = text[: articles[0][1]].strip()
        if estimate_tokens(lead) > 20:
            for window in _token_windows(lead, size, ov):
                chunks.append(build(window, "preamble", None, 0, articles[0][1]))

        for ref, start, end in articles:
            body = text[start:end].strip()
            if not body:
                continue
            if estimate_tokens(body) <= size:
                chunks.append(build(body, "article", ref, start, end))
                continue

            # Oversized article: split on its own sub-items first.
            parts = _pack(_split_subitems(body), size)
            if len(parts) > 1:
                for part in parts:
                    if estimate_tokens(part) <= size:
                        chunks.append(build(part, "article-sub", ref, start, end))
                    else:
                        for window in _token_windows(part, size, ov):
                            chunks.append(build(window, "article-window", ref, start, end))
            else:
                for window in _token_windows(body, size, ov):
                    chunks.append(build(window, "article-window", ref, start, end))

    for i, chunk in enumerate(chunks):
        chunk.index = i
    return [c for c in chunks if c.text.strip()]
