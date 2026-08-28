"""Derive mevzuat identity (title, type, number, Resmî Gazete reference) from content.

EPDK filenames are opaque download-ID slugs, so identity must come from the
document's own opening lines. The patterns here were derived by surveying real
extracted output from this corpus, which opens in a handful of recognisable
shapes:

    ELEKTRİK PİYASASI KANUNU              <- title first, ALL CAPS
    Kanun No.  : 4628

    5 Temmuz 2012 PERŞEMBE  Resmî Gazete  Sayı : 28344   <- RG banner first
    KANUN                                                <- type marker
    YARGI HİZMETLERİNİN ... DAİR KANUN                   <- then the title

    Enerji Piyasası Düzenleme Kurumundan:   <- issuing-body preamble
    KURUL KARARI
    Karar No: 10695

Anything that cannot be read off the text with reasonable confidence is left as
None and flagged. Guessing a plausible-looking title would be worse than
admitting ignorance, because downstream citations would silently attribute text
to the wrong regulation.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .extract import ExtractedDoc, tr_lower, tr_upper

# How much of the document counts as "the first page or two" for title hunting.
HEAD_CHARS = 4000
HEAD_LINES = 40

MEVZUAT_TYPES = ("Kanun", "Yönetmelik", "Tebliğ", "Kurul Kararı", "Usul ve Esaslar")

# Lines that are structural markers rather than the title itself.
_TYPE_MARKERS = {
    "kanun": "Kanun",
    "yönetmelik": "Yönetmelik",
    "yönetmeli̇k": "Yönetmelik",
    "tebliğ": "Tebliğ",
    "kurul kararı": "Kurul Kararı",
    "bakanlar kurulu kararı": "Kurul Kararı",
    "cumhurbaşkanı kararı": "Kurul Kararı",
    "usul ve esaslar": "Usul ve Esaslar",
}

# Preamble lines that precede the real title and must not be mistaken for it.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"[\w\s]*kurumundan\s*:?"          # "Enerji Piyasası Düzenleme Kurumundan:"
    r"|[\w\s]*bakanlığından\s*:?"
    r"|resm[iî]\s*gazete.*"
    r"|karar\s*(?:no|sayısı)\s*[:.].*"
    r"|kanun\s*(?:no|numarası)\s*[:.].*"
    r"|kabul\s*tarihi\s*[:.].*"
    r"|yayım(?:landığı)?\s*.*"
    r"|karar\s*tarihi\s*[:.].*"
    r"|amaç(?:\s*ve\s*kapsam)?"
    r"|dayanak"
    r")\s*$",
    re.IGNORECASE,
)

# "5 Temmuz 2012 PERŞEMBE  Resmî Gazete  Sayı : 28344"
_MONTHS = {
    "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6,
    "temmuz": 7, "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12,
}
_RG_BANNER_RE = re.compile(
    r"(\d{1,2})\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)\s+(\d{4}).{0,40}?resm[iî]\s*gazete.*?"
    r"say[ıi]\s*[:：]?\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
# "Yayımlandığı R. Gazete : Tarih : 3/3/2001 Sayı : 24335"
_RG_FIELD_RE = re.compile(
    r"yayım(?:landığı)?\s*(?:r\.?\s*|resm[iî]\s*)?gazete\s*[:：]?\s*"
    r"(?:tarih\s*[:：]?\s*)?(\d{1,2}[./]\d{1,2}[./]\d{4}).{0,30}?say[ıi]\s*[:：]?\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
# "Yayım Tarihi : 3/3/2001 tarih ve 24335 Sayılı Resmi Gazete"
_RG_INLINE_RE = re.compile(
    r"(\d{1,2}[./]\d{1,2}[./]\d{4})\s*tarih(?:li)?\s*ve\s*(\d+)\s*say[ıi]l[ıi]\s*resm[iî]?\s*gazete",
    re.IGNORECASE,
)

# The label and its value are separated by an arbitrary run of punctuation and
# whitespace in real documents ("Kanun No.\t: 4628", "Kanun Numarası\t\t: 5346").
_NUMBER_PATTERNS = (
    re.compile(r"kanun\s*(?:no|numarası)\b[\s.:：]{0,8}(\d{3,5})", re.IGNORECASE),
    re.compile(r"karar\s*(?:no|sayısı)\b[\s.:：]{0,8}([\d/\-]+)", re.IGNORECASE),
    re.compile(r"mevzuat\s*no\b[\s.:：]{0,8}(\d+)", re.IGNORECASE),
)


# Type words, matched suffix-tolerantly because Turkish agglutinates:
# "YÖNETMELİK" also appears as "YÖNETMELİĞİ" / "YÖNETMELİĞİNDE" (with the
# k -> ğ mutation), and "KANUN" as "KANUNUNDA". A bare \bword\b test would miss
# all of those. The lookahead on `karar` excludes "KARARNAME", a different word
# that merely starts the same way. Order matters: the list is checked top-down,
# so a title ending "...KANUNUNDA DEĞİŞİKLİK ... DAİR YÖNETMELİK" resolves to
# Yönetmelik rather than Kanun.
_TYPE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\busul\s+ve\s+esas\w*"), "Usul ve Esaslar"),
    (re.compile(r"\byönetmeli[kğ]\w*"), "Yönetmelik"),
    (re.compile(r"\btebli[ğg]\w*"), "Tebliğ"),
    (re.compile(r"\bkurul\s+karar\w*"), "Kurul Kararı"),
    (re.compile(r"\bkarar(?!name)\w*"), "Kurul Kararı"),
    (re.compile(r"\bkanun\w*"), "Kanun"),
)


@dataclass
class TitleInfo:
    """Identity derived from a document's content: title, type, number, RG reference."""

    title: str | None = None
    mevzuat_type: str | None = None
    number: str | None = None
    rg_date: str | None = None  # ISO yyyy-mm-dd
    rg_number: str | None = None
    confidence: str = "none"  # high | medium | low | none
    flags: list[str] = field(default_factory=list)


def _head_lines(text: str) -> list[str]:
    head = text[:HEAD_CHARS]
    return [ln.strip() for ln in head.splitlines() if ln.strip()][:HEAD_LINES]


def _is_probably_title(line: str, min_len: int = 12) -> bool:
    """Heuristic: titles are long-ish, mostly uppercase, and not boilerplate.

    `min_len` is relaxed for continuation lines: a heading wrapped across lines
    often ends on a very short tail such as "DAİR KANUN", and dropping it both
    truncates the title and hides the type word that classification relies on.
    """
    if len(line) < min_len or len(line) > 300:
        return False
    if _PREAMBLE_RE.match(line):
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < max(3, min(8, min_len)):
        return False
    upper = sum(1 for c in letters if tr_upper(c) == c)
    return upper / len(letters) >= 0.75


def _looks_like_marker(line: str) -> str | None:
    key = tr_lower(unicodedata.normalize("NFC", line)).strip(" :.-–—")
    return _TYPE_MARKERS.get(key)


def _classify_type(title: str | None, text: str) -> tuple[str | None, str | None]:
    """Infer mevzuat type from the title's ending, then from marker lines.

    Returns (type, evidence). Checks the most specific words first: a title
    ending in "... KANUNUNDA DEĞİŞİKLİK YAPILMASINA DAİR YÖNETMELİK" is a
    Yönetmelik, not a Kanun, so trailing-word order matters.
    """
    if title:
        low = tr_lower(title)
        for pattern, name in _TYPE_PATTERNS:
            matches = list(pattern.finditer(low))
            if matches:
                idx = matches[-1].start()
                # Prefer a match near the end, where the type word normally sits.
                if idx >= len(low) - 60:
                    return name, f"title ends with {matches[-1].group(0)!r}"
    for line in _head_lines(text)[:12]:
        marker = _looks_like_marker(line)
        if marker:
            return marker, f"marker line {line[:40]!r}"
    if title:
        low = tr_lower(title)
        for word, name in (
            ("yönetmelik", "Yönetmelik"),
            ("tebliğ", "Tebliğ"),
            ("kanun", "Kanun"),
        ):
            if word in low:
                return name, f"title contains {word!r}"
    return None, None


# Fields that commonly follow the title on the SAME line, e.g.
#   "... DAİR KANUN Kanun No. 7226 Kabul Tarihi: 25/3/2020 MADDE 1 – ..."
# Cutting here recovers the heading from an otherwise mixed-case line.
_TRAILING_FIELD_RE = re.compile(
    r"\s+(?="
    r"Kanun\s*(?:No|Numarası)\b"
    r"|Karar\s*(?:No|Sayısı|Tarihi)\b"
    r"|Kabul\s*Tarihi\b"
    r"|Yayım(?:landığı)?\b"
    r"|Resm[iî]\s*Gazete\b"
    r"|(?:GEÇİCİ\s+|EK\s+)?MADDE\s+\d"
    r"|Amaç\b"
    r")",
    re.IGNORECASE,
)

# "Enerji Piyasası Düzenleme Kurumundan: 6446 SAYILI ... TEBLİĞ"
_ISSUER_PREFIX_RE = re.compile(
    r"^\s*[\w\s.]*?(?:Kurumundan|Bakanlığından|Kurulundan)\s*:\s*", re.IGNORECASE
)


# The Resmî Gazete banner ("5 Temmuz 2012 PERŞEMBE  Resmî Gazete  Sayı : 28344")
# is publication metadata, never the title, but its uppercase weekday can push
# it past the heading test -- so it is recognised and skipped explicitly.
_RG_BANNER_LINE_RE = re.compile(
    r"^\s*\d{1,2}\s+\S+\s+\d{4}\b.*resm[iî]\s*gazete", re.IGNORECASE
)


def _clean_title_line(line: str) -> str:
    """Strip an issuer preamble and any metadata trailing the heading.

    Corpus lines routinely run the title straight into "Kanun No. ...",
    "Kabul Tarihi: ..." or the first MADDE, and prefix it with
    "Enerji Piyasası Düzenleme Kurumundan:". Both are removed here so the
    heading test sees the heading alone.
    """
    candidate = _ISSUER_PREFIX_RE.sub("", line).strip()
    cut = _TRAILING_FIELD_RE.search(candidate)
    if cut:
        candidate = candidate[: cut.start()].strip()
    return re.sub(r"\s+", " ", candidate).strip(" .;:,")


def _extract_title(lines: list[str]) -> tuple[str | None, str]:
    """Find the title, joining consecutive uppercase lines that form one heading."""
    for i, line in enumerate(lines[:20]):
        if _looks_like_marker(line) or _RG_BANNER_LINE_RE.match(line):
            continue
        cleaned = _clean_title_line(line)
        if not cleaned or not _is_probably_title(cleaned):
            continue
        how = "uppercase heading" if cleaned == line.strip() else "heading recovered from mixed line"

        parts = [cleaned]
        # A heading wrapped across lines continues while the next line is also
        # title-shaped. Stop once metadata was trimmed, since the trimmed field
        # marks the end of the heading.
        if cleaned == line.strip():
            for nxt in lines[i + 1 : i + 4]:
                if _looks_like_marker(nxt) or _RG_BANNER_LINE_RE.match(nxt):
                    break
                nxt_clean = _clean_title_line(nxt)
                # Short tails ("DAİR KANUN") are legitimate continuations.
                if not nxt_clean or not _is_probably_title(nxt_clean, min_len=4):
                    break
                parts.append(nxt_clean)
                if nxt_clean != nxt.strip() or len(" ".join(parts)) > 260:
                    break
        title = re.sub(r"\s+", " ", " ".join(parts)).strip(" .;:")
        return title, how
    return None, "no uppercase heading found"


def _extract_rg(text: str) -> tuple[str | None, str | None]:
    head = text[:HEAD_CHARS]

    m = _RG_FIELD_RE.search(head)
    if m:
        return _iso_date(m.group(1)), m.group(2)

    m = _RG_INLINE_RE.search(head)
    if m:
        return _iso_date(m.group(1)), m.group(2)

    m = _RG_BANNER_RE.search(head)
    if m:
        day, month_name, year, number = m.groups()
        month = _MONTHS.get(tr_lower(month_name))
        if month:
            return f"{int(year):04d}-{month:02d}-{int(day):02d}", number
        return None, number
    return None, None


def _iso_date(raw: str) -> str | None:
    parts = re.split(r"[./]", raw)
    if len(parts) != 3:
        return None
    day, month, year = parts
    try:
        d, mo, y = int(day), int(month), int(year)
    except ValueError:
        return None
    if not (1 <= mo <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _extract_number(text: str) -> str | None:
    head = text[:HEAD_CHARS]
    for pattern in _NUMBER_PATTERNS:
        m = pattern.search(head)
        if m:
            return m.group(1).strip(" .:-")
    return None


def extract_title(doc: ExtractedDoc) -> TitleInfo:
    """Derive identity from document content. Unknown fields stay None + flagged."""
    info = TitleInfo()
    text = doc.text
    if not text.strip():
        info.flags.append("title: document has no text to derive identity from")
        return info

    lines = _head_lines(text)
    title, how = _extract_title(lines)
    info.title = title

    mevzuat_type, type_evidence = _classify_type(title, text)
    info.mevzuat_type = mevzuat_type
    info.number = _extract_number(text)
    info.rg_date, info.rg_number = _extract_rg(text)

    # Confidence reflects how much corroborating evidence was found, so a
    # downstream consumer can require "high" before trusting a citation.
    signals = sum(
        bool(x) for x in (info.title, info.mevzuat_type, info.number, info.rg_number)
    )
    if info.title and signals >= 3:
        info.confidence = "high"
    elif info.title and signals >= 2:
        info.confidence = "medium"
    elif info.title:
        info.confidence = "low"
    else:
        info.confidence = "none"

    if not info.title:
        info.flags.append(f"title-missing: {how}")
    if not info.mevzuat_type:
        info.flags.append("mevzuat-type-unknown: no type word in title or marker line")
    if not info.number:
        info.flags.append("number-missing: no 'Kanun No'/'Karar No' field found")
    if not info.rg_number:
        info.flags.append("rg-missing: no Resmî Gazete reference found")
    if info.title and info.confidence == "low":
        info.flags.append("title-low-confidence: title found but no corroborating metadata")

    return info
