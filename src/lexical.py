"""BM25 lexical retrieval over the stored chunks, with Turkish diacritic folding.

This exists because Step 4's calibration measured two failure modes that dense
embeddings cannot fix on their own:

1. Domain-vocabulary mismatch. "Doğal gaz dağıtım şirketlerinin abone bağlantı
   bedeli" retrieved the *electricity* Dağıtım Bağlantı Bedelleri Tebliği at
   cosine 0.6405 -- above the ANSWER threshold. Every word matched except the
   one that mattered. A lexical index scores that query low precisely because
   "doğal gaz" appears nowhere in an electricity-only corpus.

2. Diacritic sensitivity. ASCII-typed queries scored up to 0.23 lower and
   flipped 8 of 21 gate decisions. BM25 fixes this outright by folding
   diacritics symmetrically on both sides of the match.

`fold_diacritics()` is deliberately NOT `extract.tr_lower()`. tr_lower/tr_upper
implement *correct* Turkish casing (I->ı, İ->i) and are used wherever real
Turkish text is produced or compared. fold_diacritics() destroys exactly that
information on purpose, to make "önlisans" and "onlisans" the same index key.
The two must never be composed: folded text is a lexical matching key, not
Turkish, and is never displayed or embedded.
"""
from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

from . import config

# Turkish-specific letters -> plain ASCII neighbour. Lossy and intentionally so:
# this is a lexical matching key, never text shown to a user.
_FOLD_MAP = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U",
    # Circumflex vowels appear in older mevzuat ("hâl", "kâr") and must fold to
    # the same key as their plain forms or those terms split into two tokens.
    "â": "a", "î": "i", "û": "u", "Â": "A", "Î": "I", "Û": "U",
})


def fold_diacritics(text: str) -> str:
    """Map Turkish-specific letters to their plain ASCII neighbours.

    For LEXICAL MATCHING ONLY. This is not case folding and is not a Turkish
    operation: it is deliberately lossy so that a query typed without
    diacritics ("onlisans suresi") produces the same tokens as the correctly
    spelled corpus text ("önlisans süresi").

    Never combine with extract.tr_lower()/tr_upper(). Those implement real
    Turkish casing rules and operate on real Turkish text; this operates on
    text that is about to stop being Turkish. Composing them would apply
    Turkish casing to already-ASCII-folded input, which is meaningless, and
    would corrupt any text that is subsequently displayed.
    """
    return text.translate(_FOLD_MAP)


# Applied AFTER folding, so plain ASCII casing is correct by construction:
# every i-variant (i, ı, İ, I) has already collapsed to i/I by this point.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately small and purely functional: conjunctions, postpositions,
# pronouns and question words. Legal-domain nouns are never stopworded, because
# in this corpus "bedel", "süre" or "usul" are exactly the discriminating
# terms. Written in real Turkish for readability, folded once at import.
_STOPWORDS_TR = {
    "ve", "veya", "ile", "ya", "ancak", "fakat", "ama",
    "bu", "şu", "o", "bir", "birer", "her", "hiç", "tüm", "bütün",
    "için", "gibi", "göre", "kadar", "sonra", "önce", "üzere", "ise", "de", "da",
    "ki", "mi", "mı", "mu", "mü", "ne", "nedir", "nasıl", "hangi", "kim", "kimdir",
    "olan", "olarak", "olur", "olan", "eden", "edilir", "yapılır",
    "en", "çok", "az", "daha", "ayrıca", "ancak",
}
STOPWORDS = {fold_diacritics(w).lower() for w in _STOPWORDS_TR}

# NO STEMMING / SUFFIX STRIPPING -- measured, not assumed.
#
# scripts/eval_stemming.py A/B'd a light suffix stripper (Turkish plural/case/
# possessive endings, max two passes, min stem 4) against this tokenizer over
# all 27,047 chunks, scoring recall@k on the 15 answerable calibration
# questions labelled by the document each should retrieve from:
#
#     depth 10:  12/15 vs 12/15   (+0)
#     depth 25:  13/15 vs 12/15   (-1)
#     depth 50:  14/15 vs 14/15   (+0)
#
# No gain anywhere, a regression at 25, and it made the not-answerable
# "rafinerici / ulusal petrol stoku" question score HIGHER (23.94 -> 31.74),
# which is the wrong direction for the exact failure mode BM25 is here to fix.
# It also damages the multi-word terms this corpus turns on:
#
#     iletim tarifesi    -> ['iletim', 'tarif']     (tarife/tarifname collide)
#     bağlantı anlaşması -> ['baglant', 'anlasm']   (stem is not a word)
#     dağıtım bedeli / dağıtım bedelleri -> both ['dagitim', 'bedel']
#
# Turkish is agglutinative enough that correct stemming needs a real
# morphological analyzer, not a suffix list; a wrong one silently merges
# distinct legal terms for no measured benefit. Re-run eval_stemming.py before
# revisiting this.


def tokenize(text: str) -> list[str]:
    """Fold diacritics, lowercase, split on non-alphanumerics, drop stopwords.

    Applied identically to indexed chunk text and to incoming queries -- the
    symmetry is the whole point, and is what makes "önlisans süresi" and
    "onlisans suresi" retrieve the same chunks.
    """
    folded = fold_diacritics(text).lower()
    return [t for t in _TOKEN_RE.findall(folded) if t not in STOPWORDS and len(t) > 1]


@dataclass
class BM25Index:
    """An in-memory BM25 index over the active chunks.

    Built once at startup alongside the dense matrix. The postings list is a
    plain dict of term -> [(doc position, term frequency)], which for 27k
    chunks costs well under a second to build and a few milliseconds to query
    -- no need for a separate search engine dependency.
    """

    ids: list[int]  # chunk_id per document position
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    doc_lengths: list[int] = field(default_factory=list)
    avg_doc_length: float = 0.0
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(
        cls,
        documents: list[tuple[int, str]],
        k1: float | None = None,
        b: float | None = None,
    ) -> "BM25Index":
        """Build from (chunk_id, text) pairs, in the order given."""
        k1 = config.BM25_K1 if k1 is None else k1
        b = config.BM25_B if b is None else b

        ids: list[int] = []
        doc_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for position, (chunk_id, text) in enumerate(documents):
            tokens = tokenize(text)
            ids.append(chunk_id)
            doc_lengths.append(len(tokens))
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for term, tf in counts.items():
                postings[term].append((position, tf))

        avg = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 0.0
        return cls(
            ids=ids,
            postings=dict(postings),
            doc_lengths=doc_lengths,
            avg_doc_length=avg,
            k1=k1,
            b=b,
        )

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> "BM25Index":
        """Build over every active chunk, ordered by id to match the dense matrix."""
        rows = conn.execute(
            "SELECT id, text FROM chunks WHERE active = 1 ORDER BY id"
        ).fetchall()
        return cls.build([(row[0], row[1]) for row in rows])

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF, floored at 0 so a near-universal term cannot
        contribute a negative score and push a document below one that lacks it."""
        n = len(self.ids)
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Score the query against the index. Returns (chunk_id, score), best first.

        A query whose terms appear nowhere in the corpus scores nothing at all
        and returns an empty list -- which is the desired answer for a "doğal
        gaz" question against an electricity-only corpus, and the signal that
        fusion uses to demote it.
        """
        terms = tokenize(query)
        if not terms or not self.ids or top_k <= 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            if idf <= 0.0:
                continue
            for position, tf in postings:
                length_norm = 1.0 - self.b + self.b * (
                    self.doc_lengths[position] / self.avg_doc_length
                    if self.avg_doc_length
                    else 1.0
                )
                scores[position] += idf * (tf * (self.k1 + 1.0)) / (
                    tf + self.k1 * length_norm
                )

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [(self.ids[position], score) for position, score in ranked]
