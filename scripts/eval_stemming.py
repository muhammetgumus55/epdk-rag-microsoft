"""Does light Turkish suffix stripping help BM25 on this corpus? Measure, don't guess.

src/lexical.py deliberately ships with NO stemming. This script is the evidence
for that decision, and exists so the decision can be revisited with numbers
rather than re-argued from intuition.

Method. Each of the 15 answerable calibration questions is labelled with the
document it should retrieve from -- labels taken from the Step 4 dense
calibration output, checked by reading the retrieved article. Recall@k is then
the share of questions for which BM25's top k chunks include at least one chunk
from the labelled document. That is a document-level label, which is coarse but
verifiable, unlike a chunk-level one.

The 6 not-answerable questions are scored the other way: BM25 SHOULD return
little or nothing for them, so we report how many retrieve anything at all.

Separately, the multi-word legal terms the corpus turns on are tokenized under
both schemes and printed, because a stemmer that improves recall while merging
"dağıtım bedeli" into something that also matches "dağıtım bedelleri
tarifesi" has not actually helped.

Usage:
    .venv\\Scripts\\python.exe scripts\\eval_stemming.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import store  # noqa: E402
from src.extract import _force_utf8_stdout  # noqa: E402
from src.lexical import STOPWORDS, BM25Index, fold_diacritics  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Ordered longest-first: Turkish agglutinates, so a short suffix would strip a
# prefix of a longer one and leave a fragment. Applied at most twice per token,
# which covers the common plural + case stack ("sirketlerin" -> "sirket").
_SUFFIXES = (
    "lerinin", "larinin", "lerini", "larini", "lerin", "larin", "leri", "lari",
    "ndan", "nden", "nin", "nun", "nun", "tan", "ten", "dan", "den",
    "ler", "lar", "in", "un", "ye", "ya", "yi", "yu", "de", "da", "te", "ta",
    "si", "su", "i", "e", "a", "u",
)
MIN_STEM = 4


def stem(token: str) -> str:
    """Light suffix stripping: at most two passes, never below MIN_STEM chars."""
    for _ in range(2):
        for suffix in _SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM:
                token = token[: -len(suffix)]
                break
        else:
            break
    return token


def tokenize_stemmed(text: str) -> list[str]:
    folded = fold_diacritics(text).lower()
    return [
        stem(t) for t in _TOKEN_RE.findall(folded) if t not in STOPWORDS and len(t) > 1
    ]


# (question, substring that must appear in the retrieved chunk's document_title)
LABELLED = [
    ("Elektrik piyasasında üretim lisansı almak için başvuru sahibinin sağlaması "
     "gereken şartlar nelerdir?", "LİSANS YÖNETMELİĞİ"),
    ("Gün öncesi piyasasında teklif verme ve eşleştirme süreci nasıl işler?",
     "DENGELEME VE UZLAŞTIRMA"),
    ("Lisanssız elektrik üretiminde çatı tipi güneş enerjisi santralleri için "
     "kurulu güç sınırı nedir?", "LİSANSSIZ ELEKTRİK"),
    ("Yenilenebilir enerji kaynak belgesi (YEK belgesi) nasıl alınır?",
     "YENİLENEBİLİR ENERJİ"),
    ("Dağıtım şirketinin tüketiciye planlı kesinti öncesinde bildirim yapma "
     "yükümlülüğü nedir?", "BAĞLANTI VE SİSTEM KULLANIM"),
    ("Bağlantı anlaşması hangi hallerde sona erer veya feshedilir?", "BAĞLANTI"),
    ("Yan hizmetler kapsamında primer frekans kontrol hizmeti nasıl tedarik edilir?",
     "YAN HİZMETLER"),
    ("Serbest tüketici kimdir ve tedarikçisini değiştirme hakkını nasıl kullanır?",
     "TÜKETİCİ"),
    ("Kapasite mekanizması kapsamında yapılacak ödemeler nasıl hesaplanır?",
     "KAPASİTE MEKANİZMASI"),
    ("Elektrik enerjisi ithalat ve ihracat faaliyeti için hangi lisans gereklidir?",
     "İTHALAT VE İHRACAT"),
    ("Sayaçların okunması ve tüketim değerlerinin belirlenmesine ilişkin usul ve "
     "esaslar nelerdir?", "SAYAÇ"),
    ("Dağıtım tarifesinin düzenlenmesinde gelir tavanı nasıl belirlenir?",
     "DAĞITIM TARİFESİ"),
    ("Piyasa işletmecisine verilecek teminatların türleri ve tutarı nasıl hesaplanır?",
     "TEMİNAT"),
    ("Lisans sahiplerine uygulanacak idari para cezaları nelerdir?",
     "ELEKTRİK PİYASASI KANUNU"),
    ("İletim sistemi işletmecisinin arz güvenliğine ilişkin yükümlülükleri nelerdir?",
     "ŞEBEKE YÖNETMELİĞİ"),
]

NOT_ANSWERABLE = [
    "Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli nasıl hesaplanır?",
    "LPG otogaz istasyonlarında sorumlu müdür bulundurma zorunluluğu nedir?",
    "Akaryakıt bayilik lisansı için aranan asgari sermaye şartı nedir?",
    "Rafinerici lisansı sahiplerinin ulusal petrol stoku tutma yükümlülüğü nedir?",
    "Konut kira sözleşmesinin feshi için ihtarname süresi ne kadardır?",
    "Deniz balıkçılığında av yasağı dönemleri hangi aylardır?",
]

# The terms that must not be damaged, whatever the recall numbers say.
PHRASES = [
    "dağıtım bedeli", "iletim tarifesi", "serbest tüketici", "önlisans süresi",
    "dağıtım bedelleri", "serbest tüketiciler", "bağlantı anlaşması",
]

DEPTHS = (10, 25, 50)


def recall_at(index: BM25Index, titles: dict[int, str], depth: int) -> tuple[int, list[str]]:
    hits, misses = 0, []
    for question, expected in LABELLED:
        found = index.search(question, depth)
        if any(expected in (titles.get(cid) or "") for cid, _ in found):
            hits += 1
        else:
            misses.append(f"{expected} <- {question[:60]}")
    return hits, misses


def main() -> int:
    _force_utf8_stdout()
    conn = store.connect()
    rows = conn.execute(
        "SELECT id, text, document_title FROM chunks "
        "WHERE active = 1 AND indexable = 1 ORDER BY id"
    ).fetchall()
    titles = {row[0]: row[2] for row in rows}
    documents = [(row[0], row[1]) for row in rows]
    print(f"Corpus: {len(documents):,} active chunks\n")

    print("Building baseline index (no stemming) ...")
    baseline = BM25Index.build(documents)

    print("Building stemmed index ...")
    import src.lexical as lexical

    original = lexical.tokenize
    lexical.tokenize = tokenize_stemmed
    try:
        stemmed = BM25Index.build(documents)
    finally:
        lexical.tokenize = original

    print(f"  vocabulary: baseline {baseline.vocabulary_size:,} terms, "
          f"stemmed {stemmed.vocabulary_size:,} terms "
          f"({100 * (1 - stemmed.vocabulary_size / baseline.vocabulary_size):.1f}% smaller)\n")

    print("=" * 78)
    print("RECALL@k ON THE 15 LABELLED ANSWERABLE QUESTIONS")
    print("=" * 78)
    print(f"{'depth':>6}  {'no stemming':>12}  {'stemmed':>12}   delta")
    all_misses = {}
    for depth in DEPTHS:
        # Stemmed index needs stemmed queries -- the symmetry that makes it work.
        lexical.tokenize = original
        base_hits, base_misses = recall_at(baseline, titles, depth)
        lexical.tokenize = tokenize_stemmed
        stem_hits, stem_misses = recall_at(stemmed, titles, depth)
        lexical.tokenize = original
        all_misses[depth] = (base_misses, stem_misses)
        n = len(LABELLED)
        print(f"{depth:>6}  {base_hits:>4}/{n} ({base_hits / n:>5.1%})  "
              f"{stem_hits:>4}/{n} ({stem_hits / n:>5.1%})   {stem_hits - base_hits:+d}")

    print()
    print("Misses at depth 50 (no stemming):")
    for miss in all_misses[50][0] or ["  none"]:
        print(f"  {miss}")
    print("Misses at depth 50 (stemmed):")
    for miss in all_misses[50][1] or ["  none"]:
        print(f"  {miss}")

    print()
    print("=" * 78)
    print("NOT-ANSWERABLE QUESTIONS (fewer/weaker hits is better)")
    print("=" * 78)
    for question in NOT_ANSWERABLE:
        lexical.tokenize = original
        b = baseline.search(question, 10)
        lexical.tokenize = tokenize_stemmed
        s = stemmed.search(question, 10)
        lexical.tokenize = original
        print(f"  baseline {len(b):>2} hits (top {b[0][1]:.2f})  |  "
              f"stemmed {len(s):>2} hits (top {s[0][1]:.2f})  |  {question[:52]}"
              if b and s else
              f"  baseline {len(b):>2} hits  |  stemmed {len(s):>2} hits  |  {question[:52]}")

    print()
    print("=" * 78)
    print("PHRASE INTEGRITY (do the multi-word legal terms survive?)")
    print("=" * 78)
    for phrase in PHRASES:
        lexical.tokenize = original
        base = original(phrase)
        stemmed_tokens = tokenize_stemmed(phrase)
        mark = "  " if base != stemmed_tokens else "= "
        print(f"{mark}{phrase:<24} {base} -> {stemmed_tokens}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
