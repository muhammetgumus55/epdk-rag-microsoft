"""Domain-scope classification: which corpus text is actually about electricity regulation.

Why this module exists
----------------------
The corpus was downloaded from EPDK's electricity mevzuat pages, so every file
*arrived* under an electricity heading. Its CONTENT is not all electricity. A
large share of the Kanunlar tree is "torba kanun" -- Turkish omnibus acts that
amend dozens of unrelated codes in one instrument. A single such file legitimately
contains the article that amended the Electricity Market Law *and*, three articles
later, one amending the Criminal Procedure Code, the Tax Procedure Law, or the
Trade Unions Act.

Indexing those whole meant questions with no connection to electricity
(kıdem tazminatı, trafik cezası, vergi levhası) retrieved real, well-scoring
chunks and were answered. That is a corpus-scope defect, not a gate-tuning
defect: no confidence cutoff can fix retrieving genuinely relevant text for a
question the corpus should never have been able to answer at all.

What it does
------------
`document_scope()` sorts every document into one of three dispositions, and the
ingest path acts on that:

1. EXCLUDED  -- a single-subject law that is simply not EPDK electricity-market
   regulation, listed by hand in `_MANUAL_EXCLUSIONS`. None of its chunks are
   indexed. The omnibus classifier cannot catch these: they are perfectly
   coherent single-subject documents, they just belong to a neighbouring
   regulator. Adjacent-but-not-EPDK law is a recurring category (nuclear today;
   plausibly gas, petroleum or mining instruments tomorrow), so the override is
   a first-class mechanism with a recorded reason per entry, not a special case.

2. OMNIBUS   -- a "torba kanun". Its articles are filtered one by one by
   `classify_text()`: is this article about electricity/energy-market
   regulation, clearly about something else, or genuinely unclear? Unclear is
   its own outcome, reported for human review rather than guessed at.

3. IN_SCOPE  -- everything else, the overwhelming majority: yönetmelikler,
   tebliğler, kurul kararları, and the Electricity Market Law itself. Indexed
   whole, never filtered, never even classified.

The conservative direction is deliberate and asymmetric. Wrongly excluding an
electricity article makes a real question unanswerable; wrongly keeping an
off-domain one puts the system back where it started. Both are bad, so anything
this module cannot decide it declines to decide, and `AMBIGUOUS` is treated as
keep-and-flag by the ingest path (see src/store.py) so nothing is silently lost.

Classification is lexical, not learned: the vocabulary of Turkish legal text is
narrow and the domain markers are unambiguous, so a term-dominance rule is both
accurate here and -- unlike a model -- auditable line by line, which matters
because every exclusion this makes is recorded in docs/decisions/.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .extract import tr_lower

ScopeLabel = Literal["ELECTRICITY", "OFF_DOMAIN", "AMBIGUOUS"]
DocumentDisposition = Literal["IN_SCOPE", "OMNIBUS", "EXCLUDED"]


# --------------------------------------------------------------------------
# Manual exclusions: single-subject documents outside EPDK electricity scope
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ManualExclusion:
    """One hand-listed document that is out of scope despite not being omnibus."""

    name: str      # what the document is, in human terms
    reason: str    # why it is out of scope -- goes into the audit trail verbatim
    pattern: re.Pattern[str]  # matched against the title and the document opening


def _exclusion(name: str, reason: str, pattern: str) -> ManualExclusion:
    return ManualExclusion(name, reason, re.compile(pattern, re.IGNORECASE))


# Matched against title + opening text rather than source_path, deliberately.
# EPDK's filenames are opaque and actively misleading -- the Nuclear Regulation
# Law is stored as "guncel-6446-sayl-elektrik-piyasas-kanunu-degisiklik-5.pdf",
# named for a law it has nothing to do with -- and the same act recurs under
# several such names as a content duplicate. Matching what the document SAYS IT
# IS survives both, and survives the corpus being re-downloaded under new names.
#
# Each entry must be specific enough not to fire on a genuine electricity
# document that merely mentions the subject: an electricity regulation may well
# reference nuclear plants, so the pattern keys on the act's own name and number.
_MANUAL_EXCLUSIONS: tuple[ManualExclusion, ...] = (
    _exclusion(
        name="Nükleer Düzenleme Kanunu (7381)",
        reason=(
            "Nuclear safety regulation, supervised by the Nükleer Düzenleme Kurumu "
            "(NDK), not EPDK. A single-subject act, so the omnibus classifier cannot "
            "reach it. The project's scope is electricity-market regulation strictly; "
            "nuclear licensing and radiation safety fall outside it even though "
            "nuclear plants generate electricity."
        ),
        pattern=r"n[üu]kleer\s+d[üu]zenleme\s+kanunu|kanun\s+no\.?\s*7381",
    ),
)


def manual_exclusion(title: str | None, text: str = "") -> ManualExclusion | None:
    """The manual exclusion covering this document, or None.

    Both the title and the document's opening are searched, because title
    extraction is heuristic and truncates: the Nuclear Regulation Law's title
    comes back as "DÜZENLEME KANUNU" with the word that identifies it missing.
    """
    haystack = f"{title or ''}\n{text[:1500]}"
    for exclusion in _MANUAL_EXCLUSIONS:
        if exclusion.pattern.search(haystack):
            return exclusion
    return None


# --------------------------------------------------------------------------
# Omnibus document detection
# --------------------------------------------------------------------------

# Turkish omnibus acts name themselves. The invariant marker is a PLURAL,
# unnamed set of amended instruments -- "bazı kanunlarda", "bazı kanun ve kanun
# hükmünde kararnamelerde", "diğer bazı kanunlarda". A title naming exactly one
# instrument ("Elektrik Piyasası Kanununda Değişiklik Yapılmasına Dair Kanun")
# is a single-subject amendment and is NOT omnibus, which is why "bazı" (some)
# rather than "değişiklik" (amendment) is what this keys on.
_OMNIBUS_TITLE_RE = re.compile(
    r"baz[ıi]\s+(?:vergi\s+)?kanun"          # "bazı kanunlarda", "bazı vergi kanunları"
    r"|di[ğg]er\s+baz[ıi]\s+kanun"           # "diğer bazı kanunlarda"
    r"|b[üu]t[çc]e\s+kanunlar[ıi]nda\s+yer\s+alan\s+baz[ıi]",
    re.IGNORECASE,
)

# A title is not always recovered (src/titles.py falls back to None on documents
# whose first heading is a Resmî Gazete banner). For those, the body announces
# itself the same way in its own opening lines.
#
# The noun is left to `\w*` rather than spelled out case by case: Turkish
# agglutination gives "kanunlarda", "kanunlarında", "kanunda" and "kanunları"
# for the same phrase, and an earlier attempt to enumerate the suffixes silently
# matched none of them -- every omnibus document in the corpus was being caught
# by its title instead, leaving this path dead and untested. The window between
# the noun and "değişiklik" absorbs the "ve kanun hükmünde kararnamelerde" that
# many of these titles carry, while still requiring the two halves to be part of
# one phrase rather than merely both present somewhere in the opening.
_OMNIBUS_BODY_RE = re.compile(
    r"(?:di[ğg]er\s+)?baz[ıi]\s+(?:vergi\s+)?kanun\w*"
    r"(?:\s+\S+){0,8}?"
    r"\s+de[ğg]i[şs]iklik\s+yap[ıi]lmas[ıi]",
    re.IGNORECASE,
)


def is_omnibus_document(title: str | None, text: str = "") -> bool:
    """Whether a document is a Turkish omnibus act ("torba kanun").

    Checked against the title first, then the document's opening text for the
    files whose title extraction returned a Resmî Gazete banner instead of the
    act's name. Only the opening is consulted: an omnibus act declares itself in
    its own name, whereas a single-subject electricity law may well *mention*
    other laws deep in its body without being one.
    """
    if title and _OMNIBUS_TITLE_RE.search(title):
        return True
    return bool(text) and bool(_OMNIBUS_BODY_RE.search(text[:1500]))


# --------------------------------------------------------------------------
# The one entry point ingestion and the audit both call
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentScope:
    """What to do with a whole document, and why."""

    disposition: DocumentDisposition
    reason: str

    @property
    def filters_articles(self) -> bool:
        """Whether individual articles still need classifying."""
        return self.disposition == "OMNIBUS"

    @property
    def excluded_entirely(self) -> bool:
        """Whether every chunk of this document is non-indexable."""
        return self.disposition == "EXCLUDED"


def document_scope(title: str | None, text: str = "") -> DocumentScope:
    """Decide a document's disposition. Manual exclusions win over everything.

    Order matters: an entry in `_MANUAL_EXCLUSIONS` is a human decision about a
    specific act and must not be second-guessed by either heuristic below it.
    """
    excluded = manual_exclusion(title, text)
    if excluded is not None:
        return DocumentScope("EXCLUDED", f"manual exclusion: {excluded.name} -- {excluded.reason}")
    if is_omnibus_document(title, text):
        return DocumentScope("OMNIBUS", "omnibus act: articles filtered individually")
    return DocumentScope("IN_SCOPE", "single-subject electricity document: indexed whole")


# --------------------------------------------------------------------------
# Article-level domain classification
# --------------------------------------------------------------------------

# The strongest available signal, and the one this classifier leads with: an
# omnibus article always names the code it amends -- "MADDE 7- 3213 sayılı
# Kanunun 16 ncı maddesi...", "MADDE 2 - 31/8/1956 tarihli ve 6831 sayılı Orman
# Kanununun...". The amended code's subject IS the article's subject, which is
# far more reliable than counting vocabulary in text that is mostly the
# mechanical language of amendment ("ibaresi ... şeklinde değiştirilmiştir").
_LAW_CODE_RE = re.compile(r"\b(\d{3,5})\s*say[ıi]l[ıi]", re.IGNORECASE)

# Codes whose subject matter IS electricity-market regulation.
_ELECTRICITY_CODES = frozenset({
    "6446",  # Elektrik Piyasası Kanunu
    "4628",  # Elektrik Piyasası Kanunu (mülga) / EPDK Teşkilat ve Görevleri
    "5346",  # Yenilenebilir Enerji Kaynaklarının Elektrik Enerjisi Üretimi
    "3096",  # Elektrik üretim/iletim/dağıtım tesislerinin görevlendirilmesi
    "4283",  # Yap-İşlet modeli ile elektrik enerjisi üretim tesisleri
    "5627",  # Enerji Verimliliği Kanunu
    "2819",  # Elektrik İşleri Etüt İdaresi
    "6094",  # 5346 sayılı Kanunda değişiklik (YEK destekleme)
    "7257",  # Elektrik Piyasası Kanunu ile bazı kanunlarda değişiklik
})

# Codes whose subject matter is definitively another field. Membership here is
# narrower than "not an electricity code" on purpose: several non-electricity
# codes are invoked constantly BY electricity regulation and appear in genuine
# electricity articles, so listing them would strip real provisions. Kept out
# for exactly that reason, and left to the vocabulary rule instead:
#
#   2942 Kamulaştırma      - every transmission line and power plant needs it
#   4046 Özelleştirme      - the whole distribution-privatisation history
#   6102 TTK / 6098 TBK    - the corporate and contract law licensees operate under
#   4734 / 2886 ihale      - how public electricity works are tendered
#   3194 İmar / 2872 Çevre / 6831 Orman / 4342 Mera - power plant siting and permits
#   7201 Tebligat, 3095 faiz, 5429 TÜİK - procedural machinery used throughout
#   4562 OSB / 4691 TGB / 4737 Endüstri Bölgeleri - zones that distribute electricity
_OFF_DOMAIN_CODES = frozenset({
    # Tax
    "213",   # Vergi Usul Kanunu
    "193",   # Gelir Vergisi Kanunu
    "5520",  # Kurumlar Vergisi Kanunu
    "3065",  # Katma Değer Vergisi Kanunu
    "4760",  # Özel Tüketim Vergisi Kanunu
    "488",   # Damga Vergisi Kanunu
    "492",   # Harçlar Kanunu
    "7338",  # Veraset ve İntikal Vergisi Kanunu
    "1319",  # Emlak Vergisi Kanunu
    "6183",  # Amme Alacaklarının Tahsil Usulü
    # Public personnel, pay, social security
    "657",   # Devlet Memurları Kanunu
    "375",   # 375 sayılı KHK (mali ve sosyal haklar)
    "5510",  # Sosyal Sigortalar ve Genel Sağlık Sigortası
    "4447",  # İşsizlik Sigortası Kanunu
    "926",   # Türk Silahlı Kuvvetleri Personel Kanunu
    # Labour
    "4857",  # İş Kanunu
    "1475",  # eski İş Kanunu (kıdem tazminatı md. 14)
    "6356",  # Sendikalar ve Toplu İş Sözleşmesi Kanunu
    "854",   # Deniz İş Kanunu
    "5953",  # Basın İş Kanunu
    # Criminal, procedure, enforcement, judiciary
    "5237",  # Türk Ceza Kanunu
    "5271",  # Ceza Muhakemesi Kanunu
    "5275",  # Ceza ve Güvenlik Tedbirlerinin İnfazı
    "2004",  # İcra ve İflas Kanunu
    "6100",  # Hukuk Muhakemeleri Kanunu
    "2577",  # İdari Yargılama Usulü Kanunu
    "2575",  # Danıştay Kanunu
    "2802",  # Hâkimler ve Savcılar Kanunu
    "5326",  # Kabahatler Kanunu
    # Traffic, family, civil status
    "2918",  # Karayolları Trafik Kanunu
    "4721",  # Türk Medeni Kanunu
    "5490",  # Nüfus Hizmetleri Kanunu
    # Education, health, defence
    "2547",  # Yükseköğretim Kanunu
    "2809",  # Yükseköğretim Kurumları Teşkilatı Kanunu
    "1111",  # Askerlik Kanunu
    "3359",  # Sağlık Hizmetleri Temel Kanunu
    # Mining, other energy sectors (energy-adjacent but not electricity market)
    "3213",  # Maden Kanunu
    "2804",  # Maden Tetkik ve Arama
    "6491",  # Türk Petrol Kanunu
    "4646",  # Doğal Gaz Piyasası Kanunu
    "5015",  # Petrol Piyasası Kanunu
    "5307",  # Sıvılaştırılmış Petrol Gazları (LPG) Piyasası Kanunu
    # Agriculture, food, animals
    "5996",  # Veteriner Hizmetleri, Bitki Sağlığı, Gıda ve Yem
    "5199",  # Hayvanları Koruma Kanunu
    "5488",  # Tarım Kanunu
    # Tourism, culture, sport, arms
    "2634",  # Turizmi Teşvik Kanunu
    "2863",  # Kültür ve Tabiat Varlıklarını Koruma
    "3289",  # Spor Genel Müdürlüğü
    "6136",  # Ateşli Silahlar ve Bıçaklar
    # Finance and telecoms
    "5411",  # Bankacılık Kanunu
    "6362",  # Sermaye Piyasası Kanunu
    "5684",  # Sigortacılık Kanunu
    "5809",  # Elektronik Haberleşme Kanunu
    "6112",  # Radyo ve Televizyonların Kuruluşu
})

# Terms that mark text as electricity / energy-market regulation. Deliberately
# specific: "piyasa" or "lisans" alone appear across Turkish regulation
# generally, so the list leans on vocabulary that does not occur outside this
# sector (uzlaştırma, YEKDEM, önlisans, kWh, TEİAŞ) plus the unmistakable
# "elektrik"/"enerji" stems.
_ELECTRICITY_TERMS = (
    "elektrik", "elektrik enerjisi", "elektrik piyasası", "enerji piyasası",
    "enerji bakanlığı", "enerji ve tabii kaynaklar",
    "epdk", "teiaş", "tedaş", "eüaş", "tetaş", "epiaş", "epiaş",
    "dağıtım şirketi", "dağıtım bölgesi", "dağıtım lisansı", "dağıtım sistemi",
    "dağıtım bedeli", "dağıtım faaliyeti", "dağıtım tesisi",
    "iletim sistemi", "iletim şirketi", "iletim faaliyeti", "iletim tarifesi",
    "üretim tesisi", "üretim lisansı", "üretim şirketi", "üretim faaliyeti",
    "önlisans", "ön lisans", "lisanssız üretim", "lisanssız elektrik",
    "tedarik lisansı", "tedarikçi", "görevli tedarik", "son kaynak tedarik",
    "perakende satış", "serbest tüketici", "abone", "abonelik",
    "kwh", "mwh", "gwh", "megavat", "kilovat", "mw", "kv",
    "gerilim", "trafo", "transformatör", "şebeke", "sayaç", "osos",
    "dengeleme", "uzlaştırma", "gün öncesi piyasası", "dengeleme güç piyasası",
    "yekdem", "yek-g", "yenilenebilir enerji", "rüzgâr", "rüzgar",
    "güneş enerjisi", "hidroelektrik", "jeotermal", "biyokütle",
    "kojenerasyon", "santral", "santrali", "enterkonneksiyon",
    "kapasite mekanizması", "piyasa işletmecisi", "yan hizmet",
    "sistem kullanım", "bağlantı anlaşması", "elektrik tüketimi",
    "elektrik üretimi", "elektrik tarifesi", "kayıp kaçak", "puant",
)

# Terms that mark text as belonging to a clearly different body of law. Every
# entry here names a subject an EPDK electricity question can never be about.
#
# Deliberately EXCLUDED from this list, though they are non-electricity codes:
# imar, kamulaştırma, orman, mera, çevre, tapu, ihale. Electricity law
# genuinely and constantly invokes those (a power plant needs expropriation,
# forest permits, zoning and an EIA), so treating them as off-domain markers
# would strip real electricity provisions. They are left unmarked; an article
# that is only about them and mentions no electricity term lands in AMBIGUOUS
# for review rather than being excluded automatically.
_OFF_DOMAIN_TERMS = (
    # Tax
    "vergi usul kanunu", "gelir vergisi", "kurumlar vergisi", "katma değer vergisi",
    "damga vergisi", "veraset ve intikal", "emlak vergisi", "motorlu taşıtlar vergisi",
    "vergi dairesi", "vergi levhası", "beyanname", "matrah", "mükellef",
    "vergi ziyaı", "tarhiyat", "gelir idaresi başkanlığı", "varlık barışı",
    # Civil service / public personnel
    "devlet memurları kanunu", "memuriyet", "kadro ihdas", "ek gösterge",
    "aylık ve özlük", "özlük hakları", "sözleşmeli personel", "kadro unvan",
    "657 sayılı", "375 sayılı kanun hükmünde kararname",
    # Social security / pensions
    "sosyal sigortalar", "sosyal güvenlik kurumu", "emekli aylığı", "prim borcu",
    "5510 sayılı", "genel sağlık sigortası",
    # Labour
    "kıdem tazminatı", "ihbar tazminatı", "iş sözleşmesi", "iş kanunu",
    "sendika", "toplu iş sözleşmesi", "grev", "lokavt", "asgari ücret",
    "işçi ve işveren", "4857 sayılı", "6356 sayılı",
    # Criminal / procedure / enforcement
    "türk ceza kanunu", "ceza muhakemesi", "adlî kontrol", "tutuklama",
    "infaz", "hükümlü", "cumhuriyet savcı", "ceza zamanaşımı",
    "icra ve iflas", "haciz", "hukuk muhakemeleri", "idari yargılama usulü",
    "5271 sayılı", "5237 sayılı", "2004 sayılı icra",
    # Traffic
    "karayolları trafik", "trafik cezası", "sürücü belgesi", "araç tescil",
    "trafik para cezası", "2918 sayılı",
    # Family / civil status
    "türk medeni kanunu", "velayet", "nafaka", "boşanma", "nüfus hizmetleri",
    "evlenme", "miras",
    # Education / health / military
    "yükseköğretim", "öğretim üyesi", "milli eğitim", "öğrenci affı",
    "sağlık bakanlığı", "hastane", "eczane", "ilaç fiyat",
    "askerlik", "türk silahlı kuvvetleri", "jandarma",
    # Agriculture / food
    "gıda tarım ve hayvancılık", "hayvancılık destek", "tarımsal destekleme",
    "bitki sağlığı", "veteriner",
    # Sport / culture / tourism / arms
    "spor federasyon", "futbol", "sporcu", "kültür ve turizm",
    "turizm işletmesi belgesi", "turizm yatırımı", "plaj işletme",
    "ateşli silah", "yivsiz tüfek", "tabanca",
    # Finance / telecoms unrelated to energy markets
    "bankacılık kanunu", "sermaye piyasası kurulu", "sigortacılık kanunu",
    "elektronik haberleşme", "evrensel hizmet", "hazine payı", "telsiz ücreti",
    # Mining and the other energy sectors. These are the neighbouring-domain
    # cases the Step 5 calibration flagged as unfixable by term statistics alone
    # (see config.FUSION_THRESHOLD's note on "doğal gaz"): the fix is keeping
    # their articles out of the index, not scoring them lower.
    "maden ruhsat", "ruhsat sahibi", "maden sahası", "işletme ruhsatı",
    "arama ruhsatı", "devlet hakkı", "ocak başı satış", "rödovans",
    "madencilik faaliyet", "maden kanunu",
    "doğal gaz piyasası", "petrol piyasası", "akaryakıt", "lpg",
    "rafineri", "boru hattı", "petrol arama", "sıvılaştırılmış petrol",
    "nükleer madde", "radyoaktif", "nükleer tesis",
)


@dataclass(frozen=True)
class ScopeVerdict:
    """One article's classification, with the evidence that produced it."""

    label: ScopeLabel
    electricity_hits: tuple[str, ...]
    off_domain_hits: tuple[str, ...]
    electricity_codes: tuple[str, ...] = ()
    off_domain_codes: tuple[str, ...] = ()

    @property
    def indexable(self) -> bool:
        """Whether this article should be embedded and made searchable.

        AMBIGUOUS counts as indexable: the asymmetry is deliberate (see the
        module docstring) -- an article this module cannot classify is kept and
        flagged for review, never dropped on a guess.
        """
        return self.label != "OFF_DOMAIN"

    @property
    def reason(self) -> str:
        """Human-readable justification, for the audit trail."""
        parts = [self.label]
        if self.electricity_codes:
            parts.append("amends " + "/".join(self.electricity_codes) + " (electricity)")
        if self.off_domain_codes:
            parts.append("amends " + "/".join(self.off_domain_codes) + " (off-domain)")
        parts.append("elec terms: " + (", ".join(self.electricity_hits[:5]) or "-"))
        parts.append("off terms: " + (", ".join(self.off_domain_hits[:5]) or "-"))
        return " | ".join(parts)


def _compile(terms: tuple[str, ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Anchor each term at a word start, but leave its end open.

    Turkish is agglutinative, so a bare substring test is wrong in both
    directions: "abone" must still match "aboneliğin" and "abonesine" (open
    end), while the abbreviations in these lists -- "kv", "mw", "lpg" -- must
    not fire inside an unrelated word (anchored start). Prefix-anchored
    matching is the rule that gets both right.
    """
    return tuple((t, re.compile(r"\b" + re.escape(t))) for t in terms)


_ELECTRICITY_PATTERNS = _compile(_ELECTRICITY_TERMS)
_OFF_DOMAIN_PATTERNS = _compile(_OFF_DOMAIN_TERMS)


def _hits(lowered: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> tuple[str, ...]:
    return tuple(term for term, pattern in patterns if pattern.search(lowered))


def cited_codes(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The electricity and off-domain law codes an article cites, in order of appearance."""
    seen: list[str] = []
    for match in _LAW_CODE_RE.finditer(text):
        code = match.group(1)
        if code not in seen:
            seen.append(code)
    return (
        tuple(c for c in seen if c in _ELECTRICITY_CODES),
        tuple(c for c in seen if c in _OFF_DOMAIN_CODES),
    )


def classify_text(text: str) -> ScopeVerdict:
    """Classify one article of an omnibus act: which body of law is it about?

    Two signals, in priority order.

    1. Which code the article amends. An omnibus article names its target
       ("6446 sayılı Kanunun 4 üncü maddesine...") and that target's subject is
       the article's subject. This is decisive when the article touches codes
       from only one side.

    2. Subject-matter vocabulary, by dominance rather than presence. Needed
       because chunking splits long articles, and a continuation fragment
       carries the substance without the amendment header that named the code.
       Dominance, not presence, because omnibus articles cross-reference
       constantly: an electricity provision may cite the Tax Procedure Law for
       how a levy is collected. Whichever vocabulary genuinely carries the text
       wins; a near-tie is not decided at all.

    Never call this on a single-subject document -- it is calibrated for the
    mixed-subject case only, and `is_omnibus_document()` is what decides that.
    """
    lowered = tr_lower(text)
    elec = _hits(lowered, _ELECTRICITY_PATTERNS)
    off = _hits(lowered, _OFF_DOMAIN_PATTERNS)
    elec_codes, off_codes = cited_codes(text)

    def verdict(label: ScopeLabel) -> ScopeVerdict:
        return ScopeVerdict(label, elec, off, elec_codes, off_codes)

    # -- signal 1: the amended code, when it points one way only ---------------
    if elec_codes and not off_codes:
        return verdict("ELECTRICITY")
    if off_codes and not elec_codes and not elec:
        # An off-domain target with no electricity vocabulary anywhere in the
        # article. The `not elec` guard matters: an omnibus act's electricity
        # article sometimes amends a tax or personnel code *for* the electricity
        # sector (an exemption for generation licensees, say), and that article
        # belongs in the index.
        return verdict("OFF_DOMAIN")

    # -- signal 2: vocabulary dominance ---------------------------------------
    if not elec and not off:
        # A whole article of an omnibus act carrying no electricity marker at
        # all -- not the word "elektrik", not a sector term, not an electricity
        # code. In a single-subject document this would be meaningless
        # (procedural boilerplate is subject-neutral), which is exactly why
        # is_omnibus_document() gates this classifier: here the base rate is
        # the other way round. These articles amend the Notaries Act, the Sugar
        # Act, civil aviation, public procurement. Absence of evidence IS
        # evidence when the document is a grab-bag by construction.
        return verdict("OFF_DOMAIN")
    if elec and not off:
        return verdict("ELECTRICITY")
    if off and not elec:
        return verdict("OFF_DOMAIN")
    if len(elec) >= 2 * len(off):
        return verdict("ELECTRICITY")
    if len(off) >= 2 * len(elec):
        return verdict("OFF_DOMAIN")
    return verdict("AMBIGUOUS")


# --------------------------------------------------------------------------
# Article-level classification -- the unit this is actually applied at
# --------------------------------------------------------------------------


def classify_chunks(items: Sequence[tuple[str | None, str]]) -> list[ScopeVerdict]:
    """Classify one omnibus document's chunks, deciding per ARTICLE not per chunk.

    `items` is (article_ref, text) in document order, exactly as chunk_document()
    emits them.

    Grouping matters because chunking splits a long article into several chunks
    (strategies `article-sub` and `article-window`), and only the first of those
    carries the amendment header that names the code being amended. A
    continuation fragment reads as subject-neutral on its own -- "Genel
    Müdürlükten alacağı şeklinde, üçüncü fıkrasında yer alan..." -- while the
    article it belongs to is unmistakably about mining or noteries. Classifying
    the article's full text once and applying that verdict to all of its chunks
    is both more accurate and the only way the result is auditable as "this
    article was excluded", which is what docs/decisions/ records.

    A chunk with no article_ref attaches to the article in progress -- but only
    if there IS one. On a document whose article headings were never detected
    (the `token-window` strategy: every chunk carries article_ref=None) that
    rule would swallow the entire document into a single group and classify
    hundreds of unrelated articles by one verdict, so there each chunk is
    judged on its own text instead.
    """
    groups: list[tuple[list[int], list[str]]] = []
    current_ref: str | None = None
    for position, (article_ref, text) in enumerate(items):
        continues = groups and current_ref is not None and article_ref in (None, current_ref)
        if continues:
            groups[-1][0].append(position)
            groups[-1][1].append(text)
        else:
            groups.append(([position], [text]))
        current_ref = article_ref

    verdicts: list[ScopeVerdict | None] = [None] * len(items)
    for positions, texts in groups:
        verdict = classify_text("\n".join(texts))
        for position in positions:
            verdicts[position] = verdict
    return [v for v in verdicts if v is not None]


_EXCLUDED_VERDICT = ScopeVerdict("OFF_DOMAIN", (), ())


def scope_chunks(
    title: str | None, items: Sequence[tuple[str | None, str]]
) -> tuple[DocumentScope, list[ScopeVerdict | None]]:
    """Decide a whole document, then every chunk in it. The one call ingestion makes.

    Returns the document's disposition alongside a per-chunk verdict list.
    A verdict of None means "never evaluated" -- the document is a single-subject
    electricity document, so its chunks were indexed without being classified at
    all, and the store records that as a NULL scope_label rather than pretending
    a judgement was made.
    """
    opening = "\n".join(text for _ref, text in items[:2])
    decision = document_scope(title, opening)

    if decision.excluded_entirely:
        return decision, [_EXCLUDED_VERDICT] * len(items)
    if decision.filters_articles:
        return decision, list(classify_chunks(items))
    return decision, [None] * len(items)
