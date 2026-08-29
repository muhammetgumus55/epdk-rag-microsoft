# Excluding out-of-scope text from the index

**Date:** 2026-08-29
**Status:** applied
**Code:** `src/scope.py`, `src/store.py`, `src/lexical.py`, `scripts/audit_omnibus.py`
**Tests:** `tests/test_scope.py`
**Reproduce the audit:** `python -m scripts.audit_omnibus`

## What went wrong

Three questions with no connection to electricity regulation were answered from
real, well-scoring chunks of the indexed corpus:

| Query | dense_top1 | idf_coverage | confidence | gate |
|---|---|---|---|---|
| Kıdem tazminatı nasıl hesaplanır? | 0.5222 | 0.5592 | 0.29203 | ANSWER |
| Trafik cezası itiraz süresi nedir? | 0.3904 | 0.6140 | 0.23971 | ANSWER |
| Vergi levhası nereye asılır? | 0.2792 | 0.3702 | 0.10336 | NOT_FOUND |

Tracing each retrieved chunk to its source document showed these were **two
different defects**, not one:

- `trafik cezası` and `vergi levhası` retrieved chunks whose text is İcra ve
  İflas Kanunu, Ceza Muhakemesi Kanunu, Sendikalar Kanunu and Vergi Usul Kanunu
  — indexed because they sit inside **omnibus acts** ("torba kanun") that EPDK
  publishes on its electricity pages. A genuine corpus-scope defect.
- `kıdem tazminatı` retrieved **only legitimate single-subject electricity
  documents** — Kalite Yönetmeliği MADDE 6/18/3, the outage-compensation
  (*kesinti tazminatı*) formulas. Nothing about the corpus is wrong there. It is
  a gate problem, and it is documented separately in
  [`2026-08-29-kidem-tazminati-gate-limit.md`](2026-08-29-kidem-tazminati-gate-limit.md).

## Why the corpus contains it at all

The corpus was downloaded from EPDK's electricity mevzuat pages, so every file
*arrived* under an electricity heading, and the filenames assert it: the Nuclear
Regulation Law is stored as `guncel-6446-sayl-elektrik-piyasas-kanunu-degisiklik-5.pdf`,
named for a law it has nothing to do with. Content was never checked against
filename.

A Turkish omnibus act amends dozens of unrelated codes in one instrument. One
file legitimately contains the article amending the Electricity Market Law and,
three articles later, one amending the Notaries Act. Indexing the file whole
indexes both.

## The audit

`scripts/audit_omnibus.py`, over the 27,047 active chunks:

```
Active documents                    553   (599 files on disk; 46 are content-
Active chunks                    27,047    duplicates deduped by file_sha256)
Omnibus documents                    23
Manually excluded documents           1

Chunks in omnibus documents         754   (2.8% of the corpus)
  ELECTRICITY (kept)                173
  OFF_DOMAIN  (excluded)            550
  AMBIGUOUS   (kept, flagged)        31

TOTAL NON-INDEXABLE                 606   (2.24% of the corpus)
  from omnibus article filtering    550
  from manual document exclusion     56
```

**The leak is small in share but structural in kind:** every one of the 606
chunks sits in 24 files out of 553, all under `Kanunlar/`. The other 529
documents contain no out-of-scope text at all.

### Per-document exclusions

| ELEC | OFF | AMB | Document title |
|---:|---:|---:|---|
| 2 | 89 | 0 | DEVLET MEMURLARI KANUNU İLE BAZI KANUNLARDA VE 375 SAYILI KHK'DE DEĞİŞİKLİK |
| 3 | 72 | 4 | BAZI VERGİ KANUNLARI İLE DİĞER BAZI KANUNLARDA DEĞİŞİKLİK |
| 6 | 53 | 1 | SANAYİNİN GELİŞTİRİLMESİ... BAZI KANUN VE KHK'LERDE DEĞİŞİKLİK |
| 2 | 48 | 6 | MADEN KANUNU İLE BAZI KANUNLARDA VE KHK'DE DEĞİŞİKLİK |
| 9 | 45 | 5 | VERGİ KANUNLARI İLE BAZI KANUN VE KHK'LERDE DEĞİŞİKLİK |
| 5 | 31 | 0 | DEVLET SU İŞLERİ... İLE BAZI KANUNLARDA... DEĞİŞİKLİK |
| 21 | 28 | 0 | BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN |
| 1 | 27 | 0 | VERGİ USUL KANUNU İLE BAZI KANUNLARDA DEĞİŞİKLİK |
| 37 | 22 | 1 | ELEKTRİK PİYASASI KANUNU İLE BAZI KANUNLARDA DEĞİŞİKLİK |
| 5 | 20 | 1 | BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN |
| 5 | 20 | 1 | BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN |
| 8 | 17 | 1 | ORGANİZE SANAYİ BÖLGELERİ KANUNU İLE BAZI KANUNLARDA DEĞİŞİKLİK |
| 8 | 17 | 1 | ORGANİZE SANAYİ BÖLGELERİ KANUNU İLE BAZI KANUNLARDA DEĞİŞİKLİK |
| 7 | 16 | 5 | BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN |
| 8 | 16 | 0 | ELEKTRİK PİYASASI KANUNU İLE BAZI KANUNLARDA VE 375 SAYILI KHK'DE DEĞİŞİKLİK |
| 8 | 11 | 0 | MADEN KANUNU İLE BAZI KANUNLARDA DEĞİŞİKLİK |
| 2 | 9 | 0 | BAZI KANUNLARDA DEĞİŞİKLİK YAPILMASINA DAİR KANUN |
| 1 | 3 | 0 | YARGI HİZMETLERİNİN ETKİNLEŞTİRİLMESİ AMACIYLA BAZI KANUNLARDA DEĞİŞİKLİK |
| 4 | 3 | 0 | BAZI KANUN VE KHK'LERDE DEĞİŞİKLİK YAPILMASINA DAİR KANUN |
| 1 | 2 | 0 | BÜTÇE KANUNLARINDA YER ALAN BAZI HÜKÜMLERİN... EKLENMESİNE DAİR KANUN |
| 1 | 1 | 0 | AMME ALACAKLARININ TAHSİL USULÜ HAKKINDA KANUN İLE BAZI KANUNLARDA DEĞİŞİKLİK |
| 14 | 0 | 3 | ELEKTRİK PİYASASI KANUNU İLE BAZI KANUNLARDA DEĞİŞİKLİK |
| 15 | 0 | 2 | ELEKTRİK PİYASASI KANUNU İLE BAZI KANUNLARDA DEĞİŞİKLİK |
| — | 56 | — | **NÜKLEER DÜZENLEME KANUNU (7381)** — manual exclusion, whole document |

Excluded subject matter includes: Noterlik Kanunu, Terörle Mücadele Kanunu,
Şeker Kanunu, sivil havacılık, Kamu İhale Kanunu, kadro ihdas cetvelleri, maden
ruhsatları, LPG/doğal gaz piyasası, turizm işletme belgeleri, ateşli silahlar.

## The classification logic

`src/scope.py` makes two decisions, in order.

### 1. Document disposition — `document_scope()`

- **EXCLUDED** — a hand-listed single-subject act that is simply not EPDK
  electricity-market regulation. Matched on the document's *title and opening
  text*, never its path: EPDK filenames are opaque and actively misleading, and
  the same act recurs under several of them as a content duplicate. Currently
  one entry, the Nuclear Regulation Law (7381) — nuclear safety is NDK's remit,
  not EPDK's. Adjacent-but-not-EPDK law is a recurring category, so this is a
  first-class list with a recorded reason per entry, not a special case.
- **OMNIBUS** — a torba kanun, detected by the plural, unnamed set of amended
  instruments its own name declares ("bazı kanunlarda", "diğer bazı kanunlarda").
  Keyed on *bazı* rather than *değişiklik*, so a single-instrument amendment
  ("Elektrik Piyasası Lisans Yönetmeliğinde Değişiklik...") is correctly not
  omnibus. Its articles are filtered individually.
- **IN_SCOPE** — everything else, 529 of 553 documents. Indexed whole, never
  filtered, never even classified; their `scope_label` is NULL, recording that
  no judgement was made rather than implying one.

### 2. Article classification — `classify_text()`

Applied **only** inside omnibus acts, per article rather than per chunk. Two
signals, in priority order:

1. **Which code the article amends.** An omnibus article names its target
   ("MADDE 7- 3213 sayılı Kanunun 16 ncı maddesi..."), and the target's subject
   is the article's subject. Decisive when the cited codes point one way only.
   `_ELECTRICITY_CODES` and `_OFF_DOMAIN_CODES` list them explicitly.
2. **Subject-vocabulary dominance**, for chunk fragments that lost the
   amendment header when a long article was split. Dominance, not presence:
   omnibus articles cross-reference constantly, so an electricity provision may
   cite the Tax Procedure Law for how a levy is collected.

Grouping is per-article: all chunks of one article share the verdict computed
from the article's full text. Documents whose article headings were never
detected (every chunk `article_ref = NULL`) are judged chunk by chunk instead —
without that guard, one verdict swallowed an entire 56-chunk document.

An article of an omnibus act with **no electricity marker at all** is classified
OFF_DOMAIN. In a single-subject document that would be meaningless, which is
exactly why this classifier is gated behind `document_scope()`: in a grab-bag
act, absence of evidence is evidence.

Codes deliberately **left out** of `_OFF_DOMAIN_CODES` despite not being
electricity codes, because electricity regulation genuinely and constantly
invokes them: 2942 Kamulaştırma, 4046 Özelleştirme, 6102 TTK, 4734/2886 ihale,
3194 İmar, 2872 Çevre, 6831 Orman, 4342 Mera, 7201 Tebligat, 4562 OSB. Listing
them would have stripped real electricity provisions.

## What "excluded" means mechanically

Excluded chunks are **extracted and stored in full** — the corpus keeps complete
provenance of what the source documents contain — but:

- `indexable = 0`
- `embedding IS NULL`, `embedded_at IS NULL` — never sent to the embedding model
- absent from `store.fetch_active_embeddings()` (the dense matrix)
- absent from `lexical.BM25Index.from_connection()` (the BM25 index)

Both rankers matter. The dense side excludes them for free since they have no
vector, but **BM25 indexes text, and the excluded rows still hold text** — the
lexical half is what matched the omnibus chunks behind the trafik/vergi failures
in the first place. `tests/test_scope.py::TestExcludedTextIsNotRetrievable`
asserts absence from both, not merely a low score.

This is why the fix is at ingestion rather than query time: there is nothing to
retrieve, not something that scores badly.

## Verification

`AMBIGUOUS` is treated as **keep and flag**, never dropped. Of 550 omnibus
chunks excluded, only 5 contain the word "elektrik" at all; all five were read
by hand and are correctly excluded (mining permits on olive groves, two LPG
Kanunu articles, an Ar-Ge personnel provision, a commercial-spam article). No
genuine electricity provision is excluded.

**31 AMBIGUOUS chunks remain indexed and await review** — genuinely straddling
cases such as EÜAŞ coal procurement, YEK resource-area expropriation, and doğal
gaz iletim tarifesi articles inside electricity torba acts. List them with
`python -m scripts.audit_omnibus --ambiguous`.

### Effect on the three failures

| Query | before | after |
|---|---|---|
| Trafik cezası itiraz süresi | **ANSWER** (0.23971) | **NOT_FOUND** (0.15201) |
| Vergi levhası nereye asılır | NOT_FOUND (0.10336) | NOT_FOUND (0.10845) |
| Kıdem tazminatı nasıl hesaplanır | **ANSWER** (0.29203) | **ANSWER** (0.31580) |

`trafik cezası` is fixed by the corpus filter alone: coverage fell from 0.614 to
0.389 once the omnibus chunks supplying "itiraz", "ceza" and "süre" were gone.
`kıdem tazminatı` is unchanged, and slightly *higher* — removing omnibus chunks
let more genuine Kalite Yönetmeliği chunks into the top-5. As diagnosed, the
corpus fix does not and cannot address it.

## Re-running ingestion

The hash-based resume path deliberately skips any document whose `file_sha256`
is unchanged, which makes a policy change invisible to it: the bytes on disk are
identical, only our judgement about them moved. `--reprocess-scope` handles this:

```
python -m src.store --reprocess-scope --dry-run   # list what would be rebuilt
python -m src.store --reprocess-scope             # rebuild just those documents
```

It selects documents using the same classifier that runs during ingest, so the
set cannot drift from a hand-maintained list, then **deletes** their stored
chunks before re-ingesting. Deletion rather than `active = 0` is required:
`UNIQUE(file_sha256, chunk_index)` ignores `active`, so re-inserting unchanged
content after retiring it would collide and be silently dropped by
`INSERT OR IGNORE`, leaving those documents with no active chunks at all.

The run of 2026-08-29 rebuilt 24 documents, embedded 204 chunks, stored 606
non-indexable, and took 26.3s. The other 26,237 chunks were verified untouched
by comparing a SHA-256 fingerprint over their `(id, source_path, chunk_index,
text, embedding)` before and after:
`c21f8f1b4076bc51dab27006d524fb9286f390a8f5fa4b4c31a6e25b112241a8`, identical.

## Schema change

`chunks` gained `indexable INTEGER NOT NULL DEFAULT 1` and `scope_label TEXT`,
and `embedding` / `embedded_at` were relaxed to NULLable. The first two are a
plain `ALTER TABLE`; relaxing NOT NULL is not expressible as an ALTER in SQLite,
so `store._migrate()` does the standard table rebuild inside one transaction.
Migration is automatic on `store.connect()` and takes ~2s on the 27k-chunk store.
