# Why "kıdem tazminatı" cannot be gated out by recalibration

**Date:** 2026-08-29
**Status:** known limitation, accepted and measured
**Reproduce:** `.venv\Scripts\python.exe scripts\calibrate_gate.py`
**Related:** [`2026-08-29-omnibus-scope-filter.md`](2026-08-29-omnibus-scope-filter.md)

## The claim

Of the three out-of-scope questions reported on 2026-08-29, two were caused by
the corpus containing text it should never have contained, and are fixed. The
third, **"Kıdem tazminatı nasıl hesaplanır?"**, is a different defect that
neither the corpus fix nor any choice of cutoff can resolve. It is recorded here
rather than papered over.

| Query | before fix | after fix + recalibration |
|---|---|---|
| Trafik cezası itiraz süresi nedir? | ANSWER (0.23971) | **NOT_FOUND** (0.15201) |
| Vergi levhası nereye asılır? | NOT_FOUND (0.10336) | **NOT_FOUND** (0.10845) |
| Kıdem tazminatı nasıl hesaplanır? | ANSWER (0.29203) | **ANSWER_WEAK** (0.31580) |

`ANSWER_WEAK` still generates an answer, with `low_confidence` set and a warning
shown to the user. It is a mitigation, not a fix.

## Why the corpus fix does not touch it

All five chunks retrieved for this question come from **legitimate
single-subject electricity documents**, not omnibus acts:

```
[1] ELEKTRİK PİYASASINDA DAĞITIM VE PERAKENDE SATIŞ ... KALİTE YÖNETMELİĞİ / MADDE 18
[2] ... KALİTE YÖNETMELİĞİNDE DEĞİŞİKLİK YAPILMASINA DAİR / MADDE 6
[3] ... KALİTE YÖNETMELİĞİ / MADDE 3
```

These articles define *kesinti tazminatı* — the compensation a distribution
company owes users for supply interruptions — and they literally contain the
sentence "ödenecek tazminat miktarı aşağıdaki formüle göre hesaplanır". The
corpus is answering "how is compensation calculated" with real, correctly
retrieved electricity law. Nothing here is out of scope.

The score in fact went **up** after the scope filter (0.29203 → 0.31580):
removing the omnibus chunks freed slots in the top-5 for more Kalite Yönetmeliği
articles, which are a better semantic match.

## Why no cutoff separates it

`fusion_confidence = dense_top1 × idf_coverage`. Both factors are legitimately
high, for reasons the signal was designed to reward:

```
dense_top1 = 0.57010
coverage   = 0.55394
confidence = 0.31580
```

The IDF-coverage breakdown shows exactly where the confidence comes from:

| term | df | idf weight | covered in top-5 |
|---|---:|---:|---|
| `kidem` | 13 | 7.5800 | **no** |
| `tazminati` | 38 | 6.5321 | yes |
| `hesaplanir` | 1482 | 2.8812 | yes |
| | | **16.9933 total** | **9.4133 covered → 0.554** |

Two of the three query terms are genuinely present in the corpus and genuinely
relevant. `kıdem` — the single word that carries the entire domain distinction —
is one term of three, and it is not even absent: it occurs in 13 indexed chunks
(6446 Elektrik Piyasası Kanunu, 4628 EPDK Teşkilat Kanunu, YEK mevzuatı), where
it appears incidentally in staff and transitional provisions. Because df > 0, it
cannot be weighted at max IDF, so the coverage penalty for missing it is
bounded. The signal is behaving exactly as specified; the specification cannot
express "one of these words changes what the question is about".

### The cost of forcing it

To gate this question to `NOT_FOUND`, `FUSION_FLOOR` would have to exceed
0.31580. Against the current answerable distribution that refuses **3 of 15
(20%) genuinely answerable questions outright**:

```
0.18804  İletim sistemi işletmecisinin arz güvenliğine ilişkin yükümlülükleri
0.23964  Dağıtım şirketinin tüketiciye planlı kesinti öncesinde bildirim yükümlülüğü
0.24807  Lisanssız elektrik üretiminde çatı tipi GES kurulu güç sınırı
```

These are core questions this assistant exists to answer. Trading three of them
for one labour-law question is not a trade worth making, so `FUSION_FLOOR` stays
anchored at the lowest answerable score (0.18804) as it always has been.

`FUSION_THRESHOLD` **was** raised, 0.23963 → 0.32979, which is what demotes
kıdem tazminatı from `ANSWER` to `ANSWER_WEAK`. That move is justified
independently by the Youden analysis on the enlarged negative set, and its cost
is three answerable questions moving `ANSWER` → `ANSWER_WEAK` — still answered,
flagged as low confidence.

## This is a known structural failure mode, not a new one

`config.py` already records the same shape of problem for natural gas:

> "Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli" is still ANSWER
> (0.50644). BM25 cannot demote it because "doğal" and "gaz" both genuinely
> occur in this electricity corpus [...] Distinguishing "a question ABOUT
> natural gas" from "an electricity rule that MENTIONS natural gas" needs
> document-level domain filtering or a reranker, not term statistics.

`kıdem tazminatı` is that failure mode again, with `tazminat` in the role of
`dağıtım/bağlantı/bedel`: every word matches except the one a human would use to
decide, and that word is not rare enough in this corpus to dominate the IDF sum.

## What would actually fix it

Not a cutoff. The options, none of which is in scope for this change:

1. **A query-side domain gate.** Check the *question* for out-of-domain
   vocabulary before retrieval and refuse, rather than judging the retrieved
   text. This is the only option that addresses the general class, and it would
   also fix the natural-gas case. It is a new component that needs its own
   calibration and its own false-positive analysis — refusing a legitimate
   question because it contains "tazminat" would be a worse failure than the one
   being fixed.
2. **A cross-encoder reranker** over the top-k, which can represent "this
   passage is about outage compensation, the question is about severance pay" in
   a way a bag-of-terms coverage score structurally cannot.
3. **Accept it.** `ANSWER_WEAK` plus the low-confidence warning is a reasonable
   product behaviour for a question the corpus genuinely has adjacent material
   for.

## Remaining out-of-domain leakage

The recalibration measures the whole class, not just this one question. Of 19
out-of-domain questions (excluding the two known energy-adjacent
domain-mismatch cases), **12 gate correctly to NOT_FOUND** and 7 reach
ANSWER_WEAK:

```
0.31580  Kıdem tazminatı nasıl hesaplanır?
0.28518  Toplu iş sözleşmesi en fazla kaç yıl süreyle yapılabilir?
0.26760  İşçinin yıllık ücretli izin süresi kaç gündür?
0.23986  Sürücü belgesi kaç yılda bir yenilenir?
0.22738  Konut kira sözleşmesinin feshi için ihtarname süresi ne kadardır?
0.20110  Katma değer vergisi beyannamesi hangi tarihe kadar verilir?
0.19375  LPG otogaz istasyonlarında sorumlu müdür bulundurma zorunluluğu nedir?
```

None reaches `ANSWER`. All are the same mechanism: a legal-procedural question
whose non-domain-specific words ("süre", "hesaplanır", "belirlenir", "itiraz",
"yenilenir") are common in regulatory Turkish, matched against an electricity
corpus that uses those words constantly. Every one of them needs option 1 or 2
above, not a cutoff.

The three questions in criminal and family law that share no vocabulary at all
(`kasten yaralama` 0.05841, `tutukluluk` 0.05641, `nafaka` 0.06514) gate
correctly and comfortably — which is the evidence that the mechanism is
vocabulary overlap, not a broken gate.
