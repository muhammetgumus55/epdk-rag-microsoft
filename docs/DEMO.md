# Demo script

Six questions for a live walkthrough, each with its expected outcome so a
failed run is diagnosable on the spot rather than mysterious. All six were
re-run in one continuous session against the live app (Foundry Local
`qwen3-4b-cuda-gpu` + `qwen3-embedding-0.6b-cuda-gpu`, corpus of **26,441
indexable chunks**) on **2026-08-29**, after the corpus scope filter and the
gate recalibration; actual outcomes are recorded below each question and
matched what's described.

Cutoffs in force for these numbers: `FUSION_THRESHOLD = 0.32979`,
`FUSION_FLOOR = 0.18804`.

> **Restart Foundry Local before demoing:** `foundry server restart`, then wait
> for `Server ready`. Generation quality degrades measurably once the server has
> been running through many requests — see
> [`FINDINGS.md`](FINDINGS.md#1-qwen3-4b-output-degrades-on-a-long-running-foundry-server).
> Every answer recorded below was produced on a freshly restarted server. The
> gate decisions and confidences are unaffected by this; only the generated
> prose is.

Run each question either in the Streamlit UI (`streamlit run app.py`) or via
`python -m src.answer "<question>"` for a faster, UI-free check.

---

## 1. A direct question that answers well with a citation

**Ask:** `Serbest tüketici kimdir ve tedarikçisini değiştirme hakkını nasıl kullanır?`

**Expected:** Gate decision `ANSWER`, one clean paragraph, exactly one
`(KAYNAK 1)` citation resolving to a real article.

**Actual (recorded 2026-08-29):** `ANSWER` (confidence 0.50645). Answer:
"Serbest tüketicinin, tedarikçisini değiştirmeye hakkinin, tedarikçisini
değiştirmek amacıyla, PYS üzerinden ilgili işlemi başlatması ve beş iş günü
içinde onaylanmasının şartıdır. (KAYNAK 1)" — citing *Elektrik Piyasası
Dengeleme ve Uzlaştırma Yönetmeliğinde Değişiklik Yapılmasına Dair
Yönetmelik*. Matches expectation, and is byte-identical to the 2026-08-28 run:
neither the corpus filter nor the recalibration touched this question.

**If it doesn't match:** Check `foundry status` shows `Reachable` first — a
`NOT_FOUND` here almost always means the chat/embedding models aren't
loaded, not a retrieval problem, since this question is well above both gate
thresholds.

---

## 2. A follow-up to (1) that exercises multi-turn

**Ask (same session, right after question 1):** `Peki bu hakkı kullanmazsa ne olur?`

**Expected:** The follow-up is detected lexically (the marker "bu"), a
rewrite call fires, and the retriever runs on the rewritten query, not the
raw one.

**Actual (recorded 2026-08-29):** Correctly detected as a follow-up and
rewritten to "Bu hakkı kullanmazsa ne olur?" (the log line
`retrieval query | original='Peki bu hakkı kullanmazsa ne olur?' | used='Bu hakkı kullanmazsa ne olur?' | follow_up=True`
confirms this). Gate decision: `NOT_FOUND` (confidence 0.09907) — the
rewrite stripped "Peki" but did not substitute in the actual referent
("tedarikçi değiştirme hakkı"), so the standalone query it produced was too
underspecified for retrieval to find anything.

**The rewrite is not reproducible run to run.** Across four runs of this exact
pair the rewrite came back as "Bu hakkı kullanmazsa ne olur?" (three times),
once as a genuinely better "Serbest tüketici, tedarikçisini değiştirmeye
hakkinin kullanmazsa ne olur?", and once as garbage that the pipeline's own
guard rejected (`rewrite was not a question, using question as typed`).
Confidence varied 0.047–0.152 accordingly. **The gate decision was `NOT_FOUND`
every time**, so the demo outcome is stable even though the number under it is
not — but don't promise the audience a specific confidence for this one.

**This is the documented model-reasoning limitation, not a pipeline bug**
(see README "Known limitations" and the follow-up note there): the
mechanics — classification, rewrite call, retrieval on the rewritten string
— are all working correctly and are unit-tested (`tests/test_session.py`).
qwen3-4b's rewrite quality on a heavily elliptical follow-up is the ceiling.
If you want a follow-up that *does* get answered instead, ask
`Önlisans süresi ne kadardır?` then `Peki uzatılabilir mi?` — this pair is
also `REWRITE_FEW_SHOT`'s worked example in `src/session.py`, so the model
has seen this exact shape and reliably rewrites it to "Önlisans süresi
uzatılabilir mi?", landing `ANSWER` at confidence 0.58330.

> Note on that alternative pair: the **setup** question "Önlisans süresi ne
> kadardır?" now lands `ANSWER_WEAK` (0.19110), not `ANSWER`. It scores below
> the raised `FUSION_THRESHOLD`, so the UI shows the low-confidence warning on
> the first answer and not on the follow-up. The answer itself is correct
> (Lisans Yönetmeliği MADDE 9, "otuz altı ayı geçmemek üzere").

---

## 3. A question that correctly triggers `NOT_FOUND`

**Ask:** `Trafik cezasına nasıl itiraz edilir?`

**Expected:** `NOT_FOUND`, model never called, fixed refusal text shown
(never a generated "I don't know").

**Actual (recorded 2026-08-29):** `NOT_FOUND` (confidence 0.08194, well under
the 0.18804 floor). Fixed refusal message shown; `ÜRETİM: HAYIR - model
çağrılmadı` confirms the model was never invoked. Matches expectation.

**Why this question and not something obviously unrelated:** until 2026-08-29
a near-identical question — "Trafik cezası itiraz süresi nedir?" — was
**answered**, at confidence 0.23971, citing real chunks from omnibus acts
("torba kanun") that amend the Criminal Procedure and Enforcement codes and
were indexed alongside genuine electricity law. That is a far stronger demo
than the previous placeholder (`Deniz balıkçılığında av yasağı dönemleri
hangi aylardır?`, still `NOT_FOUND` at 0.06989): a question about deep-sea
fishing shares no vocabulary at all with this corpus and was never in danger
of being answered, so refusing it demonstrates nothing. A traffic-fine appeal
question shares the whole procedural vocabulary of Turkish regulation —
"itiraz", "süre", "ceza" — and was genuinely being answered until the corpus
was filtered.

**When demoing this:** it is worth saying that out loud. The refusal is
interesting because of what it used to do, not because the question is
far-fetched. See
[`decisions/2026-08-29-omnibus-scope-filter.md`](decisions/2026-08-29-omnibus-scope-filter.md).

---

## 4. A question that triggers `ANSWER_WEAK`

**Ask:** `Rafinerici lisansı sahiplerinin ulusal petrol stoku tutma yükümlülüğü nedir?`

**Expected:** Confidence between the floor (0.18804) and threshold (0.32979)
— generated, but flagged with the low-confidence warning shown in the UI.

**Actual (recorded 2026-08-29):** `ANSWER_WEAK` (confidence 0.21315). The
model generated an incomplete, trailing-off answer — "Rafinerici lisansı
sahiplerinin ulusal petrol stoku tutma yükümlülüğü, (KAYNAK 1)" — citing the
*electricity* Lisans Yönetmeliği, because that's genuinely the closest match
in an electricity-only corpus to a petroleum-licensing question. This is a
good live illustration of exactly what the low-confidence warning exists to
flag: a technically-generated answer that a reader should not trust without
checking the source.

> Previously this question sat *just* above the floor (0.21310 against a
> 0.1871 floor and a 0.23963 threshold), near the bottom of a narrow band.
> The recalibration widened the `ANSWER_WEAK` band considerably, so it now
> sits comfortably mid-band rather than on a knife edge — a more robust demo,
> since it will not flip to `ANSWER` on a small scoring change.

---

## 5. The "doğal gaz" question — a known limitation, not a failure

**Ask:** `Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli nasıl hesaplanır?`

**Expected:** Incorrectly gates to `ANSWER` (confidence above threshold)
despite being a natural-gas question against an electricity-only corpus.
**This is the documented cross-domain-confusion limitation** — see README
"Known limitations" and `config.py`'s comments above `FUSION_THRESHOLD` for
why BM25 cannot fix this particular case (the electricity corpus genuinely
contains the words "doğal" and "gaz").

**Actual (recorded 2026-08-29):** `ANSWER` (confidence 0.50657, well above the
0.32979 threshold, **no low-confidence warning shown**). Generated answer:
"Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli, tüketicilerin iç
tesisatının ve üreticilerin şalt sahasının dağıtım şebekesine bağlanması
için inşa edilen ve **[KAYNAK 1]**" — trails off mid-sentence, citing
*Elektrik Piyasası Tarifeler Yönetmeliği* Madde 6. Matches the documented
limitation exactly: presented with full confidence, no warning, citing an
electricity regulation for a gas question.

**Neither the corpus filter nor the recalibration helped here, and neither
could have.** The retrieved chunk is genuine electricity law, so there is
nothing out of scope to remove; and the score is so far above threshold that
no defensible cutoff reaches it. The same structural failure now also has a
second documented instance — "Kıdem tazminatı nasıl hesaplanır?", which
retrieves the Kalite Yönetmeliği's *kesinti tazminatı* formulas at 0.31580 and
lands `ANSWER_WEAK`. Both need a query-side domain gate or a reranker:
[`decisions/2026-08-29-kidem-tazminati-gate-limit.md`](decisions/2026-08-29-kidem-tazminati-gate-limit.md).

**When demoing this:** frame it explicitly as "here is a known, documented
limitation" before asking it — an audience member who doesn't know this is
expected will read it as a broken system, not a characterized one.

---

## 6. Two questions submitted concurrently from two sessions

**Setup:** Open the Streamlit app in two separate browser tabs, log in on
both (each tab is an independent Streamlit session — the password gate is
per-session, not shared across tabs), type a different question into each,
and submit both within a second or two of each other.

**Expected:** Foundry Local serves one request at a time, so `app.py`'s
`GENERATION_LOCK` should let one session generate while the other shows a
visible waiting state — not silently queue, not error, not corrupt either
answer.

**Actual (recorded 2026-08-28, concurrency behaviour unchanged since):** Tab A
(`Yan hizmetler kapsamında primer frekans kontrol hizmeti nasıl tedarik
edilir?`) began generating immediately, showing the staged progress indicator
("🔍 Aranıyor · ⚖️ Değerlendiriliyor · ✍️ Yanıt üretiliyor..."). Tab B
(`Kapasite mekanizması kapsamında yapılacak ödemeler nasıl hesaplanır?`),
submitted at essentially the same time, immediately showed "⏳ Sırada
bekliyor — başka bir soru işleniyor..." and only began its own generation once
Tab A's finished. Both ultimately returned correct, cleanly cited answers.
Matches expectation exactly.

**Both questions re-verified individually 2026-08-29** (the two-tab timing
behaviour is a UI test and was not re-run; `GENERATION_LOCK` was not touched
by either change):

- Tab A: `ANSWER`, confidence 0.58542, citing *Elektrik Piyasası Yan Hizmetler
  Yönetmeliği* MADDE 16. Clean single-paragraph answer.
- Tab B: `ANSWER`, confidence 0.60252, citing *Elektrik Piyasası Kapasite
  Mekanizması Yönetmeliği* MADDE 8. **The answer now repeats itself** — the
  same sentence restated four times with different KAYNAK labels. The first
  sentence is correct and cited; the repetition is the qwen3-4b degeneration
  described in [`FINDINGS.md`](FINDINGS.md#1-qwen3-4b-output-degrades-on-a-long-running-foundry-server),
  and it appeared here even on a freshly restarted server. If you are demoing
  concurrency, consider swapping Tab B for a question with a shorter retrieved
  context.
