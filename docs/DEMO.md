# Demo script

Six questions for a live walkthrough, each with its expected outcome so a
failed run is diagnosable on the spot rather than mysterious. All six were
run once against the live app (Foundry Local `qwen3-4b-cuda-gpu` +
`qwen3-embedding-0.6b-cuda-gpu`, corpus of 27,047 active chunks) on
2026-08-28; actual outcomes are recorded below each question and matched
what's described.

Run each question either in the Streamlit UI (`streamlit run app.py`) or via
`python -m src.answer "<question>"` for a faster, UI-free check.

---

## 1. A direct question that answers well with a citation

**Ask:** `Serbest tüketici kimdir ve tedarikçisini değiştirme hakkını nasıl kullanır?`

**Expected:** Gate decision `ANSWER`, one clean paragraph, exactly one
`(KAYNAK 1)` citation resolving to a real article.

**Actual (recorded):** `ANSWER` (confidence 0.50645). Answer: "Serbest
tüketicinin, tedarikçisini değiştirmeye hakkinin, tedarikçisini değiştirmek
amacıyla, PYS üzerinden ilgili işlemi başlatması ve beş iş günü içinde
onaylanmasının şartıdır. (KAYNAK 1)" — citing *Elektrik Piyasası Dengeleme
ve Uzlaştırma Yönetmeliğinde Değişiklik Yapılmasına Dair Yönetmelik*.
Matches expectation.

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

**Actual (recorded):** Correctly detected as a follow-up and rewritten to
"Bu hakkı kullanmazsa ne olur?" (the log line
`retrieval query | original='Peki bu hakkı kullanmazsa ne olur?' | used='Bu hakkı kullanmazsa ne olur?' | follow_up=True`
confirms this). Gate decision: `NOT_FOUND` (confidence 0.09816) — the
rewrite stripped "Peki" but did not substitute in the actual referent
("tedarikçi değiştirme hakkı"), so the standalone query it produced was too
underspecified for retrieval to find anything.

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

---

## 3. A question that correctly triggers `NOT_FOUND`

**Ask:** `Deniz balıkçılığında av yasağı dönemleri hangi aylardır?`

**Expected:** `NOT_FOUND`, model never called, fixed refusal text shown
(never a generated "I don't know").

**Actual (recorded):** `NOT_FOUND` (confidence 0.07942, well under the
0.1871 floor). Fixed refusal message shown; `ÜRETİM: HAYIR - model
çağrılmadı` confirms the model was never invoked. Matches expectation.

---

## 4. A question that triggers `ANSWER_WEAK`

**Ask:** `Rafinerici lisansı sahiplerinin ulusal petrol stoku tutma yükümlülüğü nedir?`

**Expected:** Confidence between the floor (0.1871) and threshold (0.23963)
— generated, but flagged with the low-confidence warning shown in the UI.

**Actual (recorded):** `ANSWER_WEAK` (confidence 0.21310 — exactly the value
recorded in `config.py`'s comments above `FUSION_THRESHOLD`). The model
generated an incomplete, trailing-off answer — "Rafinerici lisansı
sahiplerinin ulusal petrol stoku tutma yükümlülüğü, (KAYNAK 1)" — citing the
*electricity* Lisans Yönetmeliği, because that's genuinely the closest match
in an electricity-only corpus to a petroleum-licensing question. This is a
good live illustration of exactly what the low-confidence warning exists to
flag: a technically-generated answer that a reader should not trust without
checking the source.

---

## 5. The "doğal gaz" question — a known limitation, not a failure

**Ask:** `Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli nasıl hesaplanır?`

**Expected:** Incorrectly gates to `ANSWER` (confidence above threshold)
despite being a natural-gas question against an electricity-only corpus.
**This is the documented cross-domain-confusion limitation** — see README
"Known limitations" and `config.py`'s comments above `FUSION_THRESHOLD` for
why BM25 cannot fix this particular case (the electricity corpus genuinely
contains the words "doğal" and "gaz").

**Actual (recorded):** `ANSWER` (confidence 0.50644, well above the 0.23963
threshold, **no low-confidence warning shown**). Generated answer: "Doğal
gaz dağıtım şirketlerinin abone bağlantı bedeli, tüketicilerin iç
tesisatının ve üreticilerin şalt sahasının dağıtım şebekesine bağlanması
için inşa edilen ve **[KAYNAK 1]**" — trails off mid-sentence, citing
*Elektrik Piyasası Tarifeler Yönetmeliği* Madde 6. Matches the documented
limitation exactly: presented with full confidence, no warning, citing an
electricity regulation for a gas question.

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

**Actual (recorded):** Tab A (`Yan hizmetler kapsamında primer frekans
kontrol hizmeti nasıl tedarik edilir?`) began generating immediately,
showing the staged progress indicator ("🔍 Aranıyor · ⚖️ Değerlendiriliyor ·
✍️ Yanıt üretiliyor..."). Tab B (`Kapasite mekanizması kapsamında yapılacak
ödemeler nasıl hesaplanır?`), submitted at essentially the same time,
immediately showed "⏳ Sırada bekliyor — başka bir soru işleniyor..." and
only began its own generation once Tab A's finished. Both ultimately
returned correct, cleanly cited answers (Tab A: *Elektrik Piyasası Yan
Hizmetler Yönetmeliği* Madde 16; Tab B: *Elektrik Piyasası Kapasite
Mekanizması Yönetmeliği* Madde 8). Matches expectation exactly.
