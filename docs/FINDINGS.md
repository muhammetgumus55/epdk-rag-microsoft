# Open findings

Things observed and characterized but **not acted on**. Each records what was
seen, how reliably it reproduces, and what would need to happen to close it, so
that a later decision starts from evidence rather than from scratch.

Closed decisions live in [`DECISIONS.md`](DECISIONS.md) and
[`decisions/`](decisions/); this file is only for what is still open.

---

## 1. qwen3-4b output degrades on a long-running Foundry server

**Status:** open, not acted on. Observed 2026-08-29.
**Affects:** generated answer text only. Retrieval, gate decisions and
confidence scores are unaffected.

### What was seen

After the Foundry Local server had served many requests in one session, the
chat model's output degenerated in two ways, sometimes together:

- **Chinese-character drift.** Turkish output with CJK spans spliced into it:

  ```
  Doğal gaz dağıtım益 şirket<公司><的><连接><问><是><怎><样 prove. (KAYNAK 1)
  Kapas nóng kapasite mekanizması ... sabit maliyet闪光 maliyet bileşenleri
  Serbest tüketiciiciler ... 闪光的使用场所的电力消费
  ```

- **Repetition loops.** The completion budget spent restating one clause:

  ```
  Önlisans süresi, mücbir sebep hâ闪光除外闪光闪光闪光闪光 n n n n n n n n n ...
  ```

It also hit the query-rewrite path, where the pipeline's own guard caught it and
fell back correctly:

```
rewrite was not a question, using question as typed
  | 'Peki bu hakkı kullanmazsa ne olur?' -> 'Bu闪光使用通过PYS aracılığıyla kullanır. ()'
```

### It is server-state dependent, not random

This is the part worth knowing. On a **freshly restarted** server the same
questions produce clean Turkish:

| | fresh server | long-running server |
|---|---|---|
| Doğal gaz / abone bağlantı bedeli | clean, 0 CJK chars | `şirket<公司><的><连接>...` |
| Yan hizmetler / primer frekans | clean, 0 CJK chars | clean |
| Kapasite mekanizması ödemeleri | repetition only | `Kapas nóng ... maliyet闪光` |
| Serbest tüketici | clean, 0 CJK chars | `闪光的使用场所的电力消费` |
| Önlisans süresi uzatılabilir mi | repetition only | CJK + `n n n n ...` loop |

Five sequential answers on a freshly restarted server: **zero** CJK characters.
The full DEMO.md walkthrough (nine generations) on a fresh server: **zero** CJK
characters.

The confidences are identical across both states — 0.50657, 0.58542, 0.60252 —
which is what rules retrieval out as the cause.

### Probable relationship to the known VRAM problem

`src/config.py`'s `CHAT_EFFECTIVE_CONTEXT` comment already documents that a
failed CUDA allocation fragments VRAM and is not cleaned up, so every request
after an OOM fails at sizes that work fine on a fresh server, and that
`foundry server restart` between measurements is mandatory. The degradation
here follows the same shape and appeared in sessions that had previously hit:

```
Failed to handle OpenAI embeddings: CUDA error in CudaMallocArray
at .../cuda_common.h:131 - an illegal memory access was encountered
```

That makes accumulated allocator state the leading hypothesis, but it is a
hypothesis — **not verified**. Degraded sampling after an illegal memory access
is plausible and consistent with what was seen, and no attempt was made to
confirm it against onnxruntime-genai internals.

Repetition is separately partly expected: `CHAT_FREQUENCY_PENALTY = 1.1` exists
precisely because greedy decoding on a 4B model fed long, partly duplicated
legal text falls into loops. The Kapasite and Önlisans answers show it still
happens on a clean server when several near-identical chunks are retrieved, so
the penalty reduces the problem without eliminating it.

### What would close it

1. Reproduce deliberately: fresh server, then N generations with a CJK-character
   and repetition-ratio counter per response, to find where degradation starts
   and whether an induced OOM triggers it immediately.
2. If accumulated allocator state is confirmed, the cheap mitigation is a
   restart policy (periodic, or on detecting a degenerate response) rather than
   a model change.
3. Independently: detect degenerate output before showing it — a CJK-character
   check and a repeated-sentence check on the completion would let the app
   suppress or regenerate rather than display it. That is a small, targeted
   change and does not depend on diagnosing the root cause.

### For now

`foundry server restart` before any demo or recorded measurement, and treat the
first clean run as the reference. This is stated at the top of
[`DEMO.md`](DEMO.md).

---

## 2. 31 AMBIGUOUS chunks awaiting a scope decision

**Status:** open, not acted on. Flagged 2026-08-29.
**Affects:** corpus scope. These chunks **are currently indexed and
retrievable.**

### What they are

`src/scope.py` classifies each article of an omnibus act as `ELECTRICITY`,
`OFF_DOMAIN` or `AMBIGUOUS`. The classifier deliberately refuses to guess when
the evidence is balanced, and `AMBIGUOUS` is treated as **keep and flag** —
wrongly excluding an electricity article makes a real question unanswerable, so
anything undecidable is retained and reported rather than dropped.

31 chunks currently hold that label. List them in full with:

```
python -m scripts.audit_omnibus --ambiguous
```

### Why they are genuinely undecidable

They straddle two bodies of law in one article. Representative cases (chunk ids
as of the 2026-08-29 reprocess — note that reprocessing reassigns ids, so
re-derive them from the audit script rather than trusting these forever):

| chunk | what it is |
|---|---|
| 27110/27111 | EÜAŞ coal procurement — amends 3096 (electricity) *and* 3213 (Maden Kanunu) |
| 27295, 27720 | YEK resource-area expropriation — amends 5346 (electricity) *and* 5490/5510 (nüfus, SGK) |
| 27324, 27749 | Doğal gaz **iletim tarifesi** — amends 4646, inside an electricity omnibus act |
| 27084–27086 | Mining permits in protected areas, mentioning jeotermal and santral |
| 27653 | Income-tax exemption for buildings selling surplus self-generated electricity to the son kaynak tedarik şirketi — amends 193 (Gelir Vergisi Kanunu) |

Chunk 27653 is the clearest illustration of why a rule cannot settle these: it
is an article of the Gelir Vergisi Kanunu, which is unambiguously out of
domain, that exists **only** to regulate an electricity-market transaction.

### What would close it

A human ruling per chunk, or per pattern. Two questions decide most of them:

1. Do provisions in other codes that exist *for* the electricity sector (tax
   exemptions for generators, expropriation for YEK areas) count as in scope?
   Answering yes keeps roughly half.
2. Does natural-gas content inside an electricity omnibus act follow the
   existing "gas is out of scope" line? Answering yes excludes 536/1921 and
   aligns them with the LPG and petroleum articles already excluded.

Both answers can be encoded in `src/scope.py` — the first as a rule, the second
by adding 4646 handling — so this does not need a per-chunk allowlist.

### Scale

31 chunks out of 26,441 indexable (0.12%). Small enough that leaving them in
place is not a material scope leak, which is why they were left indexed rather
than excluded pending review.
