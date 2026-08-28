# Design decisions

Consolidated from `config.py`'s comments and the commit history into one
narrative. Each decision links back to where the evidence for it lives, so
it can be re-verified rather than taken on faith.

## Why Foundry Local

The assistant serves EPDK electricity-market legislation for internal
KEPSAŞ use — regulatory text, potentially sensitive in aggregate, on
infrastructure that should work air-gapped. Foundry Local runs the chat and
embedding models entirely on local hardware behind an OpenAI-compatible
endpoint, so no document text or question ever leaves the machine, and the
rest of the pipeline (`src/llm.py`, `src/embed.py`) could be written against
the standard `openai` client library rather than a bespoke integration.

## Why article-boundary chunking

Turkish legal text is written and cited by article (MADDE): "MADDE 5"
answers a question in a way no arbitrary token window does, and a citation
that says "MADDE 5" is verifiable by a reader in a way "characters 4,200–4,700"
never is. `src/chunk.py` chunks on article boundaries first and only falls
back to sub-item splitting, then token windows, when an article is too
large or a document has no article structure at all — see the module
docstring's ordering of the four strategies.

## Why hybrid BM25 + dense retrieval, fused with RRF

Dense-only retrieval (the Step 4 baseline, kept callable as
`retrieve_dense()`) has two measured failure modes calibration could not fix
with a better cutoff:

1. **Domain-vocabulary mismatch.** "Doğal gaz dağıtım şirketlerinin abone
   bağlantı bedeli" retrieves the *electricity* Dağıtım Bağlantı Bedelleri
   Tebliği at cosine 0.6405 — every word matches except the one that
   matters semantically to a human, "doğal gaz" vs. "elektrik".
2. **Diacritic sensitivity.** ASCII-typed queries ("onlisans suresi" instead
   of "önlisans süresi") scored up to 0.21 lower on the same question and
   flipped 8 of 21 calibration decisions, 2 of them from a correct `ANSWER`
   to `NOT_FOUND`.

BM25 (`src/lexical.py`) fixes diacritic sensitivity outright by folding
diacritics symmetrically before indexing and querying, and gives the domain-
mismatch case a signal dense similarity structurally cannot have: a query
about natural gas scores low against an electricity-only corpus because
"doğal gaz" as a *phrase* doesn't dominate the electricity documents' term
statistics the way it should for a true natural-gas corpus. The two rankers'
ranked lists are combined by Reciprocal Rank Fusion (RRF), which uses only
rank position — not raw score — specifically because BM25's unbounded IDF
sums and dense cosine's `[-1, 1]` range have no stable fixed weighting that
would hold across queries. See `scripts/calibrate_gate.py` and
`scripts/eval_stemming.py` for the measurements (the latter also documents
*why* no stemming was added: it helped nothing and actively merged distinct
legal terms like "dağıtım bedeli" and "dağıtım bedelleri").

## Why raw RRF turned out to be ungateable, and what replaced it

The first hybrid calibration run thresholded the raw RRF score directly and
it failed outright: RRF is computed from rank positions alone, so a top-1
result scores ~0.0328 whenever both rankers agree and ~0.0164 when only one
does — almost regardless of whether the corpus actually contains the
answer. Measured over the calibration set, RRF separated answerable from
not-answerable questions *worse* than dense-only cosine had.

`scripts/eval_gate_signals.py` then compared six candidate confidence
signals on the same questions, scored by Youden's J:

| signal | Youden's J | fold-stable (typed vs. ASCII-folded) |
|---|---|---|
| dense_top1 (Step 4 baseline) | 0.500 | 16/21 |
| lexical_norm | 0.567 | 20/21 |
| coverage_topk | 0.700 | 18/21 |
| dense × lexnorm | 0.600 | 17/21 |
| mean(dense, coverage) | 0.767 | 18/21 |
| **dense_top1 × idf_coverage** | **0.767** | **19/21** |

`dense_top1 × idf_coverage` won on both separation and stability, and it has
an intuitive reading: a result must be both *semantically close* (dense) and
*actually contain the terms asked about, weighted by how rare and
discriminating they are* (IDF-weighted coverage). This is
`fusion_confidence()` in `src/retrieval.py`, and it is what `gate()`
thresholds — `config.FUSION_THRESHOLD` / `FUSION_FLOOR`, not the RRF score
that produced the ranking. RRF still decides *which* chunks come back;
`dense_top1 × idf_coverage` decides whether to trust them enough to answer.

## Why citations are assembled in code, never generated

The model is shown retrieved text with every citable identifier scrubbed
out — no document title, no article number, no Resmî Gazete reference, no
date, not just as omitted metadata but stripped from the chunk *body* too,
since mevzuat text is full of "MADDE 9 - (Değişik:RG-24/2/2017-29989)" and a
model that can read that can repeat it. The model answers using only
`(KAYNAK n)` labels; `src/answer.py` then maps whichever labels the model
actually used back to the real metadata row and builds the citation from
that. A citation assembled this way cannot be wrong about which provision it
names — it is a database lookup, not a generation. A citation *written* by a
4B model reading legal text can be wrong, and would be wrong in the most
expensive way available for a regulatory tool: a confident, well-formatted,
fabricated article reference. A label the model references that was never
supplied is counted as a hallucinated reference and dropped from the
citation list, not silently ignored — see `tests/test_answer.py`, which
asserts this invariant directly against the bytes sent to the server rather
than trusting a docstring.

## Why versioning is active/inactive, not full snapshot

`src/store.py` tracks document identity by `source_path`, and by
`file_sha256` (a content hash) within that path. When a document's content
changes — EPDK republishes an amended version under the same path — its old
chunks are marked `active = 0` and the new content's chunks are inserted as
`active = 1`; nothing is deleted. Retrieval only ever reads `active = 1`
rows, so a single query can never mix chunks from two versions of the same
regulation. This is deliberately *not* a full version history: there is no
queryable "what did this article say on a given past date" — only "what is
currently active" and "what used to be active, retained for audit." A full
temporal store was judged out of scope for what this tool needs to do
(answer questions against current law), and the active/inactive flag is the
minimum mechanism that prevents the one failure that actually matters: a
stale and a current provision being retrieved together and presented as if
both were in force.

## Why diacritic folding is kept separate from Turkish case-folding

`src/extract.py` implements `tr_lower()`/`tr_upper()` — *correct* Turkish
casing (`I` → `ı`, `İ` → `i`), used wherever real Turkish text is produced,
compared, or displayed. `src/lexical.py` implements a separate
`fold_diacritics()` that *destroys* the ı/i distinction on purpose, mapping
"önlisans" and "onlisans" to the same lexical key. These do two incompatible
jobs and are never composed: folded text is a matching key for BM25, never
Turkish, and must never reach a user or an embedding call. Composing them
would apply Turkish casing rules to text that has already stopped being
Turkish, which is meaningless, and risks corrupting genuine Turkish text
that later needs to be displayed. Keeping the functions in separate modules
with distinct names is the guardrail against ever calling the wrong one
where real text is in play.
