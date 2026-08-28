# epdk-rag-microsoft

An offline, self-hosted Turkish RAG assistant over EPDK (Enerji Piyasası
Düzenleme Kurumu) electricity-market legislation, built on **Microsoft
Foundry Local**. It runs entirely on local hardware — no data leaves the
machine — and was built for a KEPSAŞ unit's internal use: employees ask a
question in plain Turkish and get an answer grounded in the actual indexed
regulation text, with a citation to the exact article and document.

## Architecture

Documents are **extracted** from `.doc`/`.docx`/`.pdf` into clean Turkish
text, **chunked** on article (MADDE) boundaries so each chunk is one citable
provision, **embedded** and written to a **SQLite** store alongside a BM25
index. A question runs through **hybrid retrieval** (dense cosine similarity
fused with BM25 via Reciprocal Rank Fusion), a **confidence gate** decides
whether the retrieved evidence is strong enough to answer at all, Foundry
Local's local chat model **generates** the answer from the retrieved text
only, **citations are assembled in code** from the store's own metadata
(never written by the model), and everything is served through a
**Streamlit** chat UI. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for
the full component map and data flow, and [docs/DECISIONS.md](docs/DECISIONS.md)
for why each piece is built the way it is.

## Setup (clean Windows machine)

1. **Python 3.11.** Verify with `python --version`. Create and activate a
   virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Foundry Local.** Install it (via `winget install Microsoft.FoundryLocal`,
   or download it from Microsoft's Foundry Local page), then start its
   daemon:
   ```
   foundry server start
   ```
   The two models this project uses are resolved to hardware-specific
   variants (e.g. `qwen3-4b-cuda-gpu` on an NVIDIA GPU) and downloaded
   automatically the first time they're loaded — no separate download step is
   needed. Check `foundry status` to confirm the service is `Reachable`
   before running anything else. Everything in this repo needs Foundry Local
   running; nothing will work without it.

3. **LibreOffice**, needed only to convert legacy `.doc` files to `.docx`
   during extraction (`.pdf` and `.docx` need no external tool). Install via
   `winget install TheDocumentFoundation.LibreOffice`, or point
   `src/config.py`'s `LIBREOFFICE_PATH` at an existing install if
   auto-detection (`soffice` on PATH, or the default Program Files location)
   doesn't find it.

4. **Point `src/config.py` at the right models**, if not using the defaults.
   `CHAT_MODEL` and `EMBEDDING_MODEL` are Foundry Local aliases (currently
   `qwen3-4b` and `qwen3-embedding-0.6b`); everything downstream — chunk
   sizing, context budget, the embedding dimension the store schema uses —
   assumes these two, so changing either means re-reading the comments above
   `CHAT_EFFECTIVE_CONTEXT` and `EMBEDDING_DIM` in that file before swapping
   models. `CHAT_EFFECTIVE_CONTEXT` in particular is a *measured* VRAM
   ceiling for a specific 6 GB GPU running both models at once — re-measure
   it on different hardware rather than assuming it carries over (see
   `scripts/validate_tokenizer.py` and the calibration scripts for the
   pattern to follow).

5. Run the test suite to confirm the install: `pytest` (see
   [Final verification](#final-verification) below).

## Adding new documents / re-running ingest

Drop new files under `data/mevzuat/raw/` (any mix of `.doc`, `.docx`, `.pdf`
— other types are reported and skipped, not silently dropped) and re-run:

```
python -m src.store
```

This re-extracts the whole corpus and re-embeds only what's new. Identity is
tracked by **content hash** (`file_sha256`), not filename — EPDK's own
filenames are opaque download IDs, so they're never trusted for identity.
For a document whose content changed since the last ingest (same
`source_path`, different hash), the old chunks are marked `active = 0`
rather than deleted, and the new content's chunks are inserted as
`active = 1`. Retrieval only ever sees `active = 1` rows, so a query never
mixes text from two versions of the same regulation. Nothing is a hard
delete: the old version's rows stay in the database for audit, just excluded
from search. Re-running ingest on an unchanged corpus embeds nothing new —
each chunk is keyed by `(file_sha256, chunk_index)`, so already-active chunks
are skipped, and a crash mid-run resumes from wherever it stopped rather
than re-embedding from scratch.

## Running it

With Foundry Local's daemon running (`foundry status` shows `Reachable`):

```
python -m src.extract          # optional: extraction-only report (types, quality flags, titles)
python -m src.store             # full pipeline: extract -> chunk -> embed -> write to data/epdk.db
$env:EPDK_UI_PASSWORD = "choose-a-password"
streamlit run app.py
```

`EPDK_UI_PASSWORD` gates the whole UI behind a single shared password
(`hmac`-compared, never hardcoded) — set it as an environment variable
before launch, or in `.streamlit/secrets.toml` as `EPDK_UI_PASSWORD = "..."`
(gitignored; never commit it). Without it set, the login page tells the user
so explicitly rather than silently failing open.

For one-off CLI use without the UI:

```
python -m src.retrieval "<question>"        # retrieval + gate decision only, no generation
python -m src.answer "<question>" --then "<follow-up>"   # full pipeline, optionally multi-turn
```

## `src/config.py` reference

All tunables live in one file so nothing requires touching implementation
code to adjust. Every value below is measured or calibrated against the real
corpus and models — see the comments in `config.py` itself and
[docs/DECISIONS.md](docs/DECISIONS.md) for how each number was derived.

| Setting | Value | Controls |
|---|---|---|
| `CHUNK_SIZE` | 512 tokens | Target chunk size (article-boundary chunking overflows this when an article is naturally longer; see `CHUNK_OVERLAP`) |
| `CHUNK_OVERLAP` | 50 tokens | Overlap between adjacent windows, only in the token-window fallback paths |
| `TOKENS_PER_WORD` | 2.0 | Cheap token-count estimate used while chunking (deliberately over-estimates; see `MEASURED_TOKENS_PER_WORD`) |
| `MEASURED_TOKENS_PER_WORD` | 2.77 | The real ratio, measured against qwen3-embedding-0.6b's tokenizer over the whole corpus; recorded for reference, not consumed by chunking |
| `TOP_K` | 5 | Chunks returned per query after fusion |
| `SIMILARITY_THRESHOLD` / `SIMILARITY_FLOOR` | 0.606 / 0.53 | Superseded dense-only gate cutoffs, kept as the baseline `retrieve_dense()`/`gate_dense()` must beat |
| `BM25_K1` / `BM25_B` | 1.5 / 0.75 | Standard Robertson BM25 parameters (literature defaults; no evidence yet that tuning them beats tuning fusion) |
| `RRF_K` | 60 | Reciprocal Rank Fusion damping constant (Cormack et al.'s original value) |
| `FUSION_THRESHOLD` / `FUSION_FLOOR` | 0.23963 / 0.1871 | **The gate cutoffs actually used** — thresholds `dense_top1 × idf_coverage`, not raw RRF (see DECISIONS.md) |
| `FUSION_CANDIDATES` | 50 | How deep each ranker (dense, BM25) searches before fusion combines them |
| `CHAT_MODEL` / `EMBEDDING_MODEL` | `qwen3-4b` / `qwen3-embedding-0.6b` | Foundry Local model aliases |
| `CHAT_TEMPERATURE` | 0.0 | Pinned for reproducibility — a regulatory answer must not vary between identical runs |
| `CHAT_FREQUENCY_PENALTY` | 1.1 | Breaks a repetition loop this model falls into at temperature 0 on long, partly duplicated legal text |
| `CHAT_CONTEXT_WINDOW` | 40960 tokens | The model's *declared* context window (not what budgeting uses) |
| `CHAT_EFFECTIVE_CONTEXT` | 4000 tokens | The *measured* usable window on a 6 GB GPU running both models at once — budgeting against the declared window OOMs this hardware |
| `CHAT_MAX_COMPLETION_TOKENS` | 900 | Reserved for the answer, including Qwen3's always-present `<think>` block |
| `CONTEXT_SAFETY_MARGIN` | 96 tokens | Slack for chat-template overhead our own tokenizer count doesn't reproduce exactly |
| `SESSION_MAX_TURNS` | 3 | Prior exchanges kept in multi-turn history — a context-budget consequence, not a UX guess |
| `EMBEDDING_DIM` | 1024 | qwen3-embedding-0.6b's real dimension; the store schema depends on this exactly |
| `EMBED_BATCH_SIZE` / `EMBED_MAX_RETRIES` | 32 / 3 | Chunks per embedding call, and retries on transient failures |
| `DB_PATH` | `data/epdk.db` | SQLite chunk + vector store location |

## Known limitations

This is a regulatory tool, so these are stated plainly rather than buried.

- **Cross-domain confusion.** A question about a *neighbouring* energy
  domain that shares vocabulary with electricity law — natural gas, LPG,
  petroleum — can score high enough on dense similarity to incorrectly gate
  to `ANSWER`. Example, measured: "Doğal gaz dağıtım şirketlerinin abone
  bağlantı bedeli nasıl hesaplanır?" retrieves the *electricity* Dağıtım
  Bağlantı Bedelleri Tebliği at a fused confidence of 0.50644, well above the
  0.23963 threshold — see `config.py`'s comments above `FUSION_THRESHOLD` for
  the full measurement and demonstrated live in
  [docs/DEMO.md](docs/DEMO.md). BM25 can't fix this because the electricity
  corpus genuinely contains the words "doğal" and "gaz" (the Electricity
  Market Law references natural gas), so lexical coverage stays high.
  Fixing this needs a reranker or document-level domain filtering, both out
  of scope for this version.
- **Weaker reasoning on nuanced follow-ups.** The multi-turn mechanics
  (follow-up detection, query rewriting, history budgeting) are correct and
  tested; qwen3-4b's *reasoning* on a follow-up that depends heavily on
  unstated context is the ceiling, not the pipeline. A follow-up can be
  correctly classified and sent to the rewriter, and the rewriter can still
  under-specify the standalone query it produces, which then legitimately
  finds nothing. See `docs/DEMO.md` question 2 for a real example.
- **Effective context is ~4,000 tokens, not the declared 40,960.** The chat
  and embedding models are co-resident on a 6 GB GPU; asking the server for
  a prompt sized to the declared context window causes a CUDA OOM. This is a
  hardware ceiling (`CHAT_EFFECTIVE_CONTEXT`), not a model limit — raise it
  on a GPU with more VRAM, but re-measure rather than guess (see
  `config.py`'s comments and `scripts/validate_tokenizer.py`).
- **~11% of the source corpus is out of scope and not indexed** —
  spreadsheets (`.xlsx`, `.xls`) and archives (`.zip`) are reported by
  `python -m src.extract` and skipped, not silently dropped, but they carry
  no regulation prose an extractor here can use.
- **No full legal-version history.** The store tracks *active/inactive*
  chunks per content-hash change (see "Adding new documents" above), not a
  queryable diff between mevzuat versions over time. You can tell that a
  document changed and see its current version; you cannot ask "what did
  this article say on 2020-01-01?"

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — component map, one
  question's full data flow, and the store schema.
- [docs/DECISIONS.md](docs/DECISIONS.md) — why each non-obvious design
  choice was made, consolidated from the code comments and commit history.
- [docs/DEMO.md](docs/DEMO.md) — six demo questions with recorded actual
  outcomes, for a live walkthrough.

## Final verification

Full test suite, run from a clean checkout: **341 passed** (`pytest`, ~3s).
All six `docs/DEMO.md` questions were run once against the live app
(Foundry Local + Streamlit) with outcomes matching what's documented there.
