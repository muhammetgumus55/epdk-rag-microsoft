# Central place for tunable parameters, so benchmarking never requires touching implementation code.

# Chunking
CHUNK_SIZE = 512  # tokens
# ~10% of CHUNK_SIZE. Legal text is chunked on MADDE (article) boundaries, so
# adjacent chunks are usually semantically self-contained and overlap matters
# less than in prose. Overlap only applies to the token-window fallback paths
# (oversized articles, documents with no article structure), where it exists to
# stop a sentence spanning a cut from being lost to both neighbours. Larger
# values mostly duplicate text in the vector store for little retrieval gain.
CHUNK_OVERLAP = 50  # tokens

# Token counting is an ESTIMATE, not qwen3's real tokenizer -- counting exactly
# would mean loading the model's tokenizer for every chunk. Turkish is
# agglutinative, so a BPE tokenizer emits well over one token per word; this
# ratio is calibrated to over-estimate slightly, which makes chunks a little
# smaller than the target rather than overflowing the embedding window.
# Verify against the real tokenizer before trusting CHUNK_SIZE as a hard limit.
TOKENS_PER_WORD = 2.0

# Extraction
# Resolved at runtime by src.extract; set explicitly to override autodetection.
LIBREOFFICE_PATH = None
# .doc -> .docx conversions issued per soffice invocation. One process per file
# is ~10x slower; very large batches risk a single bad file stalling the batch.
LIBREOFFICE_BATCH_SIZE = 20
LIBREOFFICE_TIMEOUT = 300  # seconds per batch

# Quality-check thresholds (see src.extract.quality_flags)
MIN_DOC_CHARS = 200  # below this a document is flagged near-empty
MIN_PAGE_CHARS = 30  # below this a PDF page is flagged near-empty / needs OCR
MAX_REPLACEMENT_RATIO = 0.001  # U+FFFD share of text before flagging
MIN_TURKISH_CHAR_COUNT = 1  # a Turkish regulation with 0 is almost certainly mis-decoded
MAX_GIBBERISH_RUN = 40  # longest run of non-space chars before flagging

# Measured directly against the real qwen3-embedding-0.6b tokenizer by
# scripts/validate_tokenizer.py (tokenizing all ~27k real corpus chunks), not
# estimated. NOT consumed by chunk.py -- TOKENS_PER_WORD above remains the one
# actual chunking parameter, deliberately left unchanged. Recorded here for
# future reference only: the validation found the estimate imprecise (chunks
# run larger than the word-count guess implies) but every chunk stays far
# under the model's 32768-token context window, so re-chunking would have been
# churn without benefit. See scripts/validate_tokenizer.py's report for detail.
# Result (2026-08-20, real qwen3-embedding-0.6b tokenizer, all 27,047 real
# corpus chunks): mean 2.77 real tokens/word vs the 2.0 estimate -- the
# estimate UNDER-shoots by ~28%. 41.1% of chunks land over the 512 CHUNK_SIZE
# target once measured for real, but only 1.14% exceed it by more than 50%,
# and the largest chunk in the whole corpus is 1,509 real tokens against a
# 32,768-token model context window (21x headroom). No re-chunking needed.
MEASURED_TOKENS_PER_WORD = 2.77

# Retrieval
TOP_K = 5

# Confidence gate cutoffs for src.retrieval.gate(), on top-1 cosine similarity:
#   score >= SIMILARITY_THRESHOLD -> ANSWER
#   score >= SIMILARITY_FLOOR     -> ANSWER_WEAK
#   below                         -> NOT_FOUND
#
# Measured, not guessed: scripts/calibrate_gate.py run against all 27,047 real
# corpus chunks (2026-08-20, qwen3-embedding-0.6b-cuda-gpu) with 15 questions
# answerable from the indexed mevzuat and 6 written to sound related but not be
# covered. Observed top-1 distributions:
#
#   answerable    (n=15): min 0.5361  p25 0.6069  median 0.6294  max 0.7468
#   not answerable (n=6): min 0.3875  p25 0.4977  median 0.5487  max 0.6405
#
# The two distributions OVERLAP -- 0.5361 (answerable) sits below 0.6405 (not
# answerable) -- so no single cutoff separates them and the two values do
# different jobs. THRESHOLD is the observed cutoff maximizing Youden's J
# (0.6069, J=0.633: 80% of answerable at or above it, 17% of not-answerable),
# rounded down to 0.606. FLOOR is anchored just under the lowest answerable
# score (0.5361 -> 0.53) so no genuinely answerable question is ever refused;
# it rejects the two clearly-unrelated questions (0.3875, 0.4977) outright and
# leaves the rest in ANSWER_WEAK.
#
# Known limit these numbers cannot fix: a question about a NEIGHBOURING energy
# domain (natural gas / LPG / petroleum) retrieves the electricity analogue at
# a genuinely high cosine -- "Doğal gaz dağıtım şirketlerinin abone bağlantı
# bedeli" hit the electricity Dağıtım Bağlantı Bedelleri Tebliği at 0.6405,
# above THRESHOLD. Dense similarity has no way to see that "doğal gaz" is the
# one word that matters. Lexical/BM25 fusion is the fix, not a higher cutoff.
#
# Second known limit, and the bigger one: these cutoffs assume the query is
# typed with Turkish diacritics. calibrate_gate.py's diacritics probe re-ran
# all 21 questions ASCII-folded ("önlisans süresi" -> "onlisans suresi", which
# is how people actually type) and 8 of 21 gate decisions changed, two of them
# from a correct ANSWER to NOT_FOUND. Mean score change -0.0376, worst -0.2062.
# The retrieved chunk is often still correct -- only the score collapses -- so
# this is a scoring problem, not a ranking one. Needs a real fix (query
# normalization, diacritic-restoring, or lexical fusion) before these cutoffs
# can be trusted against untreated user input.
#
# SUPERSEDED (Step 5): these two are the DENSE-ONLY cutoffs. They are no longer
# what gate() uses -- FUSION_THRESHOLD / FUSION_FLOOR below are. Kept because
# retrieve_dense() remains callable as the baseline hybrid must beat, and
# gate_dense() still scores against these.
SIMILARITY_THRESHOLD = 0.606
SIMILARITY_FLOOR = 0.53

# --------------------------------------------------------------------------
# Lexical (BM25) retrieval -- src/lexical.py
# --------------------------------------------------------------------------
# Standard Robertson BM25 parameters. k1 controls term-frequency saturation,
# b controls length normalization. Left at the literature defaults: the corpus
# is chunked to a fairly uniform size already (article-level), so there is
# little length variance for b to correct, and no evidence yet that tuning
# these beats tuning the fusion instead.
BM25_K1 = 1.5
BM25_B = 0.75

# --------------------------------------------------------------------------
# Fusion -- Reciprocal Rank Fusion of dense + BM25
# --------------------------------------------------------------------------
# RRF scores a document as sum over rankers of 1/(k + rank). k damps the
# influence of top ranks: small k lets a single ranker's #1 dominate, large k
# flattens toward a plain rank average. 60 is the value from the original
# Cormack et al. RRF paper and is what the calibration below was run with.
RRF_K = 60
# Confidence gate cutoffs -- what gate() actually uses. These REPLACE the
# dense-only SIMILARITY_* values above.
#
# They threshold retrieval.fusion_confidence() = dense_top1 x idf_coverage,
# NOT the RRF score. The first hybrid calibration run did threshold raw RRF and
# it failed outright: RRF is computed from rank positions alone, so top-1
# scores clustered at ~0.0328 (both rankers agree) or ~0.0164 (one does), and
# the classes separated WORSE than dense-only (answerable min 0.01639 vs
# not-answerable max 0.03252). RRF ranks well and measures confidence not at
# all. scripts/eval_gate_signals.py then compared six candidate signals on the
# same questions; dense_top1 x idf_coverage won on both separation and
# stability under ASCII folding:
#
#   signal              Youden's J   fold-stable
#   dense_top1              0.500        16/21     <- Step 4 baseline
#   lexical_norm            0.567        20/21
#   coverage_topk           0.700        18/21
#   dense_x_lexnorm         0.600        17/21
#   mean_dense_cov          0.767        18/21
#   dense_x_coverage        0.767        19/21     <- chosen
#
# Calibrated 2026-08-20 over all 27,047 chunks, same 15 answerable + 6
# not-answerable questions as Step 4, each run as typed and ASCII-folded:
#
#   answerable    (n=15): min 0.18711  median 0.41196  max 0.67547
#   not answerable (n=6): min 0.07942  median 0.18834  max 0.50644
#
# Classes still overlap, so as in Step 4 the two cutoffs do different jobs:
# THRESHOLD is the Youden-optimal observed cutoff (0.23963; 93% of answerable
# at or above it, 17% of not-answerable), FLOOR is anchored at the lowest
# answerable score so nothing answerable is ever refused.
#
# Measured effect vs the Step 4 dense-only gate:
#   gate accuracy, ASCII-folded input :  16/21 -> 18/21
#   gate accuracy, as typed           :  17/21 -> 17/21
#   decisions that flip under folding :      8 -> 4  (7 of Step 4's 8 now agree)
#
# STILL UNFIXED: "Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli" is
# still ANSWER (0.50644). BM25 cannot demote it because "doğal" and "gaz" both
# genuinely occur in this electricity corpus (the Electricity Market Law
# references natural gas), so IDF coverage stays high at 0.796. Distinguishing
# "a question ABOUT natural gas" from "an electricity rule that MENTIONS
# natural gas" needs document-level domain filtering or a reranker, not term
# statistics. The related "rafinerici / petrol stoku" question did improve
# sharply (0.5536 -> 0.21310, now just above the floor rather than mid-band).
#
# --------------------------------------------------------------------------
# RECALIBRATED 2026-08-29, after the corpus scope fix. Previous values were
# THRESHOLD 0.23963 / FLOOR 0.1871.
#
# Two things changed under these cutoffs, and only one of them is a tuning
# change:
#
#  1. The corpus itself. 606 chunks (2.24%) of out-of-scope text -- omnibus-act
#     articles amending tax, labour, criminal and traffic codes, plus the
#     Nuclear Regulation Law -- are no longer indexed at all. See
#     docs/decisions/2026-08-29-omnibus-scope-filter.md. Retrieval now runs over
#     26,441 chunks, not 27,047.
#  2. The calibration set. The not-answerable class went from 6 questions to 21.
#     The old six were almost all energy-ADJACENT (doğal gaz, LPG, akaryakıt,
#     petrol), which measured only the hardest case and left the easy case
#     unmeasured -- so nothing in the old numbers could have revealed that
#     questions about kıdem tazminatı or trafik cezası were being answered. The
#     new negatives span labour, tax, traffic, criminal, family and civil-service
#     law, and include the three real failures reported by a user.
#
# Measured over 26,441 chunks, 15 answerable + 21 not-answerable, each run as
# typed and ASCII-folded:
#
#   answerable    (n=15): min 0.18804  median 0.41094  max 0.67547
#   not answerable (n=21): min 0.05641  median 0.17950  max 0.50657
#
# Classes still overlap, so the two cutoffs keep doing their separate jobs.
# THRESHOLD is the Youden-optimal observed cutoff, which moved UP from 0.23963
# to 0.32979 (J improved 0.767 -> 0.752 on a set 71% larger and far harder; the
# old J was measured against six mostly-adjacent negatives and was optimistic).
# At 0.32979, 80% of answerable questions are at or above it against 5% of
# not-answerable. FLOOR stays anchored at the lowest answerable score, so no
# genuinely answerable question is ever refused.
#
# What the higher threshold costs, stated plainly: three answerable questions
# move from ANSWER to ANSWER_WEAK (arz güvenliği 0.18804, planlı kesinti
# bildirimi 0.23964, çatı tipi GES kurulu güç 0.24807). They are still answered,
# with the low-confidence flag set. That is the price of demoting kıdem
# tazminatı out of ANSWER, and it is worth paying; refusing them outright would
# not be, which is why FLOOR did not move up with THRESHOLD.
#
# STILL UNFIXED, and it cannot be fixed here: "Kıdem tazminatı nasıl
# hesaplanır?" scores 0.31580 and lands in ANSWER_WEAK, not NOT_FOUND. It
# retrieves genuine electricity law -- the Kalite Yönetmeliği's kesinti
# tazminatı formulas -- so the corpus filter does not touch it and no cutoff
# separates it: a FLOOR above 0.31580 would refuse 3 of 15 (20%) real
# answerable questions. See
# docs/decisions/2026-08-29-kidem-tazminati-gate-limit.md for the full argument.
FUSION_THRESHOLD = 0.32979
FUSION_FLOOR = 0.18804
# How deep each ranker goes before fusion. Fusion can only rescue a chunk that
# at least one ranker surfaced, so this is set well above TOP_K -- the whole
# point is for BM25 to promote something dense ranked 30th, and vice versa.
FUSION_CANDIDATES = 50

# Models (Foundry Local, OpenAI-compatible endpoint)
# Aliases; the server resolves these to hardware variants (e.g. qwen3-4b-cuda-gpu).
CHAT_MODEL = "qwen3-4b"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

# --------------------------------------------------------------------------
# Generation -- src/llm.py, src/answer.py
# --------------------------------------------------------------------------
# Regulatory answers must be reproducible: the same question must not produce a
# different answer on a second run. Sampling variance is a defect here, not a
# feature, so temperature is pinned at 0 and never exposed as a tunable.
CHAT_TEMPERATURE = 0.0

# Greedy decoding (temperature 0) on a 4B model fed long, partly duplicated legal
# text reliably falls into a repetition loop: an observed answer repeated
# "...tedarikçisini seçme hakkını kullanmayan serbest tüketici" for the entire
# 900-token completion budget. A frequency penalty breaks the loop and, unlike
# raising the temperature, is a deterministic logit transform -- identical
# questions still produce identical answers, so reproducibility is preserved.
# Verified honoured by this Foundry Local build (unlike chat_template_kwargs).
CHAT_FREQUENCY_PENALTY = 1.1

# qwen3-4b's DECLARED context window, read from its genai_config.json in
# Foundry Local's model cache (~/.foundry/cache/models/**/qwen3-4b-*/genai_config.json,
# "model.context_length"). Exact, not estimated -- Step 1 recorded the embedding
# dimension but never the chat window, so it was measured for Step 5.
#
# NOT the number budgeting uses. See CHAT_EFFECTIVE_CONTEXT.
CHAT_CONTEXT_WINDOW = 40960

# The window this machine can ACTUALLY serve, measured 2026-08-26. The declared
# 40,960 is unreachable here and budgeting against it crashes the server.
#
# Why: the RAG pipeline needs both models resident at once (embedding for the
# query, chat for generation). On this 6 GB RTX 4050 that is qwen3-4b (2.6 GB) +
# qwen3-embedding-0.6b (478 MB), leaving ~2.7 GB for the KV cache. Asking for a
# prompt the declared window permits makes onnxruntime-genai fail its KV
# allocation with a CUDA OOM, surfaced as an HTTP 500:
#
#   "Failed to handle OpenAI completion: CUDA error in CudaMallocArray
#    at .../cuda_common.h:131 - out of memory"
#
# The budget is on prompt + completion together, not the prompt alone: the KV
# cache grows with the whole sequence, so a longer answer costs the same
# resource a longer prompt does.
#
# Measured with both models loaded, prompt sizes counted with qwen3-4b's real
# tokenizer:
#
#   total tokens        result
#   ~1,951              OK   (VRAM immediately 5897 / 6141 MiB)
#   ~3,583              OK   one-shot
#   ~4,007 x6           OK   sustained over 6 sequential calls, VRAM flat at 5907 MiB
#   ~4,111              OK   one-shot
#   ~12,000             OOM  outright
#
# IMPORTANT measurement caveat, because it produced a wrong number the first
# time: a failed allocation fragments VRAM and is not cleaned up, so every
# request after an OOM fails at sizes that work fine on a fresh server. An early
# probe read ~3,873 as the ceiling purely because a previous request had already
# OOMed. `foundry server restart` between probes is required, and a ceiling
# measured without it will understate the real one.
#
# 4000 total is the largest configuration proven stable across repeated calls
# with a clean server. This is a hardware limit, not a model limit -- on a GPU
# with more VRAM, raise it toward CHAT_CONTEXT_WINDOW. Re-measure, do not guess.
CHAT_EFFECTIVE_CONTEXT = 4000

# Reserved for the answer itself. Qwen3 emits a <think> block before its answer
# even when thinking is suppressed with /no_think, and that block spends
# completion tokens, so this cannot be trimmed to just the visible answer.
#
# 400 was measurably too small: the model spent it restating the question and was
# cut off mid-sentence before citing any KAYNAK label, which loses the citation
# entirely. SYSTEM_PROMPT now forbids restating the question and caps the answer
# at four sentences, and this was raised alongside that.
CHAT_MAX_COMPLETION_TOKENS = 900

# Slack between our own tokenizer count and the server's. The server applies a
# chat template (role markers, turn delimiters) that our per-message count
# approximates rather than reproduces exactly; budgeting is only safe if it
# errs high. Measured overhead was well under this on every probe.
CONTEXT_SAFETY_MARGIN = 96

# Per-message framing cost in the chat template, added on top of the token count
# of each message's content when budgeting.
TOKENS_PER_MESSAGE = 4

# Multi-turn history -- src/session.py
# Only the last N question/answer exchanges are kept. Three is what the context
# budget above can afford alongside retrieved chunks; it is not a guess about
# what users need.
SESSION_MAX_TURNS = 3

# Budget for the follow-up classification + query rewrite call. The rewrite
# returns one short standalone question, so this stays small.
REWRITE_MAX_COMPLETION_TOKENS = 160

# Observed empirically in scripts/verify_foundry.py — not from documentation.
# The vector store schema depends on this exact value.
EMBEDDING_DIM = 1024

# Step 5 will benchmark EMBEDDING_MODEL against this alternative on real retrieval
# quality. Note it cannot co-reside with a chat model on 6GB VRAM (5.5GB alone),
# so benchmarking it requires load/unload cycling.
EMBEDDING_MODEL_ALT = "qwen3-embedding-8b"  # 4096-dim per docs; NOT yet verified locally

# Embedding / storage
# Chunks sent per Foundry Local embeddings.create() call. Large enough to
# amortize the request round trip over ~27k chunks, small enough that one
# batch failure (e.g. a transient GPU hiccup) only costs re-embedding this many.
EMBED_BATCH_SIZE = 32
EMBED_MAX_RETRIES = 3

DB_PATH = "data/epdk.db"
