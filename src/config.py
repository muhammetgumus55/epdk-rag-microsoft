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
SIMILARITY_THRESHOLD = None  # calibrated in Step 5

# Models (Foundry Local, OpenAI-compatible endpoint)
# Aliases; the server resolves these to hardware variants (e.g. qwen3-4b-cuda-gpu).
CHAT_MODEL = "qwen3-4b"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

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
