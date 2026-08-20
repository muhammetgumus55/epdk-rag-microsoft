# Central place for tunable parameters, so benchmarking never requires touching implementation code.

# Chunking
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50  # tokens

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
