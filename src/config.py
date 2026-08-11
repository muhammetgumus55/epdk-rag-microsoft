# Central place for tunable parameters, so benchmarking never requires touching implementation code.

# Chunking
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50  # tokens

# Retrieval
TOP_K = 5
SIMILARITY_THRESHOLD = None  # calibrated in Step 5

# Models
EMBEDDING_MODEL = None  # decided in Step 1
LLM_MODEL = None  # decided in Step 1
