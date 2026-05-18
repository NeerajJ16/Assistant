import json
import faiss
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# =========================================================
# CONFIG
# =========================================================

CHUNK_FILE = Path(
    "scraped_output/chunked_data.json"
)

VECTOR_STORE_DIR = Path(
    "vector_store"
)

VECTOR_STORE_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"

METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =========================================================
# LOAD CHUNKS
# =========================================================

print("\nLoading chunks...")

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")

# =========================================================
# PREPARE TEXTS
# =========================================================

texts = []

metadata = []

for chunk in chunks:

    text = chunk.get("text", "")

    title = chunk.get("title", "")

    chunk_type = chunk.get(
        "chunk_type",
        ""
    )

    chunk_metadata = chunk.get(
        "metadata",
        {}
    )

    # =====================================================
    # IMPORTANT IMPROVED EMBEDDING TEXT
    # =====================================================

    combined_text = f"""
    Chunk Type:
    {chunk_type}

    Title:
    {title}

    Content:
    {text}

    Metadata:
    {json.dumps(chunk_metadata)}
    """

    texts.append(combined_text)

    metadata.append({
        "chunk_id": chunk.get("chunk_id"),
        "title": title,
        "chunk_type": chunk_type,
        "text": text,
        "metadata": chunk_metadata
    })

print(f"\nPrepared {len(texts)} texts")

# =========================================================
# GENERATE EMBEDDINGS
# =========================================================

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embeddings generated.")

# =========================================================
# NORMALIZE
# =========================================================

faiss.normalize_L2(embeddings)

# =========================================================
# CREATE INDEX
# =========================================================

dimension = embeddings.shape[1]

print(f"\nEmbedding dimension: {dimension}")

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS index size: {index.ntotal}")

# =========================================================
# SAVE INDEX
# =========================================================

faiss.write_index(
    index,
    str(FAISS_INDEX_PATH)
)

print(f"\nSaved FAISS index -> {FAISS_INDEX_PATH}")

# =========================================================
# SAVE METADATA
# =========================================================

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"Saved metadata -> {METADATA_PATH}")

print("\n===================================")
print("EMBEDDING PIPELINE COMPLETE")
print("===================================")