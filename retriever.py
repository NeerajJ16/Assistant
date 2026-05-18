import json
import faiss
import numpy as np

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

# =====================================================
# CONFIG
# =====================================================

VECTOR_STORE_DIR = Path("vector_store")

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"

METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_RETRIEVE = 10

TOP_K_FINAL = 3

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

print("\nLoading embedding model...")

embed_model = SentenceTransformer(
    EMBED_MODEL
)

print("Embedding model loaded.")

# =====================================================
# LOAD RERANKER
# =====================================================

print("\nLoading reranker model...")

reranker = CrossEncoder(
    RERANK_MODEL
)

print("Reranker loaded.")

# =====================================================
# LOAD FAISS
# =====================================================

print("\nLoading FAISS index...")

index = faiss.read_index(
    str(FAISS_INDEX_PATH)
)

print(f"FAISS index loaded: {index.ntotal}")

# =====================================================
# LOAD METADATA
# =====================================================

with open(METADATA_PATH, "r", encoding="utf-8") as f:

    metadata = json.load(f)

print(f"Loaded {len(metadata)} metadata entries")

# =====================================================
# SEARCH + RERANK
# =====================================================

def search(query):

    print(f"\nQuery: {query}")

    # -------------------------------------------------
    # EMBED QUERY
    # -------------------------------------------------

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    )

    # -------------------------------------------------
    # NORMALIZE
    # -------------------------------------------------

    faiss.normalize_L2(query_embedding)

    # -------------------------------------------------
    # FAISS SEARCH
    # -------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        TOP_K_RETRIEVE
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx == -1:
            continue

        retrieved_chunks.append(
            metadata[idx]
        )

    # =================================================
    # RERANKING
    # =================================================

    pairs = []

    for chunk in retrieved_chunks:

        pair = (
            query,
            chunk["text"]
        )

        pairs.append(pair)

    rerank_scores = reranker.predict(pairs)

    # =================================================
    # COMBINE SCORES
    # =================================================

    reranked_results = []

    for chunk, score in zip(
        retrieved_chunks,
        rerank_scores
    ):

        reranked_results.append({
            "score": float(score),
            "title": chunk["title"],
            "chunk_type": chunk["chunk_type"],
            "text": chunk["text"],
            "metadata": chunk["metadata"]
        })

    # =================================================
    # SORT BY RERANK SCORE
    # =================================================

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x["score"],
        reverse=True
    )

    # =================================================
    # RETURN TOP RESULTS
    # =================================================

    return reranked_results[:TOP_K_FINAL]

# =====================================================
# CLI LOOP
# =====================================================

while True:

    query = input("\nAsk something: ")

    if query.lower() in ["exit", "quit"]:
        break

    results = search(query)

    print("\n===================================")
    print("RERANKED RESULTS")
    print("===================================")

    for i, result in enumerate(results, start=1):

        print(f"\n[{i}]")

        print(f"Rerank Score: {result['score']:.4f}")

        print(f"Title: {result['title']}")

        print(f"Type: {result['chunk_type']}")

        print("\nTEXT:")

        print(result["text"][:1000])