import os
import json
import faiss

from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from groq import Groq

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env file"
    )

# =========================================================
# CONFIG
# =========================================================

VECTOR_STORE_DIR = Path("vector_store")

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"

METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL = "openai/gpt-oss-120b"

TOP_K_RETRIEVE = 10

TOP_K_FINAL = 3

# =========================================================
# LOAD MODELS
# =========================================================

print("\nLoading embedding model...")

embed_model = SentenceTransformer(
    EMBED_MODEL
)

print("Embedding model loaded.")

# ---------------------------------------------------------

print("\nLoading reranker model...")

reranker = CrossEncoder(
    RERANK_MODEL
)

print("Reranker loaded.")

# =========================================================
# LOAD FAISS
# =========================================================

print("\nLoading FAISS index...")

index = faiss.read_index(
    str(FAISS_INDEX_PATH)
)

print(f"FAISS index loaded: {index.ntotal}")

# =========================================================
# LOAD METADATA
# =========================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print(f"Loaded {len(metadata)} metadata entries")

# =========================================================
# GROQ CLIENT
# =========================================================

client = Groq(
    api_key=GROQ_API_KEY
)

# =========================================================
# PROMPT TEMPLATES
# =========================================================

REWRITE_PROMPT = """
Given the following conversation history and a follow-up question, 
rephrase the follow-up question to be a standalone question.

HISTORY:
{history}

FOLLOW-UP:
{question}

STANDALONE:
"""

PROMPT_TEMPLATE = """
You are an AI assistant representing Neeraj's
portfolio, resume, projects, and experience.

Your task is to answer interview questions ONLY
using the provided context.

STRICT RULES:
- Answer ONLY using the context
- Keep answers concise and professional
- Answer like a real interview candidate
- If the answer is unavailable in context say:
  "I could not find that information in the portfolio."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# =========================================================
# SESSION HISTORY
# =========================================================

chat_history = []

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def rewrite_query(query):
    if not chat_history:
        return query
    
    history_text = "\n".join([f"User: {c['user']}\nAssistant: {c['assistant']}" for c in chat_history[-3:]])
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(history=history_text, question=query)}],
        temperature=0,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

# =========================================================
# RETRIEVAL + RERANKING
# =========================================================

def retrieve_context(query):

    # -----------------------------------------------------
    # EMBED QUERY
    # -----------------------------------------------------

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    )

    # -----------------------------------------------------
    # NORMALIZE
    # -----------------------------------------------------

    faiss.normalize_L2(query_embedding)

    # -----------------------------------------------------
    # SEARCH FAISS
    # -----------------------------------------------------

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

    # =====================================================
    # RERANK
    # =====================================================

    pairs = []

    for chunk in retrieved_chunks:

        pairs.append(
            (query, chunk["text"])
        )

    rerank_scores = reranker.predict(pairs)

    reranked_results = []

    for chunk, score in zip(
        retrieved_chunks,
        rerank_scores
    ):

        reranked_results.append({
            "score": float(score),
            "title": chunk["title"],
            "chunk_type": chunk["chunk_type"],
            "text": chunk["text"]
        })

    # =====================================================
    # SORT
    # =====================================================

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x["score"],
        reverse=True
    )

    # =====================================================
    # TOP FINAL
    # =====================================================

    top_results = reranked_results[
        :TOP_K_FINAL
    ]

    return top_results

# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(results):

    context_parts = []

    for result in results:

        chunk = f"""
TITLE:
{result['title']}

TYPE:
{result['chunk_type']}

CONTENT:
{result['text']}
"""

        context_parts.append(chunk)

    return "\n\n".join(context_parts)

# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(query):

    # -----------------------------------------------------
    # RETRIEVE
    # -----------------------------------------------------

    results = retrieve_context(query)

    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    context = build_context(results)

    # -----------------------------------------------------
    # FINAL PROMPT
    # -----------------------------------------------------

    final_prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=query
    )

    # -----------------------------------------------------
    # GROQ CALL
    # -----------------------------------------------------

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a grounded RAG assistant."
                )
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0.2,
        max_tokens=500
    )

    answer = response.choices[0].message.content

    return answer, results

# =========================================================
# CLI CHAT LOOP
# =========================================================

print("\n===================================")
print("PORTFOLIO RAG CHATBOT READY")
print("Type 'exit' to quit")
print("===================================")

while True:

    query = input("\nAsk Interview Question: ")

    if query.lower() in ["exit", "quit"]:
        break

    try:
        # 1. Rewrite Query
        standalone_query = rewrite_query(query)

        # 2. Generate Answer
        answer, retrieved = generate_answer(standalone_query)

        # 3. Save History
        chat_history.append({"user": query, "assistant": answer})

        print("\n===================================")
        print("ANSWER")
        print("===================================\n")

        print(answer)

        print("\n===================================")
        print("RETRIEVED CONTEXT")
        print("===================================")

        for i, item in enumerate(
            retrieved,
            start=1
        ):

            print(f"\n[{i}]")

            print(
                f"Title: {item['title']}"
            )

            print(
                f"Type: {item['chunk_type']}"
            )

            print(
                f"Score: {item['score']:.4f}"
            )

    except Exception as e:

        print("\nERROR:")
        print(e)