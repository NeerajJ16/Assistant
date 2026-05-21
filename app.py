import os
import json
import faiss
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from groq import Groq

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Neeraj Portfolio Assistant",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f172a;
}

/* MAIN CONTAINER */
.main {
    padding-top: 1rem;
}

/* CHAT WRAPPER */
.chat-wrapper {
    max-width: 1200px;
    margin: auto;
    padding-bottom: 120px;
}

/* TITLE */
.chat-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 40px;
}

/* ROWS */
.user-row {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 22px;
}

.assistant-row {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 22px;
}

/* USER BUBBLE */
.user-bubble {
    background: linear-gradient(
        135deg,
        #2563eb,
        #1d4ed8
    );

    color: white;

    padding: 16px 20px;

    border-radius: 20px;

    max-width: 72%;

    font-size: 16px;

    line-height: 1.7;

    box-shadow:
        0 4px 14px rgba(0,0,0,0.25);

    word-wrap: break-word;
}

/* ASSISTANT BUBBLE */
.assistant-bubble {
    background: white;

    color: #111827;

    padding: 16px 20px;

    border-radius: 20px;

    max-width: 72%;

    font-size: 16px;

    line-height: 1.8;

    box-shadow:
        0 4px 14px rgba(0,0,0,0.12);

    word-wrap: break-word;
}

/* INPUT CONTAINER */
.stChatInputContainer {

    background: transparent !important;

    border: none !important;

    padding-bottom: 20px;
}

/* INPUT BOX */
[data-testid="stChatInput"] {

    max-width: 1200px;

    margin: auto;

    background: #111827 !important;

    border: 1px solid #374151 !important;

    border-radius: 14px !important;

    padding: 8px 12px !important;
}

/* TEXTAREA */
[data-testid="stChatInput"] textarea {

    background: transparent !important;

    color: white !important;

    border: none !important;

    font-size: 15px !important;

    min-height: 22px !important;

    padding-top: 8px !important;
}

/* PLACEHOLDER */
[data-testid="stChatInput"] textarea::placeholder {

    color: #9ca3af !important;
}

/* REMOVE OUTLINE */
textarea:focus,
button:focus {

    outline: none !important;

    box-shadow: none !important;
}

/* SEND BUTTON */
[data-testid="stChatInputSubmitButton"] {

    background: #2563eb !important;

    border: none !important;

    border-radius: 10px !important;

    width: 36px !important;

    height: 36px !important;
}

/* SEND ICON */
[data-testid="stChatInputSubmitButton"] svg {

    color: white !important;
}


/* SCROLLBAR */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env")
    st.stop()

# =========================================================
# CONFIG
# =========================================================

VECTOR_STORE_DIR = Path("vector_store")

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"

METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL = "openai/gpt-oss-20b"

TOP_K_RETRIEVE = 12

TOP_K_FINAL = 5


# =========================================================
# AUTO REBUILD INDEX IF DATA CHANGED
# =========================================================

import time
import subprocess

CHUNK_FILE = Path("scraped_output/chunked_data.json")
TIMESTAMP_FILE = Path("vector_store/last_built.txt")

def needs_rebuild():
    if not FAISS_INDEX_PATH.exists():
        return True
    if not TIMESTAMP_FILE.exists():
        return True
    last_built = float(TIMESTAMP_FILE.read_text())
    last_modified = os.path.getmtime(CHUNK_FILE)
    return last_modified > last_built

def rebuild_index():
    from sentence_transformers import SentenceTransformer
    import numpy as np

    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = []
    meta = []

    for chunk in chunks:
        text = chunk.get("text", "")
        title = chunk.get("title", "")
        chunk_type = chunk.get("chunk_type", "")
        chunk_metadata = chunk.get("metadata", {})

        combined = f"""
        Chunk Type: {chunk_type}
        Title: {title}
        Content: {text}
        Metadata: {json.dumps(chunk_metadata)}
        """

        texts.append(combined)
        meta.append({
            "chunk_id": chunk.get("chunk_id"),
            "title": title,
            "chunk_type": chunk_type,
            "text": text,
            "metadata": chunk_metadata
        })

    embeddings = model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(embeddings)

    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    faiss.write_index(new_index, str(FAISS_INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    TIMESTAMP_FILE.write_text(str(time.time()))

if needs_rebuild():
    with st.spinner("New data detected. Rebuilding index..."):
        rebuild_index()


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    embed_model = SentenceTransformer(
        EMBED_MODEL
    )

    reranker = CrossEncoder(
        RERANK_MODEL
    )

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    client = Groq(
        api_key=GROQ_API_KEY
    )

    return (
        embed_model,
        reranker,
        index,
        metadata,
        client
    )

(
    embed_model,
    reranker,
    index,
    metadata,
    client
) = load_models()

# =========================================================
# SESSION STATE
# =========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# REWRITE PROMPT
# =========================================================

REWRITE_PROMPT = """
You are a query rewriting assistant for a RAG system.

Given the conversation history and follow-up question,
rewrite the question into a standalone semantic search query.

Rules:
- Preserve original meaning
- Expand ambiguous references
- Keep technical terms
- Keep concise
- Do NOT answer the question

CHAT HISTORY:
{history}

FOLLOW-UP QUESTION:
{question}

STANDALONE SEARCH QUERY:
"""

# =========================================================
# FINAL PROMPT
# =========================================================

PROMPT_TEMPLATE = """
You are an AI assistant representing Neeraj's
portfolio, resume, projects, research, skills,
and professional experience.

STRICT RULES:
- Answer naturally like a real human.
- Use conversational interview style.
- Be detailed but concise.
- Never use bullet points.
- Speak in first person.
- Use only provided context.
- Never hallucinate.
- Always complete the response fully.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# =========================================================
# QUERY REWRITER
# =========================================================

def rewrite_query(query):

    if not st.session_state.chat_history:
        return query

    history_text = "\n".join([
        f"User: {c['user']}\nAssistant: {c['assistant']}"
        for c in st.session_state.chat_history[-3:]
    ])

    try:

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": REWRITE_PROMPT.format(
                        history=history_text,
                        question=query
                    )
                }
            ],
            temperature=0,
            max_tokens=100
        )

        rewritten = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        if not rewritten:
            return query

        return rewritten

    except Exception:
        return query

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_context(query):

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        TOP_K_RETRIEVE
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx == -1:
            continue

        chunk = metadata[idx]

        retrieved_chunks.append(chunk)

    # =====================================================
    # RERANK
    # =====================================================

    pairs = []

    for chunk in retrieved_chunks:

        rerank_text = f"""
        TITLE:
        {chunk['title']}

        TYPE:
        {chunk['chunk_type']}

        CONTENT:
        {chunk['text']}
        """

        pairs.append(
            (query, rerank_text)
        )

    rerank_scores = reranker.predict(
        pairs
    )

    results = []

    for chunk, score in zip(
        retrieved_chunks,
        rerank_scores
    ):

        results.append({
            "score": float(score),
            "title": chunk["title"],
            "chunk_type": chunk["chunk_type"],
            "text": chunk["text"],
            "metadata": chunk.get(
                "metadata",
                {}
            )
        })

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:TOP_K_FINAL]

# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(results):

    context_parts = []

    for result in results:

        metadata_text = json.dumps(
            result.get("metadata", {}),
            indent=2
        )

        text = f"""
TITLE:
{result['title']}

TYPE:
{result['chunk_type']}

CONTENT:
{result['text']}

METADATA:
{metadata_text}
"""

        context_parts.append(text)

    return "\n\n".join(context_parts)

# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(query):

    standalone_query = rewrite_query(query)

    retrieved = retrieve_context(
        standalone_query
    )

    context = build_context(
        retrieved
    )

    final_prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=query
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content":
                "You are a grounded RAG assistant."
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0.3,
        top_p=0.95,
        max_tokens=850,
        stream=True
    )

    return response

# =========================================================
# TITLE
# =========================================================



# =========================================================
# CHAT HISTORY
# =========================================================

st.markdown(
    '<div class="chat-wrapper">',
    unsafe_allow_html=True
)

for chat in st.session_state.chat_history:

    # USER MESSAGE
    st.markdown(
        f"""
        <div class="user-row">
            <div class="user-bubble">
                {chat["user"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ASSISTANT MESSAGE
    st.markdown(
        f"""
        <div class="assistant-row">
            <div class="assistant-bubble">
                {chat["assistant"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# INPUT
# =========================================================

query = st.chat_input(
    "Ask me anything about Neeraj..."
)

# =========================================================
# PROCESS QUERY
# =========================================================

if query:

    # USER MESSAGE DISPLAY
    st.markdown(
        f"""
        <div class="user-row">
            <div class="user-bubble">
                {query}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ASSISTANT PLACEHOLDER
    assistant_placeholder = st.empty()

    full_response = ""

    try:

        with st.spinner("Thinking..."):

            completion = generate_answer(
                query
            )

            token_buffer = ""

            for chunk in completion:

                try:

                    delta = (
                        chunk
                        .choices[0]
                        .delta
                        .content
                    )

                    if delta:

                        token_buffer += delta
                        full_response += delta

                        # BUFFERED RENDER
                        if len(token_buffer) > 20:

                            assistant_placeholder.markdown(
                                f"""
                                <div class="assistant-row">
                                    <div class="assistant-bubble">
                                        {full_response}▌
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            token_buffer = ""

                except Exception:
                    pass

        full_response = full_response.strip()

        # FINAL RENDER
        assistant_placeholder.markdown(
            f"""
            <div class="assistant-row">
                <div class="assistant-bubble">
                    {full_response}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        full_response = (
            f"Error generating response: {str(e)}"
        )

        assistant_placeholder.markdown(
            f"""
            <div class="assistant-row">
                <div class="assistant-bubble">
                    {full_response}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # SAVE CHAT
    # =====================================================

    st.session_state.chat_history.append({
        "user": query,
        "assistant": full_response
    })
