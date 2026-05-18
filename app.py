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
    page_title="Neeraj AI Portfolio Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =========================================================
ROOT
========================================================= */

:root {
    --color-primary: #fd4520;
    --color-secondary: #f4f5f6;
    --color-tertiary: #0d1013;
    --color-gray: #f6f6f6;
    --background-color-1: linear-gradient(145deg, #1e2024, #23272b);
    --background-color-2: #212428;
    --shadow-1: 10px 10px 19px #1c1e22,
                -10px -10px 19px #262a2e;
    --shadow-2: inset 21px 21px 19px #181a1d,
                inset -21px -21px 19px #202225;
    --color-heading: #ffffff;
    --color-body: #878e99;
}

/* =========================================================
APP
========================================================= */

.stApp {
    background: #0d1013;
    color: white;
}

/* =========================================================
MAIN CONTAINER
========================================================= */

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background: #181a1d;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* =========================================================
TITLE
========================================================= */

.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    color: white;
    line-height: 1.2;
}

.highlight {
    color: #fd4520;
}

/* =========================================================
HERO CARD
========================================================= */

.hero-card {
    background: rgba(255,255,255,0.03);
    padding: 35px;
    border-radius: 30px;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 25px;
}

/* =========================================================
CHAT MESSAGES
========================================================= */

[data-testid="stChatMessage"] {
    background: #212428;
    border-radius: 20px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-1);
    border: 1px solid rgba(255,255,255,0.05);
}

/* USER MESSAGE */

[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(145deg, #f95230, #fb2c02);
}

/* =========================================================
CHAT INPUT
========================================================= */

.stChatInputContainer {
    background: #1e2024;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: var(--shadow-1);
}

/* =========================================================
BUTTONS
========================================================= */

.stButton button {
    background: linear-gradient(145deg, #f95230, #fb2c02);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton button:hover {
    transform: translateY(-2px);
}

/* =========================================================
METRIC CARDS
========================================================= */

[data-testid="metric-container"] {
    background: #212428;
    border-radius: 20px;
    padding: 15px;
    box-shadow: var(--shadow-1);
}

/* =========================================================
SCROLLBAR
========================================================= */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #fd4520;
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
    st.error("Missing GROQ_API_KEY")
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

TOP_K_RETRIEVE = 30

TOP_K_FINAL = 10

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
# PROMPT
# =========================================================

PROMPT_TEMPLATE = """
You are Neeraj's AI Portfolio Assistant.

Answer professionally and naturally.

Use ONLY the provided context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

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

    retrieved = []

    for idx in indices[0]:

        if idx == -1:
            continue

        retrieved.append(metadata[idx])

    return retrieved[:TOP_K_FINAL]

# =========================================================
# CONTEXT BUILDER
# =========================================================

def build_context(results):

    context_parts = []

    for result in results:

        text = f"""
TITLE:
{result['title']}

TYPE:
{result['chunk_type']}

CONTENT:
{result['text']}
"""

        context_parts.append(text)

    return "\n\n".join(context_parts)

# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(query):

    retrieved = retrieve_context(query)

    context = build_context(retrieved)

    final_prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=query
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a grounded portfolio assistant."
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0.2,
        max_tokens=500,
        stream=True
    )

    return response, retrieved

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    # 🤖 Neeraj AI Assistant

    AI-powered portfolio assistant built using:
    - FAISS
    - Sentence Transformers
    - Cross Encoder Reranking
    - Groq LLM
    - Streamlit
    """)

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero-card">

<div class="main-title">
Neeraj <span class="highlight">AI Portfolio Assistant</span>
</div>

<br>

<p style="
color:#878e99;
font-size:18px;
line-height:1.8;
">
Ask anything about projects, AI systems,
cloud engineering, research, certifications,
skills, and professional experience.
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric("Projects", "15+")

col2.metric("AI Stack", "RAG + LLM")

col3.metric("Experience", "3+ Years")

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# CHAT HISTORY
# =========================================================

for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.markdown(chat["user"])

    with st.chat_message("assistant"):
        st.markdown(chat["assistant"])

# =========================================================
# INPUT
# =========================================================

query = st.chat_input(
    "Ask interview question..."
)

# =========================================================
# CHAT FLOW
# =========================================================

if query:

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""

        with st.spinner("Thinking..."):

            completion, retrieved = generate_answer(
                query
            )

            for chunk in completion:

                delta = chunk.choices[0].delta.content

                if delta:

                    full_response += delta

                    response_placeholder.markdown(
                        full_response +
                        " <span style='color:#fd4520'>▌</span>",
                        unsafe_allow_html=True
                    )

        response_placeholder.markdown(
            full_response
        )

        # =================================================
        # SOURCES
        # =================================================

        with st.expander("📚 Retrieved Sources"):

            for item in retrieved:

                st.markdown(f"""
                <div style="
                    background:#1e2024;
                    padding:20px;
                    border-radius:20px;
                    margin-bottom:15px;
                    box-shadow: 10px 10px 19px #1c1e22,
                                -10px -10px 19px #262a2e;
                ">

                <h4 style="
                    color:#fd4520;
                    margin-bottom:10px;
                ">
                    {item['title']}
                </h4>

                <p style="
                    color:#878e99;
                    margin-bottom:10px;
                ">
                    {item['chunk_type']}
                </p>

                <p style="
                    color:white;
                    line-height:1.7;
                ">
                    {item['text'][:300]}...
                </p>

                </div>
                """, unsafe_allow_html=True)

    # =====================================================
    # SAVE CHAT
    # =====================================================

    st.session_state.chat_history.append({
        "user": query,
        "assistant": full_response
    })
