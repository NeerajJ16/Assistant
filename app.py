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
# QUERY REWRITE PROMPT
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

Your task is to answer interview questions
using the provided context.

STRICT STYLE RULES:
- Answer in a SINGLE, cohesive, and professional paragraph.
- DO NOT use bullet points, lists, or numbered sequences.
- Speak in the first person ("I am...", "I worked on...") as Neeraj's representative.
- Flow naturally from one point to the next using transitions.
- Provide COMPLETE and detailed information from the context.
- NEVER leave a sentence or information incomplete. Ensure the paragraph reaches a logical conclusion.
- ALWAYS include relevant links (e.g., GitHub, Demo) for projects if they exist in the context.
- Sound like a real person during an interview, not a robot or a list generator.

STRICT CONTENT RULES:
- Answer ONLY using context.
- Do NOT hallucinate or make up experience.
- When asked about experience or education, present them chronologically (newest first) within the paragraph.
- If information is unavailable say:
  "I don't have enough information about that in Neeraj's portfolio."

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

        rewritten = response.choices[0].message.content.strip()
        
        # Fallback if empty or failed
        if not rewritten or len(rewritten) < 3:
            return query
            
        return rewritten
    except Exception:
        return query

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_context(query):

    # =====================================================
    # EMBED QUERY
    # =====================================================

    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(query_embedding)

    # =====================================================
    # SEARCH
    # =====================================================

    scores, indices = index.search(
        query_embedding,
        TOP_K_RETRIEVE
    )

    query_lower = query.lower()

    retrieved_chunks = []

    preferred_types = []

    # =====================================================
    # QUERY INTENT DETECTION (Improved)
    # =====================================================
    
    # We check for presence of any of these keywords
    exp_keywords = ["experience", "work", "job", "career", "employment", "internship", "role"]
    edu_keywords = ["education", "degree", "college", "university", "study", "studied", "masters", "bachelors"]
    skill_keywords = ["skill", "skills", "technology", "tech stack", "tools", "languages", "databases"]
    proj_keywords = ["project", "projects", "portfolio", "built", "developed"]
    res_keywords = ["research", "paper", "publication", "initiative", "initiatives"]
    cert_keywords = ["certification", "certifications", "certificate"]

    if any(k in query_lower for k in exp_keywords):
        preferred_types.append("experience")

    if any(k in query_lower for k in edu_keywords):
        preferred_types.append("education")

    if any(k in query_lower for k in skill_keywords):
        preferred_types.append("skill")

    if any(k in query_lower for k in proj_keywords):
        preferred_types.append("project")

    if any(k in query_lower for k in res_keywords):
        preferred_types.append("research")

    if any(k in query_lower for k in cert_keywords):
        preferred_types.append("certification")

    # =====================================================
    # BUILD RETRIEVED CHUNKS
    # =====================================================

    for idx in indices[0]:

        if idx == -1:
            continue

        chunk = metadata[idx]

        boost = 0

        # =================================================
        # STRONG TYPE BOOST
        # =================================================

        if chunk["chunk_type"] in preferred_types:
            boost += 1000

        # =================================================
        # RECENCY BOOST
        # =================================================

        if (
            "recent" in query_lower or
            "latest" in query_lower or
            "current" in query_lower
        ):

            if chunk["chunk_type"] == "experience":

                boost += 500

                date_value = str(
                    chunk.get(
                        "metadata",
                        {}
                    ).get("date", "")
                )

                if "2025" in date_value:
                    boost += 300

                elif "2024" in date_value:
                    boost += 200

                elif "2023" in date_value:
                    boost += 100

        # =================================================
        # TITLE BOOST
        # =================================================

        title_lower = chunk["title"].lower()

        for word in query_lower.split():

            if word in title_lower:
                boost += 50

        chunk["_boost"] = boost

        retrieved_chunks.append(chunk)

    # =====================================================
    # BOOST SORT
    # =====================================================

    retrieved_chunks = sorted(
        retrieved_chunks,
        key=lambda x: x["_boost"],
        reverse=True
    )

    # =====================================================
    # RERANK INPUT
    # =====================================================

    pairs = []

    for chunk in retrieved_chunks:

        rerank_text = f"""
        Chunk Type:
        {chunk['chunk_type']}

        Title:
        {chunk['title']}

        Content:
        {chunk['text']}
        """

        pairs.append(
            (query, rerank_text)
        )

    # =====================================================
    # RERANK
    # =====================================================

    rerank_scores = reranker.predict(
        pairs
    )

    # =====================================================
    # COMBINE SCORES
    # =====================================================

    results = []

    for chunk, score in zip(
        retrieved_chunks,
        rerank_scores
    ):

        final_score = (
            float(score) +
            chunk["_boost"]
        )

        results.append({
            "score": final_score,
            "title": chunk["title"],
            "chunk_type": chunk["chunk_type"],
            "text": chunk["text"],
            "metadata": chunk.get(
                "metadata",
                {}
            )
        })

    # =====================================================
    # FINAL SORT
    # =====================================================

    def get_date_score(item):
        date_str = str(item.get("metadata", {}).get("date", "")).lower()
        if "present" in date_str:
            return 2026
        import re
        match = re.search(r"20\d{2}", date_str)
        if match:
            return int(match.group())
        return 0

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # If it's a general experience/education query, keep top results but sort 
    # them chronologically for the LLM.
    if any(t in preferred_types for t in ["experience", "education"]):
        top_n = results[:TOP_K_FINAL]
        results = sorted(top_n, key=get_date_score, reverse=True)
    else:
        results = results[:TOP_K_FINAL]

    # =====================================================
    # DEDUPLICATION
    # =====================================================

    unique_results = []

    seen_titles = set()

    for item in results:

        if item["title"] in seen_titles:
            continue

        seen_titles.add(item["title"])

        unique_results.append(item)

    return unique_results[:TOP_K_FINAL]

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
CHUNK TYPE:
{result['chunk_type']}

TITLE:
{result['title']}

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

    standalone_query = rewrite_query(
        query
    )

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
        max_tokens=850,
        stream=True
    )

    return response, retrieved

# =========================================================
# SIDEBAR
# =========================================================

    # with st.sidebar:

    #     st.title("👨‍💻 Neeraj AI Assistant")

    #     st.markdown("""
    # Ask about:
    # - Projects
    # - Experience
    # - Skills
    # - Education
    # - Research
    # - AI work
    # - Cloud technologies
    # - Certifications
    # """)

    #     st.divider()

    #     if st.button("🗑️ Clear Chat"):
    #         st.session_state.chat_history = []
    #         st.rerun()

# =========================================================
# MAIN UI
# =========================================================

#st.title("🤖 Neeraj Portfolio Assistant")

for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.markdown(chat["user"])

    with st.chat_message("assistant"):
        st.markdown(chat["assistant"])

query = st.chat_input(
    "Ask interview question..."
)

if query:

    # =====================================================
    # USER MESSAGE
    # =====================================================

    with st.chat_message("user"):
        st.markdown(query)

    # =====================================================
    # ASSISTANT
    # =====================================================

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
                        full_response + "▌"
                    )

        response_placeholder.markdown(
            full_response
        )

        # =================================================
        # SOURCES
        # =================================================


    # =====================================================
    # SAVE HISTORY
    # =====================================================

    st.session_state.chat_history.append({
        "user": query,
        "assistant": full_response
    })
