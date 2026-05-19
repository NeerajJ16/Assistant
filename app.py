import os
import json
import faiss
import streamlit as st

from datetime import date
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

# Current date injected into prompts
TODAY = date.today().strftime("%B %d, %Y")

# =========================================================
# DYNAMIC TOKEN CALCULATOR
# =========================================================

def estimate_max_tokens(query: str) -> int:
    """
    Return a token budget that scales with question complexity.

    Simple factual questions  → ~300 tokens
    Standard interview questions → ~600 tokens
    Broad / multi-part questions → ~1 200 tokens
    'Tell me everything' style   → ~2 000 tokens
    """
    q = query.lower()

    # Very broad / exhaustive questions
    broad_keywords = [
        "tell me everything", "full background", "complete profile",
        "walk me through your entire", "summarize your whole",
        "all your experience", "everything about", "full resume",
        "entire career", "describe everything"
    ]
    if any(k in q for k in broad_keywords):
        return 2000

    # Multi-part or detailed interview questions
    detailed_keywords = [
        "explain", "describe in detail", "how did you", "what challenges",
        "walk me through", "elaborate", "give me an overview of all",
        "tell me about all", "compare", "what is your experience with",
        "projects", "research", "education and experience",
        "skills and experience", "certifications and"
    ]
    if any(k in q for k in detailed_keywords):
        return 1200

    # Standard single-topic interview questions
    standard_keywords = [
        "what", "who", "where", "when", "how", "which",
        "tell me about", "describe", "experience", "skill",
        "project", "work", "role", "education", "degree",
        "certification", "strength", "weakness", "goal", "why"
    ]
    if any(k in q for k in standard_keywords):
        return 700

    # Short / factual fallback
    return 400

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    embed_model = SentenceTransformer(EMBED_MODEL)

    reranker = CrossEncoder(RERANK_MODEL)

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    client = Groq(api_key=GROQ_API_KEY)

    return embed_model, reranker, index, metadata, client

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
Today's date is {today}.

Given the conversation history and follow-up question,
rewrite the question into a standalone semantic search query.

Rules:
- Preserve original meaning
- Expand ambiguous references (e.g. "recent" means near {today})
- Keep technical terms intact
- Keep concise — one sentence max
- Do NOT answer the question

CHAT HISTORY:
{history}

FOLLOW-UP QUESTION:
{question}

STANDALONE SEARCH QUERY:
"""

# =========================================================
# FINAL ANSWER PROMPT
# =========================================================

PROMPT_TEMPLATE = """
You are an AI assistant representing Neeraj in a job interview.
Today's date is {today}.

Your role is to answer interview questions using ONLY the provided context,
speaking in first person as Neeraj's representative.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STYLE RULES (follow every single one):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Write in flowing, professional paragraphs — NO bullet points, NO numbered lists.
2. Use smooth transitions between ideas ("Additionally…", "Building on that…", "Most recently…").
3. Speak in first person: "I have…", "I built…", "I led…".
4. For experience / education: present chronologically, NEWEST FIRST, within the paragraph.
5. ALWAYS include GitHub or demo links for projects when they appear in the context.
6. Sound natural and confident — like a senior engineer in a real interview, not a resume bot.
7. Match answer length to question complexity:
   - Simple fact → 2–3 sentences
   - Standard question → 1–2 paragraphs
   - Broad / multi-part question → 3–4 full paragraphs
   NEVER truncate mid-sentence. ALWAYS finish the thought completely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTENT RULES (never break these):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Use ONLY information present in the CONTEXT below.
2. Do NOT hallucinate, invent, or extrapolate facts not in the context.
3. If the context does not contain the answer, say exactly:
   "I don't have enough information about that in Neeraj's portfolio."
4. When "current", "latest", or "recent" appears in the question,
   treat it relative to today ({today}) and highlight the most recent items first.
5. Cover ALL relevant chunks from the context — do not silently skip any.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON INTERVIEW QUESTION GUIDE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• "Tell me about yourself" → summarize education, experience, key projects, career goal
• "What are your strengths?" → draw from skills, impactful projects, measurable outcomes
• "Walk me through your experience" → all work experiences newest-to-oldest, with context
• "What projects have you worked on?" → cover ALL projects with tech stack + links
• "What is your tech stack / skills?" → list all skills grouped naturally in prose
• "Tell me about your education" → degrees, institutions, dates, relevant coursework
• "Where do you see yourself in 5 years?" → align with Neeraj's research / career direction
• "What are your certifications?" → all certifications with issuer and date
• "Why should we hire you?" → blend skills, experience, projects, and attitude

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTERVIEW QUESTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{question}

ANSWER (complete, never cut off mid-sentence):
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
                        today=TODAY,
                        history=history_text,
                        question=query
                    )
                }
            ],
            temperature=0,
            max_tokens=120
        )

        rewritten = response.choices[0].message.content.strip()

        if not rewritten or len(rewritten) < 3:
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

    scores, indices = index.search(query_embedding, TOP_K_RETRIEVE)

    query_lower = query.lower()

    retrieved_chunks = []
    preferred_types = []

    # ── Intent Detection ───────────────────────────────────
    exp_keywords  = ["experience", "work", "job", "career", "employment", "internship", "role"]
    edu_keywords  = ["education", "degree", "college", "university", "study", "studied", "masters", "bachelors"]
    skill_keywords= ["skill", "skills", "technology", "tech stack", "tools", "languages", "databases"]
    proj_keywords = ["project", "projects", "portfolio", "built", "developed"]
    res_keywords  = ["research", "paper", "publication", "initiative", "initiatives"]
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

    # "Tell me about yourself" → pull everything
    general_keywords = ["tell me about yourself", "introduce yourself", "background", "who are you"]
    if any(k in query_lower for k in general_keywords):
        preferred_types = ["experience", "education", "skill", "project", "certification", "research"]

    # ── Build chunks with boost ────────────────────────────
    for idx in indices[0]:

        if idx == -1:
            continue

        chunk = metadata[idx]
        boost = 0

        if chunk["chunk_type"] in preferred_types:
            boost += 1000

        # Recency boost
        if any(k in query_lower for k in ["recent", "latest", "current", "now"]):

            if chunk["chunk_type"] == "experience":
                boost += 500
                date_value = str(chunk.get("metadata", {}).get("date", ""))
                if "2025" in date_value or "present" in date_value.lower():
                    boost += 400
                elif "2024" in date_value:
                    boost += 200
                elif "2023" in date_value:
                    boost += 100

        # Title keyword match boost
        title_lower = chunk["title"].lower()
        for word in query_lower.split():
            if len(word) > 3 and word in title_lower:
                boost += 50

        chunk["_boost"] = boost
        retrieved_chunks.append(chunk)

    # ── Sort by boost then rerank ──────────────────────────
    retrieved_chunks = sorted(
        retrieved_chunks,
        key=lambda x: x["_boost"],
        reverse=True
    )

    pairs = []
    for chunk in retrieved_chunks:
        rerank_text = f"""
Chunk Type: {chunk['chunk_type']}
Title: {chunk['title']}
Content: {chunk['text']}
"""
        pairs.append((query, rerank_text))

    rerank_scores = reranker.predict(pairs)

    results = []
    for chunk, score in zip(retrieved_chunks, rerank_scores):
        final_score = float(score) + chunk["_boost"]
        results.append({
            "score": final_score,
            "title": chunk["title"],
            "chunk_type": chunk["chunk_type"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {})
        })

    # ── Chronological sort for experience/education ────────
    import re

    def get_date_score(item):
        date_str = str(item.get("metadata", {}).get("date", "")).lower()
        if "present" in date_str:
            return 2026
        match = re.search(r"20\d{2}", date_str)
        if match:
            return int(match.group())
        return 0

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    if any(t in preferred_types for t in ["experience", "education"]):
        top_n = results[:TOP_K_FINAL]
        results = sorted(top_n, key=get_date_score, reverse=True)
    else:
        results = results[:TOP_K_FINAL]

    # ── Deduplication ──────────────────────────────────────
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
CHUNK TYPE: {result['chunk_type']}
TITLE: {result['title']}
CONTENT:
{result['text']}
METADATA:
{metadata_text}
"""
        context_parts.append(text)

    return "\n\n---\n\n".join(context_parts)

# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(query):

    standalone_query = rewrite_query(query)

    retrieved = retrieve_context(standalone_query)

    context = build_context(retrieved)

    final_prompt = PROMPT_TEMPLATE.format(
        today=TODAY,
        context=context,
        question=query
    )

    # Dynamic token budget based on question type
    max_tokens = estimate_max_tokens(query)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a grounded RAG assistant for a portfolio chatbot. "
                    f"Today's date is {TODAY}. "
                    f"Never truncate your answer mid-sentence. "
                    f"Always complete every thought fully before stopping."
                )
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0.25,
        max_tokens=max_tokens,
        stream=True
    )

    return response, retrieved

# =========================================================
# MAIN UI
# =========================================================

for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.markdown(chat["user"])

    with st.chat_message("assistant"):
        st.markdown(chat["assistant"])

query = st.chat_input("Ask your questions here...")

if query:

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):

            completion, retrieved = generate_answer(query)

            for chunk in completion:

                delta = chunk.choices[0].delta.content

                if delta:
                    full_response += delta
                    response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.chat_history.append({
        "user": query,
        "assistant": full_response
    })
