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

import re



def make_links_clickable(text):

    url_pattern = r'((?:https?://|www\.)[^\s<]+[^<.,:;"\')\]\s])'

    def replace_link(match):

        url = match.group(0)

        href = url

        # add https if missing
        if url.startswith("www."):

            href = "https://" + url

        return f'''
        <a
            href="{href}"
            target="_blank"
            style="
                color:#60a5fa;
                text-decoration:none;
                font-weight:500;
            "
        >
            {url}
        </a>
        '''

    return re.sub(
        url_pattern,
        replace_link,
        text
    )

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
# VECTOR STORE SYSTEM
# =========================================================

import uuid
import hashlib
import numpy as np

RAW_FILE = Path(
    "scraped_output/portfolio_scraped.json"
)

CHUNK_FILE = Path(
    "scraped_output/chunked_data.json"
)

VECTOR_STORE_DIR = Path("vector_store")

FAISS_INDEX_PATH = (
    VECTOR_STORE_DIR / "faiss.index"
)

METADATA_PATH = (
    VECTOR_STORE_DIR / "metadata.json"
)

HASH_FILE = (
    VECTOR_STORE_DIR / "data.hash"
)

EMBED_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# TEXT CLEANER
# =========================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\n", " ")

    return " ".join(text.split()).strip()


# =========================================================
# HASH CHECK
# =========================================================

def get_file_hash(path):

    with open(path, "rb") as f:

        return hashlib.md5(
            f.read()
        ).hexdigest()


def needs_rebuild():

    if not RAW_FILE.exists():
        return False

    if not FAISS_INDEX_PATH.exists():
        return True

    if not METADATA_PATH.exists():
        return True

    if not HASH_FILE.exists():
        return True

    current_hash = get_file_hash(
        RAW_FILE
    )

    old_hash = HASH_FILE.read_text()

    return current_hash != old_hash


# =========================================================
# CHUNK CREATOR
# =========================================================

def add_chunk(
    chunks,
    chunk_type,
    title,
    content,
    metadata=None
):

    chunks.append({

        "chunk_id": str(uuid.uuid4()),

        "chunk_type": chunk_type,

        "title": clean_text(title),

        "text": clean_text(content),

        "metadata": metadata or {}
    })


# =========================================================
# CHUNKING
# =========================================================

def build_chunks():

    with open(
        RAW_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    chunks = []

    # =====================================================
    # HERO
    # =====================================================

    hero = data.get("hero", {})

    add_chunk(
        chunks,
        "hero",
        hero.get("name", ""),
        f"""
        Name:
        {hero.get('name', '')}

        Roles:
        {", ".join(hero.get('roles', []))}

        Tagline:
        {hero.get('tagline', '')}

        Email:
        {hero.get('email', '')}

        GitHub:
        {hero.get('github', '')}

        LinkedIn:
        {hero.get('linkedin', '')}
        """,
        hero
    )

    # =====================================================
    # SKILLS
    # =====================================================

    for skill in data.get("skills", []):

        add_chunk(
            chunks,
            "skill",
            skill.get("title", ""),
            f"""
            Skill Category:
            {skill.get('title', '')}

            Skills:
            {skill.get('items', '')}
            """,
            skill
        )

    # =====================================================
    # EXPERIENCE
    # =====================================================

    for exp in data.get("experience", []):

        add_chunk(
            chunks,
            "experience",
            f"{exp.get('role')} - {exp.get('company')}",
            f"""
            Role:
            {exp.get('role', '')}

            Company:
            {exp.get('company', '')}

            Location:
            {exp.get('location', '')}

            Duration:
            {exp.get('period', '')}

            Description:
            {exp.get('description', '')}
            """,
            exp
        )

    # =====================================================
    # EDUCATION
    # =====================================================

    for edu in data.get("education", []):

        add_chunk(
            chunks,
            "education",
            edu.get("degree", ""),
            f"""
            Degree:
            {edu.get('degree', '')}

            Institution:
            {edu.get('institution', '')}

            GPA:
            {edu.get('gpa', '')}

            Courses:
            {edu.get('courses', '')}
            """,
            edu
        )

    # =====================================================
    # RESEARCH
    # =====================================================

    for research in data.get("research", []):

        add_chunk(
            chunks,
            "research",
            research.get("title", ""),
            f"""
            Research:
            {research.get('title', '')}

            Description:
            {research.get('description', '')}

            Link:
            {research.get('link', '')}
            """,
            research
        )

    # =====================================================
    # PROJECTS
    # =====================================================

    for project in data.get(
        "archiveProjects",
        []
    ):

        add_chunk(
            chunks,
            "project",
            project.get("title", ""),
            f"""
            Project:
            {project.get('title', '')}

            Category:
            {project.get('category', '')}

            Stack:
            {project.get('stack', '')}

            Description:
            {project.get('description', '')}

            Demo:
            {project.get('demo', '')}

            GitHub:
            {project.get('github', '')}
            """,
            project
        )

    # =====================================================
    # DASHBOARDS
    # =====================================================

    for dashboard in data.get(
        "dashboards",
        []
    ):

        add_chunk(
            chunks,
            "dashboard",
            dashboard.get("title", ""),
            f"""
            Dashboard:
            {dashboard.get('title', '')}

            Stack:
            {dashboard.get('stack', '')}

            Description:
            {dashboard.get('description', '')}

            Overview:
            {dashboard.get('overview', '')}
            """,
            dashboard
        )

    # =====================================================
    # SAVE CHUNKS
    # =====================================================

    with open(
        CHUNK_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    return chunks


# =========================================================
# BUILD VECTOR STORE
# =========================================================

def rebuild_vector_store():

    st.info(
        "New portfolio data detected. Rebuilding embeddings..."
    )

    VECTOR_STORE_DIR.mkdir(
        exist_ok=True
    )

    chunks = build_chunks()

    model = SentenceTransformer(
        EMBED_MODEL
    )

    texts = []
    metadata = []

    for chunk in chunks:

        combined_text = f"""
        {chunk['title']}

        {chunk['text']}
        """

        texts.append(
            clean_text(combined_text)
        )

        metadata.append(chunk)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    faiss.normalize_L2(
        embeddings
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH)
    )

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

    HASH_FILE.write_text(
        get_file_hash(RAW_FILE)
    )

    st.cache_resource.clear()

    st.success(
        f"Successfully built {len(chunks)} chunks."
    )


# =========================================================
# AUTO REBUILD
# =========================================================

if needs_rebuild():

    rebuild_vector_store()



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
                                         {make_links_clickable(full_response)}▌
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
                    {make_links_clickable(full_response)}
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
                    {make_links_clickable(full_response)}
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
