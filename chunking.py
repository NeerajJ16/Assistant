import json
import uuid
from pathlib import Path

# ======================================================
# LOAD SCRAPED DATA
# ======================================================

INPUT_FILE = Path(
    "scraped_output/portfolio_scraped.json"
)

OUTPUT_FILE = Path(
    "scraped_output/chunked_data.json"
)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ======================================================
# STORAGE
# ======================================================

chunks = []

# ======================================================
# HELPERS
# ======================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text.strip()

# ------------------------------------------------------

def make_chunk(
    chunk_type,
    title,
    text,
    metadata=None
):

    chunk = {
        "chunk_id": str(uuid.uuid4()),
        "chunk_type": chunk_type,
        "title": clean_text(title),
        "text": clean_text(text),
        "metadata": metadata or {}
    }

    chunks.append(chunk)

# ======================================================
# MAIN PAGE
# ======================================================

main = data.get("main", {})

# ======================================================
# HERO
# ======================================================

hero = main.get("hero", {})

hero_text = f"""
Section: About / Hero
Name: {hero.get('name', '')}
Tagline: {hero.get('tagline', '')}
Bio: {hero.get('bio', '')}
"""

make_chunk(
    chunk_type="hero",
    title="Hero Section",
    text=hero_text,
    metadata={
        "section": "hero"
    }
)

# ======================================================
# SKILLS
# ======================================================

skills = main.get("skills", [])

for skill in skills:

    text = f"""
Section: Skills
Category: {skill.get('category', '')}
Tools: {skill.get('tools', '')}
"""

    make_chunk(
        chunk_type="skill",
        title=skill.get("category", ""),
        text=text,
        metadata={
            "category": skill.get("category", "")
        }
    )

# ======================================================
# RESUME
# ======================================================

resume = main.get("resume", {})

# ------------------------------------------------------
# EXPERIENCE
# ------------------------------------------------------

for exp in resume.get("experience", []):

    text = f"""
Section: Work Experience
Role: {exp.get('company', '')}
Company: {exp.get('role', '')}
Date: {exp.get('date', '')}
Description: {exp.get('description', '')}
"""

    make_chunk(
        chunk_type="experience",
        title=f"{exp.get('company', '')} at {exp.get('role', '')}",
        text=text,
        metadata={
            "company": exp.get("role", ""),
            "role": exp.get("company", ""),
            "date": exp.get("date", "")
        }
    )

# ------------------------------------------------------
# EDUCATION
# ------------------------------------------------------

for edu in resume.get("education", []):

    text = f"""
Section: Education
Degree: {edu.get('role', '')}
Institution: {edu.get('company', '')}
Date: {edu.get('date', '')}
"""

    make_chunk(
        chunk_type="education",
        title=edu.get("role", ""),
        text=text,
        metadata={
            "institution": edu.get("company", "")
        }
    )

# ------------------------------------------------------
# CERTIFICATIONS
# ------------------------------------------------------

for cert in resume.get("certifications", []):

    make_chunk(
        chunk_type="certification",
        title=cert.get("text", ""),
        text=cert.get("text", ""),
        metadata={}
    )

# ======================================================
# RESEARCH
# ======================================================

researches = main.get("researches", [])

for research in researches:

    links = research.get("links", {})

    text = f"""
Section: Research & Initiatives
Title: {research.get('title', '')}
Description: {research.get('description', '')}
Links: {json.dumps(links)}
"""

    make_chunk(
        chunk_type="research",
        title=research.get("title", ""),
        text=text,
        metadata={
            "links": links
        }
    )

# ======================================================
# PROJECTS
# ======================================================

archives = data.get("archives", {})

projects = archives.get(
    "projects_table",
    []
)

for project in projects:

    stack = ", ".join(
        project.get("stack", [])
    )

    links = project.get("links", [])

    text = f"""
Section: Projects
Project Name: {project.get('name', '')}
Tech Stack: {stack}
Description: {project.get('description', '')}
Links: {json.dumps(links)}
"""

    make_chunk(
        chunk_type="project",
        title=project.get("name", ""),
        text=text,
        metadata={
            "stack": project.get("stack", []),
            "links": links
        }
    )

# ======================================================
# DASHBOARDS
# ======================================================

dashboards = archives.get(
    "dashboards_table",
    []
)

for dashboard in dashboards:

    text = f"""
Section: Dashboards
Dashboard Name: {dashboard.get('name', '')}
Stack: {', '.join(dashboard.get('stack', []))}
Description: {dashboard.get('description', '')}
"""

    make_chunk(
        chunk_type="dashboard",
        title=dashboard.get("name", ""),
        text=text,
        metadata={
            "stack": dashboard.get("stack", [])
        }
    )

# ======================================================
# EXTRA CURRICULARS
# ======================================================

extra = main.get(
    "extra_curriculars",
    []
)

for item in extra:

    text = f"""
Section: Extra Curricular Activities
Title: {item.get('title', '')}
Description: {item.get('description', '')}
"""

    make_chunk(
        chunk_type="extra_curricular",
        title=item.get("title", ""),
        text=text
    )

# ======================================================
# CONTACT
# ======================================================

contact = main.get("contact", {})

contact_text = f"""
Section: Contact Information
Name: {contact.get('name', '')}
Contact Links: {json.dumps(contact.get('links', {}))}
"""

make_chunk(
    chunk_type="contact",
    title="Contact Information",
    text=contact_text
)

# ======================================================
# SAVE OUTPUT
# ======================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\n===================================")
print("CHUNKING COMPLETE")
print(f"Total Chunks: {len(chunks)}")
print(f"Saved -> {OUTPUT_FILE}")
print("===================================")