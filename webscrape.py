"""
Portfolio Scraper — neerajj-portfolio.netlify.app
Selectors verified against live DOM diagnosis.
Outputs: JSON, Markdown, TXT
"""

import asyncio, json, re
from pathlib import Path
from playwright.async_api import async_playwright

BASE_URL  = "https://neerajj-portfolio.netlify.app"
OUTPUT_DIR = Path(__file__).parent / "scraped_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

async def scrape_main(page) -> dict:
    await page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
    # scroll full page to trigger all reveal animations
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await asyncio.sleep(2)
    await page.evaluate("window.scrollTo(0, 0)")
    await asyncio.sleep(2)

    # ── HERO ─────────────────────────────────────────────────────────────────
    name = clean(await page.locator("#heroName").inner_text())
    bio  = clean(await page.locator("#heroBio").inner_text())

    tagline = ""
    try:
        tagline = clean(await page.locator("#typingText").inner_text())
    except Exception:
        pass

    social_links = {}
    for a in await page.locator("#socialLinks a[href]").all():
        href  = await a.get_attribute("href") or ""
        aria  = await a.get_attribute("aria-label") or ""
        title = await a.get_attribute("title") or ""
        label = aria or title or href
        if href:
            social_links[label] = href

    hero = {"name": name, "bio": bio, "tagline": tagline, "social_links": social_links}

    # ── SKILLS ───────────────────────────────────────────────────────────────
    # #skillsGrid > .skill-card > h4 + p
    skills = []
    for card in await page.locator("#skillsGrid .skill-card").all():
        icon  = clean(await card.locator(".skill-icon-box").inner_text()) if await card.locator(".skill-icon-box").count() else ""
        title = clean(await card.locator("h4").inner_text())             if await card.locator("h4").count() else ""
        tools = clean(await card.locator("p").inner_text())              if await card.locator("p").count() else ""
        if title:
            skills.append({"icon": icon, "category": title, "tools": tools})

    # ── RESUME ───────────────────────────────────────────────────────────────
    resume = {}

    # Experience — always visible (active tab on load)
    resume["experience"] = await _scrape_timeline(page, "#expTimeline")

    # Education
    resume["education"] = await _scrape_timeline(page, "#eduTimeline")

    # Certifications — click tab 3 (index 2)
    await page.locator(".tab-btn").nth(2).click()
    await asyncio.sleep(0.8)
    certs = []
    for card in await page.locator("#certGrid .cert-card, #certGrid .card, #certGrid > div").all():
        txt  = clean(await card.inner_text())
        link = ""
        a_el = card.locator("a[href]")
        if await a_el.count():
            link = await a_el.first.get_attribute("href") or ""
        if txt:
            certs.append({"text": txt, "link": link})
    resume["certifications"] = certs

    # Leadership — click tab 4 (index 3)
    await page.locator(".tab-btn").nth(3).click()
    await asyncio.sleep(0.8)
    resume["leadership"] = await _scrape_timeline(page, "#leaderTimeline")

    # ── RESEARCHES ───────────────────────────────────────────────────────────
    # #researchGrid > .research-card > h4 + p
    await page.locator("#researchGrid").scroll_into_view_if_needed()
    await asyncio.sleep(1)
    researches = []
    for card in await page.locator("#researchGrid .research-card").all():
        title = clean(await card.locator("h4").inner_text()) if await card.locator("h4").count() else ""
        desc  = clean(await card.locator("p").inner_text())  if await card.locator("p").count() else ""
        links = {}
        for a in await card.locator("a[href]").all():
            href  = await a.get_attribute("href") or ""
            label = clean(await a.inner_text()) or await a.get_attribute("title") or href
            if href:
                links[label] = href
        researches.append({"title": title, "description": desc, "links": links})

    # ── PROJECTS (homepage) ───────────────────────────────────────────────────
    # #projectsGrid — scroll to trigger reveal, then scrape
    await page.locator("#projectsGrid").scroll_into_view_if_needed()
    await asyncio.sleep(1.5)
    projects_home = []
    for card in await page.locator("#projectsGrid .project-card, #projectsGrid > div").all():
        title = ""
        for sel in ["h3", "h4", ".project-title"]:
            el = card.locator(sel)
            if await el.count():
                title = clean(await el.first.inner_text())
                break
        desc  = clean(await card.locator("p").first.inner_text()) if await card.locator("p").count() else ""
        stack = [clean(await b.inner_text()) for b in await card.locator(".si, .badge, .tag, .stack-item").all()]
        links = {}
        for a in await card.locator("a[href]").all():
            href  = await a.get_attribute("href") or ""
            label = clean(await a.inner_text()) or await a.get_attribute("title") or href
            if href:
                links[label] = href
        if title:
            projects_home.append({"title": title, "description": desc, "stack": stack, "links": links})

    # ── EXTRA CURRICULARS ────────────────────────────────────────────────────
    # #extrasGrid > .extra-card > .extra-thumb + .extra-body > h3 + p
    await page.locator("#extrasGrid").scroll_into_view_if_needed()
    await asyncio.sleep(1)
    extras = []
    for card in await page.locator("#extrasGrid .extra-card").all():
        icon  = clean(await card.locator(".extra-thumb").inner_text()) if await card.locator(".extra-thumb").count() else ""
        title = clean(await card.locator("h3").inner_text())           if await card.locator("h3").count() else ""
        desc  = clean(await card.locator("p").inner_text())            if await card.locator("p").count() else ""
        if title:
            extras.append({"icon": icon, "title": title, "description": desc})

    # ── CONTACT ──────────────────────────────────────────────────────────────
    contact = {
        "name":  clean(await page.locator("#contactName").inner_text()),
        "role":  clean(await page.locator("#contactRole").inner_text()),
        "links": {}
    }
    for a in await page.locator("#contactDetails a[href]").all():
        href  = await a.get_attribute("href") or ""
        label = clean(await a.inner_text()) or href
        if href:
            contact["links"][label] = href

    # ── NAV ──────────────────────────────────────────────────────────────────
    nav = {}
    for a in await page.locator("#desktopNav a[href]").all():
        href  = await a.get_attribute("href") or ""
        label = clean(await a.inner_text())
        if href and label:
            nav[label] = href

    return {
        "hero":              hero,
        "skills":            skills,
        "resume":            resume,
        "researches":        researches,
        "projects_homepage": projects_home,
        "extra_curriculars": extras,
        "contact":           contact,
        "nav":               nav,
    }


async def _scrape_timeline(page, timeline_id: str) -> list:
    """Extract structured entries from a .timeline div using exact field classes."""
    entries = []
    for item in await page.locator(f"{timeline_id} .timeline-item").all():
        date    = clean(await item.locator(".t-date").inner_text())    if await item.locator(".t-date").count()    else ""
        role    = clean(await item.locator(".t-role").inner_text())    if await item.locator(".t-role").count()    else ""
        company = clean(await item.locator(".t-company").inner_text()) if await item.locator(".t-company").count() else ""
        desc    = clean(await item.locator(".t-desc").inner_text())    if await item.locator(".t-desc").count()    else ""
        links   = {}
        for a in await item.locator("a[href]").all():
            href  = await a.get_attribute("href") or ""
            label = clean(await a.inner_text()) or href
            if href:
                links[label] = href
        entries.append({"date": date, "role": role, "company": company, "description": desc, "links": links})
    return entries


# ─────────────────────────────────────────────────────────────────────────────
#  ARCHIVES PAGE
# ─────────────────────────────────────────────────────────────────────────────

async def scrape_archives(page) -> dict:
    await page.goto(f"{BASE_URL}/archives", wait_until="networkidle", timeout=60000)
    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await asyncio.sleep(2)
    await page.evaluate("window.scrollTo(0, 0)")
    await asyncio.sleep(1)

    # Projects table — tbody id is #PT
    projects = await _scrape_archive_table(page, "#PT")

    # Dashboards table — tbody id is #DT
    dashboards = await _scrape_archive_table(page, "#DT")

    # Collect internal slug URLs from dashboard link cells
    slug_urls = set()
    for row in dashboards:
        for lnk in row.get("links", []):
            href = lnk.get("href", "")
            if href and not href.startswith("http"):
                slug_urls.add(BASE_URL + "/archives/" + href.lstrip("/"))
            elif href and BASE_URL in href:
                slug_urls.add(href)

    dashboard_details = []
    for url in slug_urls:
        dashboard_details.append(await _scrape_slug(page, url))

    medium_link = ""
    for a in await page.locator("a[href*='medium']").all():
        medium_link = await a.get_attribute("href") or ""
        break

    return {
        "projects_table":       projects,
        "dashboards_table":     dashboards,
        "dashboard_slug_pages": dashboard_details,
        "medium_blog_link":     medium_link or "https://neerajjawahirani.medium.com/",
    }


async def _scrape_archive_table(page, tbody_id: str) -> list:
    """
    Scrape an archive table by tbody id.
    Stack chips: .si  |  Link anchors: .il
    """
    rows = []
    for tr in await page.locator(f"{tbody_id} tr").all():
        cells = await tr.locator("td").all()
        if len(cells) < 4:
            continue
        name  = clean(await cells[0].inner_text())
        stack = [clean(await s.inner_text()) for s in await cells[1].locator(".si").all()]
        desc  = clean(await cells[2].inner_text())
        links = []
        for a in await cells[3].locator("a.il, a[href]").all():
            href  = await a.get_attribute("href") or ""
            title = await a.get_attribute("title") or ""
            aria  = await a.get_attribute("aria-label") or ""
            label = title or aria or clean(await a.inner_text()) or href
            if href:
                links.append({"label": label, "href": href})
        rows.append({"name": name, "stack": stack, "description": desc, "links": links})
    return rows


async def _scrape_slug(page, url: str) -> dict:
    """Visit a dashboard slug page and extract its full content."""
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)
        title = ""
        for sel in ["h1", "h2", ".dashboard-title", ".title"]:
            el = page.locator(sel).first
            if await el.count():
                title = clean(await el.inner_text())
                break
        content = ""
        for sel in ["main", ".content", ".dashboard-content", "article", "body"]:
            el = page.locator(sel).first
            if await el.count():
                content = clean(await el.inner_text())
                break
        return {"url": url, "title": title, "content": content[:4000]}
    except Exception as e:
        return {"url": url, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIUM (RSS — reliable, no JS needed)
# ─────────────────────────────────────────────────────────────────────────────

async def scrape_medium(page) -> list:
    import xml.etree.ElementTree as ET
    blogs = []
    try:
        await page.goto("https://neerajjawahirani.medium.com/feed", wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)
        content = await page.locator("body").inner_text()
        root = ET.fromstring(content)
        for item in root.findall(".//item"):
            title   = (item.findtext("title")      or "").strip()
            link    = (item.findtext("link")        or "").strip()
            pub     = (item.findtext("pubDate")     or "").strip()
            summary = (item.findtext("description") or "").strip()
            summary = re.sub(r"<[^>]+>", "", summary).strip()[:300]
            if title:
                blogs.append({"title": title, "url": link, "published": pub, "summary": summary})
    except Exception as e:
        blogs.append({"error": f"Medium RSS failed: {e}"})
    return blogs


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def to_markdown(main: dict, arch: dict, blogs: list) -> str:
    lines = []

    def h(n, t): lines.append(f"\n{'#'*n} {t}\n")
    def p(t):
        if t: lines.append(t + "\n")

    h(1, "Neeraj Jawahirani — Portfolio")

    h(2, "About")
    hero = main["hero"]
    p(f"**Name:** {hero['name']}")
    p(f"**Bio:** {hero['bio']}")
    if hero.get("tagline"):
        p(f"**Tagline:** {hero['tagline']}")
    h(3, "Social Links")
    for label, url in hero["social_links"].items():
        lines.append(f"- [{label}]({url})")

    h(2, "Site Navigation")
    for label, url in main["nav"].items():
        lines.append(f"- [{label}]({url})")

    h(2, "Skills")
    for s in main["skills"]:
        lines.append(f"- **{s['icon']} {s['category']}**: {s['tools']}")

    h(2, "Resume")
    h(3, "Experience")
    for e in main["resume"].get("experience", []):
        lines.append(f"\n**{e['role']}** | {e['company']} | _{e['date']}_")
        if e["description"]: lines.append(e["description"])
    h(3, "Education")
    for e in main["resume"].get("education", []):
        lines.append(f"\n**{e['role']}** | {e['company']} | _{e['date']}_")
        if e["description"]: lines.append(e["description"])
    h(3, "Certifications")
    for c in main["resume"].get("certifications", []):
        line = f"- {c['text']}"
        if c.get("link"): line += f" ([link]({c['link']}))"
        lines.append(line)
    h(3, "Leadership")
    for e in main["resume"].get("leadership", []):
        lines.append(f"\n**{e['role']}** | {e['company']} | _{e['date']}_")
        if e["description"]: lines.append(e["description"])

    h(2, "Researches & Initiatives")
    for r in main["researches"]:
        h(3, r["title"])
        p(r["description"])
        for lbl, url in r.get("links", {}).items():
            lines.append(f"- [{lbl}]({url})")

    h(2, "Featured Projects (Homepage)")
    for proj in main["projects_homepage"]:
        h(3, proj["title"])
        p(proj["description"])
        if proj["stack"]: lines.append(f"**Stack:** {', '.join(proj['stack'])}")
        for lbl, url in proj.get("links", {}).items():
            lines.append(f"- [{lbl}]({url})")

    h(2, "Extra Curriculars")
    for e in main["extra_curriculars"]:
        lines.append(f"- **{e['icon']} {e['title']}**: {e['description']}")

    h(2, "Contact")
    c = main["contact"]
    p(f"**{c['name']}** — {c['role']}")
    for lbl, url in c["links"].items():
        lines.append(f"- [{lbl}]({url})")

    h(1, "Project Archives")
    h(2, "All Projects")
    for row in arch["projects_table"]:
        h(3, row["name"])
        p(row["description"])
        if row["stack"]: lines.append(f"**Stack:** {', '.join(row['stack'])}")
        for lnk in row["links"]:
            lines.append(f"- [{lnk['label']}]({lnk['href']})")

    h(2, "Dashboards")
    for row in arch["dashboards_table"]:
        h(3, row["name"])
        p(row["description"])
        if row["stack"]: lines.append(f"**Stack:** {', '.join(row['stack'])}")
        for lnk in row["links"]:
            lines.append(f"- [{lnk['label']}]({lnk['href']})")

    h(2, "Dashboard Detail Pages")
    for d in arch.get("dashboard_slug_pages", []):
        h(3, d.get("title") or d["url"])
        p(f"URL: {d['url']}")
        p(d.get("content") or d.get("error", ""))

    h(2, "Medium Blog Posts")
    p(f"Blog: {arch['medium_blog_link']}")
    for b in blogs:
        if "error" not in b:
            lines.append(f"- **[{b['title']}]({b['url']})** _{b.get('published','')}_")
            if b.get("summary"): lines.append(f"  {b['summary']}")

    return "\n".join(lines)


def to_txt(main: dict, arch: dict, blogs: list) -> str:
    parts = []

    def section(title):
        parts.append(f"\n{'='*60}\n{title.upper()}\n{'='*60}")

    section("ABOUT")
    hero = main["hero"]
    parts.append(f"Name: {hero['name']}")
    parts.append(f"Bio: {hero['bio']}")
    parts.append(f"Tagline: {hero.get('tagline','')}")
    parts.append("Social links:")
    for k, v in hero["social_links"].items():
        parts.append(f"  {k}: {v}")

    section("SKILLS")
    for s in main["skills"]:
        parts.append(f"{s['icon']} {s['category']}: {s['tools']}")

    section("EXPERIENCE")
    for e in main["resume"].get("experience", []):
        parts.append(f"\n{e['date']}")
        parts.append(f"  Role:    {e['role']}")
        parts.append(f"  Company: {e['company']}")
        parts.append(f"  Details: {e['description']}")

    section("EDUCATION")
    for e in main["resume"].get("education", []):
        parts.append(f"\n{e['date']}")
        parts.append(f"  Degree:  {e['role']}")
        parts.append(f"  School:  {e['company']}")
        parts.append(f"  Details: {e['description']}")

    section("CERTIFICATIONS")
    for c in main["resume"].get("certifications", []):
        parts.append(f"  - {c['text']}")

    section("LEADERSHIP")
    for e in main["resume"].get("leadership", []):
        parts.append(f"\n{e['date']}")
        parts.append(f"  Role:    {e['role']}")
        parts.append(f"  Org:     {e['company']}")
        parts.append(f"  Details: {e['description']}")

    section("RESEARCHES & INITIATIVES")
    for r in main["researches"]:
        parts.append(f"\n{r['title']}")
        parts.append(f"  {r['description']}")

    section("FEATURED PROJECTS (HOMEPAGE)")
    for proj in main["projects_homepage"]:
        parts.append(f"\n{proj['title']}")
        parts.append(f"  Stack: {', '.join(proj['stack'])}")
        parts.append(f"  {proj['description']}")
        for lbl, url in proj.get("links", {}).items():
            parts.append(f"  Link: {lbl} -> {url}")

    section("EXTRA CURRICULARS")
    for e in main["extra_curriculars"]:
        parts.append(f"  {e['icon']} {e['title']}: {e['description']}")

    section("CONTACT")
    c = main["contact"]
    parts.append(f"{c['name']} | {c['role']}")
    for lbl, url in c["links"].items():
        parts.append(f"  {lbl}: {url}")

    section("ALL PROJECTS (ARCHIVES)")
    for row in arch["projects_table"]:
        parts.append(f"\nProject: {row['name']}")
        parts.append(f"  Stack: {', '.join(row['stack'])}")
        parts.append(f"  Description: {row['description']}")
        for lnk in row["links"]:
            parts.append(f"  Link [{lnk['label']}]: {lnk['href']}")

    section("DASHBOARDS (ARCHIVES)")
    for row in arch["dashboards_table"]:
        parts.append(f"\nDashboard: {row['name']}")
        parts.append(f"  Stack: {', '.join(row['stack'])}")
        parts.append(f"  Description: {row['description']}")
        for lnk in row["links"]:
            parts.append(f"  Link [{lnk['label']}]: {lnk['href']}")

    section("DASHBOARD DETAIL PAGES")
    for d in arch.get("dashboard_slug_pages", []):
        parts.append(f"\n{d.get('title', d['url'])}")
        parts.append(f"  URL: {d['url']}")
        parts.append(f"  {d.get('content', d.get('error', ''))}")

    section("MEDIUM BLOG POSTS")
    parts.append(f"Blog URL: {arch['medium_blog_link']}")
    for b in blogs:
        if "error" not in b:
            parts.append(f"\nTitle: {b['title']}")
            parts.append(f"  URL: {b['url']}")
            parts.append(f"  Published: {b.get('published','')}")
            parts.append(f"  Summary: {b.get('summary','')}")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120"
        )
        page = await context.new_page()

        print("── Scraping main page...")
        main_data = await scrape_main(page)
        print(f"   skills:           {len(main_data['skills'])} cards")
        print(f"   experience:       {len(main_data['resume']['experience'])} entries")
        print(f"   education:        {len(main_data['resume']['education'])} entries")
        print(f"   certifications:   {len(main_data['resume']['certifications'])} entries")
        print(f"   leadership:       {len(main_data['resume']['leadership'])} entries")
        print(f"   researches:       {len(main_data['researches'])} cards")
        print(f"   projects (home):  {len(main_data['projects_homepage'])} cards")
        print(f"   extra curricular: {len(main_data['extra_curriculars'])} cards")

        print("── Scraping archives page...")
        arch_data = await scrape_archives(page)
        print(f"   projects table:   {len(arch_data['projects_table'])} rows")
        print(f"   dashboards table: {len(arch_data['dashboards_table'])} rows")
        print(f"   slug pages:       {len(arch_data['dashboard_slug_pages'])} visited")

        print("── Scraping Medium RSS...")
        blogs = await scrape_medium(page)
        print(f"   blog posts:       {len(blogs)}")

        await browser.close()

    full = {"source": BASE_URL, "main": main_data, "archives": arch_data, "medium_blogs": blogs}

    json_path = OUTPUT_DIR / "portfolio_scraped.json"
    json_path.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = OUTPUT_DIR / "portfolio_scraped.md"
    md_path.write_text(to_markdown(main_data, arch_data, blogs), encoding="utf-8")

    txt_path = OUTPUT_DIR / "portfolio_scraped.txt"
    txt_path.write_text(to_txt(main_data, arch_data, blogs), encoding="utf-8")

    print(f"\nDone. Files saved to: {OUTPUT_DIR.resolve()}")
    print(f"  {json_path.name}")
    print(f"  {md_path.name}")
    print(f"  {txt_path.name}")


if __name__ == "__main__":
    asyncio.run(main())