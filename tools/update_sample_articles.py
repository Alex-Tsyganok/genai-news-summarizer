"""
Resolve hub/topic URLs in data/sample_articles.json to concrete article URLs.

For each entry:
- Validate if the URL looks like a direct article.
- If it's a hub/topic page, fetch and try to find a recent article link.
- Fetch the article page to extract title and infer expected topics.
- Update sample_articles.json in place.

Note: Best-effort scraping with minimal heuristics; no external APIs used.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Iterable

import requests
from bs4 import BeautifulSoup


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "sample_articles.json"


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}


def fetch_html(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200 and r.text:
            return r.text
    except requests.RequestException:
        return None
    return None


def first_match(hrefs: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        for href in hrefs:
            if pat in href:
                return href
    return None


def absolutize(base: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        from urllib.parse import urljoin

        return urljoin(base, href)
    from urllib.parse import urljoin

    return urljoin(base + ("/" if not base.endswith("/") else ""), href)


def extract_article_from_hub(url: str, html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    hrefs = []
    for a in soup.find_all("a", href=True):
        hrefs.append(absolutize(url, a["href"]))

    if "theverge.com" in url:
        # Prefer AI-related articles with year in path /202x/
        for href in hrefs:
            if (
                "theverge.com/20" in href
                and (
                    "/ai-" in href
                    or "/ai/" in href
                    or "/artificial-intelligence" in href
                )
                and "/deals" not in href
            ):
                return href
        # Fallback: any article-like 20xx link excluding deals
        for href in hrefs:
            if "theverge.com/20" in href and "/deals" not in href:
                return href

    if "apnews.com" in url:
        for href in hrefs:
            if "/article/" in href and re.search(r"\b(ai|artificial)\b", href, re.I):
                return href

    if "wired.com" in url:
        for href in hrefs:
            if (
                ("wired.com/story/" in href or "wired.com/article/" in href)
                and re.search(r"\b(ai|artificial|gpt|openai)\b", href, re.I)
            ):
                return href

    if "reuters.com" in url:
        for href in hrefs:
            if "/technology/" in href and re.search(r"/\d{4}-\d{2}-\d{2}/", href):
                return href
        # Reuters new style sometimes includes id near end
        for href in hrefs:
            if "/technology/" in href:
                return href

    return None


def extract_title_and_topics(url: str, html: str) -> Tuple[str, List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    # Prefer h1
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    if not title:
        if soup.title and soup.title.get_text(strip=True):
            title = soup.title.get_text(strip=True)
    title = title.strip() or url

    # Infer topics from title words (simple heuristic)
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", title) if len(w) > 2]
    core = []
    for w in words:
        if w in {"the", "and", "with", "from", "into", "over", "about", "this", "that", "have"}:
            continue
        core.append(w)
        if len(core) >= 5:
            break

    # Domain tag
    domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
    topics = list(dict.fromkeys(core + [domain.split(".")[0], "ai"]))
    return title, topics


def looks_like_article(url: str) -> bool:
    return any(
        key in url
        for key in [
            "/20",  # year in path
            "/article/",
            "/story/",
            "/news/",
        ]
    ) and not any(key in url for key in ["/tag/", "/hub/", "/ai-artificial-intelligence"])  # topic pages


def process_entry(url: str) -> Tuple[str, Optional[str], Optional[List[str]]]:
    html = fetch_html(url)
    if not html:
        return url, None, None

    if looks_like_article(url):
        title, topics = extract_title_and_topics(url, html)
        return url, title, topics

    # Hub/topic page; try to pick an article link
    article_url = extract_article_from_hub(url, html)
    if not article_url:
        return url, None, None

    article_html = fetch_html(article_url)
    if not article_html:
        return url, None, None

    title, topics = extract_title_and_topics(article_url, article_html)
    return article_url, title, topics


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] in {"--generate", "-g"}:
        count = 50
        if len(args) > 1 and args[1].isdigit():
            count = int(args[1])
        return generate(count)

    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    updated = 0
    for i, entry in enumerate(data):
        orig_url = entry["url"].strip()
        new_url, title, topics = process_entry(orig_url)
        if new_url != orig_url or title or topics:
            if new_url:
                entry["url"] = new_url
            if title:
                entry["title"] = title
            if topics:
                entry["expected_topics"] = topics
            updated += 1
            print(f"Updated [{i}]: {orig_url} -> {entry['url']}")
        else:
            print(f"No change [{i}]: {orig_url}")

    DATA_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Entries updated: {updated}/{len(data)}")
    return 0


def generate(target_count: int = 50) -> int:
    seeds: List[str] = [
        # General/AI hubs
        "https://www.bbc.com/news/technology",
        "https://www.bbc.com/news/articles/c0k3700zljjo",  # concrete
        "https://www.reuters.com/technology/",
        "https://www.reuters.com/technology/ai/",
        "https://apnews.com/hub/technology",
        "https://apnews.com/hub/artificial-intelligence",
        "https://www.theverge.com/ai-artificial-intelligence",
        "https://www.wired.com/tag/artificial-intelligence/",
        "https://arstechnica.com/tag/artificial-intelligence/",
        "https://www.theguardian.com/technology/artificialintelligenceai",
        "https://techcrunch.com/tag/ai/",
        "https://venturebeat.com/category/ai/",
        "https://www.cnbc.com/ai/",
        # Company research blogs
        "https://blog.google/technology/ai/",
        "https://openai.com/blog",
        "https://www.microsoft.com/en-us/research/blog/category/artificial-intelligence/",
        "https://ai.facebook.com/blog/",
        "https://blogs.nvidia.com/blog/category/ai/",
        "https://aws.amazon.com/blogs/machine-learning/",
        "https://blog.cloudflare.com/tag/ai/",
        "https://www.anthropic.com/news",
        "https://www.deepmind.com/blog",
        "https://stability.ai/blog",
        "https://huggingface.co/blog",
        # Academia/research
        "https://hai.stanford.edu/news",
        "https://www.technologyreview.com/topic/artificial-intelligence/",
        # Security and other topics
        "https://www.bleepingcomputer.com/",
        "https://www.theregister.com/security/",
        "https://www.zdnet.com/topic/artificial-intelligence/",
        "https://www.scientificamerican.com/artificial-intelligence/",
    ]

    # Start with preserving the first existing entry if available
    entries: List[dict] = []
    if DATA_FILE.exists():
        try:
            existing = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            if isinstance(existing, list) and existing:
                entries.append(existing[0])
        except Exception:
            pass

    seen_urls = {e["url"] for e in entries}

    def add_article(u: str):
        nonlocal entries
        if u in seen_urls:
            return
        html = fetch_html(u)
        if not html:
            return
        if not looks_like_article(u):
            # try resolving from hub
            aurl = extract_article_from_hub(u, html)
            if not aurl:
                return
            u = aurl
            html = fetch_html(u) or ""
            if not html:
                return
        title, topics = extract_title_and_topics(u, html)
        entries.append({
            "url": u,
            "title": title,
            "expected_topics": topics,
        })
        seen_urls.add(u)

    for seed in seeds:
        if len(entries) >= target_count:
            break
        html = fetch_html(seed)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        hrefs = []
        for a in soup.find_all("a", href=True):
            hrefs.append(absolutize(seed, a["href"]))
        # prefer likely article links
        candidates = [h for h in hrefs if looks_like_article(h)]
        # small cap per seed to diversify
        for h in candidates[:3]:
            if len(entries) >= target_count:
                break
            add_article(h)

    # If still short, do a second pass allowing hub resolution
    if len(entries) < target_count:
        for seed in seeds:
            if len(entries) >= target_count:
                break
            add_article(seed)

    # Truncate to target_count
    entries = entries[:target_count]
    DATA_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(entries)} entries into {DATA_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
