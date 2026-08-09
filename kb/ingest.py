"""Scrape shasuncollege.edu.in into a clean, normalized corpus.

Ingest path, in priority order:

1. **WordPress REST API** (``/wp-json/wp/v2/...``). The site exposes it
   unauthenticated, and ``content.rendered`` is the post body *without* the
   site nav and footer. That matters more than convenience: extracting the same
   pages from rendered HTML measured **94.9% boilerplate**, so a naive scrape
   would have filled the index with 200 near-identical copies of the menu and
   drowned the actual answers.
2. **HTML fallback** for documents the API returns thin or empty. Elementor
   stores its layout in postmeta rather than ``content.rendered``, so
   Elementor-built pages come back blank from the API and must be scraped from
   the page itself, where boilerplate is subtracted by frequency (see
   ``_strip_shared_boilerplate``).

Output is ``corpus.jsonl`` — one JSON object per document, already sanitized.

    python -m kb.ingest              # full scrape
    python -m kb.ingest --limit 20   # quick smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .sanitize import (
    clean_text,
    extract_directory,
    html_to_text,
    injection_signals,
    page_html_to_text,
)

SITE = "https://shasuncollege.edu.in"
API = f"{SITE}/wp-json/wp/v2"
UA = "sia-college-assistant/1.0 (+knowledge base ingest)"

# REST collections worth indexing. `portfolio` is listed but returns 0 via the
# API on this install, so it falls through to the sitemap/HTML path.
REST_TYPES = ("pages", "knowledge-economy", "mec-events", "portfolio")

# Post types that are pure site furniture — indexing them would add noise.
SITEMAP_SKIP = ("elementor-hf", "jupiterx-codes", "ae_global_templates", "post_tag", "mec_category")

OUT_DIR = Path(__file__).resolve().parent.parent / "kb-data"
CORPUS = OUT_DIR / "corpus.jsonl"
QUARANTINE = OUT_DIR / "quarantine.jsonl"

# Raw fetched HTML, kept on disk. The site serves ~1.4 MB per page and throttles
# to roughly 10 s each, so a full crawl is ~40 minutes — far too slow to iterate
# on extraction quality against. With the cache, re-running ingest after an
# extractor change re-parses locally in seconds and refetches nothing. It also
# makes the crawl resumable, which matters when a run that long gets
# interrupted. Delete the directory (or pass --refetch) to pull fresh pages.
HTML_CACHE = OUT_DIR / "html-cache"

MIN_DOC_CHARS = 120  # below this a "document" is a stub, not an answer source

# A REST document needs at least this many prose lines to count as real content.
# Elementor stores page layout in postmeta, so for Elementor-built pages
# `content.rendered` contains only the in-page navigation widget — a stack of
# 2-4 word menu items. Those come back long enough to look like a document while
# containing no answer, and indexing them buries the pages that do. Requiring
# real sentences distinguishes the two and sends the rest to the HTML fallback.
MIN_PROSE_LINES = 2
PROSE_MIN_WORDS = 10


def prose_lines(text: str) -> int:
    return sum(1 for ln in text.splitlines() if len(ln.split()) >= PROSE_MIN_WORDS)


def is_real_content(text: str) -> bool:
    """Does a REST document carry actual content, or only its nav widget?

    Only meaningful on the REST path. Elementor pages return `content.rendered`
    containing nothing but a stack of 2-4 word menu items, which is long enough
    to look like a document, so requiring real sentences is what separates them.
    """
    return len(text) >= MIN_DOC_CHARS and prose_lines(text) >= MIN_PROSE_LINES


def is_real_page(text: str) -> bool:
    """Does an HTML-extracted page carry content? Length only, deliberately.

    The prose test must NOT be applied here. `page_html_to_text` has already
    removed nav structurally, and plenty of real pages are legitimately made of
    short lines: the NAAC page is 317 characters reading "NAAC Accreditation:",
    "A++ Grade (2023)", "B++ Grade (June 2018)" — every fact a visitor wants,
    and zero lines of ten words or more. Requiring prose there silently dropped
    it from the corpus, and no amount of retrieval tuning can find a page that
    was never indexed.
    """
    return len(text) >= MIN_DOC_CHARS


@dataclass
class Doc:
    id: str
    url: str
    title: str
    text: str
    source: str        # "rest" | "html"
    kind: str
    modified: str = ""


def _get(url: str, *, retries: int = 3, timeout: int = 45) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401, 403, 404):
                raise
            last = exc
        except Exception as exc:  # noqa: BLE001 - network flakiness, retry
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise last if last else RuntimeError(f"failed: {url}")


def _get_json(url: str):
    return json.loads(_get(url).decode("utf-8", "replace"))


def _cache_path(url: str) -> Path:
    return HTML_CACHE / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + ".html")


def _get_page(url: str, *, refetch: bool = False) -> str | None:
    """Fetch a page, reading from (and writing to) the on-disk HTML cache."""
    path = _cache_path(url)
    if not refetch and path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        html = _get(url).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] {url}: {exc}")
        return None
    HTML_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8", errors="replace")
    return html


# --- REST ---------------------------------------------------------------------

def fetch_rest(kind: str) -> list[Doc]:
    docs: list[Doc] = []
    page = 1
    while True:
        url = (
            f"{API}/{kind}?per_page=100&page={page}"
            "&_fields=id,link,title,content,modified,status"
        )
        try:
            batch = _get_json(url)
        except urllib.error.HTTPError:
            break  # past the last page, or type not exposed
        if not isinstance(batch, list) or not batch:
            break

        for item in batch:
            text = html_to_text((item.get("content") or {}).get("rendered", ""))
            title = clean_text(html_to_text((item.get("title") or {}).get("rendered", "")))
            docs.append(
                Doc(
                    id=f"{kind}-{item.get('id')}",
                    url=item.get("link", ""),
                    title=title,
                    text=text,
                    source="rest",
                    kind=kind,
                    modified=item.get("modified", "") or "",
                )
            )
        if len(batch) < 100:
            break
        page += 1
    return docs


# --- sitemap + HTML fallback ---------------------------------------------------

def sitemap_urls() -> list[str]:
    """Every content URL the site advertises, minus template/furniture types."""
    index = _get(f"{SITE}/sitemap_index.xml").decode("utf-8", "replace")
    subs = [m for m in _locs(index) if not any(s in m for s in SITEMAP_SKIP)]
    urls: list[str] = []
    for sub in subs:
        try:
            urls.extend(_locs(_get(sub).decode("utf-8", "replace")))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] sitemap {sub}: {exc}")
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(urls))


def _locs(xml: str) -> list[str]:
    import re

    return [m.strip() for m in re.findall(r"<loc>(.*?)</loc>", xml, re.S)]


def _drop_residual_chrome(pages: dict[str, str]) -> dict[str, str]:
    """Second pass: remove any line still shared by most pages.

    Structural extraction (``page_html_to_text``) does the heavy lifting; this
    only sweeps up stragglers such as a call-to-action banner that sits inside
    the content region on every page. The threshold is deliberately high —
    subtracting text by frequency was tried as the *primary* mechanism and it
    deleted real content along with the nav, so it is kept as a light finishing
    pass rather than the main filter.
    """
    if len(pages) < 8:
        return pages
    counts: Counter[str] = Counter()
    for text in pages.values():
        counts.update({ln.strip() for ln in text.splitlines() if ln.strip()})
    threshold = max(8, int(0.6 * len(pages)))
    chrome = {line for line, n in counts.items() if n >= threshold and len(line) <= 90}
    return {
        url: "\n".join(ln for ln in text.splitlines() if ln.strip() and ln.strip() not in chrome)
        for url, text in pages.items()
    }


def fetch_html(urls: Iterable[str], *, refetch: bool = False) -> list[Doc]:
    urls = list(urls)
    raw: dict[str, str] = {}
    titles: dict[str, str] = {}
    directory: list[str] = []
    fetched = 0
    for i, url in enumerate(urls, 1):
        cached = not refetch and _cache_path(url).exists()
        html = _get_page(url, refetch=refetch)
        if html is None:
            continue
        raw[url] = page_html_to_text(html)
        titles[url] = _title(html)
        # The staff directory is stamped identically into every page, so the
        # first page carrying it supplies the whole thing and the rest are
        # ignored. Taking the largest keeps a page that happens to render a
        # partial list from winning.
        if len(entries := extract_directory(html)) > len(directory):
            directory = entries
        if i % 25 == 0:
            print(f"  html {i}/{len(urls)} ({fetched} fetched, {i - fetched} cached)...")
        if not cached:
            fetched += 1
            time.sleep(0.15)  # be a polite guest on someone else's server

    cleaned = _drop_residual_chrome(raw)
    docs = [
        Doc(
            id="html-" + (urllib.parse.urlparse(url).path.strip("/").replace("/", "-") or "root"),
            url=url,
            title=titles.get(url, ""),
            text=text,
            source="html",
            kind="page",
        )
        for url, text in cleaned.items()
    ]

    if directory:
        print(f"  faculty directory: {len(directory)} entries lifted out of the popup markup")
        docs.append(
            Doc(
                id="directory-faculty",
                url=f"{SITE}/faculty/",
                title="Faculty and Staff Directory",
                text="Faculty and staff of Shasun Jain College for Women.\n"
                + "\n".join(directory),
                source="directory",
                kind="faculty",
            )
        )
    return docs


def _title(html: str) -> str:
    import re

    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    if not m:
        return ""
    return clean_text(html_to_text(m.group(1))).split("|")[0].strip()


# --- driver -------------------------------------------------------------------

def build(limit: int | None = None, refetch: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_url: dict[str, Doc] = {}

    print("== WordPress REST API ==")
    nav_only = 0
    for kind in REST_TYPES:
        docs = fetch_rest(kind)
        kept = [d for d in docs if is_real_content(d.text)]
        thin = sum(1 for d in docs if len(d.text) >= MIN_DOC_CHARS and not is_real_content(d.text))
        nav_only += thin
        print(f"  {kind:20s} {len(docs):4d} fetched, {len(kept):4d} usable, {thin:3d} nav-only -> HTML")
        for d in kept:
            by_url[d.url] = d
    print(f"  {nav_only} Elementor/nav-only documents deferred to the HTML fallback")

    print("\n== sitemap (HTML fallback) ==")
    all_urls = sitemap_urls()
    missing = [u for u in all_urls if u not in by_url]
    if limit:
        missing = missing[:limit]
    print(f"  {len(all_urls)} sitemap urls, {len(missing)} not covered by usable REST content")
    for d in fetch_html(missing, refetch=refetch):
        if is_real_page(d.text):
            by_url[d.url] = d

    # Quarantine anything carrying injection-shaped text rather than indexing it.
    clean: list[Doc] = []
    flagged: list[dict] = []
    for doc in by_url.values():
        signals = injection_signals(doc.text)
        if signals:
            flagged.append({**asdict(doc), "signals": signals})
        else:
            clean.append(doc)

    with CORPUS.open("w", encoding="utf-8") as fh:
        for doc in sorted(clean, key=lambda d: d.url):
            fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    with QUARANTINE.open("w", encoding="utf-8") as fh:
        for item in flagged:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    chars = sum(len(d.text) for d in clean)
    print(f"\n{len(clean)} documents -> {CORPUS}")
    print(f"  {chars:,} chars (~{chars // 4:,} tokens)")
    print(f"  {len(flagged)} quarantined for injection-shaped content -> {QUARANTINE.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scrape the college site into a clean corpus.")
    ap.add_argument("--limit", type=int, help="cap the HTML-fallback fetches (smoke test)")
    ap.add_argument(
        "--refetch",
        action="store_true",
        help="ignore the on-disk HTML cache and pull every page fresh",
    )
    args = ap.parse_args()
    build(args.limit, refetch=args.refetch)
