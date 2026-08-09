"""Sanitization for scraped content — the first line of prompt-injection defense.

The scraped website is the injection surface. Anyone who can put text on the
WordPress install (a plugin, an event listing, a comment, an uploaded page) can
plant instructions that the model would otherwise read as if the operator wrote
them. Two rules follow:

1. Nothing invisible survives ingest. Text a human visitor cannot see has no
   legitimate reason to reach the model, and hidden text is the classic carrier
   for injected instructions — HTML comments, ``display:none`` blocks,
   zero-width characters, bidi overrides.
2. Sanitizing is not sufficient on its own. It raises the cost of an attack but
   cannot prove text is safe, so the prompt layer (``prompts.py``) still treats
   every retrieved chunk as hostile data rather than instructions.
"""

from __future__ import annotations

import re
import unicodedata

from selectolax.parser import HTMLParser

# Elements that never carry visible prose.
_DROP_TAGS = (
    "script", "style", "noscript", "svg", "canvas", "template", "iframe",
    "object", "embed", "form", "input", "button", "select", "textarea",
)

# Inline CSS that hides an element from a sighted visitor.
_HIDDEN_CSS = re.compile(
    r"(display\s*:\s*none)|(visibility\s*:\s*hidden)|(opacity\s*:\s*0(\.0+)?\s*[;\"']?)"
    r"|(font-size\s*:\s*0)|(text-indent\s*:\s*-\d{3,})"
    r"|(position\s*:\s*absolute\s*;?\s*(left|top)\s*:\s*-\d{3,})",
    re.I,
)

# Zero-width and directional-override characters. These render as nothing but
# survive naive text extraction, so they can smuggle text past a human reviewer.
_INVISIBLE = re.compile(
    "["
    "​-‏"  # zero-width space/joiners, LRM/RLM
    "‪-‮"  # bidi embedding/override
    "⁠-⁤"  # word joiner, invisible operators
    "⁪-⁯"  # deprecated format controls
    "﻿"         # BOM / zero-width no-break space
    "­"         # soft hyphen
    "]"
)

# Private-use area — used by icon fonts, and a known smuggling channel.
_PRIVATE_USE = re.compile("[-\U000f0000-\U000ffffd]")

MAX_DOC_CHARS = 60_000


def _visible_text(html: str) -> str:
    """Extract only what a sighted visitor would actually read."""
    tree = HTMLParser(html)

    for tag in _DROP_TAGS:
        for node in tree.css(tag):
            node.decompose()

    # Comments are invisible to visitors; selectolax exposes them as -comment nodes.
    for node in tree.css("-comment"):
        node.decompose()

    for node in tree.css("[style]"):
        if _HIDDEN_CSS.search(node.attributes.get("style") or ""):
            node.decompose()

    for node in tree.css("[hidden], [aria-hidden=true]"):
        node.decompose()

    body = tree.body or tree.root
    return body.text(separator="\n") if body else ""


def clean_text(raw: str) -> str:
    """Normalize extracted text into something safe to embed and to show a model."""
    text = _INVISIBLE.sub("", raw)
    text = _PRIVATE_USE.sub("", text)
    # NFKC folds look-alike/compatibility forms so an attacker can't evade a
    # downstream check by spelling a word with fullwidth or math-styled glyphs.
    text = unicodedata.normalize("NFKC", text)
    # Control characters except tab/newline.
    text = "".join(c for c in text if c in "\t\n" or not unicodedata.category(c).startswith("C"))
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(html: str) -> str:
    """Full ingest path for one HTML fragment: strip hidden nodes, then normalize."""
    if not html:
        return ""
    return clean_text(_visible_text(html))[:MAX_DOC_CHARS]


# Site chrome, identified structurally. This is far more reliable than
# subtracting text that repeats across pages: on this site the shared nav is
# ~95% of every page's text, and frequency subtraction was aggressive enough to
# delete the real content along with it (the NAAC page reduced to 205 chars, the
# milestones page to zero). Selecting the content region instead recovers both.
_CHROME_SELECTORS = (
    "nav", "header", "footer",
    "[role=navigation]", "[role=banner]", "[role=contentinfo]",
    "[class*=breadcrumb]", "[id*=breadcrumb]",
    "[class*=nav-menu]",
    "[class*=sidebar]", "[class*=widget-area]",
    "[class*=cookie]", "[class*=popup]", "[class*=modal]",
    # Popup Maker renders the *entire* 226-entry faculty directory into the
    # markup of every page as hidden popups — 535 KB of identical text per page,
    # on 145 pages. `[class*=popup]` does not catch it, because the plugin's
    # classes are `pum-*` and `popmake-*`; that near-miss is what let it through.
    #
    # The cost was not merely wasted bytes. The chunker subtracts lines that
    # repeat across documents, so this bundle looked exactly like site chrome and
    # took **82% of the corpus with it** (3.35M chars in, 642K indexed) —
    # deleting genuine faculty specializations and syllabus lines corpus-wide
    # because they happened to appear inside it. Removing it here is what makes
    # the frequency filter in `index.py` safe to keep.
    #
    # The directory is not discarded: `extract_directory()` below emits it once
    # as its own document, so "who teaches psychology" still has something to
    # retrieve. Deduplicated, not deleted.
    "[class*=pum-]", "[class*=popmake]", "[id*=popmake]", "[id*=pum-]", "[class*=pum_]",
    ".screen-reader-text", ".skip-link",
)

# The Popup Maker container that holds one directory entry: name, designation,
# qualifications, experience, email.
_DIRECTORY_ENTRY_SELECTOR = "[class*=pum-container]"

# Containers that are *sometimes* navigation and sometimes content, so they are
# judged by link density rather than by class name. Elementor renders both its
# sub-menus and its factual bullet lists as `icon-list`: dropping the class
# outright cost the NAAC page its accreditation grades, which live in one.
_AMBIGUOUS_LIST_SELECTORS = ("[class*=icon-list]", "ul", "ol")

# A list whose text is almost entirely inside links is a menu; a list that
# carries standalone text is content.
_NAV_LINK_TEXT_RATIO = 0.9
_NAV_MIN_LINKS = 3


# Tab and accordion panels. Elementor marks every *inactive* panel
# `aria-hidden="true"`, but a visitor reaches that text with one click — it is
# real page content, not concealed text. Treating it as hidden cost the NAAC
# page its accreditation grades, which sit in the second tab.
#
# This narrows the hidden-text defense slightly, and that is a deliberate
# trade: the text is still sanitized, still quarantine-scanned, and still framed
# as untrusted data in the prompt, and anyone able to publish a tab panel on the
# CMS could equally publish visible text. The rule still removes what is
# genuinely invisible — `display:none`, zero-width characters, offscreen nodes.
_REVEALABLE_SELECTORS = (
    "[role=tabpanel]",
    "[class*=tab-content]",
    "[class*=tab-pane]",
    "[class*=accordion]",
    "[class*=toggle-content]",
)


def _unhide_revealable(tree) -> None:
    """Clear the hidden flags on tab/accordion panels before the hidden sweep.

    Elementor marks each inactive panel `hidden="hidden"` on the panel element
    itself, so clearing it there is enough — the text inside is not separately
    flagged.
    """
    for selector in _REVEALABLE_SELECTORS:
        for panel in tree.css(selector):
            for attr in ("hidden", "aria-hidden"):
                try:
                    del panel.attrs[attr]
                except (KeyError, TypeError, AttributeError):
                    pass


def _looks_like_navigation(node) -> bool:
    links = node.css("a")
    if len(links) < _NAV_MIN_LINKS:
        return False
    total = len("".join(node.text(separator=" ").split()))
    if total == 0:
        return True
    linked = len("".join(" ".join(a.text(separator=" ") for a in links).split()))
    return linked / total >= _NAV_LINK_TEXT_RATIO

# Where the page's own content lives, most specific first.
_CONTENT_ROOTS = ("article", "[role=main]", "main", ".entry-content", "#content")


def page_html_to_text(html: str) -> str:
    """Extract one full web page's own content, without the surrounding site chrome."""
    if not html:
        return ""

    tree = HTMLParser(html)
    _unhide_revealable(tree)
    for tag in _DROP_TAGS:
        for node in tree.css(tag):
            node.decompose()
    for node in tree.css("-comment"):
        node.decompose()
    for selector in _CHROME_SELECTORS:
        for node in tree.css(selector):
            node.decompose()
    for selector in _AMBIGUOUS_LIST_SELECTORS:
        for node in tree.css(selector):
            if _looks_like_navigation(node):
                node.decompose()
    for node in tree.css("[style]"):
        if _HIDDEN_CSS.search(node.attributes.get("style") or ""):
            node.decompose()
    for node in tree.css("[hidden], [aria-hidden=true]"):
        node.decompose()

    root = None
    for selector in _CONTENT_ROOTS:
        root = tree.css_first(selector)
        if root is not None:
            break
    root = root or tree.body or tree.root
    if root is None:
        return ""

    text = clean_text(root.text(separator="\n"))

    # Elementor repeats each section heading as an in-content tab label, so the
    # same line often lands twice in a row. Collapse repeats to keep chunks
    # dense with information rather than with menu echoes.
    seen: set[str] = set()
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(stripped)

    return "\n".join(lines)[:MAX_DOC_CHARS]


# Widget furniture that sits inside a directory entry but says nothing about the
# person, plus the mojibake left by the site's broken icon-font characters.
_DIRECTORY_NOISE = re.compile(r"^(close|read more|view profile|x|\W*)$", re.I)


def extract_directory(html: str) -> list[str]:
    """Pull the faculty directory out of a page's hidden Popup Maker markup.

    The plugin stamps the same 226-entry staff list into every page. It is real,
    answerable content — "who teaches psychology", "what are the lecturer's
    qualifications" — but repeated 145 times it is indistinguishable from site
    chrome, and the frequency filter downstream treats it as such.

    So it is lifted out exactly once, here, and indexed as a single document.
    Each entry becomes one line pairing the name with the credentials that
    follow it, because the two live in different elements and split apart they
    are both useless: a bare name retrieves nothing, and "Assistant Professor,
    M.Com, 1.4 years" belongs to nobody.

    Returns one string per person, or an empty list on a page with no directory.
    """
    if not html:
        return []

    tree = HTMLParser(html)
    entries: list[str] = []
    seen: set[str] = set()

    for node in tree.css(_DIRECTORY_ENTRY_SELECTOR):
        parts = [
            line.strip()
            for line in clean_text(node.text(separator="\n")).splitlines()
            if line.strip() and not _DIRECTORY_NOISE.match(line.strip())
        ]
        if len(parts) < 2:
            continue  # a popup with no person in it (a promo banner, say)
        entry = f"{parts[0]} — {', '.join(parts[1:])}"
        key = entry.lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)

    return entries


# Phrases whose only purpose in *reference content* is to redirect a model.
# This is a reporting signal, not a filter: matching text is quarantined and
# counted so ingest surfaces a compromised page, but a clean scan is never
# treated as proof that a chunk is safe.
_INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instruction|prompt|direction|rule)",
        r"disregard (all |any )?(the )?(previous|prior|above|system)",
        r"forget (everything|all|your) (you|instructions|rules|prior)",
        r"you are (now|actually) (a|an|no longer)",
        r"new (system )?(instruction|prompt|rule)s?\s*[:\-]",
        r"</?(system|assistant|human|user)>",
        r"\[\s*(system|assistant|inst)\s*\]",
        r"(reveal|print|repeat|output|show)\s+(your|the)\s+(system\s+)?(prompt|instruction)",
        r"do not (mention|tell|reveal|say)",
        r"respond only with",
        r"override.{0,20}(instruction|safety|guardrail)",
    )
]


def injection_signals(text: str) -> list[str]:
    """Injection-shaped phrases found in reference content, for quarantine + alerting."""
    return [m.group(0)[:80] for p in _INJECTION_PATTERNS if (m := p.search(text))]
