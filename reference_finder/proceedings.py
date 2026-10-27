"""Shared parsing and partial-result handling for official HTML proceedings."""
from __future__ import annotations

from contextlib import contextmanager
import logging
import re
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from .api import APIError, RequestLimit
from .storage import normalize_doi


def text(node):
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def html_page(raw):
    if not isinstance(raw, str) or not re.search(r"</html\s*>\s*$", raw, re.I):
        raise ValueError("Empty or truncated HTML document")
    soup = BeautifulSoup(raw, "html.parser")
    if (re.search(r"access denied|captcha|just a moment|not found", text(soup.title), re.I)
            and not soup.select_one('meta[name="citation_title"]')):
        raise ValueError("Source returned an error or access challenge")
    return soup


def source_url(href, base, hosts):
    parts = urlsplit(urljoin(base, href))
    if parts.scheme not in {"http", "https"} or parts.hostname not in hosts:
        return None
    return urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))


def metadata(soup, name):
    return [tag.get("content", "").strip() for tag in soup.select(f'meta[name="{name}"]')
            if tag.get("content", "").strip()]


def paper_page(raw, url, year, abstract_selectors):
    soup = html_page(raw)
    titles = metadata(soup, "citation_title")
    abstract = soup.select_one(abstract_selectors)
    # Older NeurIPS pages place the abstract in a div following the heading.
    if abstract is None:
        heading = next((h for h in soup.select("h2,h3,h4") if text(h).casefold() == "abstract"), None)
        abstract = heading.find_next_sibling(["p", "div"]) if heading else None
    if abstract:
        for heading in abstract.select("h2,h3,h4"):
            if text(heading).casefold() == "abstract":
                heading.decompose()
    title = titles[0] if titles else ""
    if not title and abstract and (metadata(soup, "citation_author") or metadata(soup, "citation_pdf_url")):
        # Some PMLR pages interpolate unescaped quotes into citation_title,
        # leaving an empty content attribute. The article h1 is intact.
        title = text(soup.select_one("h1"))
    if not title:
        raise ValueError("Paper page is missing citation_title and a verified article heading: " + url)
    dois = metadata(soup, "citation_doi")
    return {"external_id": url, "title": title, "authors": metadata(soup, "citation_author"),
            "year": year, "url": url, "doi": normalize_doi(dois[0]) if dois else None,
            "abstract": text(abstract) or None}


def result():
    return {"records": [], "urls": [], "status": "incomplete", "note": ""}


def progress(output):
    count = len(output["records"])
    if count == 1 or count % 25 == 0:
        abstracts = sum(bool(r.get("abstract")) for r in output["records"])
        logging.getLogger(__name__).info("Official-source progress: %d papers read, %d abstracts (%s)",
                                        count, abstracts, output["records"][-1]["url"])


@contextmanager
def collecting(output):
    """Keep parsed records on source failures; let the CLI persist bounded runs."""
    try:
        yield
    except RequestLimit as exc:
        output["note"] = str(exc)
        exc.partial_result = output
        raise
    except (APIError, ValueError, OSError) as exc:
        output["status"] = "incomplete"
        output["note"] = str(exc)


def read(fetch, output, url):
    if url not in output["urls"]:
        output["urls"].append(url)
    return fetch(url)


def complete(output):
    output["status"] = "complete" if output["records"] else "unavailable"
    output["note"] = (f"Read {len(output['records'])} papers from the official source; "
                      "source retrieval does not establish that current-year publication is final.")
