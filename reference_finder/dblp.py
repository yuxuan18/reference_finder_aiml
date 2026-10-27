"""Enumerate main-conference papers from DBLP's public SPARQL API.

DBLP's HTML site currently blocks automated access.  Online collection therefore
uses its documented public API, not a browser challenge or mirror workaround.
The BeautifulSoup helpers below also accept HTML saved by a user.  "complete"
means all records in DBLP's indexed main proceedings were read and counted; it
does not certify that DBLP has indexed every accepted paper for an ongoing year.

API documentation: https://blog.dblp.org/2024/09/09/introducing-our-public-sparql-query-service/
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlencode, urljoin, urlsplit

from bs4 import BeautifulSoup

from .api import RequestLimit
from .storage import normalize_doi


STREAMS = {
    "iclr": "iclr", "icml": "icml", "neurips": "nips", "aaai": "aaai",
}
SPARQL_ENDPOINT = "https://sparql.dblp.org/sparql"
PAGE_SIZE = 500
_PREFIX = "PREFIX dblp: <https://dblp.org/rdf/schema#>\n"
_REC = "https://dblp.org/rec/"
_SEPARATOR = "\x1f"


def _scope(venue_key, year):
    if venue_key not in STREAMS:
        raise ValueError("Unknown conference: " + str(venue_key))
    if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2100:
        raise ValueError("year must be an integer between 1900 and 2100")
    stream = STREAMS[venue_key]
    # The terminal anchor excludes 2024w, 2024cset, companions and other events.
    return stream, rf"^https://dblp[.]org/rec/conf/{stream}/{year}(-[0-9]+)?$"


def _query_url(query):
    # `format=json` works without a caller-specific Accept header.
    return SPARQL_ENDPOINT + "?" + urlencode({"query": query, "format": "json"})


def _bindings(raw):
    payload = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    if not isinstance(payload, dict) or payload.get("exception") or payload.get("error"):
        raise ValueError("DBLP API did not return a successful result")
    rows = payload.get("results", {}).get("bindings")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("DBLP API response is missing result bindings")
    return rows


def _value(row, name, default=""):
    value = row.get(name, {})
    if not isinstance(value, dict) or not isinstance(value.get("value", default), str):
        raise ValueError("Invalid DBLP binding for " + name)
    return value.get("value", default)


def _books_query(stream, pattern):
    return _PREFIX + """SELECT ?book (SAMPLE(?bookTitle) AS ?title)
       (SAMPLE(?tocPage) AS ?toc) (COUNT(DISTINCT ?paper) AS ?count)
WHERE {
  ?book dblp:bibtexType <http://purl.org/net/nknouf/ns/bibtex#Proceedings> ;
        dblp:publishedInStream <https://dblp.org/streams/conf/%s> .
  FILTER(REGEX(STR(?book), %s))
  OPTIONAL { ?book dblp:title ?bookTitle . }
  OPTIONAL { ?book dblp:listedOnTocPage ?tocPage . }
  OPTIONAL { ?paper a dblp:Inproceedings ; dblp:publishedAsPartOf ?book . }
} GROUP BY ?book ORDER BY ?book""" % (stream, json.dumps(pattern))


def _papers_query(books, offset):
    values = " ".join("<" + book + ">" for book in books)
    return _PREFIX + """SELECT ?paper (SAMPLE(?paperTitle) AS ?title)
       (SAMPLE(?paperDoi) AS ?doi) (SAMPLE(?document) AS ?url)
       (SAMPLE(?paperYear) AS ?publicationYear)
       (GROUP_CONCAT(DISTINCT CONCAT(STR(?authorOrdinal), ":", ?authorName); separator="\\u001f") AS ?authors)
       (GROUP_CONCAT(DISTINCT STR(?book); separator="\\u001f") AS ?books)
WHERE {
  VALUES ?book { %s }
  ?paper a dblp:Inproceedings ; dblp:publishedAsPartOf ?book ;
         dblp:title ?paperTitle .
  OPTIONAL { ?paper dblp:doi ?paperDoi . }
  OPTIONAL { ?paper dblp:primaryDocumentPage ?document . }
  OPTIONAL { ?paper dblp:yearOfPublication ?paperYear . }
  OPTIONAL {
    ?paper dblp:hasSignature ?signature .
    ?signature a dblp:AuthorSignature ; dblp:signatureOrdinal ?authorOrdinal ;
               dblp:signatureDblpName ?authorName .
  }
} GROUP BY ?paper ORDER BY ?paper LIMIT %d OFFSET %d""" % (values, PAGE_SIZE, offset)


def _ordered_authors(value):
    authors = []
    for item in filter(None, value.split(_SEPARATOR)):
        ordinal, separator, name = item.partition(":")
        if not separator or not ordinal.isdigit() or not name:
            raise ValueError("DBLP returned an invalid author signature")
        authors.append((int(ordinal), name))
    return [name for ordinal, name in sorted(authors)]


def collect_dblp(venue_key, year, fetch):
    """Return records, source URLs, status and note for an event's main roster.

    ``fetch(url)`` returns response text and must raise on HTTP errors. It is
    supplied by the runner so caching, request pacing and retries are shared.
    Publication years can differ from event years,
    so membership is determined by proceedings keys and the requested event year.
    """
    stream, pattern = _scope(venue_key, year)
    urls = []
    records = {}
    result = {"records": [], "urls": urls, "status": "incomplete", "note": ""}
    try:
        url = _query_url(_books_query(stream, pattern))
        urls.append(url)
        books = {}
        for row in _bindings(fetch(url)):
            book = _value(row, "book")
            if not re.fullmatch(pattern, book):
                raise ValueError("DBLP returned a proceedings key outside the requested scope")
            count = int(_value(row, "count"))
            if count < 0 or book in books:
                raise ValueError("DBLP returned invalid or duplicate proceedings counts")
            books[book] = count
            toc = _value(row, "toc")
            if toc:
                urls.append(toc + ("" if toc.endswith(".html") else ".html"))
        if not books:
            result.update(status="unavailable", note=(
                "No main proceedings are indexed in DBLP for this event year; "
                "this is unknown coverage, not a verified zero-paper conference."
            ))
            return result

        observed = {book: set() for book in books}
        expected_max = sum(books.values())
        offset = 0
        while offset < expected_max:
            url = _query_url(_papers_query(books, offset))
            urls.append(url)
            rows = _bindings(fetch(url))
            if not rows:
                break
            for row in rows:
                uri = _value(row, "paper")
                title = _value(row, "title").strip()
                prefixes = [_REC + "conf/" + stream + "/"]
                if not any(uri.startswith(prefix) for prefix in prefixes) or not title:
                    raise ValueError("DBLP returned a paper without a valid key/title")
                if uri in records:
                    raise ValueError("DBLP pagination repeated a paper; roster may have changed")
                parent_books = _value(row, "books").split(_SEPARATOR)
                if not parent_books or any(book not in books for book in parent_books):
                    raise ValueError("DBLP paper is missing its main proceedings membership")
                key = uri[len(_REC):]
                publication_year = _value(row, "publicationYear")
                paper = {
                    "external_id": key, "dblp_key": key, "title": title,
                    "year": year, "abstract": None,
                    "doi": normalize_doi(_value(row, "doi")),
                    "url": _value(row, "url") or uri,
                    "authors": _ordered_authors(_value(row, "authors")),
                    "proceedings": parent_books,
                }
                if publication_year.isdigit():
                    paper["publication_year"] = int(publication_year)
                records[uri] = paper
                for book in parent_books:
                    observed[book].add(uri)
            offset += len(rows)
            if len(rows) < PAGE_SIZE:
                break
        mismatches = [book for book in books if not books[book] or len(observed[book]) != books[book]]
        if mismatches:
            result["note"] = "Proceedings counts are incomplete: " + "; ".join(
                ("%s: volume has no indexed papers; cannot verify its coverage" % book.rsplit("/", 1)[-1])
                if not books[book] else
                ("%s: read %d of %d" % (book.rsplit("/", 1)[-1], len(observed[book]), books[book]))
                for book in mismatches
            )
        else:
            result.update(status="complete", note=(
                "Read all %d DBLP-indexed main-proceedings papers across %d volume(s); "
                "counts checked against the API. DBLP indexing itself may still be incomplete."
            ) % (len(records), len(books)))
    except RequestLimit:
        raise
    except Exception as exc:
        result["note"] = "DBLP collection incomplete: " + str(exc)
    result["records"] = list(records.values())
    return result


def _html_soup(html):
    """Reject saved challenges, error pages and visibly truncated documents."""
    if not isinstance(html, str) or not re.search(r"</html\s*>\s*$", html, re.I):
        raise ValueError("DBLP HTML is empty or truncated (missing closing html tag)")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True).casefold() if soup.title else ""
    if soup.find(id="anubis_challenge") or any(term in title for term in (
        "not a bot", "access denied", "oh noes", "captcha", "just a moment",
    )):
        raise ValueError("DBLP returned an access challenge, not conference records")
    return soup


def parse_dblp_index(venue_key, year, html, index_url=None):
    """Find all main annual TOC links in a user-saved DBLP series index."""
    stream, pattern = _scope(venue_key, year)
    soup = _html_soup(html)
    base = index_url or "https://dblp.org/db/conf/" + stream + "/index.html"
    urls = set()
    for entry in soup.select("li.entry"):
        key = entry.get("id", "")
        if not re.fullmatch(pattern, _REC + key):
            continue
        for link in entry.select("a[href]"):
            url = urljoin(base, link["href"])
            path = urlsplit(url).path
            if re.fullmatch(rf"/db/conf/{stream}/{stream}{year}(?:-[0-9]+)?[.]html", path):
                urls.add(url.split("#", 1)[0])
    if not soup.select("li.entry"):
        raise ValueError("Saved DBLP index contains no proceedings entries")
    return sorted(urls)


def parse_dblp_html(html, year, source_url):
    """Parse every inproceedings item in a user-saved main-conference TOC.

    Posters/keynotes appearing as individual main-proceedings papers are kept;
    section headings are retained for callers that want a narrower scope.
    Raises on empty, malformed or truncated pages instead of reporting success.
    """
    soup = _html_soup(html)
    entries = soup.select("li.entry.inproceedings")
    if not entries:
        raise ValueError("Saved DBLP TOC contains no inproceedings records")
    records = []
    seen = set()
    for entry in entries:
        key = entry.get("id", "")
        title = entry.select_one(".title")
        if not key.startswith("conf/") or title is None or not title.get_text(" ", strip=True):
            raise ValueError("DBLP TOC contains an incomplete paper entry")
        if key in seen:
            raise ValueError("DBLP TOC contains a duplicate paper key")
        seen.add(key)
        links = [urljoin(source_url, a["href"]) for a in entry.select("a[href]")]
        doi = next((normalize_doi(link) for link in links if normalize_doi(link)), None)
        electronic = entry.select_one("nav.publ li.ee a[href], nav.publ .head a[href], .publ a.ee[href], a[itemprop='url'][href]")
        publisher_url = urljoin(source_url, electronic["href"]) if electronic else None
        if not publisher_url or urlsplit(publisher_url).hostname in ("dblp.org", "dblp.uni-trier.de"):
            publisher_url = "https://doi.org/" + doi if doi else _REC + key
        authors = []
        for author in entry.select('[itemprop="author"]'):
            name = author.select_one('[itemprop="name"]')
            authors.append((name or author).get_text(" ", strip=True))
        heading = entry.find_previous(["h2", "h3", "h4"])
        records.append({
            "external_id": key, "dblp_key": key, "title": title.get_text(" ", strip=True),
            "year": year, "doi": doi, "authors": authors, "url": publisher_url,
            "abstract": None, "section": heading.get_text(" ", strip=True) if heading else None,
        })
    return records
