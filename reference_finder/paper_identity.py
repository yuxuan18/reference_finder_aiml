"""Stable official publication identifiers; never infer identity from titles."""
import re
from urllib.parse import parse_qs, urlsplit


def paper_identity(url):
    if not isinstance(url, str):
        return None
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        return None
    host = (parts.hostname or "").removeprefix("www.")
    if host == "openreview.net" and parts.path.rstrip("/") == "/forum":
        ids = parse_qs(parts.query).get("id", [])
        if len(ids) == 1 and re.fullmatch(r"[\w-]+", ids[0]):
            return "openreview:" + ids[0]
    if host == "proceedings.mlr.press" and re.fullmatch(r"/v\d+/[^/]+\.html", parts.path):
        return "pmlr:" + parts.path
    if host in {"proceedings.neurips.cc", "papers.nips.cc"}:
        match = re.fullmatch(r"/(?:paper_files/paper|paper)/(?P<year>\d{4})/hash/(?P<id>[0-9a-f]{32})-Abstract(?:-[^/]+)?\.html", parts.path, re.I)
        if match:
            return "neurips:" + match["year"] + ":" + match["id"].lower()
    if host == "proceedings.iclr.cc":
        match = re.fullmatch(r"/paper_files/paper/(?P<year>\d{4})/hash/(?P<id>[0-9a-f]{32})-Abstract(?:-[^/]+)?\.html", parts.path, re.I)
        if match:
            return "iclr:" + match["year"] + ":" + match["id"].lower()
    if host == "ojs.aaai.org":
        match = re.fullmatch(r"/index.php/AAAI/article/view/(\d+)/?", parts.path)
        if match:
            return "aaai:" + match[1]
    return None


def aaai_article_id(url=None, doi=None):
    identity = paper_identity(url)
    if identity and identity.startswith("aaai:"):
        return identity.split(":", 1)[1]
    if isinstance(doi, str):
        match = re.fullmatch(r"10\.1609/aaai\.v(\d+)i\d+\.(\d+)", doi, re.I)
        # Volume 33 (2019) encodes volume/issue/page in its DOI suffix,
        # rather than the OJS article ID. It requires explicit DOI metadata.
        if match and int(match[1]) != 33:
            return match[2]
    return None
