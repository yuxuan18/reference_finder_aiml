"""Official COLM accepted-paper lists, independent of the OpenReview API.

These pages establish acceptance and authorship but do not supply abstracts.
"""
from urllib.parse import parse_qs, urlsplit

from .proceedings import collecting, complete, html_page, read, result, source_url, text


def collect_colm(year, fetch):
    output = result()
    if year < 2024:
        output.update(status="not_applicable", note="COLM began in 2024.")
        return output
    url = f"https://colmweb.org/{year}/AcceptedPapers.html"
    with collecting(output):
        soup = html_page(read(fetch, output, url))
        if not any(text(h).casefold() == "accepted papers" for h in soup.select("h1,h2,h3")):
            raise ValueError("COLM page does not identify an accepted-paper list")
        records = {}
        for a in soup.select("a[href]"):
            target = source_url(a["href"], url, {"openreview.net"})
            if not target or urlsplit(target).path != "/forum":
                continue
            ids = parse_qs(urlsplit(target).query).get("id", [])
            if len(ids) != 1 or not ids[0] or not text(a):
                raise ValueError("Malformed COLM accepted-paper link")
            external_id = ids[0]
            parent = a.find_parent("p")
            authors = text(parent.select_one("em")) if parent else ""
            record = {"external_id": external_id, "title": text(a), "year": year,
                      "url": target, "authors": [name.strip() for name in authors.split(",") if name.strip()],
                      "abstract": None}
            if external_id in records and records[external_id]["title"] != record["title"]:
                raise ValueError("Conflicting titles for a COLM accepted-paper ID")
            records[external_id] = record
        output["records"] = list(records.values())
        if not records:
            raise ValueError("COLM page contains no recognizable accepted papers")
        complete(output)
        output["note"] += " This is a roster-only source; abstracts require OpenReview or API enrichment."
    return output
