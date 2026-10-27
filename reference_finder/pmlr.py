"""ICML main proceedings and abstracts from PMLR's public HTML catalog."""
import re
from urllib.parse import urlsplit

from .proceedings import collecting, complete, html_page, paper_page, progress, read, result, source_url, text

CATALOG = "https://proceedings.mlr.press/"
HOSTS = {"proceedings.mlr.press"}


def discover_volumes(raw, year):
    soup = html_page(raw)
    volumes = set()
    for item in soup.select("li"):
        label = re.sub(r"^Volume\s+\d+\s*", "", text(item), flags=re.I)
        # Exact catalog descriptions avoid ICML-affiliated workshop volumes.
        if not re.fullmatch(rf"Proceedings of (?:ICML|(?:the \d+(?:st|nd|rd|th) )?International Conference on Machine Learning) {year}", label, re.I):
            continue
        for a in item.select("a[href]"):
            url = source_url(a["href"], CATALOG, HOSTS)
            if url and re.fullmatch(r"/v\d+/?", urlsplit(url).path):
                volumes.add(url.rstrip("/") + "/")
    return sorted(volumes)


def collect_pmlr(year, fetch):
    output = result()
    with collecting(output):
        volumes = discover_volumes(read(fetch, output, CATALOG), year)
        if not volumes:
            output.update(status="unavailable", note="No ICML volume for this year in the PMLR catalog.")
            return output
        seen = set()
        for volume in volumes:
            soup = html_page(read(fetch, output, volume))
            heading = text(soup.select_one("h2"))
            if "International Conference on Machine Learning" not in heading or str(year) not in heading:
                raise ValueError("PMLR volume does not identify the requested ICML year")
            nodes = soup.select("div.paper")
            if not nodes:
                raise ValueError("PMLR volume contains no recognizable paper entries")
            for node in nodes:
                links = [source_url(a["href"], volume, HOSTS) for a in node.select("a[href]")
                         if text(a).casefold() == "abs"]
                if len(links) != 1 or not links[0] or not text(node.select_one(".title")):
                    raise ValueError("Malformed PMLR paper entry")
                url = links[0]
                if url in seen:
                    raise ValueError("Duplicate PMLR paper entry")
                seen.add(url)
                record = {"external_id": url, "title": text(node.select_one(".title")),
                          "year": year, "url": url, "abstract": None,
                          "authors": [a.strip() for a in text(node.select_one(".authors")).split(",") if a.strip()]}
                output["records"].append(record)
                parsed = paper_page(read(fetch, output, url), url, year, "#abstract, .abstract")
                record.update(parsed)
                progress(output)
        complete(output)
    return output
