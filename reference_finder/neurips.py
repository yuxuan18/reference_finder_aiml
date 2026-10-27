"""NeurIPS main proceedings, including historical NIPS abstract-page layouts."""
import re
from urllib.parse import urlsplit

from .proceedings import collecting, complete, html_page, paper_page, progress, read, result, source_url, text

CATALOG = "https://proceedings.neurips.cc/"
HOSTS = {"proceedings.neurips.cc", "papers.nips.cc"}


def discover_volumes(raw, year):
    soup = html_page(raw)
    volumes = set()
    for a in soup.select("a[href]"):
        url = source_url(a["href"], CATALOG, HOSTS)
        if not url or not re.fullmatch(rf"/(?:paper_files/paper|paper|papers)/{year}(?:/vol[^/]+)?/?", urlsplit(url).path):
            continue
        label = text(a)
        if "Neural Information Processing Systems" in label and not re.search(
                r"creative|workshop|datasets and benchmarks", label, re.I):
            volumes.add(url.rstrip("/"))
    return sorted(volumes)


def collect_neurips(year, fetch):
    output = result()
    with collecting(output):
        volumes = discover_volumes(read(fetch, output, CATALOG), year)
        if not volumes:
            output.update(status="unavailable", note="No main NeurIPS proceedings for this year in the official catalog.")
            return output
        seen = set()
        for volume in volumes:
            soup = html_page(read(fetch, output, volume))
            links = []
            for a in soup.select('a[href]'):
                url = source_url(a["href"], volume, HOSTS)
                if url and re.fullmatch(rf"/(?:paper_files/paper|paper)/{year}/hash/[^/]+-Abstract(?:-[^/]+)?\.html", urlsplit(url).path):
                    if not text(a):
                        raise ValueError("NeurIPS paper has no title")
                    links.append((url, text(a)))
            if not links:
                raise ValueError("NeurIPS volume contains no recognizable paper entries")
            count = soup.select_one(".paper-count")
            if count:
                expected = re.search(r"[\d,]+", text(count))
                if not expected or int(expected[0].replace(",", "")) != len(links):
                    raise ValueError("NeurIPS paper count does not match its roster")
            for url, title in links:
                if url in seen:
                    raise ValueError("Duplicate NeurIPS paper entry")
                seen.add(url)
                record = {"external_id": url, "title": title, "year": year, "url": url, "abstract": None}
                output["records"].append(record)
                record.update(paper_page(read(fetch, output, url), url, year,
                                         ".paper-abstract, #abstract, .abstract"))
                progress(output)
        complete(output)
    return output
