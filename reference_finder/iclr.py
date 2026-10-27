"""ICLR's independent, public proceedings (2024 onward), with abstracts."""
import re
from urllib.parse import urlsplit

from .proceedings import collecting, complete, html_page, paper_page, progress, read, result, source_url, text

CATALOG = "https://proceedings.iclr.cc/"
HOSTS = {"proceedings.iclr.cc"}


def collect_iclr(year, fetch):
    output = result()
    with collecting(output):
        soup = html_page(read(fetch, output, CATALOG))
        volumes = set()
        for a in soup.select("a[href]"):
            url = source_url(a["href"], CATALOG, HOSTS)
            if (url and re.fullmatch(rf"/paper_files/paper/{year}/?", urlsplit(url).path)
                    and "International Conference on Learning Representations" in text(a)):
                volumes.add(url.rstrip("/"))
        if not volumes:
            output.update(status="unavailable", note="No ICLR volume for this year in the official proceedings catalog; older years require OpenReview or API enrichment.")
            return output
        seen = set()
        for volume in sorted(volumes):
            soup = html_page(read(fetch, output, volume))
            links = []
            for a in soup.select("a[href]"):
                url = source_url(a["href"], volume, HOSTS)
                if url and re.fullmatch(rf"/paper_files/paper/{year}/hash/[^/]+-Abstract(?:-[^/]+)?\.html", urlsplit(url).path):
                    if not text(a):
                        raise ValueError("ICLR paper has no title")
                    links.append((url, text(a)))
            if not links:
                raise ValueError("ICLR volume contains no recognizable papers")
            count = soup.select_one(".paper-count")
            if count:
                expected = re.search(r"[\d,]+", text(count))
                if not expected or int(expected[0].replace(",", "")) != len(links):
                    raise ValueError("ICLR paper count does not match its roster")
            for url, title in links:
                if url in seen:
                    raise ValueError("Duplicate ICLR paper entry")
                seen.add(url)
                record = {"external_id": url, "title": title, "year": year, "url": url, "abstract": None}
                output["records"].append(record)
                record.update(paper_page(read(fetch, output, url), url, year,
                                         ".paper-abstract, #abstract, .abstract"))
                progress(output)
        complete(output)
    return output
