"""AAAI OJS archives, including the migrated 2017/2018 proceedings.

The annual volumes mix AAAI with IAAI/EAAI and ancillary programs. Membership
comes from explicit AAAI research section headings, not article title keywords.
"""
import re

from .proceedings import collecting, complete, html_page, paper_page, progress, read, result, source_url, text

ARCHIVE = "https://ojs.aaai.org/index.php/AAAI/issue/archive"
HOSTS = {"ojs.aaai.org"}
EXCLUDED = re.compile(r"\b(?:IAAI\d*|EAAI\d*|workshops?|demos?|demonstrations?|doctoral|student abstracts?|"
                      r"posters?|tutorials?|preface|front matter|invited|senior (?:member|track))\b|"
                      r"innovative applications|educational advances|model ai assignments|what.?s hot abstracts|"
                      r"undergraduate consortium|new faculty|journal track|emerging trends|student papers|"
                      r"sister conference|^Errata$", re.I)
LEGACY_RESEARCH_SECTIONS = {
    "cognitive modeling and cognitive systems", "human-aware artificial intelligence",
    "machine learning methods",
    "computational sustainability and artificial intelligence", "intelligent robots (rob)",
}


def issue_year(label):
    match = re.search(r"\bVol\.?\s*(\d+)", label, re.I)
    if match:
        # Annual OJS volume 31 is AAAI 2017, volume 40 is AAAI 2026.
        return int(match[1]) + 1986
    match = re.search(r"\((20\d\d)\)", label)
    return int(match[1]) if match else None


def discover_issues(year, fetch, output):
    url, seen, issues = ARCHIVE, set(), set()
    while url:
        if url in seen:
            raise ValueError("AAAI archive repeated a pagination link")
        seen.add(url)
        soup = html_page(read(fetch, output, url))
        cards = soup.select(".obj_issue_summary")
        if not cards:
            raise ValueError("AAAI archive contains no recognizable issue entries")
        for card in cards:
            title = card.select_one(".title")
            if issue_year(text(card.select_one(".series")) or text(card)) != year:
                continue
            link = title if title and title.name == "a" else card.select_one('a[href*="/issue/view/"]')
            target = source_url(link["href"], url, HOSTS) if link else None
            if not target:
                raise ValueError("Malformed AAAI issue link")
            issues.add(target)
        next_link = soup.select_one('a.next[href], a[rel="next"][href]')
        url = source_url(next_link["href"], url, HOSTS) if next_link else None
        if next_link and not url:
            raise ValueError("Unsupported AAAI archive pagination URL")
    return sorted(issues)


def classify_section(label, issue_label=""):
    """Classify official section headings, preserving unknowns for review."""
    if EXCLUDED.search(label):
        return "excluded"
    if label.casefold() == "articles":
        return "research" if re.search(r"AAAI.*Technical", issue_label, re.I) else "unknown"
    if (re.search(r"AAAI.*(?:technical|special|track)|technical papers|^(?:main|special) track", label, re.I)
            or label.casefold() in LEGACY_RESEARCH_SECTIONS):
        return "research"
    return "unknown"


def issue_entries(raw, issue):
    soup = html_page(raw)
    sections = soup.select(".section")
    if not sections:
        raise ValueError("AAAI issue has no recognizable sections")
    issue_label = text(soup.select_one(".current_issue_title, h1")) + " " + text(soup.title)
    for section in sections:
        label = text(section.select_one("h2,h3"))
        classification = classify_section(label, issue_label)
        for paper in section.select(".obj_article_summary"):
            link = paper.select_one("h3 a[href], .title a[href]")
            url = source_url(link["href"], issue, HOSTS) if link else None
            if not url or not text(link) or "/AAAI/article/view/" not in url:
                raise ValueError("Malformed AAAI paper entry")
            yield dict(url=url, title=text(link), section=label, classification=classification,
                       evidence_url=issue)


def collect_aaai(year, fetch):
    output = result()
    with collecting(output):
        issues = discover_issues(year, fetch, output)
        if not issues:
            output.update(status="unavailable", note="No AAAI research issue for this year in the official OJS archive.")
            return output
        seen, uncertain = set(), 0
        for issue in issues:
            for entry in issue_entries(read(fetch, output, issue), issue):
                label, url = entry["section"], entry["url"]
                if entry["classification"] == "excluded":
                    continue
                if entry["classification"] != "research":
                    uncertain += 1
                    continue
                if url in seen:
                    raise ValueError("Duplicate AAAI paper entry")
                seen.add(url)
                record = {"external_id": url, "title": entry["title"], "year": year,
                          "url": url, "abstract": None, "section": label}
                output["records"].append(record)
                record.update(paper_page(read(fetch, output, url), url, year,
                                         "section.abstract, .item.abstract"))
                progress(output)
        complete(output)
        if uncertain:
            output.update(status="incomplete", note=f"Excluded {uncertain} entries in unrecognized AAAI sections; review required.")
    return output
