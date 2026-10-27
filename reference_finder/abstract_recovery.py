"""Recover abstracts for existing roster members without replacing source scans."""
import html
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import parse_qs, urlencode, urlsplit

from .api import APIError, OA_FIELDS, RequestLimit
from .openreview import acceptance, scope, value
from .paper_identity import aaai_article_id
from .proceedings import paper_page
from .storage import normalize_doi, normalize_title, reconstruct_abstract


class ScopeReviewRequired(ValueError):
    pass


def author_key(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\s+\d{4}$", "", name)
    if "," in name:
        family, given = name.split(",", 1)
        name = given + " " + family
    return "".join(c for c in unicodedata.normalize("NFKD", name).casefold() if c.isalnum())


def api_match(paper, record):
    """A title-search result needs corroborating authors, year and DOI."""
    title = record.get("display_name") or record.get("title") or ""
    if normalize_title(html.unescape(title)) != normalize_title(html.unescape(paper["title"])):
        return False
    if not isinstance(record.get("publication_year"), int) or record["publication_year"] not in {paper["event_year"], paper["publication_year"]}:
        return False
    old_doi, new_doi = normalize_doi(paper.get("doi")), normalize_doi(record.get("doi"))
    if old_doi and new_doi and old_doi != new_doi:
        return False
    known = {author_key(a) for a in json.loads(paper["authors_json"]) if author_key(a)}
    found = {author_key(a.get("author", {}).get("display_name", ""))
             for a in record.get("authorships") or []}
    found.discard("")
    return bool(known and found and len(known & found) / max(len(known), len(found)) >= 0.6)


def official_record(paper, fetch):
    url = paper.get("url") or ""
    if paper["venue"] == "aaai":
        article = aaai_article_id(url, paper.get("doi"))
        if article:
            url = "https://ojs.aaai.org/index.php/AAAI/article/view/" + article
    url = url.replace("http://", "https://").replace("papers.nips.cc", "proceedings.neurips.cc")
    parts = urlsplit(url)
    if parts.hostname == "openreview.net" and parts.path == "/forum":
        ids = parse_qs(parts.query).get("id", [])
        if len(ids) != 1:
            raise ValueError("Missing OpenReview paper identifier")
        base, group, version = scope(paper["venue"], paper["event_year"])
        evidence = base + "/notes?" + urlencode(dict(id=ids[0], details="directReplies"))
        notes = json.loads(fetch(evidence))["notes"]
        if len(notes) != 1 or ids[0] not in (notes[0].get("id"), notes[0].get("forum")):
            raise ValueError("OpenReview returned a different note")
        note = notes[0]
        if acceptance(note, group, version) is False:
            raise ScopeReviewRequired("OpenReview marks this submission as not accepted; scope review required")
        content = note.get("content") or {}
        abstract = value(content, "abstract")
        return "openreview", dict(external_id=note["id"], url=url, doi=value(content, "doi"),
                                  abstract=abstract if isinstance(abstract, str) else None), evidence
    provider = {"proceedings.iclr.cc": "iclr", "proceedings.mlr.press": "pmlr",
                "proceedings.neurips.cc": "neurips", "ojs.aaai.org": "aaai"}.get(parts.hostname)
    if not provider:
        raise ValueError("No supported official paper URL")
    record = paper_page(fetch(url), url, paper["event_year"],
                        ".paper-abstract,#abstract,.abstract,section.abstract,.item.abstract")
    return provider, record, url


def recover_missing(db, fetch, openalex=None, apply=False, snapshots=None):
    """Enrich existing IDs only; preserve roster entries and source-run status."""
    if snapshots is None:
        snapshots = json.loads(Path(__file__).with_name("abstract_snapshots.json").read_text())
    snapshots = {s["dblp_key"]: s for s in snapshots}
    rows = [dict(r) for r in db.conn.execute("""SELECT p.*,r.venue,r.year AS event_year
        FROM active_roster_entries r JOIN papers p ON p.id=r.paper_id
        WHERE p.abstract IS NULL GROUP BY p.id,r.venue,r.year ORDER BY p.id""")]
    results = []
    for paper in rows:
        result = dict(paper_id=paper["id"], venue=paper["venue"], year=paper["event_year"],
                      title=paper["title"], status="missing", attempts=[])
        try:
            keys = [r[0] for r in db.conn.execute("SELECT external_id FROM source_records WHERE provider='dblp' AND paper_id=?", (paper["id"],))]
            snapshot = next((snapshots[key] for key in keys if key in snapshots), None)
            provider, record, evidence = None, None, None
            if snapshot:
                provider, record, evidence = snapshot["provider"], dict(snapshot), snapshot["evidence_url"]
            else:
                try:
                    provider, record, evidence = official_record(paper, fetch)
                    result["attempts"].append(dict(provider=provider, url=evidence,
                                                   has_abstract=bool(record.get("abstract"))))
                except ScopeReviewRequired:
                    raise
                except (APIError, ValueError, OSError) as exc:
                    result["attempts"].append(dict(provider="official", error=str(exc)))
            if not (record or {}).get("abstract") and openalex:
                response = openalex.http.request("/works", dict(search=paper["title"].rstrip("."),
                                                                per_page=5, select=OA_FIELDS))
                matches = [r for r in response["results"] if api_match(paper, r) and reconstruct_abstract(r.get("abstract_inverted_index"))]
                if len(matches) == 1:
                    source = matches[0]
                    provider, evidence = "openalex", source["id"]
                    record = dict(external_id=source["id"], url=source["id"],
                                  abstract=reconstruct_abstract(source.get("abstract_inverted_index")))
                    result["attempts"].append(dict(provider=provider, url=evidence, has_abstract=True))
                elif len(matches) > 1:
                    raise ValueError("Multiple API records match; reviewed identity required")
            if record and record.get("abstract"):
                doi = normalize_doi(record.get("doi"))
                if paper["doi"] and doi and paper["doi"] != doi:
                    raise ValueError("Official abstract page has a conflicting DOI")
                existing = db.conn.execute("SELECT paper_id FROM source_records WHERE provider=? AND external_id=?",
                                           (provider, record["external_id"])).fetchone()
                if existing and existing[0] != paper["id"]:
                    raise ValueError("Abstract source is already linked to another paper; reviewed alias required")
                # This is abstract enrichment, not a replacement bibliography.
                # Keep the existing title, author list, DOI and publication year.
                canonical = dict(external_id=record["external_id"], title=paper["title"],
                    authors=json.loads(paper["authors_json"]), year=paper["publication_year"],
                    doi=paper["doi"], url=record.get("url"), abstract=record["abstract"].strip())
                if apply:
                    with db.conn:
                        current = db.conn.execute("SELECT abstract FROM papers WHERE id=?", (paper["id"],)).fetchone()
                        if not current or current[0]:
                            result["status"] = "already updated"
                            results.append(result)
                            continue
                        # upsert's official-record shape also works for API text
                        # after reconstruction, without mislabeling the provider.
                        if provider == "openalex":
                            source = dict(source, display_name=paper["title"], doi=paper["doi"],
                                          publication_year=paper["publication_year"], authorships=[
                                              {"author": {"display_name": a}} for a in canonical["authors"]])
                            db.upsert(provider, source, known_paper_id=paper["id"])
                        else:
                            db.upsert(provider, canonical, known_paper_id=paper["id"])
                result.update(status="recovered" if apply else "available", provider=provider,
                              evidence_url=evidence, abstract_characters=len(canonical["abstract"]))
        except RequestLimit:
            result["status"] = "request limit"
            results.append(result)
            break
        except (APIError, ValueError, OSError) as exc:
            result.update(status="review required", error=str(exc))
        results.append(result)
    return results
