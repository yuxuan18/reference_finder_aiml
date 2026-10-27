"""Resumable conference discovery and DOI-based abstract enrichment."""
from __future__ import annotations

import json
import logging
import re

from .api import APIError
from .storage import normalize_doi
from .venues import is_excluded, source_matches

LOG = logging.getLogger(__name__)


def job_key(stage, venue, **kwargs):
    # Version 1 rejected real papers carrying Semantic Scholar's mixed Book tag.
    # Re-read those discovery pages once; existing records still deduplicate.
    version = 2 if stage == "s2-discover" else 1
    return json.dumps({"version": version, "stage": stage, "venue": venue, **kwargs}, sort_keys=True)


def is_paper(provider, record):
    """Exclude volume-level records and front matter, without title keyword search."""
    title = (record.get("title") or record.get("display_name") or "").strip()
    if not title:
        return False
    if provider == "openalex" and record.get("type") in ("book", "paratext", "editorial", "erratum"):
        return False
    types = set(record.get("publicationTypes") or [])
    if "Book" in types and not types & {"Conference", "JournalArticle"}:
        return False
    if types & {"Review", "Editorial", "Erratum"} and not types & {"Conference", "JournalArticle"}:
        return False
    if re.match(r"^proceedings\b", title, re.I):
        return False
    # S2 can label individual papers as Book+Conference. Reject explicit
    # event-volume titles rather than dropping every mixed-tag paper.
    if re.match(r"^[A-Z][A-Z0-9 &+.'’\-]{0,30}['’]?\d{2,4}\s*:\s*(?:proceedings\b|\d+(?:st|nd|rd|th)\s+(?:(?:annual|international|ACM|IEEE)\s+)*(?:workshop|conference|symposium)\b)", title, re.I):
        return False
    if re.fullmatch(r"(?:preface|foreword|front matter|back matter|table of contents|author index|organizing committee|program committee|copyright.*)", title, re.I):
        return False
    return True


def s2_matches(venue, record, year):
    if record.get("year") != year or not is_paper("semantic_scholar", record):
        return False
    publication_venue = record.get("publicationVenue") or {}
    labels = [record.get("venue"), publication_venue.get("name")]
    if any(is_excluded(label) for label in labels):
        return False
    # Alternate acronyms can collide with unrelated conferences in S2. Require
    # a matching displayed/canonical label, not merely an alternate-name hit.
    return any(venue.matches(label) for label in labels if label)


class Importer:
    def __init__(self, db, venues, start, end, openalex=None, semantic_scholar=None):
        self.db, self.venues, self.start, self.end = db, venues, start, end
        self.openalex, self.semantic_scholar = openalex, semantic_scholar

    def discover_s2(self):
        for venue in self.venues:
            for year in range(max(self.start, venue.first_year), self.end + 1):
                job = job_key("s2-discover", venue.key, year=year)
                state = self.db.checkpoint(job)
                if state and state["done"]:
                    continue
                token = state["cursor"] if state else None
                while True:
                    response = self.semantic_scholar.works(venue.semantic_scholar_names, year, token)
                    records = response.get("data")
                    if not isinstance(records, list):
                        raise APIError("Semantic Scholar returned no data array; checkpoint retained.")
                    following = response.get("token")
                    if following and following == token:
                        raise APIError("Semantic Scholar repeated its pagination token; checkpoint retained.")
                    accepted = 0
                    with self.db.conn:
                        for record in records:
                            if s2_matches(venue, record, year):
                                self.db.upsert("semantic_scholar", record, venue.key, year)
                                accepted += 1
                        self.db.save_checkpoint(job, following, not following, len(records), accepted)
                    LOG.info("S2 %-15s %s: %s/%s records accepted", venue.key, year, accepted, len(records))
                    if len(records) >= 10 and accepted < len(records) * 0.2:
                        LOG.warning("S2 %s %s: over 80%% of results were rejected; check venue/type metadata before treating this as coverage", venue.key, year)
                    if not following:
                        break
                    token = following

    def discover_sources(self, venue):
        for query in venue.openalex_queries:
            job = job_key("oa-sources", venue.key, query=query)
            state = self.db.checkpoint(job)
            if state and state["done"]:
                continue
            cursor = state["cursor"] if state else "*"
            while True:
                response = self.openalex.sources(query, cursor)
                records = response.get("results")
                if not isinstance(records, list):
                    raise APIError("OpenAlex returned no source results array; checkpoint retained.")
                following = (response.get("meta") or {}).get("next_cursor")
                if following and following == cursor:
                    raise APIError("OpenAlex repeated its source cursor; checkpoint retained.")
                accepted = 0
                with self.db.conn:
                    for source in records:
                        if source_matches(venue, source):
                            self.db.conn.execute("INSERT OR REPLACE INTO openalex_sources VALUES (?,?,?)",
                                                 (venue.key, source["id"], source["display_name"]))
                            accepted += 1
                    self.db.save_checkpoint(job, following, not following or not records, len(records), accepted)
                if not following or not records:
                    break
                cursor = following
        return self.db.conn.execute("SELECT * FROM openalex_sources WHERE venue=? ORDER BY source_id", (venue.key,)).fetchall()

    def discover_oa(self):
        for venue in self.venues:
            sources = self.discover_sources(venue)
            if not sources:
                LOG.warning("OpenAlex: no specific source for %s; relying on S2 discovery and DOI enrichment", venue.key)
            for source in sources:
                for year in range(max(self.start, venue.first_year), self.end + 1):
                    job = job_key("oa-discover", venue.key, year=year, source=source["source_id"])
                    state = self.db.checkpoint(job)
                    if state and state["done"]:
                        continue
                    cursor = state["cursor"] if state else "*"
                    while True:
                        response = self.openalex.works(source["source_id"], year, cursor)
                        records = response.get("results")
                        if not isinstance(records, list):
                            raise APIError("OpenAlex returned no works array; checkpoint retained.")
                        following = (response.get("meta") or {}).get("next_cursor")
                        if following and following == cursor:
                            raise APIError("OpenAlex repeated its works cursor; checkpoint retained.")
                        accepted = 0
                        with self.db.conn:
                            for record in records:
                                if record.get("publication_year") == year and is_paper("openalex", record):
                                    self.db.upsert("openalex", record, venue.key, year)
                                    accepted += 1
                            self.db.save_checkpoint(job, following, not following or not records, len(records), accepted)
                        LOG.info("OA %-15s %s: %s/%s records accepted (%s)", venue.key, year, accepted, len(records), source["source_id"].rsplit("/", 1)[-1])
                        if not following or not records:
                            break
                        cursor = following

    def enrich_oa(self):
        keys = [v.key for v in self.venues]
        while True:
            rows = self.db.missing_for("openalex", keys, self.start, self.end)
            if not rows:
                return
            filled = 0
            for row in rows:
                record = self.openalex.doi(row["doi"])
                with self.db.conn:
                    if record:
                        if normalize_doi(record.get("doi")) != row["doi"]:
                            raise APIError("OpenAlex DOI lookup returned a different DOI; checkpoint retained.")
                        paper_id = self.db.upsert("openalex", record, known_paper_id=row["id"])
                        has_abstract = self.db.conn.execute("SELECT abstract IS NOT NULL FROM papers WHERE id=?", (paper_id,)).fetchone()[0]
                        filled += int(has_abstract)
                        outcome = "abstract" if has_abstract else "no_abstract"
                    else:
                        outcome = "not_found"
                    self.db.mark_enrichment("openalex", row["doi"], outcome)
            LOG.info("OpenAlex DOI enrichment: %s checked, %s abstracts filled", len(rows), filled)

    def enrich_s2(self):
        keys = [v.key for v in self.venues]
        while True:
            rows = self.db.missing_for("semantic_scholar", keys, self.start, self.end, limit=100)
            if not rows:
                return
            records = self.semantic_scholar.dois([r["doi"] for r in rows])
            if not isinstance(records, list) or len(records) != len(rows):
                raise APIError("Semantic Scholar batch response length mismatch; checkpoint retained.")
            filled = 0
            with self.db.conn:
                for row, record in zip(rows, records):
                    if record:
                        response_doi = normalize_doi((record.get("externalIds") or {}).get("DOI"))
                        if response_doi and response_doi != row["doi"]:
                            raise APIError("Semantic Scholar batch returned a different DOI; checkpoint retained.")
                        paper_id = self.db.upsert("semantic_scholar", record, known_paper_id=row["id"])
                        has_abstract = self.db.conn.execute("SELECT abstract IS NOT NULL FROM papers WHERE id=?", (paper_id,)).fetchone()[0]
                        filled += int(has_abstract)
                        outcome = "abstract" if has_abstract else "no_abstract"
                    else:
                        outcome = "not_found"
                    self.db.mark_enrichment("semantic_scholar", row["doi"], outcome)
            LOG.info("Semantic Scholar DOI enrichment: %s checked, %s abstracts filled", len(rows), filled)

    def run(self):
        errors = []
        # Let the other provider work if one is unavailable. Failed stages keep
        # their cursor and return a nonzero exit status; they are never marked done.
        for name, client, discover in (
            ("Semantic Scholar", self.semantic_scholar, self.discover_s2),
            ("OpenAlex", self.openalex, self.discover_oa),
        ):
            if client:
                try:
                    discover()
                except APIError as exc:
                    errors.append(str(exc))
                    LOG.error("%s discovery incomplete: %s", name, exc)
        for name, client, enrich in (
            ("OpenAlex", self.openalex, self.enrich_oa),
            ("Semantic Scholar", self.semantic_scholar, self.enrich_s2),
        ):
            if client:
                try:
                    enrich()
                except APIError as exc:
                    errors.append(str(exc))
                    LOG.error("%s enrichment incomplete: %s", name, exc)
        return errors
