"""Link explicit proceedings rosters to the API corpus without losing evidence."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .storage import normalize_doi, normalize_title, now, review_metadata
from .paper_identity import paper_identity


class RosterStore:
    def __init__(self, db):
        self.db = db
        aliases_path = Path(__file__).with_name("roster_aliases.json")
        aliases = json.loads(aliases_path.read_text()) if aliases_path.exists() else []
        self.aliases = {(a["provider"], a["external_id"], a["venue"], a["year"]): a["dblp_key"] for a in aliases}
        self.alias_reasons = {(a["provider"], a["external_id"], a["venue"], a["year"]): a["reason"] for a in aliases}
        reviews_path = Path(__file__).with_name("roster_review.json")
        reviews = json.loads(reviews_path.read_text()) if reviews_path.exists() else []
        self.reviews = {(r["provider"], r["external_id"], r["venue"], r["year"]): r for r in reviews}
        self.dblp_ids = {}
        self.paper_identities = {}
        for row in db.conn.execute("SELECT provider,external_id,venue,year,url FROM roster_entries"):
            identity = paper_identity(row["url"])
            if identity:
                self.paper_identities.setdefault((row["venue"], row["year"], identity), set()).add((row["provider"], row["external_id"]))
        for row in db.conn.execute("SELECT external_id,dblp_key FROM source_records WHERE provider='semantic_scholar'"):
            key = row["dblp_key"]
            if key:
                self.dblp_ids.setdefault(key, set()).add(row["external_id"])

    def _match(self, provider, record, venue, year):
        db = self.db.conn
        doi = normalize_doi(record.get("doi"))
        key = record.get("dblp_key") or (record["external_id"] if provider == "dblp" else None)
        alias = self.aliases.get((provider, record["external_id"], venue, year))
        if alias:
            linked = db.execute("""SELECT p.id,p.doi FROM source_records s JOIN papers p ON p.id=s.paper_id
                WHERE s.provider='dblp' AND s.external_id=?""", (alias,)).fetchone()
            if linked and (not doi or not linked["doi"] or linked["doi"] == doi):
                return linked["id"]
        candidates = set()
        # Cache stable provider IDs, not mutable local paper IDs: DOI bridges
        # can merge/delete local IDs during the same import.
        for external_id in self.dblp_ids.get(key, set()):
            identity = db.execute("SELECT paper_id FROM source_records WHERE provider='semantic_scholar' AND external_id=?", (external_id,)).fetchone()
            if identity:
                candidates.add(identity[0])
        if doi:
            candidates.update(r[0] for r in db.execute("SELECT id FROM papers WHERE doi=?", (doi,)))
        candidates.update(r[0] for r in db.execute("SELECT paper_id FROM source_records WHERE provider=? AND external_id=?", (provider, record["external_id"])))
        if doi:
            candidates = {pid for pid in candidates if
                          db.execute("SELECT doi FROM papers WHERE id=?", (pid,)).fetchone()[0] in (None, doi)}
        if len(candidates) == 1:
            return candidates.pop()
        identity = paper_identity(record.get("url"))
        if identity:
            # Resolve provider identities at lookup time: merges can delete
            # local IDs, and SQLite can reuse those IDs for unrelated papers.
            sources = self.paper_identities.get((venue, year, identity), set())
            compatible = set()
            for source, external_id in sources:
                row = db.execute("""SELECT p.id,p.doi FROM source_records s JOIN papers p ON p.id=s.paper_id
                    WHERE s.provider=? AND s.external_id=?""", (source, external_id)).fetchone()
                if row and (not doi or not row["doi"] or row["doi"] == doi):
                    compatible.add(row["id"])
            if len(compatible) == 1:
                return compatible.pop()
        url = record.get("url")
        if url:
            parsed = urlsplit(url)
            host = parsed.netloc.removeprefix("www.")
            urls = [urlunsplit((scheme, prefix + host, parsed.path.rstrip("/"), parsed.query, ""))
                    for scheme in ("http", "https") for prefix in ("", "www.")]
            marks = ",".join("?" for _ in urls)
            rows = db.execute(f"""SELECT DISTINCT p.id,p.doi FROM roster_entries r JOIN papers p ON p.id=r.paper_id
                WHERE r.venue=? AND r.year=? AND r.url IN ({marks}) AND p.normalized_title=?""",
                (venue, year, *urls, normalize_title(record["title"]))).fetchall()
            ids = {r["id"] for r in rows if not doi or not r["doi"] or doi == r["doi"]}
            if len(ids) == 1:
                return ids.pop()
        # Title-only evidence requires the same event year. DOI/DBLP identifiers
        # above may bridge a different API publication year without guessing.
        rows = db.execute("""SELECT DISTINCT p.* FROM papers p JOIN paper_venues v ON v.paper_id=p.id
            WHERE p.normalized_title=? AND v.venue=? AND v.year=?""",
            (normalize_title(record["title"]), venue, year)).fetchall()
        compatible = [r for r in rows if not doi or not r["doi"] or r["doi"] == doi]
        rostered = [r for r in compatible if db.execute("SELECT 1 FROM roster_entries WHERE paper_id=? AND venue=? AND year=? LIMIT 1", (r["id"], venue, year)).fetchone()]
        if len(rostered) == 1:
            return rostered[0]["id"]
        if len(compatible) == 1:
            return compatible[0]["id"]
        return None

    def save(self, provider, venue, year, result):
        source_records = result.get("records") or []
        status = result.get("status", "incomplete")
        if status == "complete" and not source_records:
            status = "incomplete"
        records, exclusions = [], []
        urls = list(result.get("urls") or [])
        for source_record in source_records:
            record = dict(source_record)
            review = self.reviews.get((provider, record["external_id"], venue, year))
            if review:
                if review["action"] not in {"exclude", "annotate"}:
                    raise ValueError("Unknown roster review action: " + str(review["action"]))
                record["roster_review"] = {k: review[k] for k in ("action", "note", "evidence_urls")}
                urls.extend(url for url in review["evidence_urls"] if url not in urls)
                if review["action"] == "exclude":
                    exclusions.append(record)
                    continue
            records.append(record)
        note = result.get("note") or ""
        if exclusions:
            detail = "; ".join(r["external_id"] + ": " + r["roster_review"]["note"] for r in exclusions)
            note = (note + " Reviewed exclusions (%d): %s" % (len(exclusions), detail)).strip()
        adjusted = dict(result, records=records, status=status, note=note, urls=urls)
        with self.db.conn:
            if status == "complete":
                self.db.conn.execute("DELETE FROM roster_entries WHERE provider=? AND venue=? AND year=?", (provider, venue, year))
            # A reviewed exclusion is affirmative evidence, even if another
            # part of this source scan failed.  Preserve the underlying corpus
            # and source identities; remove only this source's roster membership.
            for record in exclusions:
                self.db.conn.execute("DELETE FROM roster_entries WHERE provider=? AND external_id=? AND venue=? AND year=?",
                                     (provider, record["external_id"], venue, year))
            for record in records:
                alias_key = (provider, record["external_id"], venue, year)
                if alias_key in self.aliases:
                    record = dict(record, roster_match={"method": "reviewed_alias", "dblp_key": self.aliases[alias_key],
                                                       "reason": self.alias_reasons.get(alias_key, "Reviewed identity")})
                existing = self._match(provider, record, venue, year)
                paper_id = self.db.upsert(provider, record, venue, year, known_paper_id=existing)
                identity = paper_identity(record.get("url"))
                if identity:
                    self.paper_identities.setdefault((venue, year, identity), set()).add((provider, record["external_id"]))
                if provider == "dblp" and record.get("publication_year"):
                    self.db.conn.execute("UPDATE papers SET publication_year=? WHERE id=?", (record["publication_year"], paper_id))
                self.db.conn.execute("""INSERT INTO roster_entries VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(provider,external_id,venue,year) DO UPDATE SET paper_id=excluded.paper_id,
                    title=excluded.title,doi=excluded.doi,url=excluded.url,authors_json=excluded.authors_json,
                    section=excluded.section,review_json=excluded.review_json""",
                    (provider, record["external_id"], venue, year, paper_id, record["title"], normalize_doi(record.get("doi")),
                     record.get("url"), json.dumps(record.get("authors") or [], ensure_ascii=False),
                     record.get("section"), json.dumps(review_metadata(record), ensure_ascii=False)))
            self.db.conn.execute("""INSERT OR REPLACE INTO roster_runs VALUES (?,?,?,?,?,?,?,?)""",
                (provider, venue, year, status, note, json.dumps(urls), len(records), now()))
        return adjusted

    def finished(self, provider, venue, year):
        row = self.db.conn.execute("SELECT status FROM roster_runs WHERE provider=? AND venue=? AND year=?", (provider, venue, year)).fetchone()
        return bool(row and row[0] == "complete")

    def coverage(self, venue, year):
        row = dict(self.db.conn.execute("""SELECT COUNT(DISTINCT p.id) AS listed,
            COUNT(DISTINCT CASE WHEN p.abstract IS NOT NULL THEN p.id END) AS abstracts
            FROM active_roster_entries r JOIN papers p ON p.id=r.paper_id WHERE r.venue=? AND r.year=?""", (venue, year)).fetchone())
        row["excluded"] = self.db.conn.execute("SELECT COUNT(*) FROM scope_exclusions WHERE venue=? AND year=?", (venue, year)).fetchone()[0]
        row["runs"] = [dict(r) for r in self.db.conn.execute("SELECT * FROM roster_runs WHERE venue=? AND year=? ORDER BY provider", (venue, year))]
        row["unverified_api"] = self.db.conn.execute("""SELECT COUNT(DISTINCT v.paper_id) FROM active_paper_venues v
            WHERE v.venue=? AND v.year=? AND v.provider IN ('openalex','semantic_scholar')
            AND NOT EXISTS (SELECT 1 FROM roster_entries r WHERE r.paper_id=v.paper_id AND r.venue=v.venue AND r.year=v.year)""", (venue, year)).fetchone()[0]
        return row

    def papers(self, venues, start, end):
        marks = ",".join("?" for _ in venues)
        rows = self.db.conn.execute(f"""SELECT p.*,r.venue,r.year AS event_year,
            GROUP_CONCAT(DISTINCT r.provider) AS roster_sources
            FROM active_roster_entries r JOIN papers p ON p.id=r.paper_id
            WHERE r.venue IN ({marks}) AND r.year BETWEEN ? AND ?
            GROUP BY r.venue,r.year,p.id ORDER BY r.venue,r.year,p.title""", (*venues, start, end)).fetchall()
        for row in rows:
            result = dict(row)
            evidence = self.db.conn.execute("""SELECT provider,external_id,title,url,section,doi,review_json FROM roster_entries
                WHERE paper_id=? AND venue=? AND year=? ORDER BY provider""", (row["id"], row["venue"], row["event_year"])).fetchall()
            result["evidence"] = []
            for entry in evidence:
                entry = dict(entry)
                review = json.loads(entry.pop("review_json")).get("roster_review") or {}
                entry["review_note"] = review.get("note", "")
                entry["review_evidence_urls"] = review.get("evidence_urls", [])
                result["evidence"].append(entry)
            # Prefer the actual proceedings title over an API preprint title.
            if evidence:
                result["title"] = evidence[0]["title"]
            yield result
