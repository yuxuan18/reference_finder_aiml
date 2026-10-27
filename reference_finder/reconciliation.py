"""Conservative identity merges and reversible, evidence-based scope review."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin

from .aaai import issue_entries, issue_year
from .paper_identity import aaai_article_id, paper_identity
from .proceedings import html_page, text
from .storage import normalize_doi, now


def cached_aaai_evidence(db, cache_dir):
    """Read only cache files named by source-run URLs and archive issue links."""
    cache = Path(cache_dir)

    def read(url):
        path = cache / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if not path.exists():
            return None
        entry = json.loads(path.read_text())
        return entry.get("body") if entry.get("url") == url else None

    issues, archives, years = {}, set(), set()
    for row in db.conn.execute("SELECT year,urls_json FROM roster_runs WHERE provider='aaai'"):
        years.add(row["year"])
        for url in json.loads(row["urls_json"]):
            if url.startswith("https://ojs.aaai.org/index.php/AAAI/issue/view/"):
                issues[url] = row["year"]
            if url.startswith("https://ojs.aaai.org/index.php/AAAI/issue/archive"):
                archives.add(url)
    for url in sorted(archives):
        raw = read(url)
        if not raw:
            continue
        for card in html_page(raw).select(".obj_issue_summary"):
            link = card.select_one('a[href*="/issue/view/"]')
            year = issue_year(text(card.select_one(".series")) or text(card))
            if link and year in years:
                target = urljoin(url, link["href"])
                if target.startswith("https://ojs.aaai.org/index.php/AAAI/issue/view/"):
                    issues[target] = year
    evidence, missing = [], []
    for url, year in sorted(issues.items()):
        raw = read(url)
        if raw is None:
            missing.append(url)
            continue
        for entry in issue_entries(raw, url):
            entry = dict(entry, year=year, article_id=aaai_article_id(entry["url"]))
            if year == 2019 and entry["classification"] == "excluded":
                article = read(entry["url"])
                if article:
                    meta = html_page(article).select_one('meta[name="citation_doi"]')
                    if meta:
                        entry["doi"] = normalize_doi(meta.get("content"))
            evidence.append(entry)
    return evidence, missing


def counts(db):
    conn = db.conn
    return {"papers": conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            "source_records": conn.execute("SELECT COUNT(*) FROM source_records").fetchone()[0],
            "roster_entries": conn.execute("SELECT COUNT(*) FROM roster_entries").fetchone()[0],
            "active_papers": conn.execute("SELECT COUNT(*) FROM active_corpus_papers").fetchone()[0],
            "active_abstracts": conn.execute("SELECT COUNT(*) FROM active_corpus_papers WHERE abstract IS NOT NULL").fetchone()[0],
            "excluded_memberships": conn.execute("SELECT COUNT(*) FROM scope_exclusions").fetchone()[0]}


def reconcile(db, evidence=(), apply=False, aliases=None, reviews=None):
    """Plan and optionally apply in one locked transaction. Never match titles."""
    conn = db.conn
    report = dict(generated_at=now(), applied=apply, merges=[], exclusions=[], conflicts=[], unresolved=[])
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        report["before"] = counts(db)
        papers = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM papers")}
        memberships = defaultdict(set)
        identities = defaultdict(set)
        article_ids = defaultdict(set)
        dois = defaultdict(set)
        for pid, paper in papers.items():
            if paper["doi"]:
                dois[pid].add(normalize_doi(paper["doi"]))
        for row in conn.execute("SELECT paper_id,venue,year,url,doi FROM roster_entries"):
            pid, venue, year = row["paper_id"], row["venue"], row["year"]
            memberships[pid].add((venue, year))
            identity = paper_identity(row["url"])
            if identity:
                identities[venue, year, identity].add(pid)
            if row["doi"]:
                dois[pid].add(normalize_doi(row["doi"]))
            if venue == "aaai":
                aid = aaai_article_id(row["url"], row["doi"])
                if aid:
                    article_ids[pid, year].add(aid)
                    identities[venue, year, "aaai:" + aid].add(pid)
        for pid, paper in papers.items():
            for venue, year in memberships[pid]:
                identity = paper_identity(paper["url"])
                if identity:
                    identities[venue, year, identity].add(pid)
                if venue == "aaai":
                    aid = aaai_article_id(paper["url"], paper["doi"])
                    if aid:
                        article_ids[pid, year].add(aid)
                        identities[venue, year, "aaai:" + aid].add(pid)
        # Curated cross-source links require an explicit reviewed alias. Text
        # similarity alone never creates a merge edge here.
        if aliases is None:
            aliases = json.loads(Path(__file__).with_name("roster_aliases.json").read_text())
        source_ids = {(r["provider"], r["external_id"]): r["paper_id"]
                      for r in conn.execute("SELECT provider,external_id,paper_id FROM source_records")}
        reviewed_proof = {}
        for alias in aliases:
            left = source_ids.get((alias["provider"], alias["external_id"]))
            right = source_ids.get(("dblp", alias["dblp_key"]))
            event = alias["venue"], alias["year"]
            if (left is None or right is None or left == right
                    or event not in memberships[left] or event not in memberships[right]):
                continue
            key = (*event, "reviewed:" + alias["provider"] + ":" + alias["external_id"])
            identities[key].update((left, right))
            reviewed_proof[key] = {k: alias[k] for k in ("reason", "evidence_urls") if k in alias}
        # Connected components handle multiple source URLs for one publication.
        parent = {pid: pid for pid in papers}

        def root(pid):
            while parent[pid] != pid:
                parent[pid] = parent[parent[pid]]
                pid = parent[pid]
            return pid

        for ids in identities.values():
            if len(ids) > 1:
                first = min(ids)
                for pid in ids:
                    parent[root(pid)] = root(first)
        groups = defaultdict(set)
        for pid in papers:
            groups[root(pid)].add(pid)
        canonical = {pid: pid for pid in papers}
        for ids in groups.values():
            if len(ids) < 2:
                continue
            identifiers = sorted(set().union(*(dois[pid] for pid in ids)))
            proof = [{"venue": v, "year": y, "identity": identity, **reviewed_proof.get((v, y, identity), {})}
                     for (v, y, identity), linked in identities.items() if len(linked & ids) > 1]
            if len(identifiers) > 1:
                report["conflicts"].append(dict(paper_ids=sorted(ids), dois=identifiers, evidence=proof))
                continue
            keep = min(ids)
            report["merges"].append(dict(keep=keep, remove=sorted(ids - {keep}), evidence=proof,
                                         titles={str(pid): papers[pid]["title"] for pid in sorted(ids)}))
            for pid in ids:
                canonical[pid] = keep
                if apply and pid != keep:
                    db._merge(keep, pid)
        lookup, doi_lookup = defaultdict(list), defaultdict(list)
        for entry in evidence:
            lookup[entry["year"], entry["article_id"]].append(entry)
            if entry.get("doi"):
                doi_lookup[entry["year"], normalize_doi(entry["doi"])].append(entry)
        decisions = defaultdict(list)
        for (pid, year), ids in article_ids.items():
            for aid in ids:
                decisions[canonical[pid], year].extend(lookup[year, aid])
        for pid, venues in memberships.items():
            for venue, year in venues:
                if venue == "aaai":
                    for doi in dois[pid]:
                        decisions[canonical[pid], year].extend(doi_lookup[year, doi])
        for (pid, year), entries in sorted(decisions.items()):
            classes = {e["classification"] for e in entries}
            if "research" in classes:
                if "excluded" in classes:
                    report["conflicts"].append(dict(paper_ids=[pid], reason="Conflicting AAAI section evidence", evidence=entries))
                if apply:
                    conn.execute("DELETE FROM scope_exclusions WHERE paper_id=? AND venue='aaai' AND year=?", (pid, year))
                continue
            if "excluded" not in classes:
                continue
            entry = next(e for e in entries if e["classification"] == "excluded")
            exclusion = dict(paper_id=pid, venue="aaai", year=year, title=papers[pid]["title"],
                             section=entry["section"], evidence_url=entry["evidence_url"],
                             reason="Official issue section is outside AAAI main and special research tracks")
            report["exclusions"].append(exclusion)
            if apply:
                conn.execute("""INSERT INTO scope_exclusions VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(paper_id,venue,year) DO UPDATE SET section=excluded.section,
                    evidence_url=excluded.evidence_url,reason=excluded.reason,reviewed_at=excluded.reviewed_at""",
                    (pid, "aaai", year, exclusion["section"], exclusion["evidence_url"], exclusion["reason"], now()))
        if reviews is None:
            path = Path(__file__).with_name("scope_reviews.json")
            reviews = json.loads(path.read_text()) if path.exists() else []
        report["annotations"] = []
        for review in reviews:
            original = source_ids.get((review["provider"], review["external_id"]))
            if original is None or (review["venue"], review["year"]) not in memberships[original]:
                continue
            pid = canonical[original]
            if review["action"] not in {"exclude", "annotate"}:
                raise ValueError("Unknown scope review action")
            if review["action"] == "exclude":
                exclusion = dict(paper_id=pid, venue=review["venue"], year=review["year"],
                                 title=papers[pid]["title"], section=review["section"],
                                 evidence_url=review["evidence_urls"][0], reason=review["reason"])
                report["exclusions"] = [e for e in report["exclusions"] if
                    (e["paper_id"], e["venue"], e["year"]) != (pid, review["venue"], review["year"])]
                report["exclusions"].append(exclusion)
                if apply:
                    conn.execute("""INSERT OR REPLACE INTO scope_exclusions VALUES (?,?,?,?,?,?,?)""",
                                 (pid, review["venue"], review["year"], review["section"],
                                  exclusion["evidence_url"], review["reason"], now()))
            else:
                report["annotations"].append(dict(paper_id=pid, **review))
            if apply:
                for row in conn.execute("""SELECT rowid,review_json FROM roster_entries
                        WHERE paper_id=? AND venue=? AND year=?""", (pid, review["venue"], review["year"])).fetchall():
                    metadata = json.loads(row["review_json"])
                    metadata["roster_review"] = dict(action=review["action"], note=review["reason"],
                                                     evidence_urls=review["evidence_urls"])
                    conn.execute("UPDATE roster_entries SET review_json=? WHERE rowid=?",
                                 (json.dumps(metadata, ensure_ascii=False), row["rowid"]))
        if apply:
            if conn.execute("PRAGMA foreign_key_check").fetchone():
                raise ValueError("Reconciliation broke a foreign key; rolled back")
            report["after"] = counts(db)
            for key in ("source_records", "roster_entries"):
                if report["after"][key] != report["before"][key]:
                    raise ValueError("Reconciliation changed source evidence counts; rolled back")
        for row in conn.execute("""SELECT DISTINCT p.id,p.title,r.venue,r.year,p.doi,p.url
                FROM active_roster_entries r JOIN papers p ON p.id=r.paper_id
                WHERE p.abstract IS NULL ORDER BY r.venue,r.year,p.id"""):
            report["unresolved"].append(dict(row))
    return report
