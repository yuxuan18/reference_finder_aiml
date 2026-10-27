#!/usr/bin/env python3
"""Recover missing roster abstracts with reviewed identities and official/API sources."""
import argparse
from contextlib import closing
from datetime import datetime, timezone, date
import json
import os
from pathlib import Path
import sqlite3

from reference_finder.abstract_recovery import recover_missing
from reference_finder.api import Budget, HTTP, OpenAlex
from reference_finder.config import load_environment
from reference_finder.corpus_report import export_corpus
from reference_finder.reconciliation import cached_aaai_evidence, counts, reconcile
from reference_finder.roster import RosterStore
from reference_finder.storage import Database, now
from reference_finder.venues import DEFAULT_VENUES, VENUES
from reference_finder.webfetch import WebFetcher


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", default="data/corpus")
    parser.add_argument("--cache-dir", default="data/page_cache")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Fetch and preview; do not apply paper changes")
    args = parser.parse_args(argv)
    path, output = Path(args.db).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if not path.is_file() or args.max_requests < 1:
        parser.error("Existing database and positive max-requests required")
    if output == path or output.is_file():
        parser.error("Output must be a directory separate from the database")
    load_environment()
    backup = None
    if not args.dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.parent / "backups" / (path.stem + ".before-abstract-recovery-" + stamp + path.suffix)
        backup.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as src, closing(sqlite3.connect(backup)) as dst:
            src.backup(dst)
            if dst.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Backup verification failed")
    db = Database(path)
    try:
        before = counts(db)
        evidence, missing = cached_aaai_evidence(db, args.cache_dir)
        review = reconcile(db, evidence, apply=not args.dry_run)
        budget = Budget(args.max_requests)
        key = os.environ.get("OPENALEX_API_KEY")
        oa = OpenAlex(HTTP("OpenAlex", "https://api.openalex.org", budget,
                           {"Authorization": "Bearer " + key}, interval=0.2, retries=2, timeout=25)) if key else None
        recovered = recover_missing(db, WebFetcher(args.cache_dir, budget=budget), oa, apply=not args.dry_run)
        result = dict(generated_at=now(), applied=not args.dry_run, backup=str(backup) if backup else None,
                      before=before, after=counts(db), reconciliation=review, recovery=recovered,
                      missing_issue_pages=missing, http_attempts=budget.used)
        result["remaining_missing"] = [dict(r) for r in db.conn.execute("""SELECT DISTINCT p.id,p.title,
            r.venue,r.year FROM active_roster_entries r JOIN papers p ON p.id=r.paper_id
            WHERE p.abstract IS NULL ORDER BY r.venue,r.year,p.id""")]
        if db.conn.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Database failed its foreign-key check")
        output.mkdir(parents=True, exist_ok=True)
        (output / "abstract-recovery.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if not args.dry_run:
            export_corpus(RosterStore(db), [VENUES[v] for v in DEFAULT_VENUES],
                          date.today().year - 9, date.today().year, output)
        print(json.dumps(dict(before=before, after=result["after"], recovered=sum(
            r["status"] == "recovered" for r in recovered), duplicates=sum(len(m["remove"]) for m in review["merges"]),
            backup=result["backup"], audit=str(output / "abstract-recovery.json")), indent=2))
        return int(any(r["status"] not in {"recovered", "already updated", "available"} for r in recovered))
    finally:
        db.conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
