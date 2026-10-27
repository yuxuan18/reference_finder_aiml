#!/usr/bin/env python3
"""Audit stable publication identities and AAAI section scope using cached evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from reference_finder.reconciliation import cached_aaai_evidence, reconcile
from reference_finder.storage import Database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--cache-dir", default="data/page_cache")
    parser.add_argument("--output", default="data/corpus/reconciliation.json")
    parser.add_argument("--apply", action="store_true", help="Apply reviewed identities and section exclusions, after making a backup")
    args = parser.parse_args()
    path = Path(args.db).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not path.is_file():
        parser.error("Database does not exist")
    if output == path or (output.exists() and output.samefile(path)):
        parser.error("Audit output must differ from the database")
    backup = None
    if args.apply:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.parent / "backups" / (path.stem + ".before-reconciliation-" + stamp + path.suffix)
        backup.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as source, sqlite3.connect(str(backup)) as destination:
            source.backup(destination)
            if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Backup verification failed")
    db = Database(str(path))
    try:
        evidence, missing = cached_aaai_evidence(db, args.cache_dir)
        report = reconcile(db, evidence, args.apply)
        report.update(backup=str(backup) if backup else None, missing_issue_pages=missing)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"applied": args.apply, "merge_groups": len(report["merges"]),
                          "duplicates": sum(len(m["remove"]) for m in report["merges"]),
                          "excluded_memberships": len(report["exclusions"]),
                          "conflicts": len(report["conflicts"]), "missing_issue_pages": len(missing),
                          "before": report["before"], "after": report.get("after"),
                          "audit": str(output.resolve()), "backup": str(backup) if backup else None}, indent=2))
    finally:
        db.conn.close()


if __name__ == "__main__":
    main()
