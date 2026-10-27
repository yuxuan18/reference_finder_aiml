#!/usr/bin/env python3
"""Build an explicit proceedings roster, retrieve official abstracts, and export it."""
from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
import sys

from reference_finder.api import APIError, Budget, RequestLimit
from reference_finder.corpus_report import export_corpus
from reference_finder.roster import RosterStore
from reference_finder.storage import Database
from reference_finder.venues import DEFAULT_VENUES, VENUES
from reference_finder.webfetch import WebFetcher


SOURCES = {
    "dblp": set(VENUES) - {"colm"},
    "iclr": {"iclr"},
    "colm": {"colm"},
    "openreview": {"iclr", "colm"},
    "pmlr": {"icml"},
    "neurips": {"neurips"},
    "aaai": {"aaai"},
}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", default="data/papers.sqlite3")
    p.add_argument("--start-year", type=int, default=date.today().year - 9)
    p.add_argument("--end-year", type=int, default=date.today().year)
    p.add_argument("--venues", nargs="+", choices=list(VENUES), default=list(DEFAULT_VENUES))
    p.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    p.add_argument("--output", default="data/corpus")
    p.add_argument("--cache-dir", default="data/page_cache")
    p.add_argument("--refresh", action="store_true", help="Refetch source lists and rebuild their roster membership")
    p.add_argument("--report-only", action="store_true", help="Regenerate CSV/HTML from SQLite, with no network")
    p.add_argument("--max-requests", type=int, help="Bound HTTP attempts; rerun without --refresh to resume")
    p.add_argument("--interval", type=float, default=1.0, help="Minimum seconds between requests to the same host")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not 2017 <= args.start_year <= args.end_year <= date.today().year:
        p.error("use 2017 <= start-year <= end-year <= current year")
    if args.interval < 0 or (args.max_requests is not None and args.max_requests <= 0):
        p.error("interval must be nonnegative and max-requests positive")
    if args.report_only and args.refresh:
        p.error("--report-only cannot be combined with --refresh")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not args.report_only:
        try:
            from reference_finder.config import load_environment
            load_environment()
            from reference_finder.dblp import collect_dblp
            from reference_finder.openreview import collect_openreview
            from reference_finder.pmlr import collect_pmlr
            from reference_finder.neurips import collect_neurips
            from reference_finder.aaai import collect_aaai
            from reference_finder.iclr import collect_iclr
            from reference_finder.colm import collect_colm
        except ImportError as exc:
            if exc.name == "bs4":
                p.error("BeautifulSoup is required: python3 -m pip install -r requirements.txt (or use .venv/bin/python)")
            if exc.name == "dotenv":
                p.error("Install collection dependencies: uv add -r requirements-webscraping.txt (or python3 -m pip install -r requirements-webscraping.txt)")
            raise
    db_path = Path(args.db).expanduser()
    db = Database(str(db_path))
    store = RosterStore(db)
    venues = [VENUES[v] for v in dict.fromkeys(args.venues)]
    budget = Budget(args.max_requests)
    fetch = WebFetcher(args.cache_dir, args.refresh, args.interval, budget)
    status = 0
    try:
        if not args.report_only:
            for provider in dict.fromkeys(args.sources):
                for venue in venues:
                    if venue.key not in SOURCES[provider]:
                        continue
                    for year in range(max(args.start_year, venue.first_year), args.end_year + 1):
                        if not args.refresh and store.finished(provider, venue.key, year):
                            continue
                        try:
                            logging.info("Collecting %s %s %s (%s)", provider, venue.key, year,
                                         "roster only" if provider in {"dblp", "colm"} else "roster and available abstracts")
                            if provider == "dblp":
                                result = collect_dblp(venue.key, year, fetch)
                            elif provider == "openreview":
                                result = collect_openreview(venue.key, year, fetch)
                            elif provider == "iclr":
                                result = collect_iclr(year, fetch)
                            elif provider == "colm":
                                result = collect_colm(year, fetch)
                            elif provider == "pmlr":
                                result = collect_pmlr(year, fetch)
                            elif provider == "neurips":
                                result = collect_neurips(year, fetch)
                            elif provider == "aaai":
                                result = collect_aaai(year, fetch)
                            else:
                                raise ValueError("Unknown corpus source: " + provider)
                        except RequestLimit as exc:
                            if hasattr(exc, "partial_result"):
                                store.save(provider, venue.key, year, exc.partial_result)
                            raise
                        except APIError as exc:
                            result = {"records": [], "urls": [], "status": "incomplete", "note": str(exc)}
                        result = store.save(provider, venue.key, year, result)
                        abstracts = sum(bool(r.get("abstract")) for r in result["records"])
                        detail = "roster only; this source does not supply abstracts" if provider in {"dblp", "colm"} else f"{abstracts} source abstracts"
                        logging.info("%s %-15s %s: %s papers, %s [%s] %s", provider, venue.key, year,
                                     len(result["records"]), detail, result["status"], result.get("note", ""))
                        if result["status"] == "incomplete":
                            status = 1
    except RequestLimit as exc:
        logging.info("%s", exc)
        status = 2
    except KeyboardInterrupt:
        logging.warning("Interrupted; saved source lists will resume on rerun.")
        status = 130
    finally:
        export_corpus(store, venues, args.start_year, args.end_year, args.output)
        logging.info("HTTP requests this run: %s", budget.used)
        db.conn.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
