#!/usr/bin/env python3
"""Download indexed conference abstracts into SQLite (AI/ML venues by default)."""
from __future__ import annotations

import argparse
from datetime import date
import logging
import os
import sqlite3
import sys

from reference_finder.api import APIError, Budget, HTTP, OpenAlex, RequestLimit, SemanticScholar
from reference_finder.importer import Importer
from reference_finder.reporting import report
from reference_finder.storage import Database
from reference_finder.venues import DEFAULT_VENUES, VENUES


def parser():
    current_year = date.today().year
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", default="data/papers.sqlite3", help="SQLite database path")
    p.add_argument("--start-year", type=int, default=current_year - 9, help="First publication year, inclusive")
    p.add_argument("--end-year", type=int, default=current_year, help="Last publication year, inclusive")
    p.add_argument("--venues", nargs="+", choices=list(VENUES), default=list(DEFAULT_VENUES), help="Conference keys")
    p.add_argument("--providers", nargs="+", choices=["openalex", "semantic-scholar"], default=["openalex", "semantic-scholar"])
    p.add_argument("--phase", choices=["all", "discover", "enrich"], default="all")
    p.add_argument("--refresh", action="store_true", help="Recheck selected venue/years and missing abstracts; keep existing papers")
    p.add_argument("--report-only", action="store_true", help="Print stored coverage without calling APIs")
    p.add_argument("--max-requests", type=int, help="Pause after this many HTTP requests (including retries); rerun to resume")
    p.add_argument("--retries", type=int, default=5, help="Retries per transient failure")
    p.add_argument("--timeout", type=float, default=60, help="HTTP timeout in seconds")
    p.add_argument("--openalex-interval", type=float, default=0.2, help="Minimum seconds between OpenAlex requests")
    p.add_argument("--s2-interval", type=float, default=1.1, help="Minimum seconds between Semantic Scholar requests")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not 1900 <= args.start_year <= args.end_year <= date.today().year:
        p.error("years must satisfy 1900 <= start-year <= end-year <= current year")
    if args.max_requests is not None and args.max_requests < 1:
        p.error("--max-requests must be positive")
    if args.retries < 0 or args.timeout <= 0 or min(args.openalex_interval, args.s2_interval) < 0:
        p.error("retries and request intervals must be nonnegative, and timeout must be positive")
    if args.refresh and args.report_only:
        p.error("--refresh cannot be used with --report-only")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not args.report_only:
        from reference_finder.config import load_environment
        try:
            load_environment()
        except ImportError as exc:
            if exc.name == "dotenv":
                p.error("Install collection dependencies: uv add -r requirements-webscraping.txt (or python3 -m pip install -r requirements-webscraping.txt)")
            raise
    venues = [VENUES[key] for key in dict.fromkeys(args.venues)]
    db = Database(args.db)
    budget = Budget(args.max_requests)
    status = 0
    try:
        if args.report_only:
            report(db, venues, args.start_year, args.end_year)
            return 0
        if args.refresh:
            db.refresh([v.key for v in venues], args.start_year, args.end_year)
        oa, s2 = None, None
        if "openalex" in args.providers:
            key = os.environ.get("OPENALEX_API_KEY")
            headers = {"Authorization": "Bearer " + key} if key else {}
            oa = OpenAlex(HTTP("OpenAlex", "https://api.openalex.org", budget, headers, args.openalex_interval, args.retries, args.timeout))
            if not key:
                logging.warning("OPENALEX_API_KEY is unset; anonymous quota is small. A free key is recommended for the full import.")
        if "semantic-scholar" in args.providers:
            key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
            headers = {"x-api-key": key} if key else {}
            s2 = SemanticScholar(HTTP("Semantic Scholar", "https://api.semanticscholar.org/graph/v1", budget, headers, args.s2_interval, args.retries, args.timeout))
            if not key:
                logging.warning("SEMANTIC_SCHOLAR_API_KEY is unset; anonymous requests may be rate limited.")
        importer = Importer(db, venues, args.start_year, args.end_year, oa, s2)
        logging.info("Importing %s–%s into %s; reruns resume saved progress", args.start_year, args.end_year, args.db)
        if args.phase == "all":
            status = 1 if importer.run() else 0
        else:
            methods = [(s2, importer.discover_s2), (oa, importer.discover_oa)] if args.phase == "discover" else [(oa, importer.enrich_oa), (s2, importer.enrich_s2)]
            for client, method in methods:
                if client:
                    try:
                        method()
                    except APIError as exc:
                        logging.error("%s", exc)
                        status = 1
    except RequestLimit as exc:
        logging.info("%s", exc)
        status = 2
    except KeyboardInterrupt:
        logging.warning("Interrupted; committed pages are saved. Rerun to resume.")
        status = 130
    finally:
        if not args.report_only:
            report(db, venues, args.start_year, args.end_year)
            logging.info("HTTP requests this run: %s", budget.used)
        db.conn.close()
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (APIError, sqlite3.Error, OSError, ValueError) as exc:
        logging.error("%s", exc)
        sys.exit(1)
