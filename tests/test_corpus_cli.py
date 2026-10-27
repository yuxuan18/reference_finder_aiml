import contextlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import build_corpus
import download_abstracts
from reference_finder.api import RequestLimit
from reference_finder.storage import Database
from reference_finder.venues import DEFAULT_VENUES


class CorpusCliTests(unittest.TestCase):
    def test_only_ai_ml_venues_are_selectable(self):
        for parser in (build_corpus.parser(), download_abstracts.parser()):
            self.assertEqual(parser.parse_args([]).venues, list(DEFAULT_VENUES))
            self.assertEqual(parser.parse_args(["--venues", "icml", "colm"]).venues, ["icml", "colm"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--venues", "unsupported-conference"])

    def run_build(self, directory, *options):
        with contextlib.redirect_stdout(io.StringIO()), patch("reference_finder.config.load_environment"):
            return build_corpus.main(["--db", str(Path(directory) / "papers.sqlite3"),
                                      "--output", str(Path(directory) / "report"),
                                      "--cache-dir", str(Path(directory) / "cache"), *options])

    def test_source_routing_resume_and_first_colm_year(self):
        with tempfile.TemporaryDirectory() as directory:
            data = {"records": [{"external_id": "x", "title": "A model", "year": 2024}],
                    "status": "complete", "urls": [], "note": ""}
            with patch("reference_finder.openreview.collect_openreview", return_value=data) as collect, \
                    patch("reference_finder.dblp.collect_dblp") as dblp:
                options = ("--venues", "colm", "--sources", "openreview", "dblp", "--start-year", "2023", "--end-year", "2024")
                self.assertEqual(self.run_build(directory, *options), 0)
                self.assertEqual(collect.call_count, 1)
                self.assertEqual(collect.call_args.args[:2], ("colm", 2024))
                dblp.assert_not_called()
                self.assertEqual(self.run_build(directory, *options), 0)
                self.assertEqual(collect.call_count, 1)
                self.assertEqual(self.run_build(directory, *options, "--refresh"), 0)
                self.assertEqual(collect.call_count, 2)
            coverage = (Path(directory) / "report" / "coverage.csv").read_text()
            self.assertIn("not applicable; conference not yet established", coverage)

    def test_bounded_run_saves_partial_result_and_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = {"records": [{"external_id": "x", "title": "A model", "year": 2025}],
                       "status": "incomplete", "urls": [], "note": "budget exhausted"}
            stop = RequestLimit("bounded")
            stop.partial_result = partial
            options = ("--venues", "icml", "--sources", "pmlr", "--start-year", "2025", "--end-year", "2025")
            with patch("reference_finder.pmlr.collect_pmlr", side_effect=stop):
                self.assertEqual(self.run_build(directory, *options), 2)
            db = Database(Path(directory) / "papers.sqlite3")
            self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
            self.assertEqual(db.conn.execute("SELECT status FROM roster_runs").fetchone()[0], "incomplete")
            db.conn.close()
            with patch("reference_finder.pmlr.collect_pmlr", return_value=dict(partial, status="complete")) as collect:
                self.assertEqual(self.run_build(directory, *options), 0)
                collect.assert_called_once()

    def test_report_only_does_not_collect(self):
        with tempfile.TemporaryDirectory() as directory, patch("reference_finder.pmlr.collect_pmlr") as collect:
            self.assertEqual(self.run_build(directory, "--report-only", "--venues", "icml"), 0)
            collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
