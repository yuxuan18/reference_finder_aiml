"""Coverage reports must not confuse successful queries with full coverage."""

from contextlib import redirect_stdout
import io
import json
import unittest

from reference_finder.importer import job_key
from reference_finder.reporting import report
from reference_finder.storage import Database
from reference_finder.venues import VENUES


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.conn.close)
        self.venue = VENUES["icml"]

    def output(self, venues=None, start=2024, end=2024):
        output = io.StringIO()
        with redirect_stdout(output):
            report(self.db, venues or [self.venue], start, end)
        return output.getvalue()

    def finish_source_searches(self, venue):
        for query in venue.openalex_queries:
            self.db.save_checkpoint(job_key("oa-sources", venue.key, query=query), None, True, 0, 0)

    def test_no_jobs_are_reported_as_unfinished_not_empty_completed_searches(self):
        output = self.output(end=2025)
        self.assertIn("0/2 conference-years finished; 2 unfinished or not run", output)
        self.assertIn("OpenAlex source searches: 0/2 finished", output)
        self.assertIn("venues with unfinished searches: icml.", output)
        self.assertIn("OpenAlex no matching source after finished searches: none.", output)
        self.assertIn("absent papers are not counted", output)

    def test_completed_empty_jobs_are_distinct_from_unfinished_jobs(self):
        self.db.save_checkpoint(job_key("s2-discover", "icml", year=2024), None, True, 0, 0)
        self.finish_source_searches(self.venue)
        output = self.output()
        self.assertIn("1/1 conference-years finished; 0 unfinished or not run", output)
        self.assertIn("OpenAlex source searches: 2/2 finished", output)
        self.assertIn("venues with unfinished searches: none.", output)
        self.assertIn("OpenAlex no matching source after finished searches: icml.", output)
        self.assertIn("no matching source does not mean no papers exist", output)

    def test_legacy_s2_checkpoint_is_not_current_completion(self):
        legacy = json.dumps({"version": 1, "stage": "s2-discover", "venue": "icml", "year": 2024},
                            sort_keys=True)
        self.assertNotEqual(legacy, job_key("s2-discover", "icml", year=2024))
        self.db.save_checkpoint(legacy, None, True, 347, 1)
        output = self.output()
        self.assertIn("0/1 conference-years finished; 1 unfinished or not run", output)

    def test_every_known_source_and_selected_year_requires_a_finished_job(self):
        for source in ("https://openalex.org/S1", "https://openalex.org/S2"):
            self.db.conn.execute("INSERT INTO openalex_sources VALUES (?,?,?)", ("icml", source, "ICML"))
        self.finish_source_searches(self.venue)
        self.db.save_checkpoint(job_key("oa-discover", "icml", year=2024,
                                       source="https://openalex.org/S1"), None, True, 0, 0)
        self.db.save_checkpoint(job_key("oa-discover", "icml", year=2025,
                                       source="https://openalex.org/S1"), "next-page", False, 100, 100)
        self.db.save_checkpoint(job_key("oa-discover", "icml", year=2023,
                                       source="https://openalex.org/S2"), None, True, 0, 0)
        output = self.output(end=2025)
        self.assertIn("OpenAlex discovery: 1/4 known-source/year jobs finished; 3 unfinished or not run", output)
        self.assertIn("OpenAlex no matching source after finished searches: none.", output)

    def test_one_unfinished_source_query_keeps_venue_in_unfinished_list(self):
        first_query = self.venue.openalex_queries[0]
        self.db.save_checkpoint(job_key("oa-sources", "icml", query=first_query), None, True, 0, 0)
        iclr = VENUES["iclr"]
        self.finish_source_searches(iclr)
        output = self.output(venues=[self.venue, iclr])
        self.assertIn("venues with unfinished searches: icml.", output)
        self.assertIn("OpenAlex no matching source after finished searches: iclr.", output)

    def test_counts_explain_no_abstract_and_report_does_not_write(self):
        for number, abstract, doi in ((1, "Present.", None), (2, None, "10.1145/example"), (3, None, None)):
            self.db.upsert("semantic_scholar", {
                "paperId": str(number), "title": f"A paper {number}", "year": 2024,
                "abstract": abstract, "externalIds": {"DOI": doi},
            }, "icml", 2024)
        self.db.conn.commit()
        self.db.conn.execute("PRAGMA query_only=ON")
        changes = self.db.conn.total_changes
        output = self.output()
        row = next(line for line in output.splitlines() if line.startswith("icml "))
        self.assertEqual(row.split(), ["icml", "2024", "3", "1", "2", "1"])
        self.assertIn("No abstract", output)
        self.assertNotIn("Missing", output)
        self.assertEqual(changes, self.db.conn.total_changes)


if __name__ == "__main__":
    unittest.main()
