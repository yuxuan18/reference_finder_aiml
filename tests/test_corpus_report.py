"""Check exported inventory contents and safe treatment of untrusted titles."""
import contextlib
import csv
from datetime import date
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from reference_finder.corpus_report import export_corpus, status_for
from reference_finder.roster import RosterStore
from reference_finder.storage import Database
from reference_finder.venues import VENUES
from tests.test_roster import api_record, roster_record, snapshot


class CorpusReportTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.conn.close)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.store = RosterStore(self.db)

    def export(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return export_corpus(self.store, [VENUES["icml"]], 2025, 2025, self.output)

    def read_csv(self, name):
        with (self.output / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_roster_totals_deduplicate_sources_and_exclude_api_only_records(self):
        self.db.upsert("semantic_scholar", api_record(
            title="An API-only candidate", abstract="Not proceedings verified"), "icml", 2025)
        self.store.save("dblp", "icml", 2025, snapshot(
            roster_record("dblp/one", title="First paper", doi="10.1007/one"),
            roster_record("dblp/two", title="Second paper", doi="10.1007/two")))
        self.store.save("pmlr", "icml", 2025, snapshot(roster_record(
            "pmlr-one", title="First paper", doi="10.1007/one", abstract="First abstract.")))
        coverage = self.export()
        papers = self.read_csv("papers.csv")
        counts = self.read_csv("coverage.csv")[0]
        self.assertEqual(len(papers), 2)
        self.assertEqual({p["title"] for p in papers}, {"First paper", "Second paper"})
        self.assertEqual(counts["listed_papers"], "2")
        self.assertEqual(counts["with_abstract"], "1")
        self.assertEqual(counts["without_abstract"], "1")
        self.assertEqual(counts["api_only_unverified"], "1")
        self.assertEqual(coverage[0]["listed"], 2)
        first = next(p for p in papers if p["title"] == "First paper")
        self.assertEqual(set(first["roster_sources"].split(",")), {"dblp", "pmlr"})
        self.assertEqual(first["abstract_source"], "pmlr")

    def test_csv_preserves_full_text_proceedings_title_and_evidence(self):
        abstract = 'First paragraph, with "quotes".\nSecond paragraph: αβ.'
        self.store.save("pmlr", "icml", 2025, snapshot(roster_record(
            title='A title, with "quotes" & Unicode α', authors=["Ada Researcher", "Zoë Scientist"],
            abstract=abstract, doi="10.1007/paper")))
        self.db.upsert("semantic_scholar", api_record(title="An API preprint title", externalIds={"DOI": "10.1007/paper"}), "icml", 2025)
        self.export()
        paper = self.read_csv("papers.csv")[0]
        self.assertEqual(paper["title"], 'A title, with "quotes" & Unicode α')
        self.assertEqual(paper["abstract"], abstract)
        self.assertEqual(paper["has_abstract"], "1")
        self.assertIn("https://proceedings.example/paper/", paper["evidence_urls"])
        metadata = json.loads((self.output / "coverage.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["generated_at"])
        self.assertEqual(metadata["coverage"][0]["runs"][0]["provider"], "pmlr")

    def test_html_payload_escapes_script_terminators_and_source_labels(self):
        title = '</script><img src=x onerror="alert(1)"> α & β'
        self.store.save("pmlr", "icml", 2025, snapshot(roster_record(
            title=title, url="javascript:alert(1)", abstract="Available")))
        malicious_provider = '<img src=x onerror="alert(2)">'
        self.db.conn.execute("UPDATE roster_runs SET provider=?", (malicious_provider,))
        self.export()
        page = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertNotIn(title, page)
        self.assertNotIn(malicious_provider, page)
        self.assertIn("&lt;img", page)
        self.assertEqual(page.count("<script>"), 1)
        self.assertEqual(page.count("</script>"), 1)
        payload = re.search(r"const papers=(.*?);let shown=", page, flags=re.S).group(1)
        self.assertEqual(json.loads(payload)[0]["title"], title)
        self.assertIn("\\u003c/script>", payload)
        self.assertIn("td.textContent=value", page)
        self.assertIn("/^https?:\\/\\//.test(url)", page)

    def test_unresolved_source_discrepancy_is_exported_with_the_paper(self):
        record = roster_record(title="A bibliography discrepancy")
        record["roster_review"] = {"note": "Absent from the current official contents; not a confirmed withdrawal.",
                                  "evidence_urls": ["https://proceedings.example/contents.pdf"]}
        self.store.save("dblp", "icml", 2025, snapshot(record))
        self.export()
        self.assertIn("not a confirmed withdrawal", self.read_csv("papers.csv")[0]["review_notes"])
        page = (self.output / "index.html").read_text()
        self.assertIn("not a confirmed withdrawal", page)
        self.assertIn("note.textContent=e.review_note", page)

    def test_source_failure_and_current_year_are_not_reported_as_final_complete(self):
        base = {"listed": 2, "abstracts": 1, "runs": [{"status": "complete"}]}
        self.assertEqual(status_for(base, date.today().year), "current year; provisional")
        self.assertEqual(status_for(base, date.today().year - 1), "source lists retrieved")
        partial = dict(base, runs=[{"status": "complete"}, {"status": "incomplete"}])
        self.assertEqual(status_for(partial, date.today().year - 1), "source lists retrieved; another source incomplete")
        self.assertEqual(status_for(partial, date.today().year), "current year; provisional; another source incomplete")
        failed = dict(base, runs=[{"status": "incomplete"}])
        self.assertEqual(status_for(failed, date.today().year - 1), "incomplete")
        self.assertEqual(status_for(dict(failed, listed=0), date.today().year), "incomplete")
        self.assertEqual(status_for({"listed": 0, "runs": []}, date.today().year), "not attempted")
        empty = {"listed": 0, "runs": [{"status": "unavailable"}]}
        self.assertEqual(status_for(empty, date.today().year), "unavailable / not indexed")


if __name__ == "__main__":
    unittest.main()
