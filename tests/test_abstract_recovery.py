import json
import unittest

from reference_finder.abstract_recovery import api_match, recover_missing
from reference_finder.api import RequestLimit
from reference_finder.reconciliation import reconcile
from reference_finder.roster import RosterStore
from reference_finder.storage import Database
from reference_finder.webfetch import FetchError


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.conn.close)
        self.store = RosterStore(self.db)

    def paper(self, title="Test paper", url="https://proceedings.mlr.press/v267/test.html", abstract=None):
        self.store.save("dblp", "icml", 2025, dict(status="complete", records=[dict(
            external_id="conf/icml/Test25", title=title, year=2025, url=url,
            authors=["Alice Example", "Bob Researcher"], abstract=abstract)]))
        return self.db.conn.execute("SELECT id FROM papers").fetchone()[0]

    def test_snapshot_enriches_known_id_without_changing_roster_scan(self):
        pid = self.paper()
        scan = tuple(self.db.conn.execute("SELECT * FROM roster_runs").fetchone())
        snapshots = [dict(dblp_key="conf/icml/Test25", provider="pmlr", external_id="archived-paper",
                          url="https://proceedings.mlr.press/v267/test.html", evidence_url="https://example.org/pinned",
                          abstract="Verified publisher abstract.")]
        results = recover_missing(self.db, lambda url: self.fail("No HTTP needed"), apply=True, snapshots=snapshots)
        self.assertEqual(results[0]["status"], "recovered")
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        self.assertEqual(self.db.conn.execute("SELECT id,abstract_source FROM papers").fetchone()[:], (pid, "pmlr"))
        self.assertEqual(tuple(self.db.conn.execute("SELECT * FROM roster_runs").fetchone()), scan)
        self.assertEqual(self.store.coverage("icml", 2025)["abstracts"], 1)
        self.assertEqual(recover_missing(self.db, lambda url: self.fail("Already complete"), apply=True, snapshots=snapshots), [])

    def test_dry_run_and_request_limit_do_not_write_abstracts(self):
        self.paper()
        snapshot = [dict(dblp_key="conf/icml/Test25", provider="pmlr", external_id="test",
                         url="https://example.org/test", evidence_url="https://example.org/pinned", abstract="Abstract")]
        results = recover_missing(self.db, lambda url: None, snapshots=snapshot)
        self.assertEqual(results[0]["status"], "available")
        self.assertIsNone(self.db.conn.execute("SELECT abstract FROM papers").fetchone()[0])

        def limited(url):
            raise RequestLimit("bounded")

        self.assertEqual(recover_missing(self.db, limited, apply=True, snapshots=[])[0]["status"], "request limit")

    def test_title_search_requires_authors_year_and_compatible_doi(self):
        paper = dict(title="A & B", authors_json='["Alice Example", "Bob Researcher 0001"]',
                     doi="10.1234/published", publication_year=2025, event_year=2025)
        record = dict(display_name="A &amp; B", publication_year=2025, doi="10.1234/published",
                      authorships=[{"author": {"display_name": "Example, Alice"}},
                                   {"author": {"display_name": "Bob Researcher"}}])
        self.assertTrue(api_match(paper, record))
        for changed in (dict(record, publication_year=2024), dict(record, doi="10.1234/other"),
                        dict(record, authorships=[{"author": {"display_name": "Other Person"}}]),
                        dict(record, display_name="A different paper"), dict(record, publication_year=None)):
            self.assertFalse(api_match(paper, changed))

    def test_api_enrichment_is_labeled_as_api_and_preserves_bibliography(self):
        self.paper()
        record = dict(id="https://openalex.org/W123", display_name="Test paper", publication_year=2025,
                      abstract_inverted_index={"API": [0], "abstract": [1]}, authorships=[
                          {"author": {"display_name": "Alice Example"}}, {"author": {"display_name": "Bob Researcher"}}])

        class API:
            def request(self, path, params):
                return {"results": [record]}

        class OA:
            http = API()

        def missing(url):
            raise FetchError("HTTP 404")

        recovered = recover_missing(self.db, missing, OA(), apply=True, snapshots=[])
        self.assertEqual(recovered[0]["provider"], "openalex")
        paper = self.db.conn.execute("SELECT * FROM papers").fetchone()
        self.assertEqual(paper["abstract_source"], "openalex")
        self.assertEqual(paper["title"], "Test paper")
        self.assertEqual(paper["abstract"], "API abstract")

    def test_reviewed_alias_and_scope_review_preserve_all_source_keys(self):
        first = self.paper(title="Published title")
        other = self.db.upsert("openreview", dict(external_id="note", title="Revised title", year=2025,
                                                abstract="Abstract", url="https://openreview.net/forum?id=note"), "icml", 2025)
        self.db.conn.execute("INSERT INTO roster_entries VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            "openreview", "note", "icml", 2025, other, "Revised title", None,
            "https://openreview.net/forum?id=note", "[]", None, "{}"))
        self.db.conn.commit()
        alias = dict(provider="openreview", external_id="note", venue="icml", year=2025,
                     dblp_key="conf/icml/Test25", reason="Reviewed primary sources")
        review = dict(provider="dblp", external_id="conf/icml/Test25", venue="icml", year=2025,
                      action="exclude", section="Retracted paper", reason="Publisher retracted it",
                      evidence_urls=["https://example.org/publisher-notice"])
        result = reconcile(self.db, apply=True, aliases=[alias], reviews=[review])
        self.assertEqual(len(result["merges"]), 1)
        self.assertEqual(result["after"]["source_records"], 2)
        self.assertEqual(result["after"]["roster_entries"], 2)
        self.assertEqual(result["after"]["active_papers"], 0)
        self.assertEqual(self.db.conn.execute("SELECT paper_id FROM scope_exclusions").fetchone()[0], first)
        self.assertEqual(json.loads(self.db.conn.execute("SELECT review_json FROM roster_entries LIMIT 1").fetchone()[0])["roster_review"]["action"], "exclude")

    def test_rejected_openreview_note_requires_review_before_api_fallback(self):
        self.store.save("dblp", "iclr", 2025, dict(status="complete", records=[dict(
            external_id="test", title="Test", year=2025, url="https://openreview.net/forum?id=x")]))
        note = dict(id="x", content={"venueid": {"value": "ICLR.cc/2025/Conference/Withdrawn_Submission"},
                                     "abstract": {"value": "Draft abstract"}})
        class OA:
            @property
            def http(self):
                raise AssertionError("Negative venue status must not be hidden by API fallback")
        result = recover_missing(self.db, lambda url: json.dumps({"notes": [note]}), OA(), apply=True, snapshots=[])
        self.assertEqual(result[0]["status"], "review required")
        self.assertIsNone(self.db.conn.execute("SELECT abstract FROM papers").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
