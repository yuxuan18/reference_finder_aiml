"""Exercise roster/API identity and snapshot safety using real SQLite state."""
import json
import unittest

from reference_finder.roster import RosterStore
from reference_finder.storage import Database


def roster_record(identifier="proceedings/paper1", **changes):
    record = {
        "external_id": identifier, "title": "A practical learning result", "year": 2025,
        "doi": None, "authors": ["Ada Researcher"], "abstract": None,
        "url": "https://proceedings.example/paper/" + identifier,
    }
    record.update(changes)
    return record


def api_record(identifier="api1", **changes):
    record = {
        "paperId": identifier, "title": "A practical learning result", "year": 2025,
        "externalIds": {}, "authors": [{"name": "Ada Researcher"}], "abstract": None,
    }
    record.update(changes)
    return record


def snapshot(*records, status="complete"):
    return {"records": list(records), "status": status, "urls": ["https://proceedings.example/2025"], "note": "Test roster"}


class RosterStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.conn.close)

    def store(self):
        return RosterStore(self.db)

    def rows(self):
        return self.db.conn.execute("SELECT * FROM papers ORDER BY id").fetchall()

    def test_doi_links_api_to_event_despite_title_and_publication_year_difference(self):
        paper_id = self.db.upsert("semantic_scholar", api_record(
            year=2026, title="Earlier API title", externalIds={"DOI": "10.1007/BOOK_1"},
            abstract="API abstract"), "neurips", 2026)
        store = self.store()
        store.save("dblp", "neurips", 2025, snapshot(roster_record(doi="https://doi.org/10.1007/book_1")))
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(store.coverage("neurips", 2025)["abstracts"], 1)
        paper = list(store.papers(["neurips"], 2025, 2025))[0]
        self.assertEqual(paper["id"], paper_id)
        self.assertEqual(paper["event_year"], 2025)
        self.assertEqual(paper["publication_year"], 2026)
        self.assertEqual(paper["title"], "A practical learning result")

    def test_dblp_external_id_links_api_without_doi_across_publication_years(self):
        paper_id = self.db.upsert("semantic_scholar", api_record(
            title="API wording", year=2026, externalIds={"DBLP": "conf/neurips/Example25"},
            abstract="The indexed abstract."), "neurips", 2026)
        store = self.store()
        store.save("dblp", "neurips", 2025, snapshot(roster_record("conf/neurips/Example25")))
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(list(store.papers(["neurips"], 2025, 2025))[0]["id"], paper_id)
        self.assertEqual(store.coverage("neurips", 2025)["abstracts"], 1)

    def test_exact_title_links_non_doi_api_and_official_abstract_survives_later_api_updates(self):
        paper_id = self.db.upsert("semantic_scholar", api_record(authors=[]), "iclr", 2025)
        store = self.store()
        store.save("iclr", "iclr", 2025, snapshot(roster_record(abstract="Official abstract.")))
        for abstract in (None, "Lower-priority API version."):
            with self.subTest(abstract=abstract):
                self.db.upsert("semantic_scholar", api_record(abstract=abstract), "iclr", 2025)
                paper = self.rows()[0]
                self.assertEqual(paper["id"], paper_id)
                self.assertEqual(paper["abstract"], "Official abstract.")
                self.assertEqual(paper["abstract_source"], "iclr")
                self.assertEqual(len(self.rows()), 1)

    def test_same_title_in_adjacent_editions_is_not_identity_without_identifier(self):
        self.db.upsert("semantic_scholar", api_record(year=2024), "icml", 2024)
        store = self.store()
        store.save("pmlr", "icml", 2025, snapshot(roster_record()))
        self.assertEqual(len(self.rows()), 2)
        api_id = self.db.conn.execute("SELECT paper_id FROM source_records WHERE provider='semantic_scholar'").fetchone()[0]
        roster_id = self.db.conn.execute("SELECT paper_id FROM roster_entries").fetchone()[0]
        self.assertNotEqual(api_id, roster_id)

    def test_different_titles_in_same_venue_and_year_remain_distinct(self):
        self.db.upsert("semantic_scholar", api_record(title="A different result"), "icml", 2025)
        store = self.store()
        store.save("pmlr", "icml", 2025, snapshot(roster_record()))
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(store.coverage("icml", 2025)["unverified_api"], 1)

    def test_matching_titles_with_conflicting_dois_are_not_merged(self):
        self.db.upsert("semantic_scholar", api_record(externalIds={"DOI": "10.1007/first"}), "icml", 2025)
        store = self.store()
        store.save("dblp", "icml", 2025, snapshot(roster_record(doi="10.1007/second")))
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual({r["doi"] for r in self.rows()}, {"10.1007/first", "10.1007/second"})

    def test_conflicting_doi_disqualifies_bad_api_dblp_cross_reference(self):
        self.db.upsert("semantic_scholar", api_record(externalIds={
            "DOI": "10.1007/first", "DBLP": "conf/icml/Example25"}), "icml", 2025)
        store = self.store()
        store.save("dblp", "icml", 2025, snapshot(roster_record("conf/icml/Example25", doi="10.1007/second")))
        self.assertEqual(len(self.rows()), 2)
        roster = list(store.papers(["icml"], 2025, 2025))[0]
        self.assertEqual(roster["doi"], "10.1007/second")

    def test_later_identifier_merge_preserves_roster_references_and_official_abstract(self):
        api_id = self.db.upsert("semantic_scholar", api_record(
            title="API preprint title", externalIds={"DOI": "10.1007/linked"}, abstract="API version"), "icml", 2025)
        store = self.store()
        store.save("pmlr", "icml", 2025, snapshot(roster_record(abstract="Official version")))
        roster_id = self.db.conn.execute("SELECT paper_id FROM roster_entries").fetchone()[0]
        self.assertNotEqual(api_id, roster_id)
        self.db.upsert("pmlr", roster_record(doi="10.1007/linked", abstract="Official version"), "icml", 2025)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.db.conn.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(self.db.conn.execute("SELECT paper_id FROM roster_entries").fetchone()[0], api_id)
        self.assertEqual(self.rows()[0]["abstract_source"], "pmlr")
        self.assertEqual(store.coverage("icml", 2025)["listed"], 1)
        self.assertEqual(store.coverage("icml", 2025)["abstracts"], 1)

    def test_cached_dblp_identity_survives_a_database_merge_during_the_import(self):
        api_id = self.db.upsert("semantic_scholar", api_record(
            "with-doi", title="Earlier wording", externalIds={"DOI": "10.1007/linked"}), "icml", 2025)
        self.db.upsert("semantic_scholar", api_record(
            "with-dblp", externalIds={"DBLP": "conf/icml/Example25"}), "icml", 2025)
        store = self.store()
        # A subsequent API record bridges identities after RosterStore built its
        # DBLP lookup; source_records now points at the surviving paper ID.
        self.db.upsert("semantic_scholar", api_record(
            "with-dblp", externalIds={"DOI": "10.1007/linked", "DBLP": "conf/icml/Example25"}), "icml", 2025)
        store.save("dblp", "icml", 2025, snapshot(roster_record("conf/icml/Example25", doi="10.1007/linked")))
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(list(store.papers(["icml"], 2025, 2025))[0]["id"], api_id)

    def test_failed_snapshot_preserves_evidence_and_successful_snapshot_removes_stale_entries(self):
        store = self.store()
        first = roster_record("first", title="First paper")
        second = roster_record("second", title="Second paper")
        store.save("dblp", "icml", 2025, snapshot(first, second))
        self.assertTrue(store.finished("dblp", "icml", 2025))
        store.save("dblp", "icml", 2025, snapshot(status="incomplete"))
        self.assertFalse(store.finished("dblp", "icml", 2025))
        self.assertEqual(store.coverage("icml", 2025)["listed"], 2)
        store.save("dblp", "icml", 2025, snapshot(second))
        self.assertTrue(store.finished("dblp", "icml", 2025))
        self.assertEqual(store.coverage("icml", 2025)["listed"], 1)
        self.assertEqual([r["title"] for r in store.papers(["icml"], 2025, 2025)], ["Second paper"])
        self.assertEqual(len(self.rows()), 2, "Old corpus records must survive roster reconciliation")

    def test_empty_claimed_complete_snapshot_cannot_erase_previous_roster(self):
        store = self.store()
        store.save("dblp", "icml", 2025, snapshot(roster_record()))
        store.save("dblp", "icml", 2025, snapshot())
        self.assertFalse(store.finished("dblp", "icml", 2025))
        self.assertEqual(store.coverage("icml", 2025)["listed"], 1)

    def test_shared_accepted_papers_page_is_not_a_paper_identity(self):
        store = self.store()
        url = "https://ojs.aaai.org/index.php/AAAI/issue/view/1"
        store.save("aaai", "aaai", 2026, snapshot(
            roster_record("a", title="First research paper", year=2026, url=url),
            roster_record("b", title="Second research paper", year=2026, url=url)))
        self.assertEqual(store.coverage("aaai", 2026)["listed"], 2)

    def test_matching_roster_url_avoids_duplicates_when_api_titles_are_ambiguous(self):
        for ident, author in (("one", "A Author"), ("two", "B Author")):
            self.db.upsert("semantic_scholar", api_record(ident, authors=[{"name": author}]), "iclr", 2025)
        store = self.store()
        url = "https://proceedings.iclr.cc/paper_files/paper/2025/hash/example-Abstract-Conference.html"
        store.save("dblp", "iclr", 2025, snapshot(roster_record("conf/iclr/Example25", url=url)))
        store.save("iclr", "iclr", 2025, snapshot(roster_record(url, url=url, abstract="Official.")))
        self.assertEqual(store.coverage("iclr", 2025)["listed"], 1)
        self.assertEqual(store.coverage("iclr", 2025)["abstracts"], 1)

    def test_reviewed_alias_bridges_a_changed_title(self):
        store = self.store()
        store.save("dblp", "icml", 2025, snapshot(roster_record("conf/icml/Published25", title="Published title", doi="10.1007/example")))
        store.aliases[("pmlr", "123", "icml", 2025)] = "conf/icml/Published25"
        store.save("pmlr", "icml", 2025, snapshot(roster_record("123", title="Earlier title", abstract="Official abstract.")))
        self.assertEqual(store.coverage("icml", 2025)["listed"], 1)
        self.assertEqual(store.coverage("icml", 2025)["abstracts"], 1)
        review = json.loads(self.db.conn.execute("SELECT review_json FROM roster_entries WHERE provider='pmlr'").fetchone()[0])
        self.assertEqual(review["roster_match"], {"method": "reviewed_alias", "dblp_key": "conf/icml/Published25", "reason": "Reviewed identity"})

    def test_reviewed_schedule_only_exclusions_adjust_counts_without_removing_api_data(self):
        store = self.store()
        exclusions = [("conf/iclr/Invited18", 2018), ("conf/iclr/Panel20", 2020)]
        for identifier, year in exclusions:
            with self.subTest(identifier=identifier):
                store.reviews[("dblp", identifier, "iclr", year)] = dict(action="exclude",
                    note="Official program identifies an invited program entry.",
                    evidence_urls=["https://iclr.cc/virtual/program"])
                api = api_record(identifier, year=year, title="Schedule-only contribution " + identifier)
                paper_id = self.db.upsert("semantic_scholar", api, "iclr", year)
                original_source = dict(self.db.conn.execute("SELECT * FROM source_records WHERE provider='semantic_scholar' AND external_id=?", (identifier,)).fetchone())
                source = snapshot(roster_record(identifier, year=year, title=api["title"]),
                                  roster_record("research/" + identifier, year=year, title="Research paper " + identifier))
                adjusted = store.save("dblp", "iclr", year, source)
                self.assertEqual(len(source["records"]), 2, "Do not mutate collector/cached source data")
                self.assertEqual(len(adjusted["records"]), 1)
                self.assertEqual(adjusted["status"], "complete")
                run = store.coverage("iclr", year)["runs"][0]
                self.assertEqual(run["record_count"], 1)
                self.assertIn(identifier, run["note"])
                self.assertIn("Reviewed exclusions (1)", run["note"])
                self.assertIn(store.reviews[("dblp", identifier, "iclr", year)]["evidence_urls"][0], json.loads(run["urls_json"]))
                preserved = self.db.conn.execute("SELECT * FROM source_records WHERE provider='semantic_scholar' AND external_id=?", (identifier,)).fetchone()
                self.assertEqual(preserved["paper_id"], paper_id)
                self.assertEqual(dict(preserved), original_source)
                self.assertFalse(self.db.conn.execute("SELECT 1 FROM roster_entries WHERE provider='dblp' AND external_id=?", (identifier,)).fetchone())


    def test_exclusion_is_scoped_to_reviewed_event_year(self):
        store = self.store()
        adjusted = store.save("dblp", "iclr", 2019, snapshot(
            roster_record("conf/iclr/Invited18", year=2019)))
        self.assertEqual(len(adjusted["records"]), 1)
        self.assertNotIn("roster_review", adjusted["records"][0])
        self.assertEqual(store.coverage("iclr", 2019)["listed"], 1)

    def test_reviewed_exclusion_removes_old_roster_membership_on_partial_rescan(self):
        store = self.store()
        review = dict(action="exclude", note="Invited program entry.", evidence_urls=["https://iclr.cc/virtual/program"])
        records = [roster_record("conf/iclr/Invited18", year=2018, title="Invited talk"),
                   roster_record("research/one", year=2018, title="Research paper")]
        store.save("dblp", "iclr", 2018, snapshot(*records))
        self.assertEqual(store.coverage("iclr", 2018)["listed"], 2)
        store.reviews[("dblp", "conf/iclr/Invited18", "iclr", 2018)] = review
        adjusted = store.save("dblp", "iclr", 2018, snapshot(records[0], status="incomplete"))
        self.assertEqual(adjusted["records"], [])
        self.assertEqual(adjusted["status"], "incomplete")
        self.assertEqual(store.coverage("iclr", 2018)["listed"], 1)
        self.assertEqual(len(self.rows()), 2, "Keep old corpus records and their provenance")

    def test_roster_keeps_only_normalized_data_and_curated_review_metadata(self):
        store = self.store()
        record = roster_record(abstract="An abstract stored only on the paper.",
                               unused_response="Unneeded original scraper payload.",
                               roster_review={"action": "annotate", "note": "Reviewed source.",
                                              "evidence_urls": ["https://evidence.example"],
                                              "unused": "Drop this too."})
        store.save("iclr", "iclr", 2025, snapshot(record))
        row = dict(self.db.conn.execute("SELECT * FROM roster_entries").fetchone())
        self.assertNotIn("raw_json", row)
        self.assertEqual(json.loads(row["review_json"]), {"roster_review": {
            "action": "annotate", "note": "Reviewed source.", "evidence_urls": ["https://evidence.example"]}})
        self.assertNotIn(record["abstract"], json.dumps(row))
        self.assertNotIn(record["unused_response"], json.dumps(row))
        self.assertEqual(self.rows()[0]["abstract"], record["abstract"])


if __name__ == "__main__":
    unittest.main()
