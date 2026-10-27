import tempfile
from pathlib import Path
import unittest

from reference_finder.aaai import ARCHIVE, classify_section, collect_aaai
from reference_finder.embeddings import read_corpus
from reference_finder.paper_identity import aaai_article_id, paper_identity
from reference_finder.reconciliation import reconcile
from reference_finder.roster import RosterStore
from reference_finder.storage import Database


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "papers.sqlite3"
        self.db = Database(self.path)
        self.addCleanup(self.db.conn.close)

    def add(self, provider, title, url, abstract=None, venue="icml", doi=None):
        # Separate stores simulate pre-fix collectors which did not recognize
        # a shared publication URL when titles changed.
        pid = self.db.upsert(provider, dict(external_id=title, title=title, url=url,
                             abstract=abstract, doi=doi, year=2025), venue, 2025)
        self.db.conn.execute("INSERT INTO roster_entries VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (provider, title, venue, 2025, pid, title, doi, url, "[]", None, "{}"))
        self.db.conn.commit()
        return pid

    def test_stable_identity_merges_changed_titles_preserving_all_evidence(self):
        first = self.add("dblp", "Original title", "http://proceedings.mlr.press/v267/test25.html")
        self.add("pmlr", "Final title", "https://proceedings.mlr.press/v267/test25.html", "Abstract")
        preview = reconcile(self.db)
        self.assertEqual(len(preview["merges"]), 1)
        self.assertEqual(preview["before"]["papers"], 2)
        result = reconcile(self.db, apply=True)
        self.assertEqual(result["after"]["papers"], 1)
        self.assertEqual(result["before"]["source_records"], result["after"]["source_records"])
        self.assertEqual(result["before"]["roster_entries"], result["after"]["roster_entries"])
        self.assertEqual(self.db.conn.execute("SELECT abstract FROM papers WHERE id=?", (first,)).fetchone()[0], "Abstract")
        self.assertEqual(reconcile(self.db, apply=True)["merges"], [])
        self.assertEqual(self.db.conn.execute("PRAGMA foreign_key_check").fetchall(), [])
        store = RosterStore(self.db)
        store.save("pmlr", "icml", 2025, dict(status="complete", records=[dict(
            external_id="new-id", title="Another final title", year=2025,
            url="https://proceedings.mlr.press/v267/test25.html", abstract="Abstract")]))
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)

    def test_conflicting_dois_and_unrecognized_urls_are_not_merged(self):
        url = "https://proceedings.mlr.press/v267/test25.html"
        self.add("dblp", "Title one", url, doi="10.1234/one")
        self.add("pmlr", "Title two", url, doi="10.1234/two")
        self.add("dblp", "Identical", "https://example.org/one")
        self.add("pmlr", "Identical", "https://example.org/two")
        result = reconcile(self.db, apply=True)
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertEqual(result["merges"], [])
        self.assertEqual(result["after"]["papers"], 4)

    def test_identifier_chain_with_doi_conflict_is_not_partially_merged(self):
        one = "https://proceedings.mlr.press/v267/one25.html"
        two = "https://proceedings.mlr.press/v267/two25.html"
        first = self.add("dblp", "One", one, doi="10.1234/one")
        bridge = self.add("pmlr", "Bridge", one)
        self.add("dblp", "Two", two, doi="10.1234/two")
        with self.db.conn:
            self.db.conn.execute("UPDATE papers SET url=? WHERE id=?", (two, bridge))
        result = reconcile(self.db, apply=True)
        self.assertEqual(result["merges"], [])
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertEqual(result["after"]["papers"], 3)

    def test_unresolved_inventory_counts_a_paper_once_across_source_urls(self):
        self.add("dblp", "Missing abstract", "http://proceedings.mlr.press/v267/test25.html")
        RosterStore(self.db).save("pmlr", "icml", 2025, dict(status="complete", records=[dict(
            external_id="official", title="Missing abstract", year=2025,
            url="https://proceedings.mlr.press/v267/test25.html")]))
        self.assertEqual(len(reconcile(self.db)["unresolved"]), 1)

    def test_match_cache_follows_merges_even_when_sqlite_reuses_deleted_id(self):
        keeper = self.add("dblp", "Original", "https://proceedings.mlr.press/v267/original.html")
        url = "https://proceedings.mlr.press/v267/final.html"
        removed = self.add("pmlr", "Final", url, "Abstract")
        store = RosterStore(self.db)
        with self.db.conn:
            self.db._merge(keeper, removed)
        reused = self.add("dblp", "Unrelated", "https://proceedings.mlr.press/v267/unrelated.html")
        self.assertEqual(reused, removed)
        store.save("pmlr", "icml", 2025, dict(status="incomplete", records=[dict(
            external_id="changed-id", title="Updated final title", year=2025, url=url)]))
        row = self.db.conn.execute("SELECT paper_id FROM source_records WHERE external_id='changed-id'").fetchone()
        self.assertEqual(row[0], keeper)
        self.assertEqual(self.db.conn.execute("SELECT title FROM papers WHERE id=?", (reused,)).fetchone()[0], "Unrelated")

    def test_scope_is_reversible_and_applies_to_report_and_embedding_exports(self):
        pid = self.add("dblp", "Student example", None, venue="aaai", doi="10.1609/aaai.v39i1.123")
        evidence = [dict(article_id="123", year=2025, section="AAAI Student Abstracts",
                         classification="excluded", evidence_url="https://ojs.aaai.org/index.php/AAAI/issue/view/1")]
        result = reconcile(self.db, evidence, apply=True)
        self.assertEqual(result["after"]["papers"], 1)
        self.assertEqual(result["after"]["active_papers"], 0)
        self.assertEqual(RosterStore(self.db).coverage("aaai", 2025)["listed"], 0)
        self.assertEqual(list(read_corpus(self.path)), [])
        with self.db.conn:
            self.db.conn.execute("INSERT INTO paper_venues VALUES (?,?,?,?)", (pid, "icml", 2025, "pmlr"))
        exported = list(read_corpus(self.path))
        self.assertEqual(len(exported), 1)
        self.assertNotIn("aaai", exported[0]["venues_json"])
        evidence[0]["classification"] = "research"
        reconcile(self.db, evidence, apply=True)
        self.assertEqual(RosterStore(self.db).coverage("aaai", 2025)["listed"], 1)

    def test_neurips_alias_and_track_suffix_resolve_same_identifier(self):
        digest = "a" * 32
        self.assertEqual(paper_identity(f"http://papers.nips.cc/paper/2020/hash/{digest}-Abstract.html"),
                         paper_identity(f"https://proceedings.neurips.cc/paper_files/paper/2020/hash/{digest}-Abstract-Conference.html"))
        self.assertIsNone(paper_identity("https://example.org/forum?id=123"))
        self.assertIsNone(aaai_article_id(doi="10.1609/aaai.v33i01.33019522"))

    def test_2019_scope_requires_official_doi_mapping_and_merge_transfers_scope(self):
        doi = "10.1609/aaai.v33i01.33019522"
        pid = self.add("dblp", "Old program title", None, venue="aaai", doi=doi)
        evidence = [dict(article_id="4983", year=2025, doi=doi,
                         section="IAAI Technical Track", classification="excluded", evidence_url="https://example.org/issue")]
        reconcile(self.db, evidence, apply=True)
        self.assertEqual(self.db.conn.execute("SELECT paper_id FROM scope_exclusions").fetchone()[0], pid)
        other = self.add("aaai", "Final program title", "https://ojs.aaai.org/index.php/AAAI/article/view/4983", venue="aaai")
        # A bridge from the official URL gives positive identity evidence.
        with self.db.conn:
            self.db.conn.execute("UPDATE roster_entries SET url=? WHERE paper_id=?", ("https://ojs.aaai.org/index.php/AAAI/article/view/4983", pid))
        result = reconcile(self.db, evidence, apply=True)
        self.assertEqual(len(result["merges"]), 1)
        self.assertIsNone(self.db.conn.execute("SELECT id FROM papers WHERE id=?", (other,)).fetchone())
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM scope_exclusions").fetchone()[0], 1)

    def test_mixed_aaai_issue_research_is_discovered_and_programs_excluded(self):
        archive = '''<html><body><div class="obj_issue_summary"><a class="title"
            href="https://ojs.aaai.org/index.php/AAAI/issue/view/1">IAAI, EAAI, AAAI Special Track and Student Papers</a>
            <div class="series">Vol. 39</div></div></body></html>'''
        fixture = Path(__file__).parent / "fixtures/ml"
        issue = (fixture / "aaai_issue.html").read_text().replace("AAAI Technical Track: Machine Learning", "AAAI Special Track on AI for Social Impact")
        responses = iter([archive, issue, (fixture / "aaai_paper.html").read_text()])
        result = collect_aaai(2025, lambda url: next(responses))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["records"]), 1)
        for label in ("AAAI Journal Track", "AAAI Undergraduate Consortium", "New Faculty Highlights", "Errata"):
            self.assertEqual(classify_section(label), "excluded")
        self.assertEqual(classify_section("Unclassified research"), "unknown")

    def test_associated_issue_is_scanned_but_does_not_add_research(self):
        fixture = Path(__file__).parent / "fixtures/ml"
        archive = (fixture / "aaai_archive.html").read_text().replace(
            '<a class="next" href="' + ARCHIVE + '/2">Next</a>',
            '<div class="obj_issue_summary"><a class="title" '
            'href="https://ojs.aaai.org/index.php/AAAI/issue/view/999">IAAI-25 Technical Papers</a>'
            '<div class="series">Vol. 39 No. 26</div></div>')
        issue = (fixture / "aaai_issue.html").read_text()
        associated = issue.replace("AAAI Technical Track: Machine Learning", "IAAI Technical Papers")
        responses = iter([archive, issue, (fixture / "aaai_paper.html").read_text(), associated])
        requested = []

        def fetch(url):
            requested.append(url)
            return next(responses)

        result = collect_aaai(2025, fetch)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["records"]), 1)
        self.assertTrue(requested[-1].endswith("/999"))


if __name__ == "__main__":
    unittest.main()
