"""Small offline fixtures exercise source formats, membership and failure modes."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from reference_finder.aaai import ARCHIVE, collect_aaai, issue_year
from reference_finder.api import RequestLimit
from reference_finder.neurips import collect_neurips
from reference_finder.openreview import acceptance, collect_openreview
from reference_finder.pmlr import collect_pmlr, discover_volumes
from reference_finder.roster import RosterStore
from reference_finder.storage import Database
from reference_finder.webfetch import FetchError
from reference_finder.proceedings import html_page, paper_page

FIXTURES = Path(__file__).parent / "fixtures" / "ml"


def fixture(name):
    return (FIXTURES / name).read_text()


class Fetch:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value


class PmlrTests(unittest.TestCase):
    def test_broken_quoted_citation_title_uses_verified_article_heading(self):
        raw = '''<html><head><meta name="citation_title" content=""Why?": A Test">
            <meta name="citation_author" content="Alice"></head><body>
            <h1>"Why?": A Test</h1><div id="abstract">Full abstract.</div></body></html>'''
        output = paper_page(raw, "https://proceedings.mlr.press/v267/test.html", 2025, "#abstract")
        self.assertEqual(output["title"], '"Why?": A Test')
        self.assertEqual(output["abstract"], "Full abstract.")

    def test_captcha_in_paper_title_is_not_an_access_challenge(self):
        raw = fixture("pmlr_paper.html").replace("Learning with Dependent Experts", "CAPTCHA Solving for Agents")
        self.assertIsNotNone(html_page(raw))
        with self.assertRaisesRegex(ValueError, "challenge"):
            html_page("<html><head><title>CAPTCHA: Access denied</title></head></html>")
    def test_catalog_excludes_affiliated_workshops(self):
        self.assertEqual(discover_volumes(fixture("pmlr_catalog.html"), 2025),
                         ["https://proceedings.mlr.press/v267/"])
        self.assertEqual(discover_volumes(fixture("pmlr_catalog.html"), 2017),
                         ["https://proceedings.mlr.press/v70/"])

    def test_collect_title_abstract_and_authors_without_doi(self):
        fetch = Fetch(fixture("pmlr_catalog.html"), fixture("pmlr_volume.html"), fixture("pmlr_paper.html"))
        output = collect_pmlr(2025, fetch)
        self.assertEqual(output["status"], "complete")
        record = output["records"][0]
        self.assertEqual(record["abstract"], "We study dependent experts.")
        self.assertEqual(record["authors"], ["Alice Example", "Bob Example"])
        self.assertIsNone(record["doi"])
        self.assertEqual(record["url"], "https://proceedings.mlr.press/v267/experts25a.html")

    def test_unpublished_year_is_unavailable(self):
        self.assertEqual(collect_pmlr(2026, Fetch(fixture("pmlr_catalog.html")))["status"], "unavailable")

    def test_missing_abstract_is_a_valid_title_only_paper(self):
        raw = fixture("pmlr_paper.html").replace('id="abstract"', 'id="none"').replace("<h4>Abstract</h4>", "")
        output = collect_pmlr(2025, Fetch(fixture("pmlr_catalog.html"), fixture("pmlr_volume.html"), raw))
        self.assertEqual(output["status"], "complete")
        self.assertIsNone(output["records"][0]["abstract"])

    def test_failed_article_preserves_its_roster_membership(self):
        output = collect_pmlr(2025, Fetch(fixture("pmlr_catalog.html"), fixture("pmlr_volume.html"), FetchError("403")))
        self.assertEqual(output["status"], "incomplete")
        self.assertEqual(len(output["records"]), 1)
        self.assertIsNone(output["records"][0]["abstract"])

    def test_wrong_volume_and_truncation_cannot_be_complete(self):
        for raw in (fixture("pmlr_volume.html").replace("2025", "2024"),
                    fixture("pmlr_volume.html").replace("</html>", "")):
            with self.subTest(raw=raw):
                output = collect_pmlr(2025, Fetch(fixture("pmlr_catalog.html"), raw))
                self.assertEqual(output["status"], "incomplete")

    def test_request_limit_carries_partial_result(self):
        with self.assertRaises(RequestLimit) as caught:
            collect_pmlr(2025, Fetch(fixture("pmlr_catalog.html"), fixture("pmlr_volume.html"), RequestLimit("paused")))
        self.assertEqual(len(caught.exception.partial_result["records"]), 1)
        self.assertEqual(caught.exception.partial_result["status"], "incomplete")


class NeuripsTests(unittest.TestCase):
    def test_main_volume_discovery_and_multiple_published_tracks(self):
        modern = fixture("pmlr_paper.html").replace('id="abstract"', 'class="paper-abstract"')
        fetch = Fetch(fixture("neurips_catalog.html"), fixture("neurips_volume.html"), modern, modern)
        output = collect_neurips(2025, fetch)
        self.assertEqual(output["status"], "complete")
        self.assertEqual(len(output["records"]), 2)
        self.assertTrue(fetch.urls[1].endswith("vol38-main-conference"))
        self.assertIn("Datasets_and_Benchmarks_Track", fetch.urls[-1])

    def test_legacy_volume_and_abstract_layout(self):
        volume = '<html><body><ul><li><a href="/paper/2017/hash/abc-Abstract.html">A paper</a></li></ul></body></html>'
        output = collect_neurips(2017, Fetch(fixture("neurips_catalog.html"), volume, fixture("neurips_legacy_paper.html")))
        self.assertEqual(output["status"], "complete")
        self.assertEqual(output["records"][0]["abstract"], "We study dependent experts.")

    def test_bad_counts_and_error_documents_cannot_be_complete(self):
        for raw in (fixture("neurips_volume.html").replace("2 papers", "3 papers"),
                    "<html><head><title>Access denied</title></head></html>",
                    fixture("neurips_volume.html").replace("</html>", "")):
            with self.subTest(raw=raw):
                output = collect_neurips(2025, Fetch(fixture("neurips_catalog.html"), raw))
                self.assertEqual(output["status"], "incomplete")

    def test_unpublished_year_is_unavailable(self):
        self.assertEqual(collect_neurips(2026, Fetch(fixture("neurips_catalog.html")))["status"], "unavailable")


class OpenReviewTests(unittest.TestCase):
    def test_iclr_2018_acceptance_decision_invitation(self):
        data = fixture("openreview_v1.json").replace("2023", "2018").replace("/-/Decision", "/-/Acceptance_Decision")
        output = collect_openreview("iclr", 2018, Fetch(data))
        self.assertEqual(output["status"], "complete")
        self.assertEqual([r["external_id"] for r in output["records"]], ["accepted"])

    def test_only_2019_official_ac_meta_recommendation_is_a_decision(self):
        group = "ICLR.cc/2019/Conference"
        reply = {"invitation": group + "/Paper1/-/Meta_Review",
                 "signatures": [group + "/Paper1/Area_Chair1"], "content": {"recommendation": "Accept (Poster)"}}
        note = {"content": {}, "details": {"directReplies": [reply]}}
        self.assertTrue(acceptance(note, group, 1))
        reply["content"]["recommendation"] = "Reject"
        self.assertFalse(acceptance(note, group, 1))
        reply["content"]["recommendation"] = "Accept (Poster)"
        reply["signatures"] = [group + "/Paper1/AnonReviewer1"]
        self.assertIsNone(acceptance(note, group, 1))
        self.assertIsNone(acceptance(note, "ICLR.cc/2020/Conference", 1))
    def test_v2_accepted_notes_and_missing_abstract(self):
        fetch = Fetch(fixture("openreview_v2.json"))
        output = collect_openreview("colm", 2024, fetch)
        self.assertEqual(output["status"], "complete")
        self.assertEqual(len(output["records"]), 2)
        self.assertIsNone(output["records"][1]["abstract"])
        query = parse_qs(urlsplit(fetch.urls[0]).query)
        self.assertEqual(query["content.venueid"], ["colmweb.org/COLM/2024/Conference"])
        self.assertEqual(urlsplit(fetch.urls[0]).hostname, "api2.openreview.net")
        self.assertEqual(query["count"], ["true"])
        self.assertNotIn("offset", query)

    def test_legacy_official_decisions(self):
        fetch = Fetch(fixture("openreview_v1.json"))
        output = collect_openreview("iclr", 2023, fetch)
        self.assertEqual(output["status"], "complete")
        self.assertEqual([r["external_id"] for r in output["records"]], ["accepted"])
        self.assertEqual(urlsplit(fetch.urls[0]).hostname, "api.openreview.net")
        self.assertEqual(parse_qs(urlsplit(fetch.urls[0]).query)["details"], ["directReplies"])
        self.assertNotIn("count", parse_qs(urlsplit(fetch.urls[0]).query))

    def test_legacy_venueid_alone_does_not_prove_acceptance(self):
        data = json.loads(fixture("openreview_v1.json"))
        group = "ICLR.cc/2023/Conference"
        for note in data["notes"][:2]:
            note["content"]["venueid"] = group
        output = collect_openreview("iclr", 2023, Fetch(json.dumps(data)))
        self.assertEqual([r["external_id"] for r in output["records"]], ["accepted"])
        self.assertIsNone(acceptance({"content": {"venueid": group}}, group, 1))

    def test_2017_uses_lowercase_group_and_submission_invitation(self):
        fetch = Fetch('{"count":0,"notes":[]}')
        collect_openreview("iclr", 2017, fetch)
        self.assertEqual(parse_qs(urlsplit(fetch.urls[0]).query)["invitation"],
                         ["ICLR.cc/2017/conference/-/submission"])

    def test_legacy_plain_author_strings(self):
        data = json.loads(fixture("openreview_v1.json"))
        data["notes"][0]["content"]["authors"] = "Alice Example, Bob Example"
        output = collect_openreview("iclr", 2023, Fetch(json.dumps(data)))
        self.assertEqual(output["status"], "complete")
        self.assertEqual(output["records"][0]["authors"], ["Alice Example", "Bob Example"])

    def test_blank_title_does_not_prevent_later_valid_abstract_retrieval(self):
        data = json.loads(fixture("openreview_v2.json"))
        blank = dict(data["notes"][0], id="blank", content=dict(data["notes"][0]["content"], title={"value": ""}))
        data["notes"].insert(0, blank)
        data["count"] += 1
        output = collect_openreview("colm", 2024, Fetch(json.dumps(data)))
        self.assertEqual(output["status"], "incomplete")
        self.assertEqual(len(output["records"]), 2)
        self.assertIn("1 accepted notes", output["note"])

    def test_rejected_withdrawn_under_review_and_unknown_are_never_accepted(self):
        group = "ICLR.cc/2025/Conference"
        for status in ("Rejected_Submission", "Withdrawn_Submission", "Submission", "Desk_Rejected_Submission"):
            self.assertFalse(acceptance({"content": {"venueid": {"value": group + "/" + status}}}, group, 2))
        self.assertFalse(acceptance({"ddate": 100, "content": {"venueid": group}}, group, 2))
        self.assertIsNone(acceptance({"content": {}}, group, 2))

    def test_review_rating_and_other_venues_decisions_do_not_establish_acceptance(self):
        note = {"content": {}, "details": {"directReplies": [
            {"invitation": "ICLR.cc/2023/Conference/Paper1/-/Official_Review", "content": {"decision": "Accept"}},
            {"invitation": "Other/Venue/-/Decision", "content": {"decision": "Accept"}},
        ]}}
        self.assertIsNone(acceptance(note, "ICLR.cc/2023/Conference", 1))
        output = collect_openreview("iclr", 2023, Fetch(json.dumps({"count": 1, "notes": [dict(note, id="x")]})))
        self.assertEqual(output["status"], "incomplete")
        self.assertEqual(output["records"], [])

    def test_pagination_checks_counts_and_repetition(self):
        data = json.loads(fixture("openreview_v2.json"))
        first = json.dumps(dict(data, notes=data["notes"][:1]))
        last = json.dumps(dict(data, notes=data["notes"][1:]))
        fetch = Fetch(first, last)
        with patch("reference_finder.openreview.PAGE_SIZE", 1):
            self.assertEqual(collect_openreview("colm", 2024, fetch)["status"], "complete")
        self.assertEqual(parse_qs(urlsplit(fetch.urls[1]).query)["offset"], ["1"])
        for second in (first, '{"count":3,"notes":[]}', '{"count":2,"notes":[]}'):
            with self.subTest(second=second), patch("reference_finder.openreview.PAGE_SIZE", 1):
                output = collect_openreview("colm", 2024, Fetch(first, second))
                self.assertEqual(output["status"], "incomplete")
                self.assertEqual(len(output["records"]), 1)

    def test_api2_omits_count_on_following_pages(self):
        data = json.loads(fixture("openreview_v2.json"))
        first = json.dumps(dict(data, notes=data["notes"][:1]))
        last = json.dumps({"notes": data["notes"][1:]})
        with patch("reference_finder.openreview.PAGE_SIZE", 1):
            output = collect_openreview("colm", 2024, Fetch(first, last))
        self.assertEqual(output["status"], "complete")
        self.assertEqual(len(output["records"]), 2)

    def test_invalid_payload_and_blocked_access_are_incomplete(self):
        for raw in ("{}", "<html>blocked</html>", '{"count":1,"notes":[{}]}', FetchError("403")):
            with self.subTest(raw=raw):
                self.assertEqual(collect_openreview("colm", 2024, Fetch(raw))["status"], "incomplete")

    def test_colm_before_first_edition_does_not_fetch(self):
        fetch = Fetch()
        self.assertEqual(collect_openreview("colm", 2023, fetch)["status"], "not_applicable")
        self.assertEqual(fetch.urls, [])


class AaaiTests(unittest.TestCase):
    def test_2019_combined_proceedings_issue_is_not_excluded(self):
        archive = '''<html><body><div class="obj_issue_summary">
            <a class="title" href="https://ojs.aaai.org/index.php/AAAI/issue/view/246">AAAI-19, IAAI-19, EAAI-20</a>
            <div class="series">Vol. 33 No. 01</div></div></body></html>'''
        output = collect_aaai(2019, Fetch(archive, fixture("aaai_issue.html"), fixture("aaai_paper.html")))
        self.assertEqual(output["status"], "complete")
        self.assertEqual(len(output["records"]), 1)

    def test_legacy_numbered_associated_events_are_excluded(self):
        issue = fixture("aaai_issue.html").replace("IAAI Technical Papers", "IAAI18 - Deployed")
        issue = issue.replace("AAAI Student Abstracts", "EAAI18 - Full Papers")
        output = collect_aaai(2017, Fetch(fixture("aaai_archive_legacy.html"), issue, fixture("aaai_paper.html")))
        self.assertEqual(output["status"], "complete")
        self.assertEqual(len(output["records"]), 1)
    def test_archive_pagination_and_main_research_sections(self):
        fetch = Fetch(fixture("aaai_archive.html"), fixture("aaai_archive_legacy.html"),
                      fixture("aaai_issue.html"), fixture("aaai_paper.html"))
        output = collect_aaai(2025, fetch)
        self.assertEqual(output["status"], "complete")
        self.assertEqual(len(output["records"]), 1)
        self.assertEqual(output["records"][0]["doi"], "10.1609/aaai.example")
        self.assertEqual(output["records"][0]["abstract"], "We study dependent experts. We give a bound.")
        self.assertTrue(fetch.urls[1].endswith("/archive/2"))
        self.assertFalse(any("999" in u or "456" in u or "789" in u for u in fetch.urls))

    def test_migrated_2017_issue(self):
        issue = fixture("aaai_issue.html").replace("AAAI Technical Track: Machine Learning", "Machine Learning Methods")
        fetch = Fetch(fixture("aaai_archive_legacy.html"), issue, fixture("aaai_paper.html"))
        output = collect_aaai(2017, fetch)
        self.assertEqual(output["status"], "complete")
        self.assertIn("/issue/view/302", fetch.urls[1])
        self.assertEqual(output["records"][0]["year"], 2017)
        self.assertEqual(issue_year("Vol. 32 No. 1 (2018): AAAI"), 2018)

    def test_historical_main_and_special_tracks(self):
        for label in ("Main Track: Machine Learning Applications", "Special Track on Cognitive Systems",
                      "Cognitive Modeling and Cognitive Systems", "Human-Aware Artificial Intelligence"):
            with self.subTest(label=label):
                issue = fixture("aaai_issue.html").replace("AAAI Technical Track: Machine Learning", label)
                fetch = Fetch(fixture("aaai_archive_legacy.html"), issue, fixture("aaai_paper.html"))
                self.assertEqual(collect_aaai(2017, fetch)["status"], "complete")

    def test_articles_heading_in_technical_issue(self):
        issue = fixture("aaai_issue.html").replace("AAAI Technical Track: Machine Learning", "Articles")
        output = collect_aaai(2025, Fetch(fixture("aaai_archive.html"), fixture("aaai_archive_legacy.html"),
                                         issue, fixture("aaai_paper.html")))
        self.assertEqual(output["status"], "complete")

    def test_unknown_sections_and_archive_loop_are_incomplete(self):
        issue = fixture("aaai_issue.html").replace("AAAI Technical Track: Machine Learning", "Unclassified")
        output = collect_aaai(2025, Fetch(fixture("aaai_archive.html"), fixture("aaai_archive_legacy.html"), issue))
        self.assertEqual(output["status"], "incomplete")
        self.assertEqual(output["records"], [])
        loop = fixture("aaai_archive.html").replace(ARCHIVE + "/2", ARCHIVE)
        self.assertEqual(collect_aaai(2025, Fetch(loop))["status"], "incomplete")


class MlStorageTests(unittest.TestCase):
    def test_official_abstract_wins_and_rosters_deduplicate_without_doi(self):
        db = Database(":memory:")
        self.addCleanup(db.conn.close)
        store = RosterStore(db)
        record = {"external_id": "conf/icml/Example25", "title": "Learning with Dependent Experts",
                  "year": 2025, "authors": ["Alice Example"], "abstract": None}
        store.save("dblp", "icml", 2025, {"records": [record], "status": "complete"})
        official = dict(record, external_id="https://proceedings.mlr.press/v267/example25.html", abstract="Official abstract")
        store.save("pmlr", "icml", 2025, {"records": [official], "status": "complete"})
        db.upsert("semantic_scholar", {"paperId": "s2", "title": record["title"], "year": 2025,
                  "authors": [{"name": "Alice Example"}], "abstract": "API abstract"}, "icml", 2025)
        self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        saved = db.conn.execute("SELECT * FROM papers").fetchone()
        self.assertEqual(saved["abstract"], "Official abstract")
        self.assertEqual(saved["abstract_source"], "pmlr")
        self.assertEqual(store.coverage("icml", 2025)["listed"], 1)


if __name__ == "__main__":
    unittest.main()
