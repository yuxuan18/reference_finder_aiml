"""Regression tests for the OpenReview verification failure observed live."""
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import build_corpus
from reference_finder.api import Budget
from reference_finder.colm import collect_colm
from reference_finder.iclr import collect_iclr
from reference_finder.roster import RosterStore
from reference_finder.storage import Database
from reference_finder.webfetch import AccessChallenge, WebFetcher
from test_ml_proceedings import Fetch, fixture
from test_webfetch import response


class ChallengeTests(unittest.TestCase):
    def test_challenge_is_actionable_and_not_repeated_or_cached(self):
        error = urllib.error.HTTPError("https://api2.openreview.net/notes", 403, "Forbidden", {},
                    io.BytesIO(json.dumps({"name": "ChallengeRequiredError", "message": "private diagnostic"}).encode()))
        with tempfile.TemporaryDirectory() as cache, \
                patch("reference_finder.webfetch.urllib.request.urlopen", side_effect=error) as open_url:
            fetch = WebFetcher(cache, interval=0, budget=Budget(10))
            for query in ("?venue=iclr", "?venue=colm"):
                with self.assertRaises(AccessChallenge) as caught:
                    fetch("https://api2.openreview.net/notes" + query)
                self.assertIn("ChallengeRequiredError", str(caught.exception))
                self.assertIn("OPENREVIEW_TOKEN", str(caught.exception))
                self.assertNotIn("private diagnostic", str(caught.exception))
            self.assertEqual(open_url.call_count, 1)
            self.assertEqual(fetch.budget.used, 1)
            self.assertEqual(fetch.session, {})

    def test_authorization_only_goes_to_openreview_hosts(self):
        with tempfile.TemporaryDirectory() as cache, \
                patch.dict("os.environ", {"OPENREVIEW_TOKEN": "test-private-token"}), \
                patch("reference_finder.webfetch.urllib.request.urlopen",
                      side_effect=[response('{"notes":[],"count":0}'), response()]) as open_url:
            fetch = WebFetcher(cache, interval=0)
            fetch("https://api2.openreview.net/notes")
            fetch("https://proceedings.iclr.cc/")
            api_request, html_request = [call.args[0] for call in open_url.call_args_list]
            self.assertEqual(api_request.get_header("Authorization"), "Bearer test-private-token")
            self.assertNotIn("Authorization", api_request.headers)
            self.assertEqual(api_request.unredirected_hdrs["Authorization"], "Bearer test-private-token")
            self.assertIsNone(html_request.get_header("Authorization"))
            self.assertNotIn("test-private-token", api_request.full_url)

    def test_browser_challenge_html_is_not_cached_as_paper_data(self):
        with tempfile.TemporaryDirectory() as cache, \
                patch("reference_finder.webfetch.urllib.request.urlopen",
                      return_value=response('<html><title>Verifying your browser</title></html>')):
            fetch = WebFetcher(cache, interval=0)
            with self.assertRaisesRegex(Exception, "challenge"):
                fetch("https://proceedings.iclr.cc/")
            self.assertFalse(fetch.session)


class IndependentSourceTests(unittest.TestCase):
    def test_iclr_uses_independent_proceedings_with_abstracts(self):
        catalog = ('<html><body><a href="/paper_files/paper/2025">'
                   'International Conference on Learning Representations 2025</a></body></html>')
        volume = fixture("neurips_volume.html")
        fetch = Fetch(catalog, volume, fixture("pmlr_paper.html"), fixture("pmlr_paper.html"))
        output = collect_iclr(2025, fetch)
        self.assertEqual(output["status"], "complete")
        self.assertEqual(len(output["records"]), 2)
        self.assertTrue(all(r["abstract"] for r in output["records"]))
        self.assertTrue(all(url.startswith("https://proceedings.iclr.cc/") for url in fetch.urls))

    def test_iclr_unavailable_legacy_year_is_explicit(self):
        output = collect_iclr(2017, Fetch("<html><body>ICLR Proceedings</body></html>"))
        self.assertEqual(output["status"], "unavailable")
        self.assertIn("older years require OpenReview", output["note"])

    def test_colm_accepted_list_preserves_note_id_and_authors(self):
        html = '''<html><body><h2>Accepted Papers</h2>
            <p><a href="https://openreview.net/forum?id=accepted">A Language Model</a>
            <br><em>Alice Example, Bob Example</em></p></body></html>'''
        output = collect_colm(2024, Fetch(html))
        self.assertEqual(output["status"], "complete")
        self.assertEqual(output["records"][0]["external_id"], "accepted")
        self.assertEqual(output["records"][0]["authors"], ["Alice Example", "Bob Example"])
        self.assertIsNone(output["records"][0]["abstract"])
        self.assertIn("roster-only", output["note"])
        db = Database(":memory:")
        self.addCleanup(db.conn.close)
        store = RosterStore(db)
        store.save("colm", "colm", 2024, output)
        enriched = dict(output["records"][0], abstract="An official abstract.")
        store.save("openreview", "colm", 2024, dict(output, records=[enriched]))
        self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        self.assertEqual(store.coverage("colm", 2024)["abstracts"], 1)

    def test_new_sources_are_default_and_cli_routes_them(self):
        self.assertIn("iclr", build_corpus.parser().parse_args([]).sources)
        self.assertIn("colm", build_corpus.parser().parse_args([]).sources)
        data = {"records": [{"external_id": "x", "title": "A model", "year": 2024,
                             "abstract": "An abstract"}], "status": "complete"}
        with tempfile.TemporaryDirectory() as tmp, \
                patch("reference_finder.config.load_environment"), \
                patch("reference_finder.iclr.collect_iclr", return_value=data) as iclr, \
                patch("reference_finder.colm.collect_colm", return_value=data) as colm, \
                contextlib.redirect_stdout(io.StringIO()):
            code = build_corpus.main(["--db", tmp + "/papers.db", "--output", tmp + "/report",
                                      "--cache-dir", tmp + "/cache", "--venues", "iclr", "colm",
                                      "--sources", "iclr", "colm", "--start-year", "2024", "--end-year", "2024"])
            self.assertEqual(code, 0)
            iclr.assert_called_once()
            colm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
