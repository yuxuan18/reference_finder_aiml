"""Offline contracts for the public-source fetcher's cache and request budget."""
import hashlib
import gzip
import io
import json
from email.message import Message
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from reference_finder.api import Budget, RequestLimit
from reference_finder.webfetch import FetchError, USER_AGENT, WebFetcher


URL = "https://proceedings.iclr.cc/paper_files/paper/2025"
SPARQL_URL = "https://sparql.dblp.org/sparql?query=test&format=json"
HTML = "<html><body>A public proceedings list.</body></html>"


def response(body=HTML):
    result = io.BytesIO(body.encode("utf-8"))
    result.headers = Message()
    result.headers["Content-Type"] = "text/html; charset=utf-8"
    return result


def http_error(code, retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError(URL, code, "server message", headers, io.BytesIO(b"private diagnostic"))


class WebFetcherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name)
        self.urlopen = self.enter_patch("reference_finder.webfetch.urllib.request.urlopen")
        self.sleep = self.enter_patch("reference_finder.webfetch.time.sleep")
        self.enter_patch("reference_finder.webfetch.time.monotonic", return_value=1000.0)
        self.enter_patch("reference_finder.webfetch.time.time", return_value=1234.0)

    def enter_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def fetcher(self, **kwargs):
        return WebFetcher(cache_dir=self.cache, interval=0, **kwargs)

    def cache_path(self, url=URL):
        return self.cache / (hashlib.sha256(url.encode()).hexdigest() + ".json")

    def test_memory_and_disk_hits_avoid_network_and_budget(self):
        self.urlopen.return_value = response()
        first = self.fetcher()
        self.assertEqual(first(URL), HTML)
        self.assertEqual(first(URL), HTML)
        second = self.fetcher(budget=Budget(0))
        self.assertEqual(second(URL), HTML)
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertEqual(first.budget.used, 1)
        self.assertEqual(second.budget.used, 0)
        saved = json.loads(self.cache_path().read_text())
        self.assertEqual(saved["url"], URL)
        self.assertEqual(saved["body"], HTML)
        self.assertEqual(saved["fetched_at"], 1234.0)

    def test_refresh_refetches_disk_cache_but_reuses_current_session(self):
        self.cache_path().write_text(json.dumps({"url": URL, "body": "Old version"}))
        self.urlopen.return_value = response()
        fetch = self.fetcher(refresh=True)
        self.assertEqual(fetch(URL), HTML)
        self.assertEqual(fetch(URL), HTML)
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertEqual(json.loads(self.cache_path().read_text())["body"], HTML)

    def test_url_allowlist_rejects_other_hosts_and_insecure_http_before_request(self):
        fetch = self.fetcher()
        for url in ("http://proceedings.iclr.cc/example", "https://example.org/", "file:///tmp/example",
                    "https://proceedings.iclr.cc.attacker.example/", "https://dblp.org/db/conf/icml/"):
            with self.subTest(url=url), self.assertRaises(FetchError):
                fetch(url)
        self.urlopen.assert_not_called()
        self.assertEqual(fetch.budget.used, 0)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_honest_identifying_agent_and_service_accept_headers(self):
        self.urlopen.side_effect = [response(), response('{"results":{"bindings":[]}}')]
        fetch = self.fetcher(timeout=12)
        fetch(URL)
        fetch(SPARQL_URL)
        first, second = self.urlopen.call_args_list
        self.assertEqual(first.args[0].get_header("User-agent"), USER_AGENT)
        self.assertEqual(first.args[0].get_header("Accept"), "text/html")
        self.assertEqual(second.args[0].get_header("Accept"), "application/sparql-results+json")
        self.assertEqual(first.kwargs["timeout"], 12)

    def test_challenges_never_enter_session_or_disk_cache(self):
        for challenge in ("anubis_version", "anubis_challenge", "Making sure you're not a bot",
                          "Just a moment...", "cf-chl-test"):
            with self.subTest(challenge=challenge):
                self.urlopen.reset_mock()
                self.urlopen.return_value = response("<html>" + challenge + "</html>")
                fetch = self.fetcher()
                with self.assertRaisesRegex(FetchError, "challenge"):
                    fetch(URL)
                self.assertEqual(self.urlopen.call_count, 1)
                self.assertNotIn(URL, fetch.session)
                self.assertEqual(list(self.cache.iterdir()), [])

    def test_empty_response_never_cached(self):
        self.urlopen.return_value = response(" \n ")
        fetch = self.fetcher()
        with self.assertRaises(FetchError):
            fetch(URL)
        self.assertNotIn(URL, fetch.session)
        self.assertFalse(self.cache_path().exists())

    def test_429_retry_after_seconds_counts_attempts_then_succeeds(self):
        self.urlopen.side_effect = [http_error(429, "7.5"), response()]
        fetch = self.fetcher()
        with self.assertLogs("reference_finder.webfetch", level="WARNING"):
            self.assertEqual(fetch(URL), HTML)
        self.assertEqual(fetch.budget.used, 2)
        self.sleep.assert_any_call(7.5)

    def test_429_retry_after_http_date_is_respected(self):
        # Mocked current time is 1234; this HTTP date is epoch second 1246.
        self.urlopen.side_effect = [http_error(429, "Thu, 01 Jan 1970 00:20:46 GMT"), response()]
        with self.assertLogs("reference_finder.webfetch", level="WARNING"):
            self.assertEqual(self.fetcher()(URL), HTML)
        self.sleep.assert_any_call(12.0)

    def test_pacing_is_per_host_and_cache_hits_need_no_wait(self):
        self.urlopen.side_effect = [response(), response(), response()]
        fetch = WebFetcher(cache_dir=self.cache, interval=1.5)
        fetch(URL)
        fetch(URL + "?view=all")
        fetch("https://proceedings.mlr.press/")
        fetch(URL)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [0, 1.5, 0])
        self.assertEqual(self.urlopen.call_count, 3)

    def test_server_wait_over_one_minute_stops_without_early_retry(self):
        self.urlopen.side_effect = http_error(429, "120")
        with self.assertRaisesRegex(FetchError, "resume later"):
            self.fetcher()(URL)
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertFalse(any(call.args[0] > 0 for call in self.sleep.call_args_list))
        self.assertFalse(self.cache_path().exists())

    def test_request_limit_propagates_during_retry(self):
        self.urlopen.side_effect = http_error(429, "2")
        fetch = self.fetcher(budget=Budget(1))
        with self.assertLogs("reference_finder.webfetch", level="WARNING"):
            with self.assertRaises(RequestLimit):
                fetch(URL)
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertEqual(fetch.budget.used, 1)
        self.assertFalse(self.cache_path().exists())

    def test_repeated_transient_failure_stops_at_three_attempts(self):
        self.urlopen.side_effect = http_error(503)
        fetch = self.fetcher()
        with self.assertLogs("reference_finder.webfetch", level="WARNING"):
            with self.assertRaisesRegex(FetchError, "resume later"):
                fetch(URL)
        self.assertEqual(self.urlopen.call_count, 3)
        self.assertEqual(fetch.budget.used, 3)

    def test_403_is_terminal_without_browser_or_subprocess_fallback(self):
        self.urlopen.side_effect = http_error(403)
        with patch("subprocess.run", side_effect=AssertionError("No fallback allowed")) as run:
            with self.assertRaisesRegex(FetchError, "HTTP 403") as caught:
                self.fetcher()(URL)
        run.assert_not_called()
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertNotIn("private diagnostic", str(caught.exception))
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_truncated_cache_json_is_replaced_with_successful_response(self):
        self.cache_path().write_text('{"url":')
        self.urlopen.return_value = response()
        self.assertEqual(self.fetcher()(URL), HTML)
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertEqual(json.loads(self.cache_path().read_text())["body"], HTML)

    def test_leftover_atomic_write_temporary_is_not_treated_as_a_cache_hit(self):
        self.cache_path().with_suffix(".tmp").write_text('{"url":')
        self.urlopen.return_value = response()
        self.assertEqual(self.fetcher()(URL), HTML)
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertFalse(self.cache_path().with_suffix(".tmp").exists())

    def test_wrong_url_cache_entry_cannot_supply_another_sources_body(self):
        self.cache_path().write_text(json.dumps({"url": "https://proceedings.mlr.press/", "body": "Different source"}))
        self.urlopen.return_value = response()
        self.assertEqual(self.fetcher()(URL), HTML)
        self.assertEqual(self.urlopen.call_count, 1)

    def test_bad_cache_shapes_or_body_types_are_refetched(self):
        for entry in (None, [], {}, {"url": URL}, {"url": URL, "body": None},
                      {"url": URL, "body": {}}, {"url": URL, "body": ""}):
            with self.subTest(entry=entry):
                self.cache_path().write_text(json.dumps(entry))
                self.urlopen.reset_mock()
                self.urlopen.return_value = response()
                self.assertEqual(self.fetcher()(URL), HTML)
                self.assertEqual(self.urlopen.call_count, 1)

    def test_cached_challenge_is_not_replayed_as_a_successful_page(self):
        self.cache_path().write_text(json.dumps({"url": URL, "body": "<html>anubis_challenge</html>"}))
        self.urlopen.return_value = response()
        self.assertEqual(self.fetcher()(URL), HTML)
        self.assertEqual(self.urlopen.call_count, 1)

    def test_invalid_sparql_json_cannot_poison_resume_cache(self):
        self.urlopen.return_value = response('{"results":')
        fetch = self.fetcher()
        with self.assertRaises(FetchError):
            fetch(SPARQL_URL)
        self.assertFalse(self.cache_path(SPARQL_URL).exists())
        self.assertNotIn(SPARQL_URL, fetch.session)

    def test_invalid_sparql_json_in_existing_cache_is_refetched(self):
        self.cache_path(SPARQL_URL).write_text(json.dumps({"url": SPARQL_URL, "body": '{"results":'}))
        good = '{"results":{"bindings":[]}}'
        self.urlopen.return_value = response(good)
        self.assertEqual(self.fetcher()(SPARQL_URL), good)
        self.assertEqual(self.urlopen.call_count, 1)

    def test_new_html_hosts_and_gzip_responses(self):
        for url in ("https://proceedings.mlr.press/", "https://proceedings.neurips.cc/",
                    "https://papers.nips.cc/paper/2017", "https://ojs.aaai.org/index.php/AAAI/issue/archive"):
            with self.subTest(url=url):
                compressed = io.BytesIO(gzip.compress(HTML.encode()))
                compressed.headers = response().headers
                self.urlopen.return_value = compressed
                self.assertEqual(self.fetcher()(url), HTML)

    def test_invalid_new_html_never_poison_cache(self):
        url = "https://proceedings.mlr.press/"
        self.urlopen.return_value = response("<html><body>Truncated")
        with self.assertRaisesRegex(FetchError, "truncated"):
            self.fetcher()(url)
        self.assertFalse(self.cache_path(url).exists())

    def test_openreview_requires_valid_notes_and_valid_count_when_present(self):
        url = "https://api2.openreview.net/notes?limit=100"
        for raw in ("{}", '{"notes":[],"count":true}', '{"notes":[],"count":-1}', "<html>error</html>"):
            self.urlopen.return_value = response(raw)
            with self.subTest(raw=raw), self.assertRaises(FetchError):
                self.fetcher()(url)
            self.assertFalse(self.cache_path(url).exists())
        raw = '{"notes":[],"count":0}'
        self.urlopen.return_value = response(raw)
        self.assertEqual(self.fetcher()(url), raw)
        self.assertEqual(self.urlopen.call_args.args[0].get_header("Accept"), "application/json")
        self.urlopen.return_value = response('{"notes":[]}')
        self.assertEqual(self.fetcher()(url + "&offset=1"), '{"notes":[]}')


if __name__ == "__main__":
    unittest.main()
