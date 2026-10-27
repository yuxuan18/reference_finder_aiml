"""Offline contract tests for API requests, retry behavior, and safe errors."""
import io
import json
import unittest
import urllib.error
import urllib.parse
from unittest.mock import patch

from reference_finder.api import APIError, Budget, HTTP, OpenAlex, RequestLimit, SemanticScholar


def response(payload):
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


def http_error(code, retry_after=None, secret="secret-key"):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError(
        "https://example.test/?api_key=" + secret,
        code,
        "server echoed " + secret,
        headers,
        io.BytesIO(secret.encode("utf-8")),
    )


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.urlopen = self.enter_patch("reference_finder.api.urllib.request.urlopen")
        self.sleep = self.enter_patch("reference_finder.api.time.sleep")
        self.enter_patch("reference_finder.api.time.monotonic", return_value=1000.0)
        self.enter_patch("reference_finder.api.random.random", return_value=0.0)
        self.http = HTTP("test service", "https://example.test", Budget(), interval=0)

    def enter_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def test_retry_after_seconds_then_exponential_backoff(self):
        self.urlopen.side_effect = [http_error(429, "7.5"), http_error(503), response({"ok": True})]
        with self.assertLogs("reference_finder.api", level="WARNING"):
            result = self.http.request("/works")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(self.http.budget.used, 3)
        delays = [call.args[0] for call in self.sleep.call_args_list if call.args[0] > 0]
        self.assertEqual(delays, [7.5, 2.0])

    def test_retry_after_http_date(self):
        self.enter_patch("reference_finder.api.time.time", return_value=0.0)
        self.urlopen.side_effect = [
            http_error(429, "Thu, 01 Jan 1970 00:00:12 GMT"), response({"ok": True})
        ]
        with self.assertLogs("reference_finder.api", level="WARNING"):
            self.http.request("/works")
        self.sleep.assert_any_call(12.0)

    def test_invalid_retry_after_uses_backoff(self):
        self.urlopen.side_effect = [http_error(503, "not a date"), response({})]
        with self.assertLogs("reference_finder.api", level="WARNING"):
            self.http.request("/works")
        self.sleep.assert_any_call(1.0)

    def test_long_retry_after_stops_without_hammering_server(self):
        self.urlopen.side_effect = http_error(429, "120")
        with self.assertRaisesRegex(APIError, "resume later"):
            self.http.request("/works")
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertFalse(any(call.args[0] > 0 for call in self.sleep.call_args_list))

    def test_request_budget_counts_retries_and_stops_before_next_request(self):
        self.http.budget = Budget(1)
        self.urlopen.side_effect = http_error(503)
        with self.assertLogs("reference_finder.api", level="WARNING"):
            with self.assertRaises(RequestLimit):
                self.http.request("/works")
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertEqual(self.http.budget.used, 1)

    def test_credentials_are_absent_from_errors_and_retry_logs(self):
        secret = "super-secret-credential"
        self.http.headers["Authorization"] = "Bearer " + secret
        self.http.retries = 1
        for error in (
            http_error(401, secret=secret),
            http_error(429, secret=secret),
            urllib.error.URLError("failed URL ?api_key=" + secret),
        ):
            with self.subTest(error=type(error).__name__, code=getattr(error, "code", None)):
                self.urlopen.side_effect = error
                with patch("reference_finder.api.LOG.warning") as warning:
                    with self.assertRaises(APIError) as caught:
                        self.http.request("/works", {"api_key": secret})
                self.assertNotIn(secret, str(caught.exception))
                self.assertNotIn(secret, str(warning.call_args_list))
                self.assertTrue(caught.exception.__suppress_context__)

    def test_auth_failure_does_not_retry(self):
        self.urlopen.side_effect = http_error(403)
        with self.assertRaisesRegex(APIError, "API key"):
            self.http.request("/works")
        self.assertEqual(self.urlopen.call_count, 1)

    def test_invalid_json_retries_and_does_not_expose_response(self):
        self.http.retries = 1
        self.urlopen.side_effect = [io.BytesIO(b"secret response"), response({"ok": True})]
        with self.assertLogs("reference_finder.api", level="WARNING") as logs:
            self.assertEqual(self.http.request("/works"), {"ok": True})
        self.assertNotIn("secret response", " ".join(logs.output))
        self.assertEqual(self.urlopen.call_count, 2)

    def test_optional_not_found_returns_none_without_retry(self):
        self.urlopen.side_effect = http_error(404)
        self.assertIsNone(self.http.request("/works/missing", allow_not_found=True))
        self.assertEqual(self.urlopen.call_count, 1)

    def test_credentials_are_sent_as_headers(self):
        for header, value in (("Authorization", "Bearer oa-key"), ("x-api-key", "s2-key")):
            with self.subTest(header=header):
                client = HTTP("test", "https://example.test", Budget(), headers={header: value}, interval=0)
                self.urlopen.return_value = response({})
                client.request("/works")
                request = self.urlopen.call_args.args[0]
                headers = {name.lower(): item for name, item in request.header_items()}
                self.assertEqual(headers[header.lower()], value)
                self.assertNotIn(value, request.full_url)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.urlopen_patcher = patch("reference_finder.api.urllib.request.urlopen")
        self.urlopen = self.urlopen_patcher.start()
        self.addCleanup(self.urlopen_patcher.stop)
        self.sleep_patcher = patch("reference_finder.api.time.sleep")
        self.sleep_patcher.start()
        self.addCleanup(self.sleep_patcher.stop)
        self.http = HTTP("test", "https://example.test", Budget(), interval=0)

    def query(self):
        return urllib.parse.parse_qs(urllib.parse.urlsplit(self.urlopen.call_args.args[0].full_url).query)

    def test_openalex_cursor_requests_have_100_results_and_abstracts(self):
        client = OpenAlex(self.http)
        self.urlopen.return_value = response({"meta": {"next_cursor": "next+token/="}, "results": []})
        page = client.works("S123", 2026)
        first_query = self.query()
        self.assertEqual(first_query["cursor"], ["*"])
        self.assertEqual(first_query["per_page"], ["100"])
        self.assertEqual(first_query["filter"], ["locations.source.id:S123,publication_year:2026"])
        self.assertIn("abstract_inverted_index", first_query["select"][0].split(","))
        self.urlopen.return_value = response({"meta": {"next_cursor": None}, "results": []})
        client.works("S123", 2026, cursor=page["meta"]["next_cursor"])
        self.assertEqual(self.query()["cursor"], ["next+token/="])
        self.assertEqual(self.query()["per_page"], ["100"])

    def test_source_search_accepts_cursor_and_encodes_search(self):
        self.urlopen.return_value = response({"results": []})
        OpenAlex(self.http).sources("Learning & Representations", cursor="next cursor+")
        self.assertEqual(self.query(), {"search": ["Learning & Representations"], "cursor": ["next cursor+"], "per_page": ["100"]})

    def test_doi_is_one_escaped_path_component(self):
        self.urlopen.return_value = response({"abstract_inverted_index": {"Found": [0]}})
        doi = "10.1234/paper?part=1#appendix"
        result = OpenAlex(self.http).doi(doi)
        url = urllib.parse.urlsplit(self.urlopen.call_args.args[0].full_url)
        self.assertEqual(url.path.count("/"), 2)
        self.assertEqual(urllib.parse.unquote(url.path), "/works/https://doi.org/" + doi)
        self.assertEqual(url.fragment, "")
        self.assertEqual(result["abstract_inverted_index"], {"Found": [0]})
        self.assertIn("abstract_inverted_index", self.query()["select"][0].split(","))

    def test_semantic_scholar_bulk_can_filter_venues_without_a_query(self):
        self.urlopen.return_value = response({"data": [], "token": "next/page+="})
        client = SemanticScholar(self.http)
        first_page = client.works(["ICML", "ACM ICML"], 2026)
        query = self.query()
        self.assertEqual(query["venue"], ["ICML,ACM ICML"])
        self.assertEqual(query["year"], ["2026"])
        self.assertNotIn("query", query)
        self.assertNotIn("token", query)
        self.assertIn("abstract", query["fields"][0].split(","))
        self.urlopen.return_value = response({"data": []})
        client.works(["ICML"], 2026, token=first_page["token"])
        self.assertEqual(self.query()["token"], ["next/page+="])

    def test_semantic_scholar_batch_keeps_nulls_aligned_with_dois(self):
        papers = [{"paperId": "first", "abstract": "First abstract"}, None, {"paperId": "third"}]
        self.urlopen.return_value = response(papers)
        result = SemanticScholar(self.http).dois(["10.1/a", "10.1/missing", "10.1/c"])
        self.assertEqual(result, papers)
        request = self.urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(urllib.parse.urlsplit(request.full_url).path, "/paper/batch")
        self.assertEqual(json.loads(request.data), {"ids": ["DOI:10.1/a", "DOI:10.1/missing", "DOI:10.1/c"]})
        self.assertIn("abstract", self.query()["fields"][0].split(","))


if __name__ == "__main__":
    unittest.main()
