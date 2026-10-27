"""Use dummy secrets only; never load the developer's .env in tests."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from reference_finder.api import Budget, RequestLimit
from reference_finder.config import load_environment
from reference_finder.webfetch import AccessChallenge, FetchError, WebFetcher
from test_webfetch import response


class EnvironmentTests(unittest.TestCase):
    def test_dotenv_quoting_literal_password_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / ".env"
            path.write_text('OPENREVIEW_USERNAME=example\nOPENREVIEW_PASSWORD="space $literal #test"\nOPENALEX_API_KEY=file-key\n')
            os.environ["OPENALEX_API_KEY"] = "shell-key"
            load_environment(path)
            self.assertEqual(os.environ["OPENREVIEW_USERNAME"], "example")
            self.assertEqual(os.environ["OPENREVIEW_PASSWORD"], "space $literal #test")
            self.assertEqual(os.environ["OPENALEX_API_KEY"], "shell-key")

    def test_missing_dotenv_is_optional(self):
        load_environment("/tmp/nonexistent-reference-finder-env")


class LoginTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"OPENREVIEW_USERNAME": "dummy-user", "OPENREVIEW_PASSWORD": "dummy-password",
                                     "OPENREVIEW_TOKEN": ""})
        env.start()
        self.addCleanup(env.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        opener = patch("reference_finder.webfetch.urllib.request.urlopen")
        self.open = opener.start()
        self.addCleanup(opener.stop)

    def fetcher(self, budget=None):
        return WebFetcher(self.tmp.name, interval=0, budget=budget)

    def test_lazy_login_once_per_host_and_no_secrets_in_cache(self):
        self.open.side_effect = [response('{"token":"dummy-token"}'), response('{"notes":[],"count":0}'),
                                 response('{"notes":[],"count":0}'), response('{"token":"legacy-token"}'),
                                 response('{"notes":[],"count":0}')]
        fetch = self.fetcher()
        fetch("https://api2.openreview.net/notes?offset=0")
        fetch("https://api2.openreview.net/notes?offset=100")
        fetch("https://api.openreview.net/notes?offset=0")
        calls = [c.args[0] for c in self.open.call_args_list]
        self.assertEqual([r.full_url for r in calls if r.get_method() == "POST"],
                         ["https://api2.openreview.net/login", "https://api.openreview.net/login"])
        self.assertEqual(json.loads(calls[0].data)["password"], "dummy-password")
        self.assertEqual(calls[1].get_header("Authorization"), "Bearer dummy-token")
        for p in Path(self.tmp.name).iterdir():
            cached = p.read_text()
            self.assertNotIn("dummy-password", cached)
            self.assertNotIn("dummy-token", cached)
            self.assertNotIn("/login", cached)
        self.assertEqual(fetch.budget.used, 5)

    def test_explicit_token_takes_precedence_over_login(self):
        self.open.return_value = response('{"notes":[],"count":0}')
        with patch.dict(os.environ, {"OPENREVIEW_TOKEN": "existing-token"}):
            self.fetcher()("https://api2.openreview.net/notes")
        self.assertEqual(self.open.call_count, 1)
        self.assertIsNone(self.open.call_args.args[0].data)

    def test_login_failure_is_sanitized_and_not_retried(self):
        error = urllib.error.HTTPError("https://api2.openreview.net/login", 401, "Unauthorized", {},
                                       io.BytesIO(b'{"message":"dummy-password dummy-user"}'))
        self.open.side_effect = error
        fetch = self.fetcher()
        for offset in (0, 100):
            with self.assertRaises(FetchError) as caught:
                fetch(f"https://api2.openreview.net/notes?offset={offset}")
            self.assertNotIn("dummy-password", str(caught.exception))
            self.assertNotIn("dummy-user", str(caught.exception))
        self.assertEqual(self.open.call_count, 1)

    def test_login_challenge_and_mfa_are_explicit(self):
        self.open.side_effect = urllib.error.HTTPError("https://api2.openreview.net/login", 403, "Forbidden", {},
                                       io.BytesIO(b'{"name":"ChallengeRequiredError"}'))
        with self.assertRaises(AccessChallenge):
            self.fetcher()("https://api2.openreview.net/notes")
        self.open.side_effect = None
        self.open.return_value = response('{"mfaPending":true,"mfaPendingToken":"private-pending-token"}')
        with self.assertRaisesRegex(FetchError, "multi-factor") as caught:
            self.fetcher()("https://api2.openreview.net/notes")
        self.assertNotIn("private-pending-token", str(caught.exception))

    def test_login_respects_request_budget(self):
        self.open.return_value = response('{"token":"dummy-token"}')
        fetch = self.fetcher(Budget(1))
        with self.assertRaises(RequestLimit):
            fetch("https://api2.openreview.net/notes")
        self.assertEqual(self.open.call_count, 1)
        self.assertFalse(list(Path(self.tmp.name).iterdir()))

    def test_login_rate_limit_is_not_reported_as_bad_credentials(self):
        self.open.side_effect = urllib.error.HTTPError("https://api2.openreview.net/login", 429, "Too many", {}, io.BytesIO())
        with self.assertRaisesRegex(FetchError, "rate limited") as caught:
            self.fetcher()("https://api2.openreview.net/notes")
        self.assertNotIn("Check account credentials", str(caught.exception))

    def test_incomplete_credentials_only_affect_openreview(self):
        with patch.dict(os.environ, {"OPENREVIEW_PASSWORD": ""}):
            fetch = self.fetcher()
        with self.assertRaisesRegex(FetchError, "Set both"):
            fetch("https://api2.openreview.net/notes")
        self.open.return_value = response()
        fetch("https://proceedings.iclr.cc/")
        self.assertIsNone(self.open.call_args.args[0].data)
        self.assertIsNone(self.open.call_args.args[0].get_header("Authorization"))

    def test_expired_login_session_can_be_renewed_once(self):
        unauthorized = urllib.error.HTTPError("https://api2.openreview.net/notes", 401, "Unauthorized", {}, io.BytesIO())
        self.open.side_effect = [response('{"token":"old-token"}'), unauthorized,
                                 response('{"token":"new-token"}'), response('{"notes":[],"count":0}')]
        fetch = self.fetcher()
        self.assertEqual(json.loads(fetch("https://api2.openreview.net/notes"))["count"], 0)
        self.assertEqual(self.open.call_count, 4)
        self.assertEqual(self.open.call_args.args[0].get_header("Authorization"), "Bearer new-token")


if __name__ == "__main__":
    unittest.main()
