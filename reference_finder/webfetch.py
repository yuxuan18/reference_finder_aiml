"""Polite, cached retrieval of public bibliography pages and the DBLP API."""
from __future__ import annotations

import hashlib
import gzip
import email.utils
import json
import logging
import os
import re
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from .api import Budget, APIError

LOG = logging.getLogger(__name__)
USER_AGENT = "reference-finder/2.0 (academic bibliography; Python)"


class FetchError(APIError):
    pass


class AccessChallenge(FetchError):
    """The service requires user verification; do not repeat anonymous requests."""


class WebFetcher:
    def __init__(self, cache_dir="data/page_cache", refresh=False, interval=1.0, budget=None, timeout=45):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh, self.interval, self.timeout = refresh, interval, timeout
        self.budget = budget or Budget()
        self.last = {}
        self.session = {}
        self.blocked_hosts = {}
        self.openreview_token = os.environ.get("OPENREVIEW_TOKEN", "").strip()
        self.openreview_username = os.environ.get("OPENREVIEW_USERNAME", "").strip()
        self.openreview_password = os.environ.get("OPENREVIEW_PASSWORD", "")
        self.openreview_sessions = {}

    def _challenge_error(self, host):
        error = AccessChallenge(
            f"{host}: HTTP 403 (ChallengeRequiredError). OpenReview requires verification for this request. "
            "Configure authorized API access with OPENREVIEW_USERNAME/OPENREVIEW_PASSWORD or OPENREVIEW_TOKEN, "
            "or use independent official proceedings sources where available. Browser verification alone may "
            "not authorize this Python process. Further requests to this host are skipped for this run; "
            "retry after access is resolved.")
        self.blocked_hosts[host] = error
        return error

    def _pace(self, host):
        self.budget.take()
        time.sleep(max(0, self.interval - (time.monotonic() - self.last.get(host, 0))))
        self.last[host] = time.monotonic()

    def _openreview_authorization(self, host):
        if self.openreview_token:
            return self.openreview_token if self.openreview_token.startswith("Bearer ") else "Bearer " + self.openreview_token
        if not self.openreview_username and not self.openreview_password:
            return None
        if not self.openreview_username or not self.openreview_password:
            raise FetchError("Set both OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD in .env, or provide OPENREVIEW_TOKEN.")
        saved = self.openreview_sessions.get(host)
        if saved and time.monotonic() - saved[1] < 3300:
            return "Bearer " + saved[0]
        self._pace(host)
        request = urllib.request.Request(
            "https://" + host + "/login",
            data=json.dumps({"id": self.openreview_username, "password": self.openreview_password,
                             "expiresIn": 3600}).encode("utf-8"),
            headers={"User-Agent": USER_AGENT, "Accept": "application/json", "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                details = json.loads(exc.read(65536))
            except (ValueError, OSError):
                details = {}
            if exc.code == 403 and isinstance(details, dict) and details.get("name") == "ChallengeRequiredError":
                raise self._challenge_error(host) from None
            if exc.code == 429:
                reason = "Login is rate limited; wait before retrying."
            elif exc.code in {408, 500, 502, 503, 504}:
                reason = "Login service is temporarily unavailable; retry later."
            else:
                reason = "Check account credentials/access."
            error = FetchError(f"{host}: OpenReview login failed (HTTP {exc.code}). {reason} Login will not be repeated this run.")
            self.blocked_hosts[host] = error
            raise error from None
        except (urllib.error.URLError, OSError, ValueError):
            raise FetchError(f"{host}: OpenReview login failed: network error or invalid response; retry later.") from None
        if isinstance(payload, dict) and payload.get("mfaPending"):
            error = FetchError("OpenReview requires multi-factor authentication. Use OPENREVIEW_TOKEN from an authorized session.")
            self.blocked_hosts[host] = error
            raise error
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            error = FetchError("OpenReview login returned no access token; authentication is incomplete.")
            self.blocked_hosts[host] = error
            raise error
        self.openreview_sessions[host] = (token, time.monotonic())
        LOG.info("Authenticated with %s; access token kept in memory", host)
        return "Bearer " + token

    def _validate(self, body, host):
        if not isinstance(body, str) or not body.strip():
            raise FetchError(f"{host}: empty or invalid response")
        if any(token in body.lower() for token in ("anubis_version", "anubis_challenge", "making sure you're not a bot", "just a moment...", "cf-chl-", "verifying your browser")):
            raise FetchError(f"{host}: bot challenge; page was not parsed or cached.")
        if host in {"proceedings.mlr.press", "proceedings.neurips.cc", "papers.nips.cc", "ojs.aaai.org", "proceedings.iclr.cc", "colmweb.org"}:
            if not re.search(r"</html\s*>\s*$", body, re.I):
                raise FetchError(f"{host}: truncated or invalid HTML; not cached")
        if host == "sparql.dblp.org":
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict) or not isinstance(payload.get("results", {}).get("bindings"), list):
                    raise ValueError("Missing bindings")
            except (ValueError, AttributeError, TypeError):
                raise FetchError(f"{host}: invalid SPARQL response; not cached") from None

        if host in {"api.openreview.net", "api2.openreview.net"}:
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict) or not isinstance(payload.get("notes"), list):
                    raise ValueError("Missing notes")
                if "count" in payload and (not isinstance(payload["count"], int)
                                           or isinstance(payload["count"], bool) or payload["count"] < 0):
                    raise ValueError("Invalid count")
            except (ValueError, TypeError):
                raise FetchError(f"{host}: invalid OpenReview response; not cached") from None

    def __call__(self, url):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in {
            "sparql.dblp.org",
            "proceedings.mlr.press", "proceedings.neurips.cc", "papers.nips.cc",
            "api.openreview.net", "api2.openreview.net", "ojs.aaai.org", "proceedings.iclr.cc", "colmweb.org"
        }:
            raise FetchError("Unsupported source URL; use a documented public source or saved HTML.")
        if url in self.session:
            return self.session[url]
        path = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if path.exists() and not self.refresh:
            try:
                entry = json.loads(path.read_text())
                if isinstance(entry, dict) and entry.get("url") == url:
                    self._validate(entry.get("body"), parts.hostname)
                    self.session[url] = entry["body"]
                    return entry["body"]
            except (ValueError, KeyError, FetchError):
                pass
        if parts.hostname in self.blocked_hosts:
            raise self.blocked_hosts[parts.hostname]
        authorization = None
        if parts.hostname in {"api.openreview.net", "api2.openreview.net"}:
            authorization = self._openreview_authorization(parts.hostname)
        renewed = False
        for attempt in range(3):
            self._pace(parts.hostname)
            headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json" if parts.hostname == "sparql.dblp.org" else "text/html"}
            if parts.hostname in {"api.openreview.net", "api2.openreview.net"}:
                headers["Accept"] = "application/json"
            try:
                request = urllib.request.Request(url, headers=headers)
                if authorization:
                    # urllib does not forward unredirected headers to a redirect
                    # target (for example, a browser-verification web page).
                    request.add_unredirected_header("Authorization", authorization)
                with urllib.request.urlopen(request, timeout=self.timeout) as r:
                    data = r.read()
                    if data.startswith(b"\x1f\x8b"):
                        data = gzip.decompress(data)
                    body = data.decode(r.headers.get_content_charset() or "utf-8", errors="replace")
                self._validate(body, parts.hostname)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"url": url, "fetched_at": time.time(), "body": body}, ensure_ascii=False))
                temporary.replace(path)
                self.session[url] = body
                return body
            except urllib.error.HTTPError as exc:
                if (exc.code == 401 and authorization and not self.openreview_token and not renewed
                        and attempt < 2):
                    self.openreview_sessions.pop(parts.hostname, None)
                    authorization = self._openreview_authorization(parts.hostname)
                    renewed = True
                    continue
                if exc.code == 403 and parts.hostname in {"api.openreview.net", "api2.openreview.net"}:
                    # Read only the structured error name, never echo response
                    # bodies, authorization headers, or arbitrary error text.
                    try:
                        payload = json.loads(exc.read(65536))
                    except (ValueError, OSError):
                        payload = {}
                    if isinstance(payload, dict) and payload.get("name") == "ChallengeRequiredError":
                        raise self._challenge_error(parts.hostname) from None
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise FetchError(f"{parts.hostname}: HTTP {exc.code} ({parts.path})") from None
                try:
                    delay = max(2 ** (attempt + 1), float(exc.headers.get("Retry-After", 0)))
                except ValueError:
                    delay = 2 ** (attempt + 1)
                    try:
                        delay = max(delay, email.utils.parsedate_to_datetime(exc.headers["Retry-After"]).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
                if delay > 60 or attempt == 2:
                    raise FetchError(f"{parts.hostname}: HTTP {exc.code}; resume later") from None
                LOG.warning("%s: HTTP %s, waiting %.0fs", parts.hostname, exc.code, delay)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError, EOFError) as exc:
                if attempt == 2:
                    raise FetchError(f"{parts.hostname}: network request failed; resume later") from None
                time.sleep(2 ** (attempt + 1))
