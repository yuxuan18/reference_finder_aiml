import json
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from reference_finder.dblp import collect_dblp, parse_dblp_html, parse_dblp_index
from reference_finder.api import RequestLimit


def binding(**values):
    return {key: {"type": "literal", "value": str(value)} for key, value in values.items()}


def response(*rows):
    return json.dumps({"head": {"vars": []}, "results": {"bindings": list(rows)}})


def book(key, count):
    stream, name = key.rsplit("/", 1)
    return binding(book="https://dblp.org/rec/" + key, count=count,
                   toc="https://dblp.org/db/" + stream + "/" + stream.split("/")[-1] + name)


def paper(key, book_key, **kwargs):
    values = dict(paper="https://dblp.org/rec/" + key, title="A paper about machine learning.",
                  books="https://dblp.org/rec/" + book_key,
                  doi="https://doi.org/10.1007/TEST_1", authors="2:Bob\x1f1:Alice",
                  publicationYear="2025", url="https://doi.org/10.1007/TEST_1")
    values.update(kwargs)
    return binding(**values)


class Fetch:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


class DblpApiTests(unittest.TestCase):
    def test_ml_conference_streams(self):
        for venue, stream in (("iclr", "iclr"), ("icml", "icml"), ("neurips", "nips"), ("aaai", "aaai")):
            key = f"conf/{stream}/2025"
            with self.subTest(venue=venue):
                fetch = Fetch(response(book(key, 1)), response(paper(f"conf/{stream}/Example25", key)))
                output = collect_dblp(venue, 2025, fetch)
                self.assertEqual(output["status"], "complete")
                self.assertIn(f"streams/conf/{stream}", parse_qs(urlsplit(fetch.urls[0]).query)["query"][0])


    def test_all_parts_and_event_year(self):
        fetch = Fetch(response(book("conf/nips/2024-1", 1), book("conf/nips/2024-2", 1)),
                      response(paper("conf/nips/A24", "conf/nips/2024-1"),
                               paper("conf/nips/B24", "conf/nips/2024-2")))
        result = collect_dblp("neurips", 2024, fetch)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["records"]), 2)
        record = result["records"][0]
        self.assertEqual(record["year"], 2024)
        self.assertEqual(record["publication_year"], 2025)
        self.assertEqual(record["doi"], "10.1007/test_1")
        self.assertEqual(record["authors"], ["Alice", "Bob"])
        query = parse_qs(urlsplit(fetch.urls[0]).query)["query"][0]
        self.assertIn("publishedInStream <https://dblp.org/streams/conf/nips>", query)
        self.assertIn("2024(-[0-9]+)?$", query)
        self.assertNotIn("yearOfPublication", query)

    def test_pagination_and_count_validation(self):
        fetch = Fetch(response(book("conf/icml/2024-1", 3)),
                      response(paper("conf/icml/A24", "conf/icml/2024-1"),
                               paper("conf/icml/B24", "conf/icml/2024-1")),
                      response(paper("conf/icml/C24", "conf/icml/2024-1")))
        with patch("reference_finder.dblp.PAGE_SIZE", 2):
            result = collect_dblp("icml", 2024, fetch)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["records"]), 3)
        self.assertIn("OFFSET 2", parse_qs(urlsplit(fetch.urls[-1]).query)["query"][0])

    def test_absent_year_is_unavailable_not_complete_zero(self):
        result = collect_dblp("icml", 2026, Fetch(response()))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["records"], [])
        self.assertIn("unknown coverage", result["note"])

    def test_empty_volume_is_incomplete(self):
        result = collect_dblp("icml", 2024, Fetch(response(book("conf/icml/2024-1", 0))))
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("volume has no indexed papers", result["note"])

    def test_short_response_preserves_partial_records(self):
        result = collect_dblp("icml", 2024, Fetch(response(book("conf/icml/2024-1", 2)),
                              response(paper("conf/icml/A24", "conf/icml/2024-1"))))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["records"]), 1)
        self.assertIn("read 1 of 2", result["note"])

    def test_failed_page_preserves_partial_records(self):
        with patch("reference_finder.dblp.PAGE_SIZE", 1):
            result = collect_dblp("icml", 2024, Fetch(response(book("conf/icml/2024-1", 2)),
                                  response(paper("conf/icml/A24", "conf/icml/2024-1")),
                                  OSError("rate limited")))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["records"]), 1)
        self.assertIn("rate limited", result["note"])

    def test_repeated_page_is_incomplete(self):
        row = paper("conf/icml/A24", "conf/icml/2024-1")
        with patch("reference_finder.dblp.PAGE_SIZE", 1):
            result = collect_dblp("icml", 2024, Fetch(response(book("conf/icml/2024-1", 2)),
                                  response(row), response(row)))
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("repeated", result["note"])

    def test_workshop_companion_and_other_stream_are_rejected(self):
        for key in ("conf/aaai/2024w", "conf/aaai/2024-companion", "conf/iclr/2024"):
            with self.subTest(key=key):
                result = collect_dblp("aaai", 2024, Fetch(response(book(key, 5))))
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["records"], [])

    def test_error_html_and_bad_api_json_are_incomplete(self):
        for raw in ("<html>challenge</html>", "{}", '{"results":{"bindings":{}}}',
                    '{"exception":"rate limited"}'):
            with self.subTest(raw=raw):
                result = collect_dblp("iclr", 2024, Fetch(raw))
                self.assertEqual(result["status"], "incomplete")

    def test_missing_metadata_and_wrong_parent_are_incomplete(self):
        for kwargs in (dict(title=""), dict(books="https://dblp.org/rec/conf/icml/2024w")):
            with self.subTest(kwargs=kwargs):
                result = collect_dblp("icml", 2024, Fetch(response(book("conf/icml/2024-1", 1)),
                                      response(paper("conf/icml/A24", "conf/icml/2024-1", **kwargs))))
                self.assertEqual(result["status"], "incomplete")

    def test_invalid_scope_fails_before_network(self):
        for venue, year in (("unknown", 2024), ("icml", "2024"), ("icml", True)):
            with self.subTest(venue=venue, year=year), self.assertRaises(ValueError):
                collect_dblp(venue, year, Fetch())

    def test_request_budget_stops_entire_run(self):
        with self.assertRaises(RequestLimit):
            collect_dblp("icml", 2024, Fetch(RequestLimit("stop")))
        with self.assertRaises(RequestLimit):
            collect_dblp("icml", 2024, Fetch(response(book("conf/icml/2024-1", 2)),
                                              RequestLimit("stop")))


def html(body):
    return '<html><head><title>dblp: Test conference</title></head><body>' + body + '</body></html>'


def html_paper(key="conf/icml/Test24", title="A <i>Learning</i> Method.", doi=True):
    link = ('https://doi.org/10.1007/Test_1' if doi else
            'https://proceedings.iclr.cc/paper_files/paper/2024/hash/test-Abstract-Conference.html')
    return f'''<li class="entry inproceedings" id="{key}">
      <nav class="publ"><ul><li class="drop-down"><div class="head"><a href="{link}">view</a></div>
      <div class="body"><ul><li class="ee"><a href="{link}">electronic edition</a></li></ul></div></li></ul></nav>
      <cite><span itemprop="author"><a><span itemprop="name">Alice A.</span></a></span>
      <span itemprop="author"><a><span itemprop="name">Bob B.</span></a></span>
      <span class="title" itemprop="name">{title}</span></cite></li>'''


class DblpHtmlTests(unittest.TestCase):
    def test_papers_authors_doi_markup_and_section(self):
        source = html('<h2>Main Proceedings</h2><ul>' + html_paper() + '</ul>'
                      '<h3>Poster Session</h3><ul>' + html_paper("conf/icml/Poster24") + '</ul>')
        rows = parse_dblp_html(source, 2024, "https://dblp.org/db/conf/icml/icml2024-1.html")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["title"], "A Learning Method.")
        self.assertEqual(rows[0]["authors"], ["Alice A.", "Bob B."])
        self.assertEqual(rows[0]["doi"], "10.1007/test_1")
        self.assertEqual(rows[1]["section"], "Poster Session")

    def test_iclr_link_without_doi(self):
        rows = parse_dblp_html(html('<ul>' + html_paper("conf/iclr/Test24", doi=False) + '</ul>'),
                               2024, "https://dblp.org/db/conf/iclr/uss2024.html")
        self.assertIsNone(rows[0]["doi"])
        self.assertTrue(rows[0]["url"].startswith("https://proceedings.iclr.cc/"))

    def test_reject_truncated_empty_challenge_and_malformed_records(self):
        sources = [html(html_paper())[:-7], html("No records"),
                   '<html><head><title>Making sure you are not a bot!</title></head></html>',
                   html('<script id="anubis_challenge">{}</script>'),
                   html('<li class="entry inproceedings" id="conf/icml/Broken24"></li>'),
                   html(html_paper() + html_paper())]
        for source in sources:
            with self.subTest(source=source[:60]), self.assertRaises(ValueError):
                parse_dblp_html(source, 2024, "https://dblp.org/db/conf/icml/icml2024.html")

    def test_index_all_main_parts_and_exclude_companion(self):
        entries = ''.join(f'<li class="entry editor toc" id="conf/icml/{part}">'
                          f'<a href="icml{part}.html">[contents]</a></li>'
                          for part in ("2024-1", "2024-2", "2024-10", "2024w", "2025-1"))
        urls = parse_dblp_index("icml", 2024, html('<ul>' + entries + '</ul>'))
        self.assertEqual(len(urls), 3)
        self.assertIn("https://dblp.org/db/conf/icml/icml2024-10.html", urls)

    def test_absent_year_on_valid_index_has_no_urls(self):
        source = html('<li class="entry editor" id="conf/icml/2025"><a href="icml2025.html">contents</a></li>')
        self.assertEqual(parse_dblp_index("icml", 2026, source), [])


if __name__ == "__main__":
    unittest.main()
