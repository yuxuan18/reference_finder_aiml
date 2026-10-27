"""Offline imports exercise real SQLite transactions, checkpoints, and identity."""
import unittest
from unittest.mock import Mock, call

from reference_finder.api import APIError, RequestLimit
from reference_finder.importer import Importer, job_key
from reference_finder.storage import Database
from reference_finder.venues import VENUES


SOURCE_ID = "https://openalex.org/S4306460212"
YEAR = 2025


def s2_paper(number=1, **overrides):
    record = {
        "paperId": "%040x" % number,
        "title": "A learning result %s" % number,
        "year": YEAR,
        "venue": "ICML '25",
        "publicationVenue": {
            "id": "9ca72042-7524-46ee-b8e8-df443d922a5d",
            "name": "International Conference on Machine Learning",
        },
        "externalIds": {"DOI": "10.1145/1000000.%s" % number},
        "abstract": None,
        "publicationTypes": ["Conference"],
        "authors": [{"name": "A Researcher"}],
    }
    record.update(overrides)
    return record


def oa_paper(number=1, **overrides):
    record = {
        "id": "https://openalex.org/W%s" % (1000000000 + number),
        "display_name": "A learning result %s" % number,
        "publication_year": YEAR,
        "doi": "https://doi.org/10.1145/1000000.%s" % number,
        "abstract_inverted_index": None,
        "type": "article",
        "authorships": [{"author": {"display_name": "A Researcher"}}],
        "primary_location": {"source": {"id": SOURCE_ID}},
    }
    record.update(overrides)
    return record


def oa_page(records, cursor=None):
    return {"results": records, "meta": {"next_cursor": cursor}}


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.conn.close)
        self.venue = VENUES["icml"]
        self.s2 = Mock()
        self.oa = Mock()
        self.importer = Importer(self.db, [self.venue], YEAR, YEAR, self.oa, self.s2)

    def papers(self):
        return self.db.conn.execute("SELECT * FROM papers ORDER BY id").fetchall()

    def seed_source(self):
        with self.db.conn:
            self.db.conn.execute("INSERT INTO openalex_sources VALUES (?,?,?)", (
                self.venue.key, SOURCE_ID, "International Conference on Machine Learning"
            ))
            for query in self.venue.openalex_queries:
                self.db.save_checkpoint(job_key("oa-sources", self.venue.key, query=query), None, True, 1, 1)

    def test_colm_skips_api_discovery_before_2024(self):
        venue = VENUES["colm"]
        self.s2.works.return_value = {"data": []}
        importer = Importer(self.db, [venue], 2017, 2024, None, self.s2)
        importer.discover_s2()
        self.s2.works.assert_called_once_with(venue.semantic_scholar_names, 2024, None)

    def test_ml_venue_discovery_accepts_conference_labels(self):
        for number, key in enumerate(("iclr", "icml", "neurips", "aaai", "colm"), 1):
            venue = VENUES[key]
            self.s2.works.return_value = {"data": [s2_paper(number, venue=venue.semantic_scholar_names[0], publicationVenue={})]}
            Importer(self.db, [venue], YEAR, YEAR, None, self.s2).discover_s2()
            self.assertEqual(self.db.coverage(key, YEAR)["papers"], 1)


    def test_s2_multiple_pages_deduplicate_and_keep_missing_abstract(self):
        self.s2.works.side_effect = [
            {"data": [s2_paper(1)], "token": "page-two"},
            {"data": [s2_paper(1), s2_paper(2, abstract="A complete abstract.")]},
        ]
        self.importer.discover_s2()
        self.assertEqual(self.s2.works.call_args_list, [
            call(self.venue.semantic_scholar_names, YEAR, None),
            call(self.venue.semantic_scholar_names, YEAR, "page-two"),
        ])
        rows = self.papers()
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["abstract"])
        self.assertEqual(rows[1]["abstract"], "A complete abstract.")
        state = self.db.checkpoint(job_key("s2-discover", "icml", year=YEAR))
        self.assertTrue(state["done"])
        self.assertEqual(state["fetched"], 3)
        self.assertEqual(self.db.coverage("icml", YEAR)["papers"], 2)
        self.assertEqual(self.db.coverage("icml", YEAR)["abstracts"], 1)
        self.importer.discover_s2()
        self.assertEqual(self.s2.works.call_count, 2)

    def test_s2_stopped_run_resumes_committed_cursor(self):
        self.s2.works.side_effect = [
            {"data": [s2_paper(1)], "token": "resume-here"}, RequestLimit("test budget")
        ]
        with self.assertRaises(RequestLimit):
            self.importer.discover_s2()
        job = job_key("s2-discover", "icml", year=YEAR)
        self.assertEqual(self.db.checkpoint(job)["cursor"], "resume-here")
        self.assertFalse(self.db.checkpoint(job)["done"])
        self.s2.reset_mock()
        self.s2.works.side_effect = [{"data": [s2_paper(1), s2_paper(2)]}]
        restarted = Importer(self.db, [self.venue], YEAR, YEAR, semantic_scholar=self.s2)
        restarted.discover_s2()
        self.s2.works.assert_called_once_with(self.venue.semantic_scholar_names, YEAR, "resume-here")
        self.assertEqual(len(self.papers()), 2)
        self.assertTrue(self.db.checkpoint(job)["done"])

    def test_s2_failed_page_retains_last_checkpoint(self):
        self.s2.works.side_effect = [
            {"data": [s2_paper()], "token": "resume-here"}, APIError("temporary failure")
        ]
        with self.assertRaises(APIError):
            self.importer.discover_s2()
        state = self.db.checkpoint(job_key("s2-discover", "icml", year=YEAR))
        self.assertEqual(state["cursor"], "resume-here")
        self.assertEqual(state["fetched"], 1)
        self.assertFalse(state["done"])
        self.assertEqual(len(self.papers()), 1)

    def test_s2_missing_data_is_an_error_and_does_not_finish_job(self):
        self.s2.works.side_effect = [
            {"data": [s2_paper()], "token": "resume-here"}, {"message": "rate limit"}
        ]
        with self.assertRaisesRegex(APIError, "no data array"):
            self.importer.discover_s2()
        state = self.db.checkpoint(job_key("s2-discover", "icml", year=YEAR))
        self.assertEqual(state["cursor"], "resume-here")
        self.assertFalse(state["done"])
        self.assertEqual(state["fetched"], 1)

    def test_s2_repeated_pagination_token_does_not_commit_page(self):
        self.s2.works.side_effect = [
            {"data": [s2_paper(1)], "token": "same-token"},
            {"data": [s2_paper(2)], "token": "same-token"},
        ]
        with self.assertRaisesRegex(APIError, "repeated"):
            self.importer.discover_s2()
        self.assertEqual(len(self.papers()), 1)

    def test_page_records_and_checkpoint_are_one_transaction(self):
        broken = s2_paper(2)
        del broken["paperId"]
        self.s2.works.return_value = {"data": [s2_paper(1), broken]}
        with self.assertRaises(KeyError):
            self.importer.discover_s2()
        self.assertEqual(len(self.papers()), 0)
        self.assertIsNone(self.db.checkpoint(job_key("s2-discover", "icml", year=YEAR)))

    def test_s2_filters_workshops_wrong_years_and_volumes(self):
        self.s2.works.return_value = {"data": [
            s2_paper(1),
            s2_paper(2, venue="ICML Workshop on Privacy"),
            s2_paper(3, title="Proceedings of the 2025 International Conference on Machine Learning"),
            s2_paper(4, publicationTypes=["Book"]),
            s2_paper(5, title="Preface"),
            s2_paper(6, year=YEAR - 1),
            s2_paper(7, venue="Control and Communication Systems", publicationVenue={
                "name": "Control and Communication Systems", "alternate_names": ["ICML"]
            }),
            # Real providers sometimes supply only the canonical venue name.
            s2_paper(8, venue="", publicationVenue={
                "id": "canonical-venue-id", "name": "International Conference on Machine Learning"
            }),
        ]}
        self.importer.discover_s2()
        self.assertEqual([row["doi"] for row in self.papers()], ["10.1145/1000000.1", "10.1145/1000000.8"])

    def test_openalex_sources_paginate_and_use_labels_and_aliases(self):
        alias_source = {"id": SOURCE_ID, "display_name": "Conference proceedings", "alternate_titles": [
            "International Conference on Machine Learning"
        ]}
        second_source = {"id": "https://openalex.org/S4306460213", "display_name": "ICML '24"}
        workshop_source = {"id": "https://openalex.org/S4306460214", "display_name": "ICML Workshops",
                           "alternate_titles": ["International Conference on Machine Learning"]}
        self.oa.sources.side_effect = [
            oa_page([alias_source, workshop_source], "second-source-page"),
            oa_page([second_source]),
            oa_page([alias_source]),
        ]
        sources = self.importer.discover_sources(self.venue)
        first_query, second_query = self.venue.openalex_queries
        self.assertEqual(self.oa.sources.call_args_list, [
            call(first_query, "*"), call(first_query, "second-source-page"), call(second_query, "*")
        ])
        self.assertEqual([row["source_id"] for row in sources], [SOURCE_ID, "https://openalex.org/S4306460213"])
        self.importer.discover_sources(self.venue)
        self.assertEqual(self.oa.sources.call_count, 3)

    def test_book_mislabeled_as_icml_conference_is_excluded(self):
        # Real S2 shape: a workshop proceedings volume is indexed under ICML and
        # simultaneously classified as Book, JournalArticle, and Conference.
        self.s2.works.return_value = {"data": [s2_paper(
            title="MLWS '24: 23rd Workshop on Representation Learning (MLWS)",
            publicationTypes=["Book", "JournalArticle", "Conference"],
        )]}
        self.importer.discover_s2()
        self.assertEqual(self.papers(), [])

    def test_actual_conference_papers_with_mixed_book_tags_are_kept(self):
        self.s2.works.return_value = {"data": [
            s2_paper(1, title="Learning Representations from Limited Labels",
                     publicationTypes=["JournalArticle", "Book", "Conference"]),
            s2_paper(2, publicationTypes=["Book", "Conference"]),
            s2_paper(3, publicationTypes=["Book", "JournalArticle"]),
        ]}
        self.importer.discover_s2()
        self.assertEqual(len(self.papers()), 3)

    def test_old_filter_checkpoint_does_not_skip_corrected_discovery(self):
        import json
        legacy = json.dumps({"version": 1, "stage": "s2-discover", "venue": "icml", "year": YEAR}, sort_keys=True)
        with self.db.conn:
            self.db.save_checkpoint(legacy, None, True, 347, 1)
        self.s2.works.return_value = {"data": [s2_paper(1, publicationTypes=["Book", "Conference"])]}
        self.importer.discover_s2()
        self.s2.works.assert_called_once()
        self.assertEqual(len(self.papers()), 1)
        self.assertTrue(self.db.checkpoint(job_key("s2-discover", "icml", year=YEAR))["done"])

    def test_openalex_source_failure_resumes_its_own_cursor(self):
        self.oa.sources.side_effect = [
            oa_page([{"id": SOURCE_ID, "display_name": "ICML"}], "source-resume"), APIError("temporary failure")
        ]
        with self.assertRaises(APIError):
            self.importer.discover_sources(self.venue)
        self.oa.sources.side_effect = [oa_page([]), oa_page([])]
        self.oa.sources.reset_mock()
        sources = self.importer.discover_sources(self.venue)
        self.assertEqual(self.oa.sources.call_args_list[0], call(self.venue.openalex_queries[0], "source-resume"))
        self.assertEqual(len(sources), 1)

    def test_openalex_multiple_pages_resume_without_duplicates(self):
        self.seed_source()
        self.oa.works.side_effect = [oa_page([oa_paper(1)], "works-resume"), RequestLimit("test budget")]
        with self.assertRaises(RequestLimit):
            self.importer.discover_oa()
        job = job_key("oa-discover", "icml", year=YEAR, source=SOURCE_ID)
        self.assertEqual(self.db.checkpoint(job)["cursor"], "works-resume")
        self.assertFalse(self.db.checkpoint(job)["done"])
        self.oa.reset_mock()
        self.oa.works.side_effect = [oa_page([oa_paper(1), oa_paper(2, abstract_inverted_index={"Abstract": [0]})])]
        restarted = Importer(self.db, [self.venue], YEAR, YEAR, openalex=self.oa)
        restarted.discover_oa()
        self.oa.works.assert_called_once_with(SOURCE_ID, YEAR, "works-resume")
        self.assertEqual(len(self.papers()), 2)
        self.assertIsNone(self.papers()[0]["abstract"])
        self.assertEqual(self.papers()[1]["abstract"], "Abstract")
        self.assertTrue(self.db.checkpoint(job)["done"])
        self.assertEqual(self.db.checkpoint(job)["fetched"], 3)
        restarted.discover_oa()
        self.assertEqual(self.oa.works.call_count, 1)

    def test_openalex_missing_results_retains_checkpoint(self):
        self.seed_source()
        self.oa.works.side_effect = [oa_page([oa_paper(1)], "works-resume"), {"error": "bad response"}]
        with self.assertRaisesRegex(APIError, "no works array"):
            self.importer.discover_oa()
        state = self.db.checkpoint(job_key("oa-discover", "icml", year=YEAR, source=SOURCE_ID))
        self.assertEqual(state["cursor"], "works-resume")
        self.assertEqual(state["fetched"], 1)
        self.assertFalse(state["done"])

    def test_openalex_filters_volumes_front_matter_and_wrong_year(self):
        self.seed_source()
        self.oa.works.return_value = oa_page([
            oa_paper(1), oa_paper(2, type="book"), oa_paper(3, type="paratext"),
            oa_paper(4, display_name="Proceedings of ICML 2025: Part I"),
            oa_paper(5, display_name="Table of contents"), oa_paper(6, publication_year=YEAR - 1),
        ])
        self.importer.discover_oa()
        self.assertEqual([row["doi"] for row in self.papers()], ["10.1145/1000000.1"])

    def test_null_s2_doi_backfill_is_remembered_until_refresh(self):
        with self.db.conn:
            self.db.upsert("openalex", oa_paper(), "icml", YEAR)
        self.s2.dois.return_value = [None]
        self.importer.enrich_s2()
        self.importer.enrich_s2()
        self.s2.dois.assert_called_once_with(["10.1145/1000000.1"])
        self.assertIsNone(self.papers()[0]["abstract"])
        self.db.refresh(["icml"], YEAR, YEAR)
        self.s2.dois.return_value = [s2_paper(abstract="Recovered abstract")]
        self.importer.enrich_s2()
        self.assertEqual(self.s2.dois.call_count, 2)
        self.assertEqual(len(self.papers()), 1)
        self.assertEqual(self.papers()[0]["abstract"], "Recovered abstract")
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM source_records").fetchone()[0], 2)

    def test_null_openalex_doi_backfill_is_remembered_until_refresh(self):
        with self.db.conn:
            self.db.upsert("semantic_scholar", s2_paper(), "icml", YEAR)
        self.oa.doi.return_value = None
        self.importer.enrich_oa()
        self.importer.enrich_oa()
        self.oa.doi.assert_called_once_with("10.1145/1000000.1")
        self.db.refresh(["icml"], YEAR, YEAR)
        self.oa.doi.return_value = oa_paper(abstract_inverted_index={"Recovered": [0], "abstract": [1]})
        self.importer.enrich_oa()
        self.assertEqual(self.oa.doi.call_count, 2)
        self.assertEqual(len(self.papers()), 1)
        self.assertEqual(self.papers()[0]["abstract"], "Recovered abstract")

    def test_s2_batch_mismatch_does_not_mark_attempts(self):
        with self.db.conn:
            self.db.upsert("openalex", oa_paper(), "icml", YEAR)
        self.s2.dois.return_value = []
        with self.assertRaisesRegex(APIError, "length mismatch"):
            self.importer.enrich_s2()
        self.assertEqual(len(self.db.missing_for("semantic_scholar", ["icml"], YEAR, YEAR)), 1)

    def test_s2_wrong_doi_rolls_back_entire_backfill_batch(self):
        with self.db.conn:
            self.db.upsert("openalex", oa_paper(1), "icml", YEAR)
            self.db.upsert("openalex", oa_paper(2), "icml", YEAR)
        self.s2.dois.return_value = [s2_paper(1, abstract="Uncommitted"), s2_paper(3, abstract="Wrong DOI")]
        with self.assertRaisesRegex(APIError, "different DOI"):
            self.importer.enrich_s2()
        self.assertTrue(all(row["abstract"] is None for row in self.papers()))
        self.assertEqual(len(self.db.missing_for("semantic_scholar", ["icml"], YEAR, YEAR)), 2)

    def test_all_requested_years_include_empty_years(self):
        self.s2.works.side_effect = lambda names, year, token: {
            "data": [s2_paper(year, year=year)] if year == 2026 else []
        }
        importer = Importer(self.db, [self.venue], 2017, 2026, semantic_scholar=self.s2)
        importer.discover_s2()
        self.assertEqual([args.args[1] for args in self.s2.works.call_args_list], list(range(2017, 2027)))
        for year in range(2017, 2027):
            self.assertTrue(self.db.checkpoint(job_key("s2-discover", "icml", year=year))["done"])
            self.assertEqual(self.db.coverage("icml", year)["papers"], int(year == 2026))

    def test_provider_failure_still_allows_other_provider_to_import(self):
        self.seed_source()
        self.s2.works.side_effect = APIError("Semantic Scholar unavailable")
        self.oa.works.return_value = oa_page([oa_paper(abstract_inverted_index={"Found": [0]})])
        with self.assertLogs("reference_finder.importer", level="ERROR"):
            errors = self.importer.run()
        self.assertEqual(errors, ["Semantic Scholar unavailable"])
        self.assertEqual(self.papers()[0]["abstract"], "Found")
        self.assertIsNone(self.db.checkpoint(job_key("s2-discover", "icml", year=YEAR)))
        self.s2.dois.assert_not_called()


if __name__ == "__main__":
    unittest.main()
