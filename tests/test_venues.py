import unittest

from reference_finder.venues import DEFAULT_VENUES, VENUES, is_excluded, source_matches


class VenueTests(unittest.TestCase):
    def test_only_five_ai_ml_venues_are_available(self):
        self.assertEqual(tuple(VENUES), DEFAULT_VENUES)
        self.assertEqual(DEFAULT_VENUES, ('iclr', 'icml', 'neurips', 'aaai', 'colm'))
        self.assertEqual(VENUES['colm'].first_year, 2024)

    def test_official_names_proceedings_and_aliases(self):
        examples = {
            'iclr': ['International Conference on Learning Representations', 'ICLR 2025'],
            'icml': ['Proceedings of the 42nd International Conference on Machine Learning', "ICML '25"],
            'neurips': ['Advances in Neural Information Processing Systems 38', 'NeurIPS 2025', 'NIPS 2017'],
            'aaai': ['Proceedings of the AAAI Conference on Artificial Intelligence', 'AAAI-25'],
            'colm': ['Conference on Language Modeling', 'COLM 2024'],
        }
        for key, labels in examples.items():
            for label in labels:
                with self.subTest(key=key, label=label):
                    self.assertEqual([v.key for v in VENUES.values() if v.matches(label)], [key])

    def test_adjacent_events_and_generic_series_are_not_main_venues(self):
        for label in ('ICLR 2025 Workshop', 'ICML 2024 Tutorials', 'NeurIPS 2025 Workshops',
                      'AAAI 2025 Doctoral Consortium', 'COLM 2025 Workshop',
                      'Proceedings of Machine Learning Research', 'PMLR', 'Machine Learning',
                      'AAAI/ACM Conference on AI, Ethics, and Society', 'COLING 2024', 'ACML 2024', '', None):
            with self.subTest(label=label):
                self.assertFalse(any(v.matches(label) for v in VENUES.values()))

    def test_queries_and_canonical_names_match_the_catalog(self):
        for key, venue in VENUES.items():
            self.assertEqual(key, venue.key)
            self.assertTrue(venue.openalex_queries)
            for name in venue.semantic_scholar_names:
                self.assertTrue(venue.matches(name), name)

    def test_source_alias_cannot_override_an_explicit_workshop_label(self):
        self.assertTrue(source_matches(VENUES['icml'], {'display_name': 'Unrecognized', 'alternate_titles': ['ICML']}))
        self.assertFalse(source_matches(VENUES['icml'], {'display_name': 'ICML Workshops', 'alternate_titles': ['ICML']}))
        self.assertFalse(source_matches(VENUES['icml'], {'display_name': 'Proceedings of Machine Learning Research'}))

    def test_exclusion_helper(self):
        self.assertTrue(is_excluded('AAAI Student Abstracts'))
        self.assertTrue(is_excluded('ICML Companion'))
        self.assertFalse(is_excluded('ICML'))
        self.assertFalse(is_excluded(None))


if __name__ == '__main__':
    unittest.main()
