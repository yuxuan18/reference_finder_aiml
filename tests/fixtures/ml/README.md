These are small, synthetic source-format fixtures. They retain the navigation,
metadata, and section structures needed by the collectors without reproducing
full conference pages or real paper abstracts.

Formats checked against official sources in October 2026:

- PMLR catalog and ICML volume: https://proceedings.mlr.press/ and https://proceedings.mlr.press/v267/
- NeurIPS modern volumes and historical paper layout: https://proceedings.neurips.cc/
- AAAI OJS archive/issue structures, including migrated 2017/2018 volumes: https://ojs.aaai.org/index.php/AAAI/issue/view/302
- OpenReview API 1 and 2 notes and accepted venue IDs: https://docs.openreview.net/reference

OpenReview API requests were denied by the network during development; its
fixtures cover documented structures. AAAI access was intermittent, but current
and migrated 2017 issue and article layouts were checked successfully. Full
live coverage must be checked when collecting the corpus.
