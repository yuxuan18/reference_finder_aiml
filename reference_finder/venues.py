"""Conference catalog and conservative matching of provider venue labels.

Provider venue names are metadata, not paper titles. Never use ``matches`` to
classify a paper from a keyword in its title or abstract. A generic publication
series is not evidence of a particular conference.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold().replace("&", " and ")
    value = "".join(c for c in value if not unicodedata.combining(c))
    # API records include both "ICML '25" and "ICLR2025" spellings.
    value = re.sub(r"(?<=[a-z])(?=\d)", " ", value)
    value = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " ", value)
    value = re.sub(r"\b\d{2,4}\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


_EXCLUDED = re.compile(
    r"\b(?:workshops?|companions?|posters?|doctoral|tutorials?|"
    r"student abstracts?)\b|\bextended abstracts\b|\bsummer school\b"
)


def is_excluded(name: str) -> bool:
    """Whether a venue label identifies an excluded associated event."""
    return isinstance(name, str) and bool(_EXCLUDED.search(_normalize(name)))

# Full conference names may appear inside a proceedings title followed by a
# place, dates, or a volume number. Short acronyms are matched only as labels.
_FULL_NAMES = {
    "iclr": (r"\binternational conference on learning representations\b",),
    "icml": (r"\binternational conference on machine learning\b",),
    "neurips": (r"\b(?:advances in|conference on) neural information processing systems\b",),
    "aaai": (r"\baaai conference on artificial intelligence\b",),
    "colm": (r"\bconference on language modeling\b",),
}

_SHORT_NAMES = {
    "iclr": {"iclr"},
    "icml": {"icml"},
    "neurips": {"neurips", "nips", "neural information processing systems"},
    "aaai": {"aaai"},
    "colm": {"colm"},
}

@dataclass(frozen=True)
class Venue:
    key: str
    name: str
    openalex_queries: tuple[str, ...]
    semantic_scholar_names: tuple[str, ...]
    first_year: int = 1900

    def matches(self, name: str) -> bool:
        """Recognize a main-conference venue label, rejecting adjacent events."""
        if not isinstance(name, str) or not name.strip():
            return False
        normalized = _normalize(name)
        if _EXCLUDED.search(normalized):
            return False
        if any(re.search(pattern, normalized) for pattern in _FULL_NAMES[self.key]):
            return True
        label = re.sub(r"^(?:proceedings(?: of)?(?: the)? |the )", "", normalized)
        label = re.sub(r" (?:proceedings|part [ivx]+)$", "", label)
        return label in _SHORT_NAMES[self.key]


VENUES = {
    "iclr": Venue(
        "iclr", "International Conference on Learning Representations (ICLR)",
        ("International Conference on Learning Representations", "ICLR"),
        ("International Conference on Learning Representations", "ICLR"),
    ),
    "icml": Venue(
        "icml", "International Conference on Machine Learning (ICML)",
        ("International Conference on Machine Learning", "ICML"),
        ("International Conference on Machine Learning", "ICML"),
    ),
    "neurips": Venue(
        "neurips", "Neural Information Processing Systems (NeurIPS)",
        ("Advances in Neural Information Processing Systems", "Neural Information Processing Systems"),
        ("Advances in Neural Information Processing Systems", "Conference on Neural Information Processing Systems", "NeurIPS", "NIPS"),
    ),
    "aaai": Venue(
        "aaai", "AAAI Conference on Artificial Intelligence",
        ("AAAI Conference on Artificial Intelligence",),
        ("AAAI Conference on Artificial Intelligence", "AAAI"),
    ),
    "colm": Venue(
        "colm", "Conference on Language Modeling (COLM)",
        ("Conference on Language Modeling", "COLM"),
        ("Conference on Language Modeling", "COLM"), first_year=2024,
    ),
}

DEFAULT_VENUES = ("iclr", "icml", "neurips", "aaai", "colm")


def source_matches(venue: Venue, source: dict) -> bool:
    """Check an OpenAlex source's label and alternate labels, never its ID."""
    if is_excluded(source.get("display_name")):
        return False
    names = [source.get("display_name"), *(source.get("alternate_titles") or [])]
    return any(venue.matches(name) for name in names if isinstance(name, str))
