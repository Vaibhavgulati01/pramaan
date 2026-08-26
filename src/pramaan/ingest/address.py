"""Transliteration-tolerant address canonicalisation and PIN-bucketed
fuzzy matching (PRAMAAN_v2_architecture.md Sec.4 L0).

Addresses can't be reduced to one exact canonical key the way phone/email
can - "Flat 12, Opp City Mall, Nr Station" and "flat no 12 opposite city
mall near railway station" are the same place typed differently. So this
module normalises to a comparable form (Unicode NFKC, Devanagari->Latin
transliteration, abbreviation expansion, stopword removal) and exposes a
fuzzy match that is only ever evaluated within an exact PIN-code bucket -
comparing full free-text address strings nationwide would be both slow
and a fuzzy-matching false-positive minefield.
"""

from __future__ import annotations

import re
import unicodedata

from indic_transliteration import sanscript
from rapidfuzz import fuzz

_DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")

# Expansions applied as whole-word replacements, post-transliteration,
# pre-stopword-removal. Deliberately conservative: only common Indian
# addressing abbreviations named in the spec plus a few obvious siblings.
_ABBREVIATIONS = {
    "flt": "flat",
    "opp": "opposite",
    "nr": "near",
    "rd": "road",
    "st": "street",
    "apt": "apartment",
    "soc": "society",
    "bldg": "building",
    "ext": "extension",
    "ph": "phase",
    "sec": "sector",
    "blk": "block",
    "hno": "house number",
    "no": "number",
    "co": "care of",
    "flr": "floor",
    "twr": "tower",
}

# Low-information filler words, removed after abbreviation expansion so
# an expansion like "opp" -> "opposite" is never itself stripped.
_STOPWORDS = frozenset({"the", "and", "of", "at", "in", "on", "to", "a", "an"})

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def _transliterate_if_devanagari(text: str) -> str:
    if _DEVANAGARI_RANGE.search(text):
        return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    return text


def canonicalize_address(raw: str | None) -> str:
    """Normalises free-text address into a space-joined token string
    suitable for fuzzy comparison. Returns "" for empty/missing input
    (never None - an empty canonical address simply can never fuzzy-match
    anything, which is the correct behaviour for a blocking key)."""
    if not raw:
        return ""

    text = unicodedata.normalize("NFKC", raw)
    text = _transliterate_if_devanagari(text)
    text = text.lower()
    text = _NON_ALNUM.sub(" ", text)

    tokens = [_ABBREVIATIONS.get(tok, tok) for tok in text.split()]
    # Abbreviation expansion can introduce multi-word phrases (e.g. "house
    # number"); re-split so every stopword-removal/comparison unit is one word.
    tokens = " ".join(tokens).split()
    tokens = [tok for tok in tokens if tok not in _STOPWORDS]

    return _WHITESPACE.sub(" ", " ".join(tokens)).strip()


def canonicalize_pin(raw: str | None) -> str | None:
    """6-digit Indian PIN code, or None if it doesn't parse as one."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    return digits if len(digits) == 6 else None


def extract_numeric_tokens(norm_address: str) -> set[str]:
    """House/flat/plot numbers from a canonicalised address, including the
    parts of compound forms like `59/73` (which canonicalises to the two
    tokens `59` and `73`)."""
    return {tok for tok in norm_address.split() if tok.isdigit()}


def addresses_match(
    norm_address_a: str,
    pin_a: str | None,
    norm_address_b: str,
    pin_b: str | None,
    threshold: float = 85.0,
) -> bool:
    """True only if all of the following hold:

    1. Both addresses are in the exact same PIN-code bucket. Load-bearing,
       not an optimisation: without it, a high fuzzy similarity between two
       genuinely different addresses in different cities is entirely
       plausible and would wrongly merge two claimants' entities.
    2. Their canonicalised forms meet `threshold` fuzzy token-set similarity.
    3. If BOTH carry numeric tokens (house/flat/plot numbers), those sets
       must intersect.

    Rule 3 exists because `token_set_ratio` is deliberately forgiving about
    differing tokens, which is exactly wrong for addresses: within one PIN
    code the house number often IS the entire distinguishing signal, while
    street/locality words are shared by design. Without it, real
    (empirically observed, see tests) false merges occur - e.g.
    `H.No. 97 Chanda Marg, Faridabad` vs `H.No. 51 Garg Marg, Faridabad`
    scores above 85 on text alone despite being two different households.
    Since a false merge silently contaminates a train/test split - the
    precise failure `eval/entity_leakage_audit.py` exists to prevent -
    over-merging is not a benign error here.

    When only one side has numbers (or neither does), rule 3 is skipped
    and text similarity decides: a missing house number is weak evidence
    of difference, not evidence of sameness.
    """
    if not pin_a or not pin_b or pin_a != pin_b:
        return False
    if not norm_address_a or not norm_address_b:
        return False
    if fuzz.token_set_ratio(norm_address_a, norm_address_b) < threshold:
        return False

    nums_a = extract_numeric_tokens(norm_address_a)
    nums_b = extract_numeric_tokens(norm_address_b)
    if nums_a and nums_b and not (nums_a & nums_b):
        return False
    return True
