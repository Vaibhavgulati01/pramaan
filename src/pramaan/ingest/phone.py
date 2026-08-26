"""India phone canonicalisation (PRAMAAN_v2_architecture.md Sec.4 L0).

Strip +91, leading 0, spaces, and hyphens down to a bare 10-digit key so
the same claimant's phone number canonicalises identically regardless of
how it was typed. This key is one of the exact-match signals
`ingest/identity.py` uses to cluster claims into canonical identities for
the entity-leakage audit.
"""

from __future__ import annotations

import re

_NON_DIGIT = re.compile(r"\D")


def canonicalize_phone(raw: str | None) -> str | None:
    """Returns a 10-digit canonical key, or None if `raw` doesn't reduce
    to one (missing, malformed, not an Indian mobile-length number).
    """
    if not raw:
        return None

    digits = _NON_DIGIT.sub("", raw)

    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]

    if len(digits) == 10 and digits.isdigit():
        return digits
    return None
