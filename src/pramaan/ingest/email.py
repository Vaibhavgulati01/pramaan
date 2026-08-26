"""Email canonicalisation (PRAMAAN_v2_architecture.md Sec.4 L0).

Lowercase; for Gmail-family domains, strip dots and everything from '+'
onward in the local part, since Gmail treats `a.b+tag@gmail.com`,
`ab@gmail.com`, and `ab+anything@googlemail.com` as the same inbox. This
is a well-known abuse vector for "distinct" claimant accounts that are
really one person, so canonicalising it matters for the leakage audit.
"""

from __future__ import annotations

_GMAIL_FAMILY_DOMAINS = frozenset({"gmail.com", "googlemail.com"})


def canonicalize_email(raw: str | None) -> str | None:
    """Returns a canonical `local@domain` string, or None if `raw` isn't
    a syntactically plausible email (no '@', empty local/domain part)."""
    if not raw:
        return None

    cleaned = raw.strip().lower()
    local, sep, domain = cleaned.rpartition("@")
    if not sep or not local or not domain:
        return None

    if domain in _GMAIL_FAMILY_DOMAINS:
        local = local.split("+", 1)[0]
        local = local.replace(".", "")
        domain = "gmail.com"

    if not local:
        return None
    return f"{local}@{domain}"
