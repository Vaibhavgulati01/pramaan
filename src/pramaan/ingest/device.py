"""Device fingerprint canonicalisation (PRAMAAN_v2_architecture.md Sec.4 L0).

Hashes user-agent + screen resolution + timezone + font list into a
single fingerprint key.

**Instability caveat (documented, not hidden):** unlike phone/email,
this is NOT a stable identity signal. A browser or OS update changes the
UA string; installing/removing an app can change the enumerable font
list; travel changes timezone. Two fingerprints differing does not mean
two different people, and two matching fingerprints only mean "same
device configuration observed," not "same person." For that reason
`ingest/identity.py` does NOT use this as a hard entity-linking signal
for the entity-leakage audit (only phone/email/address are used there) -
it is retained purely as a P4 behavioural feature (device reuse across
claims is informative even when noisy), computed here and consumed
downstream in `pillars/p4_behaviour.py` (Phase 2).
"""

from __future__ import annotations

import hashlib


def canonicalize_device(
    user_agent: str | None,
    screen_resolution: str | None,
    timezone: str | None,
    font_list: list[str] | None,
) -> str | None:
    """Returns a sha256 hex digest fingerprint, or None if every input
    field is empty (nothing to fingerprint)."""
    if not any([user_agent, screen_resolution, timezone, font_list]):
        return None

    fonts_sorted = ",".join(sorted(font_list)) if font_list else ""
    fingerprint_input = "|".join(
        part.strip().lower()
        for part in (
            user_agent or "",
            screen_resolution or "",
            timezone or "",
            fonts_sorted,
        )
    )
    return hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
