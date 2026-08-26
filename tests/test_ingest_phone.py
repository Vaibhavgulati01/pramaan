import pytest

from pramaan.ingest.phone import canonicalize_phone


@pytest.mark.parametrize(
    "raw",
    [
        "9876543210",
        "+91 98765 43210",
        "+919876543210",
        "919876543210",
        "09876543210",
        "0 98765 43210",
        "+91-98765-43210",
        "  9876543210  ",
    ],
)
def test_all_common_formats_canonicalize_to_same_key(raw: str) -> None:
    assert canonicalize_phone(raw) == "9876543210"


@pytest.mark.parametrize("bad", [None, "", "12345", "notaphone", "98765432101234"])
def test_unparseable_returns_none(bad: str | None) -> None:
    assert canonicalize_phone(bad) is None
