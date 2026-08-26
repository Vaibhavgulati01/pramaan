import pytest

from pramaan.ingest.email import canonicalize_email


@pytest.mark.parametrize(
    "raw",
    [
        "abc.def@gmail.com",
        "abcdef@gmail.com",
        "abc.def+refund@gmail.com",
        "ABC.DEF@GMAIL.COM",
        "a.b.c.d.e.f@googlemail.com",
        "abc.def+anything+more@gmail.com",
    ],
)
def test_gmail_family_dot_and_plus_stripping(raw: str) -> None:
    assert canonicalize_email(raw) == "abcdef@gmail.com"


def test_non_gmail_domain_only_lowercased_not_dot_stripped() -> None:
    assert canonicalize_email("A.B+tag@Yahoo.com") == "a.b+tag@yahoo.com"


def test_googlemail_normalizes_domain_to_gmail() -> None:
    assert canonicalize_email("someone@googlemail.com") == "someone@gmail.com"


@pytest.mark.parametrize("bad", [None, "", "not-an-email", "@nolocal.com", "nodomain@"])
def test_unparseable_returns_none(bad: str | None) -> None:
    assert canonicalize_email(bad) is None
