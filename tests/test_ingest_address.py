import pytest

from pramaan.ingest.address import (
    addresses_match,
    canonicalize_address,
    canonicalize_pin,
    extract_numeric_tokens,
)


def test_expands_known_abbreviations() -> None:
    norm = canonicalize_address("Flt 12, Opp City Mall, Nr Station")
    assert "flat" in norm
    assert "opposite" in norm
    assert "near" in norm
    assert "flt" not in norm.split()
    assert "opp" not in norm.split()
    assert "nr" not in norm.split()


def test_removes_stopwords_without_touching_expanded_words() -> None:
    norm = canonicalize_address("House of the near station")
    tokens = norm.split()
    assert "of" not in tokens
    assert "the" not in tokens
    assert "near" in tokens  # not a stopword itself


def test_transliterates_devanagari_to_ascii() -> None:
    norm = canonicalize_address("मुंबई नगर")
    assert norm.isascii()
    assert norm != ""


def test_empty_input_returns_empty_string() -> None:
    assert canonicalize_address(None) == ""
    assert canonicalize_address("") == ""


def test_case_and_punctuation_insensitive() -> None:
    a = canonicalize_address("FLAT-12, City Mall!!")
    b = canonicalize_address("flat 12 city mall")
    assert a == b


def test_canonicalize_pin_valid_and_invalid() -> None:
    assert canonicalize_pin("400001") == "400001"
    assert canonicalize_pin(" 400 001 ") == "400001"
    assert canonicalize_pin("4000") is None
    assert canonicalize_pin(None) is None


def test_addresses_match_requires_same_pin() -> None:
    addr = canonicalize_address("Flat 12 City Mall")
    assert not addresses_match(addr, "400001", addr, "400002")


def test_addresses_match_same_pin_similar_text() -> None:
    a = canonicalize_address("Flat 12, Opp City Mall, Nr Station")
    b = canonicalize_address("flat no 12 opposite city mall near railway station")
    assert addresses_match(a, "400001", b, "400001")


def test_addresses_match_same_pin_unrelated_text() -> None:
    a = canonicalize_address("Flat 12 Green Society Bandra")
    b = canonicalize_address("Shop 45 Industrial Estate Andheri")
    assert not addresses_match(a, "400001", b, "400001")


def test_addresses_match_missing_pin_is_false() -> None:
    a = canonicalize_address("Flat 12 City Mall")
    assert not addresses_match(a, None, a, None)


def test_extract_numeric_tokens() -> None:
    assert extract_numeric_tokens(canonicalize_address("H.No. 97 Chanda Marg")) == {"97"}
    # Compound forms like 59/73 canonicalise to two separate numeric tokens.
    assert extract_numeric_tokens(canonicalize_address("59/73 Pillai Road")) == {"59", "73"}
    assert extract_numeric_tokens(canonicalize_address("Green Society Bandra")) == set()


@pytest.mark.parametrize(
    ("addr_a", "addr_b"),
    [
        # Both pairs were observed merging incorrectly at threshold=85 on
        # text similarity alone, in a 2000-claim simulated corpus - two
        # different households sharing a street/locality name inside one
        # PIN code. This is the regression that motivated the numeric-token
        # rule in addresses_match.
        ("H.No. 97 Chanda Marg, Faridabad", "H.No. 51 Garg Marg, Faridabad"),
        (
            "59/73 Pillai Road, Raurkela Industrial Township",
            "18/26 Oak Road, Raurkela Industrial Township",
        ),
    ],
)
def test_different_house_numbers_same_locality_do_not_merge(addr_a: str, addr_b: str) -> None:
    a, b = canonicalize_address(addr_a), canonicalize_address(addr_b)
    assert not addresses_match(a, "110001", b, "110001")


def test_same_house_number_and_locality_still_matches() -> None:
    a = canonicalize_address("H.No. 97 Chanda Marg, Faridabad")
    b = canonicalize_address("House No 97, Chanda Marg, Faridabad")
    assert addresses_match(a, "110001", b, "110001")


def test_numeric_rule_skipped_when_one_side_has_no_number() -> None:
    # A missing house number is weak evidence of difference, not evidence
    # of sameness, so the numeric-intersection rule does not apply and
    # text similarity alone decides - here it's a containment match.
    a = canonicalize_address("Flat 12, Green Society, Bandra")
    b = canonicalize_address("Green Society, Bandra")
    assert extract_numeric_tokens(b) == set()
    assert addresses_match(a, "400001", b, "400001")


def test_neither_side_has_numbers_falls_back_to_text_similarity() -> None:
    a = canonicalize_address("Green Society, Bandra West")
    b = canonicalize_address("green society bandra west")
    assert addresses_match(a, "400001", b, "400001")
