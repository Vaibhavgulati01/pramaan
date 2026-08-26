from pramaan.ingest.device import canonicalize_device


def test_identical_inputs_produce_identical_fingerprint() -> None:
    a = canonicalize_device("UA/1.0", "1920x1080", "Asia/Kolkata", ["Arial", "Verdana"])
    b = canonicalize_device("UA/1.0", "1920x1080", "Asia/Kolkata", ["Arial", "Verdana"])
    assert a == b


def test_font_list_order_does_not_matter() -> None:
    a = canonicalize_device("UA/1.0", "1920x1080", "Asia/Kolkata", ["Arial", "Verdana"])
    b = canonicalize_device("UA/1.0", "1920x1080", "Asia/Kolkata", ["Verdana", "Arial"])
    assert a == b


def test_case_insensitive() -> None:
    a = canonicalize_device("UA/1.0", "1920X1080", "Asia/Kolkata", ["Arial"])
    b = canonicalize_device("ua/1.0", "1920x1080", "asia/kolkata", ["arial"])
    assert a == b


def test_different_inputs_produce_different_fingerprints() -> None:
    a = canonicalize_device("UA/1.0", "1920x1080", "Asia/Kolkata", ["Arial"])
    b = canonicalize_device("UA/2.0", "1920x1080", "Asia/Kolkata", ["Arial"])
    assert a != b


def test_all_fields_missing_returns_none() -> None:
    assert canonicalize_device(None, None, None, None) is None


def test_returns_a_hex_digest() -> None:
    fp = canonicalize_device("UA/1.0", None, None, None)
    assert fp is not None
    assert len(fp) == 64
    int(fp, 16)  # raises ValueError if not valid hex
