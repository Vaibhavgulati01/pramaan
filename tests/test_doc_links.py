"""The doc-link checker, and the slug rule it gets wrong by default.

Under repo-only judging the README is the deliverable, so a dead anchor
is a defect in the product. These tests pin the one subtle rule --
GitHub does not collapse whitespace when slugifying -- because the first
version of the checker did collapse it and reported a *correct* link as
broken.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_doc_links", ROOT / "scripts" / "check_doc_links.py"
)
assert _spec and _spec.loader
check_doc_links = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_doc_links)

slugify = check_doc_links.slugify


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        # The load-bearing case: "&" is stripped but both spaces around it
        # survive as separate hyphens. Collapsing gives "shift-robustness",
        # which is not the anchor GitHub generates.
        ("Shift & robustness", "shift--robustness"),
        ("The problem", "the-problem"),
        ("Why false positives cost more than false negatives",
         "why-false-positives-cost-more-than-false-negatives"),
        # Inline code is unwrapped, not dropped.
        ("Running `pramaan all`", "running-pramaan-all"),
        # Punctuation goes; the words around it do not merge.
        ("Roughly two-thirds of the forensics signal is source, not fraud",
         "roughly-two-thirds-of-the-forensics-signal-is-source-not-fraud"),
        ("What follows from that", "what-follows-from-that"),
    ],
)
def test_slugify_matches_github_rules(heading: str, expected: str) -> None:
    assert slugify(heading) == expected


def test_duplicate_headings_get_numeric_suffixes(tmp_path: Path) -> None:
    doc = tmp_path / "d.md"
    doc.write_text("# Notes\n\n## Detail\n\n## Detail\n\n## Detail\n", encoding="utf-8")
    assert check_doc_links.anchors_of(doc) == {"notes", "detail", "detail-1", "detail-2"}


def test_headings_inside_fenced_code_are_not_anchors(tmp_path: Path) -> None:
    """A `#` in a shell snippet is a comment, not a heading."""
    doc = tmp_path / "d.md"
    doc.write_text(
        "# Real\n\n```bash\n# Not a heading\npramaan all\n```\n", encoding="utf-8"
    )
    assert check_doc_links.anchors_of(doc) == {"real"}


def test_repo_docs_have_no_broken_links_or_dead_anchors() -> None:
    """The gate itself, so a bad link fails the suite and not only CI."""
    assert check_doc_links.main() == 0
