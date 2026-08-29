r"""Path redaction for the committed demo recording.

`assets/session.jsonl` and the GIF rendered from it go into a public
repository, so absolute paths from the machine that recorded them must
not survive capture.

The first implementation used `str.replace` and redacted nothing:
`Path.cwd()` returns `c:\Users\...` on Windows while the child processes
print `C:\Users\...`, and a case-sensitive replace silently matches
neither. Silently is the operative word -- the transcript looked fine
until it was grepped.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "record_session", ROOT / "scripts" / "record_session.py"
)
assert _spec and _spec.loader
record_session = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record_session)

redact = record_session.redact
LABEL = "~/pramaan"


def test_redacts_exact_windows_path() -> None:
    root = Path(r"C:\Users\someone\Projects\pramaan")
    line = r"INFO wrote C:\Users\someone\Projects\pramaan\reports\smoke\metrics.json"
    assert redact(line, root, LABEL) == r"INFO wrote ~/pramaan\reports\smoke\metrics.json"
    assert "someone" not in redact(line, root, LABEL)


def test_redacts_despite_drive_letter_case_mismatch() -> None:
    """The bug that shipped: cwd gives `c:`, the child prints `C:`."""
    root = Path(r"c:\Users\someone\Projects\pramaan")
    line = r"INFO wrote C:\Users\someone\Projects\pramaan\data\smoke\model"
    assert "someone" not in redact(line, root, LABEL)


def test_redacts_forward_slash_form() -> None:
    root = Path(r"C:\Users\someone\Projects\pramaan")
    line = "loading C:/Users/someone/Projects/pramaan/data/smoke/images"
    assert "someone" not in redact(line, root, LABEL)


def test_redacts_posix_paths() -> None:
    root = Path("/home/someone/pramaan")
    line = "INFO wrote /home/someone/pramaan/reports/smoke/metrics.json"
    assert redact(line, root, LABEL) == "INFO wrote ~/pramaan/reports/smoke/metrics.json"


def test_leaves_unrelated_text_alone() -> None:
    root = Path(r"C:\Users\someone\Projects\pramaan")
    line = "PR-AUC 0.1289 [0.0695, 0.2837]"
    assert redact(line, root, LABEL) == line


def test_committed_transcript_carries_no_absolute_paths() -> None:
    """The artifact itself, not just the function that produces it."""
    transcript = ROOT / "assets" / "session.jsonl"
    if not transcript.exists():
        return
    # Match against the DECODED text, not the raw file. Searching raw JSON
    # means guessing how many backslashes survived escaping, and guessing
    # wrong makes this test pass against a transcript still full of home
    # directories -- which is exactly what the first version did.
    absolute = re.compile(r"[A-Za-z]:[\\/]Users[\\/]|/home/|/Users/")
    offenders: list[str] = []
    for raw in transcript.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        text = str(record.get("line", "")) + json.dumps(record.get("meta", ""))
        if absolute.search(text):
            offenders.append(text[:110])

    assert not offenders, (
        f"{transcript.name} still contains absolute home paths in "
        f"{len(offenders)} line(s), e.g.:\n  "
        + "\n  ".join(offenders[:3])
        + "\nRe-record with scripts/record_session.py, which redacts the repo root."
    )
