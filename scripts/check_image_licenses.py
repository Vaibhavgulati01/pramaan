"""CI gate: no NC-licensed image derivative may be committed.

The corpus is built from Amazon Berkeley Objects (**CC BY-NC 4.0**) and a
GenImage-derived set (**CC BY-NC-SA 4.0**). This repository is public and
Apache-2.0. Committing an image derived from either — an ELA heatmap, an
evidence-pack thumbnail, a GIF frame, a reliability plot rendered over
real claim images — would be redistributing NC-licensed content under
incompatible terms.

`data/` is gitignored, which handles the bulk. This gate catches the
subtler case: a *derived* visual that escaped into `reports/`, `docs/`,
or `demo/` because it looked like a plot rather than like a photograph.

The rule is an allowlist, not a blocklist. Anything not explicitly
permitted fails, because the failure mode is silent and the cost of a
false alarm is a one-line allowlist entry.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"}

# Paths whose images are self-authored or synthetic and therefore safe.
# Each entry needs a reason: an allowlist nobody can justify is a
# blocklist with extra steps.
ALLOWED_PREFIXES: dict[str, str] = {
    "reports/dev/reliability_": "matplotlib plot of aggregate statistics; no image content",
    "reports/full/reliability_": "matplotlib plot of aggregate statistics; no image content",
    "reports/smoke/reliability_": "matplotlib plot of aggregate statistics; no image content",
    "docs/assets/architecture": "self-authored diagram",
    "demo/assets/": "self-authored or procedurally generated placeholder imagery",
}

# Directories where derived imagery is most likely to appear by accident.
SCANNED_DIRS = ("reports", "docs", "demo", "notebooks", "benchmarks", "src", "tests")


def tracked_files() -> list[Path]:
    """Files git actually tracks. Scanning the working tree instead would
    flag local build artifacts that were never going to be committed."""
    try:
        output = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"could not list tracked files ({exc}); falling back to a tree walk")
        return [p for d in SCANNED_DIRS for p in Path(d).rglob("*") if p.is_file()]
    return [Path(line) for line in output.splitlines() if line]


def is_allowed(path: Path) -> str | None:
    posix = path.as_posix()
    for prefix, reason in ALLOWED_PREFIXES.items():
        if posix.startswith(prefix):
            return reason
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    violations: list[Path] = []
    allowed: list[tuple[Path, str]] = []

    for path in tracked_files():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        reason = is_allowed(path)
        if reason is None:
            violations.append(path)
        else:
            allowed.append((path, reason))

    if args.verbose or allowed:
        for path, reason in allowed:
            print(f"OK  {path}  ({reason})")

    if violations:
        print("\nIMAGE LICENCE CHECK FAILED", file=sys.stderr)
        print(
            "The following image files are tracked but not on the allowlist:",
            file=sys.stderr,
        )
        for path in violations:
            print(f"  {path}", file=sys.stderr)
        print(
            "\nABO is CC BY-NC 4.0 and the GenImage-derived set is CC BY-NC-SA 4.0. "
            "This repo is public and Apache-2.0, so no image derived from either may "
            "be committed - including plots rendered over real claim images. If this "
            "file is genuinely self-authored or synthetic, add it to ALLOWED_PREFIXES "
            "in this script with a reason. See docs/DATA_CARD.md.",
            file=sys.stderr,
        )
        return 1

    print(f"image licence check OK ({len(allowed)} allowlisted, 0 violations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
