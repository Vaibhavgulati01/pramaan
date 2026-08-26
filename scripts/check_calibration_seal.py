"""CI gate: the committed calibration seal must be internally consistent
and must not have been silently replaced.

docs/GUARANTEE.md caveat 3 promises the calibration split is used once
and never changes. `src/pramaan/risk/calibration_seal.py` records the
hash; this script is the half that runs in CI.

It deliberately does NOT rebuild the corpus. CI has no `data/` (it is
gitignored, and the images are CC BY-NC), so recomputing the hash there
is impossible. What CI *can* check, and what actually catches the failure
mode worth catching, is that a seal exists, is well-formed, and has not
been quietly edited or regenerated between commits - the latter shows up
as a diff on a committed file, which is exactly the point of committing
it.

The full recomputation happens wherever the data lives: `pramaan certify`
verifies the seal before using the split, and refuses to reseal a changed
one.

Usage:
    python scripts/check_calibration_seal.py [--require dev,full]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = {"tier", "n_claims", "n_fraud", "content_sha256", "sealed_at"}


def check_seal(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path}: not valid JSON ({exc})"]

    missing = REQUIRED_FIELDS - set(data)
    if missing:
        problems.append(f"{path}: missing field(s) {sorted(missing)}")
        return problems

    sha = data["content_sha256"]
    if not isinstance(sha, str) or len(sha) != 64:
        problems.append(f"{path}: content_sha256 is not a sha256 hex digest")
    else:
        try:
            int(sha, 16)
        except ValueError:
            problems.append(f"{path}: content_sha256 is not hexadecimal")

    if data["tier"] != path.parent.name:
        problems.append(
            f"{path}: seal says tier={data['tier']!r} but lives under "
            f"{path.parent.name!r} - a seal copied between tiers certifies nothing"
        )

    if not isinstance(data["n_claims"], int) or data["n_claims"] <= 0:
        problems.append(f"{path}: n_claims must be a positive integer")
    elif not 0 <= data["n_fraud"] <= data["n_claims"]:
        problems.append(f"{path}: n_fraud ({data['n_fraud']}) outside [0, n_claims]")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root", type=Path, default=Path("reports"),
        help="Where per-tier seals live.",
    )
    parser.add_argument(
        "--require", default="",
        help="Comma-separated tiers that MUST have a seal (e.g. 'full').",
    )
    args = parser.parse_args()

    seals = sorted(args.reports_root.glob("*/calibration_seal.json"))
    problems: list[str] = []

    for path in seals:
        problems.extend(check_seal(path))
        if not problems:
            data = json.loads(path.read_text())
            print(
                f"OK  {path}  tier={data['tier']} n={data['n_claims']} "
                f"sha={data['content_sha256'][:16]}..."
            )

    required = {t.strip() for t in args.require.split(",") if t.strip()}
    present = {p.parent.name for p in seals}
    for tier in sorted(required - present):
        problems.append(
            f"tier {tier!r} has no calibration seal. Run `pramaan certify "
            f"--scale {tier}` before relying on a certificate for it."
        )

    if problems:
        print("\nCALIBRATION SEAL CHECK FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if not seals:
        print("no calibration seals recorded yet (expected before Phase 4 runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
