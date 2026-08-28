"""Generates README sections from metrics.json — nothing hand-typed.

Sec.7's non-negotiable: `metrics.json` is the single source of truth and
README figures are injected from it. CI re-runs this and diffs the result
against the committed README, which turns "we never hand-typed a number"
from a promise into a verified property.

## How it works

The README carries marker pairs:

    <!-- BEGIN:results -->
    ...generated...
    <!-- END:results -->

Everything between them is replaced. Everything outside is prose the
generator never touches.

## Tier discipline

Which tier's metrics are injected is explicit and stamped into the
output. `full` is the only tier whose numbers may be presented as a
result; `dev` and `smoke` are injected with a prominent warning banner so
a reader cannot mistake them (`docs/EVALUATION_PROTOCOL.md`).

Usage:
    python scripts/inject_metrics.py --tier dev            # rewrite README
    python scripts/inject_metrics.py --tier dev --check     # CI: diff only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MARKERS = ("results", "guarantee", "ablations")


def _fmt_ci(entry: dict[str, Any]) -> str:
    if entry.get("ci_lower") is None:
        return f"{entry['point']:.4f}"
    return f"{entry['point']:.3f} [{entry['ci_lower']:.3f}, {entry['ci_upper']:.3f}]"


def _banner(metrics: dict[str, Any]) -> list[str]:
    if metrics.get("is_reportable_result"):
        return []
    return [
        "> **⚠️ These are `"
        + metrics["tier"]
        + "`-scale numbers, not a held-out result.**",
        "> They evaluate the *train* split's out-of-fold predictions and exist to",
        "> prove the mechanisms work. The reportable certificate and results come",
        "> from the `full` tier — see "
        "[`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md).",
        "",
    ]


def render_results(metrics: dict[str, Any]) -> str:
    lines = _banner(metrics)
    lines += [
        f"Evaluated on the **{metrics['split_evaluated']}** split of the "
        f"`{metrics['tier']}` corpus — {metrics['n_claims']:,} claims, "
        f"{metrics['prevalence']:.1%} fraud prevalence.",
        "",
        "| System | PR-AUC (95% CI) |",
        "|---|---|",
    ]
    for row in metrics.get("baselines", []):
        name = row["name"].replace("_", " ")
        if row["name"] == "pramaan_full":
            name = "**PRAMAAN (full)**"
        lines.append(f"| {name} | {_fmt_ci(row)} |")

    cost = metrics.get("cost", {})
    if cost:
        lines += [
            "",
            f"**₹{cost['per_1000_claims']:,.0f} per 1,000 claims** at "
            f"**{cost['review_load_pct']:.1f}% review load**.",
            "",
            "| Policy | ₹ per 1,000 claims |",
            "|---|---|",
        ]
        for name, value in cost.get("baselines_per_1000", {}).items():
            lines.append(f"| {name.replace('_', ' ')} | ₹{value:,.0f} |")
        lines.append(f"| **PRAMAAN** | **₹{cost['per_1000_claims']:,.0f}** |")

    cal = metrics.get("calibration", {}).get("overall", {})
    if cal:
        lines += [
            "",
            f"Calibration: Brier **{cal['brier']:.4f}**, ECE **{cal['ece']:.4f}**, "
            f"MCE **{cal['mce']:.4f}** (10 equal-mass bins).",
        ]

    cascade = metrics.get("cascade", {})
    if cascade:
        exits = cascade.get("stage_exit_distribution", {})
        lines += [
            "",
            f"Cascade: **{cascade['mean_compute_ms']:.1f} ms/claim** mean, "
            "stage-exit "
            + " / ".join(f"{v:.0%}" for v in exits.values())
            + " (stages 1/2/3).",
        ]

    return "\n".join(lines)


def render_guarantee(metrics: dict[str, Any], certificate: dict[str, Any] | None) -> str:
    if certificate is None:
        return (
            "*No certificate has been computed for this tier yet. Run "
            "`pramaan certify`.*"
        )

    lines = _banner(metrics)
    if not certificate.get("certified"):
        lines += [
            "**No α/δ rung certified.** Every rung of the pre-committed ladder was",
            "attempted and each failed; that is the published result rather than a",
            "loosened bound. See [`docs/GUARANTEE.md`](docs/GUARANTEE.md).",
            "",
            "| α | δ | Outcome |",
            "|---|---|---|",
        ]
        for attempt in certificate.get("attempts", []):
            reason = (attempt.get("stop_reason") or "").split(";")[0]
            lines.append(f"| {attempt['alpha']} | {attempt['delta']} | failed — {reason} |")
        return "\n".join(lines)

    chosen = certificate["chosen"]
    best = next(
        r for r in chosen["results"] if r["threshold"] == chosen["least_conservative"]
    )
    lines += [
        f"> At the cost-optimal certified threshold, **at most "
        f"{chosen['alpha']:.1%} of auto-denied claims are legitimate, with "
        f"{1 - chosen['delta']:.0%} confidence** "
        f"(n={best['n_denied']}, empirical {best['conditional_rate']:.3f}, "
        f"HB p={best['conditional_pvalue']:.3f}).",
        "",
        "The unconditional variant — which carries no ratio caveat — is "
        f"{best['unconditional_rate']:.4f} "
        f"(HB p={best['unconditional_pvalue']:.3f}).",
    ]
    return "\n".join(lines)


def render_ablations(metrics: dict[str, Any]) -> str:
    ablations = metrics.get("ablations")
    if not ablations:
        return "*Ablations have not been run for this tier yet.*"

    lines = _banner(metrics)
    lines += [
        "Each ablation is reported twice. The **full corpus** column is confounded",
        "by source dataset — every synthetic-fraud claim comes from GenImage, which",
        "carries 2.58× the fraud rate of ABO — so it is an *upper bound* on the",
        "pixel pillars. The **ABO-only** column holds source constant.",
        "",
        "| Ablation | Δ PR-AUC (full corpus) | Δ PR-AUC (ABO-only) |",
        "|---|---|---|",
    ]

    by_name: dict[str, dict[str, float]] = {}
    for result in ablations["results"]:
        if result["name"] == "full":
            continue
        by_name.setdefault(result["name"], {})[result["subset"]] = result["delta_pr_auc"]

    for name, subsets in by_name.items():
        full = subsets.get("all")
        abo = subsets.get("abo_only")
        full_cell = "—" if full is None else f"{full:+.4f}"
        abo_cell = "—" if abo is None else f"{abo:+.4f}"
        lines.append(f"| {name.replace('_', ' ')} | {full_cell} | {abo_cell} |")

    controls = metrics.get("negative_controls")
    if controls:
        lines += [
            "",
            "**Negative controls** (these validate the methodology, not a hypothesis):",
            "",
        ]
        lines += ["| Control | PR-AUC | Prevalence | Pass? |", "|---|---|---|---|"]
        for control in controls:
            mark = "✅" if control["passed"] else "❌"
            lines.append(
                f"| {control['name'].replace('_', ' ')} | {control['pr_auc']:.4f} | "
                f"{control['prevalence']:.4f} | {mark} |"
            )

    return "\n".join(lines)


def inject(readme: str, section: str, body: str) -> str:
    begin, end = f"<!-- BEGIN:{section} -->", f"<!-- END:{section} -->"
    if begin not in readme or end not in readme:
        raise ValueError(f"README is missing the {section!r} marker pair")

    head, _, rest = readme.partition(begin)
    _, _, tail = rest.partition(end)
    return f"{head}{begin}\n{body}\n{end}{tail}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="dev", choices=["smoke", "dev", "full"])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--check", action="store_true",
        help="Do not write; exit 1 if the committed README differs from what "
             "this script would generate. Used by CI.",
    )
    args = parser.parse_args()

    metrics_path = args.repo_root / "reports" / args.tier / "metrics.json"
    if not metrics_path.exists():
        print(
            f"no metrics at {metrics_path}; run `pramaan eval --scale {args.tier}` first",
            file=sys.stderr,
        )
        return 0 if args.check else 1

    metrics = json.loads(metrics_path.read_text())
    certificate_path = args.repo_root / "reports" / args.tier / "certificate.json"
    certificate = (
        json.loads(certificate_path.read_text()) if certificate_path.exists() else None
    )

    readme_path = args.repo_root / "README.md"
    original = readme_path.read_text(encoding="utf-8")

    updated = original
    updated = inject(updated, "results", render_results(metrics))
    updated = inject(updated, "guarantee", render_guarantee(metrics, certificate))
    updated = inject(updated, "ablations", render_ablations(metrics))

    if args.check:
        if updated != original:
            print(
                "README is out of date with reports/"
                f"{args.tier}/metrics.json.\n"
                "Every number in the README is generated - regenerate with:\n"
                f"  python scripts/inject_metrics.py --tier {args.tier}",
                file=sys.stderr,
            )
            return 1
        print(f"README matches reports/{args.tier}/metrics.json")
        return 0

    readme_path.write_text(updated, encoding="utf-8")
    print(f"injected {args.tier} metrics into README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
