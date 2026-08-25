"""One-off analysis driver for Phase 0.5: runs the power curve and the
alpha/delta ladder resolution against configs/risk.yaml + configs/data.yaml,
and prints a report used to hand-fill both configs and docs/GUARANTEE.md.

Not part of the `pramaan` CLI (this is a planning tool run once per
sizing decision, not a pipeline step) - run directly with:
    python scripts/run_power_analysis.py
"""

from __future__ import annotations

import yaml

from pramaan.risk.power_analysis import power_curve, resolve_primary_target

REPO_ROOT = __file__.replace("\\", "/").rsplit("/scripts/", 1)[0]


def main() -> None:
    with open(f"{REPO_ROOT}/configs/risk.yaml") as f:
        risk_cfg = yaml.safe_load(f)
    with open(f"{REPO_ROOT}/configs/data.yaml") as f:
        data_cfg = yaml.safe_load(f)

    ladder = [(rung["alpha"], rung["delta"]) for rung in risk_cfg["alpha_delta_ladder"]]
    max_practical_full_n = risk_cfg["max_practical_full_n"]
    min_denied = risk_cfg["min_denied_for_certification"]
    fraud_prevalence = data_cfg["fraud_prevalence"]
    assumed_deny_rate = fraud_prevalence * 0.5  # DENY = confident half of the fraud tail
    test_split_fraction = 0.20

    print("=" * 78)
    print("POWER CURVE  (min certifiable denial-set n vs alpha, delta=0.10)")
    print("=" * 78)
    curve = power_curve(
        alphas=[0.03, 0.05, 0.10, 0.15, 0.20],
        delta=0.10,
        r_hat_fractions_of_alpha=(0.0, 0.3, 0.5),
        min_denied=min_denied,
    )
    print(f"{'alpha':>6} {'r_hat/alpha':>12} {'r_hat':>8} {'min n':>10}")
    for pt in curve:
        n_str = str(pt.min_denial_set_n) if pt.min_denial_set_n is not None else "unreachable"
        print(
            f"{pt.alpha:>6.2f} {pt.r_hat_as_fraction_of_alpha:>12.1f} "
            f"{pt.r_hat_assumed:>8.4f} {n_str:>10}"
        )

    print()
    print("=" * 78)
    print(
        f"LADDER RESOLUTION  (fraud_prevalence={fraud_prevalence}, "
        f"assumed_deny_rate={assumed_deny_rate}, test_split_fraction={test_split_fraction}, "
        f"max_practical_full_n={max_practical_full_n})"
    )
    print("=" * 78)
    for label, frac in (("optimistic (model comfortably clears bar)", 0.3),
                         ("conservative (model just barely clears bar)", 0.5)):
        print(f"\n--- {label}: r_hat = {frac} * alpha ---")
        chosen, attempts = resolve_primary_target(
            ladder,
            max_practical_full_n=max_practical_full_n,
            r_hat_as_fraction_of_alpha=frac,
            assumed_deny_rate=assumed_deny_rate,
            assumed_test_split_fraction=test_split_fraction,
            min_denied=min_denied,
        )
        for est in attempts:
            practical = (
                "PRACTICAL"
                if est.required_full_corpus_n is not None
                and est.required_full_corpus_n <= max_practical_full_n
                else "impractical/unreachable"
            )
            marker = " <== CHOSEN" if est is chosen else ""
            print(
                f"alpha={est.alpha:.2f} delta={est.delta:.2f}  "
                f"min_denial_n={est.min_denial_set_n}  "
                f"required_test_n={est.required_test_n}  "
                f"required_full_corpus_n={est.required_full_corpus_n}  "
                f"[{practical}]{marker}"
            )
        if frac == 0.5:
            conservative_chosen = chosen

    print()
    print("SIZING DECISION: use the CONSERVATIVE scenario (r_hat = 0.5*alpha) to size the")
    print("corpus, since undershooting is the failure mode power analysis exists to avoid.")
    print(f"PRIMARY TARGET: alpha={conservative_chosen.alpha}, delta={conservative_chosen.delta}")
    print(f"REQUIRED FULL-SCALE CORPUS SIZE: {conservative_chosen.required_full_corpus_n} claims")
    chosen = conservative_chosen

    print()
    print("=" * 78)
    print("DEV-SCALE ILLUSTRATIVE SIZING (relaxed alpha/delta, mechanism-proof only)")
    print("=" * 78)
    dev_alpha = risk_cfg["illustrative_dev"]["alpha"]
    dev_delta = risk_cfg["illustrative_dev"]["delta"]
    dev_estimate = None
    for frac in (0.3, 0.5):
        est = resolve_primary_target(
            [(dev_alpha, dev_delta)],
            max_practical_full_n=10_000,
            r_hat_as_fraction_of_alpha=frac,
            assumed_deny_rate=assumed_deny_rate,
            assumed_test_split_fraction=test_split_fraction,
            min_denied=min_denied,
        )[0]
        print(
            f"dev alpha={dev_alpha} delta={dev_delta} r_hat_frac={frac}: "
            f"min_denial_n={est.min_denial_set_n} required_full(dev)_n={est.required_full_corpus_n}"
        )
        if frac == 0.3:
            dev_estimate = est

    assert dev_estimate is not None
    print()
    print(f"CHOSEN DEV CORPUS SIZE (rounded up a bit for headroom): "
          f"{dev_estimate.required_full_corpus_n}")


if __name__ == "__main__":
    main()
