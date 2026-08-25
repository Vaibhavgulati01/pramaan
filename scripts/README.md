# scripts/

Repo-maintenance / one-off analysis scripts, as distinct from `eval/`
(evaluation logic run per-tier by the CLI) and `src/pramaan/` (the
library). Land here as their phase requires them:

- `run_power_analysis.py` (Phase 0.5, done) — drives
  `src/pramaan/risk/power_analysis.py` against `configs/risk.yaml` /
  `configs/data.yaml` to size the `dev` and `full` corpora from the
  certification power requirement; see `docs/GUARANTEE.md`. Not a
  pipeline step — a planning tool, run again only if the sizing
  assumptions change.
- `inject_metrics.py` (Phase 6) — writes README's generated sections from
  `reports/{tier}/metrics.json`; CI diffs its output against the
  committed README.
- `check_image_licenses.py` (Phase 6) — CI gate: fails if any committed
  image file isn't on the self-authored/permissive allowlist (see
  `docs/DATA_CARD.md` for why — ABO is CC BY-NC 4.0 and this is a public
  repo).
