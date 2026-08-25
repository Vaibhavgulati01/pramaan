# scripts/

Repo-maintenance scripts, as distinct from `eval/` (evaluation logic) and
`src/pramaan/` (the library). Land here as their phase requires them:

- `inject_metrics.py` (Phase 6) — writes README's generated sections from
  `reports/{tier}/metrics.json`; CI diffs its output against the
  committed README.
- `check_image_licenses.py` (Phase 6) — CI gate: fails if any committed
  image file isn't on the self-authored/permissive allowlist (see
  `docs/DATA_CARD.md` for why — ABO is CC BY-NC 4.0 and this is a public
  repo).
