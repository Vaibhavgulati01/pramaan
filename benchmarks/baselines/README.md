# benchmarks/baselines/

Outward-facing comparisons (Phase 6), not just ablations of PRAMAAN
itself: approve-all/deny-all, a rules engine (claim-count + value
threshold — what merchants actually run today), a CLIP linear probe, a
"ResNet-50 features + linear head, in the spirit of CNNSpot" detector
(named that way deliberately — it is not a faithful CNNSpot
reimplementation), and a behaviour-only LightGBM model.
