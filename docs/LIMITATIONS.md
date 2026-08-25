# Limitations

> **Status: skeleton.** This is deliberately the last document to be
> considered "done," even though it starts here in Phase 0 — every phase
> should add to it as real limitations are discovered, and Phase 9 makes
> a final pass. Per the spec: under repo-only judging this is arguably the
> highest-ROI file in the repository. It should be long and specific, and
> unflattering where warranted — a reviewer who reads a thorough
> limitations section stops looking for the flaws you hid, because you've
> demonstrated you already found them.

## Known limitations to document as each phase lands

- **Scale**: `DEV`-scale numbers anywhere in this repo are
  mechanism-validation only, not statistically meaningful — see the
  three-tier discipline in `docs/EVALUATION_PROTOCOL.md`. If `full` never
  ran, say so here plainly, with what that means for every number in the
  README.
- **Simulated behavioural data**: Pillar 4 and the claim ledger are
  simulated (no public dataset of refund claims with claimant history
  exists). Sensitivity sweep results and their implications go here.
- **Exchangeability**: the certificate's core assumption breaks under
  generator shift by construction (the attacker's whole strategy). The
  shift-matrix results — including the real ❌s — get discussed here in
  prose, not just tabulated.
- **The selective-risk ratio caveat** (§4 L3 honesty note #1) and what it
  does and doesn't let us claim.
- **C2PA / provenance**: near-universal absence in the India context
  (WhatsApp strips it); what that means for how much this pillar can
  contribute.
- **Federation**: if the federation kill-gate triggers (see the
  implementation plan), document exactly why the measured version didn't
  land and what would be needed to make it real.
- **Device canonicalisation instability** (§4 L0).
- Anything else found while building — this list is expected to grow.
