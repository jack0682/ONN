---
title: "ONN Daily — 2026-03-25 — Atomic Promotion Certificate Closure"
date: 2026-03-25 09:00:00 +0900
last_modified_at: 2026-03-25 09:00:00 +0900
categories: [Research, ONN, Daily]
tags: [ONN, ORTSF, LOGOS, APAC, CIVeto, ArtifactCompleteness, DependenceRobustness]
toc: true
toc_sticky: true
excerpt: "Hardened APAC promotion admissibility with artifact completeness and dependence-robust CI veto integration, while keeping C184 non-promoted pending immutable replay evidence."
---

[ONN Daily Index](/daily/)

## 1. Context
Today converted promotion governance for `C184` into an atomic, executable certificate by binding predicate checks, artifact completeness, and dependence-aware CI veto into one admissibility predicate. The goal was to eliminate evidence-fragile promotion outcomes under partial writes and schema drift.

## 2. Today’s Theory Target
**Target:** Atomic Promotion Certificate Closure with Dependence-Robust CI Veto for `C184`.

Why high leverage:
- Directly addresses the open reviewer objection about artifact completeness.
- Makes CI-trust and class coverage explicit promotion prerequisites.
- Separates claim-update admissibility from stronger closed-loop certification claims.

## 3. What Changed in the Theory
### Restatement (cleaned)
Define artifact completeness, classwise CI-veto, and the atomic promotion certificate predicate:

$$
M_{art} = 1\{schema\_version\in\mathcal V\}\cdot 1\{manifest\_hash\neq\varnothing\}\cdot 1\{decision\_reason\neq\varnothing\}.
$$

$$
M_{ci}=\prod_{k\in K_{act}}1\{I_k^U\le 0\}\cdot1\{ci\_trust\_flag(k)=1\}.
$$

$$
\Phi_{prom}^{APAC}=1\{M_{bind}=1\}\cdot M_{art}\cdot M_{ci}\cdot1\{class\_coverage\_pass=1\}\cdot1\{n_{forbidden}=0\}.
$$

### Proof Audit (gaps & required assumptions)
- `C211` APAC admissibility theorem: `{PROVED}` under conservative dependence-aware CI assumptions (`A98`, `A100`) and transaction atomicity (`A106`).
- `C212` strengthening theorem (`\Phi_{prom}^{APAC} \le \Phi_{prom}^{CI}`): `{PROVED}`.
- `C213` mutation-veto lemma: `{PLAUSIBLE}`, still depends on immutable storage semantics (`A112`).
- `C214` scope theorem (admissibility != closed-loop certificate): `{PROVED}`.
- `C215` C184 governance invariant (no promotion before APAC bundle replay): `{PROVED}`.

### Strengthening (new lemma / tighter condition / fix)
APAC is a strict strengthening of prior CI-gated promotion and cannot increase unsafe promotions:

$$
\Phi_{prom}^{APAC}=\Phi_{prom}^{CI}\cdot M_{art}\cdot1\{class\_coverage\_pass=1\}\cdot1\{n_{forbidden}=0\}
\Rightarrow
\Phi_{prom}^{APAC}\le \Phi_{prom}^{CI}.
$$

Practical fix set:
- Require strict non-empty `decision_reason` and allowlisted `schema_version`.
- Fail-fast veto on any missing APAC conjunct.
- Keep `C184` non-promoted until immutable replay snapshot evidence is attached.

## 4. Paper Patch Notes (actionable edits)
- `P-676`: Insert APAC admissibility theorem in `paper/sections/05_theory.typ`.
- `P-677`: Extend telemetry contract with APAC fields in `paper/sections/03_method.typ`.
- `P-678`: Add APAC replay-cube acceptance protocol in `paper/sections/06_experiments.typ`.
- `P-679`: Clarify screening vs certification boundary in `paper/sections/07_related_work.typ`.
- `P-680`: Add accepted 2025-2026 references in `paper/bib/refs.bib`.

## 5. New Literature Integrated (≥3)
- Conformal time-series benchmarking and dependence caveats for CI-veto diagnostics.
- Delay-robust primal-dual dynamics as comparator framing for delayed evidence updates.
- Monotone inclusion/primal-dual splitting assumptions used to scope proof-route alternatives.
- Non-stationary coverage backtesting used to define acceptance checks for CI trust.

## 6. Development Actions (next 72 hours)
- Implement `test_apac_requires_all_fields` and `test_schema_version_allowlist`.
- Implement `test_no_promotion_on_ci_untrusted` with classwise replay matrix fixtures.
- Add immutable snapshot hash lock checks (`test_tx_snapshot_immutability`).
- Generate class-complete APAC replay bundle for `C184` adjudication.

## 7. Open Problems (carried + new)
- Carried: `C213` still needs storage-level immutability proof evidence.
- Carried: abrupt dependence-shift robustness for CI radius conservativeness (`A107`).
- Carried: matched-budget comparator reproducibility under lag bursts.
- New: formal theorem-to-schema binding proof for all APAC mandatory fields.
- New: replay uniqueness constraints for `(tx_id, class_id, snapshot_hash)` under async writes.

## 8. Next-day Seed
Execute APAC replay-cube evidence generation with immutable snapshot enforcement and close the `C213` mutation-veto lemma from `{PLAUSIBLE}` to `{PROVED}`.

## 9. References (reference-style links only)
1. [Sabashvili, 2026, Conformal Prediction Algorithms for Time Series Forecasting (arXiv:2601.18509)](https://arxiv.org/abs/2601.18509)
2. [Shang, 2026, Conformal prediction for high-dimensional functional time series (arXiv:2603.10674)](https://arxiv.org/abs/2603.10674)
3. [Sen et al., 2026, Delay-Robust Primal-Dual Dynamics for Distributed Optimization (arXiv:2603.18236)](https://arxiv.org/abs/2603.18236)
4. [Dao et al., 2025, Primal-dual splitting for structured composite monotone inclusions (arXiv:2512.10366)](https://arxiv.org/abs/2512.10366)
5. [Retzlaff et al., 2025, Testing Marginal and Conditional Coverage in CP for Non-Stationary Time Series (PMLR 266)](https://proceedings.mlr.press/v266/retzlaff25a.html)
