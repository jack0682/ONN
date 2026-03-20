---
title: "ONN Daily — 2026-03-20 — DARC Promotion Gate Operator Closure"
date: 2026-03-20 09:00:00 +0900
last_modified_at: 2026-03-20 09:00:00 +0900
categories: [Research, ONN, Daily]
tags: [ONN, ORTSF, LOGOS, DARC, OperatorTheory, PromotionGate, Feasibility]
toc: true
toc_sticky: true
excerpt: "Formalized DARC as an operator-screening map, locked empty-feasible-set no-activation behavior, and kept C184 non-promoted until class-complete replay evidence is satisfied."
---

[ONN Daily Index](/daily/)

## 1. Context
Today closes the unresolved promotion branch from 2026-03-19 by converting DARC from heuristic screening language into an explicit operator-theoretic map with promotion governance constraints. The key objective was to preserve conservative safety behavior while tightening what is provable now versus what remains conjectural.

## 2. Today’s Theory Target
**Target:** DARC Promotion Gate with Operator-Theoretic Closure.

High-leverage reason:
- Closes claim `C184` promotion decision path.
- Makes screening map semantics explicit enough for reviewer-grade objections.
- Separates proven safety guarantees from unresolved transfer claims.

## 3. What Changed in the Theory
### Restatement (cleaned)
Define classwise feasibility and screened selection:

$$
F_k = \{B \in \mathcal{B}_k : u_k^U(B) \le u_{tol,k},\ \rho_k(B) \ge \rho_{min},\ M_{async}(B)=1\}
$$

$$
T_k(x) = \Pi_{F_k}\left(\arg\min_{B\in \mathcal{B}_k} J_k(B;x)\right)
$$

$$
J_k(B) = w_u u_k^U(B)+w_i i_k(B)+w_f f_k(B),\quad w_u \gg w_i\ge 0,\ w_f\ge 0
$$

with conservative upper bound:

$$
u_k^U(B)=u_k(B)+r_k(\delta).$$

### Proof Audit (gaps & required assumptions)
- `C193` (radius monotonicity in `n_eff_lb`): `{PROVED}` under conservative effective-count assumptions (`A98`, `A100`).
- `C195` (no activation when `F_k=\varnothing`): `{PROVED}` with atomic snapshot/tx constraints (`A96`).
- `C196` (deterministic tie-break uniqueness): `{PROVED}` under stable snapshot comparator (`A103`).
- `C197` (safety cap preserved): `{PROVED}` if async veto is enforced before selector.
- `C194` (projection nonexpansiveness in discrete setting): `{PLAUSIBLE}` only; sparse occupancy toggling still introduces discontinuity.
- `C198` (out-of-support transfer conservativeness): `{CONJECTURE}`; not promotable without matched replay evidence.

Derivative check used for `C193`:

$$
\frac{\partial r_k}{\partial n_{eff,lb}} = -\frac{1}{2}\sqrt{\frac{\log(1/\delta)}{2}}\cdot\max(1,n_{eff,lb})^{-3/2} \le 0
$$

### Strengthening (new lemma / tighter condition / fix)
- Enforced hard no-promotion criterion for `C184`: no promotion if class-complete replay cube evidence is incomplete.
- Added empty-feasible-set contract: if `F_k=\varnothing`, then `N_{act}=1` must hold; any activation is invalid.
- Added deterministic tie-break rule keyed by minimum conservative unsafe-accept (`u_k^U`) on an atomic snapshot.

## 4. Paper Patch Notes (actionable edits)
- `P-661`: method section update for DARC operator map and telemetry/no-activation contract.
- `P-662`: theory section proposition for operator screening and explicit C184 no-promotion gate.
- `P-663`: experiment section promotion matrix requiring class-complete replay pass.
- `P-664`: related-work refinement around monotone splitting and IQC non-equivalence boundary.
- `P-665`: bibliography append with operator/IQC/delay-robust references used in this closure.

## 5. New Literature Integrated (≥3)
- [Combettes 2024 — Geometry of Monotone Operator Splitting](https://doi.org/10.1017/S0962492923000065)
- [Scherer & Ebenbauer 2025 — Convex Design via IQC (Tutorial)](https://doi.org/10.1146/annurev-control-030624-012200)
- [Lessard, Recht, Packard 2016 — IQC for Optimization Algorithms](https://doi.org/10.1137/15M1009597)
- [Fazlyab et al. 2018 — IQC for Nonstrongly Convex Problems](https://doi.org/10.1137/17M1136845)
- [Firouzbahrami & Nobakhti 2025 — Delay-Robust Distributed Optimization](https://doi.org/10.1109/TAC.2025.3533424)

## 6. Development Actions (next 72 hours)
- Execute class-complete replay cube over `(k,d_sup,n_eff_lb,tau_mis,lag_skew,delta)` with feasibility occupancy reporting.
- Run atomicity stress test for stale-cache activation bypass and verify `tx_id` lock behavior.
- Promote `C194` only if feasible-set membership toggling is bounded by explicit continuity lemma conditions.
- Keep `C198` scoped as conjecture unless matched-budget replay shows no unsafe-accept regression.

## 7. Open Problems (carried + new)
- Carried: transfer conservativeness under out-of-support topology-delay regimes (`C198`).
- Carried: sparse-class feasible-set toggle continuity conditions for `\Pi_{F_k}`.
- New: derive measurable sufficient conditions under which discrete feasible projection is effectively nonexpansive.
- New: establish replay-cube coverage threshold robust to parser-loss bursts without promotion bias.

## 8. Next-day Seed
Construct a minimal theorem candidate for piecewise-nonexpansive DARC screening under bounded feasible-set switching frequency, then validate it against the full replay cube and no-activation invariants.

## 9. References (reference-style links only)
1. [Combettes, 2024, Acta Numerica](https://doi.org/10.1017/S0962492923000065)
2. [Scherer & Ebenbauer, 2025, Annual Review of Control, Robotics, and Autonomous Systems](https://doi.org/10.1146/annurev-control-030624-012200)
3. [Lessard et al., 2016, SIAM Journal on Optimization](https://doi.org/10.1137/15M1009597)
4. [Fazlyab et al., 2018, SIAM Journal on Optimization](https://doi.org/10.1137/17M1136845)
5. [Firouzbahrami & Nobakhti, 2025, IEEE TAC](https://doi.org/10.1109/TAC.2025.3533424)
6. [Borwein, Li, Tam, 2017, SIAM Journal on Optimization](https://doi.org/10.1137/15M1045223)
