# Brain-Inspired Computation for ONN-ES

This document captures (1) an annotated bibliography of mechanistic brain computation papers translatable to ONN residual graph dynamics, (2) a unified blueprint, (3) an ONN mapping table, and (4) a 24-hour minimal validation plan.

## 1) Annotated Bibliography (anchor + bridge, 20 papers)

Each entry: citation — bucket(s) — claim — mechanism — update rule — empirical signature — ONN hooks.

1. **Rao & Ballard 1999, Nature Neurosci, “Predictive coding in the visual cortex.”** (Predictive coding) — Cortex sends predictions top-down; layer 4 error neurons send mismatch. Mechanism: feedforward errors, feedback predictions. Update: \(e = x - \hat{x}; \Delta w \propto e \, x_{context}\). Empirical: extra-classical RF suppression. ONN: edge error units; top-down priors; maintain error channels per edge.
2. **Friston 2010, Nature Reviews Neurosci, “Free-energy principle.”** (Predictive coding, energy-based) — Variational free-energy minimization; errors drive states. Update: \(\dot{x} = -\partial F/\partial x\). Empirical: mismatch negativity. ONN: treat total loss as energy; residual flow with projection.
3. **Bastos et al. 2012, Neuron, “Canonical microcircuits for predictive coding.”** (Predictive coding laminar) — Layer-specific: superficial = error FF, deep = prediction FB. Update: laminar gating by frequency bands. Empirical: gamma FF, beta/alpha FB. ONN: split edge tensors into error/prediction lanes; use gating to separate FF/FB.
4. **Keller & Mrsic-Flogel 2018, Nature Neurosci, “Predictive processing: neural circuits.”** (Predictive coding review) — Converges to two-pop scheme (pred vs err). Empirical: sensorimotor prediction errors. ONN: enforce two-field representation per edge: state + error.
5. **Denève & Machens 2016, Current Opinion Neurobiol, “Efficient codes and balanced networks.”** (E/I balance) — Balanced networks implement predictive coding via cancellation. Update: spikes/errors compensate prediction. Empirical: asynchronous irregular firing. ONN: enforce E/I-like normalization on residual updates; clip/normalize residual norm.
6. **Shenoy, Sahani & Churchland 2013, Annual Review Neurosci, “Cortical control of movement: dynamical systems view.”** (Population dynamics) — Low-d manifolds generate rotations. Update: \(\dot{x} = Ax + Bu\). Empirical: jPCA rotational modes in M1. ONN: constrain node manifold to low-rank dynamics; use linear recurrent core plus projection.
7. **Saxe et al. 2021, Neuron, “A mathematical theory of neural dynamics.”** (Population dynamics) — Training drives low-d dynamics via gradient flow. Empirical: low intrinsic dimensionality. ONN: use low-rank factorization for node embeddings.
8. **Gallego et al. 2017, Neuron, “Neural manifolds for movement control.”** (Population manifolds) — Stable manifolds reused across tasks. ONN: keep node states on learned manifold via projection.
9. **Poirazi & Mel 2001, Neuron, “Impact of active dendrites and structural plasticity on neocortical pyramidal neurons.”** (Dendritic computation) — Neuron ≈ 2-layer network (subunit nonlinearity). ONN: allow edge updates to be local MLPs with gating per branch.
10. **London & Häusser 2005, Annu Rev Neurosci, “Dendritic computation.”** (Dendrites) — Back-propagating spikes gate plasticity. ONN: local Hebbian with modulatory gate for edges.
11. **Urbanczik & Senn 2014, Neuron, “Learning by the dendritic prediction of somatic spiking.”** (Local learning) — Dendrite predicts soma; error = spike - prediction. Update: \(\Delta w \propto (soma - pred)\,x_{dend}\). ONN: edge-local predictive update without global BP.
12. **Whittington & Bogacz 2017, Neural Computation, “An approximation of backpropagation in predictive coding networks.”** (Bioplausible learning) — Predictive coding inference + Hebbian = BP-like gradients. ONN: use residual inference loop; learn weights with local Hebbian on errors.
13. **Lillicrap et al. 2016, Nature Comm, “Random feedback alignment.”** (Approx gradient) — Fixed random feedback suffices. ONN: use fixed transpose-free feedback for edge parameter tuning.
14. **Scellier & Bengio 2017, Front Comput Neurosci, “Equilibrium propagation.”** (Energy-based learning) — Two-phase nudged equilibrium; update \(\Delta w \propto (x^{\beta}-x^{0}) \cdot \partial E/\partial w\). ONN: run two residual phases (free/nudged) on graph.
15. **O’Reilly 1996, Neural Computation, “Biologically plausible error-driven learning with local activation differences.”** (CHL) — Contrastive Hebbian; minus/plus phases. ONN: plus/minus projections on constraints.
16. **van Vreeswijk & Sompolinsky 1996, Science, “Chaos in balanced networks.”** (E/I balance) — Asynchronous irregular state stabilizes activity. ONN: enforce residual scaling to keep variance bounded.
17. **Tsodyks & Sejnowski 1995, Network, “Rapid state switching in balanced cortical network.”** (Stability) — Short-term depression stabilizes. ONN: include adaptive step-size or residual leak.
18. **Sherman & Guillery 2011, Phil Trans R Soc B, “Distinct functions of the transthalamic and lemniscal pathways.”** (Thalamic gating) — Thalamus gates cortical communications. ONN: thalamic gate analog = edge gating variable modulating residual flow.
19. **Maass, Natschläger & Markram 2002, Neural Comput, “Real-time computing without stable states.”** (Reservoir/LSM) — Liquid state + linear readout. ONN: treat graph residual core as reservoir; readout = ReasoningTrace.
20. **Bellec et al. 2020, Nature Comm, “Solution to the learning dilemma for recurrent networks of spiking neurons.”** (e-prop) — Local eligibility traces + global modulators approximate BPTT. ONN: use eligibility traces on edges with global scalar reward.

## 2) Unified “Brain Computation Blueprint” (text spec)

- **Variables**: graph state x = {h_i, r_ij, g_ij, u_i}; errors e_obs (sensor mismatch), e_cons (constraint violation); gates g (thalamic/E-I inspired); uncertainty u; timescales τ_fast (inference), τ_slow (learning/ES), τ_ctrl (ORTSF).
- **Update Core**: x ← P_C(x − η · (β1·e_obs + β2·e_cons) ⊙ gain(u,g)), with P_C = projection to constraints (simplex for relations, [0,1] for gates, u≥0, radius>0), gain = 1/(u+ε) · g.
- **Loops**:
  - Feedforward: error units propagate e_obs upward.
  - Feedback: predictions downward set priors; gates modulate.
  - Recurrent manifold: low-rank dynamics keep h_i on manifold; residuals perturb then project back.
- **Stability**: E/I balance ⇒ normalize residual norms; leak/decay on h_i, r_ij; clip gates; projection each step.
- **Learning**:
  - Fast: local Hebbian/CHL on edges using errors (e · pre · post).
  - Slow: ES tunes meta-params (β1, β2, η, gate thresholds, projection strength).
  - Optional: equilibrium/contrastive phases (free vs nudged) to approximate gradients.
- **Thalamic/Reservoir role**: gating variables act as controllable valves; residual core can be run as reservoir for temporal richness; readouts trained separately.

## 3) ONN Mapping Table

| Brain primitive | Role | ONN counterpart | Minimal formula/pseudocode | Risk & mitigation |
| --- | --- | --- | --- | --- |
| Error units (layer 2/3/4) | Carry mismatch | e_obs per edge | e = x_pred - x_obs; r -= η·e | Vanishing/ exploding: normalize e, clip |
| Predictive feedback (layer 5/6) | Provide priors | top-down priors on edges/nodes | x_pred = f(top_down); e = x - x_pred | Wrong priors: gate feedback, anneal |
| E/I balance | Stabilize activity | residual norm clamp | r = r / (1+λ‖r‖) | Under/over-damping: tune λ via ES |
| Thalamic gate | Route/permit comms | gate g_ij ∈[0,1] | g = σ(evidence - conflict); x += g·Δ | Gate stuck: add entropy bonus/decay |
| Neural manifolds | Low-d latent flow | low-rank h_i = AB^Tz | h = A(B^T z); project to span(A) | Loss of expressivity: adapt rank |
| Dendritic prediction | Local credit | edge-local MLP with error gate | Δw ∝ (pred_soma - act)·pre | Instability: cap weight change |
| Reservoir core | Temporal richness | run K residual steps | for k: x←f(Ax+Bu); readout R | Drift: periodic projection P_C |
| Eligibility traces | Credit assignment | per-edge trace | e_trace = γ e_trace + pre·post; Δw ∝ trace·mod | Trace blow-up: decay γ |
| Equilibrium/CHL | Approx BP | free vs nudged phases | Δw ∝ (x⁺-x⁻)·pre | Slow: limit phase count |

## 4) 24-Hour Minimal Experimental Plan (code-able)

Goal: validate two primitives in ONN codebase.

1) **Residual Projection Stability**
- Implement: x ← P_C(x − η(β1 e_obs + β2 e_cons)), with gate clamp and radius>0.
- Test: synthetic 5-node graph with contradictory edges; measure violation drop vs step count.
- Metrics: constraint violation rate, residual norm, gate saturation fraction.

2) **Thalamic Gate as Conflict Valve**
- Implement: g_ij = σ(a·evidence - b·conflict); apply to residual update.
- Test: inject conflicting observations; verify gates close conflicting edges and reopen when conflict resolves.
- Metrics: gate entropy, accepted-edge F1, convergence time.

Execution notes: run in venv; use existing tests/unit harness; add small synthetic fixtures under tests/unit for the two experiments.
