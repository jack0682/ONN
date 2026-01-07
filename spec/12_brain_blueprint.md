# ONN-ES Brain Computation Blueprint

## Scope
- Deliverable: Annotated bibliography (brain computation → implementable algorithms), unified blueprint for relation-graph state dynamics, ONN mapping table, and 24h minimal experiment plan.
- Target model: Residual-driven graph state dynamics `x_{t+1} = P_C(x_t + Δ(x_t, e, g, u))` with projection/stabilization.

---
## Annotated Bibliography (anchor + bridge, 24 papers)

### Predictive coding / cortical microcircuits
1. **Rao & Ballard (1999, Nat Neurosci)** — *Predictive coding in the visual cortex* (Bucket: predictive coding). Error units send bottom-up residuals; prediction units send top-down estimates. Update: `e = y - ŷ`; weights via local Hebbian with error modulation. Empirical: extra-classical receptive fields; mismatch negativity. **ONN hook:** separate error/prediction tensors per edge; top-down priors as constraint targets.
2. **Friston (2005, Phil Trans B)** — *A theory of cortical responses* (predictive processing). Energy function (free energy) minimized by gradient descent on prediction errors; hierarchical bidirectional message passing. **Hook:** treat `L_total` as free-energy; run inference as gradient flow with projection.
3. **Bastos et al. (2012, Neuron)** — *Canonical microcircuits for predictive coding*. Laminar mapping: feedforward = error (supragranular), feedback = prediction (infragranular); frequency bands (gamma up, beta/alpha down). **Hook:** edge-level routing of residuals vs priors; use frequency-tagged gates (rate proxies).
4. **Keller & Mrsic-Flogel (2018, Neuron, review)** — *Predictive processing: cortical computations*. Summarizes error/prediction separation, mismatch signatures. **Hook:** implement mismatch detectors per edge; drive gate suppression when persistent mismatch.
5. **Spratling (2017, Behav Brain Sci)** — *A review of predictive coding algorithms*. Shows algebraic equivalence of variants (APC, gPC). **Hook:** choose variant with symmetric weights to avoid weight transport; fits residual dynamics.

### Population dynamics / neural manifolds
6. **Churchland et al. (2012, Nature)** — *Neural population dynamics during reaching*. Rotational low-dim trajectories in motor cortex; dynamics captured by linear systems. **Hook:** enforce low-rank latent (manifold) for graph state via PCA/Koopman subspace; regularize Δ on manifold.
7. **Gallego et al. (2017, Neuron)** — *Cortical population activity within a preserved manifold*. Behavior changes move along fixed manifold; mapping to outputs adapts. **Hook:** keep ONN state in fixed subspace; adapt readout (ORTSF) via fast map.
8. **Sadtler et al. (2014, Nature)** — *Neural constraints on learning*. Learning is fast within manifold, slow outside. **Hook:** constrain updates to tangent space of learned manifold; projection `P_T` before Δ.
9. **Pandarinath et al. (2018, Nat Neuro)** — *Inferring single-trial neural population dynamics*. LFADS captures smooth latent dynamics. **Hook:** use smoothness prior on Δ; stochastic latent for uncertainty u.

### Dendritic computation / neuron-as-network
10. **Poirazi & Mel (2001, Neuron)** — *Impact of active dendrites*. Pyramidal neuron ≈ 2-layer network (subunit nonlinearities). **Hook:** allow per-node subunit gating (AND/OR) before edge integration.
11. **Urbanczik & Senn (2014, Nat Neurosci)** — *Learning by dendritic prediction of somatic spiking*. Local error: dendrite predicts soma; plasticity when mismatch. **Hook:** per-edge local predictor; update using local prediction error without global BP.
12. **Gidon et al. (2020, Science)** — *Dendritic action potentials in human cortex*. Nonlinear plateau -> gating. **Hook:** implement saturating gate on edges; thresholded integration for conflict detection.

### Excitation–Inhibition balance / stabilization
13. **van Vreeswijk & Sompolinsky (1996, Science)** — *Balanced excitation and inhibition*. Asynchronous irregular state stabilizes activity. **Hook:** normalize Δ by estimated E/I ratio; keep total drive near zero-mean.
14. **Denève & Machens (2016, Nat Neuro)** — *Efficient codes and balanced networks*. Predictive coding emerges from E/I balance; errors drive spiking. **Hook:** couple residual magnitude to inhibitory gain; clamp runaway edges.
15. **Tsodyks et al. (1997, J Neurosci)** — *Paradoxical effect of inhibitory feedback*. Strong inhibition stabilizes network. **Hook:** implement inhibitory “penalty” term proportional to norm of conflict residuals.

### Thalamocortical gating / coordination
16. **Sherman & Guillery (2013, MIT Press; review)** — *Functional connections of cortex*. Thalamus as gate/relay with driver vs modulator. **Hook:** model thalamic gate variable g controlling which residuals propagate.
17. **Halassa & Kastner (2017, Neuron)** — *Thalamic functions in distributed cognitive control*. Thalamus coordinates cortical subnetworks; context-dependent gating. **Hook:** context gate per subgraph; switch constraint sets.
18. **Mejias & Wang (2022, Neuron)** — *Mechanistic cortico-thalamic loops*. Frequency-specific gating stabilizes loops. **Hook:** multi-timescale gates (fast evidence, slow context).

### Reservoir / liquid computing
19. **Maass et al. (2002, Nat Comp)** — *Real-time computing without stable states* (LSM). High-dimensional recurrent reservoir + simple readout. **Hook:** treat ONN state updater as constrained reservoir; readout = ORTSF.
20. **Jaeger (2001, GMD report)** — *Echo State Networks*. Spectral radius < 1 for stability. **Hook:** keep Δ linear part spectral radius < 1; projection as echo-state condition.

### Biologically plausible learning / gradient alternatives
21. **Lillicrap et al. (2016, Nat Comm)** — *Random feedback alignment*. Random backward weights approximate gradients. **Hook:** use fixed random feedback for Δ of learnable modules (small MLPs), avoiding weight transport.
22. **Scellier & Bengio (2017, Front Comput Neurosci)** — *Equilibrium propagation*. Two-phase dynamics; local updates from difference between free/clamped phases. **Hook:** apply to constraint solver: run free phase, clamp constraints, update local params.
23. **Xie & Seung (2003, NIPS)** — *Equivalence of backprop and contrastive Hebbian learning*. Energy-based networks with symmetric weights. **Hook:** if symmetric edge weights, use CHL-like local updates.
24. **Whittington & Bogacz (2017, Nat Comm)** — *Approximate gradient in predictive coding network*. Shows PCN performs BP-like gradients. **Hook:** use PC residuals as gradients for small learned modules.

### Stabilization / normalization / conflict handling (bridge)
25. **Cohen & Maunsell (2011, Neuron)** — *Attention gates normalization*. **Hook:** use multiplicative gates to prioritize consistent edges.
26. **Carandini & Heeger (2012, Nat Rev Neuro)** — *Normalization as canonical neural computation*. **Hook:** apply divisive normalization to residuals before projection.
27. **Richards et al. (2019, Nat Neurosci)** — *A deep learning framework for neuroscience*. Survey of plausible credit assignment. **Hook:** provides design space for local rules.

---
## Unified Brain Computation Blueprint → ONN Dynamics
- **State variables**: node embeddings `h`, edge embeddings `r`, gates `g ∈ [0,1]`, uncertainty `u ≥ 0`, manifold latent `z` (low-dim), constraints `(C, τ)`.
- **Error signals**: observation residual `e_obs = x - x_obs`; consistency residual `e_cons = ∇_x φ_C(x)` (logic/phys); inhibitory load `e_inh = ‖e_cons‖`.
- **Update core**: `x ← P_C( x - η * (β1 * e_obs + β2 * e_cons) / (u + ε) )`; gates `g ← σ(k1 * evidence - k2 * conflict)`; uncertainty `u ← u + α*|e_obs| - κ` (clipped ≥0).
- **Stability**: spectral radius < 1 on linear part (ESN rule); divisive normalization on residuals; E/I balance via penalty λ‖e_cons‖. Projection `P_C` = simplex for probs, [0,1] clip for gates, manifold projection for latents.
- **Learning (slow)**: local plasticity on small MLPs using predictive-coding gradients or random-feedback alignment; meta-params (β1, β2, η, gate thresholds) via ES.
- **Timescales**: fast inference loop (K=3–10 iters per frame); slow meta-updates every episode; thalamic/context gate updated slower than edge evidence.

---
## ONN Mapping Table (brain primitive → ONN design)
| Brain primitive | Role | ONN counterpart | Minimal formula/pseudocode | Risk & mitigation |
| --- | --- | --- | --- | --- |
| Error units (cortical FF) | Send mismatch upward | `e_obs`, `e_cons` tensors | `e = x - x_hat`; propagate via edges | Exploding residual → clip & normalize |
| Prediction units (FB) | Send priors downward | constraint targets `(C, τ)` | top-down target update before projection | Bad priors → gate by confidence u |
| Thalamic gate | Contextual routing | gate `g_context` per subgraph | `g = σ(a·evidence - b·conflict)` | Dead gating → floor g_min |
| E/I balance | Stabilize dynamics | inhibitory penalty λ‖e_cons‖ | `Δ ← Δ - λ * e_cons` | Over-damp → schedule λ |
| Neural manifold | Low-dim latent | latent `z`, projection `P_manifold` | `z = U^T x`; `x ← U z` | Manifold drift → periodic re-fit |
| Dendritic subunits | Nonlinear sub-branches | per-edge subunit gate | `r_ij ← f_subunit(r_ij, g_ij)` | Saturation → cap activation |
| Reservoir echo-state | Rich recurrent kernel | constrain spectral radius ρ<1 | rescale W: `W ← W / ρ(W)` | Loss of memory → tune ρ to 0.9 |
| Equilibrium propagation | Local gradient | two-phase Δx; weight update ∝ (x_clamped - x_free) | Extra cost → run small K, reuse warm start |
| Random feedback alignment | Plausible credit | fixed random B for backward | `Δθ ∝ B^T e` | Slow convergence → small MLPs only |
| Normalization (Carandini-Heeger) | Prevent runaway | divisive norm on residuals | `e ← e / (σ + ‖e‖)` | Underflow → choose σ floor |

---
## 24h Minimal Experiment Plan (code)
- **Goal**: Validate two primitives in ONN codebase: (A) residual-driven projection with gates; (B) manifold-constrained updates.
- **Setup**: Use existing tests; add two small synthetic scenarios; run inside venv.
- **Task A (Residual+Gate)**:
  - Build toy graph (3 nodes, 3 edges) with conflicting relation; run LOGOS step with gate update; expect conflict edge gate ↓ and projection restores feasibility.
  - Metrics: violation before/after, gate values, residual norms.
- **Task B (Manifold projection)**:
  - Create low-rank latent U (rank 2) and random edge states; run update with manifold projection; check Δ is confined to span(U).
  - Metrics: projection error, norm outside manifold.
- **Commands (venv)**:
  - `python -m venv .venv && source .venv/bin/activate && pip install -e .`
  - `pytest tests/unit/test_core_tensors.py tests/unit/test_solver_pc.py -q`
  - Add two new tests under `tests/unit/test_brain_blueprint.py` and run them.
- **Pass criteria**: (A) violation drops, gates in [0,1]; (B) norm outside manifold < 1e-6 after projection.

---
## Refactor Pointers (where to plug)
- **LOGOS** (`src/onn/core/solver.py`, `ops/logos_solver.py`): implement residual-driven Δ, gating, normalization, projection; enforce spectral radius constraint for linear part.
- **Tensors** (`src/onn/core/tensors.py`): ensure gates `g`, uncertainty `u`, manifold latent placeholders.
- **ES** (`src/onn/es/*`): treat β1, β2, η, gate thresholds as meta-params; fitness = violation reduction + stability.
- **Eval** (`src/onn/eval/*`): add metrics for residual norm, gate sparsity, manifold leakage.

---
## Notes
- Citations emphasize explicit computation (residuals, gating, energy) to keep ONN aligned with mechanistic neuroscience.
- Use this file as spec source for ONN-ES brain-aligned dynamics.
