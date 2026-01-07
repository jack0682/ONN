# 00. High-Level Plan: CSA (Collaborative Sensing Agent)

> **Authority**: This document is the **Constitution** of the CSA project. It defines the philosophical, mathematical, and architectural axiomatic truths of the system.
> **Reference**: Based on *"Ontology Neural Network and ORTSF: A Framework for Topological Reasoning and Delay-Robust Control"*.

---

## 0. How to Read and Use This Document

<!--
This section explains to future-you and to LLMs how this file should be interpreted.
-->

- **Audience**:
    - Human architect (You)
    - LLM collaborators (Gemini, Codex, Claude)
- **Authority level**:
    - This document is **source of truth** above all other spec files.
    - If any lower-level spec conflicts with this document, this document wins.
- **Update policy**:
    - Update this document **only when the fundamental direction changes** (architecture/vision).
    - For small implementation details, use `20_impl_plan.ir.yml` and backlog instead.
- **Tone**:
    - This is a strict, axiom-driven document. It defines the "Physics" of our system.

---

## 1. System Identity

<!--
Define what this system fundamentally IS.
-->

### 1.1 System Name
**CSA (Cognitive synergy Architecture)**
*Subtitle: An ONN-Orchestrated Topological Cognitive System*

### 1.2 One-Sentence Definition
> **CSA is a closed-loop cognitive architecture where "Existence" is defined by topological relationships (via ONN), and "Action" is the delay-robust flow of continuous semantic traces (via ORTSF).**

### 1.3 Longer Description
CSA differs from traditional robotic stacks (Perception $\to$ Planning $\to$ Control) by replacing the discrete symbol layer with a continuous **Semantic Manifold**.
- It does not "detect" objects as labels; it **anchors** them as tensors in a relationship graph.
- It does not "plan" discrete steps; it **flows** intent through a curvature-guided manifold.
- It does not "command" servos directly; it **projects** a reasoning trace that the control fabric (ORTSF) chases, guaranteed stable against communication delays.

This system is designed for **dynamic, unstructured environments** where relationships (support, containment, intent) matter more than absolute metric coordinates.

### 1.4 Goals (What This System Must Enable)
1.  **Topological "Common Sense"**: The system must inherently understand that a cup *on* a table moves *with* the table, without hard-coded scripts.
2.  **Delay-Robust Control**: The system must remain stable even if the reasoning loop (ONN) runs at 5Hz and the robot controller runs at 1kHz, with variable latency ($ \Delta t $).
3.  **Self-Correction**: The system must detect logical contradictions (e.g., "A inside B" and "B inside A") and self-correct via the Projection-Consensus solver.

### 1.5 Non-Goals
- **Not a Black-Box End-to-End Model**: We do not train a single giant Transformer from pixels to torque. We explicitly model **Topology** and **Control Theory**.
- **Not a Generic SLAM Library**: We are not competing with ORB-SLAM. We use SLAM pose as input, but our map is **Semantic**, not just Point Cloud.

---

## 2. Problem Space and Use Cases

<!--
Why do we need this? What specifically does it solve that existing stacks fail at?
-->

### 2.1 The Semantic Gap
Existing approaches treat "Meaning" as labels painted on a 3D map.
- *Problem*: If a detector flickers, the object "disappears". If a segmentation mask shifts, the logic breaks.
- *CSA Solution*: **Existence is Relation**. An object persists because its relationships (supported by table, near hand) are topologically stable, even if visual detection is noisy.

### 2.2 The Control-Reasoning Gap
High-level reasoning is slow (LLMs, Graph Search); Low-level control must be fast.
- *Problem*: Connecting a 1Hz planner to a 1kHz motor loop causes stuttering and instability.
- *CSA Solution*: **ORTSF (Ontological Real-Time Semantic Fabric)**. We treat reasoning output as a **continuous signal** (Trace) and apply small-gain control theory to guarantee stability despite the delay.

### 2.3 Use Case: The "Cup-Water-Table" Scenario (V0 Benchmark)
- **Narrative**: A robot arm must pour water into a cup. The cup is on a movable tray. A human moves the tray suddenly.
- **Challenge**:
    1.  The system must maintain the relation `(Cup, SupportedBy, Tray)` instantly.
    2.  When the tray moves, the system must infer the cup moves *without* re-detecting the cup from scratch.
    3.  The pouring action must adjust smoothly (no jerk) despite the perception delay.
- **Success Criteria**:
    - **Constraint Violation Rate $\approx 0$**: No logic breaks (e.g., pouring on the old location).
    - **Smoothness**: No oscillation in the arm trajectory.

---

## 3. High-Level Architectural View

<!--
The "Map of Continents". Refines the 3-Operator Theory.
-->

### 3.1 Architectural Overview
The system is an **Operator-Based State Machine**. It does not have "modules" that pass messages blindly; it has **Operators** that transform the system state.

The State is a **Graph $G(t)$** embedded in a manifold constraints $\mathcal{C}$.

### 3.2 The Trinity of Operators
**1. SEGO (Semantic Graph Ontology Mapper)**
*The Perception Operator.*
- **Input**: Raw Sensor Stream $z_i$ (RGB-D, Proprioception).
- **Operation**: Gauge Anchoring. Projects raw data onto the Semantic Manifold.
- **Output**: Initialized Node Tensors $S_i$ and Edge Candidates $E_{ij}$.
- **Core Math**: $S_i = \text{Encoder}(z_i)$; $E_{ij}$ filtered by geometric proximity.

**2. IMAGO (Intent Modeling & Action Generation Operator)**
*The Planning Operator.*
- **Input**: Stabilized Graph $G(t)$, Goal Description.
- **Operation**: Intent Flow via Curvature.
- **Output**: **Reasoning Trace** $\mathcal{R}_{trace}(t)$.
- **Core Math**: Uses **Forman-Ricci Curvature** to identify functional clusters. Generates a target state $S_{target}$ that flows from the current state.

**3. LOGOS (Logical Ontological Generator for Self-Adjustment)**
*The Correction Operator.*
- **Input**: Constraint Violations (Energy $\mathcal{E}$), Prediction Errors.
- **Operation**: Projection-Consensus & Evolutionary Update.
- **Output**: Updated Graph Weights $w_{ij}$, Constraint Parameters $\theta$.
- **Core Math**: $S^{k+1} \leftarrow P_{\mathcal{C}}(S^k - \eta \nabla \mathcal{L})$. This is the **solver** that enforces "Common Sense".

### 3.3 High-Level Data Flows

**Flow A: The Reasoning Loop (Slow, Deep)**
1.  **Observation**: Sensors $\to$ SEGO $\to$ Raw Graph.
2.  **Consensus**: Raw Graph $\to$ LOGOS Solver $\to$ Stabilized Graph (Valid Topology).
3.  **Intent**: Valid Graph $\to$ IMAGO $\to$ Reasoning Trace $\mathcal{R}_{trace}$.

**Flow B: The Control Loop (Fast, Reactive)**
1.  **Trace Consumption**: $\mathcal{R}_{trace}$ (from ONN) $\to$ ORTSF.
2.  **Prediction**: $\mathcal{R}_{trace}(t) \to \mathcal{T}_{predict} \to \hat{x}(t+\Delta t)$.
3.  **Action**: $\hat{x} \to$ Controller $\to$ Motors.

---

## 4. Multi-LLM Collaboration Model

<!--
Defines exactly how Gemini, Codex, and Claude work together.
-->

### 4.1 Actors & Roles
- **Human Architect**: Defines Intent, Axioms, and Final Judgment. (Source of Truth)
- **Gemini (The Architect)**:
    - Domain: Global reasoning, Interface definitions (`ir.yml`), Documentation (`.md`).
    - Responsibility: "Ensure the math is right and the system fits together."
- **Codex (The Planner)**:
    - Domain: Directory structure, boilerplate generation, function stubs.
    - Responsibility: "Build the skeleton that compiles."
- **Claude (The Implementer)**:
    - Domain: Deep logical implementation, refactoring, complex algorithms.
    - Responsibility: "Write the Projection-Consensus solver loop correctly."

### 4.2 The Charter of Separation
> **No Agent Shall Override Another's Domain.**

1.  **Gemini** produces the **Plan** and **Specs**. It does not write the final Python loop.
2.  **Codex** creates the **Structure**. It does not invent new mathematical operators.
3.  **Claude** fills the **Logic**. It does not redefine the architecture or delete constraints.

---

## 5. Repository Structure (Top-Level)

### 5.1 Directory Layout
```text
.
├── spec/                 # The Law (IR files, MD specs)
│   ├── 00_high_level_plan.md
│   ├── 02_onn_math_spec.md
│   ├── 10_architecture.ir.yml
│   └── 11_interfaces.ir.yml
├── src/
│   ├── onn/              # ONN Core Package
│   │   ├── core/         # Solvers, Tensor Definitions
│   │   ├── ops/          # SEGO, LOGOS, IMAGO implementations
│   │   └── modules/      # PyTorch Modules / Curvature Ops
│   ├── ortsf/            # Control Package
│   │   ├── fabric/       # Delay compensation, Interpolators
│   │   └── interfaces/   # ROS/HW bridges
│   └── sim/              # Simulation Envs (Cup-Water tests)
├── tests/
├── config/               # Hyperparameters (Constraint weights)
└── scripts/              # Launchers, Training loops
```

### 5.2 Key Spec Files
- `00_high_level_plan.md`: The Vision (This file).
- `02_onn_math_spec.md`: The Math (Tensor shapes, update equations).
- `11_interfaces.ir.yml`: The API (Message schemas, topic names).

---

## 6. Design Principles

### 6.1 Topological Validity First
> "It is better to halt than to hallucinate."
If the Projection-Consensus solver cannot find a solution within the time budget, the system effectively "pauses" reasoning (holds last valid state) rather than emitting a broken, physically impossible graph state.

### 6.2 Existence via Relation
An object is defined by its edges.
- If $z_i$ (sensor detection) vanishes, but $E_{ij}$ (support/containment) remain strong, the object **still exists** in the graph.
- This provides robustness against occlusion.

### 6.3 Continuous Semantics
We avoid discrete switches in the control path.
- Instead of `if mode == pouring`, we have a continuous tensor $I_{pouring} \in [0, 1]$.
- This ensures the output $\mathcal{R}_{trace}$ is differentiable, allowing ORTSF to predict future states smoothly.

---

## 7. Phased Roadmap

### 7.1 Phase 1: The Solver (V0 - Current Focus)
**Goal**: Validate the Topological Reasoning Engine.
- **Scenario**: Static "Cup on Table".
- **Tasks**:
    1.  Implement Tensor Structures ($B, F, I$).
    2.  Implement `ProjectionConsensusSolver`.
    3.  Implement `FormanRicci` curvature calculation.
    4.  Verify: $S_{final}$ satisfies constraints $C$.

### 7.2 Phase 2: The Trace (ORTSF Integration)
**Goal**: Validate Delay-Robust Control.
- **Scenario**: Moving the Table while holding the Cup.
- **Tasks**:
    1.  Implement `ReasoningTrace` bundle.
    2.  Implement `DelayPredictor` (Small Gain).
    3.  Connect ONN output $\to$ Dummy Controller.
    4.  Verify: Control signal is smooth despite injected lag.

### 7.3 Phase 3: The Evolution (Learning)
**Goal**: Self-Adjustment via ES.
- **Scenario**: Unknown object interaction.
- **Tasks**:
    1.  Wrap the system in an Evolutionary Strategy loop.
    2.  Optimize Constraint Weights $W_C$ based on violation rates.
    3.  Real-time relation discovery.

---

## 8. Known Risks and Open Questions

### 8.1 Technical Risks
1.  **Solver Convergence Speed**:
    - *Risk*: The Projection-Consensus loop might be too slow for real-time (>10Hz).
    - *Mitigation*: Use a fixed-iteration budget (e.g., 5 iters) + Warm Start from previous frame.
2.  **Ricci Curvature Complexity**:
    - *Risk*: Computing curvature on dense graphs is $O(E^2)$ or higher.
    - *Mitigation*: Keep the active graph sparse via geometric filtering in SEGO.
3.  **Constraint Conflict**:
    - *Risk*: Hard constraints might make the feasible set empty ($\emptyset$).
    - *Mitigation*: Use "Exact Penalty" (Soft $\to$ Hard) to always find a *least-bad* solution.

### 8.2 Theoretical Questions
- *Q*: How do we handle "categorical" changes (e.g., object breaks) in a continuous manifold?
- *A*: Represent categories as high-dimensional simplex corners. The transition is a fast but continuous trajectory through the simplex interior.

---

## 9. Update Rules for This Document

- **When to Update**:
    - Only when the *Trinty of Operators* or the *Fundamental Math* changes.
    - Adding a new sensor or robot does NOT require updating this file (update `11_interfaces.ir.yml` instead).
- **Versioning**:
    - This is effectively **Version 2.0** of the High-Level Plan, marking the shift to the ONN/ORTSF paradigm.

---

## X. Charter of Non-Interference (Re-stated)
*To ensure system integrity:*
1.  **AI Agents must not hallucinate interfaces.** Use `11_interfaces.ir.yml`.
2.  **Code must follow the Math.** `02_onn_math_spec.md` is the law for implementation.
3.  **Human Authority is Final.**