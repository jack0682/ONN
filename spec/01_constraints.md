---

01_constraints.md

System Constraints & Foundational Principles (CSA V0)
(Version 2.1 — Optimized for ONN/ORTSF Implementation)

---

1. Platform & Environment Constraints (V0 Scope)

1.1 Runtime & Languages
- **Core Logic**: Python 3.10+ (primary for V0).
- **Tensor Engine**: PyTorch (must support autograd for solver).
- **Performance**: V0 is for *correctness* & *simulation*, not hard real-time (< 10Hz is acceptable for Reasoning).
- **OS**: macOS (Dev) / Linux (Target). No Windows dependencies.

1.2 Vendor/Framework Independence
- **No ROS dependency in Core**: `src/onn/` must be pure Python package.
- **Interfaces**: All data exchange must use schemas from `11_interfaces.ir.yml`.
- **Visualization**: Use lightweight tools (e.g., Rerun.io or Matplotlib) for V0, not Rviz unless via adapter.

---

2. Architectural Constraints (The Trinity)

2.1 Operator-Based Architecture
The system MUST be implemented as three distinct Operators (The ONN Trinity):
1.  **SEGO (Semantic Graph Ontology Mapper)**: $z \to S$. Perception only, no logic.
2.  **LOGOS (Logical Ontological Generator for Self-Adjustment)**: $S \to S_{valid}$. The ONLY place where constraints are enforced.
3.  **IMAGO (Intent Modeling & Action Generation Operator)**: $S_{valid} \to \mathcal{R}_{trace}$. The ONLY place where future plans are generated.

2.2 No Discrete Logic in Critical Paths
- **Forbidden**: `if object.name == 'Cup': ...`
- **Required**: Encode 'Cup-ness' as a tensor $F_{cup}$. Use vector operations.
- **Exception**: State machine for high-level mission phases (e.g., "Scanning" vs "Pouring") is allowed in Application Layer, but *not* inside the ONN solver loop.

2.3 Topological Validity First
- If LOGOS fails to converge (Constraints violated), the system must **halt** or **hold last valid state**.
- Never pass an invalid graph to IMAGO.
- "Existence" is defined by Edges, not Nodes. A Node with no Edges is noise.

---

3. Engineering Decisions for V0 (The "Cup-Water" Benchmark)

3.1 Sensor Configuration
- **Input**: Single RGB-D stream (simulated or RealSense).
- **Format**: $(H, W, 4)$ tensors + Camera Matrix $K$.

3.2 Semantic Manifold Dimensions
- **Boundedness ($B$)**: 16 dim (Collision primitives).
- **Formness ($F$)**: 32 dim (Visual embedding).
- **Intentionality ($I$)**: 16 dim (Affordance vectors).
- **Total State**: $N \times 64$ per timestamp.

3.3 Solver Constraints
- **Primary Method**: Projection-Consensus (Iterative Gradient Descent).
- **Max Iterations**: 10 per frame (for V0).
- **Losses**:
    - Intersection Loss (Physics).
    - Support Loss (Gravity).
    - Containment Loss (Context).

3.4 Control Interface (ORTSF)
- **Trace Horizon**: 1.0 second lookahead.
- **Update Rate**: Simulator runs at 100Hz, ONN runs at ~5-10Hz.
- **Interpolation**: Cubic Spline binding.

---

4. Multi-LLM Collaboration Constraints

4.1 Role Separation (Strict)
- **Gemini**: Updates `spec/` files. Never writes code.
- **Codex**: Updates `20_impl_plan` and `31_todo_backlog`. Checks existing code.
- **Claude**: Writes code in `src/`. Never changes architecture.

4.2 Source of Truth
- **Files > Chat**: If the user says "Change X" in chat, update `spec/` FIRST, then code.
- **Code reflects Spec**: Code that deviates from `02_onn_math_spec.md` is a bug.

---

5. Non-Negotiable Rules

1.  **Do not solve problems Recurrently**: The ONN is Markovian (State $t$ depends on $t-1$). Do not build long history buffers unless specified in `02`.
2.  **No Magic Numbers**: specific thresholds (e.g., `dist < 0.5`) must be in a Config object, not hardcoded.
3.  **Testability**: Every Operator must be testable with synthetic tensors (Unit Tests).

---
