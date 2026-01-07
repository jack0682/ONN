# ONN Mathematical Specification (V0)

> **Authority**: This document defines the mathematical laws governing the CSA system.
> **Scope**: It covers the State Space, Operators, Constraints, Loss Functions, and Algorithms for the V0 implementation.
> **Compliance**: All implementation code in `src/onn/` must implement these equations exactly.

---

## 1. State Space

The fundamental unit of reality in CSA is the **Semantic Manifold** $\mathcal{M}$. The system state at time $t$ is a graph $G(t)$ embedded in this manifold.

### 1.1 The Semantic Node ($S_i$)
A node $S_i$ represents a discrete entity (object, zone, agent). It is a tensor in $\mathbb{R}^{64}$ composed of three fiber bundles:

$$ 
S_i = [\mathbf{B}_i; \mathbf{F}_i; \mathbf{I}_i] \in \mathbb{R}^{16+32+16} 
$$ 

#### 1.1.1 Boundedness Tensor ($\mathbf{B}_i \in \mathbb{R}^{16}$)
Encodes the spatiotemporal existence and geometric priors.
-   $b_{0:3}$: Position quaternion/vector $(x, y, z, w)$ (or centroid + radius).
-   $b_{4:7}$: Orientation quaternion $(q_x, q_y, q_z, q_w)$.
-   $b_{8:11}$: Extents/Scale $(l, w, h, s)$.
-   $b_{12:15}$: Physics derivatives $(\dot{x}, \dot{y}, \dot{z}, mass)$.

#### 1.1.2 Formness Tensor ($\mathbf{F}_i \in \mathbb{R}^{32}$)
Encodes the immutable identity (visual embedding).
-   Derived from a frozen encoder (e.g., reduced CLIP/ViT embedding).
-   Invariant to rotation and translation.
-   Used for re-identification across frames.

#### 1.1.3 Intentionality Tensor ($\mathbf{I}_i \in \mathbb{R}^{16}$)
Encodes the functional affordances and current role.
-   $I_{vector}$: A continuous vector space where "containability", "graspability", etc., are directions.
-   Example: A cup has a high projection onto $\vec{v}_{contain}$.

### 1.2 The Semantic Edge ($E_{ij}$)
A directed edge $E_{ij}$ represents the topological relationship from $S_i$ to $S_j$.

$$ 
E_{ij} = \{\mathbf{r}_{ij}, w_{ij}\} 
$$ 

-   $\mathbf{r}_{ij} \in \mathbb{R}^{K}$: Relation embedding vector. (e.g., relative pose, semantic relation type).
-   $w_{ij} \in [0, 1]$: scalar weight representing the strength/confidence of the connection.

### 1.3 The Semantic Graph ($G(t)$)
The global state is the tuple:
$$ 
G(t) = (\mathcal{V}, \mathcal{E}) = (\{S_i\}_{i=1}^N, \{E_{ij}\}_{i,j \in \mathcal{V}}) 
$$ 

---

## 2. Operators

The system evolves via the sequential application of three operators.

### 2.1 SEGO: Gauge Anchoring ($\mathcal{E}_{SEGO}$)
**Purpose**: Lifts raw sensor data $z$ onto the manifold $\mathcal{M}$.

$$ 
G_{raw} = \mathcal{E}_{SEGO}(z_t) 
$$ 

**Definition**:
Let $z_t$ be the raw RGB-D frame.
1.  **Detection**: $\{o_k\} = \text{Detector}(z_t)$.
2.  **Projection**: For each detection $o_k$:
    -   $\mathbf{B}_k = \phi_{geom}(o_k)$ (Geometric back-projection).
    -   $\mathbf{F}_k = \phi_{vis}(o_k)$ (Visual encoder).
    -   $\mathbf{I}_k = \phi_{prior}(class\_id)$ (Initial affordance prior).
3.  **Edge Proposal**: $E_{ij}$ initialized based on spatial proximity $d(S_i, S_j) < \delta$.

### 2.2 LOGOS: Projection-Consensus ($\mathcal{P}_{LOGOS}$)
**Purpose**: Projects the potentially inconsistent $G_{raw}$ onto the valid constraint manifold $\mathcal{C}$.

$$ 
G_{valid} = \mathcal{P}_{LOGOS}(G_{raw}; \mathcal{C}) 
$$ 

**Optimization Problem**:
$$ 
S^* = \arg\min_S \left( ||S - S_{raw}||^2 + \lambda \mathcal{L}_{total}(S) \right) 
$$ 
Where $\mathcal{L}_{total}$ encodes the constraints (Section 4).

### 2.3 IMAGO: Intent Flow ($\mathcal{R}_{IMAGO}$)
**Purpose**: Generates the Reasoning Trace $\tau$ by analyzing the topology of $G_{valid}$.

**Method (Forman-Ricci Curvature)**:
For each edge $e_{ij}$:
$$ 
\text{Ric}(e_{ij}) = 4 - d_i - d_j + 3 \Delta_{ij} 
$$ 
Where $d_i, d_j$ are node degrees, and $\Delta_{ij}$ is the number of triangles supported by $e_{ij}$.

**Cluster Identification**:
Edges with highly negative $\text{Ric}(e_{ij})$ represent "bridges" between functional clusters.
The Intent Flow $\tau$ is defined as a geodesic path on the manifold minimizing the traversal of negative curvature bridges unless required by the task.

---

## 3. Constraints ($\mathcal{C}$)

The solver enforces these invariants.

### 3.1 Physical Constraints ($\mathcal{C}_{phys}$)
1.  **Non-Intersection**: Two rigid bodies cannot occupy the same volume.
    $$ 
    \forall i \neq j: \text{Vol}(\mathbf{B}_i \cap \mathbf{B}_j) = 0 
    $$ 
2.  **Gravity Support**: If $S_i$ is supported by $S_j$, $S_i$ must be physically above $S_j$.
    $$ 
    E_{ij}.\text{type} = \text{SUPPORT} \implies \mathbf{B}_i.z > \mathbf{B}_j.z 
    $$ 

### 3.2 Logical Constraints ($\mathcal{C}_{logic}$)
1.  **Relation Asymmetry**:
    $$ 
    E_{ij}.\text{type} = \text{INSIDE} \implies E_{ji}.\text{type} \neq \text{INSIDE} 
    $$ 
2.  **Transitivity** (Soft):
    $$ 
    (i \to j) \land (j \to k) \implies \text{Consistency}(i \to k) 
    $$ 

### 3.3 Topological Constraints ($\mathcal{C}_{topo}$)
1.  **Existence Continuity**: Objects do not vanish instantly.
    $$ 
    ||S_i(t) - S_i(t-1)|| < \epsilon_{max} 
    $$ 

---

## 4. Loss Function

The LOGOS solver minimizes this total energy:

$$ 
\mathcal{L}_{total}(S, E) = \lambda_{data}\mathcal{L}_{data} + \lambda_{phys}\mathcal{L}_{phys} + \lambda_{logic}\mathcal{L}_{logic} 
$$ 

### 4.1 Terms
1.  **Data Fidelity Loss ($\mathcal{L}_{data}$)**:
    $$ 
    \mathcal{L}_{data} = \sum_i ||\mathbf{B}_i - \mathbf{B}_{i, raw}||^2 
    $$ 
    (Don't hallucinate movement away from observation unless necessary).

2.  **Physics Loss ($\mathcal{L}_{phys}$)**:
    $$ 
    \mathcal{L}_{phys} = \sum_{i,j} \text{ReLU}(R_i + R_j - ||\vec{p}_i - \vec{p}_j||) 
    $$ 
    (Simple sphere-collision penalty for V0).

3.  **Logic Loss ($\mathcal{L}_{logic}$)**:
    $$ 
    \mathcal{L}_{logic} = \sum_{edges} (1 - \text{Validity}(E_{ij})) 
    $$ 

### 4.2 Hyperparameters (V0 Defaults)
-   $\lambda_{data} = 1.0$
-   $\lambda_{phys} = 10.0$ (Physics violations are expensive)
-   $\lambda_{logic} = 5.0$

---

## 5. Algorithms

### 5.1 LOGOS Solver (Gradient Descent with Projection)

```python
# Pseudocode - Implementation Reference
def solve_consensus(G_raw, config):
    S = G_raw.nodes.clone()
    
    for k in range(config.max_iterations):
        # 1. Compute Gradients
        S.requires_grad = True
        loss = compute_total_loss(S, G_raw.edges)
        grads = torch.autograd.grad(loss, S)
        
        # 2. Gradient Step
        S = S - config.learning_rate * grads
        
        # 3. Projection (Hard Constraints)
        S = project_bounds(S) # e.g., ensure radius > 0
        
        # 4. Convergence Check
        if loss < config.tolerance:
            break
            
    return S
```

### 5.2 Convergence Criteria
The solver is considered converged if:
1.  $||\Delta S|| < \epsilon_{step}$
2.  $\mathcal{L}_{phys} \approx 0$ (No physics violations)

If max iterations reached without convergence, the system raises a `ConstraintViolationWarning` and holds the previous valid state or proceeds with the "least bad" solution (Soft-Fail).

---

## 6. Falsifiable Predictions

These predictions serve as the acceptance criteria for the Math implementation.

| ID | Hypothesis | Validation Metric |
|---|---|---|
| **H-01** | The Solver reduces physical violations to near-zero. | $\mathcal{L}_{phys}(S_{final}) < 10^{-3}$ on "Stacked Cubes" test. |
| **H-02** | Implicit relationships (A moves with B) emerge from edge weights. | Correlation between $\vec{p}_A$ and $\vec{p}_B$ when $w_{AB} > 0.8$. |
| **H-03** | Forman-Ricci curvature peaks at bottleneck edges (e.g., Doorway). | Visual inspection of curvature heatmap on "Room Navigation" graph. |

---

## 7. Open Questions

1.  **Differentiability of Topology**: How do we handle discrete changes in edge existence ($w_{ij} \to 0$) in a differentiable manner? *Proposed Solution: Use sigmoid gating for weights.*
2.  **Manifold Dimension**: Is 64 dimensions sufficient for complex tasks? *V0 assumption: Yes.*
