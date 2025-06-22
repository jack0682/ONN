# 🧠 Ontology Neural Network (ONN)

> **A Topological-Ricci Semantic Reasoning Framework for Contextual Relational Cognition**

---

## 📚 Overview

The Ontology Neural Network (ONN) reimagines perception and reasoning as the construction of **relational, topological, and temporally evolving semantic structures**.  
Whereas conventional AI seeks to classify or segment entities, ONN seeks to **map meaning itself** as the persistence and transformation of relational forms.  

ONN embodies a **relation-first ontology**: the world is not made of isolated objects but of **webs of interaction**, where meaning is preserved by the **topological class of these webs** and their **smooth evolution through time**.

---

## 🔍 Philosophical Foundations

- **Relational ontology:**  
Objects do not possess intrinsic identity. Their meaning arises only through their relations within the semantic web.

- **Context as a topological invariant:**  
A context is meaningful if and only if its relational graph belongs to the same topological class across time:
$\mathcal{C}(t) \cong \mathcal{C}(t') \iff d_{\mathrm{PH}}(G_{\mathcal{C}}(t), G_{\mathcal{C}}(t')) < \epsilon_{\mathrm{context}}$

- **Continuity of existence:**  
Change does not destroy meaning if the **global topological structure** is preserved, even when local configurations deform.

- **Boundaries of meaning emerge naturally:**  
Context boundaries are not predefined; they arise where Ricci curvature of the relation graph exhibits meaningful discontinuity.

---

## 📊 Mathematical Backbone

### 1️⃣ Semantic State Tensor

Each object $$( o_i $$) is represented by a semantic state vector:
$$
\mathcal{S}_i(t) =
\begin{bmatrix}
\mathbb{L}_i(t) \\
\mathbb{B}_i(t) \\
\mathbb{F}_i(t) \\
\mathbb{I}_i(t)
\end{bmatrix}
\in \mathbb{R}^d
$$

where:
- \( \mathbb{L}_i(t) \): locativeness (position, reference frame)
- \( \mathbb{B}_i(t) \): boundedness (physical extent, affordance boundaries)
- \( \mathbb{F}_i(t) \): formness (geometry, appearance)
- \( \mathbb{I}_i(t) \): intentionality (affordance, functional role)

Temporal evolution:
\[
\dot{\mathcal{S}}_i(t) = \frac{d}{dt} \mathcal{S}_i(t)
\]

---

### 2️⃣ Relational Descriptor

Semantic interaction:
\[
I_{ij}(t) = \mathcal{G}\big( \mathcal{S}_i(t), \mathcal{S}_j(t), R_{ij}(t) \big)
\]

where:
\[
R_{ij}(t) =
\begin{bmatrix}
d_{ij}(t) \\
\theta_{ij}(t) \\
\phi_{ij}(t)
\end{bmatrix}
\]
with:
- \( d_{ij} \): distance
- \( \theta_{ij}, \phi_{ij} \): orientation descriptors

---

### 3️⃣ Relation Graph and Weights

Graph:
\[
G_{\mathcal{C}} = (V_{\mathcal{C}}, E_{\mathcal{C}})
\]

Weights:
\[
w(v_i) = \| \mathcal{S}_i \|_2
\]
\[
w(e_{ij}) = \| I_{ij} \|_2
\]

---

### 4️⃣ Forman Ricci Curvature

For edge \( e_{ij} \):
\[
\operatorname{Ric}_F(e_{ij}) =
w(e_{ij}) \left[
\frac{w(v_i) + w(v_j)}{w(e_{ij})}
- \sum_{e_k \sim e_{ij}} \frac{w(v_i)}{\sqrt{w(e_{ij})w(e_k)}}
- \sum_{e_k \sim e_{ij}} \frac{w(v_j)}{\sqrt{w(e_{ij})w(e_k)}}
\right]
\]

where:
- \( e_k \sim e_{ij} \): edge \( e_k \) shares a node with \( e_{ij} \)

This curvature reflects local consistency or distortion in the relational web.

---

### 5️⃣ Contextual Ricci Smoothness

Mean curvature:
\[
\bar{\operatorname{Ric}}_F(\mathcal{C}) =
\frac{1}{|E_{\mathcal{C}}|} \sum_{e \in E_{\mathcal{C}}} \operatorname{Ric}_F(e)
\]

Smoothness loss:
\[
\mathcal{L}_{\mathrm{ricci\text{-}internal}} =
\sum_{e \in E_{\mathcal{C}}}
\left(
\operatorname{Ric}_F(e) - \bar{\operatorname{Ric}}_F(\mathcal{C})
\right)^2
\]

---

### 6️⃣ Context Boundary Curvature

Encourages distinct curvature at context edges:
\[
\mathcal{L}_{\mathrm{ricci\text{-}boundary}} =
\sum_{\mathcal{C}_i, \mathcal{C}_j}
\mathbb{I}(\text{adjacent}) \,
\left(
\operatorname{Ric}_F(\mathcal{C}_i) - \operatorname{Ric}_F(\mathcal{C}_j)
\right)^{-2}
\]

where:
\[
\operatorname{Ric}_F(\mathcal{C}_i) = 
\bar{\operatorname{Ric}}_F(\mathcal{C}_i)
\]

---

### 7️⃣ Topological Preservation via Persistent Homology

Topological invariant loss:
\[
\mathcal{L}_{\mathrm{ph}} =
d_{\mathrm{PH}}(G_{\mathcal{C}}(t), G_{\mathcal{C}}(t'))
\]

Where:
- \( d_{\mathrm{PH}} \) is the bottleneck or Wasserstein distance between persistence diagrams of the graphs.

---

### 8️⃣ Context Loss

\[
\mathcal{L}_{\mathrm{context}} =
\mathcal{L}_{\mathrm{ricci\text{-}internal}} 
+ \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{ricci\text{-}boundary}} 
+ \lambda_{\mathrm{ph}} \mathcal{L}_{\mathrm{ph}}
\]

---

### 9️⃣ Full ONN Loss

\[
\mathcal{L}_{\mathrm{total}} =
\mathcal{L}_{\mathrm{pred}}
+ \lambda_1 \mathcal{L}_{\mathrm{flow}}
+ \lambda_2 \mathcal{L}_{\mathrm{relation}}
+ \lambda_3 \mathcal{L}_{\mathrm{intent}}
+ \lambda_4 \mathcal{L}_{\mathrm{context}}
\]

---

## 🌌 Philosophical Significance (Detailed)

ONN **models meaning as the persistence of relational structure**, not as the labeling of isolated objects.

- **Relational persistence:**  
  A cup next to a red ball remains a cup-next-to-red-ball, whether the ball moves slightly or the cup rotates.

- **Meaning as topological form:**  
  Meaning persists if the global shape (homology) of the relational web remains unchanged.

- **Curvature as boundary of being:**  
  Where curvature jumps, context ends and another begins. These boundaries are **discovered, not imposed**.

- **Existence as flow:**  
  Contexts flow through time, like Ricci flow smooths curvature, adapting to change without rupturing identity.

---

## 📏 Project Structure

``` bash
ONN/
├── data/
├── models/
│ ├── embedding.py
│ ├── encoder.py
│ ├── interaction.py
│ ├── ricci.py
│ ├── ph.py
├── train/
│ ├── loss.py
│ ├── trainer.py
│ ├── evaluator.py
├── utils/
│ ├── graph_tools.py
│ ├── logger.py
├── experiments/
│ ├── run_train.py
│ ├── run_eval.py
├── README.md
└── requirements.txt
```


---

## 💡 Future Directions

- Equivariant GNNs with Ricci regularization
- Multi-modal fusion (vision + language + force)
- Ontology-grounded symbolic planner integration
- Online learning of new contexts
- Meta-layer reasoning via D-LOGOS

---

## ✉ Contact

`jaehong_oh@csa-lab.ai`  
Refer to companion modules: **SEGO**, **IMAGO**, **LOGOS**

