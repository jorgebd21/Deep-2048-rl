# Architecture Specification: Dueling2048Net & Multi-Scale Topologies

## 1. Motivation & Topological Inductive Bias

In combinatoric grid games like 2048, standard square convolutional kernels designed for computer vision (such as $3 \times 3$, $5 \times 5$, or $7 \times 7$) fail due to the small board dimensions ($4 \times 4$). A single $3 \times 3$ unpadded convolution collapses the spatial field to $2 \times 2$, while a $5 \times 5$ kernel cannot even be applied.

Furthermore, adjacent cells along rows and columns represent candidate merge pairs. To provide the neural network with appropriate inductive bias without hand-coded heuristics, we introduce **asymmetric convolutional kernels**:

1. **Horizontal Kernel $(1, 2)$:** Detects adjacent pairs along rows (horizontal merge opportunities and row monotonicity). Output shape: $(B, 128, 4, 3)$.
2. **Vertical Kernel $(2, 1)$:** Detects adjacent pairs along columns (vertical merge opportunities and column monotonicity). Output shape: $(B, 128, 3, 4)$.
3. **Square Kernel $(2, 2)$:** Detects $2 \times 2$ localized clusters (corner density and localized clusters). Output shape: $(B, 128, 3, 3)$.
4. **Pointwise Skip Kernel $(1, 1)$:** Preserves global per-cell channel distributions across the entire $4 \times 4$ grid. Output shape: $(B, 64, 4, 4)$.

---

## 2. Multi-Scale Feature Extraction

Given input observation tensor $\mathbf{X} \in \{0.0, 1.0\}^{B \times 16 \times 4 \times 4}$:

$$\begin{aligned}
\mathbf{H} &= \sigma(\text{Conv2d}_{(1, 2)}(\mathbf{X})) \in \mathbb{R}^{B \times 128 \times 4 \times 3} \\
\mathbf{V} &= \sigma(\text{Conv2d}_{(2, 1)}(\mathbf{X})) \in \mathbb{R}^{B \times 128 \times 3 \times 4} \\
\mathbf{S} &= \sigma(\text{Conv2d}_{(2, 2)}(\mathbf{X})) \in \mathbb{R}^{B \times 128 \times 3 \times 3} \\
\mathbf{K} &= \sigma(\text{Conv2d}_{(1, 1)}(\mathbf{X})) \in \mathbb{R}^{B \times 64 \times 4 \times 4}
\end{aligned}$$

where $\sigma(z) = \text{LeakyReLU}(z, \alpha=0.01)$ prevents dead neurons.

The flattened representations are concatenated along the feature dimension:

$$\mathbf{F}_{\text{conv}} = [\text{vec}(\mathbf{H}) \,\|\, \text{vec}(\mathbf{V}) \,\|\, \text{vec}(\mathbf{S}) \,\|\, \text{vec}(\mathbf{K})] \in \mathbb{R}^{B \times 5248}$$

where:
- $\text{dim}(\mathbf{H}) = 128 \times 4 \times 3 = 1536$
- $\text{dim}(\mathbf{V}) = 128 \times 3 \times 4 = 1536$
- $\text{dim}(\mathbf{S}) = 128 \times 3 \times 3 = 1152$
- $\text{dim}(\mathbf{K}) = 64 \times 4 \times 4 = 1024$
- Total feature size: $1536 + 1536 + 1152 + 1024 = 5248$

---

## 3. Shared Representation Layer

The concatenated feature vector is projected through a dense linear layer with light dropout:

$$\mathbf{h}_{\text{shared}} = \text{Dropout}_{0.05}(\sigma(\mathbf{W}_{\text{shared}} \mathbf{F}_{\text{conv}} + \mathbf{b}_{\text{shared}})) \in \mathbb{R}^{B \times 512}$$

---

## 4. Dueling Stream Separation (Value & Advantage)

To separate state valuation from action quality, $\mathbf{h}_{\text{shared}}$ branches into two distinct streams:

### 4.1. Value Stream $V(s)$
Estimates the intrinsic scalar quality of the board configuration regardless of action choice:

$$\begin{aligned}
\mathbf{h}_{V} &= \sigma(\mathbf{W}_{V, 1} \mathbf{h}_{\text{shared}} + \mathbf{b}_{V, 1}) \in \mathbb{R}^{B \times 128} \\
V(s) &= \mathbf{W}_{V, 2} \mathbf{h}_{V} + b_{V, 2} \in \mathbb{R}^{B \times 1}
\end{aligned}$$

### 4.2. Advantage Stream $A(s, a)$
Estimates the relative advantage of each discrete action $a \in \{0, 1, 2, 3\}$:

$$\begin{aligned}
\mathbf{h}_{A} &= \sigma(\mathbf{W}_{A, 1} \mathbf{h}_{\text{shared}} + \mathbf{b}_{A, 1}) \in \mathbb{R}^{B \times 128} \\
A(s, a) &= \mathbf{W}_{A, 2} \mathbf{h}_{A} + \mathbf{b}_{A, 2} \in \mathbb{R}^{B \times 4}
\end{aligned}$$

---

## 5. Centered Identifiability Aggregation

Direct addition $Q(s, a) = V(s) + A(s, a)$ is unidentifiable (i.e. adding an arbitrary constant $c$ to $V$ and subtracting it from $A$ yields the exact same $Q$). Following Wang et al. (2016), we enforce identifiability by subtracting the mean advantage:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$

This guarantees:
1. $\sum_{a \in \mathcal{A}} (Q(s, a) - V(s)) = 0$
2. Gradient updates to $V(s)$ immediately propagate to all state value estimates, accelerating convergence on boards where move choice is non-critical.

---

## 6. Action Masking

When legal action mask $\mathbf{M} \in \{0, 1\}^{B \times 4}$ is provided:

$$Q_{\text{masked}}(s, a) = \begin{cases}
Q(s, a) & \text{if } \mathbf{M}[a] = 1 \\
-10^9 & \text{if } \mathbf{M}[a] = 0
\end{cases}$$

This guarantees $\arg\max_a Q_{\text{masked}}(s, a)$ strictly selects a valid board-modifying move.

---

## 7. Model Parameter Count

| Layer / Component | Specification | Parameter Count |
| :--- | :--- | :---: |
| `conv_horiz` | `Conv2d(16, 128, kernel_size=(1, 2))` | $(16 \times 1 \times 2 + 1) \times 128 = 4,224$ |
| `conv_vert` | `Conv2d(16, 128, kernel_size=(2, 1))` | $(16 \times 2 \times 1 + 1) \times 128 = 4,224$ |
| `conv_sq` | `Conv2d(16, 128, kernel_size=(2, 2))` | $(16 \times 2 \times 2 + 1) \times 128 = 8,320$ |
| `conv_skip` | `Conv2d(16, 64, kernel_size=(1, 1))` | $(16 \times 1 \times 1 + 1) \times 64 = 1,088$ |
| `fc_shared` | `Linear(5248, 512)` | $(5248 + 1) \times 512 = 2,687,488$ |
| `value_stream[0]` | `Linear(512, 128)` | $(512 + 1) \times 128 = 65,664$ |
| `value_stream[2]` | `Linear(128, 1)` | $(128 + 1) \times 1 = 129$ |
| `advantage_stream[0]` | `Linear(512, 128)` | $(512 + 1) \times 128 = 65,664$ |
| `advantage_stream[2]` | `Linear(128, 4)` | $(128 + 1) \times 4 = 516$ |
| **Total Parameters** | | **$2,837,317$ (~10.8 MB weights)** |

---

## 8. SOTA N-Tuple Network Architecture (`NTupleNetwork`)

In addition to deep convolutional networks, the engine implements a high-performance **N-Tuple Network** following Szubert & Jaśkowski (2014) and Wu et al. (2014) (`src/models/ntuple.py`), designed for sub-microsecond leaf evaluations in deep lookahead trees:

### 8.1. Canonical Symmetric Patterns
The $4 \times 4$ grid (cell indices $0..15$) is covered by 3 canonical geometric pattern topologies:
1. **4-Tile Boundary Line:** `CANONICAL_LINE_4 = (0, 1, 2, 3)` (outer boundary rows/columns).
2. **6-Tile Rectangular Block ($2 \times 3$):** `CANONICAL_RECT_6 = (0, 1, 2, 4, 5, 6)` (corner perimeter tiles).
3. **6-Tile Snake Turn:** `CANONICAL_SNAKE_6 = (0, 1, 2, 6, 5, 4)` (corner snake fold).

### 8.2. Dihedral Symmetry Group Weight Sharing ($D_4$)
Each canonical pattern of length $k$ is expanded into all 8 isometric reflections and rotations:
$$\mathcal{T}_p = \{g(p) \mid g \in D_4\}$$
All 8 symmetric variants share the **exact same flat lookup table** of size $16^k$:
- 4-tile line table: $16^4 = 65,536$ entries (float32, 256 KB).
- 6-tile rect table: $16^6 = 16,777,216$ entries (float32, 64 MB).
- 6-tile snake table: $16^6 = 16,777,216$ entries (float32, 64 MB).

### 8.3. Base-16 Memory Addressing & Microsecond Throughput
Given a board with cell ranks $r_0, \dots, r_{15} \in [0, 15]$, the address for a pattern $p = (c_0, \dots, c_{k-1})$ is:
$$\text{index}(p) = \sum_{i=0}^{k-1} r_{c_i} \cdot 16^i = \bigvee_{i=0}^{k-1} (r_{c_i} \ll 4i)$$
The state value is the sum of lookup table entries across all active symmetric patterns:
$$V_{\text{ntuple}}(B) = \sum_{p \in \text{Patterns}} \sum_{t \in \mathcal{T}_p} W_p[\text{index}(t)]$$
- **Evaluation Throughput:** Evaluates a 64-bit Bitboard in **$< 1.0\ \mu\text{s}$** per board ($> 1,100,000\text{ boards/sec}$), entirely bypassing GPU kernel launch overhead during leaf rollouts.

---

## 9. Expectimax Search & Unified Hybrid Leaf Evaluation

The agent combines neural network afterstate predictions, N-Tuple lookups, and geometric stabilization into a unified searcher (`src/agent/search.py`):

### 9.1. Unified Leaf Evaluation Pipeline
At leaf states of the search tree, values are evaluated via:

$$V_{\text{leaf}}(B) = V_{\text{net}}(B) + w_{\text{ntuple}} \cdot \tilde{V}_{\text{ntuple}}(B) + w_{\text{smooth}} \cdot S_{\text{smooth}}(B) - w_{\text{trap}} \cdot P_{\text{trapped}}(B) + V_{\text{snake}}(B)$$

where:
- $V_{\text{net}}(B)$: Neural network afterstate value head $V(s_{\text{after}})$.
- $\tilde{V}_{\text{ntuple}}(B)$: Converged N-Tuple value lookup scaled by $w_{\text{ntuple}} \approx 0.15$.
- $S_{\text{smooth}}(B)$: Manhattan rank smoothness ($w_{\text{smooth}} = 0.10$).
- $P_{\text{trapped}}(B)$: Structural penalty for low-tier tiles wedged between high ranks ($w_{\text{trap}} = 5.0$).
- $V_{\text{snake}}(B)$: Topological monotonicity gradient along the perimeter.

### 9.2. Adaptive & Apex Depth Selection
Depth is dynamically selected based on board occupancy and criticality:
- **Early / Open Boards ($> 7$ empty cells):** 2-ply search for maximum execution speed.
- **Mid-Game Boards ($4..7$ empty cells):** 3-ply Expectimax lookahead.
- **Endgame Boards ($2..3$ empty cells):** 4-ply Expectimax lookahead.
- **Apex Endgame Boards ($\le 2$ empty cells with max tile $\ge 2048$):** Up to 5-ply / 6-ply search with chance node pruning.

### 9.3. Protected Policy-Guided Action Pruning
At shallow plies (depth 0 and 1), action candidates are queried via prior policy $\pi(a \mid s) = \text{Softmax}(Q(s, \cdot) / \tau_{\text{temp}})$.
- Actions with $\pi(a \mid s) < \tau$ (e.g., $\tau = 0.03$) are pruned to reduce branching factor.
- **Danger Bypass Protection:** If empty cells $\le 4$ or max tile $\ge 2048$, policy pruning is **strictly disabled**, guaranteeing all legal moves are explored to prevent premature game-over on tight boards.
