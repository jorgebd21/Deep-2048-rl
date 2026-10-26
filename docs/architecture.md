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
