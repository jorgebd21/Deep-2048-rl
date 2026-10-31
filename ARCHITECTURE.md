# System Architecture & Technical Specifications

This document provides a comprehensive technical overview of the software architecture, data pipelines, concurrency models, and algorithmic interactions of the **Deep-2048-RL** autonomous system.

---

## 1. System Overview & Component Decomposition

The system is organized into modular packages located under [`src/`](src/), strictly separating environmental simulation, neural representations, experience storage, heuristic/tree search, and optimization loops.

```text
src/
├── env/           # Deterministic simulation, 64-bit Bitboard, and Gymnasium wrappers
├── models/        # Dueling Double Deep Q-Network and SOTA N-Tuple architectures
├── memory/        # Prioritized Experience Replay (PER) backed by a binary SumTree
├── agent/         # Expectimax lookahead search engine and policy-guided pruning
├── training/      # DDQN Agent, Polyak updates, Afterstate Expert Iteration, and Trainer
└── utils/         # Ampere TF32 configuration, ANSI terminal visualizer, and checkpointing
```

### 1.1. Core Package Responsibilities

| Package | Key Module | Core Responsibilities & Algorithmic Foundations |
| :--- | :--- | :--- |
| **`src/env`** | [`bitboard.py`](src/env/bitboard.py)<br>[`core.py`](src/env/core.py)<br>[`vectorized.py`](src/env/vectorized.py) | - In-memory 64-bit unsigned integer (`uint64`) board representation with 4-bit nibbles.<br>- Precomputed 16-bit lookup tables (LUTs, 65,536 entries) for $O(1)$ row moves ($> 485,000$ moves/s).<br>- Pure NumPy orthogonal slide primitives with inlined non-cascade collision logic.<br>- Synchronous parallel simulation across $N$ environments with auto-resetting sub-workers. |
| **`src/models`** | [`dueling_dqn.py`](src/models/dueling_dqn.py)<br>[`ntuple.py`](src/models/ntuple.py) | - Multi-scale asymmetric convolutions: $(1, 2)$ horizontal, $(2, 1)$ vertical, $(2, 2)$ localized clusters, $(1, 1)$ pointwise channel skip.<br>- Decoupled value stream $V(s)$ and advantage stream $A(s, a)$ with centered identifiability: $Q(s, a) = V(s) + (A(s, a) - \frac{1}{\lvert \mathcal{A} \rvert}\sum A)$.<br>- SOTA N-Tuple Network featuring canonical 4-line, 6-rect, and 6-snake geometries with 8-fold Dihedral $D_4$ weight sharing ($< 1.0\ \mu\text{s}$ evaluation). |
| **`src/memory`** | [`replay_buffer.py`](src/memory/replay_buffer.py) | - Complete binary heap SumTree providing $O(\log N)$ priority updates and sampling.<br>- $64\times$ RAM compression storing states as compact `uint8` tile exponents ($< 180\text{ MB}$ for 100k transitions).<br>- Annealed Importance Sampling (IS) weights $w_i = (N \cdot P(i))^{-\beta}$ compensating for non-uniform sampling bias. |
| **`src/agent`** | [`search.py`](src/agent/search.py) | - Multi-ply Expectimax tree search evaluating deterministic player choices against stochastic chance nodes ($p=0.9$ for tile 2, $p=0.1$ for tile 4).<br>- Canonical Dihedral $D_4$ Transposition Tables caching sub-tree evaluations across isomorphic rotations.<br>- Protected Policy-Guided Action Pruning with automatic safety bypass in danger states ($\text{empty cells} \le 4$ or $\text{max tile} \ge 2048$). |
| **`src/training`** | [`agent.py`](src/training/agent.py)<br>[`trainer.py`](src/training/trainer.py)<br>[`expert_iteration.py`](src/training/expert_iteration.py) | - Double DQN Bellman targets decoupling action selection ($\theta_{\text{online}}$) from evaluation ($\theta_{\text{target}}$).<br>- Native Afterstate Value Learning: optimizing $V(s_{\text{after}})$ directly over deterministic post-slide states.<br>- Polyak soft parameter updates $\theta_{\text{target}} \leftarrow \tau \theta_{\text{online}} + (1 - \tau) \theta_{\text{target}}$. |
| **`src/utils`** | [`cuda_utils.py`](src/utils/cuda_utils.py)<br>[`checkpoint.py`](src/utils/checkpoint.py)<br>[`visualizer.py`](src/utils/visualizer.py) | - TensorFloat-32 (TF32) execution and pinned memory DMA transfers on NVIDIA Ampere (RTX 3060).<br>- Defensive checkpoint resolution supporting relative, absolute, and root-relative paths.<br>- ANSI 256-color terminal renderer with real-time HUD and live Q-value distributions. |

---

## 2. End-to-End Execution Flow

The sequence diagram below outlines the runtime lifecycle of an agent decision step, from environment state extraction to tree search, afterstate valuation, and asynchronous gradient optimization.

```mermaid
sequenceDiagram
    autonumber
    participant Env as Environment (Bitboard)
    participant Agent as Expectimax / Agent
    participant NTuple as N-Tuple (CPU LUTs)
    participant NN as Dueling2048Net (GPU TF32)
    participant PER as Replay Buffer (RAM)
    participant Opt as Optimizer (AdamW)

    Note over Env, Opt: Decision & Action Selection Phase
    Env->>Agent: Current State s & Legal Action Mask M(s)
    Agent->>Agent: Expand Legal Actions a in M(s)
    
    loop Expectimax Lookahead (Plies 1..k)
        Agent->>Env: Compute Deterministic Afterstates s_after = move(s, a)
        alt Shallow Ply & High-Throughput Leaf
            Agent->>NTuple: Base-16 D4 Lookup (s_after)
            NTuple-->>Agent: Value Score V_ntuple (< 1.0 µs)
        else Deep Ply / Terminal Head
            Agent->>NN: Forward Pass Tensor (B, 16, 4, 4)
            NN-->>Agent: Afterstate Value V_nn(s_after)
        end
        Agent->>Agent: Compute Chance Node Expectations: E = 0.9*V(s+2) + 0.1*V(s+4)
    end
    
    Agent-->>Env: Execute Best Action a*
    Env->>Env: Spawn Stochastic Tile (2 or 4) -> s_next, reward r, done d

    Note over Env, Opt: Experience Ingestion & Training Phase
    Env->>PER: Store Transition (s, a, r, s_next, d) as packed uint8
    
    opt Train Step Frequency Reached
        PER->>PER: SumTree Sample Priority Mini-Batch (B = 512)
        PER->>NN: Non-blocking Pinned DMA Copy (RAM -> VRAM)
        NN->>NN: Forward Pass Q(s, ·; θ_online) & Q(s', ·; θ_target)
        NN->>Opt: Compute Huber Loss with IS Weights
        Opt->>NN: Gradient Descent (AdamW) + Polyak Soft-Update (τ = 0.005)
    end
```

---

## 3. Data Representation & Mathematical Transformations

### 3.1. Discrete State Space & Observation Tensor
The game board is represented through three complementary tiers:
1. **64-bit Bitboard Integer (`uint64`):**
   $$\text{board} = \sum_{i=0}^{15} k_i \cdot 16^i, \quad k_i = \log_2(v_i) \in [0, 15]$$
   where $k_i = 0$ denotes an empty cell, and each 4-bit nibble represents the exponent of tile $v_i = 2^{k_i}$.
2. **Compact RAM Replay Tuple:**
   Stored as `np.uint8` arrays of shape `(4, 4)` taking 16 bytes per state, achieving a $64\times$ memory reduction compared to raw 32-bit floating point representations.
3. **Multi-Channel Convolutional Tensor:**
   Input tensor $\mathbf{X} \in \{0.0, 1.0\}^{B \times 16 \times 4 \times 4}$ enforcing the spatial partition of unity:
   $$\sum_{c=0}^{15} \mathbf{X}[b, c, i, j] = 1.0 \quad \forall (i, j) \in [0, 3]^2$$

### 3.2. Asymmetric Multi-Scale Convolutional Architecture
Standard square convolutional filters ($3 \times 3$ or $5 \times 5$) fail on $4 \times 4$ grids. The feature extraction backbone of `Dueling2048Net` uses asymmetric receptive fields aligned with candidate merge directions:

$$\begin{aligned}
\mathbf{H} &= \text{LeakyReLU}(\text{Conv2d}_{(1, 2)}(\mathbf{X})) \in \mathbb{R}^{B \times 128 \times 4 \times 3} && \text{(Horizontal row merges)} \\
\mathbf{V} &= \text{LeakyReLU}(\text{Conv2d}_{(2, 1)}(\mathbf{X})) \in \mathbb{R}^{B \times 128 \times 3 \times 4} && \text{(Vertical column merges)} \\
\mathbf{S} &= \text{LeakyReLU}(\text{Conv2d}_{(2, 2)}(\mathbf{X})) \in \mathbb{R}^{B \times 128 \times 3 \times 3} && \text{(Localized 2x2 clusters)} \\
\mathbf{K} &= \text{LeakyReLU}(\text{Conv2d}_{(1, 1)}(\mathbf{X})) \in \mathbb{R}^{B \times 64 \times 4 \times 4} && \text{(Pointwise channel skip)}
\end{aligned}$$

The concatenated representation:
$$\mathbf{F}_{\text{conv}} = [\text{vec}(\mathbf{H}) \,\|\, \text{vec}(\mathbf{V}) \,\|\, \text{vec}(\mathbf{S}) \,\|\, \text{vec}(\mathbf{K})] \in \mathbb{R}^{B \times 5248}$$
is projected through a shared dense layer ($5248 \to 512$) with light dropout ($0.05$) into decoupled value $V(s) \in \mathbb{R}^{B \times 1}$ and advantage $A(s, a) \in \mathbb{R}^{B \times 4}$ heads.

---

## 4. Concurrency, Memory & Hardware Pipeline

### 4.1. Hardware Optimization Strategy (NVIDIA Ampere RTX 3060)
1. **TensorFloat-32 (TF32):**
   Configured globally in `src/utils/cuda_utils.py`:
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   torch.set_float32_matmul_precision("high")
   ```
   Provides FP32 dynamic range with FP16 speed on 3rd-Gen Tensor Cores without gradient scaler instability.
2. **Pinned Host Memory & Non-Blocking DMA:**
   Sampled replay batches in RAM are converted to page-locked memory and transferred asynchronously over PCIe:
   ```python
   states_gpu = states_cpu.pin_memory().to(device, non_blocking=True)
   ```
   Yields a measured **+34.4% throughput speedup** during host-to-device transfers.
3. **Headless In-Memory Execution:**
   All components execute without GUI dependencies, rendering loops, or IPC bottlenecks.

---

## 5. Defensive Design & Error Handling

1. **Strict Action Masking:**
   Invalid actions are masked with a $-10^9$ penalty before Softmax or Argmax selection, guaranteeing that the policy never selects illegal moves.
2. **Protected Policy Pruning in Danger States:**
   When searching at depth, branches with $\pi(a \mid s) < \tau$ are pruned to save compute. However, if the board enters a danger state (empty cells $\le 4$ or max tile $\ge 2048$), policy pruning is strictly bypassed to evaluate all legal branches and prevent sudden game loss.
3. **Robust Checkpoint Resolution:**
   The checkpoint loader ([`src/utils/checkpoint.py`](src/utils/checkpoint.py)) resolves relative paths, root-anchored paths, and directory defaults defensively, ensuring weights are loaded safely before evaluation or fine-tuning begins.

---

## 6. Detailed Architectural References

For in-depth mathematical formulations, benchmark results, and deployment pipelines, consult the specialized documentation:
- [Environment Specification & Bitboard Design](docs/environment.md)
- [Dueling Convolutional & N-Tuple Theory](docs/architecture.md)
- [DDQN, Afterstates & PER Training Mechanics](docs/training_pipeline.md)
- [GPU Acceleration & Profiling Guide](docs/gpu_optimization.md)
- [Empirical Benchmarks & Speedups](docs/benchmarks.md)
- [Evaluation, Telemetry & Export Guide](docs/evaluation.md)
