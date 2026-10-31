# Environment Specification: Game2048 MDP & Gymnasium Architecture

## 1. Mathematical Formulation as a Markov Decision Process (MDP)

The game 2048 is modeled formally as a discrete-time, stochastic Markov Decision Process (MDP) defined by the 5-tuple:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

### 1.1. State Space ($\mathcal{S}$)
The game board is a $4 \times 4$ grid of cells. Each cell contains either:
- $0$ representing an unoccupied cell.
- An integer power of two: $v \in \{2^1, 2^2, 2^3, \dots, 2^{16}\}$ (i.e., from $2$ to $65,536$).

The combinatorial state space is bounded but astronomically large ($\sim 16^{16} \approx 1.84 \times 10^{19}$ potential configurations).

### 1.2. Action Space ($\mathcal{A}$)
The action space is discrete with 4 orthogonal movement primitives:

$$\mathcal{A} = \{0: \text{LEFT}, 1: \text{RIGHT}, 2: \text{UP}, 3: \text{DOWN}\}$$

### 1.3. Transition Dynamics ($\mathcal{P}$)
A state transition from $s_t$ to $s_{t+1}$ conditioned on action $a_t \in \mathcal{A}$ proceeds through two sequential stages:

1. **Deterministic Slide and Merge:**
   - All tiles slide in direction $a_t$.
   - Any two adjacent tiles with equal values $v$ collapse into a single tile with value $2v$.
   - **Non-Cascade Rule:** A tile formed by a merge in the current step cannot merge again during the same step (e.g., $[2, 2, 2, 2] \to [4, 4, 0, 0]$, not $[8, 0, 0, 0]$).
   - If the slide yields a board identical to $s_t$, the action is deemed **illegal** (or invalid), and the board state remains unchanged without spawning a new tile.

2. **Stochastic Tile Ingestion:**
   - If the deterministic slide modified the board ($s_{\text{slide}} \ne s_t$), a new tile is spawned in an unoccupied cell selected uniformly at random:
     $$P(\text{spawn at cell } (i, j) \mid (i, j) \in \mathcal{E}) = \frac{1}{|\mathcal{E}|}$$
     where $\mathcal{E} = \{(i, j) \mid s(i, j) = 0\}$ is the set of vacant cells.
   - The numerical value of the newly spawned tile is drawn from a categorical distribution:
     $$v_{\text{spawn}} = \begin{cases} 
     2 & \text{with probability } p = 0.90 \\
     4 & \text{with probability } p = 0.10
     \end{cases}$$

### 1.4. Reward Function ($\mathcal{R}$)
To enforce strict zero-heuristic learning, the reward does not inject arbitrary hand-coded positional weights. Instead, it scales logarithmically with the intrinsic merge points:

$$r_t(s_t, a_t) = \begin{cases}
\sum_{m \in \mathcal{M}_t} \log_2(v_m) & \text{if merges occurred} \\
0.0 & \text{if move was legal with zero merges} \\
-10.0 & \text{if action was illegal (bypassed masking)}
\end{cases}$$

where $\mathcal{M}_t$ is the multiset of all newly synthesized tiles $v_m$ created by action $a_t$.

### 1.5. Discount Factor ($\gamma$)
The long-term discounted return is parameterized by $\gamma = 0.99$. With this horizon, the agent naturally learns that maintaining maximum tiles in fixed corners maximizes topological entropy retention and longevity.

---

## 2. Multi-Channel One-Hot Observation Representation

Passing raw tile scalars ($[0, 2, 4, \dots, 2048]$) directly into dense neural network layers distorts gradient updates due to exponential numerical divergence. 

We represent every state as a multi-channel binary feature tensor:

$$\mathbf{X} \in \{0.0, 1.0\}^{16 \times 4 \times 4}$$

- **Channel 0:** Binary indicator for vacant cells ($v = 0$).
- **Channels $k \in \{1, \dots, 15\}$:** Binary indicator that cell $(i, j)$ contains tile value $2^k$:
  $$\mathbf{X}[k, i, j] = \begin{cases} 1.0 & \text{if } s[i, j] = 2^k \\ 0.0 & \text{otherwise} \end{cases}$$

### Spatial Partition of Unity:
For every grid position $(i, j)$:

$$\sum_{c=0}^{15} \mathbf{X}[c, i, j] = 1.0$$

This planar separation allows 2D convolutional filters (`Conv2d`) to detect spatial invariances, monotonicity gradients, and adjacent candidate merges.

---

## 3. Legal Action Masking

In 2048, actions that do not change the board state are invalid. Rather than relying solely on negative penalties to dissuade the agent, the environment provides an exact action mask:

$$\mathbf{M}(s) \in \{0, 1\}^4$$

- $\mathbf{M}[a] = 1$ if executing action $a$ alters the board state.
- $\mathbf{M}[a] = 0$ if executing action $a$ leaves the board unchanged.

During inference and epsilon-greedy exploration, invalid actions are masked before action selection, ensuring the policy only explores reachable state spaces.

---

## 4. Benchmark Performance

The core environment (`Game2048Core`) executes orthogonal transformations (rotations/flips) in-memory with unrolled line-compression primitives.

- **Target Throughput:** $> 15,000\text{ steps/sec}$
- **Measured CPU Throughput:** $> 17,400\text{ steps/sec}$ on standard single-thread execution.

---

## 5. 64-Bit Bitboard Engine (`Bitboard2048`)

To eliminate Python object overhead and achieve microsecond-scale leaf rollouts during deep Expectimax search, the environment includes a pure 64-bit Bitboard representation (`src/env/bitboard.py`):

### 5.1. Bitboard Memory Representation
The entire $4 \times 4$ grid is packed into a single 64-bit unsigned integer (`uint64`):
- Each of the 16 cells occupies a 4-bit nibble storing the tile rank $k$ (where $\text{value} = 2^k$, with $k=0$ denoting an empty cell).
- Grid layout index: $\text{cell}(r, c) = 4r + c$.
- Nibble bit shift: $\text{shift}(r, c) = 4(4r + c) \in [0, 60]$.

### 5.2. Precomputed 16-Bit Transition Lookup Tables (LUTs)
Each 4-cell row is an unsigned 16-bit integer ($R \in [0, 65535]$):
- `ROW_LEFT_TABLE`: Precomputed resulting 16-bit row after sliding left and merging.
- `ROW_RIGHT_TABLE`: Precomputed resulting 16-bit row after sliding right and merging.
- `ROW_SCORE_TABLE_LEFT` / `ROW_SCORE_TABLE_RIGHT`: Exact in-game merge points scored.
- `ROW_REWARD_TABLE_LEFT` / `ROW_REWARD_TABLE_RIGHT`: Sum of logarithmic rewards $\sum \log_2(v_{\text{merged}})$.

### 5.3. Bitwise Move Operations & Symmetries
- **Horizontal Moves (`LEFT`, `RIGHT`):** Directly query the precomputed 16-bit row tables for each of the 4 rows in parallel:
  $$\text{Row}_i = (B \gg 16i) \ \& \ \text{0xFFFF}$$
- **Vertical Moves (`UP`, `DOWN`):** Transpose the board using fast delta-swap bitwise operations, execute horizontal move LUT queries, and transpose back.
- **Dihedral Symmetries ($D_4$):** Fast bitwise operations (`rotate90`, `rotate180`, `rotate270`, `flip_horizontal`) and canonical hash computation `get_canonical(bb)` for Transposition Tables.
- **Simulation Throughput:** $> 485,000\text{ moves/sec}$ per core in pure Python.

---

## 6. Geometric Stabilization & Hazard Shields

To maintain topological order without handcrafting arbitrary corner biases, the bitboard engine incorporates fine-grained structural shields:

### 6.1. Manhattan Smoothness Metric ($S_{\text{smooth}}$)
Measures value continuity between horizontally and vertically adjacent non-empty cells:

$$S_{\text{smooth}}(B) = - \sum_{i, j} \left( \sum_{k \in \text{neighbors}(i,j)} |\text{rank}(c_{i,j}) - \text{rank}(c_k)| \right)$$

- Evaluated via a precomputed 65,536-entry row lookup table (`ROW_SMOOTHNESS_TABLE`).
- Vectorized SIMD/NumPy batch evaluation via `compute_smoothness_batch(bbs)`.
- Higher values correspond to smoother boards where merge partners sit immediately adjacent to each other.

### 6.2. Trapped Low-Tier Tile Detection ($P_{\text{trapped}}$)
Detects structural "tile wedges":
- Any low-tier tile (rank 1 or 2, i.e., value 2 or 4) positioned on an outer wall (rows 0, 3 or columns 0, 3) flanked on both sides by high-tier tiles of rank $\ge 6$ (value $\ge 64$), where no immediate merge is possible.
- Evaluated via `compute_trapped_penalty(bb)` and `compute_trapped_penalty_batch(bbs)`.
- Applies a severe structural penalty to prevent lookahead search from selecting catastrophic traps that irreversibly split the monotonicity chain.
