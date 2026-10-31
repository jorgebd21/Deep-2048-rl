# Training Pipeline & Numerical Stabilization Specification

## 1. Double Deep Q-Network (DDQN) Mathematical Formulation

Standard Deep Q-Learning utilizes the maximum estimated action-value for the next state:

$$y_t^{\text{DQN}} = r_t + (1 - d_t) \gamma \max_{a'} Q(s_{t+1}, a'; \theta^{-})$$

Because the same parameter set $\theta^{-}$ selects and evaluates the greedy action, Jensen's inequality and statistical noise induce systematic positive overestimation of $Q$-values, leading to sub-optimal policy convergence or value explosion.

Double DQN (van Hasselt et al., 2016) eliminates this overestimation by **decoupling action selection from action evaluation**:

$$\begin{aligned}
a^*_{t+1} &= \arg\max_{a' \in \mathcal{A}_{\text{legal}}(s_{t+1})} Q(s_{t+1}, a'; \theta_{\text{online}}, \mathbf{M}_{t+1}) \\
y_t^{\text{DDQN}} &= r_t + (1 - d_t) \gamma \, Q(s_{t+1}, a^*_{t+1}; \theta_{\text{target}})
\end{aligned}$$

where:
- $\theta_{\text{online}}$: Continuously updated parameters of the policy network.
- $\theta_{\text{target}}$: Periodically synchronized parameters of the target network.
- $\mathbf{M}_{t+1}$: Legal action mask for state $s_{t+1}$, guaranteeing that the policy never selects unreachable future states.

---

## 2. Loss Function & Importance Sampling Weighting

The temporal difference (TD) error for sample $i$ is:

$$\delta_i = y_i^{\text{DDQN}} - Q(s_i, a_i; \theta_{\text{online}})$$

To provide robust gradients that do not explode on large outlier TD errors, we utilize the **Smooth L1 Loss (Huber Loss)** with transition threshold $\beta_{\text{huber}} = 1.0$:

$$\text{Huber}(\delta_i) = \begin{cases} 
0.5 \cdot \delta_i^2 & \text{if } |\delta_i| \le 1.0 \\
|\delta_i| - 0.5 & \text{otherwise}
\end{cases}$$

Weighted by Prioritized Experience Replay (PER) Importance Sampling weights $w_i = (N \cdot P(i))^{-\beta} / \max(w)$:

$$\mathcal{L}(\theta_{\text{online}}) = \frac{1}{B} \sum_{i=1}^B w_i \cdot \text{Huber}\left( y_i^{\text{DDQN}} - Q(s_i, a_i; \theta_{\text{online}}) \right)$$

---

## 3. Numerical Stabilization Mechanisms

1. **Gradient Clipping:**
   $$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{1.0}{\|\mathbf{g}\|_2}\right)$$
   Prevents catastrophic weight divergence when large merge points or sudden terminal collapses occur.

2. **Target Network Synchronization:**
   - **Polyak Soft Update (Continuous):**
     $$\theta_{\text{target}} \leftarrow \tau \theta_{\text{online}} + (1 - \tau) \theta_{\text{target}} \quad (\tau = 0.005)$$
   - **Hard Update (Periodic fallback):**
     $$\theta_{\text{target}} \leftarrow \theta_{\text{online}} \quad \text{every } C = 1,000 \text{ optimization steps}$$

3. **Ampere TF32 & Pinned PCIe Pipeline:**
   Tensors transferred with `non_blocking=True` and operations executed under `torch.set_float32_matmul_precision('high')`.

---

## 4. Exploration Strategy & Action Masking

- **Epsilon Schedule:**
  $$\epsilon_t = \max\left( \epsilon_{\min}, \, \epsilon_{\text{start}} - \frac{t}{T_{\text{decay}}} (\epsilon_{\text{start}} - \epsilon_{\min}) \right)$$
  where $\epsilon_{\text{start}} = 1.0$, $\epsilon_{\min} = 0.01$, and $T_{\text{decay}} = 150,000$ steps.

- **Legal Action Sampling:**
  During exploratory steps ($u \sim \mathcal{U}(0, 1) < \epsilon$), actions are sampled uniformly from the active legal action mask:
  $$a_t \sim \mathcal{U}\left( \{a \in \{0, 1, 2, 3\} \mid \mathbf{M}(s_t)[a] = 1\} \right)$$
  Zero exploratory budget is wasted on invalid, non-moving actions.

---

## 5. Hyperparameter Reference

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| Discount factor ($\gamma$) | $0.99$ | Horizon discount for long-term survival |
| Learning rate ($\eta$) | $1 \times 10^{-4}$ | AdamW optimizer base learning rate |
| Weight decay | $1 \times 10^{-4}$ | $L_2$ weight regularization |
| Polyak coefficient ($\tau$) | $0.005$ | Soft target network update rate |
| Max gradient norm | $1.0$ | Gradient clipping threshold |
| Mini-batch size ($B$) | $512$ | Batch size saturating RTX 3060 Tensor Cores |
| Buffer capacity ($N$) | $100,000$ | Packed uint8 transitions in RAM |
| PER exponent ($\alpha$) | $0.6$ | Degree of prioritization |
| Initial IS exponent ($\beta_0$) | $0.4$ | Importance sampling bias compensation |
| Final IS exponent ($\beta_{\text{end}}$) | $1.0$ | Annealed linearly over $150,000$ steps |
| Warmup steps | $1,000$ | Random legal steps before training start |

---

## 6. Native Afterstate Value Learning & Expert Iteration

Traditional Q-learning directly estimates $Q(s, a)$, which must simultaneously predict the deterministic consequences of action $a$ and the stochastic outcome of nature's subsequent tile spawn.

In 2048, every transition cleanly decomposes into two sequential half-steps:
$$s_t \xrightarrow{a_t} s_{t, \text{after}} \xrightarrow{\text{spawn}} s_{t+1}$$

### 6.1. Afterstate Bellman Formulation
We parameterize the value function directly over the afterstate: $V(s_{\text{after}})$.
The Bellman optimality equation for afterstates becomes:

$$V^*(s_{\text{after}}) = \mathbb{E}_{\text{spawn}} \left[ \max_{a' \in \mathcal{A}_{\text{legal}}(s')} \left( r(s', a') + \gamma V^*(s'_{\text{after}}) \right) \right]$$

- **Variance Elimination:** The deterministic merge occurs before $s_{\text{after}}$, so immediate rewards $r$ and merges are deterministic. Nature's random tile placement occurs *after* $s_{\text{after}}$, so taking an expectation over nature's spawn distribution isolates environmental stochasticity from policy evaluation.
- **Dihedral Invariance ($D_4$):** By applying $D_4$ group transformations to afterstates during training, value predictions become strictly rotation- and reflection-invariant.

### 6.2. Expert Iteration Pipeline (`src/training/expert_iteration.py`)
Combines deep tree search with neural training:
1. Tree search (Expectimax with afterstate value evaluations) acts as an expert policy $\pi^*(s)$.
2. Full game trajectories $(s_{\text{after}}, r_{\text{next}}, s'_{\text{after}}, d)$ are recorded into the prioritized replay buffer.
3. Neural network weights are optimized to minimize TD error on expert transitions, monotonically boosting both the search evaluation accuracy and downstream tree quality.

---

## 7. N-Tuple Network TD($\lambda$) Self-Play Training (`src/training/train_ntuple.py`)

The SOTA N-Tuple Network is trained via afterstate Temporal Difference self-play directly on the 64-bit Bitboard representation:

### 7.1. Fast CPU Self-Play Engine
- Operates at $> 12,500\text{ game steps/sec}$ in pure Python/NumPy.
- Generates tens of millions of transitions in minutes without GPU bottlenecks.

### 7.2. Afterstate TD Error & Symmetrical Weight Updates
For transition $s \xrightarrow{a} s_{\text{after}} \xrightarrow{\text{spawn}} s' \xrightarrow{a^*} s'_{\text{after}}$:

$$\delta = r_{\text{next}} + (1 - d) \cdot \gamma \cdot V_{\text{ntuple}}(s'_{\text{after}}) - V_{\text{ntuple}}(s_{\text{after}})$$

The error $\delta$ is back-projected across all active tuple tables across all 8 $D_4$ symmetries:

$$\Delta W_p[\text{index}(t)] = \frac{\alpha}{|\mathcal{T}|} \cdot \delta$$

where $|\mathcal{T}| = 24$ (3 patterns $\times$ 8 symmetries), calibrating the total value update to exactly $\alpha \cdot \delta$.

### 7.3. Learning Rate Decay Schedule
The learning rate $\alpha$ decays linearly over training episodes:
$$\alpha_t = \alpha_{\text{init}} - (\alpha_{\text{init}} - \alpha_{\text{min}}) \frac{t}{T_{\text{total}}}$$
(e.g., $0.05 \to 0.005$ over 30,000 self-play episodes), ensuring rapid topological structure discovery early on and fine-grained asymptotic convergence in the late phase.
