# Evaluation, Telemetry & Deployment Guide

This document details the evaluation protocol, telemetry logging via TensorBoard, interactive CLI visualization, and production deployment via ONNX export for the **Deep-2048-RL** project.

---

## 1. Telemetry & TensorBoard Metrics

The training pipeline integrates real-time telemetry logging through `torch.utils.tensorboard.SummaryWriter`. All events are recorded under the configured `runs/` directory and can be visualized live:

```bash
tensorboard --logdir runs
```

### 1.1. Monitored Metric Categories

#### A. Episode Performance (`episode/`)
- `episode/score`: Cumulative in-game score achieved per episode.
- `episode/steps`: Episode length (survival steps before terminal state).
- `episode/max_tile`: Highest tile value produced ($128, 256, 512, 1024, 2048$).
- `episode/reach_512`: Boolean indicator (1.0/0.0) recording milestone achievement.
- `episode/reach_1024`: Boolean indicator (1.0/0.0) recording milestone achievement.
- `episode/reach_2048`: Boolean indicator (1.0/0.0) recording milestone achievement.

#### B. Policy Optimization (`train/`)
- `train/loss`: Importance Sampling weighted Smooth L1 (Huber) loss.
- `train/mean_td_error`: Mean temporal difference magnitude $|\delta| = |y - Q(s, a)|$.
- `train/mean_q`: Mean predicted Q-value across the batch for chosen actions.
- `train/epsilon`: Linear exploration decay schedule value $\epsilon \in [\epsilon_{\min}, \epsilon_{\text{start}}]$.

#### C. Hardware & Simulation Throughput (`throughput/`)
- `throughput/env_steps_per_sec`: Environment transitions collected per second.
- `throughput/samples_per_sec`: Training samples optimized per second on NVIDIA RTX 3060.

#### D. Periodic Deterministic Evaluation (`eval/`)
- `eval/mean_score`: Mean score across $M$ deterministic evaluation episodes ($\epsilon=0$).
- `eval/max_tile`: Peak tile achieved across the evaluation batch.
- `eval/reach_512_rate`: Percentage of evaluation episodes reaching tile 512+.
- `eval/reach_1024_rate`: Percentage of evaluation episodes reaching tile 1024+.
- `eval/reach_2048_rate`: Percentage of evaluation episodes reaching tile 2048+.

---

## 2. Interactive Terminal Visualizer (`scripts/play.py`)

A real-time ANSI terminal visualizer enables inspecting agent policies live in the console with zero graphical overhead:

```bash
# Play using a trained checkpoint
python scripts/play.py --checkpoint checkpoints/checkpoint_best.pt --delay 0.1

# Play step-by-step (manual advance via Enter key)
python scripts/play.py --checkpoint checkpoints/checkpoint_best.pt --manual-step
```

### Visual Features:
- **ANSI Color Palette**: Distinct color coding for each tile power ($2$ to $4096$).
- **Live HUD Display**: Displays current score, max tile, step count, and action mask.
- **Q-Value Breakdown**: Shows the model's predicted Q-values across all 4 directions (`UP`, `DOWN`, `LEFT`, `RIGHT`), explicitly highlighting the chosen action and tagging illegal masked actions.

---

## 3. Monte Carlo Statistical Evaluation (`scripts/evaluate.py`)

To quantify agent capabilities across stochastic board spawns (90% $2$, 10% $4$), `scripts/evaluate.py` performs Monte Carlo rollouts with deterministic greedy action selection ($\epsilon=0$):

```bash
python scripts/evaluate.py --checkpoint checkpoints/checkpoint_best.pt --episodes 100 --save-json eval_results.json
```

### Metric Outputs:
- **Score Distribution**: Mean, standard deviation, median, min, max score.
- **Milestone Reach Percentages**: Exact empirical probability of reaching $256$, $512$, $1024$, and $2048$ tiles.
- **Tile Frequency Histogram**: Exact distribution of terminal maximum tiles.

---

## 4. ONNX Model Export & Verification (`scripts/export_onnx.py`)

For high-performance inference, edge deployment, or integration into C++/Rust runtimes, `Dueling2048Net` can be exported directly to standard **ONNX** format:

```bash
python scripts/export_onnx.py --checkpoint checkpoints/checkpoint_best.pt --output model_2048.onnx
```

### Technical Export Specifications:
- **Opset Version**: 17
- **Dynamic Axes**: Dynamic batching dimension `batch_size`:
  - Input: `observation` $\to (\text{batch\_size}, 16, 4, 4)$
  - Output: `q_values` $\to (\text{batch\_size}, 4)$
- **Numerical Parity**: Automated validation compares PyTorch native inference with ONNX Runtime (`CPUExecutionProvider` / `CUDAExecutionProvider`):
  $$\max_{i, a} |Q_{\text{PyTorch}}(s_i, a) - Q_{\text{ONNX}}(s_i, a)| < 10^{-5}$$
  Typical empirical max error observed: $< 10^{-7}$.
