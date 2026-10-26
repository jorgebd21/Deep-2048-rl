# Deep-2048-RL: Autonomous Deep Reinforcement Learning for 2048

An enterprise-grade, high-performance Deep Reinforcement Learning (DRL) system designed to master the combinatoric game **2048** from scratch without handcrafted heuristics, featuring Dueling Double Deep Q-Networks (DDQN), Prioritized Experience Replay (PER), multi-channel one-hot state representations, and hardware acceleration optimized for NVIDIA Ampere architectures (RTX 3060).

---

## Key Features

- **Zero Handcrafted Heuristics:** The agent discovers corner-packing strategies, tile monotonicity, and cascading merges autonomously purely through logarithmic fusion rewards ($r = \log_2(\text{merge\_value})$) and Bellman return maximization ($\gamma = 0.99$).
- **Headless In-Memory Simulation:** Pure Python/NumPy game core executing $> 17,500\text{ steps/s}$ with inlined integer sliding and zero intermediate heap allocations, fully compatible with the Gymnasium API (`gymnasium.Env`).
- **Synchronous Vectorized Engine:** `VectorizedGame2048Env` simulating $N$ concurrent board states simultaneously with auto-resetting sub-environments.
- **Multi-Channel Tensor Representation:** Multi-plane binary state encoding ($\mathbf{X} \in \mathbb{R}^{B \times 16 \times 4 \times 4}$) that maps tile powers $2^k$ to discrete convolutional channels with strict partition of unity.
- **Action Masking:** Real-time generation of legal action masks preventing invalid moves from degrading policy gradient estimates.
- **Dueling Double DQN Architecture:** Decoupled value $V(s)$ and advantage $A(s, a)$ streams driven by asymmetric multi-scale convolutional kernels ($(1, 2)$, $(2, 1)$, $(2, 2)$, $(1, 1)$) with centered identifiability aggregation.
- **Prioritized Experience Replay (PER):** $\mathcal{O}(\log N)$ SumTree buffer storing states as compact `uint8` exponents in RAM ($< 180\text{ MB}$ for $100,000$ transitions) with Importance Sampling (IS) weights.
- **NVIDIA Ampere Acceleration:** Native support for TensorFloat-32 (TF32) matmul/cudnn backends, non-blocking pinned memory DMA transfers (+34.4% PCIe transfer speedup), and throughput exceeding $166,000\text{ samples/s}$ on RTX 3060.
- **Telemetry & Production Export:** Real-time TensorBoard telemetry, ANSI terminal playback visualizer, Monte Carlo evaluator, and ONNX model export with parity verification ($< 10^{-7}$ error).

---

## Repository Structure

```text
Deep-2048-rl/
├── GEMINI.md                           # System operations & engineering protocol
├── README.md                           # Executive summary and documentation
├── pyproject.toml                      # Packaging & test specifications
├── requirements.txt                    # Direct Python dependencies
├── docs/                               # Detailed technical documentation
│   ├── environment.md                  # MDP formulation, observation spaces & masking
│   ├── architecture.md                 # Dueling DQN network & convolutional streams
│   ├── training_pipeline.md            # Double DQN, Bellman targets, PER & Huber loss
│   ├── gpu_optimization.md             # TF32 profiling & CUDA pipeline scaling
│   ├── benchmarks.md                   # Empirical throughput & latency benchmarks
│   └── evaluation.md                   # Telemetry, CLI visualizer & ONNX deployment
├── scripts/                            # Standalone operational tools & runners
│   ├── benchmark_hardware.py           # Hardware throughput & PCIe profiling suite
│   ├── evaluate.py                     # Monte Carlo statistical evaluator
│   ├── play.py                         # Interactive ANSI terminal viewer & HUD
│   └── export_onnx.py                  # ONNX export & parity validation CLI
├── src/                                # Core production source code
│   ├── __init__.py
│   ├── env/                            # Game simulation and Gymnasium wrappers
│   │   ├── __init__.py
│   │   ├── core.py                     # Pure NumPy 2048 high-speed engine
│   │   ├── wrapper.py                  # Gymnasium wrapper & One-Hot encoders
│   │   └── vectorized.py               # Vectorized multi-environment runner
│   ├── models/                         # Neural network architectures
│   │   ├── __init__.py
│   │   └── dueling_dqn.py              # PyTorch Dueling DQN network & ONNX exporter
│   ├── memory/                         # Experience replay structures
│   │   ├── __init__.py
│   │   └── replay_buffer.py            # SumTree / Prioritized Replay Buffer
│   ├── training/                       # Agent logic and training loop
│   │   ├── __init__.py
│   │   ├── agent.py                    # DDQN Agent (epsilon-greedy, policy/target nets)
│   │   ├── trainer.py                  # Trainer orchestrator with TensorBoard logging
│   │   └── train.py                    # Main training CLI entrypoint
│   └── utils/                          # Hardware helpers and CLI viewer
│       ├── __init__.py
│       ├── cuda_utils.py               # TF32 setup & GPU telemetry
│       └── visualizer.py               # ANSI color terminal renderer & HUD
└── tests/                              # Automated pytest suite (51/51 passed)
    ├── test_core.py                    # Core game mechanics & collision edge-cases
    ├── test_env.py                     # Gymnasium API & one-hot encoding tests
    ├── test_models.py                  # Neural network dimensions & forward pass tests
    ├── test_buffer.py                  # PER SumTree sampling & IS weight tests
    ├── test_training.py                # DDQN Bellman updates & overfitting tests
    ├── test_vectorized.py              # Parallel environment & batch replay tests
    ├── test_evaluation.py              # ANSI rendering & Monte Carlo evaluator tests
    └── test_export.py                  # ONNX export & runtime parity tests
```

---

## Prerequisites & Installation

### System Requirements
- **OS:** Windows 10/11 or Linux (Ubuntu 22.04+)
- **Python:** 3.11 recommended (CPython 3.11.9 verified)
- **GPU (Optional but recommended):** NVIDIA RTX 3060 (12GB VRAM, Ampere `sm_86`) or higher

### Virtual Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## CLI Command Suite & Usage Guide

### 1. Launch DDQN Training Pipeline
Train the Double Dueling DQN agent with Ampere TF32 acceleration and TensorBoard logging:

```bash
# Standard training run (2000 episodes, batch size 512)
python src/training/train.py --episodes 2000 --batch-size 512 --checkpoint-dir checkpoints

# Monitor live training progress via TensorBoard
tensorboard --logdir runs
```

### 2. Live Interactive ANSI Terminal Visualizer
Watch the agent play in real-time inside your console with color-coded tiles and live Q-value analysis:

```bash
# Automated playback with 100ms step delay
python scripts/play.py --checkpoint checkpoints/checkpoint_best.pt --delay 0.1

# Manual step-by-step playback (press Enter for each action)
python scripts/play.py --checkpoint checkpoints/checkpoint_best.pt --manual-step
```

### 3. Statistical Monte Carlo Evaluation
Run deterministic evaluation rollouts ($\epsilon=0$) to measure score distribution and milestone reach rates:

```bash
python scripts/evaluate.py --checkpoint checkpoints/checkpoint_best.pt --episodes 100 --save-json eval_metrics.json
```

### 4. Hardware Profiling & Latency Benchmarks
Profile environment stepping throughput, PCIe memory transfers, and GPU compute performance on your machine:

```bash
python scripts/benchmark_hardware.py
```

### 5. ONNX Export & Parity Verification
Export the trained model to ONNX format with dynamic batch sizes and verify numerical parity with ONNX Runtime:

```bash
python scripts/export_onnx.py --checkpoint checkpoints/checkpoint_best.pt --output model_2048.onnx
```

### 6. Automated Test Suite Execution
Run the full automated test suite verifying all 51 test cases:

```bash
pytest tests/ -v
```

---

## Empirical Benchmark Results (NVIDIA RTX 3060 12GB)

### Environment Simulation Throughput
| Parallel Workers ($N$) | Simulation Throughput (steps/sec) | Speedup vs $N=1$ |
| :--- | :--- | :--- |
| **$N = 1$ (Single Env)** | 13,769.2 steps/sec | 1.00x |
| **$N = 8$ (Vectorized)** | 16,070.4 steps/sec | 1.17x |
| **$N = 16$ (Vectorized)** | 16,892.3 steps/sec | 1.23x |
| **$N = 32$ (Vectorized)** | 17,358.7 steps/sec | 1.26x |
| **$N = 64$ (Vectorized)** | **17,517.8 steps/sec** | **1.27x** |

### Host-to-Device (PCIe) Transfer Speedup
| Mini-batch Size | Unpinned Transfer | Pinned Transfer | PCIe Transfer Speedup |
| :--- | :--- | :--- | :--- |
| **$B = 256$** | 0.0314 ms | **0.0234 ms** | **+34.4%** |
| **$B = 512$** | 0.0506 ms | **0.0423 ms** | **+19.4%** |
| **$B = 1024$** | 0.0836 ms | 0.0837 ms | Baseline parity |

### Neural Optimization Latency & Throughput (Ampere TF32)
| Mini-batch Size | Forward Eval Latency | Full Step (Fwd+Bwd+Opt) | Optimization Throughput | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- |
| **$B = 256$** | 0.856 ms | 3.220 ms | 79,494.9 samples/s | 76.0 MB |
| **$B = 512$** | 0.877 ms | **3.074 ms** | **166,574.8 samples/s** | 91.8 MB |
| **$B = 1024$** | 1.297 ms | 4.162 ms | 246,009.7 samples/s | 128.0 MB |

---

## Technical Documentation Index

For in-depth mathematical formulations and architectural details, refer to:
- [Environment Specification & MDP Formulation (`docs/environment.md`)](file:///c:/Users/lober/Desktop/Proyectos/Deep-2048-rl/docs/environment.md)
- [Dueling Convolutional Architecture (`docs/architecture.md`)](file:///c:/Users/lober/Desktop/Proyectos/Deep-2048-rl/docs/architecture.md)
- [DDQN Training Pipeline & Huber Loss (`docs/training_pipeline.md`)](file:///c:/Users/lober/Desktop/Proyectos/Deep-2048-rl/docs/training_pipeline.md)
- [GPU Acceleration & Memory Optimizations (`docs/gpu_optimization.md`)](file:///c:/Users/lober/Desktop/Proyectos/Deep-2048-rl/docs/gpu_optimization.md)
- [Hardware Profiling & Benchmarks (`docs/benchmarks.md`)](file:///c:/Users/lober/Desktop/Proyectos/Deep-2048-rl/docs/benchmarks.md)
- [Evaluation, Telemetry & Export Guide (`docs/evaluation.md`)](file:///c:/Users/lober/Desktop/Proyectos/Deep-2048-rl/docs/evaluation.md)

---

## License
MIT License.
