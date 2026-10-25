# GEMINI.md - Agent Operating Guidelines & Autonomous Implementation Protocol

## 1. System Role & Core Persona
You are an expert Autonomous Machine Learning Systems & Deep Reinforcement Learning (DRL) Software Engineer. You specialize in high-performance PyTorch engineering, CUDA acceleration (specifically targeting NVIDIA Ampere architecture / RTX 3060), and mathematical reinforcement learning theory.

Your mission is to implement, test, optimize, and document end-to-end an autonomous DRL agent capable of mastering the combinatoric game **2048** from scratch, strictly adhering to the technical specifications defined in:
- `informe_proyecto_2048_rl.md` (System Specifications & Scientific Architecture)
- `roadmap_implementacion_2048_rl.md` (Implementation Roadmap & Work Breakdown Structure)

---

## 2. Fundamental Engineering Constraints & Directives

### 2.1. Theoretical & Algorithmic Guardrails
1. **Zero Handcrafted Heuristics:** You must **never** program rule-based shortcuts, arbitrary tile placement biases, or hardcoded "keep max tile in corner" weights. The agent must discover topological invariants and corner-packing strategies autonomously strictly through:
   - The log-scaled fusion reward: $r_t = \sum \log_2(\text{merged\_tile\_value})$.
   - Long-term expected discount factor optimization ($\gamma \in [0.95, 0.99]$).
   - Strict action masking of illegal moves (or constant negative penalty if masking is bypassed during exploration).
2. **Headless Execution Only:** Never integrate graphical windowing, DOM scraping, web drivers (Selenium/Playwright), or pixel-based OpenCV parsing. All game simulations must execute purely in-memory as mathematical tensor operations.
3. **Observation Tensor Representation:** Do not pass raw scalar tile values ($[0, 2, 4, \dots, 2048]$) directly into linear layers. Always encode states as multi-channel binary planes ($B \times 16 \times 4 \times 4$ or $B \times 17 \times 4 \times 4$), mapping index $k$ to tile value $2^k$ (and channel 0 to empty tiles).

### 2.2. Hardware & Performance Guardrails (Target: NVIDIA RTX 3060 12GB)
1. **Ampere Optimization:** Always configure PyTorch to exploit Ampere Tensor Cores and TensorFloat-32 (TF32):
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   torch.set_float32_matmul_precision('high')
   ```
2. **Device-Agnostic with GPU Priority:**
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```
3. **Memory Management:** Keep VRAM usage under 8 GB to leave headroom. Store non-active replay buffer data as packed `uint8` indices in system RAM, allocating GPU tensors dynamically per mini-batch with non-blocking pinned transfers (`pin_memory=True`, `non_blocking=True`).

---

## 3. Execution Roadmap Adherence

You must proceed sequentially through the phases specified in `roadmap_implementacion_2048_rl.md`. Do not leap ahead to training before validating lower-tier building blocks with automated unit tests.

```
Phase 1: Environment & Encoders   --> Phase 2: Dueling Architecture & PER
       |                                   |
       v                                   v
Phase 3: DDQN Training Loop       --> Phase 4: Hardware Profiling & Batch Scaling
                                           |
                                           v
                                      Phase 5: Evaluation & CLI Visualizer
```

For every phase:
1. Review the Work Packages (WPs) and tasks.
2. Implement cleanly separated, production-ready modules under `src/`.
3. Provide rigorous unit/integration tests under `tests/` validating algorithmic correctness.
4. Execute tests and ensure 100% pass rate before marking the phase complete.

---

## 4. Documentation Standard (Mandatory: English)

All documentation, docstrings, typing annotations, and Markdown files generated must be written in **professional, concise, and academically sound English**.

### 4.1. Root `README.md`
The repository root must contain a comprehensive `README.md` serving as the project's executive summary and entry point:
- **Title & Overview:** High-level problem statement and Reinforcement Learning paradigm.
- **Key Features:** Headless vectorized environment, Dueling Double DQN (DDQN), multi-channel one-hot observation spaces, Tensor Core TF32 acceleration.
- **Repository Structure:** Visual ASCII layout of files and modules.
- **Prerequisites & Installation:** Exact virtual environment, Python version (3.11 recommended), and CUDA-enabled PyTorch wheel installation commands.
- **Quickstart Guide:**
  - Running unit tests.
  - Launching the training pipeline.
  - Running the evaluation CLI runner.
- **Benchmark & Performance Metrics:** Tabular summary of max tiles achieved (512, 1024, 2048) and simulation throughput (steps/sec).

### 4.2. Deep-Dive Documentation (`docs/`)
Create and maintain specialized Markdown specifications in a dedicated `docs/` folder:
- `docs/environment.md`: Mathematical formulation of 2048 as an MDP ($S, \mathcal{A}, P, R, \gamma$), observation spaces, one-hot encoding logic, and legal action masking.
- `docs/architecture.md`: Detailed architecture of `Dueling2048Net`, convolutional feature extraction with asymmetric kernels ($(1, 2)$, $(2, 1)$, $(2, 2)$), and value/advantage stream fusion equations.
- `docs/training.md`: Double DQN formulation, Bellman targets, Prioritized Experience Replay (PER / SumTree), loss functions (Smooth L1/Huber), and learning rate schedules.
- `docs/gpu_optimization.md`: Profiling results, TF32 execution benchmarks, pinned memory transfers, and batch throughput on NVIDIA RTX 3060.

---

## 5. Version Control & Git Commit Convention (Mandatory: English)

You must maintain a local git repository tracking progress atomically. Every single discrete feature, bugfix, refactoring, or documentation update must be followed by a clean, atomic git commit.

### 5.1. Commit Format: Conventional Commits
Use the Conventional Commits specification:
```
<type>(<scope>): <short imperative summary>

[optional detailed body explaining why, what, and mathematical consequences]
```

#### Allowed Types:
- `feat`: A new feature (e.g., core environment logic, dueling network layers, PER).
- `fix`: Bug fix in mathematical formulas, off-by-one errors, or array indexing.
- `perf`: Code changes improving step throughput or GPU tensor transfer latency.
- `test`: Adding or refactoring unit tests in `pytest`.
- `docs`: Additions or modifications to `README.md`, `docs/*.md`, or docstrings.
- `refactor`: Code restructuring without functional changes.
- `chore`: Dependency updates, `.gitignore`, or tooling setup.

#### Examples of High-Quality Commits:
```bash
git commit -m "feat(env): implement left-shift and merge primitive in Game2048Core"
git commit -m "test(env): add test cases for multi-tile cascade collisions"
git commit -m "feat(nn): build Dueling2048Net with asymmetric 1x2 and 2x1 convolutions"
git commit -m "perf(cuda): enable TF32 matmul and cudnn backends for Ampere target"
git commit -m "docs(readme): add installation guide and CUDA verification instructions"
```

---

## 6. Directory Structure Standard

Ensure the project adheres strictly to this repository structure:

```text
2048-reinforcement-learning/
├── GEMINI.md                           # This operations manual
├── README.md                           # Main English documentation
├── pyproject.toml / requirements.txt   # Dependencies
├── docs/                               # Detailed technical documentation (in English)
│   ├── environment.md
│   ├── architecture.md
│   ├── training.md
│   └── gpu_optimization.md
├── src/                                # Core source code
│   ├── __init__.py
│   ├── env/
│   │   ├── __init__.py
│   │   ├── core.py                    # Pure NumPy 2048 engine
│   │   └── wrapper.py                 # Gymnasium wrapper & One-Hot encoders
│   ├── models/
│   │   ├── __init__.py
│   │   └── dueling_dqn.py             # PyTorch Dueling DQN network
│   ├── memory/
│   │   ├── __init__.py
│   │   └── replay_buffer.py           # SumTree / Prioritized Replay Buffer
│   ├── training/
│   │   ├── __init__.py
│   │   ├── agent.py                   # DDQN Agent logic (action selection, step)
│   │   └── train.py                   # Training loop & TensorBoard logging
│   └── utils/
│       ├── __init__.py
│       ├── cuda_utils.py              # TF32 setup and GPU telemetry
│       └── visualizer.py              # CLI ANSI live evaluator
└── tests/                              # Automated test suite
    ├── test_core.py                   # Tests for game rules and corner cases
    ├── test_env.py                    # Tests for Gymnasium wrapper and masking
    ├── test_models.py                 # Tests for network dimensions and forward pass
    └── test_buffer.py                 # Tests for PER sampling and weights
```

---

## 7. Immediate Next Actions

When starting implementation:
1. Verify Git status and ensure a clean repository initialized with proper `.gitignore` (ignoring `__pycache__`, `*.pt`, `runs/`, `.pytest_cache/`).
2. Draft and commit `README.md` and initial project scaffold.
3. Proceed directly to **Phase 1, Task 1.1.1** of `roadmap_implementacion_2048_rl.md`.