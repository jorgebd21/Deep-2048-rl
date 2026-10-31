# Contributing to Deep-2048-RL

Thank you for your interest in contributing to **Deep-2048-RL**! This document outlines our development workflows, coding standards, testing requirements, and contribution guidelines.

---

## 1. Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it to ensure a welcoming and productive environment.

---

## 2. Core Architectural Principles & Directives

Before writing any code, please review the strict engineering constraints governing this project:

1. **Zero Handcrafted Heuristics:**
   Never introduce hardcoded move biases, tile placement weights (e.g., "always keep 2048 in bottom-left"), or rule-based heuristics into the core decision engine. All topological corner-packing strategies and monotonicity gradients must emerge purely from:
   - The log-scaled fusion reward: $r_t = \sum \log_2(v_{\text{merged}})$.
   - Long-term expected discounted return ($\gamma = 0.99$).
   - Strict action masking of illegal moves.
2. **Headless & In-Memory Execution:**
   All simulation primitives, neural operations, and tree searches must execute in-memory as vectorized tensor or bitwise operations. Never integrate GUI windows, browser automation, or OpenCV screen scraping.
3. **English Documentation Standard:**
   All code comments, docstrings, type annotations, commit messages, and documentation files must be written in professional, concise, and academically sound English.

---

## 3. Local Development Environment

### 3.1. Prerequisites
- **Python:** `>= 3.11` (CPython 3.11.9 recommended).
- **Git:** Standard git CLI.
- **CUDA Toolkit (Optional):** CUDA 12.x for NVIDIA Ampere / RTX 3060 acceleration.

### 3.2. Setup Instructions

```bash
# Fork and clone repository
git clone https://github.com/lobero21/deep-2048-rl.git
cd deep-2048-rl

# Initialize virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies in editable mode with development tools
pip install -r requirements.txt
pip install -e .[dev,gpu]
```

---

## 4. Testing Protocol

We maintain a strict **100% test pass rate** across all automated test suites. Any proposed pull request must pass all tests without regressions.

### Running Tests

```bash
# Execute full test suite (120 tests)
pytest tests/ -v

# Run tests with test coverage reporting
pytest --cov=src tests/

# Execute a specific test module
pytest tests/test_ntuple.py -v
pytest tests/test_search.py -v
```

### Adding New Tests
When implementing a new feature or fixing a bug:
1. Place unit and integration tests under [`tests/`](tests/).
2. Name test files with prefix `test_*.py` and classes with prefix `Test*`.
3. Include edge cases (e.g., full board deadlocks, tile spawn distributions, numerical stability with TF32).

---

## 5. Coding & Style Standards

- **Code Formatting:** Adhere to PEP 8 standards. We recommend formatting code with `black` and linting with `flake8`.
- **Type Annotations:** All public classes, functions, and method signatures must include comprehensive Python type annotations (`from typing import Optional, Tuple, ...`).
- **Docstrings:** Use Google-style or NumPy-style docstrings describing function purpose, parameters, return types, and potential raised exceptions.
- **Hardware Agnosticism:** Always support CPU fallback gracefully when CUDA is not present:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```

---

## 6. Commit Message Convention (Conventional Commits)

We enforce the **Conventional Commits** specification. Every commit must follow this format:

```text
<type>(<scope>): <short imperative description>

[optional detailed body explaining context and mathematical justification]
```

### Allowed Types
- `feat`: A new feature (e.g., bitboard engine, dueling network layers, PER).
- `fix`: Bug fix in mathematical formulas, off-by-one errors, or array indexing.
- `perf`: Code changes improving step throughput or GPU tensor transfer latency.
- `test`: Adding or refactoring unit/integration tests in `pytest`.
- `docs`: Additions or modifications to `README.md`, `docs/*.md`, or docstrings.
- `refactor`: Code restructuring without functional changes.
- `chore`: Dependency updates, `.gitignore`, or tooling setup.

### Examples of High-Quality Commits
```bash
git commit -m "feat(ntuple): implement TD(lambda) training pipeline and calibrated hybrid search"
git commit -m "perf(cuda): enable TF32 matmul and cudnn backends for Ampere target"
git commit -m "test(core): add test cases for multi-tile cascade collisions"
git commit -m "fix(eval): add robust checkpoint path resolver and ensure weights are loaded"
```

---

## 7. Pull Request & Review Process

1. **Create a Feature Branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Commit Your Changes:** Keep commits discrete and logically grouped following Conventional Commits.
3. **Verify Locally:**
   - Run `pytest tests/ -v` and ensure all 120+ tests pass.
   - Run `scripts/benchmark_hardware.py` if your changes touch performance-critical code.
4. **Push and Open a PR:**
   - Submit your pull request against the `main` branch.
   - Complete the Pull Request template detailing what was changed and attaching test evidence.
