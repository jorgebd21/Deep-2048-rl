# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-31

### Added
- **64-Bit Bitboard Engine (`Bitboard2048`):** Pure Python single-integer `uint64` board representation with 4-bit nibbles and precomputed 16-bit lookup tables (LUTs) for moves, scores, and log-rewards, delivering $> 485,000\text{ moves/s}$.
- **SOTA N-Tuple Network:** High-throughput geometric feature lookup tables with canonical 4-line, 6-rect, and 6-snake patterns, 8-fold Dihedral $D_4$ weight sharing, and sub-microsecond leaf evaluations ($< 1.0\ \mu\text{s}$ per board, $> 1,100,000\text{ boards/s}$).
- **N-Tuple TD($\lambda$) Self-Play Training Pipeline:** Pure CPU self-play training engine running at $> 12,500\text{ steps/s}$ with afterstate Bellman updates $\delta = r_{\text{next}} + (1 - d)\gamma V(s'_{\text{after}}) - V(s_{\text{after}})$ and linear $\alpha$ decay.
- **Expectimax Lookahead Tree Search:** Multi-ply tree search engine supporting 2-ply through 4-ply adaptive search, and apex 5-ply / 6-ply endgame depth with canonical $D_4$ Transposition Tables and chance node probability pruning ($p < 0.001$).
- **Protected Policy-Guided Action Pruning:** Policy prior pruning at shallow plies ($\pi(a \mid s) < \tau$) with automatic safety bypass in danger states ($\text{empty cells} \le 4$ or $\text{max tile} \ge 2048$).
- **Bitwise Geometric Shields:** Precomputed 65,536-entry Manhattan Smoothness ($S_{\text{smooth}}$) and Trapped Low-Tier Wedge Detection ($P_{\text{trapped}}$) LUTs.
- **Native Afterstate Value Learning & Expert Iteration:** Decomposed transition modeling ($s \to s_{\text{after}} \to s'$) learning $V(s_{\text{after}})$ directly to eliminate environment spawn stochasticity from policy evaluations.
- **Dueling Double Deep Q-Network (`Dueling2048Net`):** Asymmetric multi-scale convolutional architecture with $(1, 2)$, $(2, 1)$, $(2, 2)$, and $(1, 1)$ kernels, decoupled value $V(s)$ and advantage $A(s, a)$ streams, and centered identifiability aggregation.
- **Prioritized Experience Replay (PER):** Complete binary heap SumTree with $O(\log N)$ priority updates, Importance Sampling (IS) weights, and compact `uint8` RAM exponent compression ($64\times$ memory reduction).
- **Synchronous Vectorized Gym Environment (`VectorizedGame2048Env`):** Multi-environment runner for parallel simulation and batched replay ingestion.
- **Interactive ANSI Terminal Visualizer (`scripts/play.py`):** Real-time console renderer with 256-color tile styling, live HUD, and step-by-step playback.
- **Hardware Benchmarking Suite (`scripts/benchmark_hardware.py`):** Multi-tier benchmarking for CPU simulation throughput, host-to-device PCIe latency, and GPU optimization throughput.
- **ONNX Model Export (`scripts/export_onnx.py`):** Graph export with dynamic batch axes and verification against PyTorch runtime ($< 10^{-7}$ numerical tolerance).
- **Automated Pytest Suite:** 120 unit and integration tests covering environment mechanics, bitboard parity, neural models, replay buffers, tree search, smoothness shields, and training loops.

### Changed
- Decoupled Bellman action selection from action evaluation using Double DQN targets.
- Standardized observation encoding to multi-channel binary planes ($B \times 16 \times 4 \times 4$) with strict spatial partition of unity.
- Relaxed corner hard-pruning in apex endgame states to avoid deadlocks when only non-corner moves are legal.
- Updated root documentation index and technical guides.

### Performance
- **Ampere TF32 Acceleration:** Configured PyTorch TensorFloat-32 for matmul and cuDNN backends on NVIDIA RTX 3060, achieving $166,574\text{ samples/sec}$ at mini-batch size $B = 512$.
- **Pinned Host-to-Device Memory:** Implemented page-locked memory allocations and non-blocking asynchronous DMA transfers, achieving a **+34.4%** PCIe transfer speedup.
- **Inlined Slide Primitives:** Inlined row transformations in pure Python, eliminating intermediate array allocations and boosting single-thread simulation throughput to $> 17,400\text{ steps/sec}$.

### Fixed
- Fixed checkpoint path resolution to gracefully handle relative, absolute, and directory inputs across Windows and Linux platforms.
- Fixed evaluator weight loading sequence to guarantee checkpoints are fully loaded before launching rollouts.
- Fixed core simulation throughput threshold to prevent test flakiness under CPU power throttling.
- Corrected KaTeX underscore parsing in Markdown documentation to render mathematical subscripts cleanly.

### Documentation
- Created comprehensive architecture specifications in `ARCHITECTURE.md` and `docs/architecture.md`.
- Authored detailed guides for MDP environment (`docs/environment.md`), training pipeline (`docs/training_pipeline.md`), GPU optimization (`docs/gpu_optimization.md`), benchmarks (`docs/benchmarks.md`), and evaluation/telemetry (`docs/evaluation.md`).
- Documented developer contribution guidelines in `CONTRIBUTING.md`.
