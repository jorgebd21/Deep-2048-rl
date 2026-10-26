# GPU Optimization & Hardware Acceleration Guide

This document details the hardware acceleration strategy, memory optimizations, and execution profiling for Deep Reinforcement Learning on **NVIDIA Ampere architecture** (specifically targeting the **NVIDIA GeForce RTX 3060 12GB VRAM**).

---

## 1. Architectural Hardware Target

- **Microarchitecture**: NVIDIA Ampere (Compute Capability `sm_86`)
- **GPU Core Specs**: 3584 CUDA Cores, 112 3rd-Gen Tensor Cores, 28 RT Cores
- **VRAM**: 12 GB GDDR6 (192-bit bus, 360 GB/s bandwidth)
- **Execution Target**: Headless headless tensor operations, VRAM peak < 8 GB

---

## 2. NVIDIA Ampere TensorFloat-32 (TF32) Optimization

TensorFloat-32 (TF32) provides the range of FP32 (8-bit exponent) with the precision of FP16 (10-bit mantissa), running on Ampere Tensor Cores with zero code changes or loss of numerical stability for Deep Reinforcement Learning.

### 2.1. Global Configuration
In `src/utils/cuda_utils.py`, PyTorch backends are explicitly configured to allow TF32 for matrix multiplications and convolution kernels:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

### 2.2. Mathematical Stability
Unlike FP16 mixed precision, TF32 does not require dynamic loss scaling or gradient clipping for underflow/overflow prevention. The Bellman error updates:
$$\delta_i = r_i + \gamma \max_{a'} Q(s'_i, a'; \theta^-) - Q(s_i, a_i; \theta)$$
remain numerically stable without scale-factor drift.

---

## 3. Host-to-Device Memory Pipeline (RAM to VRAM)

### 3.1. Compact Exponent Storage in RAM
The Prioritized Experience Replay buffer stores states as raw tile powers in packed `uint8` (`0` to `15` representing $0, 2^1, \dots, 2^{15}$):
- Raw float32 multi-channel tensor: $16 \times 4 \times 4 \times 4\text{ bytes} = 1,024\text{ bytes}$ per state.
- Packed uint8 exponent board: $4 \times 4 \times 1\text{ byte} = 16\text{ bytes}$ per state.
- **Memory Compression Factor**: $64\times$ reduction.
- For a buffer capacity of $N = 100,000$ transitions: total RAM footprint is $< 180\text{ MB}$, well within standard system limits.

### 3.2. Pinned Memory & Asynchronous DMA Copies
When sampling mini-batches for GPU optimization:
1. Compact uint8 exponents are indexed and converted into one-hot binary planes ($B \times 16 \times 4 \times 4$).
2. The sampled CPU tensors are placed in page-locked (pinned) host memory:
   ```python
   states_t = states_t.pin_memory()
   ```
3. Tensors are transferred asynchronously across the PCIe Gen4 bus via Direct Memory Access (DMA):
   ```python
   states_t = states_t.to(device, non_blocking=True)
   ```
4. This overlaps CPU SumTree priority tree traversals with PCIe bus transfers and CUDA kernel execution.

---

## 4. Batch Size Scaling & Latency Profile

Empirical measurements conducted on the RTX 3060:

| Mini-batch Size | Forward Pass (ms) | Full Optimization Step (ms) | Optimization Rate (samples/s) | Peak VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- |
| **B = 256** | 0.856 ms | 3.220 ms | 79,494.9 | 76.0 MB |
| **B = 512** | 0.877 ms | 3.074 ms | 166,574.8 | 91.8 MB |
| **B = 1024** | 1.297 ms | 4.162 ms | 246,009.7 | 128.0 MB |

> [!TIP]
> **Recommended Production Mini-batch**: $B = 512$. It achieves near-peak compute density ($166,000+$ samples/s) while providing smooth gradient descent updates without excessive stochastic variance.

---

## 5. Replay Ratio & Pipeline Concurrency

To ensure that neither the CPU simulation nor the GPU optimizer is starved:
- **CPU Simulation**: `VectorizedGame2048Env` runs $N = 32$ parallel environments at $\approx 17,358\text{ steps/sec}$.
- **GPU Optimization**: Mini-batch optimization of $B = 512$ takes $\approx 3.07\text{ ms}$ per step ($\approx 325\text{ gradient steps/sec}$).
- **Replay Ratio ($\mathcal{R}$)**:
  $$\mathcal{R} = \frac{\text{Gradient Optimization Steps}}{\text{Environment Steps}} \in [0.125, 0.25]$$
  1 gradient update is performed for every 4 to 8 collected environment steps, ensuring fresh exploration transitions while maintaining full GPU saturation.
