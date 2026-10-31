# Security Policy

## Supported Versions

We actively maintain and provide security updates for the current major release:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## Reporting a Vulnerability

The Deep-2048-RL team takes software security seriously. If you discover a security vulnerability in this project, please report it privately.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security concerns by emailing the maintainer directly at:
- **Security Contact:** `security-drl@protonmail.com` (or create a private security advisory on GitHub).

Please include the following information in your report:
- Type of issue (e.g., buffer handling, unpickling vulnerability in checkpoint loading, dependency CVE).
- Detailed steps to reproduce the issue.
- Potential impact and affected versions.
- Any suggested remediations or patches.

You should receive an acknowledgement of your report within 48 hours, along with an initial assessment and timeline for remediation.

---

## Safe Checkpoint & Weight Deserialization Policy

This project utilizes PyTorch checkpoint files (`.pt`) and NumPy archives (`.npz`). When loading model weights in production environments:
1. Always load checkpoints originating from trusted sources.
2. In PyTorch `>= 2.4.0`, standard weights-only deserialization (`weights_only=True`) is recommended where full pickled object graphs are not required.
3. NumPy `.npz` tables for N-Tuple evaluation are loaded with `allow_pickle=False` by default to prevent arbitrary code execution during tensor deserialization.
