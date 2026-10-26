"""CUDA and hardware configuration utilities for NVIDIA Ampere architecture."""

import logging
from typing import Dict, Any
import torch

logger = logging.getLogger(__name__)


def setup_ampere_tf32() -> bool:
    """Configure PyTorch backends for NVIDIA Ampere TensorFloat-32 (TF32) acceleration.

    Enables high-throughput matrix multiplications on RTX 3060 Tensor Cores.

    Returns:
        True if CUDA is available and TF32 was enabled, False otherwise.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return True
    return False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get computation device with GPU priority and automatic fallback.

    Args:
        prefer_gpu: If True, prioritize CUDA if available.

    Returns:
        torch.device instance ('cuda' or 'cpu').
    """
    if prefer_gpu and torch.cuda.is_available():
        setup_ampere_tf32()
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_telemetry() -> Dict[str, Any]:
    """Retrieve GPU memory usage and device properties.

    Returns:
        Dictionary containing device name, total memory, and allocated memory.
    """
    if not torch.cuda.is_available():
        return {
            "device_available": False,
            "device_name": "CPU",
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "total_mb": 0.0,
        }

    device_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_id)
    allocated = torch.cuda.memory_allocated(device_id) / (1024**2)
    reserved = torch.cuda.memory_reserved(device_id) / (1024**2)
    total = props.total_memory / (1024**2)

    return {
        "device_available": True,
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "total_mb": round(total, 2),
    }
