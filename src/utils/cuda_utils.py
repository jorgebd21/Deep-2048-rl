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


_D4_PERMUTATIONS_LIST = [
    [0, 1, 2, 3],  # rot0: identity (LEFT, RIGHT, UP, DOWN)
    [2, 3, 1, 0],  # rot1: rot90
    [1, 0, 3, 2],  # rot2: rot180
    [3, 2, 0, 1],  # rot3: rot270
    [1, 0, 2, 3],  # flip_rot0: flip horizontal
    [2, 3, 0, 1],  # flip_rot1: flip + rot90
    [0, 1, 3, 2],  # flip_rot2: flip + rot180
    [3, 2, 1, 0],  # flip_rot3: flip + rot270
]


def apply_random_d4_augmentation(
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Apply a random Dihedral D4 transformation to an experience mini-batch on GPU.

    In 2048, game dynamics are isometric under 90-degree rotations and horizontal/vertical
    reflections. Transforming the batch randomly enforces rotation and reflection invariance
    in policy and value weights without extra memory consumption.

    Args:
        batch: Dictionary containing mini-batch tensors ('states', 'next_states', etc.)

    Returns:
        Augmented batch dictionary with transformed tensors.
    """
    k = int(torch.randint(0, 8, (1,)).item())
    if k == 0:
        return batch  # Identity transformation

    perm = _D4_PERMUTATIONS_LIST[k]
    st = batch["states"]
    nst = batch["next_states"]

    # Symmetries 4..7 incorporate a horizontal flip
    if k >= 4:
        st = torch.flip(st, [-1])
        nst = torch.flip(nst, [-1])

    rot_steps = k % 4
    if rot_steps > 0:
        st = torch.rot90(st, rot_steps, [-2, -1])
        nst = torch.rot90(nst, rot_steps, [-2, -1])

    batch["states"] = st
    batch["next_states"] = nst

    # Transform discrete actions if present in batch
    if "actions" in batch and batch["actions"] is not None:
        perm_t = torch.tensor(perm, dtype=torch.long, device=batch["actions"].device)
        batch["actions"] = perm_t[batch["actions"]]

    # Transform action masks if present
    if "action_masks" in batch and batch["action_masks"] is not None:
        batch["action_masks"] = batch["action_masks"][:, perm]

    if "next_action_masks" in batch and batch["next_action_masks"] is not None:
        batch["next_action_masks"] = batch["next_action_masks"][:, perm]

    return batch
