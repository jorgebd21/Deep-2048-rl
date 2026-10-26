"""Checkpoint resolution and path helper utilities."""

import os
from typing import Optional


def resolve_checkpoint_path(path: Optional[str]) -> Optional[str]:
    """Resolve checkpoint file path with alias and directory fallbacks.

    Args:
        path: User-provided checkpoint path or filename.

    Returns:
        Resolved absolute or relative path, or None if path is None.

    Raises:
        FileNotFoundError: If a non-None path cannot be resolved to an existing file.
    """
    if path is None:
        return None

    candidates = [
        path,
        os.path.join("checkpoints", path),
        os.path.join("checkpoints", os.path.basename(path)),
    ]

    base = os.path.basename(path)
    if "best_model" in base:
        candidates.append(path.replace("best_model", "checkpoint_best"))
        candidates.append(os.path.join("checkpoints", "checkpoint_best.pt"))
    elif "checkpoint_best" in base:
        candidates.append(path.replace("checkpoint_best", "best_model"))
        candidates.append(os.path.join("checkpoints", "best_model.pt"))
    elif "latest_model" in base:
        candidates.append(path.replace("latest_model", "checkpoint_latest"))
        candidates.append(os.path.join("checkpoints", "checkpoint_latest.pt"))
    elif "checkpoint_latest" in base:
        candidates.append(path.replace("checkpoint_latest", "latest_model"))
        candidates.append(os.path.join("checkpoints", "checkpoint_latest.pt"))

    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return c

    available = os.listdir("checkpoints") if os.path.exists("checkpoints") else []
    raise FileNotFoundError(
        f"Checkpoint file '{path}' not found. Available files in 'checkpoints/': {available}"
    )
