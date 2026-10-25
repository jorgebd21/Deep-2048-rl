"""Gymnasium wrapper and observation encoders for the 2048 environment."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import gymnasium as spaces_module
from gymnasium import Env, spaces

from src.env.core import Action, Game2048Core

# Precompute lookup table for mapping tile values to exponent indices:
# 0 -> 0
# 2^k -> min(k, 15) for k in 1..16
MAX_TILE_VAL = 65536
_POWER_LUT = np.zeros(MAX_TILE_VAL + 1, dtype=np.int64)
for _k in range(1, 17):
    _val = 1 << _k
    if _val <= MAX_TILE_VAL:
        _POWER_LUT[_val] = min(_k, 15)

_IDENTITY_16 = np.eye(16, dtype=np.float32)


def one_hot_encode_board(
    board: np.ndarray, num_channels: int = 16
) -> np.ndarray:
    """Encode a 4x4 integer 2048 board into a multi-channel binary tensor.

    Args:
        board: 2D array of shape (4, 4) with integer tile values.
        num_channels: Number of channels, default 16.
            Channel 0: Empty cells (0).
            Channel k (1..15): Value 2^k.

    Returns:
        3D array of shape (num_channels, 4, 4) with dtype float32.
    """
    clipped = np.clip(board, 0, MAX_TILE_VAL)
    power_indices = _POWER_LUT[clipped]
    # (4, 4, 16) -> (16, 4, 4)
    if num_channels == 16:
        return _IDENTITY_16[power_indices].transpose(2, 0, 1)
    eye = np.eye(num_channels, dtype=np.float32)
    return eye[np.clip(power_indices, 0, num_channels - 1)].transpose(2, 0, 1)


class Game2048Env(Env):
    """Headless Gymnasium-compliant environment for 2048.

    Observation space:
        Box(0.0, 1.0, shape=(16, 4, 4), dtype=np.float32)
    Action space:
        Discrete(4): 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        seed: Optional[int] = None,
        num_channels: int = 16,
        max_steps: Optional[int] = None,
    ) -> None:
        """Initialize Gymnasium 2048 environment."""
        super().__init__()
        self.num_channels = num_channels
        self.max_steps = max_steps
        self.core = Game2048Core(seed=seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_channels, 4, 4),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset game state and return initial observation and info.

        Args:
            seed: Optional PRNG seed.
            options: Optional configuration dictionary.

        Returns:
            Tuple of (observation tensor, info dict).
        """
        super().reset(seed=seed)
        board = self.core.reset(seed=seed)
        obs = one_hot_encode_board(board, num_channels=self.num_channels)
        action_mask = self.core.get_action_mask()

        info: Dict[str, Any] = {
            "score": self.core.score,
            "max_tile": int(np.max(board)),
            "steps": self.core.steps,
            "action_mask": action_mask,
            "board": board.copy(),
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one environment step.

        Args:
            action: Action integer in {0, 1, 2, 3}.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        board, reward, terminated, info = self.core.step(action)
        obs = one_hot_encode_board(board, num_channels=self.num_channels)
        action_mask = self.core.get_action_mask()

        truncated = False
        if self.max_steps is not None and self.core.steps >= self.max_steps:
            truncated = True

        info["action_mask"] = action_mask
        info["board"] = board.copy()

        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """Return boolean mask of valid actions for current state."""
        return self.core.get_action_mask()

    @property
    def board(self) -> np.ndarray:
        """Return copy of the underlying 4x4 integer board."""
        return self.core.board.copy()

    @property
    def score(self) -> int:
        """Return cumulative raw merge score."""
        return self.core.score

    def render(self) -> str:
        """Render board to ANSI text representation."""
        lines = ["+------+------+------+------+"]
        for row in self.core.board:
            row_str = "|" + "|".join(f"{val:^6}" if val > 0 else "      " for val in row) + "|"
            lines.append(row_str)
            lines.append("+------+------+------+------+")
        return "\n".join(lines)
