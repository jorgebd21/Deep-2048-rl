"""Vectorized synchronous multi-environment engine for parallel 2048 simulations."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.env.core import Action, Game2048Core
from src.env.wrapper import MAX_TILE_VAL, _POWER_LUT, _IDENTITY_16, one_hot_encode_board


class VectorizedGame2048Env:
    """Synchronous vectorized 2048 environment simulating N parallel board states.

    Executes batched game transitions and on-the-fly multi-channel one-hot encoding,
    automatically resetting individual sub-environments upon episode termination.

    Attributes:
        num_envs: Number of concurrent parallel environments.
        num_channels: One-hot binary channels (16 by default).
    """

    def __init__(
        self,
        num_envs: int = 16,
        num_channels: int = 16,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize N parallel Game2048Core instances."""
        self.num_envs = num_envs
        self.num_channels = num_channels
        self.base_seed = seed or 42

        self.envs = [
            Game2048Core(seed=self.base_seed + i) for i in range(num_envs)
        ]

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset all sub-environments simultaneously.

        Args:
            seed: Optional base seed for reseeding sub-environments.

        Returns:
            Tuple of:
                - obs_batch: Float32 array of shape (num_envs, 16, 4, 4).
                - infos: List of N info dictionaries containing masks and scores.
        """
        if seed is not None:
            self.base_seed = seed
            for i, env in enumerate(self.envs):
                env.reset(seed=self.base_seed + i)
        else:
            for env in self.envs:
                env.reset()

        return self._get_batch_obs_and_infos()

    def _get_batch_obs_and_infos(
        self,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Construct batched one-hot tensors and individual metadata dictionaries."""
        boards = np.empty((self.num_envs, 4, 4), dtype=np.int64)
        infos = []

        for i, env in enumerate(self.envs):
            boards[i] = env.board
            mask = env.get_action_mask()
            infos.append(
                {
                    "score": env.score,
                    "max_tile": int(np.max(env.board)),
                    "steps": env.steps,
                    "action_mask": mask,
                    "board": env.board.copy(),
                }
            )

        clipped = np.clip(boards, 0, MAX_TILE_VAL)
        power_indices = _POWER_LUT[clipped]  # (num_envs, 4, 4)
        obs_batch = _IDENTITY_16[power_indices].transpose(0, 3, 1, 2)  # (num_envs, 16, 4, 4)

        return obs_batch.astype(np.float32), infos

    def step(
        self, actions: Union[List[int], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Advance all sub-environments by one discrete step.

        Automatically resets sub-environments that reach terminal state, embedding
        terminal state data under 'final_info' and 'final_observation'.

        Args:
            actions: 1D array or list of length num_envs with actions in {0, 1, 2, 3}.

        Returns:
            Tuple containing:
                - next_obs_batch: (num_envs, 16, 4, 4) float32 tensor
                - rewards: (num_envs,) float32 array
                - terminated: (num_envs,) boolean array
                - truncated: (num_envs,) boolean array
                - infos: List of info dictionaries
        """
        rewards = np.empty(self.num_envs, dtype=np.float32)
        terminated = np.empty(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: List[Dict[str, Any]] = []

        for i, (env, act) in enumerate(zip(self.envs, actions)):
            board, r, term, info = env.step(int(act))
            rewards[i] = r
            terminated[i] = term

            info["action_mask"] = env.get_action_mask()
            info["board"] = board.copy()

            if term:
                # Capture terminal board state before auto-reset
                info["final_info"] = {
                    "score": info["score"],
                    "max_tile": info["max_tile"],
                    "steps": info["steps"],
                    "board": board.copy(),
                }
                info["final_observation"] = one_hot_encode_board(
                    board, num_channels=self.num_channels
                )
                # Auto-reset terminated environment
                env.reset()
                info["action_mask"] = env.get_action_mask()
                info["board"] = env.board.copy()

            infos.append(info)

        # Batch encode current boards (post-reset if applicable)
        boards = np.empty((self.num_envs, 4, 4), dtype=np.int64)
        for i, env in enumerate(self.envs):
            boards[i] = env.board

        clipped = np.clip(boards, 0, MAX_TILE_VAL)
        power_indices = _POWER_LUT[clipped]
        obs_batch = _IDENTITY_16[power_indices].transpose(0, 3, 1, 2).astype(np.float32)

        return obs_batch, rewards, terminated, truncated, infos

    def get_action_masks(self) -> np.ndarray:
        """Return (num_envs, 4) boolean array of legal action masks."""
        masks = np.empty((self.num_envs, 4), dtype=bool)
        for i, env in enumerate(self.envs):
            masks[i] = env.get_action_mask()
        return masks
