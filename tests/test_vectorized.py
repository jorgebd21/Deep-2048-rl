"""Unit tests for VectorizedGame2048Env and batched replay buffer ingestion."""

import numpy as np
import pytest
import torch

from src.env.core import Action
from src.env.vectorized import VectorizedGame2048Env
from src.memory.replay_buffer import PrioritizedReplayBuffer


class TestVectorizedEnv:
    """Tests for multi-environment synchronization, stepping, and auto-reset."""

    def test_vectorized_init_and_reset(self) -> None:
        """Verify initialization and reset observation shapes across N parallel games."""
        num_envs = 8
        vec_env = VectorizedGame2048Env(num_envs=num_envs, seed=123)

        obs, infos = vec_env.reset()

        assert obs.shape == (num_envs, 16, 4, 4)
        assert obs.dtype == np.float32
        assert len(infos) == num_envs

        # Exactly 2 non-zero tiles per reset board
        assert np.sum(obs[:, 0, :, :]) == num_envs * 14  # 14 empty cells
        assert np.sum(obs[:, 1:, :, :]) == num_envs * 2   # 2 occupied cells

        masks = vec_env.get_action_masks()
        assert masks.shape == (num_envs, 4)
        assert masks.dtype == bool
        assert np.all(np.any(masks, axis=1))  # At least one legal move per fresh board

    def test_vectorized_step_execution(self) -> None:
        """Verify synchronous batch stepping and output tensor dimensions."""
        num_envs = 16
        vec_env = VectorizedGame2048Env(num_envs=num_envs, seed=42)
        vec_env.reset()

        actions = np.zeros(num_envs, dtype=np.int64)  # Action.UP
        next_obs, rewards, terms, truncs, infos = vec_env.step(actions)

        assert next_obs.shape == (num_envs, 16, 4, 4)
        assert rewards.shape == (num_envs,)
        assert terms.shape == (num_envs,)
        assert truncs.shape == (num_envs,)
        assert len(infos) == num_envs

        for i, info in enumerate(infos):
            assert "action_mask" in info
            assert "score" in info
            assert "max_tile" in info
            assert info["action_mask"].shape == (4,)

    def test_vectorized_auto_reset_on_termination(self) -> None:
        """Verify that terminated sub-environments are automatically reset."""
        vec_env = VectorizedGame2048Env(num_envs=2, seed=99)
        vec_env.reset()

        # Artificially configure env 0 to be deadlocked
        deadlock = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ], dtype=np.int64)
        vec_env.envs[0].board = deadlock.copy()

        # Step both environments
        next_obs, rewards, terms, truncs, infos = vec_env.step([0, 0])

        assert terms[0] is True or terms[0] == 1
        assert "final_info" in infos[0]
        assert "final_observation" in infos[0]
        assert infos[0]["final_observation"].shape == (16, 4, 4)

        # Environment 0 must have been auto-reset with a fresh board
        assert np.count_nonzero(vec_env.envs[0].board) == 2
        assert not np.array_equal(vec_env.envs[0].board, deadlock)

    def test_batched_replay_buffer_integration(self) -> None:
        """Verify batch ingestion from vectorized env into PrioritizedReplayBuffer."""
        num_envs = 8
        vec_env = VectorizedGame2048Env(num_envs=num_envs, seed=10)
        buffer = PrioritizedReplayBuffer(capacity=1000)

        obs, infos = vec_env.reset()
        initial_masks = vec_env.get_action_masks()
        initial_boards = np.array([env.board.copy() for env in vec_env.envs])

        # Step
        actions = np.random.randint(0, 4, size=num_envs)
        next_obs, rewards, terms, truncs, next_infos = vec_env.step(actions)
        next_masks = vec_env.get_action_masks()
        next_boards = np.array([env.board.copy() for env in vec_env.envs])

        # Ingest batch
        buffer.add_batch(
            states=initial_boards,
            actions=actions,
            rewards=rewards,
            next_states=next_boards,
            dones=terms,
            action_masks=initial_masks,
            next_action_masks=next_masks,
        )

        assert len(buffer) == num_envs

        # Sample with pinned memory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = buffer.sample(batch_size=4, device=device, pin_memory=torch.cuda.is_available())

        assert batch["states"].shape == (4, 16, 4, 4)
        assert batch["actions"].shape == (4,)
        assert batch["weights"].shape == (4,)
        if torch.cuda.is_available():
            assert batch["states"].is_cuda
