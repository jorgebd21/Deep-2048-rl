"""Unit tests for Game2048Env Gymnasium wrapper and one-hot observation encoding."""

import numpy as np
import pytest
from gymnasium import spaces

from src.env.core import Action
from src.env.wrapper import Game2048Env, one_hot_encode_board


class TestObservationEncoder:
    """Test suite for one-hot multi-channel tensor encoding."""

    def test_empty_board_encoding(self):
        board = np.zeros((4, 4), dtype=np.int64)
        obs = one_hot_encode_board(board, num_channels=16)
        assert obs.shape == (16, 4, 4)
        assert obs.dtype == np.float32
        # Channel 0 must be 1.0 for all cells
        np.testing.assert_array_equal(obs[0], np.ones((4, 4), dtype=np.float32))
        # Channels 1..15 must be 0.0
        np.testing.assert_array_equal(obs[1:], np.zeros((15, 4, 4), dtype=np.float32))

    def test_specific_tile_channels(self):
        board = np.zeros((4, 4), dtype=np.int64)
        board[0, 0] = 2     # 2^1 -> channel 1
        board[0, 1] = 4     # 2^2 -> channel 2
        board[1, 1] = 8     # 2^3 -> channel 3
        board[2, 2] = 2048  # 2^11 -> channel 11
        board[3, 3] = 32768 # 2^15 -> channel 15

        obs = one_hot_encode_board(board, num_channels=16)

        assert obs[1, 0, 0] == 1.0
        assert obs[2, 0, 1] == 1.0
        assert obs[3, 1, 1] == 1.0
        assert obs[11, 2, 2] == 1.0
        assert obs[15, 3, 3] == 1.0

        # Channel 0 should be 0.0 where tiles exist
        assert obs[0, 0, 0] == 0.0
        assert obs[0, 0, 1] == 0.0
        assert obs[0, 1, 1] == 0.0
        assert obs[0, 2, 2] == 0.0
        assert obs[0, 3, 3] == 0.0

        # Channel 0 should be 1.0 for empty cell e.g. (0, 2)
        assert obs[0, 0, 2] == 1.0

    def test_one_hot_partition_of_unity(self):
        """Sum across channels for any grid cell must equal exactly 1.0."""
        rng = np.random.default_rng(777)
        tiles = np.array([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        board = rng.choice(tiles, size=(4, 4))
        obs = one_hot_encode_board(board, num_channels=16)

        channel_sum = np.sum(obs, axis=0)
        np.testing.assert_allclose(channel_sum, np.ones((4, 4), dtype=np.float32))


class TestGymnasiumWrapper:
    """Test suite for Gymnasium standard interface compliance."""

    def test_env_specifications(self):
        env = Game2048Env(seed=42)
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 4
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (16, 4, 4)
        assert env.observation_space.dtype == np.float32

    def test_reset_protocol(self):
        env = Game2048Env(seed=42)
        obs, info = env.reset(seed=42)

        assert obs.shape == (16, 4, 4)
        assert isinstance(info, dict)
        assert "score" in info
        assert "max_tile" in info
        assert "action_mask" in info
        assert "board" in info
        assert len(info["action_mask"]) == 4

    def test_step_protocol(self):
        env = Game2048Env(seed=100)
        obs, info = env.reset(seed=100)

        mask = info["action_mask"]
        valid_actions = np.flatnonzero(mask)
        assert len(valid_actions) > 0
        act = valid_actions[0]

        next_obs, reward, terminated, truncated, next_info = env.step(act)
        assert next_obs.shape == (16, 4, 4)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(next_info, dict)
        assert "action_mask" in next_info

    def test_max_steps_truncation(self):
        env = Game2048Env(seed=101, max_steps=5)
        env.reset(seed=101)
        truncated = False
        for _ in range(10):
            mask = env.get_action_mask()
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                break
            _, _, term, trunc, _ = env.step(valid[0])
            if term or trunc:
                truncated = trunc
                break
        assert truncated or term

    def test_ansi_render(self):
        env = Game2048Env(seed=102)
        env.reset(seed=102)
        rendered = env.render()
        assert isinstance(rendered, str)
        assert "+------+" in rendered
