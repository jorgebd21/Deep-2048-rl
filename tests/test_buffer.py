"""Unit tests for SumTree and PrioritizedReplayBuffer."""

import numpy as np
import pytest
import torch

from src.memory.replay_buffer import PrioritizedReplayBuffer, SumTree, board_to_exponents


class TestSumTree:
    """Test suite for binary SumTree prefix sum mechanics."""

    def test_sumtree_propagation(self):
        capacity = 4
        tree = SumTree(capacity)
        # Tree leaves start at index capacity - 1 = 3
        # Leaves: index 3, 4, 5, 6
        tree.update(3, 1.0)
        tree.update(4, 2.0)
        tree.update(5, 3.0)
        tree.update(6, 4.0)

        # Total priority at root must equal sum of leaves: 1 + 2 + 3 + 4 = 10
        assert tree.total_priority() == 10.0
        # Parent of 3, 4 is index 1: 1.0 + 2.0 = 3.0
        assert tree.tree[1] == 3.0
        # Parent of 5, 6 is index 2: 3.0 + 4.0 = 7.0
        assert tree.tree[2] == 7.0

    def test_sumtree_retrieval(self):
        capacity = 4
        tree = SumTree(capacity)
        tree.update(3, 10.0)  # range [0, 10)
        tree.update(4, 20.0)  # range [10, 30)
        tree.update(5, 30.0)  # range [30, 60)
        tree.update(6, 40.0)  # range [60, 100)

        assert tree.total_priority() == 100.0

        # Sample 5.0 -> should land in leaf 3 (data_idx 0)
        tree_idx, prio, data_idx = tree.get(5.0)
        assert tree_idx == 3
        assert data_idx == 0

        # Sample 25.0 -> should land in leaf 4 (data_idx 1)
        tree_idx, prio, data_idx = tree.get(25.0)
        assert tree_idx == 4
        assert data_idx == 1

        # Sample 55.0 -> should land in leaf 5 (data_idx 2)
        tree_idx, prio, data_idx = tree.get(55.0)
        assert tree_idx == 5
        assert data_idx == 2

        # Sample 95.0 -> should land in leaf 6 (data_idx 3)
        tree_idx, prio, data_idx = tree.get(95.0)
        assert tree_idx == 6
        assert data_idx == 3


class TestPrioritizedReplayBuffer:
    """Test suite for PrioritizedReplayBuffer circular storage and sampling."""

    def test_board_exponent_packing(self):
        board = np.array(
            [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]],
            dtype=np.int64,
        )
        exp = board_to_exponents(board)
        assert exp.dtype == np.uint8
        assert exp[0, 0] == 0
        assert exp[0, 1] == 1
        assert exp[0, 2] == 2
        assert exp[2, 3] == 11  # 2048 -> 2^11
        assert exp[3, 3] == 15  # 32768 -> 2^15

    def test_buffer_add_and_circular_overwrite(self):
        capacity = 10
        buf = PrioritizedReplayBuffer(capacity=capacity)
        assert len(buf) == 0

        dummy_board = np.zeros((4, 4), dtype=np.int64)
        dummy_mask = np.ones(4, dtype=bool)

        for i in range(15):
            buf.add(
                state=dummy_board,
                action=i % 4,
                reward=float(i),
                next_state=dummy_board,
                done=False,
                action_mask=dummy_mask,
                next_action_mask=dummy_mask,
            )

        assert len(buf) == capacity
        assert buf.tree.write_ptr == 5  # 15 % 10 = 5

    def test_sampling_shapes_and_weights(self):
        capacity = 100
        buf = PrioritizedReplayBuffer(capacity=capacity, alpha=0.6, beta_start=0.4)

        dummy_board = np.array(
            [[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 16]],
            dtype=np.int64,
        )
        dummy_mask = np.array([True, True, False, False], dtype=bool)

        for i in range(50):
            buf.add(
                state=dummy_board,
                action=1,
                reward=2.0,
                next_state=dummy_board,
                done=False,
                action_mask=dummy_mask,
                next_action_mask=dummy_mask,
            )

        batch = buf.sample(batch_size=16)

        assert batch["states"].shape == (16, 16, 4, 4)
        assert batch["actions"].shape == (16,)
        assert batch["rewards"].shape == (16,)
        assert batch["next_states"].shape == (16, 16, 4, 4)
        assert batch["dones"].shape == (16,)
        assert batch["action_masks"].shape == (16, 4)
        assert batch["next_action_masks"].shape == (16, 4)
        assert batch["weights"].shape == (16,)
        assert len(batch["indices"]) == 16

        # Normalized IS weights must have max close to 1.0
        assert np.isclose(batch["weights"].max().item(), 1.0, atol=1e-4)

    def test_priority_updates(self):
        capacity = 20
        buf = PrioritizedReplayBuffer(capacity=capacity, alpha=0.6)

        dummy_board = np.zeros((4, 4), dtype=np.int64)
        dummy_mask = np.ones(4, dtype=bool)

        for i in range(10):
            buf.add(
                state=dummy_board,
                action=0,
                reward=1.0,
                next_state=dummy_board,
                done=False,
                action_mask=dummy_mask,
                next_action_mask=dummy_mask,
                priority=1.0,
            )

        # Update one transition with high TD-error
        tree_idx = 0 + capacity - 1
        buf.update_priorities([tree_idx], [100.0])

        assert buf.max_priority >= 100.0**0.6
        assert buf.tree.total_priority() > 10.0
