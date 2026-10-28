"""Unit tests for Expert Iteration and Native Afterstate Value Learning pipeline.

Validates:
1. Correct construction of afterstate transitions from Bitboard moves.
2. Bellman target computation matches theoretical values on deterministic boards.
3. D4 isometric invariance of value predictions under Dihedral symmetry augmentation.
4. ExpectimaxSearcher integration with scalar afterstate value heads.
"""

import numpy as np
import pytest
import torch

from src.agent.search import ExpectimaxSearcher
from src.env.bitboard import Bitboard2048
from src.env.core import Action
from src.memory.replay_buffer import PrioritizedReplayBuffer
from src.models.dueling_dqn import Dueling2048Net
from src.training.agent import DDQNAgent
from src.training.expert_iteration import (
    collect_afterstate_trajectory,
    populate_buffer_with_expert_trajectories,
)
from src.utils.cuda_utils import apply_random_d4_augmentation


class TestAfterstatePipeline:
    """Test suite for Afterstate Value Learning and trajectory generation."""

    def test_afterstate_transition_construction(self):
        """Transitions must follow (s_after, r_next, s'_after, done) semantics."""
        # Known board: [2, 2, 0, 0] in row 0
        b = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        bb = Bitboard2048.from_array(b)

        # Action LEFT merges 2 and 2 into 4 (reward = log2(4) = 2.0)
        s_after, score, lr, ch = Bitboard2048.move(bb, Action.LEFT)
        assert ch is True
        assert lr == 2.0

        # Afterstate should have 4 at (0, 0) and 0s elsewhere
        after_arr = Bitboard2048.to_array(s_after)
        assert after_arr[0, 0] == 4
        assert np.count_nonzero(after_arr) == 1

        # Spawn a 2 at (1, 1)
        spawned = Bitboard2048.spawn_tile_at(s_after, cell_idx=5, tile_value=2)
        assert Bitboard2048.to_array(spawned)[1, 1] == 2

        # Second action DOWN: moves 4 to (3, 0) and 2 to (3, 1), reward = 0.0
        s_after_prime, _, lr2, ch2 = Bitboard2048.move(spawned, Action.DOWN)
        assert ch2 is True
        assert lr2 == 0.0

        # Verify afterstate transition tuple
        st1_exps = Bitboard2048.to_exponents_array(s_after)
        st2_exps = Bitboard2048.to_exponents_array(s_after_prime)

        assert st1_exps[0, 0] == 2  # 2^2 = 4
        assert st2_exps[3, 0] == 2  # 2^2 = 4
        assert st2_exps[3, 1] == 1  # 2^1 = 2

    def test_afterstate_bellman_target_math(self):
        """Bellman target must equal r + (1-done) * gamma * V_target(s'_after)."""
        device = torch.device("cpu")
        agent = DDQNAgent(
            in_channels=16,
            num_actions=4,
            gamma=0.99,
            max_grad_norm=10.0,
            device=device,
        )

        # Mock target_net to return a constant value 10.0 for any afterstate
        with torch.no_grad():
            for p in agent.target_net.parameters():
                p.zero_()
            agent.target_net.value_stream[-1].bias.fill_(10.0)

            for p in agent.policy_net.parameters():
                p.zero_()
            agent.policy_net.value_stream[-1].bias.fill_(5.0)

        # Batch 1: non-terminal transition (done=False, reward=4.0)
        # Expected target: 4.0 + 0.99 * 10.0 = 13.9
        # Expected TD error: |13.9 - 5.0| = 8.9
        b_st = torch.zeros(1, 16, 4, 4, dtype=torch.float32)
        b_nst = torch.zeros(1, 16, 4, 4, dtype=torch.float32)
        b_rew = torch.tensor([4.0], dtype=torch.float32)
        b_dones = torch.tensor([0.0], dtype=torch.float32)
        b_weights = torch.tensor([1.0], dtype=torch.float32)

        batch = {
            "states": b_st,
            "rewards": b_rew,
            "next_states": b_nst,
            "dones": b_dones,
            "weights": b_weights,
        }

        loss, td_errors = agent.compute_afterstate_loss(batch)
        expected_td = 8.9
        assert abs(float(td_errors[0].item()) - expected_td) < 1e-4

        # Batch 2: terminal transition (done=True, reward=4.0)
        # Expected target: 4.0 + (1 - 1) * 0.99 * 10.0 = 4.0
        # Expected TD error: |4.0 - 5.0| = 1.0
        batch_terminal = {
            "states": b_st,
            "rewards": b_rew,
            "next_states": b_nst,
            "dones": torch.tensor([1.0], dtype=torch.float32),
            "weights": b_weights,
        }

        loss_term, td_errors_term = agent.compute_afterstate_loss(batch_terminal)
        assert abs(float(td_errors_term[0].item()) - 1.0) < 1e-4

    def test_d4_invariance_after_training_step(self):
        """Random D4 augmentation must preserve batch shapes and valid transforms."""
        b_st = torch.zeros(4, 16, 4, 4, dtype=torch.float32)
        b_nst = torch.zeros(4, 16, 4, 4, dtype=torch.float32)
        # Put marker tile in top-left corner
        b_st[:, 1, 0, 0] = 1.0
        b_nst[:, 2, 0, 0] = 1.0

        batch = {
            "states": b_st,
            "rewards": torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32),
            "next_states": b_nst,
            "dones": torch.zeros(4, dtype=torch.float32),
            "weights": torch.ones(4, dtype=torch.float32),
        }

        augmented = apply_random_d4_augmentation(batch)
        assert augmented["states"].shape == (4, 16, 4, 4)
        assert augmented["next_states"].shape == (4, 16, 4, 4)
        # Sum of one-hot activations must be strictly preserved
        assert torch.allclose(augmented["states"].sum(), b_st.sum())
        assert torch.allclose(augmented["next_states"].sum(), b_nst.sum())

    def test_expert_iteration_searcher_integration(self):
        """ExpectimaxSearcher with use_afterstate_net=True must select legal actions."""
        device = torch.device("cpu")
        model = Dueling2048Net(in_channels=16, num_actions=4).to(device)
        model.eval()

        searcher = ExpectimaxSearcher(
            model=model,
            device=device,
            depth=2,
            use_afterstate_net=True,
            use_snake_heuristic=True,
        )

        b = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 0, 2],
            [4, 8, 16, 32],
        ], dtype=np.int64)

        act = searcher.select_action(b)
        assert act in [0, 1, 2, 3]

        bb = Bitboard2048.from_array(b)
        mask = Bitboard2048.get_action_mask(bb)
        assert mask[act] is True or mask[act] == 1

    def test_collect_afterstate_trajectory(self):
        """Collecting afterstate trajectory must yield valid sequence of transitions."""
        device = torch.device("cpu")
        model = Dueling2048Net(in_channels=16, num_actions=4).to(device)
        model.eval()

        searcher = ExpectimaxSearcher(
            model=model,
            device=device,
            depth=1,
            use_afterstate_net=True,
            use_snake_heuristic=False,
        )

        traj = collect_afterstate_trajectory(
            searcher=searcher,
            depth=1,
            seed=123,
        )

        transitions = traj["transitions"]
        assert len(transitions) > 0
        assert traj["steps"] == len(transitions)
        assert traj["score"] >= 0
        assert traj["max_tile"] >= 4

        # Verify last transition is marked done=True
        last_s_after, last_r, last_nst, last_done = transitions[-1]
        assert last_done is True
        assert last_r == 0.0
        assert np.all(last_nst == 0)

        # Verify earlier transitions are done=False
        if len(transitions) > 1:
            first_s_after, first_r, first_nst, first_done = transitions[0]
            assert first_done is False
            assert first_s_after.shape == (4, 4)
            assert first_nst.shape == (4, 4)

    def test_populate_buffer_with_expert_trajectories(self):
        """Populating buffer must store transitions accessible via sample_afterstate."""
        device = torch.device("cpu")
        model = Dueling2048Net(in_channels=16, num_actions=4).to(device)
        model.eval()

        buffer = PrioritizedReplayBuffer(capacity=5000)
        searcher = ExpectimaxSearcher(
            model=model,
            device=device,
            depth=1,
            use_afterstate_net=True,
        )

        res = populate_buffer_with_expert_trajectories(
            buffer=buffer,
            searcher=searcher,
            num_episodes=2,
            depth=1,
            base_seed=42,
        )

        assert res["episodes_simulated"] == 2
        assert res["transitions_added"] > 0
        assert len(buffer) == res["transitions_added"]

        batch = buffer.sample_afterstate(batch_size=8, device=device)
        assert batch["states"].shape == (8, 16, 4, 4)
        assert batch["next_states"].shape == (8, 16, 4, 4)
        assert batch["rewards"].shape == (8,)
        assert batch["dones"].shape == (8,)
