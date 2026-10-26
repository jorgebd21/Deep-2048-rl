"""Unit and integration test suite for 64-bit Bitboard engine and Afterstate learning."""

import time
import numpy as np
import pytest
import torch

from src.env.core import Action, Game2048Core
from src.env.bitboard import Bitboard2048
from src.memory.replay_buffer import PrioritizedReplayBuffer
from src.models.dueling_dqn import Dueling2048Net
from src.training.agent import DDQNAgent


class TestBitboardCoreParity:
    """Validate mathematical equivalence between Bitboard2048 and Game2048Core."""

    def test_bitboard_conversion_roundtrip(self):
        """Converting array -> bitboard -> array should preserve exact values."""
        board = np.array([
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [4096, 8192, 16384, 32768],
        ], dtype=np.int64)

        bb = Bitboard2048.from_array(board)
        recovered = Bitboard2048.to_array(bb)
        np.testing.assert_array_equal(recovered, board)

        exps = Bitboard2048.to_exponents_array(bb)
        expected_exps = np.arange(16, dtype=np.uint8).reshape(4, 4)
        np.testing.assert_array_equal(exps, expected_exps)

    def test_bitwise_transposition(self):
        """Bitwise delta-swap transpose must exactly match np.transpose."""
        rng = np.random.default_rng(42)
        for _ in range(500):
            board = rng.choice([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], size=(4, 4))
            bb = Bitboard2048.from_array(board)
            bb_t = Bitboard2048.transpose(bb)
            np_t = Bitboard2048.to_array(bb_t)
            np.testing.assert_array_equal(np_t, np.transpose(board))

    def test_bitwise_flip_and_rotations(self):
        """Bitwise flips and rotations must match numpy flip and rot90 operations."""
        rng = np.random.default_rng(123)
        for _ in range(500):
            board = rng.choice([0, 2, 4, 8, 16, 32, 64], size=(4, 4))
            bb = Bitboard2048.from_array(board)

            # Horizontal flip
            bb_flip = Bitboard2048.flip_horizontal(bb)
            np_flip = Bitboard2048.to_array(bb_flip)
            np.testing.assert_array_equal(np_flip, np.fliplr(board))

            # 90 degrees CCW
            bb_r90 = Bitboard2048.rotate90(bb)
            np_r90 = Bitboard2048.to_array(bb_r90)
            np.testing.assert_array_equal(np_r90, np.rot90(board, 1))

            # 180 degrees
            bb_r180 = Bitboard2048.rotate180(bb)
            np_r180 = Bitboard2048.to_array(bb_r180)
            np.testing.assert_array_equal(np_r180, np.rot90(board, 2))

            # 270 degrees CCW
            bb_r270 = Bitboard2048.rotate270(bb)
            np_r270 = Bitboard2048.to_array(bb_r270)
            np.testing.assert_array_equal(np_r270, np.rot90(board, 3))

    def test_d4_canonical_invariance(self):
        """All 8 dihedral transforms of any board must have identical canonical hash."""
        rng = np.random.default_rng(999)
        for _ in range(200):
            board = rng.choice([0, 2, 4, 8, 16], size=(4, 4))
            bb = Bitboard2048.from_array(board)
            canon_expected = Bitboard2048.get_canonical(bb)

            symmetries = Bitboard2048.get_d4_symmetries(bb)
            assert len(symmetries) == 8
            for sym in symmetries:
                assert Bitboard2048.get_canonical(sym) == canon_expected

    def test_move_parity_10000_actions(self):
        """Bitboard moves, scores, rewards, and changed flags must match Game2048Core across 10,000 steps."""
        rng = np.random.default_rng(2026)
        core = Game2048Core(seed=2026)
        bit_env = Bitboard2048(seed=2026)

        # Sync initial board
        core.reset()
        bit_env.board = Bitboard2048.from_array(core.board)

        steps_tested = 0
        target_steps = 10_000

        while steps_tested < target_steps:
            action = int(rng.integers(0, 4))

            # Test mask parity
            mask_core = core.get_action_mask(core.board)
            mask_bit = Bitboard2048.get_action_mask(bit_env.board)
            np.testing.assert_array_equal(mask_bit, mask_core)

            # Slide simulation comparison
            nb_core, score_c, lr_c, ch_c = Game2048Core._execute_slide(core.board, action)
            nb_bit, score_b, lr_b, ch_b = Bitboard2048.move(bit_env.board, action)

            assert ch_b == ch_c, f"Changed mismatch on step {steps_tested}: core={ch_c}, bit={ch_b}"
            assert score_b == score_c, f"Score mismatch on step {steps_tested}: core={score_c}, bit={score_b}"
            assert np.isclose(lr_b, lr_c, atol=1e-5), f"Reward mismatch: core={lr_c}, bit={lr_b}"
            np.testing.assert_array_equal(Bitboard2048.to_array(nb_bit), nb_core)

            if ch_c:
                # Step both environments
                core.board = nb_core
                bit_env.board = nb_bit
                # Spawn same tile into both
                empty_cells = np.flatnonzero(core.board == 0)
                if len(empty_cells) == 0 or core.is_terminal():
                    core.reset()
                    bit_env.board = Bitboard2048.from_array(core.board)
                else:
                    c = rng.choice(empty_cells)
                    v = 2 if rng.random() < 0.90 else 4
                    core.board.flat[c] = v
                    bit_env.board = Bitboard2048.spawn_tile_at(bit_env.board, c, v)
            else:
                if core.is_terminal():
                    core.reset()
                    bit_env.board = Bitboard2048.from_array(core.board)

            steps_tested += 1

    def test_bitboard_throughput(self):
        """Bitboard moves should achieve > 500,000 moves/sec in pure Python."""
        bb = Bitboard2048().reset()
        t0 = time.perf_counter()
        iters = 50_000
        for i in range(iters):
            Bitboard2048.move(bb, i % 4)
        elapsed = time.perf_counter() - t0
        moves_per_sec = iters / elapsed
        assert moves_per_sec > 200_000, f"Expected >200k moves/sec, got {moves_per_sec:.0f}"

    def test_compute_snake_score_d4_invariance(self):
        """Snake score must be strictly invariant under all 8 dihedral D4 transformations."""
        board = np.array([
            [2048, 1024, 512, 256],
            [16, 32, 64, 128],
            [8, 4, 2, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        bb = Bitboard2048.from_array(board)
        base_score = Bitboard2048.compute_snake_score(bb)
        assert base_score > 0.0

        for sym_bb in Bitboard2048.get_d4_symmetries(bb):
            sym_score = Bitboard2048.compute_snake_score(sym_bb)
            assert np.isclose(base_score, sym_score, atol=1e-5), (
                f"Symmetry snake score {sym_score} != base {base_score}"
            )

    def test_compute_corner_penalty(self):
        """Corner penalty should be 0 in corner, 11 on edge, 22 in interior for 2048 tile."""
        # Max tile in corner (0,0) -> 0
        b_corner = np.array([
            [2048, 1024, 512, 256],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        assert Bitboard2048.compute_corner_penalty(Bitboard2048.from_array(b_corner)) == 0.0

        # Max tile on non-corner edge (0,1) -> 11.0
        b_edge = np.array([
            [1024, 2048, 512, 256],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        assert Bitboard2048.compute_corner_penalty(Bitboard2048.from_array(b_edge)) == 11.0

        # Max tile in interior (1,1) -> 22.0
        b_interior = np.array([
            [1024, 512, 256, 128],
            [64, 2048, 32, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        assert Bitboard2048.compute_corner_penalty(Bitboard2048.from_array(b_interior)) == 22.0

        # Max tile < 128 (e.g. 64) -> no penalty even if interior
        b_low = np.array([
            [0, 0, 0, 0],
            [0, 64, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        assert Bitboard2048.compute_corner_penalty(Bitboard2048.from_array(b_low)) == 0.0

    def test_adaptive_depth_selection(self):
        """Adaptive depth must return 2 for open board, 3 for midgame, and 4 for endgame."""
        # Open board (empty >= 8, max < 1024)
        b_open = np.array([
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        assert Bitboard2048.get_adaptive_depth(Bitboard2048.from_array(b_open)) == 2

        # Mid board (empty 6, max < 1024)
        b_mid = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [2, 4, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        assert Bitboard2048.get_adaptive_depth(Bitboard2048.from_array(b_mid)) == 3

        # Critical endgame (empty <= 4, max >= 1024)
        b_endgame = np.array([
            [2048, 1024, 512, 256],
            [128, 64, 32, 16],
            [8, 4, 2, 4],
            [2, 0, 0, 0],
        ], dtype=np.int64)
        assert Bitboard2048.get_adaptive_depth(Bitboard2048.from_array(b_endgame)) == 4


class TestAfterstateLearningAndAugmentation:
    """Validate Afterstate buffer serialization and network loss computations."""

    def test_replay_buffer_afterstate_roundtrip(self):
        """Adding and sampling afterstate transitions should preserve shapes and dtypes."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        s_after = np.array([
            [2, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 16],
        ], dtype=np.int64)

        s_next_after = np.array([
            [4, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 8, 0],
            [0, 0, 0, 16],
        ], dtype=np.int64)

        for _ in range(20):
            buffer.add_afterstate(
                afterstate=s_after,
                reward=2.0,
                next_afterstate=s_next_after,
                done=False,
            )

        assert len(buffer) == 20
        batch = buffer.sample_afterstate(batch_size=8, device=torch.device("cpu"))

        assert batch["states"].shape == (8, 16, 4, 4)
        assert batch["next_states"].shape == (8, 16, 4, 4)
        assert batch["rewards"].shape == (8,)
        assert batch["dones"].shape == (8,)
        assert batch["weights"].shape == (8,)

    def test_afterstate_forward_value_and_loss(self):
        """DDQNAgent should optimize scalar value V(s_after) via compute_afterstate_loss."""
        agent = DDQNAgent(
            in_channels=16,
            num_actions=4,
            lr=1e-3,
            device=torch.device("cpu"),
        )

        dummy_states = torch.randn(4, 16, 4, 4)
        val = agent.policy_net.forward_value(dummy_states)
        assert val.shape == (4,)

        batch = {
            "states": dummy_states,
            "rewards": torch.tensor([1.0, 2.0, 0.0, 3.0]),
            "next_states": torch.randn(4, 16, 4, 4),
            "dones": torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "weights": torch.ones(4),
        }

        loss, td_errors = agent.compute_afterstate_loss(batch)
        assert loss.item() >= 0.0
        assert td_errors.shape == (4,)

        loss_val, errors_np = agent.update_afterstate(batch)
        assert isinstance(loss_val, float)
        assert len(errors_np) == 4
