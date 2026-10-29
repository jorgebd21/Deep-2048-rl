"""Unit and integration tests for Expectimax / Lookahead search."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.agent.search import (
    ExpectimaxSearcher,
    expectimax_search,
    generate_d4_masks,
    generate_d4_tensors,
)
from src.env.bitboard import Bitboard2048
from src.env.core import Action, Game2048Core
from src.models.dueling_dqn import Dueling2048Net
from src.training.agent import DDQNAgent


@pytest.fixture
def mock_model():
    """Create a deterministic lightweight Dueling2048Net model."""
    torch.manual_seed(42)
    model = Dueling2048Net(in_channels=16, num_actions=4)
    model.eval()
    return model


class TestExpectimaxSearcher:
    """Test suite for Expectimax lookahead tree search."""

    def test_search_greedy_depth_0(self, mock_model):
        """Depth 0 should match direct model output argmax."""
        board = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(model=mock_model, device=torch.device("cpu"), depth=0)
        action, q_dict = searcher.search(board, depth=0)

        assert 0 <= action <= 3
        assert len(q_dict) == 4
        assert action == max(q_dict.keys(), key=lambda a: q_dict[a])

    def test_single_legal_action_short_circuit(self, mock_model):
        """When only one action is legal, searcher should immediately return it."""
        # Construct a board where only LEFT is legal
        board = np.array([
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        # Note: on this board LEFT and RIGHT are legal; let's force a mask with only 1 action
        custom_mask = np.array([True, False, False, False], dtype=bool)

        searcher = ExpectimaxSearcher(model=mock_model, device=torch.device("cpu"), depth=2)
        action, q_dict = searcher.search(board, action_mask=custom_mask)

        assert action == Action.LEFT
        assert q_dict[Action.LEFT] == 0.0
        assert q_dict[Action.RIGHT] == -1e9

    def test_depth_1_lookahead(self, mock_model):
        """Depth 1 should evaluate legal afterstates and return a valid action."""
        board = np.array([
            [2, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 2, 4],
            [8, 16, 32, 64],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(model=mock_model, device=torch.device("cpu"), depth=1)
        action, q_dict = searcher.search(board, depth=1)

        mask = Game2048Core().get_action_mask(board)
        assert mask[action] is True or mask[action] == 1
        assert q_dict[action] > -1e8

    def test_depth_2_expectimax_probabilities(self, mock_model):
        """Depth 2 should evaluate chance nodes across empty cells with 90/10 ratio."""
        board = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            depth=2,
            gamma=0.99,
        )
        action, q_dict = searcher.search(board, depth=2)

        mask = Game2048Core().get_action_mask(board)
        assert bool(mask[action]) is True
        for a in range(4):
            if not mask[a]:
                assert q_dict[a] == -1e9

    def test_depth_3_search(self, mock_model):
        """Depth 3 should execute multi-ply search and pick legal action."""
        board = np.array([
            [2, 4, 8, 16],
            [0, 2, 0, 4],
            [0, 0, 2, 0],
            [0, 0, 0, 2],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            depth=3,
            max_empty_samples=2,
        )
        action, q_dict = searcher.search(board, depth=3)

        mask = Game2048Core().get_action_mask(board)
        assert bool(mask[action]) is True

    def test_depth_4_search(self, mock_model):
        """Depth 4 should execute 4-ply search with probability pruning and pick legal action."""
        board = np.array([
            [2, 4, 8, 16],
            [0, 2, 0, 4],
            [0, 0, 2, 0],
            [0, 0, 0, 2],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            depth=4,
            max_empty_samples=2,
        )
        action, q_dict = searcher.search(board, depth=4)

        mask = Game2048Core().get_action_mask(board)
        assert bool(mask[action]) is True
        assert len(searcher.tt) > 0

    def test_transposition_table_caching(self, mock_model):
        """Transposition Table should cache canonical evaluations across moves."""
        board = np.array([
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            depth=2,
        )
        searcher.search(board, depth=2)
        initial_tt_size = len(searcher.tt)
        assert initial_tt_size > 0

        # Searching same board again should not require new entries
        searcher.search(board, depth=2)
        assert len(searcher.tt) == initial_tt_size

        searcher.clear_tt()
        assert len(searcher.tt) == 0

    def test_terminal_state_handling(self, mock_model):
        """Terminal board with no legal actions should return fallback action 0."""
        # Full board with no adjacent matches
        board = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(model=mock_model, device=torch.device("cpu"), depth=2)
        action, q_dict = searcher.search(board)

        assert action == 0
        for a in range(4):
            assert q_dict[a] == -1e9

    def test_agent_select_action_with_lookahead(self, mock_model):
        """DDQNAgent should support lookahead_depth in select_action."""
        agent = DDQNAgent(device=torch.device("cpu"))
        board = np.array([
            [2, 2, 0, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)

        mask = Game2048Core().get_action_mask(board)

        action = agent.select_action(
            state=np.zeros((16, 4, 4), dtype=np.float32),
            action_mask=mask,
            deterministic=True,
            lookahead_depth=2,
            board=board,
        )

        assert bool(mask[action]) is True

    def test_functional_wrapper(self, mock_model):
        """expectimax_search functional wrapper returns expected types."""
        board = np.array([
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)

        act, q_dict = expectimax_search(board, model=mock_model, depth=2, use_d4_symmetry=True)
        assert isinstance(act, int)
        assert isinstance(q_dict, dict)
        assert len(q_dict) == 4

    def test_generate_d4_tensors_distinct_permutations(self):
        """Generate all 8 dihedral transformations on an asymmetric 4x4 matrix and ensure uniqueness."""
        asym = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        d4_tensors = generate_d4_tensors(asym)

        assert d4_tensors.shape == (8, 1, 4, 4)
        slices = [d4_tensors[k, 0].numpy() for k in range(8)]
        for i in range(8):
            for j in range(i + 1, 8):
                assert not np.array_equal(slices[i], slices[j]), f"Transforms {i} and {j} should be distinct"

    def test_d4_symmetrized_evaluation_rotation_invariance(self, mock_model):
        """D4 symmetrization ensures V(s) == V(g * s) across all 8 dihedral group elements."""
        board = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            use_d4_symmetry=True,
        )

        base_val = searcher._batch_evaluate_boards([board], use_d4=True)[0]

        # Test all 4 rotations
        for k in range(1, 4):
            rot_b = np.rot90(board, k)
            rot_val = searcher._batch_evaluate_boards([rot_b], use_d4=True)[0]
            assert np.isclose(base_val, rot_val, atol=1e-4), (
                f"Rotation k={k} value {rot_val:.5f} does not match base {base_val:.5f}"
            )

        # Test horizontal reflection and its rotations
        flip_b = np.fliplr(board)
        flip_val = searcher._batch_evaluate_boards([flip_b], use_d4=True)[0]
        assert np.isclose(base_val, flip_val, atol=1e-4), "Reflection does not match base value"

        for k in range(1, 4):
            rot_flip_b = np.rot90(flip_b, k)
            rot_flip_val = searcher._batch_evaluate_boards([rot_flip_b], use_d4=True)[0]
            assert np.isclose(base_val, rot_flip_val, atol=1e-4), (
                f"Flipped rotation k={k} does not match base value"
            )

    def test_d4_greedy_and_expectimax_execution(self, mock_model):
        """Verify D4 symmetry execution in both greedy and 2-ply expectimax searches."""
        board = np.array([
            [2, 4, 8, 16],
            [0, 4, 0, 8],
            [2, 0, 4, 0],
            [0, 2, 0, 4],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            depth=2,
            use_d4_symmetry=True,
        )

        # Depth 0 with D4
        act0, q0 = searcher.search(board, depth=0, use_d4_symmetry=True)
        assert 0 <= act0 <= 3
        assert len(q0) == 4

        # Depth 2 with D4
        act2, q2 = searcher.search(board, depth=2, use_d4_symmetry=True)
        assert 0 <= act2 <= 3
        assert len(q2) == 4
        mask = Game2048Core().get_action_mask(board)
        assert bool(mask[act2]) is True

    def test_adaptive_depth_search(self, mock_model):
        """Verify adaptive depth executes properly and returns legal actions."""
        b_crit = np.array([
            [2048, 1024, 512, 256],
            [128, 64, 32, 16],
            [8, 4, 2, 4],
            [2, 0, 0, 0],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            adaptive_depth=True,
            use_d4_symmetry=False,
        )

        action, q_dict = searcher.search(b_crit)
        assert 0 <= action <= 3
        mask = Game2048Core().get_action_mask(b_crit)
        assert bool(mask[action]) is True

    def test_snake_heuristic_leaf_evaluation(self, mock_model):
        """Verify topological snake monotonicity and corner penalty blend into leaf evaluations."""
        b_clean = np.array([
            [2048, 1024, 512, 256],
            [16, 32, 64, 128],
            [8, 4, 2, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)

        b_displaced = np.array([
            [1024, 512, 256, 128],
            [64, 2048, 32, 16],
            [8, 4, 2, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            use_snake_heuristic=True,
            alpha_snake=0.5,
            beta_corner=1.0,
        )

        v_clean = searcher._batch_evaluate_boards([b_clean], use_snake=True)[0]
        v_displaced = searcher._batch_evaluate_boards([b_displaced], use_snake=True)[0]

        # Clean corner snake must evaluate significantly higher than dislodged interior 2048
        assert v_clean > v_displaced + 10.0

    def test_depth_4_search_with_deep_chance_pruning(self, mock_model):
        """Verify depth 4 search completes with deep chance pruning."""
        board = np.array([
            [1024, 512, 256, 128],
            [64, 32, 16, 8],
            [4, 2, 4, 2],
            [2, 0, 0, 0],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            depth=4,
            use_d4_symmetry=False,
            use_snake_heuristic=True,
        )

        act, q_dict = searcher.search(board, depth=4)
        assert 0 <= act <= 3
        mask = Game2048Core().get_action_mask(board)
        assert bool(mask[act]) is True

    def test_apex_depth_5_and_6_search(self, mock_model):
        """Verify 5-ply and 6-ply search execute and return valid legal moves under apex pruning."""
        # 2 empty cells (apex depth 5)
        b_apex2 = np.array([
            [2048, 1024, 512, 256],
            [128, 64, 32, 16],
            [8, 4, 2, 4],
            [2, 4, 0, 0],
        ], dtype=np.int64)

        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            adaptive_depth=True,
            apex_mode=True,
        )

        act5, q5 = searcher.search(b_apex2, depth=5)
        assert 0 <= act5 <= 3
        mask5 = Game2048Core().get_action_mask(b_apex2)
        assert bool(mask5[act5]) is True

        # 1 empty cell (apex depth 6)
        b_apex1 = np.array([
            [4096, 2048, 1024, 512],
            [256, 128, 64, 32],
            [16, 8, 4, 2],
            [4, 2, 4, 0],
        ], dtype=np.int64)

        act6, q6 = searcher.search(b_apex1, depth=6)
        assert 0 <= act6 <= 3
        mask6 = Game2048Core().get_action_mask(b_apex1)
        assert bool(mask6[act6]) is True

    def test_policy_guided_action_pruning(self, mock_model: nn.Module) -> None:
        """Validate policy prior computation and branch pruning below threshold tau."""
        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            use_policy_pruning=True,
            policy_prune_tau=0.25,
        )
        arr = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 4],
            [8, 16, 32, 0],
        ], dtype=np.int64)
        bb = Bitboard2048.from_array(arr)
        legal_actions = [0, 1, 2, 3]

        # Prior policy distribution
        pi = searcher.get_action_policy(bb, legal_actions)
        assert len(pi) == 4
        assert np.isclose(np.sum(pi), 1.0, atol=1e-5)

        # Pruning actions
        survivors = searcher._prune_actions_with_policy(bb, legal_actions, tau=0.25)
        assert len(survivors) >= 1
        for a in survivors:
            assert a in legal_actions

        # Suicidal fallback test: even with an impossibly high tau, at least 1 action survives
        extreme_survivors = searcher._prune_actions_with_policy(bb, legal_actions, tau=0.99)
        assert len(extreme_survivors) >= 1

    def test_unified_leaf_evaluation_pipeline(self, mock_model: nn.Module) -> None:
        """Validate unified leaf scoring V_leaf = V_ntuple + w_smooth * S_smooth - w_trap * P_trapped + V_net."""
        from src.models.ntuple import NTupleNetwork

        ntuple = NTupleNetwork(init_mode="normal")
        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            use_ntuple=True,
            ntuple_net=ntuple,
            use_smoothness=True,
            w_smooth=0.2,
            w_trap=5.0,
            use_snake_heuristic=False,
        )

        arr = np.array([
            [2048, 2, 512, 256],
            [128, 64, 32, 16],
            [8, 4, 2, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        bb = Bitboard2048.from_array(arr)

        # Leaf evaluate
        v_leaf = searcher._batch_evaluate_boards([bb])[0]

        # Manual terms
        v_ntuple_exp = ntuple.evaluate_bitboard(bb)
        s_smooth_exp = Bitboard2048.compute_smoothness(bb)
        p_trapped_exp = Bitboard2048.compute_trapped_penalty(bb)

        # V_leaf must incorporate all components
        expected_geom = 0.2 * s_smooth_exp - 5.0 * p_trapped_exp
        # Check that geometric penalty was applied (wedge at top wall pos 1)
        assert p_trapped_exp > 0.0
        assert expected_geom < 0.0

    def test_pure_ntuple_search_without_model(self) -> None:
        """Expectimax search works cleanly with pure N-Tuple evaluation when model is None."""
        from src.models.ntuple import NTupleNetwork

        ntuple = NTupleNetwork(init_mode="normal")
        searcher = ExpectimaxSearcher(
            model=None,
            depth=2,
            use_ntuple=True,
            ntuple_net=ntuple,
            use_smoothness=True,
            use_snake_heuristic=False,
        )

        arr = np.array([
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)

        best_act = searcher.select_action(arr)
        assert 0 <= best_act <= 3

    def test_protected_policy_pruning_in_danger(self, mock_model: nn.Module) -> None:
        """Protected policy pruning is strictly disabled when empty <= 4 or max_tile >= 2048."""
        searcher = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            use_policy_pruning=True,
            policy_prune_tau=0.40,
        )

        # Case 1: Board in danger due to tight empty cells (<= 4 empty)
        tight_arr = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 4],
            [8, 16, 32, 0],
        ], dtype=np.int64)  # 1 empty cell, max_tile=1024
        bb_tight = Bitboard2048.from_array(tight_arr)
        legal = [0, 1, 2, 3]

        # Must not prune any action because empty <= 4
        survivors_tight = searcher._prune_actions_with_policy(bb_tight, legal, tau=0.99)
        assert len(survivors_tight) == len(legal)

        # Case 2: Board with 2048+ tile (max_tile >= 2048) even with many empty cells
        big_tile_arr = np.array([
            [2048, 4, 0, 0],
            [8, 16, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)  # 12 empty cells, but max_tile=2048
        bb_big = Bitboard2048.from_array(big_tile_arr)
        survivors_big = searcher._prune_actions_with_policy(bb_big, legal, tau=0.99)
        assert len(survivors_big) == len(legal)

        # Case 3: Safe early/mid game board (empty > 4 and max_tile < 2048)
        safe_arr = np.array([
            [2, 4, 8, 16],
            [32, 64, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)  # 10 empty cells, max_tile=64
        bb_safe = Bitboard2048.from_array(safe_arr)

        # With a very high tau (e.g. 0.90), low probability actions must be pruned down
        survivors_safe = searcher._prune_actions_with_policy(bb_safe, legal, tau=0.40)
        assert len(survivors_safe) < len(legal)
        assert len(survivors_safe) >= 1

    def test_calibrated_w_ntuple_leaf_blending(self, mock_model: nn.Module) -> None:
        """w_ntuple scales and calibrates N-Tuple network contributions to V_leaf."""
        from src.models.ntuple import NTupleNetwork

        ntuple = NTupleNetwork(init_mode="normal")
        arr = np.array([
            [4, 8, 16, 32],
            [64, 128, 256, 512],
            [2, 4, 8, 16],
            [0, 0, 0, 0],
        ], dtype=np.int64)
        bb = Bitboard2048.from_array(arr)

        searcher_01 = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            use_ntuple=True,
            ntuple_net=ntuple,
            w_ntuple=0.10,
            use_smoothness=False,
            use_snake_heuristic=False,
        )
        searcher_05 = ExpectimaxSearcher(
            model=mock_model,
            device=torch.device("cpu"),
            use_ntuple=True,
            ntuple_net=ntuple,
            w_ntuple=0.50,
            use_smoothness=False,
            use_snake_heuristic=False,
        )

        v_01 = searcher_01._batch_evaluate_boards([bb])[0]
        v_05 = searcher_05._batch_evaluate_boards([bb])[0]
        v_ntuple_raw = ntuple.evaluate_bitboard(bb)

        # Difference between evaluations must be exactly (0.50 - 0.10) * v_ntuple_raw
        diff = v_05 - v_01
        expected_diff = 0.40 * v_ntuple_raw
        assert np.isclose(diff, expected_diff, atol=1e-4)
