"""Unit and integration tests for Expectimax / Lookahead search."""

import numpy as np
import pytest
import torch

from src.agent.search import (
    ExpectimaxSearcher,
    expectimax_search,
    generate_d4_masks,
    generate_d4_tensors,
)
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
