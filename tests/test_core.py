"""Unit tests for Game2048Core engine."""

import time
import numpy as np
import pytest

from src.env.core import Action, Game2048Core, slide_and_merge_row


class TestRowPrimitives:
    """Test suite for slide_and_merge_row primitive."""

    def test_empty_row(self):
        row = np.array([0, 0, 0, 0], dtype=np.int64)
        new_row, score, log_r = slide_and_merge_row(row)
        np.testing.assert_array_equal(new_row, [0, 0, 0, 0])
        assert score == 0
        assert log_r == 0.0

    def test_single_element_slide(self):
        row = np.array([0, 2, 0, 0], dtype=np.int64)
        new_row, score, log_r = slide_and_merge_row(row)
        np.testing.assert_array_equal(new_row, [2, 0, 0, 0])
        assert score == 0
        assert log_r == 0.0

    def test_two_adjacent_merge(self):
        row = np.array([2, 2, 0, 0], dtype=np.int64)
        new_row, score, log_r = slide_and_merge_row(row)
        np.testing.assert_array_equal(new_row, [4, 0, 0, 0])
        assert score == 4
        assert log_r == 2.0  # log2(4) = 2

    def test_two_separated_merge(self):
        row = np.array([2, 0, 2, 0], dtype=np.int64)
        new_row, score, log_r = slide_and_merge_row(row)
        np.testing.assert_array_equal(new_row, [4, 0, 0, 0])
        assert score == 4
        assert log_r == 2.0

    def test_cascade_four_identical_tiles(self):
        # [2, 2, 2, 2] -> [4, 4, 0, 0] (no double collapse to 8)
        row = np.array([2, 2, 2, 2], dtype=np.int64)
        new_row, score, log_r = slide_and_merge_row(row)
        np.testing.assert_array_equal(new_row, [4, 4, 0, 0])
        assert score == 8
        assert log_r == 4.0  # log2(4) + log2(4) = 4

    def test_cascade_four_fours(self):
        # [4, 4, 4, 4] -> [8, 8, 0, 0]
        row = np.array([4, 4, 4, 4], dtype=np.int64)
        new_row, score, log_r = slide_and_merge_row(row)
        np.testing.assert_array_equal(new_row, [8, 8, 0, 0])
        assert score == 16
        assert log_r == 6.0  # log2(8) + log2(8) = 6

    def test_three_identical_tiles_left_priority(self):
        # [2, 2, 2, 0] -> [4, 2, 0, 0]
        row = np.array([2, 2, 2, 0], dtype=np.int64)
        new_row, score, log_r = slide_and_merge_row(row)
        np.testing.assert_array_equal(new_row, [4, 2, 0, 0])
        assert score == 4
        assert log_r == 2.0

    def test_no_merge_different_values(self):
        row = np.array([2, 4, 8, 16], dtype=np.int64)
        new_row, score, log_r = slide_and_merge_row(row)
        np.testing.assert_array_equal(new_row, [2, 4, 8, 16])
        assert score == 0
        assert log_r == 0.0


class TestCoreMechanics:
    """Test suite for full 4x4 matrix game mechanics and directions."""

    def test_initial_board_generation(self):
        core = Game2048Core(seed=42)
        assert core.board.shape == (4, 4)
        non_zero = core.board[core.board != 0]
        assert len(non_zero) == 2
        for val in non_zero:
            assert val in (2, 4)
        assert core.score == 0
        assert core.steps == 0

    def test_orthogonal_directions(self):
        # Setup specific board:
        # [2, 0, 0, 2]
        # [0, 4, 4, 0]
        # [0, 0, 0, 0]
        # [0, 0, 0, 0]
        core = Game2048Core(seed=10)
        core.board = np.array(
            [[2, 0, 0, 2], [0, 4, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int64,
        )

        # Move LEFT:
        # row 0 becomes [4, 0, 0, 0], row 1 becomes [8, 0, 0, 0]
        res, score, log_r, changed = core._execute_slide(core.board, Action.LEFT)
        assert changed
        assert score == 4 + 8
        np.testing.assert_array_equal(res[0], [4, 0, 0, 0])
        np.testing.assert_array_equal(res[1], [8, 0, 0, 0])

        # Move RIGHT on same initial board:
        # row 0 becomes [0, 0, 0, 4], row 1 becomes [0, 0, 0, 8]
        res_r, score_r, _, changed_r = core._execute_slide(core.board, Action.RIGHT)
        assert changed_r
        assert score_r == 12
        np.testing.assert_array_equal(res_r[0], [0, 0, 0, 4])
        np.testing.assert_array_equal(res_r[1], [0, 0, 0, 8])

    def test_vertical_directions(self):
        core = Game2048Core(seed=11)
        core.board = np.array(
            [[2, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0]],
            dtype=np.int64,
        )

        # Move UP:
        # column 0: [2, 2, 4, 4] -> [4, 8, 0, 0]
        res_u, score_u, log_u, changed_u = core._execute_slide(core.board, Action.UP)
        assert changed_u
        assert score_u == 4 + 8
        np.testing.assert_array_equal(res_u[:, 0], [4, 8, 0, 0])

        # Move DOWN:
        # column 0: [2, 2, 4, 4] -> [0, 0, 4, 8]
        res_d, score_d, log_d, changed_d = core._execute_slide(core.board, Action.DOWN)
        assert changed_d
        assert score_d == 4 + 8
        np.testing.assert_array_equal(res_d[:, 0], [0, 0, 4, 8])

    def test_illegal_move_detection(self):
        core = Game2048Core(seed=12)
        # All tiles packed to the left
        core.board = np.array(
            [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int64,
        )
        # Action LEFT is illegal because it does not modify board
        board_before = core.board.copy()
        new_board, reward, term, info = core.step(Action.LEFT)
        assert info["illegal_move"] is True
        assert reward == -10.0
        np.testing.assert_array_equal(new_board, board_before)

    def test_action_mask(self):
        core = Game2048Core(seed=13)
        core.board = np.array(
            [[2, 4, 8, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int64,
        )
        mask = core.get_action_mask()
        assert not mask[Action.LEFT]
        assert bool(mask[Action.RIGHT]) is True
        assert not mask[Action.UP]
        assert bool(mask[Action.DOWN]) is True

    def test_terminal_detection(self):
        core = Game2048Core(seed=14)
        # Board with empty cell -> not terminal
        core.board = np.ones((4, 4), dtype=np.int64) * 2
        core.board[3, 3] = 0
        assert not core.is_terminal()

        # Board full with adjacent merges -> not terminal
        core.board[3, 3] = 4
        assert not core.is_terminal()  # many 2s adjacent

        # Checkerboard full of non-matching values -> terminal!
        core.board = np.array(
            [
                [2, 4, 2, 4],
                [4, 2, 4, 2],
                [2, 4, 2, 4],
                [4, 2, 4, 2],
            ],
            dtype=np.int64,
        )
        assert core.is_terminal()
        assert not np.any(core.get_action_mask())

    def test_spawn_distribution(self):
        core = Game2048Core(seed=999)
        spawns = []
        for _ in range(1000):
            core.board = np.zeros((4, 4), dtype=np.int64)
            core.spawn_tile()
            val = core.board[core.board != 0][0]
            spawns.append(val)
        twos = sum(1 for v in spawns if v == 2)
        fours = sum(1 for v in spawns if v == 4)
        assert 850 <= twos <= 950  # ~90%
        assert 50 <= fours <= 150  # ~10%

    def test_simulation_throughput(self):
        """Verify simulation throughput meets performance criteria (>10,000 steps/sec, typically 15k-20k+)."""
        core = Game2048Core(seed=1234)
        num_steps = 25000
        start = time.perf_counter()
        steps_executed = 0
        for _ in range(num_steps):
            mask = core.get_action_mask()
            valid_actions = np.flatnonzero(mask)
            if len(valid_actions) == 0:
                core.reset()
                continue
            act = valid_actions[0]
            core.step(act)
            steps_executed += 1
        elapsed = time.perf_counter() - start
        throughput = steps_executed / elapsed
        print(f"\nThroughput benchmark: {throughput:,.0f} steps/second")
        assert throughput > 10000, f"Expected >10,000 steps/sec, got {throughput:.0f}"
