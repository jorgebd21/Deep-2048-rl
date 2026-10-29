"""Unit and integration tests for Bitwise Manhattan Smoothness and Trapped Low-Tier Wedge Detection."""

import numpy as np
import pytest

from src.env.bitboard import (
    Bitboard2048,
    ROW_SMOOTHNESS_TABLE,
    _WALL_WEDGE_CANDIDATES,
)


class TestManhattanSmoothness:
    """Validate Manhattan Smoothness metric S_smooth(B) on hand-crafted and synthetic boards."""

    def test_smoothness_lut_properties(self) -> None:
        """Verify precomputed 16-bit row smoothness lookup table entries."""
        assert len(ROW_SMOOTHNESS_TABLE) == 65536
        # Row with all zeros
        assert ROW_SMOOTHNESS_TABLE[0] == 0.0
        # Row with identical positive tiles [2, 2, 2, 2] -> ranks [1, 1, 1, 1] -> diff 0
        r_ident = 1 | (1 << 4) | (1 << 8) | (1 << 12)
        assert ROW_SMOOTHNESS_TABLE[r_ident] == 0.0
        # Row with ranks [1, 2, 0, 0] -> diff between rank 1 and 2 is 1 (directed: 2)
        r_pair = 1 | (2 << 4)
        assert ROW_SMOOTHNESS_TABLE[r_pair] == 2.0

    def test_empty_and_single_tile_boards(self) -> None:
        """Empty and single-tile boards must yield maximum smoothness (0.0)."""
        bb_empty = 0
        assert Bitboard2048.compute_smoothness(bb_empty) == 0.0
        assert Bitboard2048.compute_smoothness(bb_empty, directed=False) == 0.0

        # Single tile at corner (0,0) with value 2048 (rank 11)
        bb_single = Bitboard2048.from_array(np.array([[2048, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
        assert Bitboard2048.compute_smoothness(bb_single) == 0.0

    def test_uniform_board_smoothness(self) -> None:
        """Board where all non-empty tiles have identical rank has zero roughness (0.0)."""
        arr = np.full((4, 4), 16, dtype=np.int64)
        bb = Bitboard2048.from_array(arr)
        assert Bitboard2048.compute_smoothness(bb) == 0.0

    def test_handcrafted_two_tile_adjacent(self) -> None:
        """Two adjacent tiles with rank 1 (val 2) and rank 3 (val 8) horizontally."""
        arr = np.zeros((4, 4), dtype=np.int64)
        arr[0, 0] = 2  # rank 1
        arr[0, 1] = 8  # rank 3
        bb = Bitboard2048.from_array(arr)
        # Directed diff: (1 to 3) is 2, (3 to 1) is 2 -> sum = 4 -> S_smooth = -4.0
        assert Bitboard2048.compute_smoothness(bb, directed=True) == -4.0
        # Undirected diff: 2 -> S_smooth = -2.0
        assert Bitboard2048.compute_smoothness(bb, directed=False) == -2.0

    def test_handcrafted_full_board(self) -> None:
        """Full hand-crafted board with known horizontal and vertical neighbor rank differences."""
        # Ranks:
        # 1  2  3  4   (row diffs: |1-2|=1, |2-3|=1, |3-4|=1 -> sum=3, directed=6)
        # 1  2  3  4   (row diffs: 3, directed=6)
        # 1  2  3  4   (row diffs: 3, directed=6)
        # 1  2  3  4   (row diffs: 3, directed=6)
        # Total horizontal directed = 24.
        # Vertical: each column has identical ranks, so vertical diff = 0!
        # Total directed roughness = 24 -> S_smooth = -24.0
        ranks = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ], dtype=np.uint8)
        bb = Bitboard2048.from_array(ranks)
        smoothness = Bitboard2048.compute_smoothness(bb, directed=True)
        assert smoothness == -24.0
        assert Bitboard2048.compute_smoothness(bb, directed=False) == -12.0

    def test_batch_smoothness_parity(self) -> None:
        """Vectorized compute_smoothness_batch must strictly match single-board compute_smoothness."""
        rng = np.random.default_rng(1234)
        sample_bbs = []
        for _ in range(50):
            rand_grid = rng.choice([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], size=(4, 4))
            sample_bbs.append(Bitboard2048.from_array(rand_grid))

        batch_directed = Bitboard2048.compute_smoothness_batch(sample_bbs, directed=True)
        batch_undirected = Bitboard2048.compute_smoothness_batch(sample_bbs, directed=False)

        for i, bb in enumerate(sample_bbs):
            expected_dir = Bitboard2048.compute_smoothness(bb, directed=True)
            expected_undir = Bitboard2048.compute_smoothness(bb, directed=False)
            assert np.isclose(batch_directed[i], expected_dir, atol=1e-5)
            assert np.isclose(batch_undirected[i], expected_undir, atol=1e-5)


class TestTrappedTileDetector:
    """Validate trapped low-tier wedge detection and structural penalty shield."""

    def test_clean_boards_have_zero_penalty(self) -> None:
        """Clean boards (empty or monotonically organized) must give zero wedges and zero penalty."""
        # 1. Empty board
        assert Bitboard2048.detect_trapped_tiles(0) == 0
        assert Bitboard2048.compute_trapped_penalty(0) == 0.0

        # 2. Clean snake monotonic board
        arr = np.array([
            [2048, 1024, 512, 256],
            [128,   64,   32,  16],
            [8,      4,    2,   0],
            [0,      0,    0,   0],
        ], dtype=np.int64)
        bb = Bitboard2048.from_array(arr)
        assert Bitboard2048.detect_trapped_tiles(bb) == 0
        assert Bitboard2048.compute_trapped_penalty(bb) == 0.0

    def test_top_wall_wedge_detection(self) -> None:
        """Low tile (value 2, rank 1) wedged on top wall between 2048 and 512."""
        arr = np.zeros((4, 4), dtype=np.int64)
        # Row 0: 2048 (rank 11), 2 (rank 1), 512 (rank 9), 256 (rank 8)
        arr[0] = [2048, 2, 512, 256]
        # Row 1: interior tile under wedge is 64 (rank 6 != 1) -> no immediate merge
        arr[1] = [128, 64, 32, 16]
        bb = Bitboard2048.from_array(arr)

        assert Bitboard2048.detect_trapped_tiles(bb) == 1
        penalty = Bitboard2048.compute_trapped_penalty(bb, penalty_per_tile=5.0)
        assert penalty == 5.0

    def test_bottom_wall_wedge_detection(self) -> None:
        """Low tile (value 4, rank 2) wedged on bottom wall between 1024 and 2048."""
        arr = np.zeros((4, 4), dtype=np.int64)
        arr[2] = [128, 64, 32, 16]
        arr[3] = [1024, 4, 2048, 512]
        bb = Bitboard2048.from_array(arr)

        assert Bitboard2048.detect_trapped_tiles(bb) == 1
        assert Bitboard2048.compute_trapped_penalty(bb, penalty_per_tile=10.0) == 10.0

    def test_left_wall_wedge_detection(self) -> None:
        """Low tile (value 2, rank 1) wedged on left wall between 1024 and 2048."""
        arr = np.zeros((4, 4), dtype=np.int64)
        # Column 0: cell (1, 0) is 2, flanked by (0, 0)=1024 and (2, 0)=2048
        arr[:, 0] = [1024, 2, 2048, 512]
        arr[:, 1] = [128, 64, 32, 16]  # interior (1, 1) is 64 != 2
        bb = Bitboard2048.from_array(arr)

        assert Bitboard2048.detect_trapped_tiles(bb) == 1
        assert Bitboard2048.compute_trapped_penalty(bb) == 1.0

    def test_right_wall_wedge_detection(self) -> None:
        """Low tile (value 4, rank 2) wedged on right wall between 512 and 1024."""
        arr = np.zeros((4, 4), dtype=np.int64)
        arr[:, 3] = [512, 4, 1024, 2048]
        arr[:, 2] = [16, 64, 32, 16]  # interior (1, 2) is 64 != 4
        bb = Bitboard2048.from_array(arr)

        assert Bitboard2048.detect_trapped_tiles(bb) == 1
        assert Bitboard2048.compute_trapped_penalty(bb) == 1.0

    def test_immediate_merge_clears_wedge_penalty(self) -> None:
        """If an immediate merge is possible perpendicular to the wall, tile is not trapped."""
        arr = np.zeros((4, 4), dtype=np.int64)
        arr[0] = [2048, 2, 512, 256]
        # Interior neighbor (1, 1) is also 2 (same rank as center) -> can merge immediately
        arr[1] = [128, 2, 32, 16]
        bb = Bitboard2048.from_array(arr)

        assert Bitboard2048.detect_trapped_tiles(bb) == 0
        assert Bitboard2048.compute_trapped_penalty(bb) == 0.0

    def test_multiple_wedges_detection(self) -> None:
        """Board with wedges on both top and bottom walls."""
        arr = np.zeros((4, 4), dtype=np.int64)
        arr[0] = [2048, 2, 512, 256]    # Wedge at (0, 1)
        arr[1] = [128, 64, 32, 16]
        arr[2] = [64, 128, 64, 32]
        arr[3] = [1024, 4, 2048, 512]   # Wedge at (3, 1)
        bb = Bitboard2048.from_array(arr)

        assert Bitboard2048.detect_trapped_tiles(bb) == 2
        assert Bitboard2048.compute_trapped_penalty(bb, penalty_per_tile=3.0) == 6.0

    def test_batch_trapped_penalty_parity(self) -> None:
        """Batch compute_trapped_penalty_batch matches single compute_trapped_penalty."""
        arr1 = np.zeros((4, 4), dtype=np.int64)
        arr1[0] = [2048, 2, 512, 256]
        arr1[1] = [128, 64, 32, 16]

        arr2 = np.zeros((4, 4), dtype=np.int64)
        arr2[0] = [2048, 1024, 512, 256]
        arr2[1] = [128, 64, 32, 16]

        bbs = [Bitboard2048.from_array(arr1), Bitboard2048.from_array(arr2)]
        batch_res = Bitboard2048.compute_trapped_penalty_batch(bbs, penalty_per_tile=5.0)

        assert np.isclose(batch_res[0], 5.0)
        assert np.isclose(batch_res[1], 0.0)
