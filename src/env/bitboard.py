"""64-bit Bitboard simulation engine for 2048 with precomputed 16-bit lookup tables (LUTs).

Represents the 4x4 game grid in a single 64-bit unsigned integer (`uint64`),
where each of the 16 cells occupies a 4-bit nibble storing the tile rank k
(where tile_value = 2^k, with 0 meaning empty).

Precomputes 16-bit tables (65,536 entries) for slide-and-merge moves,
achieving tens of millions of moves/sec per core.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np

from src.env.core import Action, _slide_and_merge_4

MASK64 = 0xFFFFFFFFFFFFFFFF
ROW_MASK = 0xFFFF


def _build_luts() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Precompute all 16-bit row transition, score, and reward lookup tables."""
    row_left = np.zeros(65536, dtype=np.uint16)
    row_right = np.zeros(65536, dtype=np.uint16)
    score_left = np.zeros(65536, dtype=np.uint32)
    score_right = np.zeros(65536, dtype=np.uint32)
    reward_left = np.zeros(65536, dtype=np.float32)
    reward_right = np.zeros(65536, dtype=np.float32)

    for R in range(65536):
        k0 = (R >> 0) & 0xF
        k1 = (R >> 4) & 0xF
        k2 = (R >> 8) & 0xF
        k3 = (R >> 12) & 0xF

        v0 = (1 << k0) if k0 > 0 else 0
        v1 = (1 << k1) if k1 > 0 else 0
        v2 = (1 << k2) if k2 > 0 else 0
        v3 = (1 << k3) if k3 > 0 else 0

        # Slide Left: v0, v1, v2, v3
        o0, o1, o2, o3, s_l, lr_l = _slide_and_merge_4(v0, v1, v2, v3)
        rk0 = min(15, int(math.log2(o0))) if o0 > 0 else 0
        rk1 = min(15, int(math.log2(o1))) if o1 > 0 else 0
        rk2 = min(15, int(math.log2(o2))) if o2 > 0 else 0
        rk3 = min(15, int(math.log2(o3))) if o3 > 0 else 0
        row_left[R] = rk0 | (rk1 << 4) | (rk2 << 8) | (rk3 << 12)
        score_left[R] = s_l
        reward_left[R] = lr_l

        # Slide Right: slide reversed row left, then reverse result
        ro3, ro2, ro1, ro0, s_r, lr_r = _slide_and_merge_4(v3, v2, v1, v0)
        rr0 = min(15, int(math.log2(ro0))) if ro0 > 0 else 0
        rr1 = min(15, int(math.log2(ro1))) if ro1 > 0 else 0
        rr2 = min(15, int(math.log2(ro2))) if ro2 > 0 else 0
        rr3 = min(15, int(math.log2(ro3))) if ro3 > 0 else 0
        row_right[R] = rr0 | (rr1 << 4) | (rr2 << 8) | (rr3 << 12)
        score_right[R] = s_r
        reward_right[R] = lr_r

    return row_left, row_right, score_left, score_right, reward_left, reward_right


# Global precomputed LUTs
(
    ROW_LEFT_TABLE,
    ROW_RIGHT_TABLE,
    ROW_SCORE_TABLE_LEFT,
    ROW_SCORE_TABLE_RIGHT,
    ROW_REWARD_TABLE_LEFT,
    ROW_REWARD_TABLE_RIGHT,
) = _build_luts()


def _build_snake_weights(base: float = 2.8) -> np.ndarray:
    """Precompute the 8 Dihedral D4 snake monotonicity weight vectors with smooth decay."""
    p1 = [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12]
    idx_grid = np.arange(16).reshape(4, 4)
    d4_grids = [
        idx_grid,
        np.rot90(idx_grid, 1),
        np.rot90(idx_grid, 2),
        np.rot90(idx_grid, 3),
        np.fliplr(idx_grid),
        np.rot90(np.fliplr(idx_grid), 1),
        np.rot90(np.fliplr(idx_grid), 2),
        np.rot90(np.fliplr(idx_grid), 3),
    ]

    snake_weights = np.zeros((8, 16), dtype=np.float32)
    norm_factor = float(base ** 15)
    for k, grid in enumerate(d4_grids):
        for i in range(16):
            cell = p1[i]
            r, c = cell // 4, cell % 4
            mapped = int(grid[r, c])
            snake_weights[k, mapped] = float(base ** (15 - i)) / norm_factor
    return snake_weights


def compute_d4_structural_scores(exps: np.ndarray) -> np.ndarray:
    """Compute Dihedral D4 snake row monotonicity, vertical connectivity, and inversion penalties.

    Evaluates structural alignment strictly on actual non-zero tiles (ignoring empty cells):
        1. Row 0 monotonicity (decreasing left-to-right between adjacent positive tiles).
        2. Row 1 monotonicity (decreasing right-to-left between adjacent positive tiles).
        3. Vertical connectivity between row 0 end [0, 3] and row 1 start [1, 3].
        4. Tile inversion penalty: penalizes smaller positive tiles trapped between larger tiles.

    Args:
        exps: Array of shape (N, 4, 4) containing tile exponent ranks [0..15].

    Returns:
        1D float32 array of shape (N,) containing maximal structural score across all 8 D4 orientations.
    """
    N = exps.shape[0]
    if N == 0:
        return np.array([], dtype=np.float32)

    r0 = exps
    r1 = np.rot90(exps, 1, (1, 2))
    r2 = np.rot90(exps, 2, (1, 2))
    r3 = np.rot90(exps, 3, (1, 2))
    f = np.flip(exps, axis=2)
    fr0 = f
    fr1 = np.rot90(f, 1, (1, 2))
    fr2 = np.rot90(f, 2, (1, 2))
    fr3 = np.rot90(f, 3, (1, 2))

    all_d4 = np.concatenate([r0, r1, r2, r3, fr0, fr1, fr2, fr3], axis=0)  # (8*N, 4, 4)

    # 1. Row 0 monotonicity between adjacent positive tiles
    g0 = all_d4[:, 0, :].astype(np.int16)
    p01 = (g0[:, 0] > 0) & (g0[:, 1] > 0)
    p12 = (g0[:, 1] > 0) & (g0[:, 2] > 0)
    p23 = (g0[:, 2] > 0) & (g0[:, 3] > 0)
    p02_skip = (g0[:, 0] > 0) & (g0[:, 1] == 0) & (g0[:, 2] > 0)
    p13_skip = (g0[:, 1] > 0) & (g0[:, 2] == 0) & (g0[:, 3] > 0)

    non_mono0 = (
        np.maximum(0, g0[:, 1] - g0[:, 0]) * p01
        + np.maximum(0, g0[:, 2] - g0[:, 1]) * p12
        + np.maximum(0, g0[:, 3] - g0[:, 2]) * p23
        + np.maximum(0, g0[:, 2] - g0[:, 0]) * p02_skip
        + np.maximum(0, g0[:, 3] - g0[:, 1]) * p13_skip
    )
    mono0_bonus = (
        (g0[:, 0] >= g0[:, 1]) * p01
        + (g0[:, 1] >= g0[:, 2]) * p12
        + (g0[:, 2] >= g0[:, 3]) * p23
        + (g0[:, 0] >= g0[:, 2]) * p02_skip
        + (g0[:, 1] >= g0[:, 3]) * p13_skip
    ) * 0.5

    # 2. Row 1 monotonicity (decreasing right-to-left: c3 >= c2 >= c1 >= c0)
    g1 = all_d4[:, 1, :].astype(np.int16)
    q32 = (g1[:, 3] > 0) & (g1[:, 2] > 0)
    q21 = (g1[:, 2] > 0) & (g1[:, 1] > 0)
    q10 = (g1[:, 1] > 0) & (g1[:, 0] > 0)
    q31_skip = (g1[:, 3] > 0) & (g1[:, 2] == 0) & (g1[:, 1] > 0)
    q20_skip = (g1[:, 2] > 0) & (g1[:, 1] == 0) & (g1[:, 0] > 0)

    non_mono1 = (
        np.maximum(0, g1[:, 2] - g1[:, 3]) * q32
        + np.maximum(0, g1[:, 1] - g1[:, 2]) * q21
        + np.maximum(0, g1[:, 0] - g1[:, 1]) * q10
        + np.maximum(0, g1[:, 1] - g1[:, 3]) * q31_skip
        + np.maximum(0, g1[:, 0] - g1[:, 2]) * q20_skip
    )
    mono1_bonus = (
        (g1[:, 3] >= g1[:, 2]) * q32
        + (g1[:, 2] >= g1[:, 1]) * q21
        + (g1[:, 1] >= g1[:, 0]) * q10
        + (g1[:, 3] >= g1[:, 1]) * q31_skip
        + (g1[:, 2] >= g1[:, 0]) * q20_skip
    ) * 0.25

    # 3. Vertical connectivity between row 0 end [0, 3] and row 1 start [1, 3]
    t03 = all_d4[:, 0, 3].astype(np.int16)
    t13 = all_d4[:, 1, 3].astype(np.int16)
    both_pos = (t03 > 0) & (t13 > 0)
    conn = np.where(both_pos, np.where(t03 >= t13, 1.0 + np.where((t03 - t13) <= 1, 1.0, 0.0), -(t13 - t03) * 1.5), 0.0)

    # 4. Trapped tile inversion penalty on boundary lines (only applies to positive tiles)
    inv1 = ((g0[:, 1] > 0) & (g0[:, 1] < g0[:, 0]) & (g0[:, 1] < g0[:, 2]) & (g0[:, 0] >= 3) & (g0[:, 2] >= 3)) * (np.minimum(g0[:, 0], g0[:, 2]) - g0[:, 1]) * 2.0
    inv2 = ((g0[:, 2] > 0) & (g0[:, 2] < g0[:, 1]) & (g0[:, 2] < g0[:, 3]) & (g0[:, 1] >= 3) & (g0[:, 3] >= 3)) * (np.minimum(g0[:, 1], g0[:, 3]) - g0[:, 2]) * 2.0

    c0 = all_d4[:, :, 0].astype(np.int16)
    inv_col = ((c0[:, 1] > 0) & (c0[:, 1] < c0[:, 0]) & (c0[:, 1] < c0[:, 2]) & (c0[:, 0] >= 3) & (c0[:, 2] >= 3)) * (np.minimum(c0[:, 0], c0[:, 2]) - c0[:, 1]) * 2.0

    scores = mono0_bonus + mono1_bonus + conn - 1.0 * non_mono0 - 0.5 * non_mono1 - inv1 - inv2 - inv_col
    scores_8n = scores.reshape(8, N)
    return np.max(scores_8n, axis=0).astype(np.float32)


SNAKE_WEIGHTS_8x16 = _build_snake_weights(base=2.8)


class Bitboard2048:
    """High-performance 64-bit Bitboard 2048 game engine."""

    def __init__(
        self,
        board: Union[int, np.ndarray, None] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize Bitboard game with board integer or 4x4 array."""
        self.rng = np.random.default_rng(seed)
        self.score: int = 0
        self.total_log_reward: float = 0.0
        self.steps: int = 0

        if board is None:
            self.board: int = 0
            self.reset(seed=seed)
        elif isinstance(board, (int, np.integer)):
            self.board = int(board) & MASK64
        elif isinstance(board, np.ndarray):
            self.board = self.from_array(board)
        else:
            raise TypeError(f"Unsupported board type: {type(board)}")

    @staticmethod
    def transpose(bb: int) -> int:
        """Transpose 4x4 nibbles using delta-swap bitwise shifts."""
        t = (bb ^ (bb >> 12)) & 0x0000F0F00000F0F0
        bb = bb ^ t ^ (t << 12)
        t = (bb ^ (bb >> 24)) & 0x00000000FF00FF00
        bb = bb ^ t ^ (t << 24)
        return bb & MASK64

    @staticmethod
    def flip_horizontal(bb: int) -> int:
        """Flip 4x4 board horizontally by reversing nibbles in each 16-bit row."""
        return (
            (((bb << 12) & 0xF000F000F000F000) & MASK64)
            | (((bb << 4) & 0x0F000F000F000F00) & MASK64)
            | (((bb >> 4) & 0x00F000F000F000F0) & MASK64)
            | (((bb >> 12) & 0x000F000F000F000F) & MASK64)
        )

    @classmethod
    def rotate90(cls, bb: int) -> int:
        """Rotate board 90 degrees counter-clockwise."""
        return cls.transpose(cls.flip_horizontal(bb))

    @classmethod
    def rotate180(cls, bb: int) -> int:
        """Rotate board 180 degrees."""
        r90 = cls.rotate90(bb)
        return cls.rotate90(r90)

    @classmethod
    def rotate270(cls, bb: int) -> int:
        """Rotate board 270 degrees counter-clockwise."""
        r180 = cls.rotate180(bb)
        return cls.rotate90(r180)

    @classmethod
    def get_d4_symmetries(cls, bb: int) -> List[int]:
        """Compute all 8 dihedral orientations (4 rotations x 2 reflections)."""
        rot0 = bb
        rot1 = cls.rotate90(rot0)
        rot2 = cls.rotate90(rot1)
        rot3 = cls.rotate90(rot2)

        flip0 = cls.flip_horizontal(rot0)
        flip1 = cls.rotate90(flip0)
        flip2 = cls.rotate90(flip1)
        flip3 = cls.rotate90(flip2)

        return [rot0, rot1, rot2, rot3, flip0, flip1, flip2, flip3]

    @classmethod
    def get_canonical(cls, bb: int) -> int:
        """Compute unique minimum canonical hash for Transposition Tables under D4 symmetry."""
        syms = cls.get_d4_symmetries(bb)
        return min(syms)

    @staticmethod
    def move_left(bb: int) -> Tuple[int, int, float, bool]:
        """Slide board LEFT using precomputed row LUTs."""
        r0 = int(ROW_LEFT_TABLE[(bb >> 0) & ROW_MASK])
        r1 = int(ROW_LEFT_TABLE[(bb >> 16) & ROW_MASK])
        r2 = int(ROW_LEFT_TABLE[(bb >> 32) & ROW_MASK])
        r3 = int(ROW_LEFT_TABLE[(bb >> 48) & ROW_MASK])

        nb = (r0 | (r1 << 16) | (r2 << 32) | (r3 << 48)) & MASK64
        changed = (nb != bb)

        score = int(
            ROW_SCORE_TABLE_LEFT[(bb >> 0) & ROW_MASK]
            + ROW_SCORE_TABLE_LEFT[(bb >> 16) & ROW_MASK]
            + ROW_SCORE_TABLE_LEFT[(bb >> 32) & ROW_MASK]
            + ROW_SCORE_TABLE_LEFT[(bb >> 48) & ROW_MASK]
        )
        reward = float(
            ROW_REWARD_TABLE_LEFT[(bb >> 0) & ROW_MASK]
            + ROW_REWARD_TABLE_LEFT[(bb >> 16) & ROW_MASK]
            + ROW_REWARD_TABLE_LEFT[(bb >> 32) & ROW_MASK]
            + ROW_REWARD_TABLE_LEFT[(bb >> 48) & ROW_MASK]
        )
        return nb, score, reward, changed

    @staticmethod
    def move_right(bb: int) -> Tuple[int, int, float, bool]:
        """Slide board RIGHT using precomputed row LUTs."""
        r0 = int(ROW_RIGHT_TABLE[(bb >> 0) & ROW_MASK])
        r1 = int(ROW_RIGHT_TABLE[(bb >> 16) & ROW_MASK])
        r2 = int(ROW_RIGHT_TABLE[(bb >> 32) & ROW_MASK])
        r3 = int(ROW_RIGHT_TABLE[(bb >> 48) & ROW_MASK])

        nb = (r0 | (r1 << 16) | (r2 << 32) | (r3 << 48)) & MASK64
        changed = (nb != bb)

        score = int(
            ROW_SCORE_TABLE_RIGHT[(bb >> 0) & ROW_MASK]
            + ROW_SCORE_TABLE_RIGHT[(bb >> 16) & ROW_MASK]
            + ROW_SCORE_TABLE_RIGHT[(bb >> 32) & ROW_MASK]
            + ROW_SCORE_TABLE_RIGHT[(bb >> 48) & ROW_MASK]
        )
        reward = float(
            ROW_REWARD_TABLE_RIGHT[(bb >> 0) & ROW_MASK]
            + ROW_REWARD_TABLE_RIGHT[(bb >> 16) & ROW_MASK]
            + ROW_REWARD_TABLE_RIGHT[(bb >> 32) & ROW_MASK]
            + ROW_REWARD_TABLE_RIGHT[(bb >> 48) & ROW_MASK]
        )
        return nb, score, reward, changed

    @classmethod
    def move_up(cls, bb: int) -> Tuple[int, int, float, bool]:
        """Slide board UP via transpose -> move_left -> transpose."""
        t_bb = cls.transpose(bb)
        nb_t, score, reward, changed = cls.move_left(t_bb)
        return cls.transpose(nb_t), score, reward, changed

    @classmethod
    def move_down(cls, bb: int) -> Tuple[int, int, float, bool]:
        """Slide board DOWN via transpose -> move_right -> transpose."""
        t_bb = cls.transpose(bb)
        nb_t, score, reward, changed = cls.move_right(t_bb)
        return cls.transpose(nb_t), score, reward, changed

    @classmethod
    def move(cls, bb: int, action: int) -> Tuple[int, int, float, bool]:
        """Apply directional action (0=LEFT, 1=RIGHT, 2=UP, 3=DOWN) on bitboard."""
        if action == Action.LEFT:
            return cls.move_left(bb)
        elif action == Action.RIGHT:
            return cls.move_right(bb)
        elif action == Action.UP:
            return cls.move_up(bb)
        elif action == Action.DOWN:
            return cls.move_down(bb)
        else:
            raise ValueError(f"Invalid action: {action}")

    @classmethod
    def is_move_legal(cls, bb: int, action: int) -> bool:
        """Check if action changes the board state."""
        _, _, _, changed = cls.move(bb, action)
        return changed

    @classmethod
    def get_action_mask(cls, bb: int) -> np.ndarray:
        """Compute legal action mask as a 4-element boolean array."""
        mask = np.zeros(4, dtype=bool)
        mask[Action.LEFT] = cls.is_move_legal(bb, Action.LEFT)
        mask[Action.RIGHT] = cls.is_move_legal(bb, Action.RIGHT)
        mask[Action.UP] = cls.is_move_legal(bb, Action.UP)
        mask[Action.DOWN] = cls.is_move_legal(bb, Action.DOWN)
        return mask

    @classmethod
    def is_terminal(cls, bb: int) -> bool:
        """Check if board has no legal moves."""
        for act in (Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN):
            if cls.is_move_legal(bb, act):
                return False
        return True

    @classmethod
    def compute_snake_score(cls, bb: int) -> float:
        """Compute maximum Dihedral D4 snake monotonicity, connectivity, and gradient score."""
        ranks = np.array([(bb >> (4 * i)) & 0xF for i in range(16)], dtype=np.float32)
        scores = np.dot(SNAKE_WEIGHTS_8x16, ranks)
        base_dot = float(np.max(scores))

        exps = cls.to_exponents_array(bb)[None, ...]
        struct_score = float(compute_d4_structural_scores(exps)[0])
        return base_dot + struct_score

    @classmethod
    def compute_corner_penalty(cls, bb: int) -> float:
        """Compute penalty for dislodged maximum tile.

        Returns:
            0.0 if maximum tile is located in any of the 4 outer corners,
            k_max * 2.0 if dislodged to the interior cells,
            k_max * 1.0 if dislodged to outer non-corner edges.
            (Only active for max tiles >= 128, i.e., rank >= 7).
        """
        ranks = [(bb >> (4 * i)) & 0xF for i in range(16)]
        k_max = max(ranks)
        if k_max < 7:
            return 0.0

        # 4 outer corners: (0,0)->0, (0,3)->3, (3,0)->12, (3,3)->15
        if ranks[0] == k_max or ranks[3] == k_max or ranks[12] == k_max or ranks[15] == k_max:
            return 0.0

        # 4 interior cells: (1,1)->5, (1,2)->6, (2,1)->9, (2,2)->10
        if ranks[5] == k_max or ranks[6] == k_max or ranks[9] == k_max or ranks[10] == k_max:
            return float(k_max * 2.0)

        # Dislodged to an outer edge
        return float(k_max * 1.0)

    @classmethod
    def get_adaptive_depth(cls, bb: int, apex_mode: bool = False) -> int:
        """Determine dynamic Expectimax lookahead depth based on board complexity.

        Args:
            bb: 64-bit integer bitboard.
            apex_mode: If True, extends search to 5-ply and 6-ply in extreme endgames.

        Returns:
            6 for Apex Endgame (empty == 1 and max tile >= 2048, apex_mode only)
            5 for Apex Endgame (empty <= 2 and max tile >= 2048, apex_mode only)
            4 for Endgame (empty <= 4 and max tile >= 1024)
            2 for Early/Open Game (empty >= 8 and max tile < 1024)
            3 for Mid Game (otherwise)
        """
        empty_count = cls.count_empty_cells(bb)
        max_rank = max((bb >> (4 * i)) & 0xF for i in range(16))
        max_tile = (1 << max_rank) if max_rank > 0 else 0

        if apex_mode and max_tile >= 2048:
            if empty_count <= 1:
                return 6
            elif empty_count <= 2:
                return 5
            elif empty_count <= 4:
                return 4
        elif empty_count <= 4 and max_tile >= 1024:
            return 4

        if empty_count >= 8 and max_tile < 1024:
            return 2
        else:
            return 3

    @classmethod
    def compute_survival_margin(cls, nb: int) -> float:
        """Compute the survival margin for an afterstate board.

        A positive value indicates healthy continuation headroom.
        A negative value indicates immediate game-over or near-certain deadlock upon tile spawn.
        """
        empty_cells = cls.get_empty_cells(nb)
        if not empty_cells:
            # Full board after move: check if any immediate merges are possible
            mask = cls.get_action_mask(nb)
            return 1.0 if np.any(mask) else -10.0

        if len(empty_cells) >= 4:
            return float(len(empty_cells))

        min_legal = 4
        total_legal = 0
        total_spawns = 0
        dead_spawns = 0

        for c in empty_cells:
            for val in (2, 4):
                s = cls.spawn_tile_at(nb, c, val)
                mask = cls.get_action_mask(s)
                n_legal = int(np.count_nonzero(mask))
                if n_legal == 0:
                    dead_spawns += 1
                if n_legal < min_legal:
                    min_legal = n_legal
                total_legal += n_legal
                total_spawns += 1

        if total_spawns == 0:
            return -10.0

        # If all spawns lead to game over, survival margin is definitely negative
        if dead_spawns == total_spawns:
            return -10.0

        # If any spawn leads to immediate death on tight boards (empty <= 2)
        if dead_spawns > 0 and len(empty_cells) <= 2:
            return -5.0 * (dead_spawns / total_spawns)

        avg_legal = total_legal / total_spawns
        return float(len(empty_cells) + avg_legal - 2.0)

    @classmethod
    def filter_corner_preserving_actions(
        cls,
        bb: int,
        legal_actions: Sequence[int],
    ) -> List[int]:
        """Prune actions that dislodge an established corner lock unless hazardous.

        If the maximum tile (rank >= 7, >= 128) is in an outer corner:
        - Filters actions that maintain the maximum tile in an outer corner.
        - Relaxes hard filtering: If ALL corner-preserving actions have negative
          survival margin (i.e. leading to immediate death or inescapable deadlock),
          allows non-corner moves (returns all legal actions) so the agent can escape.
        """
        ranks = [(bb >> (4 * i)) & 0xF for i in range(16)]
        k_max = max(ranks)
        if k_max < 7:
            return list(legal_actions)

        corner_indices = (0, 3, 12, 15)
        locked_corners = [c for c in corner_indices if ranks[c] == k_max]
        if not locked_corners:
            return list(legal_actions)

        preserving: List[int] = []
        preserving_margins: List[float] = []

        for a in legal_actions:
            nb, _, _, ch = cls.move(bb, a)
            if not ch:
                continue
            nb_ranks = [(nb >> (4 * i)) & 0xF for i in range(16)]
            if any(nb_ranks[c] == k_max for c in corner_indices):
                margin = cls.compute_survival_margin(nb)
                preserving.append(int(a))
                preserving_margins.append(margin)

        if not preserving:
            return list(legal_actions)

        # If ALL corner-preserving moves have negative survival margin,
        # allow non-corner moves (return all legal actions).
        if all(m < 0 for m in preserving_margins):
            return list(legal_actions)

        # Filter to preserving actions with safe (non-negative) survival margin
        safe_preserving = [a for a, m in zip(preserving, preserving_margins) if m >= 0]
        return safe_preserving if safe_preserving else preserving

    @staticmethod
    def get_empty_cells(bb: int) -> List[int]:
        """Extract indices [0..15] of all empty cells using bitwise operations."""
        empty = []
        for i in range(16):
            if ((bb >> (4 * i)) & 0xF) == 0:
                empty.append(i)
        return empty

    @staticmethod
    def count_empty_cells(bb: int) -> int:
        """Count number of empty cells."""
        count = 0
        for i in range(16):
            if ((bb >> (4 * i)) & 0xF) == 0:
                count += 1
        return count

    @staticmethod
    def spawn_tile_at(bb: int, cell_idx: int, tile_value: int) -> int:
        """Spawn a tile of specific value (2 or 4) into given cell index [0..15]."""
        rank = 1 if tile_value == 2 else (2 if tile_value == 4 else int(math.log2(tile_value)))
        idx = int(cell_idx)
        return (int(bb) | ((rank & 0xF) << (4 * idx))) & MASK64

    def spawn_tile(self, bb: Optional[int] = None) -> int:
        """Spawn random tile (2 with 90%, 4 with 10%) into uniform empty cell."""
        target_bb = self.board if bb is None else bb
        empty = self.get_empty_cells(target_bb)
        if len(empty) == 0:
            return target_bb

        idx = self.rng.choice(empty)
        val = 2 if self.rng.random() < 0.90 else 4
        new_bb = self.spawn_tile_at(target_bb, idx, val)

        if bb is None:
            self.board = new_bb
        return new_bb

    def reset(self, seed: Optional[int] = None) -> int:
        """Reset game state to 0 and spawn two initial tiles."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.board = 0
        self.score = 0
        self.total_log_reward = 0.0
        self.steps = 0

        self.spawn_tile()
        self.spawn_tile()
        return self.board

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Execute one step in the bitboard game."""
        new_bb, merge_score, log_reward, changed = self.move(self.board, action)

        if not changed:
            info = {
                "illegal_move": True,
                "score": self.score,
                "max_tile": int(np.max(self.to_array())),
                "steps": self.steps,
            }
            return self.board, -10.0, self.is_terminal(self.board), info

        self.board = new_bb
        self.score += merge_score
        self.total_log_reward += log_reward
        self.steps += 1

        self.spawn_tile()
        terminated = self.is_terminal(self.board)

        info = {
            "illegal_move": False,
            "score": self.score,
            "merge_score": merge_score,
            "max_tile": int(np.max(self.to_array())),
            "steps": self.steps,
        }
        return self.board, log_reward, terminated, info

    @classmethod
    def to_array(cls, bb: int) -> np.ndarray:
        """Convert 64-bit bitboard to 4x4 integer NumPy array of tile values."""
        b = np.zeros((4, 4), dtype=np.int64)
        for r in range(4):
            for c in range(4):
                k = (bb >> (4 * (4 * r + c))) & 0xF
                b[r, c] = (1 << k) if k > 0 else 0
        return b

    @classmethod
    def to_exponents_array(cls, bb: int) -> np.ndarray:
        """Convert 64-bit bitboard to 4x4 uint8 NumPy array of tile exponents [0..15]."""
        b = np.zeros((4, 4), dtype=np.uint8)
        for r in range(4):
            for c in range(4):
                k = (bb >> (4 * (4 * r + c))) & 0xF
                b[r, c] = k
        return b

    @classmethod
    def from_array(cls, arr: np.ndarray) -> int:
        """Convert 4x4 integer array of tile values (or exponents) to 64-bit bitboard."""
        bb = 0
        for r in range(4):
            for c in range(4):
                val = int(arr[r, c])
                if val == 0:
                    k = 0
                elif val <= 15 and np.issubdtype(arr.dtype, np.unsignedinteger):
                    k = val
                else:
                    k = min(15, int(math.log2(val)))
                bb |= ((k & 0xF) << (4 * (4 * r + c)))
        return bb & MASK64
