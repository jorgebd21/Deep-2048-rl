"""High-performance SOTA N-Tuple Network value evaluation architecture for 2048.

Implements N-Tuple Network value function approximation following Szubert & Jaśkowski (2014)
and Wu et al. (2014), featuring:
    - Canonical 4-tile and 6-tile geometric patterns (boundary lines, 2x3 rectangles, snake turns).
    - Base-16 addressing: index = sum_{i=0}^{k-1} rank(c_i) * 16^i.
    - Symmetric weight sharing across all 8 Dihedral D4 symmetries (rotations and reflections).
    - Sub-microsecond leaf state evaluation directly from 64-bit Bitboard representations.
    - TD(lambda) reinforcement learning update rules with afterstates.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import os
import numpy as np

# Canonical tuple patterns covering the 4x4 grid:
# Cells are indexed 0..15: row * 4 + col
# 1. 4-tile outer boundary line (Row 0):
CANONICAL_LINE_4: Tuple[int, ...] = (0, 1, 2, 3)

# 2. 6-tile rectangular 2x3 block at corner (Cells (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)):
CANONICAL_RECT_6: Tuple[int, ...] = (0, 1, 2, 4, 5, 6)

# 3. 6-tile snake turn at corner ([0,0], [0,1], [0,2], [1,2], [1,1], [1,0]):
CANONICAL_SNAKE_6: Tuple[int, ...] = (0, 1, 2, 6, 5, 4)

DEFAULT_PATTERNS: Tuple[Tuple[int, ...], ...] = (
    CANONICAL_LINE_4,
    CANONICAL_RECT_6,
    CANONICAL_SNAKE_6,
)

_SHIFTS_64 = np.arange(0, 64, 4, dtype=np.uint64)


def _transform_cell_d4(cell: int, op: int) -> int:
    """Transform cell index (0..15) according to one of 8 Dihedral D4 operations.

    Operations:
        0: Identity
        1: Rot90 counter-clockwise
        2: Rot180
        3: Rot270 counter-clockwise
        4: Flip horizontal
        5: Flip horizontal + Rot90 (transpose)
        6: Flip horizontal + Rot180 (vertical flip)
        7: Flip horizontal + Rot270 (anti-transpose)
    """
    r, c = cell // 4, cell % 4
    if op == 0:
        nr, nc = r, c
    elif op == 1:
        nr, nc = 3 - c, r
    elif op == 2:
        nr, nc = 3 - r, 3 - c
    elif op == 3:
        nr, nc = c, 3 - r
    elif op == 4:
        nr, nc = r, 3 - c
    elif op == 5:
        nr, nc = c, r
    elif op == 6:
        nr, nc = 3 - r, c
    elif op == 7:
        nr, nc = 3 - c, 3 - r
    else:
        raise ValueError(f"Invalid D4 operation: {op}")
    return nr * 4 + nc


def generate_d4_tuple_symmetries(pattern: Sequence[int]) -> List[Tuple[int, ...]]:
    """Generate all 8 Dihedral D4 symmetric variants of a tuple pattern."""
    return [tuple(_transform_cell_d4(c, op) for c in pattern) for op in range(8)]


class NTupleNetwork:
    """SOTA N-Tuple Network for 2048 leaf evaluation and afterstate value learning.

    Stores weights in flat float32 NumPy lookup tables indexed by base-16 addresses.
    Symmetric weight sharing maps all 8 D4 rotations and reflections into the same table.
    """

    def __init__(
        self,
        patterns: Optional[Sequence[Sequence[int]]] = None,
        alpha: float = 0.0025,
        gamma: float = 0.99,
        lambd: float = 0.0,
        init_mode: str = "zeros",
    ) -> None:
        """Initialize N-Tuple network with patterns, parameters, and lookup tables.

        Args:
            patterns: Sequence of tuple cell patterns. If None, uses canonical defaults
                (4-tile line, 6-tile 2x3 rect, 6-tile snake turn).
            alpha: Learning rate for TD updates.
            gamma: Discount factor.
            lambd: Eligibility trace decay factor (0.0 for TD(0)).
            init_mode: Weight initialization mode ('zeros' or 'normal').
        """
        raw_patterns = patterns if patterns is not None else DEFAULT_PATTERNS
        self.patterns: List[Tuple[int, ...]] = [tuple(p) for p in raw_patterns]
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lambd = float(lambd)

        # Precompute 8 D4 symmetries for each canonical pattern
        self.sym_tuples: List[List[Tuple[int, ...]]] = [
            generate_d4_tuple_symmetries(p) for p in self.patterns
        ]

        # Allocate lookup tables for each canonical pattern: size = 16^k
        self.tables: List[np.ndarray] = []
        for p in self.patterns:
            size = 16 ** len(p)
            if init_mode == "zeros":
                table = np.zeros(size, dtype=np.float32)
            elif init_mode == "normal":
                table = np.random.normal(0.0, 0.01, size=size).astype(np.float32)
            else:
                table = np.zeros(size, dtype=np.float32)
            self.tables.append(table)

        # Precompute integer multiplication vectors for fast base-16 address conversion
        self.mult_vectors: List[np.ndarray] = [
            np.array([16 ** j for j in range(len(p))], dtype=np.int32)
            for p in self.patterns
        ]

        # Eligibility traces for TD(lambda): sparse dictionary per pattern {index: trace}
        self.traces: List[Dict[int, float]] = [{} for _ in self.patterns]

    def reset_traces(self) -> None:
        """Reset eligibility traces between training episodes."""
        for t in self.traces:
            t.clear()

    @staticmethod
    def extract_index(bb: int, pattern: Sequence[int]) -> int:
        """Extract base-16 address index = sum_{i=0}^{k-1} rank(c_i) * 16^i from bitboard.

        Args:
            bb: 64-bit bitboard integer.
            pattern: Sequence of cell indices in 0..15.

        Returns:
            Integer index in [0, 16^k - 1].
        """
        idx = 0
        bb_int = int(bb)
        for i, c in enumerate(pattern):
            rk = (bb_int >> (4 * c)) & 0xF
            idx |= (rk << (4 * i))
        return idx

    def evaluate_bitboard(self, bb: int) -> float:
        """Rapid leaf evaluation of a single bitboard summing symmetric lookup table values.

        Runs in microsecond time by directly indexing NumPy lookup tables.

        Args:
            bb: 64-bit bitboard integer.

        Returns:
            Scalar predicted afterstate return V(bb).
        """
        bb_int = int(bb)
        # Unpack all 16 cell ranks once
        r = (
            bb_int & 0xF,
            (bb_int >> 4) & 0xF,
            (bb_int >> 8) & 0xF,
            (bb_int >> 12) & 0xF,
            (bb_int >> 16) & 0xF,
            (bb_int >> 20) & 0xF,
            (bb_int >> 24) & 0xF,
            (bb_int >> 28) & 0xF,
            (bb_int >> 32) & 0xF,
            (bb_int >> 36) & 0xF,
            (bb_int >> 40) & 0xF,
            (bb_int >> 44) & 0xF,
            (bb_int >> 48) & 0xF,
            (bb_int >> 52) & 0xF,
            (bb_int >> 56) & 0xF,
            (bb_int >> 60) & 0xF,
        )

        total = 0.0
        for pat_idx, sym_list in enumerate(self.sym_tuples):
            tbl = self.tables[pat_idx]
            k = len(self.patterns[pat_idx])
            if k == 4:
                for c0, c1, c2, c3 in sym_list:
                    idx = r[c0] | (r[c1] << 4) | (r[c2] << 8) | (r[c3] << 12)
                    total += tbl[idx]
            elif k == 6:
                for c0, c1, c2, c3, c4, c5 in sym_list:
                    idx = (
                        r[c0]
                        | (r[c1] << 4)
                        | (r[c2] << 8)
                        | (r[c3] << 12)
                        | (r[c4] << 16)
                        | (r[c5] << 20)
                    )
                    total += tbl[idx]
            else:
                for sym_p in sym_list:
                    idx = 0
                    for j, c in enumerate(sym_p):
                        idx |= (r[c] << (4 * j))
                    total += tbl[idx]
        return float(total)

    def evaluate_board(self, board: np.ndarray) -> float:
        """Evaluate a 4x4 integer NumPy board."""
        from src.env.bitboard import Bitboard2048
        return self.evaluate_bitboard(Bitboard2048.from_array(board))

    def evaluate_batch(self, bbs: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        """Vectorized evaluation of multiple bitboards returning 1D float32 array.

        Args:
            bbs: Sequence or array of 64-bit bitboard integers.

        Returns:
            1D float32 array of shape (N,) containing V(bb) for each board.
        """
        n = len(bbs)
        if n == 0:
            return np.array([], dtype=np.float32)

        bb_arr = np.asarray(bbs, dtype=np.uint64)
        # Vectorized extraction of all 16 cell ranks: shape (N, 16)
        ranks = ((bb_arr[:, None] >> _SHIFTS_64[None, :]) & np.uint64(0xF)).astype(np.int32)

        totals = np.zeros(n, dtype=np.float32)
        for pat_idx, sym_list in enumerate(self.sym_tuples):
            tbl = self.tables[pat_idx]
            k = len(self.patterns[pat_idx])
            if k == 4:
                for c0, c1, c2, c3 in sym_list:
                    idx = (
                        ranks[:, c0]
                        | (ranks[:, c1] << 4)
                        | (ranks[:, c2] << 8)
                        | (ranks[:, c3] << 12)
                    )
                    totals += tbl[idx]
            elif k == 6:
                for c0, c1, c2, c3, c4, c5 in sym_list:
                    idx = (
                        ranks[:, c0]
                        | (ranks[:, c1] << 4)
                        | (ranks[:, c2] << 8)
                        | (ranks[:, c3] << 12)
                        | (ranks[:, c4] << 16)
                        | (ranks[:, c5] << 20)
                    )
                    totals += tbl[idx]
            else:
                mult = self.mult_vectors[pat_idx]
                for sym_p in sym_list:
                    idx = np.dot(ranks[:, sym_p], mult)
                    totals += tbl[idx]
        return totals

    def update_from_delta(
        self,
        s_after: int,
        td_error: float,
        alpha: Optional[float] = None,
        scale_by_num_tuples: bool = True,
    ) -> None:
        """Apply TD update directly to active tuple tables from a precomputed TD error delta.

        Args:
            s_after: Current afterstate bitboard integer.
            td_error: Precomputed scalar TD error delta = target - V(s_after).
            alpha: Optional learning rate overriding self.alpha.
            scale_by_num_tuples: If True, divides learning rate by total active
                symmetries (e.g. 24) to keep total value update calibrated to alpha * delta.
        """
        lr = self.alpha if alpha is None else float(alpha)
        total_syms = sum(len(syms) for syms in self.sym_tuples)
        step = (lr / total_syms * td_error) if scale_by_num_tuples else (lr * td_error)

        s_after_int = int(s_after)
        r = (
            s_after_int & 0xF,
            (s_after_int >> 4) & 0xF,
            (s_after_int >> 8) & 0xF,
            (s_after_int >> 12) & 0xF,
            (s_after_int >> 16) & 0xF,
            (s_after_int >> 20) & 0xF,
            (s_after_int >> 24) & 0xF,
            (s_after_int >> 28) & 0xF,
            (s_after_int >> 32) & 0xF,
            (s_after_int >> 36) & 0xF,
            (s_after_int >> 40) & 0xF,
            (s_after_int >> 44) & 0xF,
            (s_after_int >> 48) & 0xF,
            (s_after_int >> 52) & 0xF,
            (s_after_int >> 56) & 0xF,
            (s_after_int >> 60) & 0xF,
        )

        for pat_idx, sym_list in enumerate(self.sym_tuples):
            tbl = self.tables[pat_idx]
            k = len(self.patterns[pat_idx])
            if k == 4:
                for c0, c1, c2, c3 in sym_list:
                    idx = r[c0] | (r[c1] << 4) | (r[c2] << 8) | (r[c3] << 12)
                    tbl[idx] += step
            elif k == 6:
                for c0, c1, c2, c3, c4, c5 in sym_list:
                    idx = (
                        r[c0]
                        | (r[c1] << 4)
                        | (r[c2] << 8)
                        | (r[c3] << 12)
                        | (r[c4] << 16)
                        | (r[c5] << 20)
                    )
                    tbl[idx] += step
            else:
                for sym_p in sym_list:
                    idx = 0
                    for j, c in enumerate(sym_p):
                        idx |= (r[c] << (4 * j))
                    tbl[idx] += step

    def update(
        self,
        s_after: int,
        reward: float,
        s_after_next: Optional[int],
        done: bool = False,
    ) -> float:
        """Perform TD(lambda) learning update on afterstate value tables.

        Updates weights according to:
            Delta W = alpha * (r + gamma * V(s') - V(s)) * e_t

        Args:
            s_after: Current afterstate bitboard integer.
            reward: Immediate transition reward r.
            s_after_next: Next afterstate bitboard integer (or None if terminal/done).
            done: Whether the transition led to terminal state.

        Returns:
            Scalar TD error delta.
        """
        v_cur = self.evaluate_bitboard(s_after)
        if done or s_after_next is None:
            v_next = 0.0
        else:
            v_next = self.evaluate_bitboard(s_after_next)

        td_error = float(reward + self.gamma * v_next - v_cur)

        # Extract active indices for current state s_after
        s_after_int = int(s_after)
        r = (
            s_after_int & 0xF,
            (s_after_int >> 4) & 0xF,
            (s_after_int >> 8) & 0xF,
            (s_after_int >> 12) & 0xF,
            (s_after_int >> 16) & 0xF,
            (s_after_int >> 20) & 0xF,
            (s_after_int >> 24) & 0xF,
            (s_after_int >> 28) & 0xF,
            (s_after_int >> 32) & 0xF,
            (s_after_int >> 36) & 0xF,
            (s_after_int >> 40) & 0xF,
            (s_after_int >> 44) & 0xF,
            (s_after_int >> 48) & 0xF,
            (s_after_int >> 52) & 0xF,
            (s_after_int >> 56) & 0xF,
            (s_after_int >> 60) & 0xF,
        )

        if self.lambd <= 0.0:
            # TD(0): direct gradient step on active features
            step = self.alpha * td_error
            for pat_idx, sym_list in enumerate(self.sym_tuples):
                tbl = self.tables[pat_idx]
                k = len(self.patterns[pat_idx])
                if k == 4:
                    for c0, c1, c2, c3 in sym_list:
                        idx = r[c0] | (r[c1] << 4) | (r[c2] << 8) | (r[c3] << 12)
                        tbl[idx] += step
                elif k == 6:
                    for c0, c1, c2, c3, c4, c5 in sym_list:
                        idx = (
                            r[c0]
                            | (r[c1] << 4)
                            | (r[c2] << 8)
                            | (r[c3] << 12)
                            | (r[c4] << 16)
                            | (r[c5] << 20)
                        )
                        tbl[idx] += step
                else:
                    for sym_p in sym_list:
                        idx = 0
                        for j, c in enumerate(sym_p):
                            idx |= (r[c] << (4 * j))
                        tbl[idx] += step
        else:
            # TD(lambda): accumulate and decay eligibility traces
            decay = self.gamma * self.lambd
            for pat_idx, sym_list in enumerate(self.sym_tuples):
                tr_dict = self.traces[pat_idx]
                tbl = self.tables[pat_idx]

                # Decay existing traces
                for idx in list(tr_dict.keys()):
                    tr_dict[idx] *= decay
                    if tr_dict[idx] < 1e-5:
                        del tr_dict[idx]

                # Accumulate current active features
                k = len(self.patterns[pat_idx])
                if k == 4:
                    for c0, c1, c2, c3 in sym_list:
                        idx = r[c0] | (r[c1] << 4) | (r[c2] << 8) | (r[c3] << 12)
                        tr_dict[idx] = tr_dict.get(idx, 0.0) + 1.0
                elif k == 6:
                    for c0, c1, c2, c3, c4, c5 in sym_list:
                        idx = (
                            r[c0]
                            | (r[c1] << 4)
                            | (r[c2] << 8)
                            | (r[c3] << 12)
                            | (r[c4] << 16)
                            | (r[c5] << 20)
                        )
                        tr_dict[idx] = tr_dict.get(idx, 0.0) + 1.0
                else:
                    for sym_p in sym_list:
                        idx = 0
                        for j, c in enumerate(sym_p):
                            idx |= (r[c] << (4 * j))
                        tr_dict[idx] = tr_dict.get(idx, 0.0) + 1.0

                # Apply weight update across active traces
                step = self.alpha * td_error
                for idx, e in tr_dict.items():
                    tbl[idx] += step * e

        if done:
            self.reset_traces()

        return td_error

    def save(self, filepath: str) -> None:
        """Save N-Tuple lookup tables to compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        save_dict = {f"table_{i}": t for i, t in enumerate(self.tables)}
        save_dict["alpha"] = np.array(self.alpha, dtype=np.float64)
        save_dict["gamma"] = np.array(self.gamma, dtype=np.float64)
        save_dict["lambd"] = np.array(self.lambd, dtype=np.float64)
        np.savez_compressed(filepath, **save_dict)

    def load(self, filepath: str) -> None:
        """Load N-Tuple lookup tables from compressed .npz archive."""
        data = np.load(filepath)
        for i in range(len(self.tables)):
            key = f"table_{i}"
            if key in data:
                self.tables[i] = data[key].astype(np.float32)
        if "alpha" in data:
            self.alpha = float(data["alpha"])
        if "gamma" in data:
            self.gamma = float(data["gamma"])
        if "lambd" in data:
            self.lambd = float(data["lambd"])
