"""Pure NumPy high-performance engine for 2048 game dynamics."""

from enum import IntEnum
from typing import Optional, Tuple
import math
import numpy as np


class Action(IntEnum):
    """Discrete action enumeration matching technical roadmap specifications."""
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


def _slide_and_merge_4(
    a: int, b: int, c: int, d: int
) -> Tuple[int, int, int, int, int, float]:
    """Pure fast primitive returning (o0, o1, o2, o3, merge_score, log_reward)."""
    v0, v1, v2, v3 = 0, 0, 0, 0
    n = 0
    if a:
        v0 = a
        n = 1
    if b:
        if n == 0:
            v0 = b
        else:
            v1 = b
        n += 1
    if c:
        if n == 0:
            v0 = c
        elif n == 1:
            v1 = c
        else:
            v2 = c
        n += 1
    if d:
        if n == 0:
            v0 = d
        elif n == 1:
            v1 = d
        elif n == 2:
            v2 = d
        else:
            v3 = d
        n += 1

    if n == 0:
        return 0, 0, 0, 0, 0, 0.0

    vals = (v0, v1, v2, v3)
    o0, o1, o2, o3 = 0, 0, 0, 0
    score = 0
    log_reward = 0.0

    idx = 0
    i = 0
    while i < n:
        curr = vals[i]
        if i + 1 < n and curr == vals[i + 1]:
            merged = curr * 2
            score += merged
            log_reward += math.log2(merged)
            if idx == 0:
                o0 = merged
            elif idx == 1:
                o1 = merged
            elif idx == 2:
                o2 = merged
            elif idx == 3:
                o3 = merged
            idx += 1
            i += 2
        else:
            if idx == 0:
                o0 = curr
            elif idx == 1:
                o1 = curr
            elif idx == 2:
                o2 = curr
            elif idx == 3:
                o3 = curr
            idx += 1
            i += 1

    return o0, o1, o2, o3, score, log_reward


def slide_and_merge_row(row: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Pure optimized primitive to slide and merge a single row of length 4 to the left.

    Adheres strictly to standard 2048 collision rules:
    - Non-zero values compact to the left.
    - Identical adjacent pairs collapse left-to-right.
    - Each tile can merge at most once per turn ([2, 2, 2, 2] -> [4, 4, 0, 0]).
    - Yields raw points (sum of merged values) and log-scaled reward (sum of log2(merged values)).

    Args:
        row: 1D array of length 4 containing tile values.

    Returns:
        Tuple containing:
            - new_row: 1D numpy array of length 4 after sliding and merging.
            - merge_score: Raw sum of newly merged tile values.
            - log_reward: Sum of log2 of newly merged tile values.
    """
    o0, o1, o2, o3, score, log_reward = _slide_and_merge_4(
        int(row[0]), int(row[1]), int(row[2]), int(row[3])
    )
    return np.array([o0, o1, o2, o3], dtype=row.dtype), score, log_reward


def _row_changes_on_left_slide(a: int, b: int, c: int, d: int) -> bool:
    """Fast check whether a row of 4 integers changes when slid left."""
    # Check if empty cell can be filled from right
    if a == 0 and (b or c or d):
        return True
    if b == 0 and (c or d):
        return True
    if c == 0 and d:
        return True
    # Check if adjacent merges are possible
    if a and (a == b or (not b and (a == c or (not c and a == d)))):
        return True
    if b and (b == c or (not c and b == d)):
        return True
    if c and c == d:
        return True
    return False


class Game2048Core:
    """High-throughput headless 2048 game core.

    Executes matrix manipulations purely in-memory using orthogonal NumPy symmetries
    (zero code duplication across directions).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize game core with optional RNG seed."""
        self.rng = np.random.default_rng(seed)
        self.board: np.ndarray = np.zeros((4, 4), dtype=np.int64)
        self.score: int = 0
        self.total_log_reward: float = 0.0
        self.steps: int = 0
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset board to all zeros and spawn two initial tiles.

        Args:
            seed: Optional seed to reseed the generator.

        Returns:
            Fresh 4x4 board state.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.total_log_reward = 0.0
        self.steps = 0

        self.spawn_tile()
        self.spawn_tile()
        return self.board.copy()

    def spawn_tile(self) -> bool:
        """Spawn a new tile (2 with p=0.9, 4 with p=0.1) in a uniformly chosen empty cell.

        Returns:
            True if a tile was spawned, False if board is full.
        """
        empty_cells = np.flatnonzero(self.board == 0)
        if len(empty_cells) == 0:
            return False

        idx = self.rng.choice(empty_cells)
        val = 2 if self.rng.random() < 0.9 else 4
        self.board.flat[idx] = val
        return True

    @staticmethod
    def _execute_slide(
        board: np.ndarray, action: int
    ) -> Tuple[np.ndarray, int, float, bool]:
        """Apply directional transformation, slide left, and apply inverse transformation.

        Args:
            board: 4x4 array of current board.
            action: Action integer (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN).

        Returns:
            Tuple of:
                - transformed_board: New 4x4 board.
                - merge_score: Raw points from merges.
                - log_reward: Log-scaled merge reward.
                - changed: Boolean flag indicating if board state changed.
        """
        if action == Action.LEFT:
            oriented = board
        elif action == Action.RIGHT:
            oriented = np.fliplr(board)
        elif action == Action.UP:
            oriented = board.T
        elif action == Action.DOWN:
            oriented = np.fliplr(board.T)
        else:
            raise ValueError(f"Invalid action: {action}")

        new_oriented = np.empty_like(oriented)
        total_merge_score = 0
        total_log_reward = 0.0

        for r in range(4):
            o0, o1, o2, o3, s_r, lr_r = _slide_and_merge_4(
                int(oriented[r, 0]),
                int(oriented[r, 1]),
                int(oriented[r, 2]),
                int(oriented[r, 3]),
            )
            new_oriented[r, 0] = o0
            new_oriented[r, 1] = o1
            new_oriented[r, 2] = o2
            new_oriented[r, 3] = o3
            total_merge_score += s_r
            total_log_reward += lr_r

        changed = not np.array_equal(oriented, new_oriented)

        if action == Action.LEFT:
            result = new_oriented
        elif action == Action.RIGHT:
            result = np.fliplr(new_oriented)
        elif action == Action.UP:
            result = new_oriented.T
        elif action == Action.DOWN:
            result = np.fliplr(new_oriented).T

        return result, total_merge_score, total_log_reward, changed

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the game.

        Args:
            action: Action (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN).

        Returns:
            Tuple of:
                - board: Current 4x4 board state.
                - reward: Log-scaled reward for this step (or -10.0 if illegal move).
                - terminated: Whether the game is terminal.
                - info: Diagnostic info dictionary.
        """
        new_board, merge_score, log_reward, changed = self._execute_slide(
            self.board, action
        )

        if not changed:
            # Illegal move: board did not change
            reward = -10.0
            info = {
                "illegal_move": True,
                "score": self.score,
                "merge_score": 0,
                "max_tile": int(np.max(self.board)),
                "steps": self.steps,
            }
            return self.board.copy(), reward, self.is_terminal(), info

        # Move was legal
        self.board = new_board
        self.score += merge_score
        self.total_log_reward += log_reward
        self.steps += 1

        self.spawn_tile()
        terminated = self.is_terminal()

        info = {
            "illegal_move": False,
            "score": self.score,
            "merge_score": merge_score,
            "max_tile": int(np.max(self.board)),
            "steps": self.steps,
        }
        return self.board.copy(), log_reward, terminated, info

    def get_action_mask(self, board: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute legal action mask.

        Simulates each of the 4 moves in-memory without modifying board state.

        Args:
            board: Optional board state to inspect; defaults to current board.

        Returns:
            1D boolean numpy array of shape (4,), True for legal actions.
        """
        b = self.board if board is None else board
        mask = np.zeros(4, dtype=bool)

        # LEFT
        for r in range(4):
            if _row_changes_on_left_slide(b[r, 0], b[r, 1], b[r, 2], b[r, 3]):
                mask[Action.LEFT] = True
                break

        # RIGHT
        for r in range(4):
            if _row_changes_on_left_slide(b[r, 3], b[r, 2], b[r, 1], b[r, 0]):
                mask[Action.RIGHT] = True
                break

        # UP
        for c in range(4):
            if _row_changes_on_left_slide(b[0, c], b[1, c], b[2, c], b[3, c]):
                mask[Action.UP] = True
                break

        # DOWN
        for c in range(4):
            if _row_changes_on_left_slide(b[3, c], b[2, c], b[1, c], b[0, c]):
                mask[Action.DOWN] = True
                break

        return mask

    def is_terminal(self, board: Optional[np.ndarray] = None) -> bool:
        """Check if game is over (no empty cells and no adjacent equal tiles).

        Args:
            board: Optional board state to check; defaults to current board.

        Returns:
            True if no legal moves remain, False otherwise.
        """
        target = self.board if board is None else board
        if np.count_nonzero(target == 0) > 0:
            return False

        # Fast adjacent comparison
        if np.any(target[:, :-1] == target[:, 1:]):
            return False
        if np.any(target[:-1, :] == target[1:, :]):
            return False

        return True
