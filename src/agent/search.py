"""Lightweight Expectimax & Lookahead search for 2048 DRL agents.

Implements batched n-ply Expectimax search over deterministic slide-and-merge
actions and stochastic tile spawns (90% for 2, 10% for 4). Evaluates leaf states
using a trained Dueling Double DQN network in a single GPU forward pass with:
    - 64-bit Bitboard simulation engine with precomputed 16-bit LUTs.
    - Transposition Table (TT) with 64-bit canonical Dihedral D4 keys.
    - Dihedral D4 (8-fold rotation/reflection) symmetry invariance.
    - Probability pruning (p < 0.001) and beam search.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.core import Action, Game2048Core
from src.env.bitboard import Bitboard2048
from src.env.wrapper import _POWER_LUT, MAX_TILE_VAL, one_hot_encode_board
from src.utils.cuda_utils import get_device

# Dihedral D4 action index permutations:
# Maps original action index to transformed action index on rotated/reflected boards
# Actions: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
_D4_ACTION_PERMUTATIONS = [
    [0, 1, 2, 3],  # rot0: identity (LEFT, RIGHT, UP, DOWN)
    [2, 3, 1, 0],  # rot1: rot90
    [1, 0, 3, 2],  # rot2: rot180
    [3, 2, 0, 1],  # rot3: rot270
    [1, 0, 2, 3],  # flip_rot0: flip horizontal
    [2, 3, 0, 1],  # flip_rot1: flip + rot90
    [0, 1, 3, 2],  # flip_rot2: flip + rot180
    [3, 2, 1, 0],  # flip_rot3: flip + rot270
]

_SHIFTS_64 = np.arange(0, 64, 4, dtype=np.uint64).reshape(1, 16)


def generate_d4_tensors(x: torch.Tensor) -> torch.Tensor:
    """Generate all 8 dihedral transformations (4 rotations x 2 reflections) on GPU.

    Args:
        x: Float or long tensor of shape (B, C, 4, 4) or (B, 4, 4).

    Returns:
        Concatenated tensor of shape (8 * B, C, 4, 4) or (8 * B, 4, 4).
    """
    rot0 = x
    rot1 = torch.rot90(x, 1, [-2, -1])
    rot2 = torch.rot90(x, 2, [-2, -1])
    rot3 = torch.rot90(x, 3, [-2, -1])

    flip = torch.flip(x, [-1])
    f_rot0 = flip
    f_rot1 = torch.rot90(flip, 1, [-2, -1])
    f_rot2 = torch.rot90(flip, 2, [-2, -1])
    f_rot3 = torch.rot90(flip, 3, [-2, -1])

    return torch.cat([rot0, rot1, rot2, rot3, f_rot0, f_rot1, f_rot2, f_rot3], dim=0)


def generate_d4_masks(masks: torch.Tensor) -> torch.Tensor:
    """Generate transformed legal action masks corresponding to the 8 D4 orientations.

    Args:
        masks: Boolean tensor of shape (B, 4).

    Returns:
        Concatenated boolean tensor of shape (8 * B, 4).
    """
    return torch.cat([masks[:, perm] for perm in _D4_ACTION_PERMUTATIONS], dim=0)


class ExpectimaxSearcher:
    """High-throughput Expectimax searcher leveraging bitboard simulations and batched GPU evaluation.

    Supports:
        - Depth 0: Pure greedy evaluation via direct Q(s, a) forward pass.
        - Depth 1: 1-ply lookahead (action slide -> evaluate afterstate).
        - Depth 2: 2-ply Expectimax (action slide -> chance spawns 2/4 -> evaluate leaf states).
        - Depth 3: 3-ply Expectimax (action 1 -> chance 1 -> action 2 -> evaluate leaf states).
        - Depth 4: 4-ply Expectimax (action 1 -> chance 1 -> action 2 -> chance 2 -> evaluate leaf states).
        - Transposition Table (TT) with 64-bit canonical Dihedral D4 hashing.
        - Dihedral D4 Symmetry: 8-fold isometric ensembling for rotation/reflection invariance.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        gamma: float = 0.99,
        depth: int = 2,
        max_empty_samples: Optional[int] = None,
        use_d4_symmetry: bool = False,
        tt_max_size: int = 500_000,
    ) -> None:
        """Initialize searcher with model and search configuration.

        Args:
            model: PyTorch policy or Q-network (e.g. Dueling2048Net).
            device: Compute device (defaults to GPU if available).
            gamma: Discount factor for Bellman expectation calculation.
            depth: Default search depth (0 for greedy, 1 for 1-ply, 2 for 2-ply, 3 for 3-ply, 4 for 4-ply).
            max_empty_samples: Optional maximum number of empty cells to sample per chance node.
                None evaluates all empty cells deterministically.
            use_d4_symmetry: Whether to enable 8-fold Dihedral D4 symmetry ensembling.
            tt_max_size: Maximum entries in the Transposition Table before pruning.
        """
        self.model = model
        self.device = device or get_device(prefer_gpu=True)
        self.gamma = gamma
        self.depth = depth
        self.max_empty_samples = max_empty_samples
        self.use_d4_symmetry = use_d4_symmetry
        self.tt_max_size = tt_max_size
        self.tt: Dict[int, float] = {}
        self.core = Game2048Core()

    def clear_tt(self) -> None:
        """Clear all cached entries in the Transposition Table."""
        self.tt.clear()

    def select_action(
        self,
        board: Union[np.ndarray, int],
        action_mask: Optional[np.ndarray] = None,
        depth: Optional[int] = None,
        use_d4_symmetry: Optional[bool] = None,
    ) -> int:
        """Select best action using lookahead search.

        Args:
            board: Current 4x4 integer game board or 64-bit bitboard int.
            action_mask: Optional boolean array of legal actions.
            depth: Optional search depth overriding instance default.
            use_d4_symmetry: Optional D4 symmetry toggle overriding instance default.

        Returns:
            Selected integer action in {0, 1, 2, 3}.
        """
        best_action, _ = self.search(
            board,
            action_mask=action_mask,
            depth=depth,
            use_d4_symmetry=use_d4_symmetry,
        )
        return best_action

    def search(
        self,
        board: Union[np.ndarray, int],
        action_mask: Optional[np.ndarray] = None,
        depth: Optional[int] = None,
        use_d4_symmetry: Optional[bool] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """Perform Expectimax search from current board state.

        Args:
            board: 4x4 integer numpy array or 64-bit bitboard integer.
            action_mask: Optional boolean array of shape (4,). If None, calculated via bitboard/core.
            depth: Search depth (0, 1, 2, 3, or 4). Defaults to self.depth.
            use_d4_symmetry: Optional D4 toggle overriding instance setting.

        Returns:
            Tuple of (best_action, q_values_dict).
        """
        d = self.depth if depth is None else depth
        d4_active = self.use_d4_symmetry if use_d4_symmetry is None else use_d4_symmetry

        # Prune cache between moves if maximum capacity is exceeded
        if len(self.tt) > self.tt_max_size:
            self.tt.clear()

        # Convert to bitboard if numpy array
        if isinstance(board, np.ndarray):
            bb = Bitboard2048.from_array(board)
            np_board = board
        else:
            bb = int(board)
            np_board = None

        if action_mask is None:
            mask = Bitboard2048.get_action_mask(bb)
        else:
            mask = np.asarray(action_mask, dtype=bool)

        legal_actions = np.flatnonzero(mask)
        if len(legal_actions) == 0:
            return 0, {a: -1e9 for a in range(4)}

        # Short-circuit: if only one action is legal, no search can alter the outcome
        if len(legal_actions) == 1:
            only_act = int(legal_actions[0])
            return only_act, {a: (0.0 if a == only_act else -1e9) for a in range(4)}

        if d <= 0:
            if np_board is None:
                np_board = Bitboard2048.to_array(bb)
            return self._search_greedy(np_board, mask, use_d4=d4_active)
        elif d == 1:
            return self._search_depth_1(bb, legal_actions, use_d4=d4_active)
        elif d == 2:
            return self._search_depth_2(bb, legal_actions, use_d4=d4_active)
        elif d == 3:
            return self._search_depth_3(bb, legal_actions, use_d4=d4_active)
        elif d >= 4:
            return self._search_depth_4(bb, legal_actions, use_d4=d4_active)
        else:
            return self._search_depth_2(bb, legal_actions, use_d4=d4_active)

    def _ensure_evaluated(self, bitboards: Iterable[int], use_d4: bool) -> None:
        """Evaluate un-cached canonical bitboards in a single batched GPU forward pass."""
        unrated: Dict[int, int] = {}
        for b in bitboards:
            canon = Bitboard2048.get_canonical(int(b))
            if canon not in self.tt and canon not in unrated:
                unrated[canon] = int(b)

        if not unrated:
            return

        canon_keys = list(unrated.keys())
        rep_boards = list(unrated.values())
        v_vals = self._batch_evaluate_boards(rep_boards, use_d4=use_d4)

        for k, v in zip(canon_keys, v_vals):
            self.tt[k] = float(v)

    def _search_greedy(
        self,
        board: np.ndarray,
        mask: np.ndarray,
        use_d4: bool = False,
    ) -> Tuple[int, Dict[int, float]]:
        """Evaluate root state directly with model (depth 0), optionally symmetrized over D4."""
        obs = one_hot_encode_board(board)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=(self.device.type == "cuda"),
            ):
                if use_d4:
                    batch_d4 = generate_d4_tensors(obs_t)
                    batch_masks = generate_d4_masks(mask_t)
                    q_vals_d4 = self.model(batch_d4, action_mask=batch_masks)  # (8, 4)

                    q_orig_accum = torch.zeros(4, device=self.device)
                    for k, perm in enumerate(_D4_ACTION_PERMUTATIONS):
                        for j in range(4):
                            q_orig_accum[perm[j]] += q_vals_d4[k, j]
                    q_vals = (q_orig_accum / 8.0).cpu().numpy()
                else:
                    q_vals = (
                        self.model(obs_t, action_mask=mask_t)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )

        best_action = int(np.argmax(q_vals))
        q_dict = {a: float(q_vals[a]) for a in range(4)}
        return best_action, q_dict

    def _search_depth_1(
        self,
        bb: int,
        legal_actions: Sequence[int],
        use_d4: bool = False,
    ) -> Tuple[int, Dict[int, float]]:
        """1-ply lookahead: evaluate afterstate for each legal action."""
        after_boards: List[int] = []
        imm_rewards: List[float] = []
        valid_actions: List[int] = []

        for a in legal_actions:
            nb, _, lr, changed = Bitboard2048.move(bb, a)
            if changed:
                after_boards.append(nb)
                imm_rewards.append(lr)
                valid_actions.append(a)

        if len(after_boards) == 0:
            return int(legal_actions[0]), {a: -1e9 for a in range(4)}

        self._ensure_evaluated(after_boards, use_d4=use_d4)

        q_dict: Dict[int, float] = {a: -1e9 for a in range(4)}
        for a, lr, nb in zip(valid_actions, imm_rewards, after_boards):
            v = self.tt.get(Bitboard2048.get_canonical(nb), 0.0)
            q_dict[a] = lr + self.gamma * v

        best_action = max(valid_actions, key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _search_depth_2(
        self,
        bb: int,
        legal_actions: Sequence[int],
        use_d4: bool = False,
    ) -> Tuple[int, Dict[int, float]]:
        """2-ply Expectimax: action slide -> chance spawns 2/4 -> batched leaf evaluation.

        For each legal action:
            1. Deterministic slide-and-merge yields s_after and immediate reward r.
            2. Branches over all empty cells with spawn probabilities (90% for 2, 10% for 4).
            3. Evaluates all unique un-cached leaf states via Transposition Table and batched GPU pass.
            4. Returns action maximizing Q(s, a) = r + gamma * sum(P(spawn) * V(s')).
        """
        all_leaves: List[int] = []
        tree: Dict[int, Tuple[float, List[Tuple[float, int]]]] = {}

        for a in legal_actions:
            nb, _, lr, changed = Bitboard2048.move(bb, a)
            if not changed:
                continue

            empty = Bitboard2048.get_empty_cells(nb)
            num_empty = len(empty)

            if num_empty == 0:
                tree[a] = (lr, [(1.0, nb)])
                all_leaves.append(nb)
            else:
                if (
                    self.max_empty_samples is not None
                    and num_empty > self.max_empty_samples
                ):
                    sampled = np.random.choice(
                        empty, self.max_empty_samples, replace=False
                    )
                else:
                    sampled = empty

                w_base = 1.0 / len(sampled)
                branches: List[Tuple[float, int]] = []
                for c in sampled:
                    b2 = Bitboard2048.spawn_tile_at(nb, c, 2)
                    branches.append((0.90 * w_base, b2))
                    all_leaves.append(b2)

                    b4 = Bitboard2048.spawn_tile_at(nb, c, 4)
                    branches.append((0.10 * w_base, b4))
                    all_leaves.append(b4)

                tree[a] = (lr, branches)

        if not tree:
            return int(legal_actions[0]), {a: -1e9 for a in range(4)}

        self._ensure_evaluated(all_leaves, use_d4=use_d4)

        q_dict: Dict[int, float] = {a: -1e9 for a in range(4)}
        for a, (lr, branches) in tree.items():
            exp_v = sum(w * self.tt.get(Bitboard2048.get_canonical(b), 0.0) for w, b in branches)
            q_dict[a] = lr + self.gamma * exp_v

        best_action = max(tree.keys(), key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _search_depth_3(
        self,
        bb: int,
        legal_actions: Sequence[int],
        use_d4: bool = False,
    ) -> Tuple[int, Dict[int, float]]:
        """3-ply Expectimax: Action 1 -> Chance 1 -> Action 2 -> Leaf Evaluation.

        Backpropagates:
            V(s_chance) = max_{a2} [ r2 + gamma * V(s_after2) ]
            Q(s, a1) = r1 + gamma * sum(P(spawn) * V(s_chance))
        """
        all_leaves: List[int] = []
        # tree structure: a1 -> (r1, list of (w_chance, list of (r2, s_after2)))
        tree: Dict[int, Tuple[float, List[Tuple[float, List[Tuple[float, int]]]]]] = {}

        max_chance = (
            6 if self.max_empty_samples is None else min(6, self.max_empty_samples)
        )

        for a1 in legal_actions:
            nb1, _, lr1, changed1 = Bitboard2048.move(bb, a1)
            if not changed1:
                continue

            empty = Bitboard2048.get_empty_cells(nb1)
            num_empty = len(empty)

            if num_empty == 0:
                # Terminal or full board after slide
                tree[a1] = (lr1, [(1.0, [(0.0, nb1)])])
                all_leaves.append(nb1)
                continue

            if num_empty > max_chance:
                sampled = np.random.choice(empty, max_chance, replace=False)
            else:
                sampled = empty

            w_base = 1.0 / len(sampled)
            chance_branches: List[Tuple[float, List[Tuple[float, int]]]] = []

            for c in sampled:
                for val, p_val in [(2, 0.90), (4, 0.10)]:
                    s_chance = Bitboard2048.spawn_tile_at(nb1, c, val)
                    w_chance = p_val * w_base

                    # Legal actions from s_chance
                    mask2 = Bitboard2048.get_action_mask(s_chance)
                    legal2 = np.flatnonzero(mask2)

                    if len(legal2) == 0:
                        # Terminal chance outcome: value 0.0
                        chance_branches.append((w_chance, []))
                    else:
                        act2_moves: List[Tuple[float, int]] = []
                        for a2 in legal2:
                            nb2, _, lr2, ch2 = Bitboard2048.move(s_chance, a2)
                            if ch2:
                                act2_moves.append((lr2, nb2))
                                all_leaves.append(nb2)
                        chance_branches.append((w_chance, act2_moves))

            tree[a1] = (lr1, chance_branches)

        if not tree:
            return self._search_depth_2(bb, legal_actions, use_d4=use_d4)

        self._ensure_evaluated(all_leaves, use_d4=use_d4)

        q_dict: Dict[int, float] = {a: -1e9 for a in range(4)}
        for a1, (lr1, chance_branches) in tree.items():
            exp_v = 0.0
            for w_chance, act2_moves in chance_branches:
                if not act2_moves:
                    best_q2 = 0.0
                else:
                    best_q2 = max(
                        lr2 + self.gamma * self.tt.get(Bitboard2048.get_canonical(nb2), 0.0)
                        for lr2, nb2 in act2_moves
                    )
                exp_v += w_chance * best_q2
            q_dict[a1] = lr1 + self.gamma * exp_v

        best_action = max(tree.keys(), key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _search_depth_4(
        self,
        bb: int,
        legal_actions: Sequence[int],
        use_d4: bool = False,
    ) -> Tuple[int, Dict[int, float]]:
        """4-ply Expectimax: Action 1 -> Chance 1 -> Action 2 -> Chance 2 -> Leaf Evaluation.

        Applies probability pruning (pruning branches with p < 0.001) and beam search.
        """
        all_leaves: List[int] = []
        # tree: a1 -> (lr1, list of (w_chance1, list of (lr2, list of (w_chance2, s_leaf))))
        tree: Dict[int, Tuple[float, List[Tuple[float, List[Tuple[float, List[Tuple[float, int]]]]]]]] = {}

        max_chance1 = (
            4 if self.max_empty_samples is None else min(4, self.max_empty_samples)
        )
        max_chance2 = 4

        for a1 in legal_actions:
            nb1, _, lr1, ch1 = Bitboard2048.move(bb, a1)
            if not ch1:
                continue

            empty1 = Bitboard2048.get_empty_cells(nb1)
            num_empty1 = len(empty1)
            if num_empty1 == 0:
                tree[a1] = (lr1, [(1.0, [(0.0, [(1.0, nb1)])])])
                all_leaves.append(nb1)
                continue

            sampled1 = (
                np.random.choice(empty1, max_chance1, replace=False)
                if num_empty1 > max_chance1
                else empty1
            )
            w_base1 = 1.0 / len(sampled1)
            chance1_branches = []

            for c1 in sampled1:
                for val1, p1 in [(2, 0.90), (4, 0.10)]:
                    s1 = Bitboard2048.spawn_tile_at(nb1, c1, val1)
                    w1 = p1 * w_base1

                    mask2 = Bitboard2048.get_action_mask(s1)
                    legal2 = np.flatnonzero(mask2)
                    if len(legal2) == 0:
                        chance1_branches.append((w1, []))
                        continue

                    act2_branches = []
                    for a2 in legal2:
                        nb2, _, lr2, ch2 = Bitboard2048.move(s1, a2)
                        if not ch2:
                            continue

                        empty2 = Bitboard2048.get_empty_cells(nb2)
                        num_empty2 = len(empty2)
                        if num_empty2 == 0:
                            act2_branches.append((lr2, [(1.0, nb2)]))
                            all_leaves.append(nb2)
                            continue

                        sampled2 = (
                            np.random.choice(empty2, max_chance2, replace=False)
                            if num_empty2 > max_chance2
                            else empty2
                        )
                        w_base2 = 1.0 / len(sampled2)
                        chance2_branches = []

                        for c2 in sampled2:
                            for val2, p2 in [(2, 0.90), (4, 0.10)]:
                                p_branch = w1 * (p2 * w_base2)
                                # Probability pruning: skip branches with negligible cumulative probability
                                if p_branch < 0.001 and val2 == 4:
                                    continue

                                s2 = Bitboard2048.spawn_tile_at(nb2, c2, val2)
                                w2 = p2 * w_base2
                                chance2_branches.append((w2, s2))
                                all_leaves.append(s2)

                        if chance2_branches:
                            act2_branches.append((lr2, chance2_branches))

                    chance1_branches.append((w1, act2_branches))

            tree[a1] = (lr1, chance1_branches)

        if not tree:
            return self._search_depth_3(bb, legal_actions, use_d4=use_d4)

        self._ensure_evaluated(all_leaves, use_d4=use_d4)

        q_dict: Dict[int, float] = {a: -1e9 for a in range(4)}
        for a1, (lr1, chance1_branches) in tree.items():
            exp_v1 = 0.0
            for w1, act2_branches in chance1_branches:
                if not act2_branches:
                    best_q2 = 0.0
                else:
                    best_q2 = -1e9
                    for lr2, chance2_branches in act2_branches:
                        if not chance2_branches:
                            q2 = lr2
                        else:
                            exp_v2 = sum(
                                w2 * self.tt.get(Bitboard2048.get_canonical(s2), 0.0)
                                for w2, s2 in chance2_branches
                            )
                            q2 = lr2 + self.gamma * exp_v2
                        if q2 > best_q2:
                            best_q2 = q2
                    if best_q2 == -1e9:
                        best_q2 = 0.0
                exp_v1 += w1 * best_q2
            q_dict[a1] = lr1 + self.gamma * exp_v1

        best_action = max(tree.keys(), key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _batch_evaluate_boards(
        self,
        boards: Sequence[Union[np.ndarray, int]],
        use_d4: Optional[bool] = None,
    ) -> np.ndarray:
        """Evaluate a collection of boards in a single GPU forward pass with optional D4 symmetrization.

        Args:
            boards: Sequence of 4x4 integer boards or 64-bit bitboard integers.
            use_d4: Optional toggle overriding instance setting.

        Returns:
            1D numpy array of values V(s) = max_{a in legal} Q(s, a), or 0.0 if terminal.
        """
        n = len(boards)
        if n == 0:
            return np.array([], dtype=np.float32)

        d4_active = self.use_d4_symmetry if use_d4 is None else use_d4

        first = boards[0]
        if isinstance(first, (int, np.integer)):
            # Fast vectorized bitboard to uint8 exponent conversion
            bbs = np.fromiter(boards, dtype=np.uint64, count=n)
            exps = ((bbs[:, None] >> _SHIFTS_64) & np.uint64(0xF)).astype(np.uint8).reshape(n, 4, 4)

            # Compute legal action masks for all child bitboards
            masks_np = np.zeros((n, 4), dtype=bool)
            for i, b in enumerate(boards):
                bb_int = int(b)
                masks_np[i, 0] = Bitboard2048.is_move_legal(bb_int, Action.LEFT)
                masks_np[i, 1] = Bitboard2048.is_move_legal(bb_int, Action.RIGHT)
                masks_np[i, 2] = Bitboard2048.is_move_legal(bb_int, Action.UP)
                masks_np[i, 3] = Bitboard2048.is_move_legal(bb_int, Action.DOWN)
        else:
            boards_arr = np.stack(boards)
            clipped = np.clip(boards_arr, 0, MAX_TILE_VAL)
            exps = _POWER_LUT[clipped].astype(np.uint8)
            masks_np = np.stack([self.core.get_action_mask(b) for b in boards])

        # Non-blocking transfer of compact uint8 representation to GPU
        exps_t = torch.from_numpy(exps).to(self.device, non_blocking=True)
        obs_t = F.one_hot(exps_t.long(), 16).permute(0, 3, 1, 2).float()

        masks_t = torch.from_numpy(masks_np).bool().to(self.device, non_blocking=True)
        has_legal = masks_t.any(dim=-1)

        self.model.eval()
        with torch.inference_mode():
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=(self.device.type == "cuda"),
            ):
                if d4_active:
                    batch_d4 = generate_d4_tensors(obs_t)
                    batch_masks = generate_d4_masks(masks_t)

                    q_vals = self.model(batch_d4, action_mask=batch_masks)
                    q_d4 = q_vals.view(8, n, 4)
                    # Symmetrized value: mean of maximum legal Q-values across 8 orientations
                    v_k = q_d4.max(dim=-1).values  # (8, n)
                    v_sym = v_k.mean(dim=0)  # (n,)
                    v_vals = torch.where(has_legal, v_sym, torch.zeros_like(v_sym))
                else:
                    q_vals = self.model(obs_t, action_mask=masks_t)
                    v_vals = torch.where(
                        has_legal,
                        q_vals.max(dim=-1).values,
                        torch.zeros_like(q_vals[:, 0]),
                    )

            return v_vals.float().cpu().numpy()


def expectimax_search(
    board: Union[np.ndarray, int],
    model: nn.Module,
    device: Optional[torch.device] = None,
    depth: int = 2,
    gamma: float = 0.99,
    max_empty_samples: Optional[int] = None,
    use_d4_symmetry: bool = False,
) -> Tuple[int, Dict[int, float]]:
    """Functional convenience wrapper for ExpectimaxSearcher.search."""
    searcher = ExpectimaxSearcher(
        model=model,
        device=device,
        gamma=gamma,
        depth=depth,
        max_empty_samples=max_empty_samples,
        use_d4_symmetry=use_d4_symmetry,
    )
    return searcher.search(board, depth=depth, use_d4_symmetry=use_d4_symmetry)
