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
from src.env.bitboard import (
    Bitboard2048,
    SNAKE_WEIGHTS_8x16,
    compute_d4_structural_scores,
)
from src.env.wrapper import _POWER_LUT, MAX_TILE_VAL, one_hot_encode_board
from src.models.ntuple import NTupleNetwork
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

# Cells ordered by constraint/perimeter priority: corners first, outer edges, then interior
_CELL_PRIORITY = (0, 3, 12, 15, 1, 2, 4, 7, 8, 11, 13, 14, 5, 6, 9, 10)
_PRIORITY_ORDER = {cell: idx for idx, cell in enumerate(_CELL_PRIORITY)}


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
        - Adaptive Dynamic Depth: Automatically selects 2-ply, 3-ply, or 4-ply based on board complexity.
        - Topological Snake Monotonicity & Corner-Lock Evaluation Shield.
        - Transposition Table (TT) with 64-bit canonical Dihedral D4 hashing.
        - Dihedral D4 Symmetry: 8-fold isometric ensembling for rotation/reflection invariance.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        gamma: float = 0.99,
        depth: int = 2,
        max_empty_samples: Optional[int] = None,
        use_d4_symmetry: bool = False,
        tt_max_size: int = 500_000,
        adaptive_depth: bool = False,
        apex_mode: bool = False,
        use_snake_heuristic: bool = True,
        alpha_snake: float = 0.5,
        beta_corner: float = 1.0,
        use_afterstate_net: bool = False,
        use_ntuple: bool = False,
        ntuple_net: Optional[NTupleNetwork] = None,
        use_smoothness: bool = False,
        w_smooth: float = 0.1,
        w_trap: float = 5.0,
        use_policy_pruning: bool = False,
        policy_prune_tau: float = 0.03,
    ) -> None:
        """Initialize searcher with model and search configuration.

        Args:
            model: PyTorch policy or Q-network (e.g. Dueling2048Net). Optional if NTupleNetwork used alone.
            device: Compute device (defaults to GPU if available).
            gamma: Discount factor for Bellman expectation calculation.
            depth: Default search depth (0 for greedy, 1 for 1-ply, 2 for 2-ply, 3 for 3-ply, 4 for 4-ply).
            max_empty_samples: Optional maximum number of empty cells to sample per chance node.
                None evaluates all empty cells deterministically.
            use_d4_symmetry: Whether to enable 8-fold Dihedral D4 symmetry ensembling.
            tt_max_size: Maximum entries in the Transposition Table before pruning.
            adaptive_depth: Whether to dynamically select lookahead depth based on board state.
            apex_mode: Whether to enable 5-ply / 6-ply search in extreme endgames.
            use_snake_heuristic: Whether to blend topological snake monotonicity and corner lock shield.
            alpha_snake: Weight for snake monotonicity score (default 0.5).
            beta_corner: Weight for corner dislodgement penalty (default 1.0).
            use_afterstate_net: Whether to query scalar V(s_after) head directly instead of max_a Q(s, a).
            use_ntuple: Whether to enable microsecond SOTA N-Tuple leaf evaluation.
            ntuple_net: Optional NTupleNetwork instance (instantiated with defaults if None).
            use_smoothness: Whether to apply Manhattan smoothness and trapped low-tier wedge shield.
            w_smooth: Weight multiplier for Manhattan smoothness S_smooth(B).
            w_trap: Weight multiplier for trapped tile penalty P_trapped(B).
            use_policy_pruning: Whether to prune low-probability actions (pi(a|s) < tau) at shallow plies.
            policy_prune_tau: Cutoff threshold for policy-guided action pruning.
        """
        self.model = model
        self.device = device or get_device(prefer_gpu=True)
        self.gamma = gamma
        self.depth = depth
        self.max_empty_samples = max_empty_samples
        self.use_d4_symmetry = use_d4_symmetry
        self.tt_max_size = tt_max_size
        self.adaptive_depth = adaptive_depth
        self.apex_mode = apex_mode
        self.use_snake_heuristic = use_snake_heuristic
        self.alpha_snake = alpha_snake
        self.beta_corner = beta_corner
        # Auto-detect afterstate mode if configured on model
        self.use_afterstate_net = use_afterstate_net or (
            getattr(model, "is_afterstate", False) if model is not None else False
        )
        self.use_ntuple = use_ntuple
        self.ntuple_net = ntuple_net
        if self.use_ntuple and self.ntuple_net is None:
            self.ntuple_net = NTupleNetwork()
        self.use_smoothness = use_smoothness
        self.w_smooth = float(w_smooth)
        self.w_trap = float(w_trap)
        self.use_policy_pruning = use_policy_pruning
        self.policy_prune_tau = float(policy_prune_tau)

        self.tt: Dict[int, float] = {}
        self.core = Game2048Core()

    def clear_tt(self) -> None:
        """Clear all cached entries in the Transposition Table."""
        self.tt.clear()

    def get_action_policy(
        self,
        bb: int,
        legal_actions: Sequence[int],
    ) -> np.ndarray:
        """Compute action policy prior distribution pi(a | s) over legal actions.

        Supports dual-head policy networks, Q-networks, afterstate networks, and N-Tuple priors.

        Args:
            bb: 64-bit bitboard integer.
            legal_actions: Sequence of legal action indices in {0, 1, 2, 3}.

        Returns:
            Normalized 1D float32 array of shape (4,) representing pi(a | s).
        """
        if len(legal_actions) == 0:
            return np.zeros(4, dtype=np.float32)

        # 1. Dual-Head Network: if model exposes forward_policy
        if self.model is not None and hasattr(self.model, "forward_policy"):
            np_board = Bitboard2048.to_array(bb)
            obs = one_hot_encode_board(np_board)
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            with torch.inference_mode():
                probs = self.model.forward_policy(obs_t).squeeze(0).cpu().numpy()
            mask_arr = np.zeros(4, dtype=bool)
            mask_arr[list(legal_actions)] = True
            masked_probs = np.where(mask_arr, probs, 0.0)
            sum_p = masked_probs.sum()
            if sum_p > 1e-6:
                return (masked_probs / sum_p).astype(np.float32)

        # 2. Derive policy from Q-values or afterstate values
        q_vals = np.full(4, -1e9, dtype=np.float32)
        if self.model is not None:
            if self.use_afterstate_net:
                for a in legal_actions:
                    nb, _, lr, ch = Bitboard2048.move(bb, a)
                    if ch:
                        canon = Bitboard2048.get_canonical(nb)
                        if canon in self.tt:
                            v = self.tt[canon]
                        else:
                            v = float(self._batch_evaluate_boards([nb])[0])
                        q_vals[a] = lr + self.gamma * v
            else:
                np_board = Bitboard2048.to_array(bb)
                obs = one_hot_encode_board(np_board)
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    raw_q = self.model(obs_t).squeeze(0).cpu().numpy()
                for a in legal_actions:
                    q_vals[a] = raw_q[a]
        elif self.use_ntuple and self.ntuple_net is not None:
            for a in legal_actions:
                nb, _, lr, ch = Bitboard2048.move(bb, a)
                if ch:
                    v = self.ntuple_net.evaluate_bitboard(nb)
                    q_vals[a] = lr + self.gamma * v
        else:
            p = np.zeros(4, dtype=np.float32)
            p[list(legal_actions)] = 1.0 / len(legal_actions)
            return p

        # Softmax over legal actions
        legal_list = list(legal_actions)
        legal_q = q_vals[legal_list]
        max_q = float(np.max(legal_q))
        exp_q = np.exp(np.clip(legal_q - max_q, -50.0, 50.0))
        sum_exp = float(np.sum(exp_q))
        pi = np.zeros(4, dtype=np.float32)
        if sum_exp > 0:
            pi[legal_list] = (exp_q / sum_exp).astype(np.float32)
        else:
            pi[legal_list] = 1.0 / len(legal_list)
        return pi

    def _prune_actions_with_policy(
        self,
        bb: int,
        legal_actions: Sequence[int],
        tau: Optional[float] = None,
    ) -> List[int]:
        """Prune action branches where pi(a | s) < tau, unless all other actions are suicidal.

        Args:
            bb: 64-bit bitboard integer.
            legal_actions: Sequence of legal action indices.
            tau: Pruning threshold (defaults to self.policy_prune_tau).

        Returns:
            List of survived action indices.
        """
        if len(legal_actions) <= 1:
            return list(legal_actions)

        cutoff = self.policy_prune_tau if tau is None else tau
        pi = self.get_action_policy(bb, legal_actions)
        survivors = [a for a in legal_actions if pi[a] >= cutoff]

        def is_suicidal(a: int) -> bool:
            nb, _, _, ch = Bitboard2048.move(bb, a)
            if not ch:
                return True
            return Bitboard2048.is_terminal(nb)

        non_suicidal_legal = [a for a in legal_actions if not is_suicidal(a)]
        non_suicidal_survivors = [a for a in survivors if not is_suicidal(a)]

        if non_suicidal_survivors:
            return non_suicidal_survivors
        if non_suicidal_legal:
            return non_suicidal_legal
        if survivors:
            return survivors
        return [int(np.argmax(pi))]

    def select_action(
        self,
        board: Union[np.ndarray, int],
        action_mask: Optional[np.ndarray] = None,
        depth: Optional[int] = None,
        use_d4_symmetry: Optional[bool] = None,
        adaptive_depth: Optional[bool] = None,
        apex_mode: Optional[bool] = None,
        use_snake_heuristic: Optional[bool] = None,
        use_afterstate_net: Optional[bool] = None,
        use_ntuple: Optional[bool] = None,
        use_smoothness: Optional[bool] = None,
        use_policy_pruning: Optional[bool] = None,
    ) -> int:
        """Select best action using lookahead search.

        Args:
            board: Current 4x4 integer game board or 64-bit bitboard int.
            action_mask: Optional boolean array of legal actions.
            depth: Optional search depth overriding instance default.
            use_d4_symmetry: Optional D4 symmetry toggle overriding instance default.
            adaptive_depth: Optional adaptive depth toggle overriding instance default.
            apex_mode: Optional apex mode toggle overriding instance default.
            use_snake_heuristic: Optional snake heuristic toggle overriding instance default.
            use_afterstate_net: Optional afterstate value network toggle overriding instance default.
            use_ntuple: Optional N-Tuple leaf evaluation toggle overriding instance default.
            use_smoothness: Optional Manhattan smoothness toggle overriding instance default.
            use_policy_pruning: Optional policy pruning toggle overriding instance default.

        Returns:
            Selected integer action in {0, 1, 2, 3}.
        """
        best_action, _ = self.search(
            board,
            action_mask=action_mask,
            depth=depth,
            use_d4_symmetry=use_d4_symmetry,
            adaptive_depth=adaptive_depth,
            apex_mode=apex_mode,
            use_snake_heuristic=use_snake_heuristic,
            use_afterstate_net=use_afterstate_net,
            use_ntuple=use_ntuple,
            use_smoothness=use_smoothness,
            use_policy_pruning=use_policy_pruning,
        )
        return best_action

    def search(
        self,
        board: Union[np.ndarray, int],
        action_mask: Optional[np.ndarray] = None,
        depth: Optional[int] = None,
        use_d4_symmetry: Optional[bool] = None,
        adaptive_depth: Optional[bool] = None,
        apex_mode: Optional[bool] = None,
        use_snake_heuristic: Optional[bool] = None,
        use_afterstate_net: Optional[bool] = None,
        use_ntuple: Optional[bool] = None,
        use_smoothness: Optional[bool] = None,
        use_policy_pruning: Optional[bool] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """Perform Expectimax search from current board state.

        Args:
            board: 4x4 integer numpy array or 64-bit bitboard integer.
            action_mask: Optional boolean array of shape (4,). If None, calculated via bitboard/core.
            depth: Search depth (0, 1, 2, 3, 4, 5, or 6). Defaults to self.depth if adaptive_depth is False.
            use_d4_symmetry: Optional D4 toggle overriding instance setting.
            adaptive_depth: Optional adaptive depth toggle overriding instance setting.
            apex_mode: Optional apex mode toggle overriding instance setting.
            use_snake_heuristic: Optional snake heuristic toggle overriding instance setting.
            use_afterstate_net: Optional afterstate value net toggle overriding instance setting.
            use_ntuple: Optional N-Tuple evaluation toggle overriding instance setting.
            use_smoothness: Optional Manhattan smoothness toggle overriding instance setting.
            use_policy_pruning: Optional policy pruning toggle overriding instance setting.

        Returns:
            Tuple of (best_action, q_values_dict).
        """
        # Convert to bitboard if numpy array
        if isinstance(board, np.ndarray):
            bb = Bitboard2048.from_array(board)
            np_board = board
        else:
            bb = int(board)
            np_board = None

        is_adaptive = self.adaptive_depth if adaptive_depth is None else adaptive_depth
        is_apex = self.apex_mode if apex_mode is None else apex_mode
        if is_adaptive:
            d = Bitboard2048.get_adaptive_depth(bb, apex_mode=is_apex)
        else:
            d = self.depth if depth is None else depth

        d4_active = self.use_d4_symmetry if use_d4_symmetry is None else use_d4_symmetry
        snake_active = self.use_snake_heuristic if use_snake_heuristic is None else use_snake_heuristic
        afterstate_active = self.use_afterstate_net if use_afterstate_net is None else use_afterstate_net

        # Temporary overrides if passed
        if use_ntuple is not None:
            self.use_ntuple = use_ntuple
            if self.use_ntuple and self.ntuple_net is None:
                self.ntuple_net = NTupleNetwork()
        if use_smoothness is not None:
            self.use_smoothness = use_smoothness
        if use_policy_pruning is not None:
            self.use_policy_pruning = use_policy_pruning

        # Prune cache between moves if maximum capacity is exceeded
        if len(self.tt) > self.tt_max_size:
            self.tt.clear()

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
            if afterstate_active:
                # With afterstate value network V(s_after), greedy action selection is 1-ply search: argmax_a [r + gamma * V(s_after)]
                return self._search_depth_1(bb, legal_actions, use_d4=d4_active, use_snake=snake_active)
            if np_board is None:
                np_board = Bitboard2048.to_array(bb)
            return self._search_greedy(np_board, mask, use_d4=d4_active)
        elif d == 1:
            return self._search_depth_1(bb, legal_actions, use_d4=d4_active, use_snake=snake_active)
        elif d == 2:
            return self._search_depth_2(bb, legal_actions, use_d4=d4_active, use_snake=snake_active)
        elif d == 3:
            return self._search_depth_3(bb, legal_actions, use_d4=d4_active, use_snake=snake_active)
        elif d == 4:
            return self._search_depth_4(bb, legal_actions, use_d4=d4_active, use_snake=snake_active)
        elif d == 5:
            return self._search_depth_5(bb, legal_actions, use_d4=d4_active, use_snake=snake_active)
        elif d >= 6:
            return self._search_depth_6(bb, legal_actions, use_d4=d4_active, use_snake=snake_active)
        else:
            return self._search_depth_2(bb, legal_actions, use_d4=d4_active, use_snake=snake_active)

    def _ensure_evaluated(
        self,
        bitboards: Iterable[int],
        use_d4: bool = False,
        use_snake: bool = True,
    ) -> None:
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
        v_vals = self._batch_evaluate_boards(rep_boards, use_d4=use_d4, use_snake=use_snake)

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
        use_snake: bool = True,
    ) -> Tuple[int, Dict[int, float]]:
        """1-ply lookahead: evaluate afterstate for each legal action."""
        act_candidates = (
            self._prune_actions_with_policy(bb, legal_actions)
            if self.use_policy_pruning
            else legal_actions
        )
        after_boards: List[int] = []
        imm_rewards: List[float] = []
        valid_actions: List[int] = []

        for a in act_candidates:
            nb, _, lr, changed = Bitboard2048.move(bb, a)
            if changed:
                after_boards.append(nb)
                imm_rewards.append(lr)
                valid_actions.append(a)

        if len(after_boards) == 0:
            return int(legal_actions[0]), {a: -1e9 for a in range(4)}

        self._ensure_evaluated(after_boards, use_d4=use_d4, use_snake=use_snake)

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
        use_snake: bool = True,
    ) -> Tuple[int, Dict[int, float]]:
        """2-ply Expectimax: action slide -> chance spawns 2/4 -> batched leaf evaluation.

        For each legal action:
            1. Deterministic slide-and-merge yields s_after and immediate reward r.
            2. Branches over all empty cells with spawn probabilities (90% for 2, 10% for 4).
            3. Evaluates all unique un-cached leaf states via Transposition Table and batched GPU pass.
            4. Returns action maximizing Q(s, a) = r + gamma * sum(P(spawn) * V(s')).
        """
        act_candidates = (
            self._prune_actions_with_policy(bb, legal_actions)
            if self.use_policy_pruning
            else legal_actions
        )
        all_leaves: List[int] = []
        tree: Dict[int, Tuple[float, List[Tuple[float, int]]]] = {}

        for a in act_candidates:
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

        self._ensure_evaluated(all_leaves, use_d4=use_d4, use_snake=use_snake)

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
        use_snake: bool = True,
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

        act_candidates = (
            self._prune_actions_with_policy(bb, legal_actions)
            if self.use_policy_pruning
            else legal_actions
        )

        for a1 in act_candidates:
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
                        legal2_candidates = (
                            self._prune_actions_with_policy(s_chance, legal2)
                            if self.use_policy_pruning
                            else legal2
                        )
                        act2_moves: List[Tuple[float, int]] = []
                        for a2 in legal2_candidates:
                            nb2, _, lr2, ch2 = Bitboard2048.move(s_chance, a2)
                            if ch2:
                                act2_moves.append((lr2, nb2))
                                all_leaves.append(nb2)
                        chance_branches.append((w_chance, act2_moves))

            tree[a1] = (lr1, chance_branches)

        if not tree:
            return self._search_depth_2(bb, legal_actions, use_d4=use_d4, use_snake=use_snake)

        self._ensure_evaluated(all_leaves, use_d4=use_d4, use_snake=use_snake)

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
        use_snake: bool = True,
    ) -> Tuple[int, Dict[int, float]]:
        """4-ply Expectimax: Action 1 -> Chance 1 -> Action 2 -> Chance 2 -> Leaf Evaluation.

        Applies deep chance node pruning:
            - Chance 1 (ply 2): if empty cells > 4, prioritizes top-4 perimeter cells.
            - Chance 2 (ply 4): if empty cells > 4, prunes to top-3 most constrained/edge-adjacent empty cells.
            - Chance 2 (ply 4): only expands spawn tile 2 (weight 1.0) and prunes tile 4 unless empty cells <= 2.
            - Probability pruning (pruning branches with p < 0.001).
        """
        all_leaves: List[int] = []
        # tree: a1 -> (lr1, list of (w_chance1, list of (lr2, list of (w_chance2, s_leaf))))
        tree: Dict[int, Tuple[float, List[Tuple[float, List[Tuple[float, List[Tuple[float, int]]]]]]]] = {}

        max_chance1 = (
            4 if self.max_empty_samples is None else min(4, self.max_empty_samples)
        )

        act_candidates = (
            self._prune_actions_with_policy(bb, legal_actions)
            if self.use_policy_pruning
            else legal_actions
        )

        for a1 in act_candidates:
            nb1, _, lr1, ch1 = Bitboard2048.move(bb, a1)
            if not ch1:
                continue

            empty1 = Bitboard2048.get_empty_cells(nb1)
            num_empty1 = len(empty1)
            if num_empty1 == 0:
                tree[a1] = (lr1, [(1.0, [(0.0, [(1.0, nb1)])])])
                all_leaves.append(nb1)
                continue

            if num_empty1 > max_chance1:
                sampled1 = sorted(empty1, key=_PRIORITY_ORDER.get)[:max_chance1]
            else:
                sampled1 = empty1

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

                    legal2_candidates = (
                        self._prune_actions_with_policy(s1, legal2)
                        if self.use_policy_pruning
                        else legal2
                    )
                    act2_branches = []
                    for a2 in legal2_candidates:
                        nb2, _, lr2, ch2 = Bitboard2048.move(s1, a2)
                        if not ch2:
                            continue

                        empty2 = Bitboard2048.get_empty_cells(nb2)
                        num_empty2 = len(empty2)
                        if num_empty2 == 0:
                            act2_branches.append((lr2, [(1.0, nb2)]))
                            all_leaves.append(nb2)
                            continue

                        # Deep chance pruning:
                        # 1. If empty cells > 4, expand only top-3 most constrained cells
                        if num_empty2 > 4:
                            sampled2 = sorted(empty2, key=_PRIORITY_ORDER.get)[:3]
                        elif self.max_empty_samples is not None and num_empty2 > self.max_empty_samples:
                            sampled2 = sorted(empty2, key=_PRIORITY_ORDER.get)[:self.max_empty_samples]
                        else:
                            sampled2 = empty2

                        w_base2 = 1.0 / len(sampled2)

                        # 2. At deep ply >= 3, only expand spawn tile 2 (weight 1.0); prune tile 4 unless empty cells <= 2
                        if num_empty2 > 2:
                            spawn_candidates = [(2, 1.0)]
                        else:
                            spawn_candidates = [(2, 0.90), (4, 0.10)]

                        chance2_branches = []
                        for c2 in sampled2:
                            for val2, p2 in spawn_candidates:
                                p_branch = w1 * (p2 * w_base2)
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
            return self._search_depth_3(bb, legal_actions, use_d4=use_d4, use_snake=use_snake)

        self._ensure_evaluated(all_leaves, use_d4=use_d4, use_snake=use_snake)

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

    def _filter_corner_preserving_actions(
        self,
        bb: int,
        legal_actions: Sequence[int],
    ) -> List[int]:
        """Prune actions that dislodge a locked maximum tile from outer corners."""
        return Bitboard2048.filter_corner_preserving_actions(bb, legal_actions)

    def _search_depth_5(
        self,
        bb: int,
        legal_actions: Sequence[int],
        use_d4: bool = False,
        use_snake: bool = True,
    ) -> Tuple[int, Dict[int, float]]:
        """5-ply Expectimax: Action 1 -> Chance 1 -> Action 2 -> Chance 2 -> Action 3 -> Leaf Evaluation.

        Applies strict apex endgame pruning:
            - Prunes actions that dislodge a locked corner tile unless hazardous.
            - When empty_cells <= 2, performs exact stochastic evaluation branching both tile 2 (90%)
              and tile 4 (10%) across the empty cells.
            - Otherwise expands the primary tile spawn (tile 2) at top-1 priority cell.
        """
        all_leaves: List[int] = []
        tree: Dict[int, Tuple[float, List[Tuple[float, List[Tuple[float, List[Tuple[float, List[Tuple[float, int]]]]]]]]]] = {}

        filt_actions1 = self._filter_corner_preserving_actions(bb, legal_actions)
        if self.use_policy_pruning:
            filt_actions1 = self._prune_actions_with_policy(bb, filt_actions1)

        for a1 in filt_actions1:
            nb1, _, lr1, ch1 = Bitboard2048.move(bb, a1)
            if not ch1:
                continue

            empty1 = Bitboard2048.get_empty_cells(nb1)
            num_empty1 = len(empty1)
            if num_empty1 == 0:
                tree[a1] = (lr1, [(1.0, [(0.0, [(1.0, [(0.0, [(1.0, nb1)])])])])])
                all_leaves.append(nb1)
                continue

            if num_empty1 <= 2:
                sampled1 = empty1
                spawn_candidates1 = [(2, 0.90), (4, 0.10)]
            else:
                sampled1 = sorted(empty1, key=_PRIORITY_ORDER.get)[:1]
                spawn_candidates1 = [(2, 1.0)]

            w_base1 = 1.0 / len(sampled1)
            chance1_branches = []

            for c1 in sampled1:
                for val1, p1 in spawn_candidates1:
                    w1 = p1 * w_base1
                    s1 = Bitboard2048.spawn_tile_at(nb1, c1, val1)
                    mask2 = Bitboard2048.get_action_mask(s1)
                    legal2 = np.flatnonzero(mask2)

                    if len(legal2) == 0:
                        chance1_branches.append((w1, []))
                        continue

                    filt_actions2 = self._filter_corner_preserving_actions(s1, legal2)
                    if self.use_policy_pruning:
                        filt_actions2 = self._prune_actions_with_policy(s1, filt_actions2)
                    act2_branches = []

                    for a2 in filt_actions2:
                        nb2, _, lr2, ch2 = Bitboard2048.move(s1, a2)
                        if not ch2:
                            continue

                        empty2 = Bitboard2048.get_empty_cells(nb2)
                        num_empty2 = len(empty2)
                        if num_empty2 == 0:
                            act2_branches.append((lr2, [(1.0, [(0.0, nb2)])]))
                            all_leaves.append(nb2)
                            continue

                        if num_empty2 <= 2:
                            sampled2 = empty2
                            spawn_candidates2 = [(2, 0.90), (4, 0.10)]
                        else:
                            sampled2 = sorted(empty2, key=_PRIORITY_ORDER.get)[:1]
                            spawn_candidates2 = [(2, 1.0)]

                        w_base2 = 1.0 / len(sampled2)
                        chance2_branches = []

                        for c2 in sampled2:
                            for val2, p2 in spawn_candidates2:
                                w2 = p2 * w_base2
                                s2 = Bitboard2048.spawn_tile_at(nb2, c2, val2)
                                mask3 = Bitboard2048.get_action_mask(s2)
                                legal3 = np.flatnonzero(mask3)

                                if len(legal3) == 0:
                                    chance2_branches.append((w2, []))
                                    continue

                                filt_actions3 = self._filter_corner_preserving_actions(s2, legal3)
                                act3_branches = []

                                for a3 in filt_actions3:
                                    nb3, _, lr3, ch3 = Bitboard2048.move(s2, a3)
                                    if ch3:
                                        act3_branches.append((lr3, nb3))
                                        all_leaves.append(nb3)

                                if act3_branches:
                                    chance2_branches.append((w2, act3_branches))

                        if chance2_branches:
                            act2_branches.append((lr2, chance2_branches))

                    if act2_branches:
                        chance1_branches.append((w1, act2_branches))

            if chance1_branches:
                tree[a1] = (lr1, chance1_branches)

        if not tree:
            return self._search_depth_4(bb, legal_actions, use_d4=use_d4, use_snake=use_snake)

        self._ensure_evaluated(all_leaves, use_d4=use_d4, use_snake=use_snake)

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
                            exp_v2 = 0.0
                            for w2, act3_branches in chance2_branches:
                                if not act3_branches:
                                    best_q3 = 0.0
                                else:
                                    best_q3 = max(
                                        lr3 + self.gamma * self.tt.get(Bitboard2048.get_canonical(nb3), 0.0)
                                        for lr3, nb3 in act3_branches
                                    )
                                exp_v2 += w2 * best_q3
                            q2 = lr2 + self.gamma * exp_v2
                        if q2 > best_q2:
                            best_q2 = q2
                    if best_q2 == -1e9:
                        best_q2 = 0.0
                exp_v1 += w1 * best_q2
            q_dict[a1] = lr1 + self.gamma * exp_v1

        best_action = max(tree.keys(), key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _search_depth_6(
        self,
        bb: int,
        legal_actions: Sequence[int],
        use_d4: bool = False,
        use_snake: bool = True,
    ) -> Tuple[int, Dict[int, float]]:
        """6-ply Expectimax: Action 1 -> Chance 1 -> Action 2 -> Chance 2 -> Action 3 -> Chance 3 -> Leaf Evaluation.

        Applies apex endgame pruning for extreme endgames:
            - Prunes actions that dislodge a locked corner tile unless hazardous.
            - When empty_cells <= 2, evaluates both tile 2 (90%) and tile 4 (10%) spawns.
        """
        all_leaves: List[int] = []
        tree: Dict[int, Tuple[float, List[Tuple[float, List[Tuple[float, List[Tuple[float, List[Tuple[float, List[Tuple[float, int]]]]]]]]]]]] = {}

        filt_actions1 = self._filter_corner_preserving_actions(bb, legal_actions)
        if self.use_policy_pruning:
            filt_actions1 = self._prune_actions_with_policy(bb, filt_actions1)

        for a1 in filt_actions1:
            nb1, _, lr1, ch1 = Bitboard2048.move(bb, a1)
            if not ch1:
                continue

            empty1 = Bitboard2048.get_empty_cells(nb1)
            num_empty1 = len(empty1)
            if num_empty1 == 0:
                tree[a1] = (lr1, [(1.0, [(0.0, [(1.0, [(0.0, [(1.0, [(0.0, nb1)])])])])])])
                all_leaves.append(nb1)
                continue

            if num_empty1 <= 2:
                sampled1 = empty1
                spawn_candidates1 = [(2, 0.90), (4, 0.10)]
            else:
                sampled1 = sorted(empty1, key=_PRIORITY_ORDER.get)[:1]
                spawn_candidates1 = [(2, 1.0)]

            w_base1 = 1.0 / len(sampled1)
            chance1_branches = []

            for c1 in sampled1:
                for val1, p1 in spawn_candidates1:
                    w1 = p1 * w_base1
                    s1 = Bitboard2048.spawn_tile_at(nb1, c1, val1)
                    mask2 = Bitboard2048.get_action_mask(s1)
                    legal2 = np.flatnonzero(mask2)
                    if len(legal2) == 0:
                        chance1_branches.append((w1, []))
                        continue

                    filt_actions2 = self._filter_corner_preserving_actions(s1, legal2)
                    if self.use_policy_pruning:
                        filt_actions2 = self._prune_actions_with_policy(s1, filt_actions2)
                    act2_branches = []

                    for a2 in filt_actions2:
                        nb2, _, lr2, ch2 = Bitboard2048.move(s1, a2)
                        if not ch2:
                            continue

                        empty2 = Bitboard2048.get_empty_cells(nb2)
                        num_empty2 = len(empty2)
                        if num_empty2 == 0:
                            act2_branches.append((lr2, [(1.0, [(0.0, [(1.0, [(0.0, nb2)])])])]))
                            all_leaves.append(nb2)
                            continue

                        if num_empty2 <= 2:
                            sampled2 = empty2
                            spawn_candidates2 = [(2, 0.90), (4, 0.10)]
                        else:
                            sampled2 = sorted(empty2, key=_PRIORITY_ORDER.get)[:1]
                            spawn_candidates2 = [(2, 1.0)]

                        w_base2 = 1.0 / len(sampled2)
                        chance2_branches = []

                        for c2 in sampled2:
                            for val2, p2 in spawn_candidates2:
                                w2 = p2 * w_base2
                                s2 = Bitboard2048.spawn_tile_at(nb2, c2, val2)
                                mask3 = Bitboard2048.get_action_mask(s2)
                                legal3 = np.flatnonzero(mask3)
                                if len(legal3) == 0:
                                    chance2_branches.append((w2, []))
                                    continue

                                filt_actions3 = self._filter_corner_preserving_actions(s2, legal3)
                                act3_branches = []

                                for a3 in filt_actions3:
                                    nb3, _, lr3, ch3 = Bitboard2048.move(s2, a3)
                                    if not ch3:
                                        continue

                                    empty3 = Bitboard2048.get_empty_cells(nb3)
                                    num_empty3 = len(empty3)
                                    if num_empty3 == 0:
                                        act3_branches.append((lr3, [(1.0, nb3)]))
                                        all_leaves.append(nb3)
                                        continue

                                    if num_empty3 <= 2:
                                        sampled3 = empty3
                                        spawn_candidates3 = [(2, 0.90), (4, 0.10)]
                                    else:
                                        sampled3 = sorted(empty3, key=_PRIORITY_ORDER.get)[:1]
                                        spawn_candidates3 = [(2, 1.0)]

                                    w_base3 = 1.0 / len(sampled3)
                                    chance3_branches = []

                                    for c3 in sampled3:
                                        for val3, p3 in spawn_candidates3:
                                            w3 = p3 * w_base3
                                            s3 = Bitboard2048.spawn_tile_at(nb3, c3, val3)
                                            chance3_branches.append((w3, s3))
                                            all_leaves.append(s3)

                                    if chance3_branches:
                                        act3_branches.append((lr3, chance3_branches))

                                if act3_branches:
                                    chance2_branches.append((w2, act3_branches))

                        if chance2_branches:
                            act2_branches.append((lr2, chance2_branches))

                    if act2_branches:
                        chance1_branches.append((w1, act2_branches))

            if chance1_branches:
                tree[a1] = (lr1, chance1_branches)

        if not tree:
            return self._search_depth_5(bb, legal_actions, use_d4=use_d4, use_snake=use_snake)

        self._ensure_evaluated(all_leaves, use_d4=use_d4, use_snake=use_snake)

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
                            exp_v2 = 0.0
                            for w2, act3_branches in chance2_branches:
                                if not act3_branches:
                                    best_q3 = 0.0
                                else:
                                    best_q3 = -1e9
                                    for lr3, chance3_branches in act3_branches:
                                        if not chance3_branches:
                                            q3 = lr3
                                        else:
                                            exp_v3 = sum(
                                                w3 * self.tt.get(Bitboard2048.get_canonical(s3), 0.0)
                                                for w3, s3 in chance3_branches
                                            )
                                            q3 = lr3 + self.gamma * exp_v3
                                        if q3 > best_q3:
                                            best_q3 = q3
                                    if best_q3 == -1e9:
                                        best_q3 = 0.0
                                exp_v2 += w2 * best_q3
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
        use_snake: Optional[bool] = None,
        use_ntuple: Optional[bool] = None,
        use_smoothness: Optional[bool] = None,
    ) -> np.ndarray:
        """Evaluate a collection of boards combining neural net, N-Tuple, and geometric shield.

        Unified Leaf Scoring Pipeline:
            V_leaf(B) = V_ntuple(B) + w_smooth * S_smooth(B) - w_trap * P_trapped(B) + V_net(B)

        Args:
            boards: Sequence of 4x4 integer boards or 64-bit bitboard integers.
            use_d4: Optional toggle overriding instance setting.
            use_snake: Optional toggle overriding instance setting.
            use_ntuple: Optional toggle overriding instance setting.
            use_smoothness: Optional toggle overriding instance setting.

        Returns:
            1D numpy array of values V_leaf(B), or 0.0 if terminal.
        """
        n = len(boards)
        if n == 0:
            return np.array([], dtype=np.float32)

        d4_active = self.use_d4_symmetry if use_d4 is None else use_d4
        snake_active = self.use_snake_heuristic if use_snake is None else use_snake

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

        # 1. Base Neural Network Evaluation V_net(B)
        if self.model is not None:
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
                    if self.use_afterstate_net:
                        if d4_active:
                            batch_d4 = generate_d4_tensors(obs_t)
                            v_d4 = self.model.forward_value(batch_d4)  # (8 * n,)
                            v_k = v_d4.view(8, n)
                            v_sym = v_k.mean(dim=0)  # (n,)
                            v_vals = torch.where(has_legal, v_sym, torch.zeros_like(v_sym))
                        else:
                            v_raw = self.model.forward_value(obs_t)  # (n,)
                            v_vals = torch.where(has_legal, v_raw, torch.zeros_like(v_raw))
                    else:
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

            v_np = v_vals.float().cpu().numpy()
        else:
            v_np = np.zeros(n, dtype=np.float32)

        # 2. SOTA N-Tuple Leaf Evaluation V_ntuple(B)
        ntuple_active = self.use_ntuple if use_ntuple is None else use_ntuple
        if ntuple_active and self.ntuple_net is not None:
            if isinstance(first, (int, np.integer)):
                v_ntuple = self.ntuple_net.evaluate_batch(bbs)
            else:
                bbs_arr = np.array([Bitboard2048.from_array(b) for b in boards], dtype=np.uint64)
                v_ntuple = self.ntuple_net.evaluate_batch(bbs_arr)
        else:
            v_ntuple = np.zeros(n, dtype=np.float32)

        # 3. Geometric Shield: Manhattan Smoothness S_smooth(B) and Trapped Wedge Penalty P_trapped(B)
        smoothness_active = self.use_smoothness if use_smoothness is None else use_smoothness
        if smoothness_active:
            if isinstance(first, (int, np.integer)):
                bbs_geom = bbs
            else:
                bbs_geom = np.array([Bitboard2048.from_array(b) for b in boards], dtype=np.uint64)
            s_smooth = Bitboard2048.compute_smoothness_batch(bbs_geom)
            p_trapped = Bitboard2048.compute_trapped_penalty_batch(bbs_geom)
            geom_val = self.w_smooth * s_smooth - self.w_trap * p_trapped
        else:
            geom_val = np.zeros(n, dtype=np.float32)

        # 4. Topological Snake Monotonicity & Corner-Lock Heuristic
        if snake_active:
            ranks_flat = exps.reshape(n, 16).astype(np.float32)
            # Dot product with 8x16 snake templates and take maximum
            snake_scores = np.max(np.dot(ranks_flat, SNAKE_WEIGHTS_8x16.T), axis=1)
            # Add refined structural scores (row 0/1 monotonicity, connectivity, inversion penalty)
            struct_scores = compute_d4_structural_scores(exps)
            total_snake_scores = snake_scores + struct_scores

            # Vectorized corner penalty
            k_max = np.max(ranks_flat, axis=1)
            is_corner = (
                (ranks_flat[:, 0] == k_max)
                | (ranks_flat[:, 3] == k_max)
                | (ranks_flat[:, 12] == k_max)
                | (ranks_flat[:, 15] == k_max)
            )
            is_interior = (
                (ranks_flat[:, 5] == k_max)
                | (ranks_flat[:, 6] == k_max)
                | (ranks_flat[:, 9] == k_max)
                | (ranks_flat[:, 10] == k_max)
            )
            penalties = np.where(
                k_max < 7,
                0.0,
                np.where(is_corner, 0.0, np.where(is_interior, k_max * 2.0, k_max * 1.0)),
            )
            snake_val = self.alpha_snake * total_snake_scores - self.beta_corner * penalties
        else:
            snake_val = np.zeros(n, dtype=np.float32)

        has_legal_np = masks_np.any(axis=-1)
        v_final = np.where(
            has_legal_np,
            v_np + v_ntuple + geom_val + snake_val,
            0.0,
        )
        return v_final.astype(np.float32)


def expectimax_search(
    board: Union[np.ndarray, int],
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    depth: int = 2,
    gamma: float = 0.99,
    max_empty_samples: Optional[int] = None,
    use_d4_symmetry: bool = False,
    adaptive_depth: bool = False,
    apex_mode: bool = False,
    use_snake_heuristic: bool = True,
    alpha_snake: float = 0.5,
    beta_corner: float = 1.0,
    use_afterstate_net: bool = False,
    use_ntuple: bool = False,
    ntuple_net: Optional[NTupleNetwork] = None,
    use_smoothness: bool = False,
    w_smooth: float = 0.1,
    w_trap: float = 5.0,
    use_policy_pruning: bool = False,
    policy_prune_tau: float = 0.03,
) -> Tuple[int, Dict[int, float]]:
    """Functional convenience wrapper for ExpectimaxSearcher.search."""
    searcher = ExpectimaxSearcher(
        model=model,
        device=device,
        gamma=gamma,
        depth=depth,
        max_empty_samples=max_empty_samples,
        use_d4_symmetry=use_d4_symmetry,
        adaptive_depth=adaptive_depth,
        apex_mode=apex_mode,
        use_snake_heuristic=use_snake_heuristic,
        alpha_snake=alpha_snake,
        beta_corner=beta_corner,
        use_afterstate_net=use_afterstate_net,
        use_ntuple=use_ntuple,
        ntuple_net=ntuple_net,
        use_smoothness=use_smoothness,
        w_smooth=w_smooth,
        w_trap=w_trap,
        use_policy_pruning=use_policy_pruning,
        policy_prune_tau=policy_prune_tau,
    )
    return searcher.search(
        board,
        depth=depth,
        use_d4_symmetry=use_d4_symmetry,
        adaptive_depth=adaptive_depth,
        apex_mode=apex_mode,
        use_snake_heuristic=use_snake_heuristic,
        use_afterstate_net=use_afterstate_net,
        use_ntuple=use_ntuple,
        use_smoothness=use_smoothness,
        use_policy_pruning=use_policy_pruning,
    )
