"""Lightweight Expectimax & Lookahead search for 2048 DRL agents.

Implements batched n-ply Expectimax search over deterministic slide-and-merge
actions and stochastic tile spawns (90% for 2, 10% for 4). Evaluates leaf states
using a trained Dueling Double DQN network in a single GPU forward pass with optional
Dihedral D4 (8-fold rotation/reflection) symmetry invariance.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.core import Action, Game2048Core
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
    """High-throughput Expectimax searcher leveraging batched neural network evaluation.

    Supports:
        - Depth 0: Pure greedy evaluation via direct Q(s, a) forward pass.
        - Depth 1: 1-ply lookahead (action slide -> evaluate afterstate).
        - Depth 2: 2-ply Expectimax (action slide -> chance spawns 2/4 -> evaluate leaf states).
        - Depth 3: 3-ply Expectimax (action -> chance -> next action -> evaluate leaf states).
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
    ) -> None:
        """Initialize searcher with model and search configuration.

        Args:
            model: PyTorch policy or Q-network (e.g. Dueling2048Net).
            device: Compute device (defaults to GPU if available).
            gamma: Discount factor for Bellman expectation calculation.
            depth: Default search depth (0 for greedy, 1 for 1-ply, 2 for 2-ply expectimax).
            max_empty_samples: Optional maximum number of empty cells to sample per chance node.
                None evaluates all empty cells deterministically.
            use_d4_symmetry: Whether to enable 8-fold Dihedral D4 symmetry ensembling.
        """
        self.model = model
        self.device = device or get_device(prefer_gpu=True)
        self.gamma = gamma
        self.depth = depth
        self.max_empty_samples = max_empty_samples
        self.use_d4_symmetry = use_d4_symmetry
        self.core = Game2048Core()

    def select_action(
        self,
        board: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        depth: Optional[int] = None,
        use_d4_symmetry: Optional[bool] = None,
    ) -> int:
        """Select best action using lookahead search.

        Args:
            board: Current 4x4 integer game board.
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
        board: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        depth: Optional[int] = None,
        use_d4_symmetry: Optional[bool] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """Perform Expectimax search from current board state.

        Args:
            board: 4x4 integer numpy array.
            action_mask: Optional boolean array of shape (4,). If None, calculated via core.
            depth: Search depth (0, 1, 2, or 3). Defaults to self.depth.
            use_d4_symmetry: Optional D4 toggle overriding instance setting.

        Returns:
            Tuple of (best_action, q_values_dict).
        """
        d = self.depth if depth is None else depth
        d4_active = self.use_d4_symmetry if use_d4_symmetry is None else use_d4_symmetry

        if action_mask is None:
            mask = self.core.get_action_mask(board)
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
            return self._search_greedy(board, mask, use_d4=d4_active)
        elif d == 1:
            return self._search_depth_1(board, legal_actions, use_d4=d4_active)
        elif d == 2:
            return self._search_depth_2(board, legal_actions, use_d4=d4_active)
        elif d == 3:
            return self._search_depth_3(board, legal_actions, use_d4=d4_active)
        else:
            return self._search_depth_2(board, legal_actions, use_d4=d4_active)

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

                    # Reverse map actions for each orientation k:
                    # transformed action j corresponds to original action perm[j]
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
        board: np.ndarray,
        legal_actions: np.ndarray,
        use_d4: bool = False,
    ) -> Tuple[int, Dict[int, float]]:
        """1-ply lookahead: evaluate afterstate for each legal action."""
        after_boards = []
        imm_rewards = []
        valid_actions = []

        for a in legal_actions:
            next_b, _, lr, changed = Game2048Core._execute_slide(board, a)
            if changed:
                after_boards.append(next_b)
                imm_rewards.append(lr)
                valid_actions.append(a)

        if len(after_boards) == 0:
            return int(legal_actions[0]), {a: -1e9 for a in range(4)}

        v_vals = self._batch_evaluate_boards(after_boards, use_d4=use_d4)

        q_dict: Dict[int, float] = {a: -1e9 for a in range(4)}
        for a, lr, v in zip(valid_actions, imm_rewards, v_vals):
            q_dict[a] = lr + self.gamma * v

        best_action = max(valid_actions, key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _search_depth_2(
        self,
        board: np.ndarray,
        legal_actions: np.ndarray,
        use_d4: bool = False,
    ) -> Tuple[int, Dict[int, float]]:
        """2-ply Expectimax: simulate move -> tile spawns 2/4 -> batched leaf evaluation.

        For each legal action:
            1. Simulates deterministic slide-and-merge yielding s_after and immediate reward r.
            2. Branches over all empty cells with nature spawn probabilities (90% for 2, 10% for 4).
            3. Evaluates all resulting leaf states in a single batched GPU forward pass (optionally D4 symmetrized).
            4. Calculates expected return: Q(s, a) = r + gamma * sum(P(spawn) * max_{a'} Q(s', a')).
        """
        all_children: List[np.ndarray] = []
        all_weights: List[float] = []
        all_action_owners: List[int] = []
        action_immediate_rewards: Dict[int, float] = {}

        for a in legal_actions:
            next_b, _, lr, changed = Game2048Core._execute_slide(board, a)
            if not changed:
                continue

            action_immediate_rewards[a] = lr
            empty_cells = np.flatnonzero(next_b == 0)
            num_empty = len(empty_cells)

            if num_empty == 0:
                all_children.append(next_b.copy())
                all_weights.append(1.0)
                all_action_owners.append(a)
            else:
                if (
                    self.max_empty_samples is not None
                    and num_empty > self.max_empty_samples
                ):
                    sampled_cells = np.random.choice(
                        empty_cells, self.max_empty_samples, replace=False
                    )
                else:
                    sampled_cells = empty_cells

                w_base = 1.0 / len(sampled_cells)
                for cell_idx in sampled_cells:
                    # Spawn 2 with 90% probability
                    b2 = next_b.copy()
                    b2.flat[cell_idx] = 2
                    all_children.append(b2)
                    all_weights.append(0.90 * w_base)
                    all_action_owners.append(a)

                    # Spawn 4 with 10% probability
                    b4 = next_b.copy()
                    b4.flat[cell_idx] = 4
                    all_children.append(b4)
                    all_weights.append(0.10 * w_base)
                    all_action_owners.append(a)

        if len(all_children) == 0:
            return int(legal_actions[0]), {a: -1e9 for a in range(4)}

        v_vals = self._batch_evaluate_boards(all_children, use_d4=use_d4)

        action_exp_v: Dict[int, float] = {a: 0.0 for a in action_immediate_rewards}
        for v, w, a in zip(v_vals, all_weights, all_action_owners):
            action_exp_v[a] += w * v

        q_dict: Dict[int, float] = {a: -1e9 for a in range(4)}
        for a, lr in action_immediate_rewards.items():
            q_dict[a] = lr + self.gamma * action_exp_v[a]

        best_action = max(action_immediate_rewards.keys(), key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _search_depth_3(
        self,
        board: np.ndarray,
        legal_actions: np.ndarray,
        use_d4: bool = False,
    ) -> Tuple[int, Dict[int, float]]:
        """3-ply Expectimax: Action -> Chance Spawn -> Next Action -> Leaf Evaluation."""
        all_leaves: List[np.ndarray] = []
        leaf_metadata: List[Tuple[int, float, float]] = []
        action_immediate_rewards: Dict[int, float] = {}

        max_chance = (
            4 if self.max_empty_samples is None else min(4, self.max_empty_samples)
        )

        for a1 in legal_actions:
            nb1, _, lr1, ch1 = Game2048Core._execute_slide(board, a1)
            if not ch1:
                continue

            action_immediate_rewards[a1] = lr1
            empty_cells = np.flatnonzero(nb1 == 0)
            num_empty = len(empty_cells)

            if num_empty == 0:
                all_leaves.append(nb1.copy())
                leaf_metadata.append((a1, 1.0, 0.0))
                continue

            if num_empty > max_chance:
                sampled_cells = np.random.choice(
                    empty_cells, max_chance, replace=False
                )
            else:
                sampled_cells = empty_cells

            w_base = 1.0 / len(sampled_cells)

            for cell_idx in sampled_cells:
                for val, p_val in [(2, 0.90), (4, 0.10)]:
                    s_chance = nb1.copy()
                    s_chance.flat[cell_idx] = val
                    chance_w = p_val * w_base

                    mask2 = self.core.get_action_mask(s_chance)
                    legal2 = np.flatnonzero(mask2)

                    if len(legal2) == 0:
                        all_leaves.append(s_chance)
                        leaf_metadata.append((a1, chance_w, 0.0))
                    else:
                        for a2 in legal2:
                            nb2, _, lr2, ch2 = Game2048Core._execute_slide(
                                s_chance, a2
                            )
                            if ch2:
                                all_leaves.append(nb2)
                                leaf_metadata.append((a1, chance_w, lr2))

        if len(all_leaves) == 0:
            return self._search_depth_2(board, legal_actions, use_d4=use_d4)

        leaf_v = self._batch_evaluate_boards(all_leaves, use_d4=use_d4)

        action_exp_v: Dict[int, float] = {a: 0.0 for a in action_immediate_rewards}
        for v, (a1, chance_w, lr2) in zip(leaf_v, leaf_metadata):
            q_ply2 = lr2 + self.gamma * v
            action_exp_v[a1] += chance_w * q_ply2

        q_dict: Dict[int, float] = {a: -1e9 for a in range(4)}
        for a, lr in action_immediate_rewards.items():
            q_dict[a] = lr + self.gamma * action_exp_v[a]

        best_action = max(action_immediate_rewards.keys(), key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _batch_evaluate_boards(
        self,
        boards: List[np.ndarray],
        use_d4: Optional[bool] = None,
    ) -> np.ndarray:
        """Evaluate a collection of boards in a single GPU forward pass with optional D4 symmetrization.

        Args:
            boards: List of 4x4 integer boards.
            use_d4: Optional toggle overriding instance setting.

        Returns:
            1D numpy array of values V(s) = max_{a in legal} Q(s, a), or 0.0 if terminal.
        """
        n = len(boards)
        if n == 0:
            return np.array([], dtype=np.float32)

        d4_active = self.use_d4_symmetry if use_d4 is None else use_d4

        boards_arr = np.stack(boards)
        clipped = np.clip(boards_arr, 0, MAX_TILE_VAL)
        exps = _POWER_LUT[clipped].astype(np.uint8)

        # Non-blocking transfer of compact uint8 representation to GPU
        exps_t = torch.from_numpy(exps).to(self.device, non_blocking=True)
        obs_t = F.one_hot(exps_t.long(), 16).permute(0, 3, 1, 2).float()

        # Compute legal action masks for all child boards
        masks_np = np.stack([self.core.get_action_mask(b) for b in boards])
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
    board: np.ndarray,
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

