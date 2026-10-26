"""Lightweight Expectimax & Lookahead search for 2048 DRL agents.

Implements batched n-ply Expectimax search over deterministic slide-and-merge
actions and stochastic tile spawns (90% for 2, 10% for 4). Evaluates leaf states
using a trained Dueling Double DQN network in a single GPU forward pass.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.core import Action, Game2048Core
from src.env.wrapper import _POWER_LUT, MAX_TILE_VAL, one_hot_encode_board
from src.utils.cuda_utils import get_device


class ExpectimaxSearcher:
    """High-throughput Expectimax searcher leveraging batched neural network evaluation.

    Supports:
        - Depth 0: Pure greedy evaluation via direct Q(s, a) forward pass.
        - Depth 1: 1-ply lookahead (action slide -> evaluate afterstate).
        - Depth 2: 2-ply Expectimax (action slide -> chance spawns 2/4 -> evaluate leaf states).
        - Depth 3: 3-ply Expectimax (action -> chance -> next action -> evaluate leaf states).
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        gamma: float = 0.99,
        depth: int = 2,
        max_empty_samples: Optional[int] = None,
    ) -> None:
        """Initialize searcher with model and search configuration.

        Args:
            model: PyTorch policy or Q-network (e.g. Dueling2048Net).
            device: Compute device (defaults to GPU if available).
            gamma: Discount factor for Bellman expectation calculation.
            depth: Default search depth (0 for greedy, 1 for 1-ply, 2 for 2-ply expectimax).
            max_empty_samples: Optional maximum number of empty cells to sample per chance node.
                None evaluates all empty cells deterministically.
        """
        self.model = model
        self.device = device or get_device(prefer_gpu=True)
        self.gamma = gamma
        self.depth = depth
        self.max_empty_samples = max_empty_samples
        self.core = Game2048Core()

    def select_action(
        self,
        board: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        depth: Optional[int] = None,
    ) -> int:
        """Select best action using lookahead search.

        Args:
            board: Current 4x4 integer game board.
            action_mask: Optional boolean array of legal actions.
            depth: Optional search depth overriding instance default.

        Returns:
            Selected integer action in {0, 1, 2, 3}.
        """
        best_action, _ = self.search(board, action_mask=action_mask, depth=depth)
        return best_action

    def search(
        self,
        board: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        depth: Optional[int] = None,
    ) -> Tuple[int, Dict[int, float]]:
        """Perform Expectimax search from current board state.

        Args:
            board: 4x4 integer numpy array.
            action_mask: Optional boolean array of shape (4,). If None, calculated via core.
            depth: Search depth (0, 1, 2, or 3). Defaults to self.depth.

        Returns:
            Tuple of (best_action, q_values_dict).
        """
        d = self.depth if depth is None else depth
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
            return self._search_greedy(board, mask)
        elif d == 1:
            return self._search_depth_1(board, legal_actions)
        elif d == 2:
            return self._search_depth_2(board, legal_actions)
        elif d == 3:
            return self._search_depth_3(board, legal_actions)
        else:
            # Fallback to depth 2 for higher depths to preserve high throughput
            return self._search_depth_2(board, legal_actions)

    def _search_greedy(
        self, board: np.ndarray, mask: np.ndarray
    ) -> Tuple[int, Dict[int, float]]:
        """Evaluate root state directly with model (depth 0)."""
        obs = one_hot_encode_board(board)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            q_vals = self.model(obs_t, action_mask=mask_t).squeeze(0).cpu().numpy()

        best_action = int(np.argmax(q_vals))
        q_dict = {a: float(q_vals[a]) for a in range(4)}
        return best_action, q_dict

    def _search_depth_1(
        self, board: np.ndarray, legal_actions: np.ndarray
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

        v_vals = self._batch_evaluate_boards(after_boards)

        q_dict: Dict[int, float] = {a: -1e9 for a in range(4)}
        for a, lr, v in zip(valid_actions, imm_rewards, v_vals):
            q_dict[a] = lr + self.gamma * v

        best_action = max(valid_actions, key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _search_depth_2(
        self, board: np.ndarray, legal_actions: np.ndarray
    ) -> Tuple[int, Dict[int, float]]:
        """2-ply Expectimax: simulate move -> tile spawns 2/4 -> batched leaf evaluation.

        For each legal action:
            1. Simulates deterministic slide-and-merge yielding s_after and immediate reward r.
            2. Branches over all empty cells with nature spawn probabilities (90% for 2, 10% for 4).
            3. Evaluates all resulting leaf states in a single batched GPU forward pass.
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
                # Board full after slide: evaluate afterstate directly
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

        # Evaluate all candidate leaf states concurrently in a single GPU batch
        v_vals = self._batch_evaluate_boards(all_children)

        # Aggregate expected values per action
        action_exp_v: Dict[int, float] = {a: 0.0 for a in action_immediate_rewards}
        for v, w, a in zip(v_vals, all_weights, all_action_owners):
            action_exp_v[a] += w * v

        q_dict: Dict[int, float] = {a: -1e9 for a in range(4)}
        for a, lr in action_immediate_rewards.items():
            q_dict[a] = lr + self.gamma * action_exp_v[a]

        best_action = max(action_immediate_rewards.keys(), key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _search_depth_3(
        self, board: np.ndarray, legal_actions: np.ndarray
    ) -> Tuple[int, Dict[int, float]]:
        """3-ply Expectimax: Action -> Chance Spawn -> Next Action -> Leaf Evaluation.

        Limits empty cell sampling to keep total leaf evaluations within a single GPU batch.
        """
        all_leaves: List[np.ndarray] = []
        leaf_metadata: List[Tuple[int, float, float]] = []  # (root_act, chance_weight, next_lr)
        action_immediate_rewards: Dict[int, float] = {}

        max_chance = 4 if self.max_empty_samples is None else min(4, self.max_empty_samples)

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
                sampled_cells = np.random.choice(empty_cells, max_chance, replace=False)
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
                        # Terminal chance node
                        all_leaves.append(s_chance)
                        leaf_metadata.append((a1, chance_w, 0.0))
                    else:
                        for a2 in legal2:
                            nb2, _, lr2, ch2 = Game2048Core._execute_slide(s_chance, a2)
                            if ch2:
                                all_leaves.append(nb2)
                                leaf_metadata.append((a1, chance_w, lr2))

        if len(all_leaves) == 0:
            return self._search_depth_2(board, legal_actions)

        leaf_v = self._batch_evaluate_boards(all_leaves)

        action_exp_v: Dict[int, float] = {a: 0.0 for a in action_immediate_rewards}
        for v, (a1, chance_w, lr2) in zip(leaf_v, leaf_metadata):
            q_ply2 = lr2 + self.gamma * v
            action_exp_v[a1] += chance_w * q_ply2

        q_dict: Dict[int, float] = {a: -1e9 for a in range(4)}
        for a, lr in action_immediate_rewards.items():
            q_dict[a] = lr + self.gamma * action_exp_v[a]

        best_action = max(action_immediate_rewards.keys(), key=lambda a: q_dict[a])
        return int(best_action), q_dict

    def _batch_evaluate_boards(self, boards: List[np.ndarray]) -> np.ndarray:
        """Evaluate a collection of boards in a single GPU forward pass.

        Args:
            boards: List of 4x4 integer boards.

        Returns:
            1D numpy array of values V(s) = max_{a in legal} Q(s, a), or 0.0 if terminal.
        """
        n = len(boards)
        if n == 0:
            return np.array([], dtype=np.float32)

        boards_arr = np.stack(boards)
        clipped = np.clip(boards_arr, 0, MAX_TILE_VAL)
        exps = _POWER_LUT[clipped].astype(np.uint8)

        # Non-blocking transfer of compact uint8 representation to GPU
        exps_t = torch.from_numpy(exps).to(self.device, non_blocking=True)
        obs_t = F.one_hot(exps_t.long(), 16).permute(0, 3, 1, 2).float()

        # Compute legal action masks for all child boards
        masks_np = np.stack([self.core.get_action_mask(b) for b in boards])
        masks_t = torch.from_numpy(masks_np).bool().to(self.device, non_blocking=True)

        self.model.eval()
        with torch.no_grad():
            q_vals = self.model(obs_t, action_mask=masks_t)
            has_legal = masks_t.any(dim=-1)
            # For terminal states with no legal actions, set value to 0.0
            v_vals = torch.where(
                has_legal,
                q_vals.max(dim=-1).values,
                torch.zeros_like(q_vals[:, 0]),
            )
            return v_vals.cpu().numpy()


def expectimax_search(
    board: np.ndarray,
    model: nn.Module,
    device: Optional[torch.device] = None,
    depth: int = 2,
    gamma: float = 0.99,
    max_empty_samples: Optional[int] = None,
) -> Tuple[int, Dict[int, float]]:
    """Functional convenience wrapper for ExpectimaxSearcher.search."""
    searcher = ExpectimaxSearcher(
        model=model,
        device=device,
        gamma=gamma,
        depth=depth,
        max_empty_samples=max_empty_samples,
    )
    return searcher.search(board, depth=depth)
