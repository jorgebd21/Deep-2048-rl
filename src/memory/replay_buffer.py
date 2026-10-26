"""Prioritized Experience Replay (PER) buffer with SumTree backing."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.env.wrapper import MAX_TILE_VAL, _POWER_LUT, _IDENTITY_16


class SumTree:
    """Binary SumTree data structure for O(log N) prioritized experience sampling.

    Attributes:
        capacity: Maximum number of leaf nodes (transitions).
        tree: Array of size (2 * capacity - 1) holding prefix sums of priorities.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize SumTree with given leaf capacity."""
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_ptr = 0
        self.size = 0

    def total_priority(self) -> float:
        """Return root value representing the sum of all leaf priorities."""
        return float(self.tree[0])

    def update(self, tree_idx: int, priority: float) -> None:
        """Update leaf priority and propagate delta up to the root.

        Args:
            tree_idx: Index of leaf in the tree array.
            priority: New non-negative priority value.
        """
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        curr = tree_idx
        while curr > 0:
            curr = (curr - 1) // 2
            self.tree[curr] += delta

    def get(self, value: float) -> Tuple[int, float, int]:
        """Search tree from root to leaf for the segment matching sampled value.

        Args:
            value: Cumulative priority value sampled in [0, total_priority].

        Returns:
            Tuple containing:
                - tree_idx: Index in self.tree.
                - priority: Leaf priority.
                - data_idx: Corresponding index in circular data arrays [0, capacity - 1].
        """
        curr = 0
        tree_len = len(self.tree)

        while True:
            left = 2 * curr + 1
            right = left + 1

            if left >= tree_len:
                break

            if value <= self.tree[left]:
                curr = left
            else:
                value -= self.tree[left]
                if right < tree_len:
                    curr = right
                else:
                    curr = left
                    break

        data_idx = curr - self.capacity + 1
        return curr, float(self.tree[curr]), data_idx

    def get_batch(
        self, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized search from root to leaf for multiple sampled values simultaneously.

        Args:
            values: 1D array of shape (B,) with cumulative priority values in [0, total_priority].

        Returns:
            Tuple of:
                - tree_indices: 1D int64 array of leaf indices in self.tree.
                - priorities: 1D float64 array of leaf priorities.
                - data_indices: 1D int64 array of circular data buffer indices [0, capacity - 1].
        """
        b = len(values)
        v = np.array(values, dtype=np.float64, copy=True)
        curr = np.zeros(b, dtype=np.int64)
        tree_len = len(self.tree)
        max_depth = int(np.ceil(np.log2(tree_len)))

        for _ in range(max_depth):
            left = 2 * curr + 1
            right = left + 1

            valid_left = left < tree_len
            if not np.any(valid_left):
                break

            safe_left = np.minimum(left, tree_len - 1)
            tree_left = np.where(valid_left, self.tree[safe_left], 0.0)
            go_left = valid_left & (v <= tree_left)
            go_right = valid_left & ~go_left

            v = np.where(go_right, v - tree_left, v)
            safe_right = np.minimum(right, tree_len - 1)
            curr = np.where(go_left, safe_left, np.where(go_right, safe_right, curr))

        data_idx = np.clip(curr - self.capacity + 1, 0, self.capacity - 1)
        return curr, self.tree[curr], data_idx


def board_to_exponents(board: np.ndarray) -> np.ndarray:
    """Convert raw 4x4 integer tile values into packed uint8 powers (0..15)."""
    clipped = np.clip(board, 0, MAX_TILE_VAL)
    return _POWER_LUT[clipped].astype(np.uint8)


class PrioritizedReplayBuffer:
    """High-performance circular Prioritized Experience Replay (PER) buffer.

    Stores states in RAM as packed uint8 exponents to maintain memory footprint
    under 200 MB for 100,000+ transitions, expanding into multi-channel one-hot
    tensors on-the-fly during mini-batch sampling.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 150000,
        eps: float = 1e-5,
    ) -> None:
        """Initialize Prioritized Replay Buffer.

        Args:
            capacity: Maximum number of stored transitions.
            alpha: Priority exponent scaling (alpha=0 -> uniform, alpha=1 -> fully proportional).
            beta_start: Initial Importance Sampling (IS) exponent.
            beta_end: Final IS exponent.
            beta_steps: Number of steps to linearly anneal beta to beta_end.
            eps: Small positive constant preventing zero priority transitions.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.eps = eps

        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.total_steps = 0

        # Compact RAM-efficient circular arrays
        self.states = np.zeros((capacity, 4, 4), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 4, 4), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.action_masks = np.zeros((capacity, 4), dtype=np.bool_)
        self.next_action_masks = np.zeros((capacity, 4), dtype=np.bool_)

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self.tree.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
        priority: Optional[float] = None,
    ) -> None:
        """Append a new transition to the buffer.

        Args:
            state: 4x4 array of tile values or packed exponents.
            action: Integer action taken.
            reward: Step reward.
            next_state: Resulting 4x4 board.
            done: Episode termination flag.
            action_mask: Boolean mask of shape (4,).
            next_action_mask: Boolean mask for next state of shape (4,).
            priority: Optional priority value; defaults to current max priority.
        """
        ptr = self.tree.write_ptr
        tree_idx = ptr + self.capacity - 1

        # Pack states into uint8 exponents if raw integers
        if state.dtype != np.uint8:
            st = board_to_exponents(state)
        else:
            st = state

        if next_state.dtype != np.uint8:
            nst = board_to_exponents(next_state)
        else:
            nst = next_state

        self.states[ptr] = st
        self.actions[ptr] = action
        self.rewards[ptr] = float(reward)
        self.next_states[ptr] = nst
        self.dones[ptr] = bool(done)
        self.action_masks[ptr] = action_mask
        self.next_action_masks[ptr] = next_action_mask

        # Assign priority: max priority for new unseen experiences
        p = self.max_priority if priority is None else priority
        self.tree.update(tree_idx, p**self.alpha)

        self.tree.write_ptr = (ptr + 1) % self.capacity
        if self.tree.size < self.capacity:
            self.tree.size += 1

    def add_batch(
        self,
        states: np.ndarray,
        actions: Union[List[int], np.ndarray],
        rewards: Union[List[float], np.ndarray],
        next_states: np.ndarray,
        dones: Union[List[bool], np.ndarray],
        action_masks: np.ndarray,
        next_action_masks: np.ndarray,
        priorities: Optional[Union[List[float], np.ndarray]] = None,
    ) -> None:
        """Append a batch of transitions simultaneously to the circular buffer.

        Args:
            states: (N, 4, 4) array of tile values or packed exponents.
            actions: (N,) array of integer actions.
            rewards: (N,) array of float rewards.
            next_states: (N, 4, 4) array of boards or exponents.
            dones: (N,) array of boolean termination flags.
            action_masks: (N, 4) array of boolean masks.
            next_action_masks: (N, 4) array of boolean masks.
            priorities: Optional (N,) array of custom priorities.
        """
        n = len(actions)
        if n == 0:
            return

        if states.dtype != np.uint8:
            st = board_to_exponents(states)
        else:
            st = states

        if next_states.dtype != np.uint8:
            nst = board_to_exponents(next_states)
        else:
            nst = next_states

        actions_arr = np.asarray(actions, dtype=np.int64)
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        dones_arr = np.asarray(dones, dtype=bool)

        for i in range(n):
            ptr = self.tree.write_ptr
            tree_idx = ptr + self.capacity - 1

            self.states[ptr] = st[i]
            self.actions[ptr] = actions_arr[i]
            self.rewards[ptr] = float(rewards_arr[i])
            self.next_states[ptr] = nst[i]
            self.dones[ptr] = bool(dones_arr[i])
            self.action_masks[ptr] = action_masks[i]
            self.next_action_masks[ptr] = next_action_masks[i]

            p = self.max_priority if (priorities is None or priorities[i] is None) else float(priorities[i])
            self.tree.update(tree_idx, p**self.alpha)

            self.tree.write_ptr = (ptr + 1) % self.capacity
            if self.tree.size < self.capacity:
                self.tree.size += 1

    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        pin_memory: bool = False,
    ) -> Dict[str, Any]:
        """Sample a prioritized mini-batch with importance sampling correction.

        Args:
            batch_size: Number of transitions to sample.
            device: Target torch device (CPU or CUDA).
            pin_memory: Whether to pin host memory before asynchronous GPU transfer.

        Returns:
            Dictionary containing:
                - 'states': torch.Tensor of shape (B, 16, 4, 4)
                - 'actions': torch.Tensor of shape (B,)
                - 'rewards': torch.Tensor of shape (B,)
                - 'next_states': torch.Tensor of shape (B, 16, 4, 4)
                - 'dones': torch.Tensor of shape (B,)
                - 'action_masks': torch.Tensor of shape (B, 4)
                - 'next_action_masks': torch.Tensor of shape (B, 4)
                - 'weights': torch.Tensor of shape (B,) IS weights
                - 'indices': np.ndarray of shape (B,) tree indices for priority updates
        """
        if self.tree.size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} samples when buffer size is {self.tree.size}"
            )

        # Anneal beta linearly
        self.total_steps += 1
        fraction = min(1.0, self.total_steps / max(1, self.beta_steps))
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

        total_p = self.tree.total_priority()
        segment = total_p / batch_size

        # Vectorized sampling segments
        lows = segment * np.arange(batch_size, dtype=np.float64)
        highs = segment * np.arange(1, batch_size + 1, dtype=np.float64)
        vals = np.random.uniform(lows, highs)

        tree_indices, priorities, data_indices = self.tree.get_batch(vals)

        # Defensive boundary check
        data_indices = np.clip(data_indices, 0, max(0, self.tree.size - 1))

        # Compute Importance Sampling weights
        # P(i) = p_i / total_p
        # w_i = (N * P(i))^(-beta) / max_w
        probs = np.maximum(priorities / total_p, 1e-12)
        weights = (self.tree.size * probs) ** (-self.beta)
        weights = weights / (weights.max() + 1e-12)

        # Vectorized indexing from circular arrays
        batch_st = self.states[data_indices]
        batch_nst = self.next_states[data_indices]

        # Convert to tensors
        actions_t = torch.from_numpy(self.actions[data_indices]).long()
        rewards_t = torch.from_numpy(self.rewards[data_indices]).float()
        dones_t = torch.from_numpy(self.dones[data_indices]).float()
        masks_t = torch.from_numpy(self.action_masks[data_indices]).bool()
        next_masks_t = torch.from_numpy(self.next_action_masks[data_indices]).bool()
        weights_t = torch.from_numpy(weights.astype(np.float32)).float()

        if device is not None and device.type == "cuda":
            # Fast compact uint8 PCIe transfer (8 KB) + high-bandwidth GPU one-hot expansion
            st_raw = torch.from_numpy(batch_st)
            nst_raw = torch.from_numpy(batch_nst)
            if pin_memory:
                st_raw = st_raw.pin_memory()
                nst_raw = nst_raw.pin_memory()
                actions_t = actions_t.pin_memory()
                rewards_t = rewards_t.pin_memory()
                dones_t = dones_t.pin_memory()
                masks_t = masks_t.pin_memory()
                next_masks_t = next_masks_t.pin_memory()
                weights_t = weights_t.pin_memory()

            st_gpu = st_raw.to(device, non_blocking=True)
            nst_gpu = nst_raw.to(device, non_blocking=True)
            states_t = F.one_hot(st_gpu.long(), num_classes=16).permute(0, 3, 1, 2).float()
            next_states_t = F.one_hot(nst_gpu.long(), num_classes=16).permute(0, 3, 1, 2).float()

            actions_t = actions_t.to(device, non_blocking=True)
            rewards_t = rewards_t.to(device, non_blocking=True)
            dones_t = dones_t.to(device, non_blocking=True)
            masks_t = masks_t.to(device, non_blocking=True)
            next_masks_t = next_masks_t.to(device, non_blocking=True)
            weights_t = weights_t.to(device, non_blocking=True)
        else:
            # CPU fallback expansion
            states_one_hot = _IDENTITY_16[batch_st].transpose(0, 3, 1, 2)
            next_states_one_hot = _IDENTITY_16[batch_nst].transpose(0, 3, 1, 2)
            states_t = torch.from_numpy(states_one_hot).float()
            next_states_t = torch.from_numpy(next_states_one_hot).float()
            if device is not None:
                states_t = states_t.to(device)
                actions_t = actions_t.to(device)
                rewards_t = rewards_t.to(device)
                next_states_t = next_states_t.to(device)
                dones_t = dones_t.to(device)
                masks_t = masks_t.to(device)
                next_masks_t = next_masks_t.to(device)
                weights_t = weights_t.to(device)

        return {
            "states": states_t,
            "actions": actions_t,
            "rewards": rewards_t,
            "next_states": next_states_t,
            "dones": dones_t,
            "action_masks": masks_t,
            "next_action_masks": next_masks_t,
            "weights": weights_t,
            "indices": tree_indices,
        }

    def update_priorities(
        self,
        tree_indices: Union[np.ndarray, list],
        td_errors: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """Update transition priorities in the SumTree based on new TD-errors.

        Args:
            tree_indices: Array or list of tree leaf indices.
            td_errors: Absolute TD errors |y - Q(s, a)|.
        """
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        clipped_errors = np.abs(td_errors) + self.eps
        scaled_priorities = clipped_errors**self.alpha

        for tree_idx, prio in zip(tree_indices, scaled_priorities):
            self.tree.update(int(tree_idx), float(prio))
            if float(prio) > self.max_priority:
                self.max_priority = float(prio)
