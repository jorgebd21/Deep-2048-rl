"""Double Dueling Deep Q-Network (DDQN) Agent."""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dueling_dqn import Dueling2048Net
from src.utils.cuda_utils import get_device


class DDQNAgent:
    """Double DQN Agent incorporating Dueling architecture and legal action masking.

    Decouples action selection from action evaluation to eliminate positive maximization
    bias in Bellman target calculations:
        a* = argmax_{a'} Q(s', a'; theta_online, mask_next)
        y  = r + (1 - done) * gamma * Q(s', a*; theta_target)
    """

    def __init__(
        self,
        in_channels: int = 16,
        num_actions: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_grad_norm: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay_steps: int = 150000,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize online and target networks, optimizer, and exploration schedule."""
        self.device = device if device is not None else get_device(prefer_gpu=True)
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.num_actions = num_actions

        # Exploration hyperparameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0
        self.last_metrics: Dict[str, float] = {
            "loss": 0.0,
            "mean_td_error": 0.0,
            "mean_q": 0.0,
        }

        # Online policy network and target network
        self.policy_net = Dueling2048Net(
            in_channels=in_channels, num_actions=num_actions
        ).to(self.device)
        self.target_net = Dueling2048Net(
            in_channels=in_channels, num_actions=num_actions
        ).to(self.device)

        # Initialize target network with identical weights and freeze gradients
        self.update_target_hard()
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action_mask: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> int:
        """Select an action following epsilon-greedy policy with strict legal masking.

        Args:
            state: Observation tensor of shape (16, 4, 4) or batch (1, 16, 4, 4).
            action_mask: Boolean array of shape (4,) indicating legal actions.
            deterministic: If True, bypass exploration and choose greedy best action.

        Returns:
            Integer action in {0, 1, 2, 3}.
        """
        if isinstance(action_mask, torch.Tensor):
            mask_np = action_mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(action_mask, dtype=bool)

        valid_actions = np.flatnonzero(mask_np)
        if len(valid_actions) == 0:
            return 0  # Fallback if no legal actions remain

        # Exploration branch: select uniformly only among legal actions
        if not deterministic and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))

        # Exploitation branch: greedy action selection through online network
        self.policy_net.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                s_t = torch.from_numpy(state).float()
            else:
                s_t = state.float()

            if s_t.dim() == 3:
                s_t = s_t.unsqueeze(0)  # Add batch dimension -> (1, 16, 4, 4)

            s_t = s_t.to(self.device)
            m_t = torch.from_numpy(mask_np).bool().unsqueeze(0).to(self.device)

            q_values = self.policy_net(s_t, action_mask=m_t)
            best_action = int(q_values.argmax(dim=-1).item())

        return best_action

    def select_actions_batch(
        self,
        states: Union[np.ndarray, torch.Tensor],
        action_masks: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select actions for a batch of N environments concurrently.

        Executes a single batched GPU forward pass with legal action masking.

        Args:
            states: Batch of observations (N, 16, 4, 4) or uint8 boards (N, 4, 4).
            action_masks: (N, 4) boolean array of legal moves.
            deterministic: If True, bypass exploration and pick argmax action.

        Returns:
            1D NumPy array of shape (N,) containing selected integer actions.
        """
        if isinstance(action_masks, torch.Tensor):
            masks_np = action_masks.detach().cpu().numpy().astype(bool)
            masks_t = action_masks.bool().to(self.device, non_blocking=True)
        else:
            masks_np = np.asarray(action_masks, dtype=bool)
            masks_t = torch.from_numpy(masks_np).bool().to(self.device, non_blocking=True)

        n = len(masks_np)

        self.policy_net.eval()
        with torch.no_grad():
            if isinstance(states, np.ndarray):
                if states.ndim == 3 and states.shape[1:] == (4, 4):
                    if states.dtype != np.uint8:
                        from src.memory.replay_buffer import board_to_exponents
                        st_exp = board_to_exponents(states)
                    else:
                        st_exp = states
                    st_raw = torch.from_numpy(st_exp).to(self.device, non_blocking=True)
                    s_t = F.one_hot(st_raw.long(), num_classes=16).permute(0, 3, 1, 2).float()
                else:
                    s_t = torch.from_numpy(states).float().to(self.device, non_blocking=True)
            else:
                s_t = states.float().to(self.device, non_blocking=True)

            q_values = self.policy_net(s_t, action_mask=masks_t)
            best_actions = q_values.argmax(dim=-1).cpu().numpy()

        if not deterministic and self.epsilon > 0.0:
            explore_mask = np.random.rand(n) < self.epsilon
            for idx in np.flatnonzero(explore_mask):
                valid = np.flatnonzero(masks_np[idx])
                if len(valid) > 0:
                    best_actions[idx] = int(np.random.choice(valid))

        return best_actions

    def compute_td_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Double DQN Bellman loss and absolute TD-errors.

        Args:
            batch: Dictionary containing mini-batch tensors:
                - 'states': (B, 16, 4, 4)
                - 'actions': (B,)
                - 'rewards': (B,)
                - 'next_states': (B, 16, 4, 4)
                - 'dones': (B,)
                - 'next_action_masks': (B, 4)
                - 'weights': (B,) Importance Sampling weights

        Returns:
            Tuple of:
                - weighted_loss: Scalar torch.Tensor for backpropagation.
                - td_errors: 1D torch.Tensor of absolute TD errors |y - Q(s, a)|.
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        next_masks = batch["next_action_masks"].to(self.device)
        weights = batch["weights"].to(self.device)

        # 1. Evaluate current Q-values Q(s, a; theta_online)
        q_pred = self.policy_net(states)  # (B, 4)
        current_q = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # 2. Compute Double DQN Bellman targets
        with torch.no_grad():
            # Action selection: argmax over next state using online net with next legal mask
            q_next_online = self.policy_net(next_states, action_mask=next_masks)
            best_next_actions = q_next_online.argmax(dim=1, keepdim=True)  # (B, 1)

            # Action evaluation: evaluate chosen action with target net
            q_next_target_all = self.target_net(next_states)  # (B, 4)
            next_q_target = q_next_target_all.gather(1, best_next_actions).squeeze(1)

            # Bellman equation: y = r + (1 - done) * gamma * Q_target(s', a*)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_target

        # 3. Absolute TD errors for PER update: |y - Q(s, a)|
        td_errors = torch.abs(target_q - current_q)

        # 4. Importance Sampling weighted Smooth L1 (Huber) loss
        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction="none", beta=1.0)
        weighted_loss = (elementwise_loss * weights).mean()

        # Cache metrics for telemetry
        self.last_metrics["mean_q"] = float(current_q.detach().mean().item())
        self.last_metrics["mean_td_error"] = float(td_errors.detach().mean().item())

        return weighted_loss, td_errors

    def update(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, np.ndarray]:
        """Perform one optimization step on policy network.

        Args:
            batch: Dictionary of sampled transitions and IS weights.

        Returns:
            Tuple of:
                - loss_value: Float loss value.
                - td_errors_np: 1D NumPy array of absolute TD-errors.
        """
        self.policy_net.train()
        self.optimizer.zero_grad()

        loss, td_errors = self.compute_td_loss(batch)
        loss.backward()

        # Gradient clipping to stabilize backpropagation
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer.step()

        loss_val = float(loss.item())
        self.last_metrics["loss"] = loss_val

        return loss_val, td_errors.detach().cpu().numpy()

    def get_last_metrics(self) -> Dict[str, float]:
        """Return the most recent optimization metrics dictionary."""
        return self.last_metrics.copy()

    def update_target_hard(self) -> None:
        """Copy weights directly from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_target_soft(self, tau: Optional[float] = None) -> None:
        """Perform Polyak soft target update: theta_target <- tau * theta + (1 - tau) * theta_target."""
        t = self.tau if tau is None else tau
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                t * policy_param.data + (1.0 - t) * target_param.data
            )

    def decay_epsilon(self) -> float:
        """Decay exploration parameter epsilon linearly over configured decay steps."""
        self.total_steps += 1
        fraction = min(1.0, self.total_steps / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start - fraction * (
            self.epsilon_start - self.epsilon_min
        )
        return self.epsilon

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save network parameters, optimizer state, and training metadata."""
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(
        self, path: str, resume_epsilon: Optional[float] = None
    ) -> Dict[str, Any]:
        """Load network parameters, optimizer state, and training metadata.

        Args:
            path: Path to checkpoint file.
            resume_epsilon: Optional epsilon to override for fine-tuning exploration.

        Returns:
            Checkpoint metadata dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        metadata = {}

        if isinstance(checkpoint, dict):
            if "policy_net_state_dict" in checkpoint:
                self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            elif "model_state_dict" in checkpoint:
                self.policy_net.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.policy_net.load_state_dict(checkpoint)

            if "target_net_state_dict" in checkpoint:
                self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            else:
                self.update_target_hard()

            if "optimizer_state_dict" in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception:
                    pass

            self.total_steps = checkpoint.get("total_steps", 0)
            if resume_epsilon is not None:
                self.epsilon = float(resume_epsilon)
                self.epsilon_start = float(resume_epsilon)
            else:
                self.epsilon = checkpoint.get("epsilon", self.epsilon_min)

            metadata = checkpoint.get("metadata", {})
        else:
            self.policy_net.load_state_dict(checkpoint)
            self.update_target_hard()
            if resume_epsilon is not None:
                self.epsilon = float(resume_epsilon)
                self.epsilon_start = float(resume_epsilon)

        return metadata
