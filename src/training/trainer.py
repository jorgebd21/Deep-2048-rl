"""High-level training orchestrator for 2048 Deep Reinforcement Learning."""

from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.env.wrapper import Game2048Env
from src.memory.replay_buffer import PrioritizedReplayBuffer
from src.training.agent import DDQNAgent


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for DDQN training loop."""
    total_episodes: int = 2000
    warmup_steps: int = 1000
    batch_size: int = 512
    buffer_capacity: int = 100000
    train_freq: int = 1
    use_soft_update: bool = True
    tau: float = 0.005
    target_update_freq: int = 1000  # For hard update fallback
    eval_freq: int = 50
    eval_episodes: int = 10
    checkpoint_dir: str = "checkpoints"
    save_latest_freq: int = 100
    seed: Optional[int] = 42


class Trainer:
    """Manages the full lifecycle of agent-environment interaction, replay, and evaluation."""

    def __init__(
        self,
        env: Game2048Env,
        agent: DDQNAgent,
        buffer: Optional[PrioritizedReplayBuffer] = None,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        """Initialize trainer with environment, agent, buffer, and configuration."""
        self.config = config or TrainingConfig()
        self.env = env
        self.agent = agent
        self.buffer = buffer or PrioritizedReplayBuffer(capacity=self.config.buffer_capacity)

        self.best_eval_score = 0.0
        self.history: Dict[str, List[float]] = {
            "episode_scores": [],
            "episode_max_tiles": [],
            "episode_lengths": [],
            "losses": [],
            "eval_scores": [],
            "eval_max_tiles": [],
        }

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def warmup(self) -> None:
        """Populate experience buffer with uniform random legal transitions."""
        obs, info = self.env.reset(seed=self.config.seed)
        step_count = 0

        while step_count < self.config.warmup_steps:
            action_mask = info["action_mask"]
            valid_actions = np.flatnonzero(action_mask)
            if len(valid_actions) == 0:
                obs, info = self.env.reset()
                continue

            action = int(np.random.choice(valid_actions))
            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            next_mask = next_info["action_mask"]
            done = terminated or truncated

            self.buffer.add(
                state=info["board"],
                action=action,
                reward=reward,
                next_state=next_info["board"],
                done=done,
                action_mask=action_mask,
                next_action_mask=next_mask,
            )

            step_count += 1
            if done:
                obs, info = self.env.reset()
            else:
                obs, info = next_obs, next_info

    def evaluate(self, num_episodes: Optional[int] = None) -> Dict[str, float]:
        """Execute deterministic evaluation runs with zero exploration (epsilon=0).

        Args:
            num_episodes: Number of evaluation episodes; defaults to config.eval_episodes.

        Returns:
            Dictionary containing mean score, highest tile observed, and mean steps.
        """
        n_eps = num_episodes or self.config.eval_episodes
        eval_scores = []
        eval_max_tiles = []
        eval_lengths = []

        for ep_idx in range(n_eps):
            obs, info = self.env.reset()
            done = False
            ep_steps = 0

            while not done:
                action = self.agent.select_action(
                    state=obs,
                    action_mask=info["action_mask"],
                    deterministic=True,
                )
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_steps += 1

            eval_scores.append(float(info["score"]))
            eval_max_tiles.append(float(info["max_tile"]))
            eval_lengths.append(float(ep_steps))

        return {
            "mean_score": float(np.mean(eval_scores)),
            "max_tile": float(np.max(eval_max_tiles)),
            "mean_steps": float(np.mean(eval_lengths)),
        }

    def train_step(self) -> Optional[float]:
        """Sample mini-batch from buffer, optimize policy, and update target network."""
        if len(self.buffer) < self.config.batch_size:
            return None

        batch = self.buffer.sample(self.config.batch_size, device=self.agent.device)
        loss, td_errors = self.agent.update(batch)

        # Update PER tree priorities
        self.buffer.update_priorities(batch["indices"], td_errors)

        # Synchronize target network
        if self.config.use_soft_update:
            self.agent.update_target_soft(self.config.tau)
        elif self.agent.total_steps % self.config.target_update_freq == 0:
            self.agent.update_target_hard()

        return loss

    def train(self, max_episodes: Optional[int] = None) -> Dict[str, Any]:
        """Run complete DDQN training loop.

        Args:
            max_episodes: Optional episode override for testing.

        Returns:
            Training history dictionary.
        """
        total_episodes = max_episodes or self.config.total_episodes
        self.warmup()

        for ep in range(1, total_episodes + 1):
            obs, info = self.env.reset()
            done = False
            ep_steps = 0
            ep_losses = []

            while not done:
                # Select legal action with epsilon-greedy exploration
                action = self.agent.select_action(
                    state=obs,
                    action_mask=info["action_mask"],
                    deterministic=False,
                )

                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                done = terminated or truncated

                # Store transition
                self.buffer.add(
                    state=info["board"],
                    action=action,
                    reward=reward,
                    next_state=next_info["board"],
                    done=done,
                    action_mask=info["action_mask"],
                    next_action_mask=next_info["action_mask"],
                )

                # Gradient step
                if ep_steps % self.config.train_freq == 0:
                    loss = self.train_step()
                    if loss is not None:
                        ep_losses.append(loss)

                # Decay epsilon per environment transition
                self.agent.decay_epsilon()

                obs, info = next_obs, next_info
                ep_steps += 1

            # Episode logging
            self.history["episode_scores"].append(float(info["score"]))
            self.history["episode_max_tiles"].append(float(info["max_tile"]))
            self.history["episode_lengths"].append(float(ep_steps))
            if ep_losses:
                self.history["losses"].append(float(np.mean(ep_losses)))

            # Periodic Evaluation
            if ep % self.config.eval_freq == 0:
                eval_metrics = self.evaluate()
                self.history["eval_scores"].append(eval_metrics["mean_score"])
                self.history["eval_max_tiles"].append(eval_metrics["max_tile"])

                # Checkpoint best model
                if eval_metrics["mean_score"] > self.best_eval_score:
                    self.best_eval_score = eval_metrics["mean_score"]
                    best_path = os.path.join(self.config.checkpoint_dir, "checkpoint_best.pt")
                    self.agent.save_checkpoint(
                        best_path,
                        metadata={
                            "episode": ep,
                            "eval_mean_score": self.best_eval_score,
                            "eval_max_tile": eval_metrics["max_tile"],
                        },
                    )

            # Periodic latest checkpoint
            if ep % self.config.save_latest_freq == 0:
                latest_path = os.path.join(self.config.checkpoint_dir, "checkpoint_latest.pt")
                self.agent.save_checkpoint(
                    latest_path,
                    metadata={"episode": ep, "steps": self.agent.total_steps},
                )

        return self.history
