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
    enable_tensorboard: bool = True
    log_dir: Optional[str] = None
    pin_memory: bool = True


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
        self.total_env_steps = 0
        self.total_opt_steps = 0
        self.start_time = time.time()

        self.history: Dict[str, List[float]] = {
            "episode_scores": [],
            "episode_max_tiles": [],
            "episode_lengths": [],
            "losses": [],
            "eval_scores": [],
            "eval_max_tiles": [],
        }

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        # TensorBoard Telemetry
        self.writer = None
        if self.config.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = self.config.log_dir or os.path.join(
                    "runs", f"run_{time.strftime('%Y%m%d_%H%M%S')}"
                )
                self.writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                self.writer = None

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
            Dictionary containing evaluation summary metrics.
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

        max_tiles_np = np.array(eval_max_tiles)
        metrics = {
            "mean_score": float(np.mean(eval_scores)),
            "max_tile": float(np.max(eval_max_tiles)),
            "mean_steps": float(np.mean(eval_lengths)),
            "reach_512_rate": float(np.mean(max_tiles_np >= 512)),
            "reach_1024_rate": float(np.mean(max_tiles_np >= 1024)),
            "reach_2048_rate": float(np.mean(max_tiles_np >= 2048)),
        }

        if self.writer is not None:
            step = self.agent.total_steps
            self.writer.add_scalar("eval/mean_score", metrics["mean_score"], step)
            self.writer.add_scalar("eval/max_tile", metrics["max_tile"], step)
            self.writer.add_scalar("eval/mean_steps", metrics["mean_steps"], step)
            self.writer.add_scalar("eval/reach_512_rate", metrics["reach_512_rate"], step)
            self.writer.add_scalar("eval/reach_1024_rate", metrics["reach_1024_rate"], step)
            self.writer.add_scalar("eval/reach_2048_rate", metrics["reach_2048_rate"], step)

        return metrics

    def train_step(self) -> Optional[float]:
        """Sample mini-batch from buffer, optimize policy, and update target network."""
        if len(self.buffer) < self.config.batch_size:
            return None

        batch = self.buffer.sample(
            self.config.batch_size,
            device=self.agent.device,
            pin_memory=self.config.pin_memory,
        )
        loss, td_errors = self.agent.update(batch)
        self.total_opt_steps += 1

        # Update PER tree priorities
        self.buffer.update_priorities(batch["indices"], td_errors)

        # Synchronize target network
        if self.config.use_soft_update:
            self.agent.update_target_soft(self.config.tau)
        elif self.agent.total_steps % self.config.target_update_freq == 0:
            self.agent.update_target_hard()

        # Telemetry logging per gradient update
        if self.writer is not None and self.total_opt_steps % 10 == 0:
            agent_metrics = self.agent.get_last_metrics()
            step = self.agent.total_steps
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/mean_td_error", agent_metrics["mean_td_error"], step)
            self.writer.add_scalar("train/mean_q", agent_metrics["mean_q"], step)
            self.writer.add_scalar("train/epsilon", self.agent.epsilon, step)

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
        self.start_time = time.time()

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
                self.total_env_steps += 1

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
            score = float(info["score"])
            max_tile = float(info["max_tile"])
            self.history["episode_scores"].append(score)
            self.history["episode_max_tiles"].append(max_tile)
            self.history["episode_lengths"].append(float(ep_steps))
            if ep_losses:
                self.history["losses"].append(float(np.mean(ep_losses)))

            # Tensorboard Episode Logging
            if self.writer is not None:
                step = self.agent.total_steps
                elapsed = max(1e-5, time.time() - self.start_time)
                env_sps = self.total_env_steps / elapsed
                samples_sps = (self.total_opt_steps * self.config.batch_size) / elapsed

                self.writer.add_scalar("episode/score", score, step)
                self.writer.add_scalar("episode/steps", ep_steps, step)
                self.writer.add_scalar("episode/max_tile", max_tile, step)
                self.writer.add_scalar("episode/reach_512", 1.0 if max_tile >= 512 else 0.0, step)
                self.writer.add_scalar("episode/reach_1024", 1.0 if max_tile >= 1024 else 0.0, step)
                self.writer.add_scalar("episode/reach_2048", 1.0 if max_tile >= 2048 else 0.0, step)
                self.writer.add_scalar("throughput/env_steps_per_sec", env_sps, step)
                self.writer.add_scalar("throughput/samples_per_sec", samples_sps, step)

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
                            "reach_512_rate": eval_metrics["reach_512_rate"],
                            "reach_1024_rate": eval_metrics["reach_1024_rate"],
                            "reach_2048_rate": eval_metrics["reach_2048_rate"],
                        },
                    )

            # Periodic latest checkpoint
            if ep % self.config.save_latest_freq == 0:
                latest_path = os.path.join(self.config.checkpoint_dir, "checkpoint_latest.pt")
                self.agent.save_checkpoint(
                    latest_path,
                    metadata={"episode": ep, "steps": self.agent.total_steps},
                )

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        return self.history
