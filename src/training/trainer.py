"""High-level training orchestrator for 2048 Deep Reinforcement Learning."""

from dataclasses import dataclass, field
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from src.env.wrapper import Game2048Env
from src.env.vectorized import VectorizedGame2048Env
from src.memory.replay_buffer import PrioritizedReplayBuffer
from src.training.agent import DDQNAgent


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for DDQN training loop."""
    total_episodes: int = 2000
    warmup_steps: int = 2000
    batch_size: int = 512
    buffer_capacity: int = 100000
    num_envs: int = 32
    use_vectorized: bool = True
    train_freq: int = 1  # Number of vectorized steps between gradient updates
    gradient_steps_per_train: int = 1
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
    use_tqdm: bool = True


class Trainer:
    """Manages the full lifecycle of agent-environment interaction, replay, and evaluation."""

    def __init__(
        self,
        env: Optional[Union[Game2048Env, VectorizedGame2048Env]] = None,
        agent: Optional[DDQNAgent] = None,
        buffer: Optional[PrioritizedReplayBuffer] = None,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        """Initialize trainer with environment, agent, buffer, and configuration."""
        self.config = config or TrainingConfig()

        if env is not None:
            self.env = env
        elif self.config.use_vectorized:
            self.env = VectorizedGame2048Env(
                num_envs=self.config.num_envs, seed=self.config.seed
            )
        else:
            self.env = Game2048Env()

        self.agent = agent or DDQNAgent()
        self.buffer = buffer or PrioritizedReplayBuffer(capacity=self.config.buffer_capacity)
        self.eval_env = Game2048Env()

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

        # Defensive checkpoint directory handling
        ckpt_dir = self.config.checkpoint_dir
        if os.path.isfile(ckpt_dir):
            ckpt_dir = os.path.dirname(ckpt_dir) or "checkpoints"
            self.config.checkpoint_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

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
        if isinstance(self.env, VectorizedGame2048Env):
            obs, infos = self.env.reset(seed=self.config.seed)
            boards = np.array([inf["board"] for inf in infos])
            collected = 0

            while collected < self.config.warmup_steps:
                masks = self.env.get_action_masks()
                actions = np.array([
                    int(np.random.choice(np.flatnonzero(m))) if np.any(m) else 0
                    for m in masks
                ], dtype=np.int64)

                next_obs, rewards, terminated, truncated, next_infos = self.env.step(actions)
                next_boards = np.array([inf["board"] for inf in next_infos])
                next_masks = self.env.get_action_masks()

                self.buffer.add_batch(
                    states=boards,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_boards,
                    dones=terminated,
                    action_masks=masks,
                    next_action_masks=next_masks,
                )

                collected += self.env.num_envs
                obs, infos, boards = next_obs, next_infos, next_boards
        else:
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
            obs, info = self.eval_env.reset()
            done = False
            ep_steps = 0

            while not done:
                action = self.agent.select_action(
                    state=obs,
                    action_mask=info["action_mask"],
                    deterministic=True,
                )
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
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

    def _train_vectorized(self, total_episodes: int) -> Dict[str, Any]:
        """Execute high-throughput parallelized vectorized training loop."""
        from tqdm import tqdm

        completed_episodes = 0
        obs, infos = self.env.reset(seed=self.config.seed)
        boards = np.array([inf["board"] for inf in infos])

        step_window_time = time.time()
        step_window_count = 0

        recent_scores: List[float] = []
        recent_tiles: List[float] = []
        recent_losses: List[float] = []

        pbar = None
        if self.config.use_tqdm:
            pbar = tqdm(
                total=total_episodes,
                desc="DDQN Vectorized Training",
                dynamic_ncols=True,
                file=sys.stdout,
            )

        vec_steps = 0

        while completed_episodes < total_episodes:
            masks = self.env.get_action_masks()

            # Batched GPU action selection
            actions = self.agent.select_actions_batch(
                states=obs,
                action_masks=masks,
                deterministic=False,
            )

            # Parallel environment stepping
            next_obs, rewards, terminated, truncated, next_infos = self.env.step(actions)
            next_boards = np.array([inf["board"] for inf in next_infos])
            next_masks = self.env.get_action_masks()

            # Batch transition ingestion into circular PER buffer
            self.buffer.add_batch(
                states=boards,
                actions=actions,
                rewards=rewards,
                next_states=next_boards,
                dones=terminated,
                action_masks=masks,
                next_action_masks=next_masks,
            )

            num_env_transitions = self.env.num_envs
            self.total_env_steps += num_env_transitions
            step_window_count += num_env_transitions
            vec_steps += 1

            # Decay epsilon
            for _ in range(num_env_transitions):
                self.agent.decay_epsilon()

            # Gradient update
            if vec_steps % self.config.train_freq == 0:
                for _ in range(self.config.gradient_steps_per_train):
                    loss = self.train_step()
                    if loss is not None:
                        recent_losses.append(loss)
                        if len(recent_losses) > 50:
                            recent_losses.pop(0)

            # Process completed episodes
            for inf in next_infos:
                if "final_info" in inf:
                    fin = inf["final_info"]
                    completed_episodes += 1
                    ep_score = float(fin["score"])
                    ep_tile = float(fin["max_tile"])
                    ep_steps = float(fin["steps"])

                    self.history["episode_scores"].append(ep_score)
                    self.history["episode_max_tiles"].append(ep_tile)
                    self.history["episode_lengths"].append(ep_steps)

                    recent_scores.append(ep_score)
                    recent_tiles.append(ep_tile)
                    if len(recent_scores) > 30:
                        recent_scores.pop(0)
                        recent_tiles.pop(0)

                    # Compute instant and rolling throughput
                    now = time.time()
                    elapsed_window = now - step_window_time
                    if elapsed_window >= 1.0:
                        env_sps = step_window_count / elapsed_window
                        step_window_time = now
                        step_window_count = 0
                    else:
                        elapsed_total = max(1e-5, now - self.start_time)
                        env_sps = self.total_env_steps / elapsed_total

                    samples_sps = (self.total_opt_steps * self.config.batch_size) / max(1e-5, now - self.start_time)

                    # TensorBoard Logging
                    if self.writer is not None:
                        step = self.agent.total_steps
                        self.writer.add_scalar("episode/score", ep_score, step)
                        self.writer.add_scalar("episode/steps", ep_steps, step)
                        self.writer.add_scalar("episode/max_tile", ep_tile, step)
                        self.writer.add_scalar("episode/reach_512", 1.0 if ep_tile >= 512 else 0.0, step)
                        self.writer.add_scalar("episode/reach_1024", 1.0 if ep_tile >= 1024 else 0.0, step)
                        self.writer.add_scalar("episode/reach_2048", 1.0 if ep_tile >= 2048 else 0.0, step)
                        self.writer.add_scalar("throughput/env_steps_per_sec", env_sps, step)
                        self.writer.add_scalar("throughput/samples_per_sec", samples_sps, step)

                    # Live progress update
                    if pbar is not None:
                        pbar.update(1)
                        mean_sc = np.mean(recent_scores) if recent_scores else 0.0
                        max_t = int(max(recent_tiles)) if recent_tiles else 0
                        avg_loss = np.mean(recent_losses) if recent_losses else 0.0
                        pbar.set_postfix({
                            "sps": f"{int(env_sps)}",
                            "score": f"{mean_sc:.0f}",
                            "max_tile": f"{max_t}",
                            "loss": f"{avg_loss:.3f}",
                            "eps": f"{self.agent.epsilon:.2f}",
                        })
                        sys.stdout.flush()

                    # Periodic Evaluation
                    if completed_episodes % self.config.eval_freq == 0:
                        eval_metrics = self.evaluate()
                        self.history["eval_scores"].append(eval_metrics["mean_score"])
                        self.history["eval_max_tiles"].append(eval_metrics["max_tile"])

                        if eval_metrics["mean_score"] > self.best_eval_score:
                            self.best_eval_score = eval_metrics["mean_score"]
                            best_path = os.path.join(self.config.checkpoint_dir, "checkpoint_best.pt")
                            self.agent.save_checkpoint(
                                best_path,
                                metadata={
                                    "episode": completed_episodes,
                                    "eval_mean_score": self.best_eval_score,
                                    "eval_max_tile": eval_metrics["max_tile"],
                                },
                            )

                    # Periodic latest checkpoint
                    if completed_episodes % self.config.save_latest_freq == 0:
                        latest_path = os.path.join(self.config.checkpoint_dir, "checkpoint_latest.pt")
                        self.agent.save_checkpoint(
                            latest_path,
                            metadata={"episode": completed_episodes, "steps": self.agent.total_steps},
                        )

                    if completed_episodes >= total_episodes:
                        break

            obs, infos, boards = next_obs, next_infos, next_boards

        if pbar is not None:
            pbar.close()

        return self.history

    def _train_single(self, total_episodes: int) -> Dict[str, Any]:
        """Execute single-environment sequential training loop."""
        for ep in range(1, total_episodes + 1):
            obs, info = self.env.reset()
            done = False
            ep_steps = 0
            ep_losses = []

            while not done:
                action = self.agent.select_action(
                    state=obs,
                    action_mask=info["action_mask"],
                    deterministic=False,
                )

                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                done = terminated or truncated
                self.total_env_steps += 1

                self.buffer.add(
                    state=info["board"],
                    action=action,
                    reward=reward,
                    next_state=next_info["board"],
                    done=done,
                    action_mask=info["action_mask"],
                    next_action_mask=next_info["action_mask"],
                )

                if ep_steps % self.config.train_freq == 0:
                    loss = self.train_step()
                    if loss is not None:
                        ep_losses.append(loss)

                self.agent.decay_epsilon()
                obs, info = next_obs, next_info
                ep_steps += 1

            score = float(info["score"])
            max_tile = float(info["max_tile"])
            self.history["episode_scores"].append(score)
            self.history["episode_max_tiles"].append(max_tile)
            self.history["episode_lengths"].append(float(ep_steps))
            if ep_losses:
                self.history["losses"].append(float(np.mean(ep_losses)))

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

            if ep % self.config.eval_freq == 0:
                eval_metrics = self.evaluate()
                self.history["eval_scores"].append(eval_metrics["mean_score"])
                self.history["eval_max_tiles"].append(eval_metrics["max_tile"])

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

            if ep % self.config.save_latest_freq == 0:
                latest_path = os.path.join(self.config.checkpoint_dir, "checkpoint_latest.pt")
                self.agent.save_checkpoint(
                    latest_path,
                    metadata={"episode": ep, "steps": self.agent.total_steps},
                )

        return self.history

    def train(self, max_episodes: Optional[int] = None) -> Dict[str, Any]:
        """Run complete DDQN training loop in vectorized or single-env mode.

        Args:
            max_episodes: Optional episode override for testing.

        Returns:
            Training history dictionary.
        """
        total_episodes = max_episodes or self.config.total_episodes
        self.warmup()
        self.start_time = time.time()

        if isinstance(self.env, VectorizedGame2048Env):
            history = self._train_vectorized(total_episodes)
        else:
            history = self._train_single(total_episodes)

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        return history
