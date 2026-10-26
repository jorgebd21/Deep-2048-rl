"""Unit and integration tests for DDQN Agent and Training Pipeline."""

import os
import shutil
import numpy as np
import pytest
import torch

from src.env.wrapper import Game2048Env
from src.memory.replay_buffer import PrioritizedReplayBuffer
from src.training.agent import DDQNAgent
from src.training.trainer import Trainer, TrainingConfig


class TestDDQNAgent:
    """Test suite for DDQNAgent update rules, Bellman targets, and synchronization."""

    def test_target_network_hard_update(self):
        agent = DDQNAgent(in_channels=16, num_actions=4)

        # Perturb policy net parameters
        with torch.no_grad():
            for p in agent.policy_net.parameters():
                p.add_(1.0)

        # Verify weights currently differ
        policy_param = next(agent.policy_net.parameters())
        target_param = next(agent.target_net.parameters())
        assert not torch.allclose(policy_param, target_param)

        # Execute hard update
        agent.update_target_hard()
        for p_pol, p_tgt in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
            torch.testing.assert_close(p_pol, p_tgt)

    def test_target_network_soft_update(self):
        agent = DDQNAgent(in_channels=16, num_actions=4)
        tau = 0.2

        with torch.no_grad():
            for p in agent.policy_net.parameters():
                p.fill_(10.0)
            for p in agent.target_net.parameters():
                p.fill_(0.0)

        agent.update_target_soft(tau=tau)

        # Expected: 0.2 * 10.0 + 0.8 * 0.0 = 2.0
        for p_tgt in agent.target_net.parameters():
            expected = torch.full_like(p_tgt, 2.0)
            torch.testing.assert_close(p_tgt, expected)

    def test_compute_td_loss_shapes(self):
        agent = DDQNAgent(in_channels=16, num_actions=4)
        batch_size = 4

        batch = {
            "states": torch.randn(batch_size, 16, 4, 4),
            "actions": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "rewards": torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32),
            "next_states": torch.randn(batch_size, 16, 4, 4),
            "dones": torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
            "next_action_masks": torch.ones(batch_size, 4, dtype=torch.bool),
            "weights": torch.ones(batch_size, dtype=torch.float32),
        }

        loss, td_errors = agent.compute_td_loss(batch)

        assert loss.dim() == 0  # Scalar loss
        assert loss.requires_grad is True
        assert td_errors.shape == (batch_size,)
        assert (td_errors >= 0).all()

    def test_epsilon_decay_schedule(self):
        start_eps = 1.0
        min_eps = 0.05
        decay_steps = 100
        agent = DDQNAgent(
            epsilon_start=start_eps,
            epsilon_min=min_eps,
            epsilon_decay_steps=decay_steps,
        )

        assert agent.epsilon == 1.0
        epsilons = [agent.decay_epsilon() for _ in range(50)]
        # Monotonically non-increasing
        assert all(x >= y for x, y in zip(epsilons, epsilons[1:]))

        # After decay_steps, must be at min_eps
        for _ in range(60):
            agent.decay_epsilon()
        assert np.isclose(agent.epsilon, min_eps)

    def test_overfit_sanity_check(self):
        """Sanity check: Agent must easily overfit a fixed mini-batch to near-zero loss."""
        agent = DDQNAgent(in_channels=16, num_actions=4, lr=1e-3)
        batch_size = 8

        fixed_batch = {
            "states": torch.randn(batch_size, 16, 4, 4, device=agent.device),
            "actions": torch.randint(0, 4, (batch_size,), device=agent.device),
            "rewards": torch.ones(batch_size, device=agent.device),
            "next_states": torch.randn(batch_size, 16, 4, 4, device=agent.device),
            "dones": torch.zeros(batch_size, device=agent.device),
            "next_action_masks": torch.ones(batch_size, 4, dtype=torch.bool, device=agent.device),
            "weights": torch.ones(batch_size, device=agent.device),
        }

        initial_loss, _ = agent.update(fixed_batch)

        for _ in range(60):
            final_loss, _ = agent.update(fixed_batch)

        assert final_loss < initial_loss
        assert final_loss < 0.1, f"Expected final loss < 0.1, got {final_loss:.4f}"


class TestTrainerIntegration:
    """Integration test suite for the Trainer orchestration loop."""

    @pytest.fixture
    def test_dirs(self):
        ckpt_dir = "tests/test_checkpoints"
        yield ckpt_dir
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_warmup_and_train_loop(self, test_dirs):
        env = Game2048Env(seed=42)
        agent = DDQNAgent(
            in_channels=16,
            num_actions=4,
            epsilon_decay_steps=100,
        )
        config = TrainingConfig(
            total_episodes=2,
            warmup_steps=30,
            batch_size=8,
            buffer_capacity=1000,
            checkpoint_dir=test_dirs,
            eval_freq=1,
            eval_episodes=2,
        )

        trainer = Trainer(env=env, agent=agent, config=config)

        # Test warmup phase
        trainer.warmup()
        assert len(trainer.buffer) >= 30

        # Test evaluation phase
        eval_metrics = trainer.evaluate(num_episodes=2)
        assert "mean_score" in eval_metrics
        assert "max_tile" in eval_metrics
        assert "mean_steps" in eval_metrics
        assert eval_metrics["mean_steps"] > 0

        # Test short training run
        history = trainer.train(max_episodes=2)
        assert len(history["episode_scores"]) == 2
        assert len(history["eval_scores"]) >= 1

        # Checkpoint existence
        best_ckpt = os.path.join(test_dirs, "checkpoint_best.pt")
        assert os.path.exists(best_ckpt)
