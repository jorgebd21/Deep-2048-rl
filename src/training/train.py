"""Main entrypoint for Double Dueling DQN training on 2048."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

from src.env.wrapper import Game2048Env
from src.memory.replay_buffer import PrioritizedReplayBuffer
from src.training.agent import DDQNAgent
from src.training.trainer import Trainer, TrainingConfig
from src.utils.cuda_utils import get_device, get_gpu_telemetry, setup_ampere_tf32


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train Double Dueling Deep Q-Network on 2048 from scratch."
    )
    parser.add_argument("--episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (AdamW)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak soft update rate")
    parser.add_argument("--buffer-capacity", type=int, default=100000, help="PER buffer capacity")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup transitions")
    parser.add_argument("--eval-freq", type=int, default=50, help="Evaluation frequency in episodes")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes per evaluation run")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Initialize environment, agent, buffer, and launch training orchestrator."""
    args = parse_args()

    # Device selection and Ampere optimization
    setup_ampere_tf32()
    if args.device == "auto":
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)

    gpu_info = get_gpu_telemetry()

    print("\n" + "=" * 65)
    print(" DEEP 2048 REINFORCEMENT LEARNING - DDQN TRAINING PIPELINE")
    print("=" * 65)
    print(f"  Target Device       : {device}")
    if gpu_info["device_available"]:
        print(f"  GPU Name            : {gpu_info['device_name']}")
        print(f"  Compute Capability  : {gpu_info['compute_capability']}")
        print(f"  TF32 Enabled        : {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  Total Episodes      : {args.episodes}")
    print(f"  Mini-Batch Size     : {args.batch_size}")
    print(f"  Learning Rate       : {args.lr}")
    print(f"  Discount Factor     : {args.gamma}")
    print(f"  PER Capacity        : {args.buffer_capacity}")
    print(f"  Checkpoint Dir      : {args.checkpoint_dir}")
    print("=" * 65 + "\n")

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = Game2048Env()
    agent = DDQNAgent(
        in_channels=16,
        num_actions=4,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
    )
    buffer = PrioritizedReplayBuffer(capacity=args.buffer_capacity)

    config = TrainingConfig(
        total_episodes=args.episodes,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        tau=args.tau,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        enable_tensorboard=not args.no_tensorboard,
        seed=args.seed,
    )

    trainer = Trainer(env=env, agent=agent, buffer=buffer, config=config)

    start_time = time.time()
    history = trainer.train()
    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print(" TRAINING SUMMARY")
    print("=" * 65)
    print(f"  Total Time Elapsed  : {elapsed / 60.0:.2f} minutes")
    print(f"  Total Steps Taken   : {agent.total_steps}")
    print(f"  Peak Training Score : {max(history['episode_scores'], default=0.0):.1f}")
    print(f"  Peak Eval Score     : {trainer.best_eval_score:.1f}")
    print(f"  Peak Tile Observed  : {int(max(history['episode_max_tiles'], default=0))}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
