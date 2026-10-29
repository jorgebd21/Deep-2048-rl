"""CLI script to train N-Tuple Network value tables via afterstate TD self-play."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.train_ntuple import train_ntuple


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for N-Tuple network training."""
    parser = argparse.ArgumentParser(
        description="Train SOTA N-Tuple Network value function via afterstate TD self-play."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50000,
        help="Number of self-play games to simulate (default: 50,000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Initial learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.005,
        help="Final decayed learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor gamma (default: 0.99)",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="log",
        choices=["log", "score"],
        help="Reward formulation: 'log' for sum(log2(merges)) or 'score' for points (default: log)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/ntuple/ntuple_best.npz",
        help="Filepath where best N-Tuple weights .npz is saved (default: checkpoints/ntuple/ntuple_best.npz)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5000,
        help="Log progress and evaluate every N episodes (default: 5,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    net, metrics = train_ntuple(
        episodes=args.episodes,
        alpha_init=args.alpha,
        alpha_min=args.alpha_min,
        gamma=args.gamma,
        reward_type=args.reward_type,
        save_path=args.save_path,
        eval_every=args.eval_every,
        seed=args.seed,
    )
    print(f"Training completed successfully. Best model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
