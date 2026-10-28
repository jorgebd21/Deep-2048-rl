"""Main entrypoint for Double Dueling DQN training on 2048."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

from src.env.wrapper import Game2048Env
from src.env.vectorized import VectorizedGame2048Env
from src.memory.replay_buffer import PrioritizedReplayBuffer
from src.training.agent import DDQNAgent
from src.training.trainer import Trainer, TrainingConfig
from src.utils.checkpoint import resolve_checkpoint_path
from src.utils.cuda_utils import get_device, get_gpu_telemetry, setup_ampere_tf32


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train Double Dueling Deep Q-Network on 2048 from scratch or resume.",
        allow_abbrev=False,
    )
    parser.add_argument("--episodes", type=int, default=2000, help="Total training episodes")
    parser.add_argument("--num-envs", type=int, default=32, help="Number of concurrent vectorized environments")
    parser.add_argument("--single-env", action="store_true", help="Force single-environment sequential training")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (AdamW)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak soft update rate")
    parser.add_argument("--buffer-capacity", type=int, default=100000, help="PER buffer capacity")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup transitions")
    parser.add_argument("--train-freq", type=int, default=1, help="Vectorized steps between gradient updates")
    parser.add_argument("--eval-freq", type=int, default=50, help="Evaluation frequency in episodes")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes per evaluation run")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory where models are saved")
    parser.add_argument("--resume", "--load-checkpoint", dest="resume", type=str, default=None, help="Path to checkpoint file (.pt) to resume training from")
    parser.add_argument("--resume-epsilon", type=float, default=0.15, help="Exploration epsilon when resuming (default: 0.15)")
    parser.add_argument("--train-afterstate", action="store_true", default=False, help="Enable Expert Iteration with Native Afterstate Value Learning V(s_after)")
    parser.add_argument("--expert-lookahead-depth", type=int, default=2, help="Lookahead depth for expert self-play trajectory generation (default: 2)")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Gradient clipping max norm (default: 10.0 for afterstate, 1.0 for DDQN)")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar")
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
    if args.train_afterstate:
        mode_str = f"Expert Iteration (Afterstate V(s_after), Depth {args.expert_lookahead_depth})"
    elif args.single_env:
        mode_str = "Single Env"
    else:
        mode_str = f"Vectorized ({args.num_envs} Parallel Envs)"

    # Resolve learning rate and max_grad_norm defaults
    lr = args.lr
    if args.train_afterstate and args.lr == 1e-4:
        lr = 5e-5  # Default fine-tuning learning rate

    max_grad_norm = args.max_grad_norm
    if max_grad_norm is None:
        max_grad_norm = 10.0 if args.train_afterstate else 1.0

    # Defensive checkpoint directory resolution
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    elif args.train_afterstate:
        checkpoint_dir = "checkpoints/afterstate"
    else:
        checkpoint_dir = "checkpoints"

    if os.path.isfile(checkpoint_dir):
        checkpoint_dir = os.path.dirname(checkpoint_dir) or "checkpoints"

    pipeline_title = "EXPERT ITERATION AFTERSTATE" if args.train_afterstate else "DDQN"
    print("\n" + "=" * 65, flush=True)
    print(f" DEEP 2048 REINFORCEMENT LEARNING - {pipeline_title} PIPELINE", flush=True)
    print("=" * 65, flush=True)
    print(f"  Target Device       : {device}", flush=True)
    if gpu_info["device_available"]:
        print(f"  GPU Name            : {gpu_info['device_name']}", flush=True)
        print(f"  Compute Capability  : {gpu_info['compute_capability']}")
        print(f"  TF32 Enabled        : {torch.backends.cuda.matmul.allow_tf32}", flush=True)
    print(f"  Execution Mode      : {mode_str}", flush=True)
    print(f"  Total Episodes      : {args.episodes}", flush=True)
    print(f"  Mini-Batch Size     : {args.batch_size}", flush=True)
    print(f"  Learning Rate       : {lr}", flush=True)
    print(f"  Max Grad Norm       : {max_grad_norm}", flush=True)
    print(f"  Discount Factor     : {args.gamma}", flush=True)
    print(f"  PER Capacity        : {args.buffer_capacity}", flush=True)
    print(f"  Checkpoint Dir      : {checkpoint_dir}", flush=True)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.single_env:
        env = Game2048Env()
    else:
        env = VectorizedGame2048Env(num_envs=args.num_envs, seed=args.seed)

    agent = DDQNAgent(
        in_channels=16,
        num_actions=4,
        lr=lr,
        gamma=args.gamma,
        tau=args.tau,
        max_grad_norm=max_grad_norm,
        device=device,
    )

    # Resume from checkpoint if requested
    if args.resume is not None:
        resume_path = resolve_checkpoint_path(args.resume)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")

        meta = agent.load_checkpoint(resume_path, resume_epsilon=args.resume_epsilon)
        print(f"  Resuming From       : {resume_path}", flush=True)
        print(f"  Resumed Epsilon     : {agent.epsilon:.2f} (decaying to {agent.epsilon_min})", flush=True)
        if meta:
            prev_ep = meta.get("episode", "N/A")
            prev_score = meta.get("eval_mean_score", "N/A")
            prev_tile = meta.get("eval_max_tile", "N/A")
            print(f"  Checkpoint Metadata : Episode {prev_ep} | Score: {prev_score} | Tile: {prev_tile}", flush=True)

    print("=" * 65 + "\n", flush=True)

    buffer = PrioritizedReplayBuffer(capacity=args.buffer_capacity)

    config = TrainingConfig(
        total_episodes=args.episodes,
        warmup_steps=args.warmup_steps if args.resume is None else min(500, args.warmup_steps),
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        num_envs=args.num_envs,
        use_vectorized=not args.single_env,
        train_freq=args.train_freq,
        tau=args.tau,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        checkpoint_dir=checkpoint_dir,
        log_dir=args.log_dir,
        enable_tensorboard=not args.no_tensorboard,
        use_tqdm=not args.no_tqdm,
        seed=args.seed,
        train_afterstate=args.train_afterstate,
        expert_lookahead_depth=args.expert_lookahead_depth,
    )

    trainer = Trainer(env=env, agent=agent, buffer=buffer, config=config)

    start_time = time.time()
    history = trainer.train()
    elapsed = time.time() - start_time

    print("\n" + "=" * 65, flush=True)
    print(" TRAINING SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"  Total Time Elapsed  : {elapsed / 60.0:.2f} minutes", flush=True)
    print(f"  Total Steps Taken   : {trainer.total_env_steps}", flush=True)
    print(f"  Peak Training Score : {max(history['episode_scores'], default=0.0):.1f}", flush=True)
    print(f"  Peak Eval Score     : {trainer.best_eval_score:.1f}", flush=True)
    print(f"  Peak Tile Observed  : {int(max(history['episode_max_tiles'], default=0))}", flush=True)
    print("=" * 65 + "\n", flush=True)


if __name__ == "__main__":
    main()
