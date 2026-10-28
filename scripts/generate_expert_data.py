"""CLI script to generate high-tier expert afterstate trajectories using Bitboard Expectimax."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.agent.search import ExpectimaxSearcher
from src.models.dueling_dqn import Dueling2048Net
from src.training.expert_iteration import (
    collect_afterstate_trajectory,
    save_expert_dataset,
)
from src.utils.checkpoint import resolve_checkpoint_path
from src.utils.cuda_utils import get_device, setup_ampere_tf32


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for expert data generation."""
    parser = argparse.ArgumentParser(
        description="Generate expert 2048 afterstate trajectories via Lookahead Search."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_best.pt",
        help="Path to trained base model checkpoint (.pt)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of full expert games to simulate",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=2,
        help="Expectimax search depth for expert play (default: 2)",
    )
    parser.add_argument(
        "--use-d4-symmetry",
        action="store_true",
        default=False,
        help="Enable Dihedral D4 symmetry ensembling in search",
    )
    parser.add_argument(
        "--apex-mode",
        action="store_true",
        default=False,
        help="Enable 5/6-ply apex search depth",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/expert_afterstates.npz",
        help="Filepath to save compressed afterstate dataset (.npz)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed base",
    )
    return parser.parse_args()


def main() -> None:
    """Run expert self-play trajectory collection."""
    args = parse_args()
    setup_ampere_tf32()

    device = get_device(prefer_gpu=True) if args.device == "auto" else torch.device(args.device)
    model = Dueling2048Net(in_channels=16, num_actions=4).to(device)

    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("policy_net_state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict)
        print(f"Loaded base model weights from {ckpt_path}", flush=True)
    else:
        print(f"Warning: No valid checkpoint found at {args.checkpoint}. Using untrained model.", flush=True)

    model.eval()
    searcher = ExpectimaxSearcher(
        model=model,
        device=device,
        depth=args.lookahead_depth,
        use_d4_symmetry=args.use_d4_symmetry,
        apex_mode=args.apex_mode,
        use_snake_heuristic=True,
    )

    print(
        f"\nGenerating {args.episodes} expert self-play games with "
        f"Depth {args.lookahead_depth} lookahead on {device}...",
        flush=True,
    )

    all_transitions = []
    scores = []
    max_tiles = []
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        ep_t0 = time.time()
        traj = collect_afterstate_trajectory(
            searcher=searcher,
            depth=args.lookahead_depth,
            use_d4=args.use_d4_symmetry,
            apex_mode=args.apex_mode,
            seed=args.seed + ep,
        )
        ep_elapsed = time.time() - ep_t0

        all_transitions.extend(traj["transitions"])
        scores.append(traj["score"])
        max_tiles.append(traj["max_tile"])

        sps = traj["steps"] / max(1e-5, ep_elapsed)
        print(
            f"  Episode {ep:3d}/{args.episodes} | Score: {traj['score']:6d} | "
            f"Max Tile: {traj['max_tile']:5d} | Steps: {traj['steps']:4d} ({sps:.1f} steps/s) | "
            f"Trans: {len(traj['transitions']):4d}",
            flush=True,
        )

    out_path = save_expert_dataset(all_transitions, args.output)
    total_elapsed = time.time() - t0

    print("\n" + "=" * 65)
    print(" EXPERT TRAJECTORY GENERATION COMPLETE")
    print("=" * 65)
    print(f"  Total Games Simulated : {args.episodes}")
    print(f"  Total Transitions     : {len(all_transitions):,}")
    print(f"  Mean Score            : {np.mean(scores):.1f}")
    print(f"  Max Tile Reached      : {max(max_tiles)}")
    print(f"  Saved Dataset To      : {out_path}")
    print(f"  Total Elapsed Time    : {total_elapsed:.2f} s")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
