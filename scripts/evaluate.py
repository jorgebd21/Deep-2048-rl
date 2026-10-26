"""Monte Carlo evaluation runner for 2048 DRL agents."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.env.wrapper import Game2048Env
from src.models.dueling_dqn import Dueling2048Net
from src.utils.cuda_utils import get_device, setup_ampere_tf32


def parse_args() -> argparse.Namespace:
    """Parse evaluation command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo evaluation on 2048 Deep Q-Network."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--save-json", type=str, default=None, help="Save evaluation metrics to JSON file")
    return parser.parse_args()


def evaluate_agent(
    model: Dueling2048Net,
    episodes: int = 100,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute deterministic evaluation runs and compute aggregate performance statistics."""
    dev = device or torch.device("cpu")
    model.eval()
    env = Game2048Env()

    scores: List[float] = []
    max_tiles: List[int] = []
    steps_list: List[int] = []

    print(f"\nRunning {episodes} evaluation episodes...")
    start_time = time.time()

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        done = False
        step = 0

        while not done:
            mask = info["action_mask"]
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                break

            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
            mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(dev)

            with torch.no_grad():
                q_vals = model(obs_t, action_mask=mask_t)
                action = int(q_vals.argmax(dim=-1).item())

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        scores.append(float(info["score"]))
        max_tiles.append(int(info["max_tile"]))
        steps_list.append(step)

        if ep % 20 == 0 or ep == episodes:
            print(f"  Completed {ep:3d}/{episodes} episodes | "
                  f"Mean Score: {np.mean(scores):.1f} | "
                  f"Max Tile: {max(max_tiles)}")

    elapsed = time.time() - start_time
    tiles_arr = np.array(max_tiles)
    scores_arr = np.array(scores)

    # Unique tile counts
    unique_tiles, counts = np.unique(tiles_arr, return_counts=True)
    tile_dist = {int(t): int(c) for t, c in zip(unique_tiles, counts)}

    results = {
        "num_episodes": episodes,
        "elapsed_seconds": round(elapsed, 2),
        "mean_score": float(np.mean(scores_arr)),
        "std_score": float(np.std(scores_arr)),
        "median_score": float(np.median(scores_arr)),
        "min_score": float(np.min(scores_arr)),
        "max_score": float(np.max(scores_arr)),
        "mean_steps": float(np.mean(steps_list)),
        "reach_256_rate": float(np.mean(tiles_arr >= 256)),
        "reach_512_rate": float(np.mean(tiles_arr >= 512)),
        "reach_1024_rate": float(np.mean(tiles_arr >= 1024)),
        "reach_2048_rate": float(np.mean(tiles_arr >= 2048)),
        "reach_4096_rate": float(np.mean(tiles_arr >= 4096)),
        "tile_distribution": tile_dist,
    }

    return results


def resolve_checkpoint_path(path: Optional[str]) -> Optional[str]:
    """Resolve checkpoint file path with alias and directory fallbacks.

    Args:
        path: User-provided checkpoint path or filename.

    Returns:
        Resolved absolute or relative path, or None if path is None.

    Raises:
        FileNotFoundError: If a non-None path cannot be resolved to an existing file.
    """
    if path is None:
        return None

    candidates = [
        path,
        os.path.join("checkpoints", path),
        os.path.join("checkpoints", os.path.basename(path)),
    ]

    # Handle common alias variations (best_model <-> checkpoint_best)
    base = os.path.basename(path)
    if "best_model" in base:
        candidates.append(path.replace("best_model", "checkpoint_best"))
        candidates.append(os.path.join("checkpoints", "checkpoint_best.pt"))
    elif "checkpoint_best" in base:
        candidates.append(path.replace("checkpoint_best", "best_model"))
        candidates.append(os.path.join("checkpoints", "best_model.pt"))

    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return c

    available = os.listdir("checkpoints") if os.path.exists("checkpoints") else []
    raise FileNotFoundError(
        f"Checkpoint file '{path}' not found. Available files in 'checkpoints/': {available}"
    )


def main() -> None:
    """Run CLI evaluation."""
    args = parse_args()
    setup_ampere_tf32()

    if args.device == "auto":
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)

    model = Dueling2048Net(in_channels=16, num_actions=4).to(device)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("policy_net_state_dict", checkpoint.get("model_state_dict", checkpoint))
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded weights successfully from {checkpoint_path}")
    else:
        model.eval()
        print("No checkpoint specified. Evaluating untrained baseline model.")

    results = evaluate_agent(
        model=model,
        episodes=args.episodes,
        device=device,
        seed=args.seed,
    )

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print(" MONTE CARLO EVALUATION REPORT")
    print("=" * 65)
    print(f"  Episodes Evaluated  : {results['num_episodes']}")
    print(f"  Elapsed Time        : {results['elapsed_seconds']:.2f} s")
    print(f"  Mean Score          : {results['mean_score']:.1f} +/- {results['std_score']:.1f}")
    print(f"  Median Score        : {results['median_score']:.1f}")
    print(f"  Score Range         : [{results['min_score']:.0f}, {results['max_score']:.0f}]")
    print(f"  Mean Steps          : {results['mean_steps']:.1f}")
    print("  " + "-" * 45)
    print("  Milestone Tile Reach Rates:")
    print(f"    Reach 256+        : {results['reach_256_rate'] * 100:6.1f}%")
    print(f"    Reach 512+        : {results['reach_512_rate'] * 100:6.1f}%")
    print(f"    Reach 1024+       : {results['reach_1024_rate'] * 100:6.1f}%")
    print(f"    Reach 2048+       : {results['reach_2048_rate'] * 100:6.1f}%")
    print("  " + "-" * 45)
    print("  Max Tile Frequency Distribution:")
    for tile, count in sorted(results["tile_distribution"].items()):
        pct = (count / results["num_episodes"]) * 100.0
        print(f"    Tile {tile:4d}         : {count:3d} episodes ({pct:5.1f}%)")
    print("=" * 65 + "\n")

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to: {args.save_json}")


if __name__ == "__main__":
    main()
