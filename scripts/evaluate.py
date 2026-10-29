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

from src.agent.search import ExpectimaxSearcher
from src.env.wrapper import Game2048Env
from src.models.dueling_dqn import Dueling2048Net
from src.models.ntuple import NTupleNetwork
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
    parser.add_argument("--lookahead-depth", type=int, default=0, help="Expectimax lookahead search depth (0 for greedy, 2 for 2-ply, 3 for 3-ply, 4 for 4-ply)")
    parser.add_argument("--adaptive-depth", action="store_true", default=False, help="Dynamically adjust depth (2-ply early, 3-ply mid, 4-ply endgame)")
    parser.add_argument("--apex-mode", action="store_true", default=False, help="Enable 5-ply and 6-ply depth in apex endgames")
    parser.add_argument("--use-d4-symmetry", action="store_true", default=False, help="Enable 8-fold dihedral D4 symmetry invariance in search")
    parser.add_argument("--use-snake-heuristic", action="store_true", default=True, help="Enable topological snake monotonicity and corner shield (default: True)")
    parser.add_argument("--no-snake-heuristic", action="store_false", dest="use_snake_heuristic", help="Disable topological snake monotonicity and corner shield")
    parser.add_argument("--alpha-snake", type=float, default=0.5, help="Weight for snake monotonicity score")
    parser.add_argument("--beta-corner", type=float, default=1.0, help="Weight for corner dislodgement penalty")
    parser.add_argument("--use-afterstate-net", action="store_true", default=False, help="Use scalar afterstate value head V(s_after) for leaf evaluation")
    parser.add_argument("--use-ntuple", action="store_true", default=False, help="Enable SOTA N-Tuple Network value evaluation")
    parser.add_argument("--ntuple-path", type=str, default=None, help="Path to pretrained N-Tuple .npz weights")
    parser.add_argument("--use-smoothness", action="store_true", default=False, help="Enable bitwise Manhattan smoothness and trapped-tile shield")
    parser.add_argument("--w-smooth", type=float, default=0.1, help="Weight multiplier for Manhattan smoothness")
    parser.add_argument("--w-trap", type=float, default=5.0, help="Weight multiplier for trapped tile penalty")
    parser.add_argument("--use-policy-pruning", action="store_true", default=False, help="Enable policy-guided action pruning at shallow plies")
    parser.add_argument("--policy-prune-tau", type=float, default=0.03, help="Cutoff threshold for action policy prior pruning")
    parser.add_argument("--max-empty-samples", type=int, default=None, help="Max empty cells to sample per chance node (default: evaluate all empty cells)")
    parser.add_argument("--save-json", type=str, default=None, help="Save evaluation metrics to JSON file")
    return parser.parse_args()


def evaluate_agent(
    model: Optional[Dueling2048Net] = None,
    episodes: int = 100,
    device: Optional[torch.device] = None,
    seed: int = 42,
    lookahead_depth: int = 0,
    max_empty_samples: Optional[int] = None,
    use_d4_symmetry: bool = False,
    adaptive_depth: bool = False,
    apex_mode: bool = False,
    use_snake_heuristic: bool = True,
    alpha_snake: float = 0.5,
    beta_corner: float = 1.0,
    use_afterstate_net: bool = False,
    use_ntuple: bool = False,
    ntuple_net: Optional[NTupleNetwork] = None,
    use_smoothness: bool = False,
    w_smooth: float = 0.1,
    w_trap: float = 5.0,
    use_policy_pruning: bool = False,
    policy_prune_tau: float = 0.03,
) -> Dict[str, Any]:
    """Execute deterministic evaluation runs and compute aggregate performance statistics."""
    dev = device or torch.device("cpu")
    if model is not None:
        model.eval()
    env = Game2048Env()

    searcher: Optional[ExpectimaxSearcher] = None
    if (
        lookahead_depth > 0
        or adaptive_depth
        or apex_mode
        or use_d4_symmetry
        or use_afterstate_net
        or use_ntuple
        or use_smoothness
        or use_policy_pruning
    ):
        searcher = ExpectimaxSearcher(
            model=model,
            device=dev,
            depth=lookahead_depth,
            max_empty_samples=max_empty_samples,
            use_d4_symmetry=use_d4_symmetry,
            adaptive_depth=adaptive_depth,
            apex_mode=apex_mode,
            use_snake_heuristic=use_snake_heuristic,
            alpha_snake=alpha_snake,
            beta_corner=beta_corner,
            use_afterstate_net=use_afterstate_net,
            use_ntuple=use_ntuple,
            ntuple_net=ntuple_net,
            use_smoothness=use_smoothness,
            w_smooth=w_smooth,
            w_trap=w_trap,
            use_policy_pruning=use_policy_pruning,
            policy_prune_tau=policy_prune_tau,
        )

    scores: List[float] = []
    max_tiles: List[int] = []
    steps_list: List[int] = []

    d4_str = " + D4 Symmetry" if use_d4_symmetry else ""
    snake_str = " + Snake Shield" if use_snake_heuristic else ""
    ntuple_str = " + N-Tuple" if use_ntuple else ""
    smooth_str = " + Smoothness Shield" if use_smoothness else ""
    prune_str = " + Policy Pruning" if use_policy_pruning else ""

    if apex_mode:
        search_mode_str = f"Apex Adaptive Depth (2..6 ply) Expectimax{d4_str}{snake_str}{ntuple_str}{smooth_str}{prune_str}"
    elif adaptive_depth:
        search_mode_str = f"Adaptive Depth Expectimax{d4_str}{snake_str}{ntuple_str}{smooth_str}{prune_str}"
    elif lookahead_depth > 0:
        search_mode_str = f"{lookahead_depth}-ply Expectimax{d4_str}{snake_str}{ntuple_str}{smooth_str}{prune_str}"
    else:
        search_mode_str = f"pure greedy (0-ply{d4_str}{ntuple_str}{smooth_str})"
    print(f"\nRunning {episodes} evaluation episodes with {search_mode_str}...", flush=True)
    start_time = time.time()

    for ep in range(1, episodes + 1):
        ep_start = time.time()
        obs, info = env.reset(seed=seed + ep)
        done = False
        step = 0

        while not done:
            mask = info["action_mask"]
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                break

            if searcher is not None:
                action = searcher.select_action(env.core.board, action_mask=mask)
            else:
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

        ep_duration = max(0.001, time.time() - ep_start)
        reach_2048_running = np.mean(np.array(max_tiles) >= 2048) * 100.0
        print(
            f"  Episode {ep:2d}/{episodes} | "
            f"Score: {scores[-1]:6.0f} | "
            f"Max Tile: {max_tiles[-1]:4d} | "
            f"Steps: {step:4d} ({step/ep_duration:4.1f} steps/s) | "
            f"Mean Score: {np.mean(scores):6.1f} | "
            f"2048 Rate: {reach_2048_running:4.1f}%",
            flush=True,
        )

    elapsed = time.time() - start_time
    tiles_arr = np.array(max_tiles)
    scores_arr = np.array(scores)

    # Unique tile counts
    unique_tiles, counts = np.unique(tiles_arr, return_counts=True)
    tile_dist = {int(t): int(c) for t, c in zip(unique_tiles, counts)}

    results = {
        "num_episodes": episodes,
        "lookahead_depth": lookahead_depth,
        "adaptive_depth": adaptive_depth,
        "apex_mode": apex_mode,
        "use_d4_symmetry": use_d4_symmetry,
        "use_snake_heuristic": use_snake_heuristic,
        "alpha_snake": alpha_snake,
        "beta_corner": beta_corner,
        "use_ntuple": use_ntuple,
        "use_smoothness": use_smoothness,
        "w_smooth": w_smooth,
        "w_trap": w_trap,
        "use_policy_pruning": use_policy_pruning,
        "policy_prune_tau": policy_prune_tau,
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
        "reach_8192_rate": float(np.mean(tiles_arr >= 8192)),
        "reach_16384_rate": float(np.mean(tiles_arr >= 16384)),
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

    use_afterstate_net = args.use_afterstate_net

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("policy_net_state_dict", checkpoint.get("model_state_dict", checkpoint))
            metadata = checkpoint.get("metadata", {})
            if metadata.get("afterstate", False) or metadata.get("model_mode") == "afterstate":
                use_afterstate_net = True
                print("  Model Mode          : Native Afterstate Value Network V(s_after)")
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded weights successfully from {checkpoint_path}")
    else:
        model.eval()
        print("No checkpoint specified. Evaluating untrained baseline model.")

    ntuple_net = None
    if args.use_ntuple:
        ntuple_net = NTupleNetwork()
        if args.ntuple_path and os.path.exists(args.ntuple_path):
            ntuple_net.load(args.ntuple_path)
            print(f"Loaded N-Tuple weights successfully from {args.ntuple_path}")
        else:
            print("Initialized N-Tuple Network with canonical D4 configurations.")

    results = evaluate_agent(
        model=model,
        episodes=args.episodes,
        device=device,
        seed=args.seed,
        lookahead_depth=args.lookahead_depth,
        max_empty_samples=args.max_empty_samples,
        use_d4_symmetry=args.use_d4_symmetry,
        adaptive_depth=args.adaptive_depth,
        apex_mode=args.apex_mode,
        use_snake_heuristic=args.use_snake_heuristic,
        alpha_snake=args.alpha_snake,
        beta_corner=args.beta_corner,
        use_afterstate_net=use_afterstate_net,
        use_ntuple=args.use_ntuple,
        ntuple_net=ntuple_net,
        use_smoothness=args.use_smoothness,
        w_smooth=args.w_smooth,
        w_trap=args.w_trap,
        use_policy_pruning=args.use_policy_pruning,
        policy_prune_tau=args.policy_prune_tau,
    )

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    d4_label = " + D4 Symmetry" if args.use_d4_symmetry else ""
    snake_label = " + Snake Shield" if args.use_snake_heuristic else ""
    ntuple_label = " + N-Tuple" if args.use_ntuple else ""
    smooth_label = " + Smoothness Shield" if args.use_smoothness else ""
    prune_label = " + Policy Pruning" if args.use_policy_pruning else ""

    if args.apex_mode:
        search_label = f"Apex Adaptive Depth (2..6 ply) Expectimax{d4_label}{snake_label}{ntuple_label}{smooth_label}{prune_label}"
    elif args.adaptive_depth:
        search_label = f"Adaptive Depth Expectimax{d4_label}{snake_label}{ntuple_label}{smooth_label}{prune_label}"
    elif args.lookahead_depth > 0:
        search_label = f"{args.lookahead_depth}-ply Expectimax{d4_label}{snake_label}{ntuple_label}{smooth_label}{prune_label}"
    else:
        search_label = f"Pure Greedy (0-ply{d4_label}{ntuple_label}{smooth_label})"
    print("\n" + "=" * 65)
    print(f" MONTE CARLO EVALUATION REPORT ({search_label})")
    print("=" * 65)
    print(f"  Episodes Evaluated  : {results['num_episodes']}")
    if results['apex_mode']:
        depth_str = "Apex Adaptive (2..6 ply)"
    elif results['adaptive_depth']:
        depth_str = "Adaptive (2..4 ply)"
    else:
        depth_str = str(results['lookahead_depth'])
    print(f"  Lookahead Depth     : {depth_str}")
    print(f"  D4 Symmetrization   : {'Enabled' if results['use_d4_symmetry'] else 'Disabled'}")
    print(f"  Snake Shield        : {'Enabled (alpha={:.2f}, beta={:.2f})'.format(results['alpha_snake'], results['beta_corner']) if results['use_snake_heuristic'] else 'Disabled'}")
    print(f"  N-Tuple Evaluation  : {'Enabled' if results['use_ntuple'] else 'Disabled'}")
    print(f"  Smoothness Shield   : {'Enabled (w_smooth={:.2f}, w_trap={:.2f})'.format(results['w_smooth'], results['w_trap']) if results['use_smoothness'] else 'Disabled'}")
    print(f"  Policy Pruning      : {'Enabled (tau={:.3f})'.format(results['policy_prune_tau']) if results['use_policy_pruning'] else 'Disabled'}")
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
    print(f"    Reach 4096+       : {results['reach_4096_rate'] * 100:6.1f}%")
    print(f"    Reach 8192+       : {results['reach_8192_rate'] * 100:6.1f}%")
    print(f"    Reach 16384+      : {results.get('reach_16384_rate', 0.0) * 100:6.1f}%")
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
