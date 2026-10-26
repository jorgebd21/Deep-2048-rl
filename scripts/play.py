"""Interactive CLI playback visualizer for trained 2048 DRL models."""

import argparse
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch

from src.env.wrapper import Game2048Env
from src.models.dueling_dqn import Dueling2048Net
from src.utils.cuda_utils import get_device, setup_ampere_tf32
from src.utils.visualizer import clear_screen, render_board_ansi, render_hud


def parse_args() -> argparse.Namespace:
    """Parse command line playback arguments."""
    parser = argparse.ArgumentParser(
        description="Run real-time ANSI terminal visualizer for 2048 RL agent."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps in seconds")
    parser.add_argument("--manual-step", action="store_true", help="Press Enter to execute next move")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None, help="Seed for game initialization")
    parser.add_argument("--lookahead-depth", type=int, default=0, help="Expectimax lookahead search depth (0 for greedy, 2 for 2-ply, 3 for 3-ply, 4 for 4-ply)")
    parser.add_argument("--use-d4-symmetry", action="store_true", default=False, help="Enable 8-fold dihedral D4 symmetry invariance in search")
    return parser.parse_args()


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


def play(args: argparse.Namespace) -> None:
    """Run interactive playback loop."""
    setup_ampere_tf32()
    if args.device == "auto":
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)

    env = Game2048Env()
    obs, info = env.reset(seed=args.seed)

    # Initialize model
    model = Dueling2048Net(in_channels=16, num_actions=4).to(device)
    model_loaded = False

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("policy_net_state_dict", checkpoint.get("model_state_dict", checkpoint))
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        model_loaded = True
        print(f"Loaded weights successfully from {checkpoint_path}")
    else:
        model.eval()
        print("No checkpoint specified. Running with untrained baseline policy.")

    done = False
    step = 0

    while not done:
        clear_screen()
        print("\n" + "=" * 52)
        d4_str = " + D4" if args.use_d4_symmetry else ""
        mode_str = f"{args.lookahead_depth}-PLY EXPECTIMAX{d4_str}" if args.lookahead_depth > 0 else f"GREEDY{d4_str}"
        status = f"TRAINED ({mode_str})" if model_loaded else f"UNTRAINED ({mode_str})"
        print(f"  DEEP 2048 RL LIVE AGENT VIEWER ({status})")
        print("=" * 52 + "\n")

        # Board rendering
        print(render_board_ansi(info["board"]))
        print()

        # Compute Q-values with legal action mask
        mask = info["action_mask"]
        valid_actions = np.flatnonzero(mask)

        if len(valid_actions) == 0:
            break

        if args.lookahead_depth > 0 or args.use_d4_symmetry:
            from src.agent.search import ExpectimaxSearcher
            searcher = ExpectimaxSearcher(
                model=model,
                device=device,
                depth=args.lookahead_depth,
                use_d4_symmetry=args.use_d4_symmetry,
            )
            selected_action, q_dict = searcher.search(
                info["board"],
                action_mask=mask,
                depth=args.lookahead_depth,
                use_d4_symmetry=args.use_d4_symmetry,
            )
            q_vals_raw = np.array([q_dict[a] for a in range(4)], dtype=np.float32)
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).bool().unsqueeze(0).to(device)

            with torch.no_grad():
                q_vals_masked = model(obs_t, action_mask=mask_t).squeeze(0).cpu().numpy()
                q_vals_raw = model(obs_t).squeeze(0).cpu().numpy()

            # Greedy legal action
            selected_action = int(np.argmax(q_vals_masked))

        # HUD rendering
        print(
            render_hud(
                score=info["score"],
                max_tile=info["max_tile"],
                step=step,
                action_mask=mask,
                q_values=q_vals_raw,
                selected_action=selected_action,
            )
        )

        if args.manual_step:
            try:
                input("\n  [Press Enter to step, Ctrl+C to quit]...")
            except (KeyboardInterrupt, EOFError):
                print("\nPlayback interrupted by user.")
                return
        else:
            time.sleep(args.delay)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(selected_action)
        done = terminated or truncated
        step += 1

    # Terminal state display
    clear_screen()
    print("\n" + "=" * 52)
    print("  GAME OVER - FINAL RESULTS")
    print("=" * 52 + "\n")
    print(render_board_ansi(info["board"]))
    print()
    print(f"  Final Score      : {info['score']}")
    print(f"  Max Tile Reached : {info['max_tile']}")
    print(f"  Total Steps      : {step}")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    play(parse_args())
