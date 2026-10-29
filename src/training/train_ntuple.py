"""High-throughput TD(lambda) reinforcement learning self-play engine for N-Tuple Networks.

Trains N-Tuple Network value lookup tables using Bitboard2048 self-play simulations.
Features:
    - Pure in-memory 64-bit Bitboard CPU self-play (>13,000 steps/s).
    - Afterstate Bellman updates:
        delta = r_next + (1 - done) * gamma * V(s'_after) - V(s_after)
    - Symmetric weight updates across 8 Dihedral D4 symmetries.
    - Decaying learning rate schedule: alpha_init -> alpha_min.
    - Automated best model tracking and compressed .npz checkpointing.
"""

from typing import Any, Dict, List, Optional, Tuple
import os
import time
import numpy as np

from src.env.bitboard import Bitboard2048
from src.models.ntuple import NTupleNetwork


def train_ntuple(
    episodes: int = 50000,
    alpha_init: float = 0.05,
    alpha_min: float = 0.005,
    gamma: float = 0.99,
    reward_type: str = "log",
    scale_by_num_tuples: bool = True,
    save_path: str = "checkpoints/ntuple/ntuple_best.npz",
    eval_every: int = 5000,
    seed: int = 42,
    existing_net: Optional[NTupleNetwork] = None,
) -> Tuple[NTupleNetwork, Dict[str, Any]]:
    """Train N-Tuple Network via fast afterstate TD self-play.

    Args:
        episodes: Total number of games to simulate.
        alpha_init: Starting learning rate.
        alpha_min: Final learning rate after decay.
        gamma: Discount factor (e.g. 0.99 or 1.0).
        reward_type: 'log' for sum(log2(merged_tiles)) or 'score' for raw game points.
        scale_by_num_tuples: If True, divides alpha by number of active symmetries (24)
            so total value function update is calibrated to alpha * delta.
        save_path: Filepath where best weights .npz is saved.
        eval_every: Interval of episodes for logging progress and evaluation.
        seed: Random seed for reproducibility.
        existing_net: Optional pretrained NTupleNetwork to continue training from.

    Returns:
        Tuple of (trained NTupleNetwork, metrics dictionary).
    """
    rng = np.random.default_rng(seed)
    net = existing_net if existing_net is not None else NTupleNetwork(gamma=gamma)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    weights_path = os.path.join(os.path.dirname(os.path.abspath(save_path)), "ntuple_weights.npz")

    total_steps = 0
    best_eval_score = -1.0
    scores_history: List[int] = []
    max_tiles_history: List[int] = []

    start_time = time.time()
    last_log_time = start_time

    print(
        f"\nStarting N-Tuple TD Self-Play Training:\n"
        f"  Episodes          : {episodes:,}\n"
        f"  Learning Rate     : {alpha_init} -> {alpha_min} (decayed)\n"
        f"  Discount (gamma)  : {gamma}\n"
        f"  Reward Type       : {reward_type}\n"
        f"  Tuple Symmetries  : 8 D4 per pattern (24 active lookups)\n"
        f"  Save Path         : {save_path}\n"
        f"  Eval Frequency    : Every {eval_every:,} episodes\n",
        flush=True,
    )

    for ep in range(1, episodes + 1):
        # Linear learning rate decay
        progress = (ep - 1) / max(1, episodes - 1)
        current_alpha = alpha_init - (alpha_init - alpha_min) * progress
        net.alpha = current_alpha

        # Initialize game
        bb = Bitboard2048(seed=int(rng.integers(0, 2**31 - 1)))
        board = bb.reset()
        ep_score = 0
        ep_steps = 0

        # Step 0: Find best initial move from s_0
        best_a, best_val, best_nb, best_r, best_sc = -1, -1e9, 0, 0.0, 0
        for a in (0, 1, 2, 3):
            nb, sc, lr, ch = Bitboard2048.move(board, a)
            if not ch:
                continue
            r_val = lr if reward_type == "log" else float(sc)
            v = net.evaluate_bitboard(nb)
            val = r_val + gamma * v
            # Tiny random jitter for breaking initial ties evenly
            val += rng.uniform(-1e-6, 1e-6)
            if val > best_val:
                best_val, best_a, best_nb, best_r, best_sc = val, a, nb, r_val, sc

        if best_a == -1:
            continue

        cur_after = best_nb
        cur_v = (best_val - best_r) / gamma if gamma > 0 else net.evaluate_bitboard(cur_after)
        ep_score += best_sc

        # Episode transition loop
        while True:
            ep_steps += 1
            total_steps += 1

            # Nature spawns a tile into current afterstate
            next_b = bb.spawn_tile(cur_after)

            # Evaluate candidate moves from next_b
            next_best_a = -1
            next_best_val = -1e9
            next_best_nb = 0
            next_best_r = 0.0
            next_best_v = 0.0
            next_best_sc = 0

            for a in (0, 1, 2, 3):
                nb, sc, lr, ch = Bitboard2048.move(next_b, a)
                if not ch:
                    continue
                r_val = lr if reward_type == "log" else float(sc)
                v = net.evaluate_bitboard(nb)
                val = r_val + gamma * v
                val += rng.uniform(-1e-6, 1e-6)
                if val > next_best_val:
                    next_best_val = val
                    next_best_a = a
                    next_best_nb = nb
                    next_best_r = r_val
                    next_best_v = v
                    next_best_sc = sc

            if next_best_a == -1:
                # Terminal state reached: target is 0.0
                td_error = 0.0 - cur_v
                net.update_from_delta(
                    cur_after,
                    td_error,
                    alpha=current_alpha,
                    scale_by_num_tuples=scale_by_num_tuples,
                )
                break

            # Non-terminal Bellman afterstate update
            td_error = next_best_r + gamma * next_best_v - cur_v
            net.update_from_delta(
                cur_after,
                td_error,
                alpha=current_alpha,
                scale_by_num_tuples=scale_by_num_tuples,
            )

            cur_after = next_best_nb
            cur_v = next_best_v
            ep_score += next_best_sc

        scores_history.append(ep_score)
        max_rank = max((cur_after >> (4 * i)) & 0xF for i in range(16))
        max_tiles_history.append(1 << max_rank if max_rank > 0 else 0)

        # Periodic logging and evaluation
        if ep % eval_every == 0 or ep == episodes:
            now = time.time()
            interval_dur = max(0.001, now - last_log_time)
            interval_eps = eval_every if ep % eval_every == 0 else (ep % eval_every)
            recent_scores = scores_history[-interval_eps:]
            recent_tiles = np.array(max_tiles_history[-interval_eps:])

            mean_sc = float(np.mean(recent_scores))
            median_sc = float(np.median(recent_scores))
            max_sc = int(np.max(recent_scores))
            rate_1024 = float(np.mean(recent_tiles >= 1024) * 100.0)
            rate_2048 = float(np.mean(recent_tiles >= 2048) * 100.0)
            rate_4096 = float(np.mean(recent_tiles >= 4096) * 100.0)
            eps_per_sec = interval_eps / interval_dur

            print(
                f"  Ep {ep:6,d}/{episodes:,d} | "
                f"Alpha: {current_alpha:.5f} | "
                f"Mean: {mean_sc:6.0f} | "
                f"Med: {median_sc:6.0f} | "
                f"Peak: {max_sc:6.0f} | "
                f"1024: {rate_1024:4.1f}% | "
                f"2048: {rate_2048:4.1f}% | "
                f"4096: {rate_4096:4.1f}% | "
                f"Speed: {eps_per_sec:5.1f} eps/s",
                flush=True,
            )

            # Save latest weights checkpoint
            net.save(weights_path)

            # Update best model if rolling score improves
            if mean_sc > best_eval_score:
                best_eval_score = mean_sc
                net.save(save_path)
                print(f"    --> Saved new best N-Tuple checkpoint (Mean: {mean_sc:.1f}) to {save_path}", flush=True)

            last_log_time = now

    total_dur = max(0.001, time.time() - start_time)
    print(
        f"\nN-Tuple Training Complete:\n"
        f"  Total Episodes    : {episodes:,}\n"
        f"  Total Game Steps  : {total_steps:,}\n"
        f"  Elapsed Time      : {total_dur:.2f} s ({total_steps / total_dur:.1f} steps/s)\n"
        f"  Final Mean Score  : {np.mean(scores_history[-1000:]):.1f}\n"
        f"  Best Mean Score   : {best_eval_score:.1f}\n"
        f"  Checkpoints       : {save_path}, {weights_path}\n",
        flush=True,
    )

    metrics = {
        "episodes": episodes,
        "total_steps": total_steps,
        "elapsed_seconds": total_dur,
        "best_mean_score": best_eval_score,
        "final_mean_score": float(np.mean(scores_history[-1000:])),
        "scores_history": scores_history,
        "max_tiles_history": max_tiles_history,
    }
    return net, metrics
