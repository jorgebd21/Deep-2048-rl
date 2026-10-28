"""Expert Iteration trajectory collector and afterstate dataset utilities.

Generates high-tier expert games using Bitboard Expectimax lookahead search
and populates the Prioritized Replay Buffer with afterstate transitions:
    (s_after, r_next, s'_after, done)
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from src.agent.search import ExpectimaxSearcher
from src.env.bitboard import Bitboard2048
from src.memory.replay_buffer import PrioritizedReplayBuffer


def collect_afterstate_trajectory(
    searcher: ExpectimaxSearcher,
    depth: int = 2,
    use_d4: bool = False,
    apex_mode: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a single expert game on Bitboard engine and collect afterstate transitions.

    Args:
        searcher: ExpectimaxSearcher instance for action selection.
        depth: Search lookahead depth (e.g. 2 or 3).
        use_d4: Whether to enable Dihedral D4 symmetry ensembling during search.
        apex_mode: Whether to enable apex 5-ply / 6-ply depth in tight endgames.
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary containing:
            - "transitions": List of (s_after, r_next, next_s_after, done)
            - "score": Final merge score achieved
            - "max_tile": Highest tile value achieved
            - "steps": Total number of moves played
    """
    game = Bitboard2048()
    bb = game.reset(seed=seed)

    transitions: List[Tuple[np.ndarray, float, np.ndarray, bool]] = []
    total_score = 0
    steps = 0

    # Step 0: Initial move from starting board
    mask = Bitboard2048.get_action_mask(bb)
    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        return {
            "transitions": transitions,
            "score": 0,
            "max_tile": int(np.max(Bitboard2048.to_array(bb))),
            "steps": 0,
        }

    act = searcher.select_action(
        bb,
        action_mask=mask,
        depth=depth,
        use_d4_symmetry=use_d4,
        apex_mode=apex_mode,
    )
    curr_afterstate, merge_score, _, changed = Bitboard2048.move(bb, act)
    total_score += merge_score
    steps += 1
    curr_after_exps = Bitboard2048.to_exponents_array(curr_afterstate)

    while True:
        # Nature spawns a random tile (2 with 90%, 4 with 10%)
        next_board = game.spawn_tile(curr_afterstate)

        # Check if spawned board is terminal
        mask = Bitboard2048.get_action_mask(next_board)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            # Terminal transition: player has no legal moves from next_board
            terminal_target = np.zeros((4, 4), dtype=np.uint8)
            transitions.append((curr_after_exps, 0.0, terminal_target, True))
            break

        # Expert selects next action from spawned board
        next_act = searcher.select_action(
            next_board,
            action_mask=mask,
            depth=depth,
            use_d4_symmetry=use_d4,
            apex_mode=apex_mode,
        )
        next_afterstate, merge_score, next_lr, changed = Bitboard2048.move(
            next_board, next_act
        )
        total_score += merge_score
        steps += 1
        next_after_exps = Bitboard2048.to_exponents_array(next_afterstate)

        # Record valid non-terminal afterstate transition
        # Target value for curr_afterstate is next_lr + gamma * V(next_after_exps)
        transitions.append(
            (curr_after_exps, float(next_lr), next_after_exps, False)
        )

        curr_afterstate = next_afterstate
        curr_after_exps = next_after_exps

    max_tile = int(np.max(Bitboard2048.to_array(curr_afterstate)))
    return {
        "transitions": transitions,
        "score": total_score,
        "max_tile": max_tile,
        "steps": steps,
    }


def populate_buffer_with_expert_trajectories(
    buffer: PrioritizedReplayBuffer,
    searcher: ExpectimaxSearcher,
    num_episodes: int = 20,
    depth: int = 2,
    use_d4: bool = False,
    apex_mode: bool = False,
    min_max_tile: int = 0,
    base_seed: int = 42,
    callback: Optional[Callable[[int, int, float, int], None]] = None,
) -> Dict[str, Any]:
    """Generate multiple expert games and insert afterstate transitions into the buffer.

    Args:
        buffer: Target PrioritizedReplayBuffer instance.
        searcher: ExpectimaxSearcher used for expert action selection.
        num_episodes: Number of complete self-play games to simulate.
        depth: Lookahead search depth (default: 2).
        use_d4: Whether to use 8-fold D4 symmetry in lookahead.
        apex_mode: Whether to enable 5/6-ply apex search.
        min_max_tile: Optional filter to only store episodes reaching this max tile.
        base_seed: Initial random seed.
        callback: Optional progress callback `fn(ep_idx, score, max_tile, num_transitions)`.

    Returns:
        Summary dictionary with collection metrics.
    """
    total_transitions_added = 0
    scores: List[float] = []
    max_tiles: List[int] = []
    t0 = time.time()

    for ep in range(1, num_episodes + 1):
        ep_seed = base_seed + ep
        traj = collect_afterstate_trajectory(
            searcher=searcher,
            depth=depth,
            use_d4=use_d4,
            apex_mode=apex_mode,
            seed=ep_seed,
        )

        ep_score = traj["score"]
        ep_max_tile = traj["max_tile"]
        ep_transitions = traj["transitions"]

        if ep_max_tile >= min_max_tile:
            for s_after, r_next, next_s_after, done in ep_transitions:
                buffer.add_afterstate(
                    afterstate=s_after,
                    reward=r_next,
                    next_afterstate=next_s_after,
                    done=done,
                )
            total_transitions_added += len(ep_transitions)

        scores.append(ep_score)
        max_tiles.append(ep_max_tile)

        if callback is not None:
            callback(ep, ep_score, ep_max_tile, len(ep_transitions))

    elapsed = max(1e-5, time.time() - t0)
    return {
        "episodes_simulated": num_episodes,
        "transitions_added": total_transitions_added,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "max_tile": int(max(max_tiles)) if max_tiles else 0,
        "elapsed_sec": elapsed,
        "transitions_per_sec": float(total_transitions_added / elapsed),
    }


def save_expert_dataset(
    transitions: List[Tuple[np.ndarray, float, np.ndarray, bool]],
    filepath: str,
) -> str:
    """Save afterstate transitions to compressed .npz archive."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    n = len(transitions)
    states = np.zeros((n, 4, 4), dtype=np.uint8)
    rewards = np.zeros(n, dtype=np.float32)
    next_states = np.zeros((n, 4, 4), dtype=np.uint8)
    dones = np.zeros(n, dtype=bool)

    for i, (st, r, nst, d) in enumerate(transitions):
        states[i] = st
        rewards[i] = r
        next_states[i] = nst
        dones[i] = d

    np.savez_compressed(
        filepath,
        states=states,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )
    return filepath


def load_expert_dataset(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load afterstate dataset from compressed .npz archive."""
    data = np.load(filepath)
    return data["states"], data["rewards"], data["next_states"], data["dones"]
