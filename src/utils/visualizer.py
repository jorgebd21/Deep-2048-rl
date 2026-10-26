"""ANSI terminal visualization and HUD rendering for 2048."""

import os
import sys
from typing import Dict, List, Optional, Union
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ANSI color codes for 2048 tiles
TILE_COLORS: Dict[int, str] = {
    0: "\033[90m",        # Gray / Dim
    2: "\033[37m",        # Bright White
    4: "\033[36m",        # Cyan
    8: "\033[33m",        # Yellow
    16: "\033[32m",       # Green
    32: "\033[35m",       # Magenta
    64: "\033[31m",       # Red
    128: "\033[1;33m",    # Bold Yellow / Gold
    256: "\033[1;36m",    # Bold Cyan
    512: "\033[1;32m",    # Bold Green
    1024: "\033[1;35m",   # Bold Magenta
    2048: "\033[1;31m",   # Bold Red
    4096: "\033[1;97;41m",# White on Red background
}

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"

ACTION_NAMES = ["UP   ↑", "DOWN ↓", "LEFT ←", "RIGHT→"]


def format_tile(val: int) -> str:
    """Format single tile integer with ANSI colors and aligned padding."""
    color = TILE_COLORS.get(val, "\033[1;97;45m")
    if val == 0:
        return f"{color}   .  {RESET}"
    return f"{color}{val:^6d}{RESET}"


def render_board_ansi(board: np.ndarray) -> str:
    """Render 4x4 NumPy board as an ANSI formatted box.

    Args:
        board: 4x4 array of integer tile values.

    Returns:
        Multi-line ANSI colored string.
    """
    lines = []
    lines.append("  ┌──────┬──────┬──────┬──────┐")
    for r in range(4):
        row_str = "  │"
        for c in range(4):
            row_str += format_tile(int(board[r, c])) + "│"
        lines.append(row_str)
        if r < 3:
            lines.append("  ├──────┼──────┼──────┼──────┤")
    lines.append("  └──────┴──────┴──────┴──────┘")
    return "\n".join(lines)


def render_hud(
    score: int,
    max_tile: int,
    step: int,
    action_mask: np.ndarray,
    q_values: Optional[Union[np.ndarray, List[float]]] = None,
    selected_action: Optional[int] = None,
) -> str:
    """Render interactive heads-up display (HUD) with score, step, and Q-values.

    Args:
        score: Current game score.
        max_tile: Highest tile on current board.
        step: Episode step counter.
        action_mask: 4-element boolean array of legal moves.
        q_values: Optional 4-element array of predicted Q-values.
        selected_action: Action selected by policy.

    Returns:
        Formatted HUD string.
    """
    lines = []
    lines.append(f"  {BOLD}Score:{RESET} {CYAN}{score:<8d}{RESET}  "
                 f"{BOLD}Max Tile:{RESET} {YELLOW}{max_tile:<6d}{RESET}  "
                 f"{BOLD}Step:{RESET} {step:<6d}")
    lines.append("  " + "─" * 48)

    # Q-values section
    if q_values is not None:
        q_arr = np.asarray(q_values)
        lines.append(f"  {BOLD}Q-Values & Legal Action Distribution:{RESET}")
        for a_idx, name in enumerate(ACTION_NAMES):
            legal = bool(action_mask[a_idx])
            is_chosen = (selected_action == a_idx)

            prefix = " ► " if is_chosen else "   "
            mask_tag = f"{GREEN}[LEGAL]{RESET}" if legal else f"{RED}[MASK ]{RESET}"
            q_val_str = f"{q_arr[a_idx]:+8.2f}" if legal and q_arr[a_idx] > -1e8 else "   N/A  "

            if is_chosen:
                lines.append(f"  {BOLD}{prefix}{name} {mask_tag} Q = {CYAN}{q_val_str}{RESET}{BOLD} (Chosen){RESET}")
            else:
                lines.append(f"  {DIM}{prefix}{name} {mask_tag} Q = {q_val_str}{RESET}")
    else:
        # Legal masks only
        mask_strs = [
            f"{ACTION_NAMES[i]}: {GREEN}OK{RESET}" if action_mask[i] else f"{ACTION_NAMES[i]}: {RED}X{RESET}"
            for i in range(4)
        ]
        lines.append(f"  {BOLD}Legal Moves:{RESET} {' | '.join(mask_strs)}")

    return "\n".join(lines)


def clear_screen() -> None:
    """Clear terminal screen cross-platform."""
    os.system("cls" if os.name == "nt" else "clear")
