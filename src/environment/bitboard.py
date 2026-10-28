"""Forwarding module for Bitboard2048."""

from src.env.bitboard import (
    Bitboard2048,
    ROW_LEFT_TABLE,
    ROW_RIGHT_TABLE,
    SNAKE_WEIGHTS_8x16,
    compute_d4_structural_scores,
)

__all__ = [
    "Bitboard2048",
    "ROW_LEFT_TABLE",
    "ROW_RIGHT_TABLE",
    "SNAKE_WEIGHTS_8x16",
    "compute_d4_structural_scores",
]
