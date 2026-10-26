"""Unit tests for evaluation metrics, ANSI renderer, and HUD visualizer."""

import numpy as np
import pytest
import torch

from src.models.dueling_dqn import Dueling2048Net
from src.utils.visualizer import format_tile, render_board_ansi, render_hud
from scripts.evaluate import evaluate_agent


class TestVisualizer:
    """Tests for terminal rendering and ANSI formatting."""

    def test_format_tile(self) -> None:
        """Verify tile string formatting for zero and non-zero values."""
        zero_tile = format_tile(0)
        assert "." in zero_tile
        assert "\033[0m" in zero_tile

        tile_2048 = format_tile(2048)
        assert "2048" in tile_2048
        assert "\033[1;31m" in tile_2048

    def test_render_board_ansi(self) -> None:
        """Verify 4x4 ANSI board grid generation."""
        board = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 0],
            [0, 0, 0, 0],
        ], dtype=np.int64)

        rendered = render_board_ansi(board)
        assert "┌" in rendered and "┐" in rendered
        assert "├" in rendered and "┤" in rendered
        assert "└" in rendered and "┘" in rendered
        assert "2048" in rendered
        assert "1024" in rendered

    def test_render_hud(self) -> None:
        """Verify HUD rendering with and without Q-values."""
        action_mask = np.array([True, False, True, False])

        # Without Q-values
        hud_basic = render_hud(
            score=1024,
            max_tile=256,
            step=42,
            action_mask=action_mask,
        )
        assert "Score:" in hud_basic
        assert "1024" in hud_basic
        assert "Max Tile:" in hud_basic
        assert "256" in hud_basic

        # With Q-values and selected action
        q_values = np.array([12.5, -5.0, 18.2, 0.0])
        hud_rich = render_hud(
            score=2048,
            max_tile=512,
            step=100,
            action_mask=action_mask,
            q_values=q_values,
            selected_action=2,
        )
        assert "Chosen" in hud_rich
        assert "[LEGAL]" in hud_rich
        assert "[MASK ]" in hud_rich
        assert "+18.20" in hud_rich


class TestEvaluationRunner:
    """Tests for Monte Carlo evaluation metrics computation."""

    def test_evaluate_agent_metrics(self) -> None:
        """Verify evaluate_agent returns expected statistical distributions."""
        model = Dueling2048Net(in_channels=16, num_actions=4)
        results = evaluate_agent(model=model, episodes=3, seed=42)

        assert results["num_episodes"] == 3
        assert "mean_score" in results
        assert "std_score" in results
        assert "median_score" in results
        assert "reach_256_rate" in results
        assert "reach_512_rate" in results
        assert "reach_1024_rate" in results
        assert "reach_2048_rate" in results
        assert "tile_distribution" in results
        assert isinstance(results["tile_distribution"], dict)
        assert 0.0 <= results["reach_256_rate"] <= 1.0
