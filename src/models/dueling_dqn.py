"""Dueling Deep Q-Network (Dueling2048Net) architecture for 2048."""

import os
from typing import Optional
import torch
import torch.nn as nn


class Dueling2048Net(nn.Module):
    """Convolutional Dueling Deep Q-Network tailored for 4x4 combinatoric boards.

    Extracts multi-scale geometric relations through horizontal, vertical, and square
    asymmetric kernels, fusing them into decoupled Value V(s) and Advantage A(s, a) streams
    with centered mean-subtracted identifiability aggregation.

    Attributes:
        in_channels: Number of input binary planes (16 by default).
        num_actions: Number of discrete action choices (4: LEFT, RIGHT, UP, DOWN).
    """

    def __init__(
        self,
        in_channels: int = 16,
        num_actions: int = 4,
        dropout_p: float = 0.05,
    ) -> None:
        """Initialize convolutional branches and dueling streams."""
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions

        # Asymmetric multi-scale spatial feature extractors
        # 1. Horizontal kernel (adjacent tiles along rows) -> output (B, 128, 4, 3)
        self.conv_horiz = nn.Conv2d(in_channels, 128, kernel_size=(1, 2))
        # 2. Vertical kernel (adjacent tiles along columns) -> output (B, 128, 3, 4)
        self.conv_vert = nn.Conv2d(in_channels, 128, kernel_size=(2, 1))
        # 3. Square 2x2 kernel (local 2x2 tile clusters) -> output (B, 128, 3, 3)
        self.conv_sq = nn.Conv2d(in_channels, 128, kernel_size=(2, 2))
        # 4. Pointwise skip projection (global channel mixing) -> output (B, 64, 4, 4)
        self.conv_skip = nn.Conv2d(in_channels, 64, kernel_size=(1, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Flattened dimensions:
        # 128*4*3 (1536) + 128*3*4 (1536) + 128*3*3 (1152) + 64*4*4 (1024) = 5248
        conv_flat_dim = (128 * 4 * 3) + (128 * 3 * 4) + (128 * 3 * 3) + (64 * 4 * 4)

        # Common dense projection
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_flat_dim, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_p),
        )

        # Value Stream V(s): R^(512) -> R^(128) -> R^1
        self.value_stream = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 1),
        )

        # Advantage Stream A(s, a): R^(512) -> R^(128) -> R^4
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, num_actions),
        )

    def forward(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Q-values for input batch with optional legal action masking.

        Args:
            x: Batch of one-hot encoded boards of shape (B, 16, 4, 4).
            action_mask: Optional boolean tensor of shape (B, 4) where True
                indicates a legal action.

        Returns:
            Tensor of Q-values of shape (B, 4).
        """
        # Multi-scale convolutional extraction
        h = self.leaky_relu(self.conv_horiz(x)).flatten(1)
        v = self.leaky_relu(self.conv_vert(x)).flatten(1)
        s = self.leaky_relu(self.conv_sq(x)).flatten(1)
        k = self.leaky_relu(self.conv_skip(x)).flatten(1)

        # Concat feature maps
        features = torch.cat([h, v, s, k], dim=1)
        shared = self.fc_shared(features)

        # Dueling decomposition
        v_s = self.value_stream(shared)  # (B, 1)
        a_s = self.advantage_stream(shared)  # (B, 4)

        # Centered aggregation: Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
        q_values = v_s + (a_s - a_s.mean(dim=-1, keepdim=True))

        # Additive action masking: illegal actions set to dtype-safe negative minimum
        if action_mask is not None:
            mask_bool = action_mask.bool()
            min_val = -1e4 if q_values.dtype == torch.float16 else -1e9
            neg_inf = torch.full_like(q_values, min_val)
            q_values = torch.where(mask_bool, q_values, neg_inf)

        return q_values


def export_to_onnx(
    model: Dueling2048Net,
    output_path: str,
    dynamic_batch: bool = True,
    opset_version: int = 17,
) -> str:
    """Export Dueling2048Net to ONNX format with dynamic batch axes.

    Args:
        model: Trained or initialized Dueling2048Net instance.
        output_path: Filepath where the .onnx model will be written.
        dynamic_batch: Whether to assign dynamic batch dimension.
        opset_version: ONNX operator set version.

    Returns:
        Absolute filepath to the saved ONNX model.
    """
    model.eval()
    cpu_model = model.to("cpu")
    dummy_input = torch.randn(1, model.in_channels, 4, 4, dtype=torch.float32)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "observation": {0: "batch_size"},
            "q_values": {0: "batch_size"},
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    torch.onnx.export(
        cpu_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["q_values"],
        dynamic_axes=dynamic_axes,
    )
    return os.path.abspath(output_path)
