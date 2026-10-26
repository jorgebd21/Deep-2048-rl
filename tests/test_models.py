"""Unit tests for Dueling2048Net architecture, gradients, and action masking."""

import pytest
import torch
import torch.nn as nn

from src.models.dueling_dqn import Dueling2048Net
from src.utils.cuda_utils import get_device, setup_ampere_tf32


class TestDueling2048Net:
    """Test suite for Dueling Deep Q-Network."""

    def test_forward_dimensions(self):
        model = Dueling2048Net(in_channels=16, num_actions=4)
        model.eval()

        # Test batch size 1
        x_single = torch.randn(1, 16, 4, 4)
        out_single = model(x_single)
        assert out_single.shape == (1, 4)

        # Test mini-batch size 32
        x_batch = torch.randn(32, 16, 4, 4)
        out_batch = model(x_batch)
        assert out_batch.shape == (32, 4)

        # Test large training mini-batch 512
        x_large = torch.randn(512, 16, 4, 4)
        out_large = model(x_large)
        assert out_large.shape == (512, 4)

    def test_dueling_aggregation_property(self):
        """Verify that Q(s, a) - V(s) sums to zero across actions (identifiability constraint)."""
        model = Dueling2048Net(in_channels=16, num_actions=4)
        model.eval()

        x = torch.randn(16, 16, 4, 4)
        # Direct extraction of internal V and A
        h = model.leaky_relu(model.conv_horiz(x)).flatten(1)
        v = model.leaky_relu(model.conv_vert(x)).flatten(1)
        s = model.leaky_relu(model.conv_sq(x)).flatten(1)
        k = model.leaky_relu(model.conv_skip(x)).flatten(1)
        features = torch.cat([h, v, s, k], dim=1)
        shared = model.fc_shared(features)

        v_s = model.value_stream(shared)
        a_s = model.advantage_stream(shared)
        q_direct = v_s + (a_s - a_s.mean(dim=-1, keepdim=True))

        q_model = model(x)
        torch.testing.assert_close(q_model, q_direct)

        # Sum of advantage deviations across actions must be 0
        diff = q_model - v_s
        torch.testing.assert_close(
            diff.sum(dim=-1),
            torch.zeros(16),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_action_masking(self):
        model = Dueling2048Net(in_channels=16, num_actions=4)
        model.eval()

        x = torch.randn(4, 16, 4, 4)
        # Mask: Sample 0 allows only action 1 & 2; Sample 1 allows only action 0
        mask = torch.tensor(
            [
                [False, True, True, False],
                [True, False, False, False],
                [False, False, False, True],
                [True, True, True, True],
            ],
            dtype=torch.bool,
        )

        q_unmasked = model(x)
        q_masked = model(x, action_mask=mask)

        # Check masked actions are -1e9
        assert q_masked[0, 0].item() == -1e9
        assert q_masked[0, 3].item() == -1e9
        assert q_masked[1, 1].item() == -1e9
        assert q_masked[1, 2].item() == -1e9
        assert q_masked[1, 3].item() == -1e9
        assert q_masked[2, 0].item() == -1e9
        assert q_masked[2, 1].item() == -1e9
        assert q_masked[2, 2].item() == -1e9

        # Check unmasked actions retain exact Q values
        assert q_masked[0, 1].item() == q_unmasked[0, 1].item()
        assert q_masked[0, 2].item() == q_unmasked[0, 2].item()
        assert q_masked[1, 0].item() == q_unmasked[1, 0].item()
        assert q_masked[2, 3].item() == q_unmasked[2, 3].item()

        # Best action under mask must strictly be valid
        best_actions = q_masked.argmax(dim=-1)
        for i in range(4):
            assert mask[i, best_actions[i]].item() is True

    def test_gradient_propagation(self):
        """Ensure backpropagation propagates non-zero gradients to all layers and both streams."""
        model = Dueling2048Net(in_channels=16, num_actions=4)
        model.train()

        x = torch.randn(8, 16, 4, 4, requires_grad=True)
        q_vals = model(x)
        # Note: In dueling networks, sum_a(Q(s, a)) = 4*V(s) because sum_a(A(s, a) - mean_a(A)) = 0.
        # Thus, Q-learning targets a specific chosen action:
        loss = q_vals[:, 0].sum()
        loss.backward()

        # Check convolution gradients
        assert model.conv_horiz.weight.grad is not None
        assert model.conv_horiz.weight.grad.abs().sum().item() > 0
        assert model.conv_vert.weight.grad is not None
        assert model.conv_vert.weight.grad.abs().sum().item() > 0
        assert model.conv_sq.weight.grad is not None
        assert model.conv_sq.weight.grad.abs().sum().item() > 0
        assert model.conv_skip.weight.grad is not None
        assert model.conv_skip.weight.grad.abs().sum().item() > 0

        # Check shared FC gradients
        assert model.fc_shared[0].weight.grad is not None
        assert model.fc_shared[0].weight.grad.abs().sum().item() > 0

        # Check Value stream gradients
        assert model.value_stream[0].weight.grad is not None
        assert model.value_stream[0].weight.grad.abs().sum().item() > 0
        assert model.value_stream[2].weight.grad is not None
        assert model.value_stream[2].weight.grad.abs().sum().item() > 0

        # Check Advantage stream gradients
        assert model.advantage_stream[0].weight.grad is not None
        assert model.advantage_stream[0].weight.grad.abs().sum().item() > 0
        assert model.advantage_stream[2].weight.grad is not None
        assert model.advantage_stream[2].weight.grad.abs().sum().item() > 0

    def test_cuda_tf32_execution(self):
        """Test model execution on GPU with TF32 enabled if CUDA is available."""
        device = get_device(prefer_gpu=True)
        setup_ampere_tf32()

        model = Dueling2048Net(in_channels=16, num_actions=4).to(device)
        x = torch.randn(64, 16, 4, 4, device=device)
        mask = torch.ones(64, 4, dtype=torch.bool, device=device)

        out = model(x, action_mask=mask)
        assert out.device.type == device.type
        assert out.shape == (64, 4)
