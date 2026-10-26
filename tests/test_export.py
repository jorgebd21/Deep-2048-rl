"""Unit tests for ONNX export and runtime verification."""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.models.dueling_dqn import Dueling2048Net, export_to_onnx
from scripts.export_onnx import verify_onnx_parity


class TestONNXExport:
    """Tests for ONNX model generation and numerical equivalence."""

    def test_export_file_creation(self) -> None:
        """Verify export produces a valid file with non-zero size."""
        model = Dueling2048Net(in_channels=16, num_actions=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, "test_model.onnx")
            saved_path = export_to_onnx(model, onnx_path, dynamic_batch=True)

            assert os.path.exists(saved_path)
            # Dueling2048Net is ~2.8M parameters, so file should be > 5 MB
            assert os.path.getsize(saved_path) > 5 * 1024 * 1024

    def test_onnx_runtime_parity(self) -> None:
        """Verify ONNX Runtime inference matches PyTorch output within 1e-5."""
        model = Dueling2048Net(in_channels=16, num_actions=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, "test_model.onnx")
            export_to_onnx(model, onnx_path, dynamic_batch=True)

            max_err = verify_onnx_parity(
                model=model,
                onnx_path=onnx_path,
                test_batch_sizes=[1, 4, 16],
                tolerance=1e-5,
            )

            assert max_err < 1e-5
