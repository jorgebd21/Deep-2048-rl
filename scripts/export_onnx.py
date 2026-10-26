"""ONNX export and numerical parity verification CLI for Dueling2048Net."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.models.dueling_dqn import Dueling2048Net, export_to_onnx


def parse_args() -> argparse.Namespace:
    """Parse ONNX export command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Dueling2048Net to ONNX and verify numerical parity."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="model_2048.onnx", help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--skip-verify", action="store_true", help="Skip ONNX Runtime verification")
    return parser.parse_args()


def verify_onnx_parity(
    model: Dueling2048Net,
    onnx_path: str,
    test_batch_sizes: list = [1, 8, 32],
    tolerance: float = 1e-5,
) -> float:
    """Run ONNX Runtime inference and compare against PyTorch outputs."""
    import onnxruntime as ort

    model.eval()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    max_overall_error = 0.0

    print(f"\nVerifying ONNX numerical parity against PyTorch:")
    for bs in test_batch_sizes:
        test_input = torch.randn(bs, 16, 4, 4, dtype=torch.float32)

        # PyTorch inference
        with torch.no_grad():
            pt_out = model(test_input).cpu().numpy()

        # ONNX Runtime inference
        ort_inputs = {input_name: test_input.numpy()}
        ort_out = session.run(None, ort_inputs)[0]

        max_err = float(np.max(np.abs(pt_out - ort_out)))
        mean_err = float(np.mean(np.abs(pt_out - ort_out)))
        max_overall_error = max(max_overall_error, max_err)

        status = "[PASS]" if max_err < tolerance else "[FAIL]"
        print(
            f"  Batch {bs:<3} | Max Abs Error: {max_err:9.2e} | "
            f"Mean Abs Error: {mean_err:9.2e} | Status: {status}"
        )

        assert max_err < tolerance, (
            f"Parity check failed for batch {bs}: max error {max_err:.2e} >= {tolerance}"
        )

    return max_overall_error


def main() -> None:
    """Execute ONNX export and verification workflow."""
    args = parse_args()

    print("\n" + "=" * 60)
    print(" ONNX EXPORT PIPELINE: DUELING-DQN 2048")
    print("=" * 60)

    model = Dueling2048Net(in_channels=16, num_actions=4)

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("policy_net_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"  Loaded weights from : {args.checkpoint}")
    else:
        print("  Exporting model with fresh random weights (baseline).")

    out_path = export_to_onnx(
        model=model,
        output_path=args.output,
        dynamic_batch=True,
        opset_version=args.opset,
    )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Exported ONNX Model : {out_path}")
    print(f"  Model Size          : {size_mb:.2f} MB")
    print(f"  Opset Version       : {args.opset}")
    print(f"  Dynamic Axes        : batch_size")

    if not args.skip_verify:
        max_err = verify_onnx_parity(model, out_path)
        print(f"\n[OK] ONNX Parity Verified: Max error across all test batches = {max_err:.2e} (< 1e-5)")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
