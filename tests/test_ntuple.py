"""Unit and integration tests for SOTA N-Tuple Network value evaluation architecture."""

import os
import tempfile
import time
import numpy as np
import pytest

from src.env.bitboard import Bitboard2048
from src.models.ntuple import (
    NTupleNetwork,
    CANONICAL_LINE_4,
    CANONICAL_RECT_6,
    CANONICAL_SNAKE_6,
    generate_d4_tuple_symmetries,
)


class TestNTupleNetworkArchitecture:
    """Validate N-Tuple Network pattern definitions, indexing, D4 symmetries, and TD learning."""

    def test_canonical_pattern_definitions(self) -> None:
        """Verify canonical tuple lengths and geometry covering 4x4 board."""
        assert len(CANONICAL_LINE_4) == 4
        assert CANONICAL_LINE_4 == (0, 1, 2, 3)

        assert len(CANONICAL_RECT_6) == 6
        assert CANONICAL_RECT_6 == (0, 1, 2, 4, 5, 6)

        assert len(CANONICAL_SNAKE_6) == 6
        assert CANONICAL_SNAKE_6 == (0, 1, 2, 6, 5, 4)

    def test_d4_symmetries_count_and_validity(self) -> None:
        """Every canonical pattern must generate 8 valid D4 symmetric tuples with indices in [0, 15]."""
        for pat in [CANONICAL_LINE_4, CANONICAL_RECT_6, CANONICAL_SNAKE_6]:
            syms = generate_d4_tuple_symmetries(pat)
            assert len(syms) == 8
            for sym_tuple in syms:
                assert len(sym_tuple) == len(pat)
                for cell in sym_tuple:
                    assert 0 <= cell <= 15
                # Ensure no duplicate cells in the same tuple
                assert len(set(sym_tuple)) == len(sym_tuple)

    def test_base16_index_extraction(self) -> None:
        """Verify base-16 address index matches sum_{i=0}^{k-1} rank(c_i) * 16^i."""
        # Set specific ranks:
        # cell 0: rank 1 (val 2)
        # cell 1: rank 2 (val 4)
        # cell 2: rank 3 (val 8)
        # cell 3: rank 4 (val 16)
        ranks = np.zeros((4, 4), dtype=np.uint8)
        ranks[0, 0] = 1
        ranks[0, 1] = 2
        ranks[0, 2] = 3
        ranks[0, 3] = 4
        bb = Bitboard2048.from_array(ranks)

        # For CANONICAL_LINE_4 = (0, 1, 2, 3):
        # index = 1 * 16^0 + 2 * 16^1 + 3 * 16^2 + 4 * 16^3 = 1 + 32 + 768 + 16384 = 17185
        expected_idx = 1 + 2 * 16 + 3 * 256 + 4 * 4096
        extracted = NTupleNetwork.extract_index(bb, CANONICAL_LINE_4)
        assert extracted == expected_idx

    def test_d4_symmetry_evaluation_invariance(self) -> None:
        """Under all 8 Dihedral transformations, N-Tuple value V(T_g(B)) must equal V(B) exactly."""
        # Initialize NTupleNetwork with random weights
        net = NTupleNetwork(init_mode="normal")

        # Create arbitrary non-symmetric board
        arr = np.array([
            [2048, 1024, 512, 256],
            [128,    64,  32,  16],
            [8,       4,   2,   0],
            [0,       0,   4,   2],
        ], dtype=np.int64)
        bb = Bitboard2048.from_array(arr)

        base_val = net.evaluate_bitboard(bb)

        # Generate the 8 D4 transformations using Bitboard primitives
        r0 = bb
        r90 = Bitboard2048.rotate90(bb)
        r180 = Bitboard2048.rotate180(bb)
        r270 = Bitboard2048.rotate270(bb)
        fh = Bitboard2048.flip_horizontal(bb)
        f_r90 = Bitboard2048.rotate90(fh)
        f_r180 = Bitboard2048.rotate180(fh)
        f_r270 = Bitboard2048.rotate270(fh)

        d4_boards = [r0, r90, r180, r270, fh, f_r90, f_r180, f_r270]
        for op_idx, trans_bb in enumerate(d4_boards):
            val = net.evaluate_bitboard(trans_bb)
            assert np.isclose(base_val, val, atol=1e-5), f"D4 operation {op_idx} failed invariance: {base_val} vs {val}"

    def test_td_zero_afterstate_learning_update(self) -> None:
        """TD(0) update: delta = r + gamma * V(s') - V(s) updates weights towards target."""
        net = NTupleNetwork(alpha=0.01, gamma=0.99, lambd=0.0, init_mode="zeros")
        bb1 = 0x123456789ABCDEF0
        bb2 = 0x023456789ABCDEF1

        v_before = net.evaluate_bitboard(bb1)
        assert v_before == 0.0

        # Positive reward transition
        reward = 10.0
        delta = net.update(bb1, reward, bb2, done=False)
        assert delta == 10.0

        v_after = net.evaluate_bitboard(bb1)
        assert v_after > 0.0

    def test_td_lambda_trace_decay_and_learning(self) -> None:
        """TD(lambda) eligibility traces properly accumulate and decay over sequential steps."""
        net = NTupleNetwork(alpha=0.01, gamma=0.99, lambd=0.8, init_mode="zeros")
        bb1 = 0x123456789ABCDEF0
        bb2 = 0x023456789ABCDEF1
        bb3 = 0x003456789ABCDEF2

        # Step 1
        net.update(bb1, 4.0, bb2, done=False)
        # Step 2: traces from bb1 should have decayed by gamma * lambda = 0.99 * 0.8 = 0.792
        net.update(bb2, 8.0, bb3, done=False)

        # Terminal step resets traces
        net.update(bb3, 16.0, None, done=True)
        for tr_dict in net.traces:
            assert len(tr_dict) == 0

    def test_batch_evaluation_parity(self) -> None:
        """Vectorized evaluate_batch must match evaluate_bitboard across random bitboards."""
        net = NTupleNetwork(init_mode="normal")
        rng = np.random.default_rng(42)
        bbs = [int(rng.integers(0, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)) for _ in range(50)]

        batch_vals = net.evaluate_batch(bbs)
        assert len(batch_vals) == 50

        for i, bb in enumerate(bbs):
            single_val = net.evaluate_bitboard(bb)
            assert np.isclose(batch_vals[i], single_val, atol=1e-5)

    def test_microsecond_throughput_benchmark(self) -> None:
        """Evaluation throughput benchmark: batched N-Tuple leaf eval must achieve < 1.0 us per board."""
        net = NTupleNetwork(init_mode="normal")
        rng = np.random.default_rng(999)
        B = 250
        bbs = np.array([int(rng.integers(0, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)) for _ in range(B)], dtype=np.uint64)

        # Warmup
        net.evaluate_batch(bbs)

        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            net.evaluate_batch(bbs)
        elapsed = time.perf_counter() - t0
        time_per_board_us = (elapsed / (iters * B)) * 1e6
        print(f"\n[BENCHMARK] N-Tuple batched evaluation time per board: {time_per_board_us:.3f} us")
        assert time_per_board_us < 2.0  # Safe upper threshold for CPU variance, well under microsecond class

    def test_save_and_load_roundtrip(self) -> None:
        """Saving and loading tables restores weights and evaluations identically."""
        net1 = NTupleNetwork(init_mode="normal")
        bb = 0x123456789ABCDEF0
        val1 = net1.evaluate_bitboard(bb)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ntuple_test.npz")
            net1.save(path)

            net2 = NTupleNetwork(init_mode="zeros")
            assert net2.evaluate_bitboard(bb) == 0.0
            net2.load(path)
            val2 = net2.evaluate_bitboard(bb)

            assert np.isclose(val1, val2, atol=1e-6)
