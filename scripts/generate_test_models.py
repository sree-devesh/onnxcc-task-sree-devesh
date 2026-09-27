#!/usr/bin/env python3
"""
generate_test_models.py    Create ONNX test fixtures for onnxcc.

Architecture : 4 → 8 → 2  (ReLU after each linear layer)
Operators    : MatMul, Add, Relu  (opset 13, no Gemm)
Seed         : 42  (fixed — script is idempotent)

Generated graph stats
---------------------
  Nodes        : 6  (MatMul, Add, Relu, MatMul, Add, Relu)
  Initializers : 4  (W1 [4x8], b1 [8], W2 [8x2], b2 [2])

Outputs (written to tests/fixtures/ — not committed)
----------------------------------------------------
  mlp.onnx          ONNX model
  mlp_input.bin     raw float32 (1,4) input, 16 bytes

Usage
-----
  python3 scripts/generate_test_models.py
"""

import pathlib
import struct

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# ── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = pathlib.Path(__file__).parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

RNG_SEED = 42


def build_mlp() -> onnx.ModelProto:
    """Return a 4→8→2 MLP model using MatMul + Add + Relu at opset 13."""
    rng = np.random.default_rng(RNG_SEED)

    W1 = rng.standard_normal((4, 8)).astype(np.float32)
    b1 = rng.standard_normal((8,)).astype(np.float32)
    W2 = rng.standard_normal((8, 2)).astype(np.float32)
    b2 = rng.standard_normal((2,)).astype(np.float32)

    # 6 nodes: 2× MatMul, 2× Add, 2× Relu
    nodes = [
        helper.make_node("MatMul", inputs=["input", "W1"],  outputs=["mm1"]),
        helper.make_node("Add",    inputs=["mm1",   "b1"],  outputs=["add1"]),
        helper.make_node("Relu",   inputs=["add1"],         outputs=["relu1"]),
        helper.make_node("MatMul", inputs=["relu1", "W2"],  outputs=["mm2"]),
        helper.make_node("Add",    inputs=["mm2",   "b2"],  outputs=["add2"]),
        helper.make_node("Relu",   inputs=["add2"],         outputs=["output"]),
    ]

    input_vi  = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, 4])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

    # 4 initializers: W1, b1, W2, b2
    initializers = [
        numpy_helper.from_array(W1, name="W1"),
        numpy_helper.from_array(b1, name="b1"),
        numpy_helper.from_array(W2, name="W2"),
        numpy_helper.from_array(b2, name="b2"),
    ]

    graph = helper.make_graph(
        nodes, "mlp_4_8_2",
        [input_vi], [output_vi],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = onnx.IR_VERSION
    return model


def build_input() -> bytes:
    """Return a fixed (1,4) float32 input as 16 raw bytes."""
    rng = np.random.default_rng(RNG_SEED + 1)   # separate seed from weights
    values = rng.standard_normal((4,)).astype(np.float32)
    return struct.pack(f"{len(values)}f", *values)


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # ── model ────────────────────────────────────────────────────────────────
    model = build_mlp()
    onnx.checker.check_model(model)

    model_path = FIXTURE_DIR / "mlp.onnx"
    onnx.save(model, str(model_path))

    ops = [n.op_type for n in model.graph.node]
    print(f"wrote {model_path}  ({model_path.stat().st_size} bytes)")
    print(f"  op types    : {ops}")
    print(f"  nodes       : {len(model.graph.node)}")
    print(f"  initializers: {len(model.graph.initializer)}")

    # ── input fixture ────────────────────────────────────────────────────────
    raw = build_input()
    assert len(raw) == 16, f"expected 16 bytes, got {len(raw)}"

    bin_path = FIXTURE_DIR / "mlp_input.bin"
    bin_path.write_bytes(raw)
    print(f"wrote {bin_path}  ({len(raw)} bytes)")


if __name__ == "__main__":
    main()
