# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import tempfile
import unittest

import numpy as np

_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_WARP_NN = importlib.util.find_spec("warp_nn") is not None

if _HAS_ONNX and _HAS_TORCH and _HAS_WARP_NN:
    import onnx
    import torch
    from onnx import TensorProto, helper, numpy_helper

    from newton._src.solvers.kamino.examples.rl.onnx_policy import WarpOnnxPolicy


@unittest.skipUnless(_HAS_ONNX and _HAS_TORCH and _HAS_WARP_NN, "onnx, torch, or warp-nn not installed")
class TestKaminoRlOnnx(unittest.TestCase):
    """Test Warp-NN policy inference used by the Kamino RL example."""

    def test_policy_accepts_torch_tensor(self):
        """Evaluate an ONNX policy from a zero-copy Torch input."""
        weights = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float32)
        bias = np.array([0.25, -0.5], dtype=np.float32)
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["observation", "weight", "bias"], ["action"], transB=1)],
            "policy",
            [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, 2])],
            [numpy_helper.from_array(weights, "weight"), numpy_helper.from_array(bias, "bias")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            path = os.path.join(tmp_dir, "policy.onnx")
            onnx.save(model, path)
            policy = WarpOnnxPolicy(path, device="cpu", batch_size=2)
            observation = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32)
            actual = policy(observation)

        expected = observation @ torch.from_numpy(weights).T + torch.from_numpy(bias)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
