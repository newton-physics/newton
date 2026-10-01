# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.examples.rl.example_rl_drlegs import _build_observation_kernel
from newton._src.solvers.kamino.examples.rl.joystick import JoystickConfig, JoystickController
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim

_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_WARP_NN = importlib.util.find_spec("warp_nn") is not None

if _HAS_ONNX and _HAS_WARP_NN:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from newton._src.solvers.kamino.examples.rl.onnx_policy import WarpOnnxPolicy


class TestKaminoRlDrlegsWarp(unittest.TestCase):
    """Test the Warp-native DR Legs policy path."""

    def test_module_imports_without_torch(self):
        """Import the DR Legs example when PyTorch is unavailable."""
        code = """
import builtins

real_import = builtins.__import__


def import_without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("PyTorch is intentionally unavailable")
    return real_import(name, *args, **kwargs)


builtins.__import__ = import_without_torch
import newton._src.solvers.kamino.examples.rl.example_rl_drlegs  # noqa: F401, E402
"""
        subprocess.run([sys.executable, "-c", code], check=True, cwd=os.getcwd())

    def test_builds_observation_with_warp_arrays(self):
        """Build the policy observation without Torch tensor interop."""
        joint_positions = np.arange(43, dtype=np.float32)
        body_q = wp.array(
            [
                wp.transformf(wp.vec3f(), wp.quat_identity(dtype=wp.float32)),
                wp.transformf(wp.vec3f(0.0, 0.0, 0.265), wp.quat_identity(dtype=wp.float32)),
            ],
            dtype=wp.transformf,
            device="cpu",
        )
        body_u = wp.zeros(2, dtype=wp.spatial_vectorf, device="cpu")
        actions = wp.ones((1, 12), dtype=wp.float32, device="cpu")
        command = wp.array([wp.vec4f(0.0, 0.0, 0.0, 0.265)], dtype=wp.vec4f, device="cpu")
        phase = wp.zeros(1, dtype=wp.float32, device="cpu")
        path_heading = wp.zeros(1, dtype=wp.float32, device="cpu")
        path_position = wp.zeros(1, dtype=wp.vec2f, device="cpu")
        action_history = wp.zeros((1, 12), dtype=wp.float32, device="cpu")
        action_history_prev = wp.zeros((1, 12), dtype=wp.float32, device="cpu")
        observation = wp.zeros((1, 94), dtype=wp.float32, device="cpu")

        wp.launch(
            _build_observation_kernel,
            dim=1,
            inputs=[
                body_q,
                body_u,
                wp.array(joint_positions, dtype=wp.float32, device="cpu"),
                actions,
                command,
                phase,
                path_heading,
                path_position,
                action_history,
                action_history_prev,
                1,
                1,
                7,
                43,
                0.02,
                1.0 / 0.6,
                0.4,
                0.1,
                0.1,
                0.05,
                observation,
            ],
            device="cpu",
        )

        actual = observation.numpy()[0]
        np.testing.assert_allclose(actual[:9], np.eye(3, dtype=np.float32).reshape(-1), atol=1.0e-6)
        np.testing.assert_allclose(actual[34:70], np.concatenate((joint_positions[:1], joint_positions[8:])))
        np.testing.assert_allclose(actual[70:82], 0.4)
        np.testing.assert_allclose(actual[82:94], 0.0)

    def test_joystick_torch_free_interface(self):
        """Expose input metadata and reject path resets when tracking is disabled."""
        config = JoystickConfig(head_pitch_up=0.8, head_pitch_down=0.4)
        with contextlib.redirect_stdout(io.StringIO()):
            joystick = JoystickController(dt=0.02, device="cpu", config=config, track_path=False)

        self.assertIsNone(joystick.input_mode)
        self.assertEqual(joystick.head_pitch_up_limit, 0.8)
        self.assertEqual(joystick.head_pitch_down_limit, 0.4)
        joystick.reset()
        with self.assertRaisesRegex(RuntimeError, "Path tracking was disabled"):
            joystick.reset(root_pos_2d=object())

    def test_warp_interface_rejects_torch_reset_staging(self):
        """Report unsupported Torch-style indexed reset staging clearly."""
        wrapper = RigidBodySim.__new__(RigidBodySim)
        wrapper._use_torch = False
        with self.assertRaisesRegex(RuntimeError, "use_torch=True"):
            wrapper.set_dof()
        with self.assertRaisesRegex(RuntimeError, "use_torch=True"):
            wrapper.set_root()

    def test_body_pair_filter_stays_in_warp(self):
        """Keep the body-pair flag as a Warp array when Torch is disabled."""

        class ContactAggregationStub:
            def __init__(self, flag):
                self.body_pair_contact_flag = flag
                self.filter = None

            def set_body_pair_filter(self, body_a_index, body_b_index):
                self.filter = (body_a_index, body_b_index)

        flag = wp.zeros(2, dtype=wp.int32, device="cpu")
        aggregation = ContactAggregationStub(flag)
        wrapper = RigidBodySim.__new__(RigidBodySim)
        wrapper._use_torch = False
        wrapper._body_names = ["left", "right"]
        wrapper._contact_aggregation = aggregation

        wrapper.set_body_pair_contact_filter("left", "right")

        self.assertEqual(aggregation.filter, (0, 1))
        self.assertIs(wrapper.body_pair_contact_flag, flag)


@unittest.skipUnless(_HAS_ONNX and _HAS_WARP_NN, "onnx or warp-nn not installed")
class TestKaminoRlOnnx(unittest.TestCase):
    """Test Warp-NN policy inference used by the Kamino RL example."""

    def _save_policy(
        self,
        directory: str | os.PathLike[str],
        *,
        input_count: int = 1,
        output_count: int = 1,
        output_width: int = 2,
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """Create a small ONNX policy with configurable inputs and outputs."""
        weights = np.arange(output_width * 2, dtype=np.float32).reshape(output_width, 2)
        bias = np.arange(output_width, dtype=np.float32)
        offset = np.zeros(2, dtype=np.float32)
        scale = np.ones(2, dtype=np.float32)
        inputs = [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [None, 2])]
        inputs.extend(
            helper.make_tensor_value_info(f"extra_input_{i}", TensorProto.FLOAT, [None, 2])
            for i in range(input_count - 1)
        )
        outputs = [helper.make_tensor_value_info("action", TensorProto.FLOAT, [None, output_width])]
        outputs.extend(
            helper.make_tensor_value_info(f"extra_output_{i}", TensorProto.FLOAT, [None, output_width])
            for i in range(output_count - 1)
        )
        nodes = [
            helper.make_node("Sub", ["observation", "offset"], ["centered"]),
            helper.make_node("Div", ["centered", "scale"], ["normalized"]),
            helper.make_node("Gemm", ["normalized", "weight", "bias"], ["action"], transB=1),
        ]
        nodes.extend(
            helper.make_node("Gemm", ["normalized", "weight", "bias"], [f"extra_output_{i}"], transB=1)
            for i in range(output_count - 1)
        )
        graph = helper.make_graph(
            nodes,
            "policy",
            inputs,
            outputs,
            [
                numpy_helper.from_array(weights, "weight"),
                numpy_helper.from_array(bias, "bias"),
                numpy_helper.from_array(offset, "offset"),
                numpy_helper.from_array(scale, "scale"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        path = os.path.join(directory, "policy.onnx")
        onnx.save(model, path)
        return path, weights, bias

    def test_policy_accepts_warp_array(self):
        """Evaluate a DR Legs-style ONNX policy from a Warp input."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            path, weights, bias = self._save_policy(tmp_dir)
            policy = WarpOnnxPolicy(path, device="cpu", batch_size=2, action_width=2)
            observation = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
            actual = policy(wp.array(observation, dtype=wp.float32, device="cpu")).numpy()

        expected = observation @ weights.T + bias
        np.testing.assert_allclose(actual, expected)

    def test_policy_rejects_invalid_warp_dtype(self):
        """Reject Warp observations with an incompatible dtype."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            path, _, _ = self._save_policy(tmp_dir)
            policy = WarpOnnxPolicy(path, device="cpu", batch_size=2, action_width=2)

            with self.assertRaisesRegex(TypeError, "wp.float32"):
                policy(wp.ones((2, 2), dtype=wp.float64, device="cpu"))

    def test_policy_rejects_multiple_inputs_or_outputs(self):
        """Reject policy models that do not have one input and one output."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            path, _, _ = self._save_policy(tmp_dir, input_count=2)
            with self.assertRaisesRegex(ValueError, "exactly one input and one output"):
                WarpOnnxPolicy(path, device="cpu", batch_size=2, action_width=2)

            path, _, _ = self._save_policy(tmp_dir, output_count=2)
            with self.assertRaisesRegex(ValueError, "exactly one input and one output"):
                WarpOnnxPolicy(path, device="cpu", batch_size=2, action_width=2)

    def test_policy_rejects_invalid_action_width(self):
        """Reject a policy whose output width does not match the actions."""
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp_dir:
            path, _, _ = self._save_policy(tmp_dir, output_width=3)
            with self.assertRaisesRegex(ValueError, "output shape"):
                WarpOnnxPolicy(path, device="cpu", batch_size=2, action_width=2)


if __name__ == "__main__":
    unittest.main()
