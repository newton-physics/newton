# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Newton actuators."""

import importlib.util
import json
import math
import os
import shutil
import tempfile
import types
import unittest
import warnings
from typing import Any, ClassVar
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.utils.import_usd import parse_usd
from newton.actuators import (
    Actuator,
    ActuatorParsed,
    ClampingDCMotor,
    ClampingMaxEffort,
    ClampingPositionBased,
    ControllerNeuralLSTM,
    ControllerNeuralMLP,
    ControllerPD,
    ControllerPID,
    Delay,
    SchemaNames,
    parse_actuator_prim,
)
from newton.selection import ArticulationView

try:
    from pxr import Usd

    HAS_USD = True
except ImportError:
    HAS_USD = False

try:
    from newton_actuators import ActuatorDelayedPD, ActuatorPD, ActuatorPID

    _HAS_LEGACY_ACTUATORS = True
except ImportError:
    _HAS_LEGACY_ACTUATORS = False

_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _onnx_modules():
    """Lazily import ``onnx`` submodules used by the test ONNX builders.

    Keeping these imports out of module scope so the optional ``onnx``
    dependency is only loaded when an ONNX-gated test actually runs (and to
    satisfy the ``TID253`` lint rule that bans module-level ``onnx`` imports
    in ``newton/tests/**``).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    return onnx, TensorProto, helper, numpy_helper


def _build_mlp_onnx(
    path: str,
    weights: np.ndarray,
    bias: np.ndarray,
    metadata: dict | None = None,
    batch_dim: int | None = None,
) -> None:
    """Build a single-Gemm (transB=1) ONNX MLP at ``path``.

    Args:
        weights: (out_dim, in_dim) Linear weights (PyTorch convention).
        bias: (out_dim,) Linear bias.
    """
    onnx_mod, TensorProto, helper, numpy_helper = _onnx_modules()

    in_dim = int(weights.shape[1])
    out_dim = int(weights.shape[0])

    x_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch_dim, in_dim])
    y_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_dim, out_dim])
    W_init = numpy_helper.from_array(weights.astype(np.float32), name="W")
    b_init = numpy_helper.from_array(bias.astype(np.float32), name="b")
    gemm = helper.make_node("Gemm", ["input", "W", "b"], ["output"], alpha=1.0, beta=1.0, transB=1)
    graph = helper.make_graph([gemm], "mlp", [x_vi], [y_vi], initializer=[W_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    if metadata is not None:
        meta_prop = model.metadata_props.add()
        meta_prop.key = "metadata"
        meta_prop.value = json.dumps(metadata)
    onnx_mod.checker.check_model(model)
    onnx_mod.save(model, path)


def _build_lstm_onnx(
    path: str,
    hidden_size: int = 8,
    num_layers: int = 1,
    metadata: dict | None = None,
    rng_seed: int = 0,
) -> None:
    """Build an ONNX LSTM model with random weights, layout=0.

    Inputs : ``input`` (1, N, 2), ``h_in`` (num_layers, N, H), ``c_in`` (..., N, H)
    Outputs: ``output`` (N, 1) effort, ``h_out`` (num_layers, N, H), ``c_out`` (...)
    """
    if num_layers != 1:
        raise NotImplementedError("test fixture currently supports num_layers=1")

    onnx_mod, TensorProto, helper, numpy_helper = _onnx_modules()

    rng = np.random.default_rng(rng_seed)
    input_size = 2

    W = (rng.standard_normal((1, 4 * hidden_size, input_size)) * 0.3).astype(np.float32)
    R = (rng.standard_normal((1, 4 * hidden_size, hidden_size)) * 0.3).astype(np.float32)
    B = (rng.standard_normal((1, 8 * hidden_size)) * 0.05).astype(np.float32)
    Wd = (rng.standard_normal((1, hidden_size)) * 0.3).astype(np.float32)
    bd = np.zeros((1,), dtype=np.float32)

    x_in = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, None, input_size])
    h_in = helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [num_layers, None, hidden_size])
    c_in = helper.make_tensor_value_info("c_in", TensorProto.FLOAT, [num_layers, None, hidden_size])
    y_out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])
    h_out = helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [num_layers, None, hidden_size])
    c_out = helper.make_tensor_value_info("c_out", TensorProto.FLOAT, [num_layers, None, hidden_size])

    initializers = [
        numpy_helper.from_array(W, name="W"),
        numpy_helper.from_array(R, name="R"),
        numpy_helper.from_array(B, name="B"),
        numpy_helper.from_array(Wd, name="Wd"),
        numpy_helper.from_array(bd, name="bd"),
    ]

    lstm = helper.make_node(
        "LSTM",
        ["input", "W", "R", "B", "", "h_in", "c_in"],
        ["Y", "h_out", "c_out"],
        hidden_size=hidden_size,
        layout=0,
    )
    # Y has shape (1, 1, N, hidden_size).  Squeeze first two dims -> (N, hidden_size).
    squeeze_axes = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="squeeze_axes")
    initializers.append(squeeze_axes)
    sq = helper.make_node("Squeeze", ["Y", "squeeze_axes"], ["Y_2d"])
    # Final linear decoder: Y_2d (N, H) @ Wd^T + bd -> (N, 1)
    dec = helper.make_node("Gemm", ["Y_2d", "Wd", "bd"], ["output"], alpha=1.0, beta=1.0, transB=1)

    graph = helper.make_graph(
        [lstm, sq, dec], "lstm_test", [x_in, h_in, c_in], [y_out, h_out, c_out], initializer=initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    full_meta = {
        "input_name": "input",
        "hidden_in_name": "h_in",
        "cell_in_name": "c_in",
        "output_name": "output",
        "hidden_out_name": "h_out",
        "cell_out_name": "c_out",
        "num_layers": num_layers,
        "hidden_size": hidden_size,
    }
    if metadata is not None:
        full_meta.update(metadata)
    meta_prop = model.metadata_props.add()
    meta_prop.key = "metadata"
    meta_prop.value = json.dumps(full_meta)
    onnx_mod.checker.check_model(model)
    onnx_mod.save(model, path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_dof_values(model, array, dof_indices, values):
    """Write scalar values into specific DOF positions of a Warp array."""
    arr_np = array.numpy()
    for dof, val in zip(dof_indices, values, strict=True):
        arr_np[dof] = val
    wp.copy(array, wp.array(arr_np, dtype=float, device=model.device))


def _wp_array(values, dtype=wp.float32, device=None, requires_grad: bool = False):
    """Compact helper to build a Warp array on the current default device."""
    return wp.array(values, dtype=dtype, device=device or wp.get_device(), requires_grad=requires_grad)


@wp.kernel
def _sum_kernel(values: wp.array[float], out: wp.array[float]):
    """Atomic-add reducer used as the differentiable loss in grad tests."""
    i = wp.tid()
    wp.atomic_add(out, 0, values[i])


# -------------------------------------------------------------------------
# 1. Unit controllers (analytic and network)
# -------------------------------------------------------------------------


class TestControllerNeuralMLP(unittest.TestCase):
    """ControllerNeuralMLP — load via model_path, call compute() directly."""

    def setUp(self):
        self.device = wp.get_device()
        self._tmp_dir = tempfile.mkdtemp()

    def _save_mlp(self, weights, bias, filename="mlp.onnx", metadata=None, batch_dim=None):
        path = os.path.join(self._tmp_dir, filename)
        _build_mlp_onnx(path, weights, bias, metadata, batch_dim=batch_dim)
        return path

    def test_compute(self):
        """Constant-bias network produces known output; history rolls after update_state."""
        weights = np.zeros((1, 2), dtype=np.float32)
        bias = np.array([42.0], dtype=np.float32)
        path = self._save_mlp(weights, bias)
        n = 1
        ctrl = ControllerNeuralMLP(model_path=path)
        ctrl.finalize(self.device, n)
        state_a = ctrl.state(n, self.device)
        state_b = ctrl.state(n, self.device)

        indices = wp.array([0], dtype=wp.uint32, device=self.device)
        positions = wp.zeros(n, dtype=wp.float32, device=self.device)
        velocities = wp.zeros(n, dtype=wp.float32, device=self.device)
        target_pos = wp.array([1.0], dtype=wp.float32, device=self.device)
        target_vel = wp.zeros(n, dtype=wp.float32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)

        ctrl.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            None,
            indices,
            indices,
            indices,
            indices,
            forces,
            state_a,
            0.01,
            self.device,
        )
        self.assertAlmostEqual(forces.numpy()[0], 42.0, places=3)

        ctrl.update_state(state_a, state_b)
        self.assertAlmostEqual(
            float(state_b.pos_error_history.numpy()[0, 0]),
            1.0,
            places=4,
            msg="history should contain pos error from current step",
        )

    def test_metadata_scales(self):
        """Metadata effort_scale is applied to the network output."""
        weights = np.zeros((1, 2), dtype=np.float32)
        bias = np.array([10.0], dtype=np.float32)
        path = self._save_mlp(weights, bias, metadata={"effort_scale": 3.0})

        n = 1
        ctrl = ControllerNeuralMLP(model_path=path)
        self.assertAlmostEqual(ctrl.effort_scale, 3.0)
        ctrl.finalize(self.device, n)
        state_a = ctrl.state(n, self.device)

        indices = wp.array([0], dtype=wp.uint32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)
        ctrl.compute(
            wp.zeros(n, dtype=wp.float32, device=self.device),
            wp.zeros(n, dtype=wp.float32, device=self.device),
            wp.array([1.0], dtype=wp.float32, device=self.device),
            wp.zeros(n, dtype=wp.float32, device=self.device),
            None,
            indices,
            indices,
            indices,
            indices,
            forces,
            state_a,
            0.01,
            self.device,
        )
        self.assertAlmostEqual(forces.numpy()[0], 30.0, places=3, msg="bias=10 * effort_scale=3 -> 30")

    def test_finalize_fixed_batch_onnx_with_multiple_actuators(self):
        """Fixed-batch ONNX exports can still run one scalar per actuator."""
        weights = np.array([[2.0, 0.0]], dtype=np.float32)
        bias = np.array([1.0], dtype=np.float32)
        path = self._save_mlp(weights, bias, filename="fixed_batch_mlp.onnx", batch_dim=1)

        n = 3
        ctrl = ControllerNeuralMLP(model_path=path)
        ctrl.finalize(self.device, n)
        self.assertEqual(ctrl._network._shapes[ctrl._net_input_name], (n, 2))
        self.assertEqual(ctrl._network._shapes[ctrl._net_output_name], (n, 1))

        indices = wp.array([0, 1, 2], dtype=wp.uint32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)
        ctrl.compute(
            wp.zeros(n, dtype=wp.float32, device=self.device),
            wp.zeros(n, dtype=wp.float32, device=self.device),
            wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.device),
            wp.zeros(n, dtype=wp.float32, device=self.device),
            None,
            indices,
            indices,
            indices,
            indices,
            forces,
            ctrl.state(n, self.device),
            0.01,
            self.device,
        )
        np.testing.assert_allclose(forces.numpy(), np.array([3.0, 5.0, 7.0], dtype=np.float32), rtol=1e-5)

    def test_graph_capture_matches_eager(self):
        """ONNX-backed ``compute`` is graph-safe; replays match eager outputs.

        Captures a single ``ctrl.compute`` call as a CUDA graph and replays it
        across a target schedule. Eager and graph paths must produce identical
        forces — this is what :meth:`ControllerNeuralMLP.is_graphable` promises
        for the Warp-native ONNX runtime backend.
        """
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture not supported on CPU")

        weights = np.array([[1.0, 2.0]], dtype=np.float32)
        bias = np.array([0.5], dtype=np.float32)
        path = self._save_mlp(weights, bias)
        n = 1
        target_schedule = [0.5, 1.0, 2.0, -1.5]

        def _setup():
            ctrl = ControllerNeuralMLP(model_path=path)
            ctrl.finalize(self.device, n)
            return types.SimpleNamespace(
                ctrl=ctrl,
                state=ctrl.state(n, self.device),
                positions=wp.zeros(n, dtype=wp.float32, device=self.device),
                velocities=wp.zeros(n, dtype=wp.float32, device=self.device),
                target_pos=wp.zeros(n, dtype=wp.float32, device=self.device),
                target_vel=wp.zeros(n, dtype=wp.float32, device=self.device),
                forces=wp.zeros(n, dtype=wp.float32, device=self.device),
                indices=wp.array([0], dtype=wp.uint32, device=self.device),
            )

        def _step(b):
            b.forces.zero_()
            b.ctrl.compute(
                b.positions,
                b.velocities,
                b.target_pos,
                b.target_vel,
                None,
                b.indices,
                b.indices,
                b.indices,
                b.indices,
                b.forces,
                b.state,
                0.01,
                self.device,
            )

        eager = _setup()
        self.assertTrue(eager.ctrl.is_graphable(), "ONNX-backed MLP should be graphable")

        eager_forces = []
        for tgt in target_schedule:
            wp.copy(eager.target_pos, wp.full(n, tgt, dtype=wp.float32, device=self.device))
            _step(eager)
            eager_forces.append(eager.forces.numpy().copy())

        g = _setup()
        with wp.ScopedCapture(device=self.device) as capture:
            _step(g)
        graph = capture.graph

        graph_forces = []
        for tgt in target_schedule:
            wp.copy(g.target_pos, wp.full(n, tgt, dtype=wp.float32, device=self.device))
            wp.capture_launch(graph)
            graph_forces.append(g.forces.numpy().copy())

        for i, tgt in enumerate(target_schedule):
            np.testing.assert_allclose(
                graph_forces[i],
                eager_forces[i],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"target={tgt}: graph replay diverged from eager",
            )

    @unittest.skipUnless(_HAS_TORCH, "torch not installed")
    def test_finalize_legacy_torchscript_checkpoint(self):
        """Legacy ``.pt`` checkpoints must finalize without a KeyError.

        Regression test: the deprecated ``_TorchModuleAdapter`` populates its
        ``_shapes`` dict only after the first ``__call__()``, so the output
        shape lookup in ``finalize()`` previously raised ``KeyError("action")``
        before any inference had happened.  ``finalize()`` should now probe the
        shape with a dry forward and accept the legacy checkpoint.
        """
        import torch

        n = 1
        in_features = 2  # matches default input_idx=[0] -> 2*K = 2

        class _BiasOnlyMLP(torch.nn.Module):
            """Single Linear layer with zero weights and a known bias."""

            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(in_features, 1, bias=True)
                with torch.no_grad():
                    self.fc.weight.zero_()
                    self.fc.bias.fill_(7.0)

            def forward(self, x):
                return self.fc(x)

        model = _BiasOnlyMLP().eval()
        scripted = torch.jit.script(model)
        path = os.path.join(self._tmp_dir, "legacy_mlp.pt")
        scripted.save(path, _extra_files={"metadata.json": json.dumps({"effort_scale": 1.0})})

        ctrl = ControllerNeuralMLP(model_path=path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ctrl.finalize(self.device, n)

        self.assertEqual(ctrl._net_output_name, "action")
        self.assertEqual(ctrl._network._shapes["action"], (n, 1))

        indices = wp.array([0], dtype=wp.uint32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)
        state_a = ctrl.state(n, self.device)
        ctrl.compute(
            wp.zeros(n, dtype=wp.float32, device=self.device),
            wp.zeros(n, dtype=wp.float32, device=self.device),
            wp.array([1.0], dtype=wp.float32, device=self.device),
            wp.zeros(n, dtype=wp.float32, device=self.device),
            None,
            indices,
            indices,
            indices,
            indices,
            forces,
            state_a,
            0.01,
            self.device,
        )
        self.assertAlmostEqual(float(forces.numpy()[0]), 7.0, places=3)


@unittest.skipUnless(_HAS_ONNX, "onnx not installed")
class TestControllerNeuralLSTM(unittest.TestCase):
    """ControllerNeuralLSTM — load via model_path, call compute() directly."""

    def setUp(self):
        self.device = wp.get_device()
        self._tmp_dir = tempfile.mkdtemp()

    def _save_lstm(self, filename="lstm.onnx", hidden=8, metadata=None):
        path = os.path.join(self._tmp_dir, filename)
        _build_lstm_onnx(path, hidden_size=hidden, num_layers=1, metadata=metadata)
        return path

    def _run_lstm_compute(self, ctrl):
        """Run a single compute step and verify output."""
        n = 1
        ctrl.finalize(self.device, n)

        state_a = ctrl.state(n, self.device)
        state_b = ctrl.state(n, self.device)
        np.testing.assert_array_equal(state_a.hidden.numpy(), 0.0)

        indices = wp.array([0], dtype=wp.uint32, device=self.device)
        positions = wp.zeros(n, dtype=wp.float32, device=self.device)
        velocities = wp.array([1.0], dtype=wp.float32, device=self.device)
        target_pos = wp.array([1.0], dtype=wp.float32, device=self.device)
        target_vel = wp.zeros(n, dtype=wp.float32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)

        ctrl.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            None,
            indices,
            indices,
            indices,
            indices,
            forces,
            state_a,
            0.01,
            self.device,
        )
        ctrl.update_state(state_a, state_b)

        self.assertNotAlmostEqual(forces.numpy()[0], 0.0, places=5, msg="LSTM should produce non-zero force")
        self.assertTrue(np.any(state_b.hidden.numpy() != 0.0), "hidden state should evolve")
        return forces.numpy()[0]

    def test_compute(self):
        """LSTM produces non-zero output; hidden state evolves after update_state."""
        path = self._save_lstm()
        ctrl = ControllerNeuralLSTM(model_path=path)
        self._run_lstm_compute(ctrl)

    def test_metadata_scales(self):
        """Scale factors from metadata are applied during compute."""
        metadata = {"pos_scale": 2.0, "vel_scale": 0.5, "effort_scale": 10.0}
        path = self._save_lstm(metadata=metadata)

        ctrl = ControllerNeuralLSTM(model_path=path)
        self.assertAlmostEqual(ctrl.pos_scale, 2.0)
        self.assertAlmostEqual(ctrl.vel_scale, 0.5)
        self.assertAlmostEqual(ctrl.effort_scale, 10.0)

        self._run_lstm_compute(ctrl)

    def test_graph_capture_matches_eager(self):
        """ONNX-backed ``compute`` is graph-safe; replays match eager outputs.

        Captures a single ``ctrl.compute`` call as a CUDA graph and replays it
        across a target schedule. Eager and graph paths must produce identical
        forces — this is what :meth:`ControllerNeuralLSTM.is_graphable` promises
        for the Warp-native ONNX runtime backend.
        """
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture not supported on CPU")

        path = self._save_lstm()
        n = 1
        target_schedule = [0.5, 1.0, 2.0, -1.5]

        def _setup():
            ctrl = ControllerNeuralLSTM(model_path=path)
            ctrl.finalize(self.device, n)
            return types.SimpleNamespace(
                ctrl=ctrl,
                state=ctrl.state(n, self.device),
                positions=wp.zeros(n, dtype=wp.float32, device=self.device),
                velocities=wp.array([1.0] * n, dtype=wp.float32, device=self.device),
                target_pos=wp.zeros(n, dtype=wp.float32, device=self.device),
                target_vel=wp.zeros(n, dtype=wp.float32, device=self.device),
                forces=wp.zeros(n, dtype=wp.float32, device=self.device),
                indices=wp.array([0], dtype=wp.uint32, device=self.device),
            )

        def _step(b):
            b.forces.zero_()
            b.ctrl.compute(
                b.positions,
                b.velocities,
                b.target_pos,
                b.target_vel,
                None,
                b.indices,
                b.indices,
                b.indices,
                b.indices,
                b.forces,
                b.state,
                0.01,
                self.device,
            )

        eager = _setup()
        self.assertTrue(eager.ctrl.is_graphable(), "ONNX-backed LSTM should be graphable")

        eager_forces = []
        for tgt in target_schedule:
            wp.copy(eager.target_pos, wp.full(n, tgt, dtype=wp.float32, device=self.device))
            _step(eager)
            eager_forces.append(eager.forces.numpy().copy())

        g = _setup()
        with wp.ScopedCapture(device=self.device) as capture:
            _step(g)
        graph = capture.graph

        graph_forces = []
        for tgt in target_schedule:
            wp.copy(g.target_pos, wp.full(n, tgt, dtype=wp.float32, device=self.device))
            wp.capture_launch(graph)
            graph_forces.append(g.forces.numpy().copy())

        for i, tgt in enumerate(target_schedule):
            np.testing.assert_allclose(
                graph_forces[i],
                eager_forces[i],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"target={tgt}: graph replay diverged from eager",
            )


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestControllerNeuralLSTMLegacyTorchScript(unittest.TestCase):
    """Regression tests for the deprecated ``.pt`` LSTM checkpoint path.

    ``ControllerNeuralLSTM`` was rewritten to load ``.onnx`` checkpoints
    backed by Newton's ONNX runtime, but legacy TorchScript / dict
    checkpoints created against the pre-ONNX API are still supported via
    ``_LegacyLstmTorchAdapter`` for one deprecation cycle.  These tests
    pin that contract: a legacy ``.pt`` checkpoint exposing a
    ``torch.nn.LSTM`` attribute named ``lstm`` (``batch_first=True``,
    ``input_size=2``) must continue to load, finalize, and run end-to-end.
    """

    def setUp(self):
        self.device = wp.get_device()
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _build_legacy_lstm_checkpoint(self, path: str, hidden_size: int = 4, metadata: dict | None = None):
        """Save a TorchScript LSTM module matching the pre-ONNX controller contract."""
        import torch

        class _LegacyLSTM(torch.nn.Module):
            """Minimal stateful LSTM matching the legacy ``ControllerNeuralLSTM`` API.

            Returns ``(effort, (h_new, c_new))`` from ``forward(net_input, (h, c))``
            where ``net_input`` has shape ``(N, 1, 2)`` (``batch_first=True``).  The
            explicit type annotation on ``hc`` is required for ``torch.jit.script``
            to compile the ``self.lstm(x, hc)`` overload selection.
            """

            def __init__(self, hidden_size: int):
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size=2,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                self.fc = torch.nn.Linear(hidden_size, 1, bias=True)
                with torch.no_grad():
                    self.fc.weight.fill_(0.5)
                    self.fc.bias.fill_(0.0)

            def forward(
                self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                y, hc_new = self.lstm(x, hc)
                effort = self.fc(y[:, -1, :])
                return effort, hc_new

        model = _LegacyLSTM(hidden_size).eval()
        scripted = torch.jit.script(model)
        extra_files = {"metadata.json": json.dumps(metadata or {})}
        scripted.save(path, _extra_files=extra_files)

    def test_load_emits_deprecation_warning(self):
        """Legacy ``.pt`` LSTM checkpoints emit a ``DeprecationWarning`` on load."""
        path = os.path.join(self._tmp_dir, "legacy_lstm.pt")
        self._build_legacy_lstm_checkpoint(path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ControllerNeuralLSTM(model_path=path)

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertTrue(deprecations, "expected a DeprecationWarning when loading a .pt LSTM checkpoint")
        self.assertIn(".pt", str(deprecations[0].message))

    def test_synthesizes_metadata_from_torch_module(self):
        """``num_layers`` / ``hidden_size`` are read from ``network.lstm`` when missing."""
        path = os.path.join(self._tmp_dir, "legacy_lstm.pt")
        hidden = 6
        self._build_legacy_lstm_checkpoint(path, hidden_size=hidden, metadata={"effort_scale": 2.5})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ctrl = ControllerNeuralLSTM(model_path=path)

        self.assertEqual(ctrl._num_layers, 1)
        self.assertEqual(ctrl._hidden_size, hidden)
        self.assertAlmostEqual(ctrl.effort_scale, 2.5)

    def test_finalize_and_compute(self):
        """Legacy ``.pt`` LSTM runs end-to-end through ``compute()`` / ``update_state()``.

        Mirrors :class:`TestControllerNeuralLSTM` ONNX coverage but exercises the
        ``_LegacyLstmTorchAdapter`` path so we catch any future regression that
        breaks legacy checkpoint loading before the deprecation window closes.
        """
        path = os.path.join(self._tmp_dir, "legacy_lstm.pt")
        self._build_legacy_lstm_checkpoint(path, hidden_size=4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ctrl = ControllerNeuralLSTM(model_path=path)

        n = 1
        ctrl.finalize(self.device, n)
        # ``is_graphable`` must be False because the adapter round-trips through host.
        self.assertFalse(ctrl.is_graphable())

        state_a = ctrl.state(n, self.device)
        state_b = ctrl.state(n, self.device)
        np.testing.assert_array_equal(state_a.hidden.numpy(), 0.0)

        indices = wp.array([0], dtype=wp.uint32, device=self.device)
        positions = wp.zeros(n, dtype=wp.float32, device=self.device)
        velocities = wp.array([1.0], dtype=wp.float32, device=self.device)
        target_pos = wp.array([1.0], dtype=wp.float32, device=self.device)
        target_vel = wp.zeros(n, dtype=wp.float32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)

        ctrl.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            None,
            indices,
            indices,
            indices,
            indices,
            forces,
            state_a,
            0.01,
            self.device,
        )
        ctrl.update_state(state_a, state_b)

        # The legacy module returns a non-zero effort for non-zero pos/vel error
        # and the LSTM hidden state should evolve away from zero.
        self.assertNotAlmostEqual(float(forces.numpy()[0]), 0.0, places=6)
        self.assertTrue(np.any(state_b.hidden.numpy() != 0.0))


class TestControllerPD(unittest.TestCase):
    """PD controller: f = constant + act + kp*(target_pos - q) + kd*(target_vel - v)."""

    def test_compute(self):
        """Construct controller directly and call compute() with all terms."""
        n = 2
        kp_vals = [100.0, 200.0]
        kd_vals = [10.0, 20.0]
        const_vals = [5.0, -3.0]
        q = [0.3, -0.5]
        qd = [1.0, -2.0]
        tgt_pos = [1.0, 0.5]
        tgt_vel = [0.0, 1.0]
        ff = [3.0, -1.0]

        def _f(vals):
            return wp.array(vals, dtype=wp.float32)

        indices = wp.array(list(range(n)), dtype=wp.uint32)
        ctrl = ControllerPD(kp=_f(kp_vals), kd=_f(kd_vals), const_effort=_f(const_vals))
        forces = wp.zeros(n, dtype=wp.float32)

        ctrl.compute(
            positions=_f(q),
            velocities=_f(qd),
            target_pos=_f(tgt_pos),
            target_vel=_f(tgt_vel),
            feedforward=_f(ff),
            pos_indices=indices,
            vel_indices=indices,
            target_pos_indices=indices,
            target_vel_indices=indices,
            forces=forces,
            state=None,
            dt=0.01,
        )

        result = forces.numpy()
        for i in range(n):
            expected = const_vals[i] + ff[i] + kp_vals[i] * (tgt_pos[i] - q[i]) + kd_vals[i] * (tgt_vel[i] - qd[i])
            self.assertAlmostEqual(result[i], expected, places=4, msg=f"DOF {i}")

    def _forward_with_tape(
        self,
        kp_vals: list[float],
        kd_vals: list[float],
        target_pos_vals: list[float],
        target_vel_vals: list[float],
        current_pos_vals: list[float],
        current_vel_vals: list[float],
        grad_param: str,
    ) -> tuple[np.ndarray, wp.array]:
        """Backprop ``sum(forces)`` from :meth:`ControllerPD.compute` w.r.t. ``grad_param``."""
        n = len(kp_vals)
        rg_kp = grad_param == "kp"
        rg_kd = grad_param == "kd"
        rg_tp = grad_param == "target_pos"
        rg_tv = grad_param == "target_vel"

        kp = _wp_array(kp_vals, requires_grad=rg_kp)
        kd = _wp_array(kd_vals, requires_grad=rg_kd)
        target_pos = _wp_array(target_pos_vals, requires_grad=rg_tp)
        target_vel = _wp_array(target_vel_vals, requires_grad=rg_tv)
        current_pos = _wp_array(current_pos_vals)
        current_vel = _wp_array(current_vel_vals)

        pos_idx = _wp_array(list(range(n)), dtype=wp.uint32)
        vel_idx = _wp_array(list(range(n)), dtype=wp.uint32)
        forces = wp.zeros(n, dtype=wp.float32, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        ctrl = ControllerPD(kp=kp, kd=kd)
        tape = wp.Tape()
        with tape:
            ctrl.compute(
                current_pos,
                current_vel,
                target_pos,
                target_vel,
                None,
                pos_idx,
                vel_idx,
                pos_idx,
                vel_idx,
                forces,
                None,
                dt=0.01,
            )
            wp.launch(_sum_kernel, dim=n, inputs=[forces], outputs=[loss])
        tape.backward(loss)

        grad_array = {"kp": kp, "kd": kd, "target_pos": target_pos, "target_vel": target_vel}[grad_param]
        return forces.numpy(), grad_array.grad

    def run_test_grad(self, grad_param: str):
        """Verify per-element analytic gradient against autodiff for ``grad_param``."""
        kp_v = [3.0, 5.0]
        kd_v = [0.5, 1.5]
        tp_v = [1.0, 2.0]
        tv_v = [0.1, -0.2]
        q_v = [0.25, 0.5]
        qd_v = [0.0, 0.1]

        _, grad = self._forward_with_tape(kp_v, kd_v, tp_v, tv_v, q_v, qd_v, grad_param)
        got = grad.numpy()

        expected_map = {
            "kp": [tp_v[i] - q_v[i] for i in range(len(kp_v))],
            "kd": [tv_v[i] - qd_v[i] for i in range(len(kp_v))],
            "target_pos": kp_v,
            "target_vel": kd_v,
        }
        np.testing.assert_allclose(got, expected_map[grad_param], rtol=1e-5)

    def test_grad_wrt_kp(self):
        self.run_test_grad("kp")

    def test_grad_wrt_kd(self):
        self.run_test_grad("kd")

    def test_grad_wrt_target_pos(self):
        self.run_test_grad("target_pos")

    def test_grad_wrt_target_vel(self):
        self.run_test_grad("target_vel")


class TestControllerPID(unittest.TestCase):
    """PID controller: f = const + act + kp*e + ki*integral + kd*de."""

    def test_compute(self):
        """Construct controller directly and call compute() over multiple steps."""
        kp, ki, kd, const = 50.0, 10.0, 5.0, 2.0
        dt = 0.01
        q, qd = [0.0], [0.0]
        tgt_pos, tgt_vel = [1.0], [0.0]
        pos_error = tgt_pos[0] - q[0]
        vel_error = tgt_vel[0] - qd[0]
        device = wp.get_device()

        def _f(vals):
            return wp.array(vals, dtype=wp.float32, device=device)

        indices = wp.array([0], dtype=wp.uint32, device=device)
        ctrl = ControllerPID(
            kp=_f([kp]),
            ki=_f([ki]),
            kd=_f([kd]),
            integral_max=_f([math.inf]),
            const_effort=_f([const]),
        )
        ctrl.finalize(device, 1)

        state_0 = ctrl.state(1, device)
        state_1 = ctrl.state(1, device)

        integral = 0.0
        for step_i in range(3):
            forces = wp.zeros(1, dtype=wp.float32, device=device)
            integral += pos_error * dt
            expected = const + kp * pos_error + ki * integral + kd * vel_error

            ctrl.compute(
                positions=_f(q),
                velocities=_f(qd),
                target_pos=_f(tgt_pos),
                target_vel=_f(tgt_vel),
                feedforward=None,
                pos_indices=indices,
                vel_indices=indices,
                target_pos_indices=indices,
                target_vel_indices=indices,
                forces=forces,
                state=state_0,
                dt=dt,
                device=device,
            )
            ctrl.update_state(state_0, state_1)
            state_0, state_1 = state_1, state_0

            self.assertAlmostEqual(forces.numpy()[0], expected, places=4, msg=f"step {step_i}")

    def _forward_with_tape(
        self,
        kp_vals: list[float],
        ki_vals: list[float],
        kd_vals: list[float],
        target_pos_vals: list[float],
        target_vel_vals: list[float],
        current_pos_vals: list[float],
        current_vel_vals: list[float],
        grad_param: str,
        dt: float = 0.01,
    ) -> tuple[np.ndarray, wp.array]:
        """Backprop ``sum(forces)`` from :meth:`ControllerPID.compute` w.r.t. ``grad_param``."""
        n = len(kp_vals)
        device = wp.get_device()
        rg = {p: (grad_param == p) for p in ("kp", "ki", "kd", "target_pos", "target_vel")}

        kp = _wp_array(kp_vals, requires_grad=rg["kp"], device=device)
        ki = _wp_array(ki_vals, requires_grad=rg["ki"], device=device)
        kd = _wp_array(kd_vals, requires_grad=rg["kd"], device=device)
        target_pos = _wp_array(target_pos_vals, requires_grad=rg["target_pos"], device=device)
        target_vel = _wp_array(target_vel_vals, requires_grad=rg["target_vel"], device=device)
        current_pos = _wp_array(current_pos_vals, device=device)
        current_vel = _wp_array(current_vel_vals, device=device)
        integral_max = _wp_array([math.inf] * n, device=device)

        idx = _wp_array(list(range(n)), dtype=wp.uint32, device=device)
        forces = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        ctrl = ControllerPID(kp=kp, ki=ki, kd=kd, integral_max=integral_max)
        ctrl.finalize(device, n)
        state = ctrl.state(n, device)

        tape = wp.Tape()
        with tape:
            ctrl.compute(
                current_pos,
                current_vel,
                target_pos,
                target_vel,
                None,
                idx,
                idx,
                idx,
                idx,
                forces,
                state,
                dt=dt,
                device=device,
            )
            wp.launch(_sum_kernel, dim=n, inputs=[forces], outputs=[loss])
        tape.backward(loss)

        grad_array = {"kp": kp, "ki": ki, "kd": kd, "target_pos": target_pos, "target_vel": target_vel}[grad_param]
        return forces.numpy(), grad_array.grad

    def run_test_grad(self, grad_param: str):
        """Verify per-element analytic gradient against autodiff for ``grad_param``."""
        kp_v = [3.0, 5.0]
        ki_v = [2.0, 4.0]
        kd_v = [0.5, 1.5]
        tp_v = [1.0, 2.0]
        tv_v = [0.1, -0.2]
        q_v = [0.25, 0.5]
        qd_v = [0.0, 0.1]
        dt = 0.01

        _, grad = self._forward_with_tape(kp_v, ki_v, kd_v, tp_v, tv_v, q_v, qd_v, grad_param, dt=dt)
        got = grad.numpy()

        expected_map = {
            "kp": [tp_v[i] - q_v[i] for i in range(len(kp_v))],
            "ki": [(tp_v[i] - q_v[i]) * dt for i in range(len(kp_v))],
            "kd": [tv_v[i] - qd_v[i] for i in range(len(kp_v))],
            "target_pos": [kp_v[i] + ki_v[i] * dt for i in range(len(kp_v))],
            "target_vel": kd_v,
        }
        np.testing.assert_allclose(got, expected_map[grad_param], rtol=1e-5)

    def test_grad_wrt_kp(self):
        self.run_test_grad("kp")

    def test_grad_wrt_ki(self):
        self.run_test_grad("ki")

    def test_grad_wrt_kd(self):
        self.run_test_grad("kd")

    def test_grad_wrt_target_pos(self):
        self.run_test_grad("target_pos")

    def test_grad_wrt_target_vel(self):
        self.run_test_grad("target_vel")

    def test_state_reset_masked(self):
        """PID integral accumulator: masked reset zeros selected DOFs only."""
        n = 3
        device = wp.get_device()

        def _f(vals):
            return wp.array(vals, dtype=wp.float32, device=device)

        indices = wp.array(list(range(n)), dtype=wp.uint32, device=device)
        ctrl = ControllerPID(
            kp=_f([50.0] * n),
            ki=_f([10.0] * n),
            kd=_f([5.0] * n),
            integral_max=_f([math.inf] * n),
            const_effort=_f([0.0] * n),
        )
        ctrl.finalize(device, n)

        state_0 = ctrl.state(n, device)
        state_1 = ctrl.state(n, device)

        for _ in range(3):
            forces = wp.zeros(n, dtype=wp.float32, device=device)
            ctrl.compute(
                positions=_f([0.0] * n),
                velocities=_f([0.0] * n),
                target_pos=_f([1.0] * n),
                target_vel=_f([0.0] * n),
                feedforward=None,
                pos_indices=indices,
                vel_indices=indices,
                target_pos_indices=indices,
                target_vel_indices=indices,
                forces=forces,
                state=state_0,
                dt=0.01,
                device=device,
            )
            ctrl.update_state(state_0, state_1)
            state_0, state_1 = state_1, state_0

        integral_before = state_0.integral.numpy().copy()
        self.assertTrue(all(v > 0 for v in integral_before), "integrals should have accumulated")

        mask = wp.array([True, False, True], dtype=wp.bool, device=device)
        state_0.reset(mask)

        integral_after = state_0.integral.numpy()
        self.assertAlmostEqual(integral_after[0], 0.0, places=6, msg="DOF 0 should be reset")
        self.assertAlmostEqual(integral_after[1], integral_before[1], places=6, msg="DOF 1 should be untouched")
        self.assertAlmostEqual(integral_after[2], 0.0, places=6, msg="DOF 2 should be reset")


class TestDelayedActuator(unittest.TestCase):
    """Actuator with command-input delay — delay buffer, reset, graph capture, and gradients."""

    def test_buffer_shape(self):
        """State buffers have correct shape (buf_depth, N)."""
        n, max_delay = 2, 5
        device = wp.get_device()
        delays = wp.array([max_delay] * n, dtype=wp.int32, device=device)
        delay = Delay(delay_steps=delays, max_delay=max_delay)
        delay.finalize(device, n)

        ds = delay.state(n, device)
        self.assertEqual(ds.buffer_pos.shape, (max_delay, n))
        self.assertEqual(ds.buffer_vel.shape, (max_delay, n))
        self.assertEqual(ds.buffer_act.shape, (max_delay, n))
        self.assertEqual(ds.write_idx.numpy()[0], max_delay - 1)
        np.testing.assert_array_equal(ds.num_pushes.numpy(), [0, 0])

    def test_latency(self):
        """Delay=N gives exactly N steps of delay; empty buffer falls back to current targets."""
        n, delay_val = 1, 2
        device = wp.get_device()
        delays = wp.array([delay_val], dtype=wp.int32, device=device)
        delay = Delay(delay_steps=delays, max_delay=delay_val)
        delay.finalize(device, n)

        indices = wp.array([0], dtype=wp.uint32, device=device)
        state_0 = delay.state(n, device)
        state_1 = delay.state(n, device)

        read_history = []
        for step_i in range(delay_val + 3):
            target_val = float(step_i + 1) * 10.0
            tgt_pos = wp.array([target_val], dtype=wp.float32, device=device)
            tgt_vel = wp.zeros(1, dtype=wp.float32, device=device)

            out_pos, _out_vel, _out_act = delay.get_delayed_targets(tgt_pos, tgt_vel, None, indices, indices, state_0)
            read_history.append(out_pos.numpy()[0])
            delay.update_state(tgt_pos, tgt_vel, None, indices, indices, state_0, state_1)
            state_0, state_1 = state_1, state_0

        self.assertAlmostEqual(read_history[0], 10.0, places=4, msg="step 0: empty buffer -> current target")
        self.assertAlmostEqual(read_history[1], 10.0, places=4, msg="step 1: 1 entry, lag clamped -> oldest (10)")
        self.assertAlmostEqual(read_history[2], 10.0, places=4, msg="step 2: full delay=2 -> reads step 0 (10)")
        self.assertAlmostEqual(read_history[3], 20.0, places=4, msg="step 3: full delay=2 -> reads step 1 (20)")
        self.assertAlmostEqual(read_history[4], 30.0, places=4, msg="step 4: full delay=2 -> reads step 2 (30)")

    def test_mixed_delay_zero_and_nonzero(self):
        """delay=0 DOFs pass through current targets; delay=1 DOFs lag by one step."""
        n = 2
        device = wp.get_device()
        delays = wp.array([0, 1], dtype=wp.int32, device=device)
        delay = Delay(delay_steps=delays, max_delay=1)
        delay.finalize(device, n)

        indices = wp.array([0, 1], dtype=wp.uint32, device=device)
        state_0 = delay.state(n, device)
        state_1 = delay.state(n, device)

        history_dof0 = []
        history_dof1 = []
        for step_i in range(4):
            target_val = float(step_i + 1) * 10.0
            tgt_pos = wp.array([target_val, target_val], dtype=wp.float32, device=device)
            tgt_vel = wp.zeros(n, dtype=wp.float32, device=device)

            out_pos, _, _ = delay.get_delayed_targets(tgt_pos, tgt_vel, None, indices, indices, state_0)
            result = out_pos.numpy()
            history_dof0.append(result[0])
            history_dof1.append(result[1])
            delay.update_state(tgt_pos, tgt_vel, None, indices, indices, state_0, state_1)
            state_0, state_1 = state_1, state_0

        self.assertAlmostEqual(history_dof0[0], 10.0, places=4, msg="dof0 step 0")
        self.assertAlmostEqual(history_dof0[1], 20.0, places=4, msg="dof0 step 1")
        self.assertAlmostEqual(history_dof0[2], 30.0, places=4, msg="dof0 step 2")
        self.assertAlmostEqual(history_dof0[3], 40.0, places=4, msg="dof0 step 3")

        self.assertAlmostEqual(history_dof1[0], 10.0, places=4, msg="dof1 step 0: empty -> current")
        self.assertAlmostEqual(history_dof1[1], 10.0, places=4, msg="dof1 step 1: reads step 0 (10)")
        self.assertAlmostEqual(history_dof1[2], 20.0, places=4, msg="dof1 step 2: reads step 1 (20)")
        self.assertAlmostEqual(history_dof1[3], 30.0, places=4, msg="dof1 step 3: reads step 2 (30)")

    def test_state_reset_masked(self):
        """Push data into 4-DOF delay buffer, reset DOFs 1 and 3, verify others untouched."""
        n, max_delay = 4, 2
        device = wp.get_device()
        delays = wp.array([max_delay] * n, dtype=wp.int32, device=device)
        delay = Delay(delay_steps=delays, max_delay=max_delay)
        delay.finalize(device, n)

        state_0 = delay.state(n, device)
        state_1 = delay.state(n, device)
        indices = wp.array(list(range(n)), dtype=wp.uint32, device=device)

        for step in range(3):
            tgt = wp.array([float(step + 1) * 10] * n, dtype=wp.float32, device=device)
            vel = wp.zeros(n, dtype=wp.float32, device=device)
            delay.update_state(tgt, vel, None, indices, indices, state_0, state_1)
            state_0, state_1 = state_1, state_0

        pushes_before = state_0.num_pushes.numpy().copy()
        self.assertTrue(all(p > 0 for p in pushes_before), "all DOFs should have data")

        mask = wp.array([False, True, False, True], dtype=wp.bool, device=device)
        state_0.reset(mask)

        pushes_after = state_0.num_pushes.numpy()
        self.assertEqual(pushes_after[0], pushes_before[0], "DOF 0 should be untouched")
        self.assertEqual(pushes_after[1], 0, "DOF 1 should be reset")
        self.assertEqual(pushes_after[2], pushes_before[2], "DOF 2 should be untouched")
        self.assertEqual(pushes_after[3], 0, "DOF 3 should be reset")

        buf_pos = state_0.buffer_pos.numpy()
        for row in range(max_delay):
            self.assertEqual(buf_pos[row, 1], 0.0, f"buffer_pos[{row}, 1] should be zeroed")
            self.assertEqual(buf_pos[row, 3], 0.0, f"buffer_pos[{row}, 3] should be zeroed")
            self.assertNotEqual(buf_pos[row, 0], 0.0, f"buffer_pos[{row}, 0] should be preserved")

    def test_state_reset_full(self):
        """Full reset (mask=None) zeros everything and resets write_idx."""
        n, max_delay = 2, 3
        device = wp.get_device()
        delays = wp.array([max_delay] * n, dtype=wp.int32, device=device)
        delay = Delay(delay_steps=delays, max_delay=max_delay)
        delay.finalize(device, n)

        state = delay.state(n, device)
        indices = wp.array(list(range(n)), dtype=wp.uint32, device=device)
        state_tmp = delay.state(n, device)

        for step in range(4):
            tgt = wp.array([float(step + 1)] * n, dtype=wp.float32, device=device)
            vel = wp.zeros(n, dtype=wp.float32, device=device)
            delay.update_state(tgt, vel, None, indices, indices, state, state_tmp)
            state, state_tmp = state_tmp, state

        self.assertTrue(any(p > 0 for p in state.num_pushes.numpy()))

        state.reset()

        np.testing.assert_array_equal(state.num_pushes.numpy(), [0] * n)
        np.testing.assert_array_equal(state.buffer_pos.numpy(), np.zeros((max_delay, n)))
        np.testing.assert_array_equal(state.buffer_vel.numpy(), np.zeros((max_delay, n)))
        np.testing.assert_array_equal(state.buffer_act.numpy(), np.zeros((max_delay, n)))
        self.assertEqual(state.write_idx.numpy()[0], max_delay - 1)

    def test_state_reset_composed(self):
        """Actuator.State.reset (PID + Delay) delegates to both delay buffer and PID integral."""
        num_envs = 2
        device = wp.get_device()

        template = newton.ModelBuilder()
        link = template.add_link()
        joint = template.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
        template.add_articulation([joint])
        dof = template.joint_qd_start[joint]
        template.add_actuator(ControllerPID, index=dof, kp=50.0, ki=10.0, kd=5.0, delay_steps=2)

        builder = newton.ModelBuilder()
        builder.replicate(template, num_envs)
        model = builder.finalize()

        actuator = model.actuators[0]
        n = actuator.num_actuators
        self.assertEqual(n, num_envs)

        state = model.state()
        state_0 = actuator.state()
        state_1 = actuator.state()
        dofs = actuator.indices.numpy().tolist()

        control = model.control()
        for _step in range(3):
            _write_dof_values(model, control.joint_target_q, dofs, [10.0] * n)
            control.joint_f.zero_()
            actuator.step(state, control, state_0, state_1, 0.01)
            state_0, state_1 = state_1, state_0

        self.assertTrue(all(p > 0 for p in state_0.delay_state.num_pushes.numpy()))
        self.assertTrue(all(v > 0 for v in state_0.controller_state.integral.numpy()))

        mask = wp.array([True, False], dtype=wp.bool, device=device)
        state_0.reset(mask)

        self.assertEqual(state_0.delay_state.num_pushes.numpy()[0], 0, "env 0 delay should be reset")
        self.assertGreater(state_0.delay_state.num_pushes.numpy()[1], 0, "env 1 delay should be untouched")
        self.assertAlmostEqual(
            state_0.controller_state.integral.numpy()[0], 0.0, places=6, msg="env 0 integral should be reset"
        )
        self.assertGreater(state_0.controller_state.integral.numpy()[1], 0.0, msg="env 1 integral should be untouched")

    def _forward_with_tape(
        self,
        kp_vals: list[float],
        kd_vals: list[float],
        target_pos_vals: list[float],
        target_vel_vals: list[float],
        current_pos_vals: list[float],
        current_vel_vals: list[float],
        grad_param: str,
        delay_steps: int = 2,
    ) -> tuple[np.ndarray, wp.array]:
        """Backprop ``sum(joint_f)`` from :meth:`Actuator.step` (PD + Delay + MaxEffort).

        Single step with empty delay buffer ⇒ delay reads fall back to current
        targets; ``max_effort=1e3`` ⇒ clamping is identity. Pipeline reduces
        to the bare PD controller, exposing the same analytic gradients.
        """
        n = len(kp_vals)
        device = wp.get_device()
        rg = {p: (grad_param == p) for p in ("kp", "kd", "target_pos", "target_vel")}

        kp = _wp_array(kp_vals, requires_grad=rg["kp"], device=device)
        kd = _wp_array(kd_vals, requires_grad=rg["kd"], device=device)
        target_pos = _wp_array(target_pos_vals, requires_grad=rg["target_pos"], device=device)
        target_vel = _wp_array(target_vel_vals, requires_grad=rg["target_vel"], device=device)
        current_pos = _wp_array(current_pos_vals, device=device)
        current_vel = _wp_array(current_vel_vals, device=device)

        ctrl = ControllerPD(kp=kp, kd=kd)
        clamp = ClampingMaxEffort(max_effort=_wp_array([1e3] * n, device=device))
        delay_arr = wp.array([delay_steps] * n, dtype=wp.int32, device=device)
        delay = Delay(delay_steps=delay_arr, max_delay=delay_steps)
        indices = _wp_array(list(range(n)), dtype=wp.uint32, device=device)

        actuator = Actuator(
            indices=indices,
            controller=ctrl,
            delay=delay,
            clamping=[clamp],
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
            requires_grad=True,
        )
        state_a = actuator.state()
        state_b = actuator.state()

        sim_state = types.SimpleNamespace(joint_q=current_pos, joint_qd=current_vel)
        sim_control = types.SimpleNamespace(
            joint_target_q=target_pos,
            joint_target_qd=target_vel,
            joint_act=None,
            joint_f=wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True),
        )

        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            actuator.step(sim_state, sim_control, state_a, state_b, dt=0.01)
            wp.launch(_sum_kernel, dim=n, inputs=[sim_control.joint_f], outputs=[loss])
        tape.backward(loss)

        grad_array = {"kp": kp, "kd": kd, "target_pos": target_pos, "target_vel": target_vel}[grad_param]
        return sim_control.joint_f.numpy(), grad_array.grad

    def run_test_grad(self, grad_param: str):
        """Verify analytic gradient against autodiff for one step through ``Actuator.step``."""
        kp_v = [3.0, 5.0]
        kd_v = [0.5, 1.5]
        tp_v = [1.0, 2.0]
        tv_v = [0.1, -0.2]
        q_v = [0.25, 0.5]
        qd_v = [0.0, 0.1]

        _, grad = self._forward_with_tape(kp_v, kd_v, tp_v, tv_v, q_v, qd_v, grad_param)
        got = grad.numpy()

        expected_map = {
            "kp": [tp_v[i] - q_v[i] for i in range(len(kp_v))],
            "kd": [tv_v[i] - qd_v[i] for i in range(len(kp_v))],
            "target_pos": kp_v,
            "target_vel": kd_v,
        }
        np.testing.assert_allclose(got, expected_map[grad_param], rtol=1e-5)

    def test_grad_wrt_kp(self):
        self.run_test_grad("kp")

    def test_grad_wrt_kd(self):
        self.run_test_grad("kd")

    def test_grad_wrt_target_pos(self):
        self.run_test_grad("target_pos")

    def test_grad_wrt_target_vel(self):
        self.run_test_grad("target_vel")

    @unittest.skipUnless(
        wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()),
        "CUDA graph capture requires CUDA + mempool",
    )
    def test_graph_capture_matches_eager(self):
        """``Actuator(PD + Delay + MaxEffort)`` is graph-safe with device-side ``write_idx``.

        N=2, buf_depth=5: capture N actuator + K physics substeps as a CUDA
        graph and replay with varying targets. With ``N < buf_depth`` and
        ``N % buf_depth != 0``, this configuration previously failed when
        ``write_idx`` was a host-side scalar baked into the graph.
        """
        max_delay = 5
        N = 2  # 2 % 5 != 0, N < buf_depth, N is even
        K = 2
        dt = 0.02
        warmup_target = 0.0
        cycle_targets = [2.0, -3.0, 5.0, -1.0]

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.density = 1000.0
        link = builder.add_link()
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
        builder.add_shape_sphere(body=link, radius=0.1)
        builder.add_articulation([joint])
        dof = builder.joint_qd_start[joint]
        builder.add_actuator(
            ControllerPD,
            index=dof,
            kp=200.0,
            kd=10.0,
            delay_steps=max_delay,
            clamping=[(ClampingMaxEffort, {"max_effort": 500.0})],
        )
        model = builder.finalize()
        device = model.device
        ndof = model.joint_coord_count

        def _setup():
            solver = newton.solvers.SolverMuJoCo(model, iterations=4, ls_iterations=4)
            s0 = model.state()
            s1 = model.state()
            ctrl = model.control()
            newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
            act = model.actuators[0]
            act_a, act_b = act.state(), act.state()
            return solver, s0, s1, ctrl, act, act_a, act_b

        def _loop(solver, s0, s1, ctrl, act, act_a, act_b, n):
            sub_dt = dt / K
            for _ in range(n):
                ctrl.joint_f.zero_()
                act.step(s0, ctrl, act_a, act_b, dt=dt)
                act_a, act_b = act_b, act_a
                for _ in range(K):
                    s0.clear_forces()
                    solver.step(s0, s1, ctrl, None, sub_dt)
                    s0, s1 = s1, s0
            return s0, s1, act_a, act_b

        solver, s0, s1, ctrl, act, act_a, act_b = _setup()
        wp.copy(ctrl.joint_target_q, wp.full(ndof, warmup_target, dtype=wp.float32, device=device))
        s0, s1, act_a, act_b = _loop(solver, s0, s1, ctrl, act, act_a, act_b, max_delay)
        eager_results = []
        for tgt in cycle_targets:
            wp.copy(ctrl.joint_target_q, wp.full(ndof, tgt, dtype=wp.float32, device=device))
            s0, s1, act_a, act_b = _loop(solver, s0, s1, ctrl, act, act_a, act_b, N)
            eager_results.append(s0.joint_q.numpy().copy())

        solver_g, s0_g, s1_g, ctrl_g, act_g, act_a_g, act_b_g = _setup()
        wp.copy(ctrl_g.joint_target_q, wp.full(ndof, warmup_target, dtype=wp.float32, device=device))
        s0_g, s1_g, act_a_g, act_b_g = _loop(solver_g, s0_g, s1_g, ctrl_g, act_g, act_a_g, act_b_g, max_delay)
        sub_dt = dt / K
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(N):
                ctrl_g.joint_f.zero_()
                act_g.step(s0_g, ctrl_g, act_a_g, act_b_g, dt=dt)
                act_a_g, act_b_g = act_b_g, act_a_g
                for _ in range(K):
                    s0_g.clear_forces()
                    solver_g.step(s0_g, s1_g, ctrl_g, None, sub_dt)
                    s0_g, s1_g = s1_g, s0_g
        graph = capture.graph

        graph_results = []
        for tgt in cycle_targets:
            wp.copy(ctrl_g.joint_target_q, wp.full(ndof, tgt, dtype=wp.float32, device=device))
            wp.capture_launch(graph)
            graph_results.append(s0_g.joint_q.numpy().copy())

        for ci in range(len(cycle_targets)):
            np.testing.assert_allclose(
                graph_results[ci],
                eager_results[ci],
                rtol=1e-4,
                err_msg=f"Cycle {ci}: graph should match eager with device-side write_idx",
            )


# -------------------------------------------------------------------------
# 2. Unit clamping
# -------------------------------------------------------------------------


class TestClamping(unittest.TestCase):
    """Clamping components — direct ``modify_forces`` formula checks on a single DOF."""

    def run_test_clamp(self, clamp, samples: list[tuple[float, float, float, float]]):
        """Assert ``modify_forces`` matches ``expected`` for each ``(force, pos, vel, expected)`` sample."""
        device = wp.get_device()
        clamp.finalize(device, 1)
        indices = wp.array([0], dtype=wp.uint32, device=device)

        for force_in, pos, vel, expected in samples:
            src = wp.array([force_in], dtype=wp.float32, device=device)
            dst = wp.zeros(1, dtype=wp.float32, device=device)
            positions = wp.array([pos], dtype=wp.float32, device=device)
            velocities = wp.array([vel], dtype=wp.float32, device=device)
            clamp.modify_forces(src, dst, positions, velocities, indices, indices)
            self.assertAlmostEqual(
                dst.numpy()[0],
                expected,
                places=3,
                msg=f"force={force_in}, pos={pos}, vel={vel}",
            )

    def test_max_effort(self):
        """``ClampingMaxEffort`` clamps to ``±max_effort`` regardless of pos/vel."""
        max_f = 50.0
        clamp = ClampingMaxEffort(max_effort=wp.array([max_f], dtype=wp.float32))
        self.run_test_clamp(
            clamp,
            [
                (100.0, 0.0, 0.0, max_f),
                (-80.0, 0.0, 0.0, -max_f),
                (30.0, 0.0, 0.0, 30.0),
            ],
        )

    def test_dc_motor(self):
        """DC motor torque-speed curve: ``clamp = saturation * (1 - v / v_limit)``."""
        sat, v_lim, max_f = 100.0, 10.0, 200.0
        clamp = ClampingDCMotor(
            saturation_effort=wp.array([sat], dtype=wp.float32),
            velocity_limit=wp.array([v_lim], dtype=wp.float32),
            max_motor_effort=wp.array([max_f], dtype=wp.float32),
        )
        raw = 500.0

        def expected(qd: float) -> float:
            tau_max = min(sat * (1.0 - qd / v_lim), max_f)
            tau_min = max(sat * (-1.0 - qd / v_lim), -max_f)
            return max(min(raw, tau_max), tau_min)

        self.run_test_clamp(clamp, [(raw, 0.0, qd, expected(qd)) for qd in [0.0, 5.0, 10.0, -5.0]])

    def test_position_based(self):
        """Position-based lookup table, linearly interpolated between sample points."""
        clamp = ClampingPositionBased(lookup_positions=(-1.0, 0.0, 1.0), lookup_efforts=(10.0, 30.0, 50.0))
        raw = 999.0
        self.run_test_clamp(
            clamp,
            [(raw, p, 0.0, e) for p, e in [(-1.0, 10.0), (0.0, 30.0), (1.0, 50.0), (-0.5, 20.0), (0.5, 40.0)]],
        )


# -------------------------------------------------------------------------
# 3. Legacy newton_actuators backward compatibility
# -------------------------------------------------------------------------


class TestLegacyActuatorCompat(unittest.TestCase):
    """Deprecated ``newton_actuators`` calling conventions must still work."""

    def _make_builder(self, n_joints=2):
        builder = newton.ModelBuilder()
        links = [builder.add_link() for _ in range(n_joints)]
        joints = []
        for i, link in enumerate(links):
            parent = -1 if i == 0 else links[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=link, axis=newton.Axis.Z))
        builder.add_articulation(joints)
        dofs = [builder.joint_qd_start[j] for j in joints]
        return builder, dofs

    def test_legacy_positional_list(self):
        """add_actuator(ActuatorPD, [dof], kp=...) — old positional style."""
        builder, dofs = self._make_builder()
        with self.assertWarns(DeprecationWarning):
            builder.add_actuator(ActuatorPD, [dofs[0]], kp=50.0, kd=5.0)
        model = builder.finalize()
        self.assertEqual(len(model.actuators), 1)
        self.assertIsInstance(model.actuators[0].controller, ControllerPD)
        np.testing.assert_array_almost_equal(model.actuators[0].controller.kp.numpy(), [50.0])

    def test_legacy_keyword_input_indices(self):
        """add_actuator(ActuatorPD, input_indices=[dof], kp=...)."""
        builder, dofs = self._make_builder()
        with self.assertWarns(DeprecationWarning):
            builder.add_actuator(ActuatorPD, input_indices=[dofs[0]], kp=100.0)
        model = builder.finalize()
        self.assertEqual(len(model.actuators), 1)
        np.testing.assert_array_almost_equal(model.actuators[0].controller.kp.numpy(), [100.0])

    def test_legacy_keyword_actuator_class(self):
        """add_actuator(actuator_class=ActuatorPD, input_indices=[dof], kp=...)."""
        builder, dofs = self._make_builder()
        with self.assertWarns(DeprecationWarning):
            builder.add_actuator(actuator_class=ActuatorPD, input_indices=[dofs[0]], kp=75.0)
        model = builder.finalize()
        self.assertEqual(len(model.actuators), 1)
        np.testing.assert_array_almost_equal(model.actuators[0].controller.kp.numpy(), [75.0])

    def test_legacy_delayed_pd(self):
        """ActuatorDelayedPD maps to ControllerPD + delay."""
        builder, dofs = self._make_builder()
        with self.assertWarns(DeprecationWarning):
            builder.add_actuator(ActuatorDelayedPD, input_indices=[dofs[0]], kp=200.0, delay=3)
        model = builder.finalize()
        act = model.actuators[0]
        self.assertIsInstance(act.controller, ControllerPD)
        self.assertIsNotNone(act.delay)
        np.testing.assert_array_equal(act.delay.delay_steps.numpy(), [3])

    def test_legacy_pid(self):
        """ActuatorPID maps to ControllerPID."""
        builder, dofs = self._make_builder()
        with self.assertWarns(DeprecationWarning):
            builder.add_actuator(ActuatorPID, [dofs[0]], kp=100.0, ki=10.0, kd=20.0)
        model = builder.finalize()
        act = model.actuators[0]
        self.assertIsInstance(act.controller, ControllerPID)
        np.testing.assert_array_almost_equal(act.controller.ki.numpy(), [10.0])

    def test_legacy_max_force_becomes_clamping(self):
        """max_force kwarg creates a ClampingMaxEffort on the new actuator."""
        builder, dofs = self._make_builder()
        with self.assertWarns(DeprecationWarning):
            builder.add_actuator(ActuatorPD, input_indices=[dofs[0]], kp=150.0, max_force=50.0)
        model = builder.finalize()
        act = model.actuators[0]
        self.assertEqual(len(act.clamping), 1)
        self.assertIsInstance(act.clamping[0], ClampingMaxEffort)
        np.testing.assert_array_almost_equal(act.clamping[0].max_effort.numpy(), [50.0])

    def test_legacy_output_indices_warns(self):
        """output_indices != input_indices emits extra deprecation warning."""
        builder, dofs = self._make_builder()
        with self.assertWarns(DeprecationWarning):
            builder.add_actuator(ActuatorPD, [dofs[0]], [dofs[1]], kp=50.0)

    def test_legacy_gear_warns(self):
        """Non-unity gear emits a deprecation warning."""
        builder, dofs = self._make_builder()
        with self.assertWarns(DeprecationWarning):
            builder.add_actuator(ActuatorPD, input_indices=[dofs[0]], kp=50.0, gear=2.0)

    def test_new_api_no_warning(self):
        """New-style calls must not emit DeprecationWarning."""
        builder, dofs = self._make_builder()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.add_actuator(ControllerPD, index=dofs[0], kp=50.0)
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(dep), 0, f"Unexpected warnings: {[str(x.message) for x in dep]}")


# -------------------------------------------------------------------------
# 4. Builder — programmatic actuator construction
# -------------------------------------------------------------------------


class TestActuatorBuilder(unittest.TestCase):
    """ModelBuilder actuator construction — grouping, params, state, and index layouts."""

    def test_programmatic(self):
        """Mixed controller types, clamping, and delays via add_actuator.

        3-joint chain: PD, PID with DC motor clamping, PD with delay=4.
        Verifies grouping (3 groups), per-DOF params, and state shapes.
        """
        builder = newton.ModelBuilder()
        links = [builder.add_link() for _ in range(3)]
        joints = []
        for i, link in enumerate(links):
            parent = -1 if i == 0 else links[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=link, axis=newton.Axis.Z))
        builder.add_articulation(joints)
        dofs = [builder.joint_qd_start[j] for j in joints]

        builder.add_actuator(ControllerPD, index=dofs[0], kp=50.0, kd=5.0, const_effort=1.0)
        builder.add_actuator(
            ControllerPID,
            index=dofs[1],
            kp=100.0,
            ki=10.0,
            kd=20.0,
            clamping=[
                (ClampingDCMotor, {"saturation_effort": 80.0, "velocity_limit": 15.0, "max_motor_effort": 200.0})
            ],
        )
        builder.add_actuator(ControllerPD, index=dofs[2], kp=150.0, delay_steps=4)

        model = builder.finalize()
        self.assertEqual(len(model.actuators), 3)

        pd_plain = next(a for a in model.actuators if isinstance(a.controller, ControllerPD) and a.delay is None)
        pid_act = next(a for a in model.actuators if isinstance(a.controller, ControllerPID))
        pd_delay = next(a for a in model.actuators if isinstance(a.controller, ControllerPD) and a.delay is not None)

        self.assertEqual(pd_plain.num_actuators, 1)
        np.testing.assert_array_almost_equal(pd_plain.controller.kp.numpy(), [50.0])
        np.testing.assert_array_almost_equal(pd_plain.controller.kd.numpy(), [5.0])
        np.testing.assert_array_almost_equal(pd_plain.controller.const_effort.numpy(), [1.0])
        self.assertIsNone(pd_plain.state())

        self.assertEqual(pid_act.num_actuators, 1)
        np.testing.assert_array_almost_equal(pid_act.controller.kp.numpy(), [100.0])
        np.testing.assert_array_almost_equal(pid_act.controller.ki.numpy(), [10.0])
        np.testing.assert_array_almost_equal(pid_act.controller.kd.numpy(), [20.0])
        self.assertIsInstance(pid_act.clamping[0], ClampingDCMotor)
        self.assertAlmostEqual(pid_act.clamping[0].saturation_effort.numpy()[0], 80.0, places=3)
        self.assertAlmostEqual(pid_act.clamping[0].max_motor_effort.numpy()[0], 200.0, places=3)
        pid_state = pid_act.state()
        self.assertIsNotNone(pid_state.controller_state)
        self.assertEqual(pid_state.controller_state.integral.shape, (1,))
        np.testing.assert_array_equal(pid_state.controller_state.integral.numpy(), [0.0])

        self.assertEqual(pd_delay.num_actuators, 1)
        np.testing.assert_array_almost_equal(pd_delay.controller.kp.numpy(), [150.0])
        np.testing.assert_array_equal(pd_delay.delay.delay_steps.numpy(), [4])
        self.assertEqual(pd_delay.delay.buf_depth, 4)
        ds = pd_delay.state().delay_state
        self.assertEqual(ds.buffer_pos.shape, (4, 1))
        np.testing.assert_array_equal(ds.num_pushes.numpy(), [0])

    def test_free_joint_with_replication(self):
        """Free-joint base + 2 revolute children x 3 envs.

        Verifies:
        - pos_indices != indices when joint_q layout differs from joint_qd
        - Correct per-DOF parameter replication across environments
        - State shapes scale with num_envs
        """
        num_envs = 3

        template = newton.ModelBuilder()
        base = template.add_link()
        j_free = template.add_joint_free(child=base)
        link1 = template.add_link()
        j1 = template.add_joint_revolute(parent=base, child=link1, axis=newton.Axis.Z)
        link2 = template.add_link()
        j2 = template.add_joint_revolute(parent=link1, child=link2, axis=newton.Axis.Y)
        template.add_articulation([j_free, j1, j2])

        dof1 = template.joint_qd_start[j1]
        dof2 = template.joint_qd_start[j2]

        template.add_actuator(
            ControllerPD, index=dof1, kp=100.0, kd=10.0, pos_index=template.joint_q_start[j1], delay_steps=2
        )
        template.add_actuator(
            ControllerPD, index=dof2, kp=200.0, kd=20.0, pos_index=template.joint_q_start[j2], delay_steps=3
        )

        builder = newton.ModelBuilder()
        builder.replicate(template, num_envs)
        model = builder.finalize()

        self.assertEqual(len(model.actuators), 1)
        act = model.actuators[0]
        n = 2 * num_envs
        self.assertEqual(act.num_actuators, n)

        pos_idx = act.pos_indices.numpy()
        vel_idx = act.indices.numpy()
        self.assertFalse(
            np.array_equal(pos_idx, vel_idx),
            "pos_indices should differ from indices for free-joint articulations",
        )

        np.testing.assert_array_almost_equal(act.controller.kp.numpy(), [100.0, 200.0] * num_envs)
        np.testing.assert_array_almost_equal(act.controller.kd.numpy(), [10.0, 20.0] * num_envs)

        np.testing.assert_array_equal(act.delay.delay_steps.numpy(), [2, 3] * num_envs)
        self.assertEqual(act.delay.buf_depth, 3)

        act_state = act.state()
        self.assertEqual(act_state.delay_state.buffer_pos.shape, (3, n))
        np.testing.assert_array_equal(act_state.delay_state.num_pushes.numpy(), [0] * n)


# -------------------------------------------------------------------------
# 5. Parameter access via ArticulationView
# -------------------------------------------------------------------------


class TestActuatorSelectionAPI(unittest.TestCase):
    """Tests for actuator parameter access via ArticulationView."""

    def run_test_actuator_selection(self, use_mask: bool, use_multiple_artics_per_view: bool):
        mjcf = """<?xml version="1.0" ?>
<mujoco model="myart">
    <worldbody>
    <body name="root" pos="0 0 0">
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
      <body name="link3" pos="-0.0 -0.9 0">
        <joint name="joint3" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

        num_joints_per_articulation = 3
        num_articulations_per_world = 2
        num_worlds = 3
        num_actuators = num_joints_per_articulation * num_articulations_per_world * num_worlds

        single_articulation_builder = newton.ModelBuilder()
        single_articulation_builder.add_mjcf(mjcf)

        joint_names = [
            "myart/worldbody/root/link1/joint1",
            "myart/worldbody/root/link2/joint2",
            "myart/worldbody/root/link3/joint3",
        ]
        for i, jname in enumerate(joint_names):
            j_idx = single_articulation_builder.joint_label.index(jname)
            dof = single_articulation_builder.joint_qd_start[j_idx]
            single_articulation_builder.add_actuator(ControllerPD, index=dof, kp=100.0 * (i + 1))

        single_world_builder = newton.ModelBuilder()
        for _i in range(num_articulations_per_world):
            single_world_builder.add_builder(single_articulation_builder)

        single_world_builder.articulation_label[1] = "art1"
        if use_multiple_artics_per_view:
            single_world_builder.articulation_label[0] = "art1"
        else:
            single_world_builder.articulation_label[0] = "art0"

        builder = newton.ModelBuilder()
        for _i in range(num_worlds):
            builder.add_world(single_world_builder)

        model = builder.finalize()

        joints_to_include = ["joint3"]
        joint_view = ArticulationView(model, "art1", include_joints=joints_to_include)

        actuator = model.actuators[0]

        kp_values = joint_view.get_actuator_parameter(actuator, actuator.controller, "kp").numpy().copy()

        if use_multiple_artics_per_view:
            self.assertEqual(kp_values.shape, (num_worlds, 2))
            np.testing.assert_array_almost_equal(kp_values, [[300.0, 300.0]] * num_worlds)
        else:
            self.assertEqual(kp_values.shape, (num_worlds, 1))
            np.testing.assert_array_almost_equal(kp_values, [[300.0]] * num_worlds)

        val = 1000.0
        for world_idx in range(kp_values.shape[0]):
            for dof_idx in range(kp_values.shape[1]):
                kp_values[world_idx, dof_idx] = val
                val += 100.0

        mask = None
        if use_mask:
            mask = wp.array([False, True, False], dtype=bool, device=model.device)

        wp_kp = wp.array(kp_values, dtype=float, device=model.device)
        joint_view.set_actuator_parameter(actuator, actuator.controller, "kp", wp_kp, mask=mask)

        expected_kp = []
        if use_mask:
            if use_multiple_artics_per_view:
                expected_kp = [
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    1200.0,
                    100.0,
                    200.0,
                    1300.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    300.0,
                ]
            else:
                expected_kp = [
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    1100.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    300.0,
                ]
        else:
            if use_multiple_artics_per_view:
                expected_kp = [
                    100.0,
                    200.0,
                    1000.0,
                    100.0,
                    200.0,
                    1100.0,
                    100.0,
                    200.0,
                    1200.0,
                    100.0,
                    200.0,
                    1300.0,
                    100.0,
                    200.0,
                    1400.0,
                    100.0,
                    200.0,
                    1500.0,
                ]
            else:
                expected_kp = [
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    1000.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    1100.0,
                    100.0,
                    200.0,
                    300.0,
                    100.0,
                    200.0,
                    1200.0,
                ]

        measured_kp = actuator.controller.kp.numpy()
        for i in range(num_actuators):
            self.assertAlmostEqual(
                expected_kp[i],
                measured_kp[i],
                places=4,
                msg=f"Expected kp[{i}]={expected_kp[i]}, got {measured_kp[i]}",
            )

    def test_actuator_selection_one_per_view_no_mask(self):
        self.run_test_actuator_selection(use_mask=False, use_multiple_artics_per_view=False)

    def test_actuator_selection_two_per_view_no_mask(self):
        self.run_test_actuator_selection(use_mask=False, use_multiple_artics_per_view=True)

    def test_actuator_selection_one_per_view_with_mask(self):
        self.run_test_actuator_selection(use_mask=True, use_multiple_artics_per_view=False)

    def test_actuator_selection_two_per_view_with_mask(self):
        self.run_test_actuator_selection(use_mask=True, use_multiple_artics_per_view=True)


# -------------------------------------------------------------------------
# 6. USD parsing (parse_actuator_prim)
# -------------------------------------------------------------------------


class TestUsdParsing(unittest.TestCase):
    """Exercise the ``NewtonActuator`` USD schema and
    :func:`parse_actuator_prim`.

    Covers every registered schema (PD / PID / Neural controllers,
    MaxEffort / DCMotor / PositionBased clamping, Delay), the
    asset-driven load path used by :func:`newton.ModelBuilder.add_usd`,
    and the documented :class:`ValueError` paths in
    :func:`parse_actuator_prim`.
    """

    _ACT_PATH = "/World/Robot/Actuator"
    _JOINT_PATH = "/World/Robot/Joint1"

    _SCHEMA_FOR: ClassVar[dict[type, str]] = {
        ControllerPD: SchemaNames.PD_CONTROL,
        ControllerPID: SchemaNames.PID_CONTROL,
        ControllerNeuralMLP: SchemaNames.NEURAL_CONTROL,
        ControllerNeuralLSTM: SchemaNames.NEURAL_CONTROL,
        ClampingMaxEffort: SchemaNames.MAX_EFFORT_CLAMPING,
        ClampingDCMotor: SchemaNames.DC_MOTOR_CLAMPING,
        ClampingPositionBased: SchemaNames.POSITION_BASED_CLAMPING,
        Delay: SchemaNames.DELAY,
    }
    """Component class ↦ ``apiSchemas`` token expected by :func:`parse_actuator_prim`."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _set_newton_attr(self, prim, name: str, value: Any) -> None:
        """Author ``newton:<name>`` with an :class:`Sdf.ValueTypeName` inferred
        from ``value``.

        Maps Python ``float``/``int``/``list[float]``/:class:`Sdf.AssetPath`
        to ``Float`` / ``Int`` / ``FloatArray`` / ``Asset`` respectively —
        the same SDF types used by the generated USD schema attributes.
        """
        from pxr import Sdf

        if isinstance(value, Sdf.AssetPath):
            type_name = Sdf.ValueTypeNames.Asset
        elif isinstance(value, bool):
            type_name = Sdf.ValueTypeNames.Bool
        elif isinstance(value, int):
            type_name = Sdf.ValueTypeNames.Int
        elif isinstance(value, list | tuple):
            type_name = Sdf.ValueTypeNames.FloatArray
        else:
            type_name = Sdf.ValueTypeNames.Float
        prim.CreateAttribute(f"newton:{name}", type_name).Set(value)

    def _build_actuator_stage(
        self,
        components: list[type],
        attrs: dict[str, Any] | None = None,
        *,
        target: str | None = "joint",
        joint_type: str = "revolute",
    ) -> "Usd.Stage":
        """Build a minimal in-memory stage with one ``NewtonActuator`` prim.

        Builds a base + Link1 articulation joined by a typed
        :class:`UsdPhysics.RevoluteJoint` (or ``PrismaticJoint``), then
        defines the actuator prim, applies the Newton API schemas
        corresponding to ``components`` via :meth:`Usd.Prim.ApplyAPI`,
        and authors the ``newton:*`` attributes.

        Args:
            components: Component classes (e.g. :class:`ControllerPD`,
                :class:`ClampingMaxEffort`) whose ``apiSchemas`` tokens
                should be applied to the actuator prim.
            attrs: Mapping from camelCase attribute suffix to a Python
                value; the SDF value type is inferred. The parser
                converts each suffix to snake_case kwargs.
            target: ``"joint"`` → revolute joint (default), ``"base"`` →
                non-joint prim (error path), ``None`` → leave unset.
            joint_type: ``"revolute"`` or ``"prismatic"``.

        Returns:
            The in-memory :class:`Usd.Stage`.
        """
        from pxr import UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

        robot = UsdGeom.Xform.Define(stage, "/World/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())

        base = UsdGeom.Xform.Define(stage, "/World/Robot/Base")
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim()).CreateKinematicEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr().Set(1.0)

        link = UsdGeom.Xform.Define(stage, "/World/Robot/Link1")
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr().Set(0.5)

        joint_cls = UsdPhysics.RevoluteJoint if joint_type == "revolute" else UsdPhysics.PrismaticJoint
        joint = joint_cls.Define(stage, self._JOINT_PATH)
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([link.GetPath()])
        joint.CreateAxisAttr().Set(UsdPhysics.Tokens.z)
        UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")

        act = stage.DefinePrim(self._ACT_PATH, SchemaNames.ACTUATOR)
        for cls in components:
            act.ApplyAPI(self._SCHEMA_FOR[cls])
        if target == "joint":
            act.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
        elif target == "base":
            act.CreateRelationship("newton:targets").SetTargets([base.GetPath()])
        for name, value in (attrs or {}).items():
            self._set_newton_attr(act, name, value)
        return stage

    def _make_mlp_checkpoint(self, metadata: dict | None = None) -> str:
        path = os.path.join(self._tmp_dir, "mlp.onnx")
        weights = np.zeros((1, 2), dtype=np.float32)
        bias = np.ones((1,), dtype=np.float32)
        _build_mlp_onnx(path, weights, bias, metadata)
        return path

    def _make_lstm_checkpoint(self, metadata: dict | None = None) -> str:
        path = os.path.join(self._tmp_dir, "lstm.onnx")
        _build_lstm_onnx(path, hidden_size=8, num_layers=1, metadata=metadata)
        return path

    def _build_neural_stage(self, model_path: str, controller_cls: type = ControllerNeuralMLP) -> "Usd.Stage":
        """Convenience wrapper for the neural-controller fixture."""
        from pxr import Sdf

        return self._build_actuator_stage(
            components=[controller_cls, ClampingDCMotor],
            attrs={
                "modelPath": Sdf.AssetPath(model_path),
                "saturationEffort": 100.0,
                "velocityLimit": 20.0,
                "maxMotorEffort": 200.0,
            },
        )

    # ---- Asset-driven end-to-end load ------------------------------------

    def test_load_from_fixture_usda(self):
        """Load actuators from a fixture USDA via ``newton.ModelBuilder``.

        The asset has two actuators:
          Joint1Actuator: PD (kp=100, kd=10) + MaxEffort(50)
          Joint2Actuator: PD (kp=200, kd=20) + Delay(5)
        Different clamping/delay splits them into separate groups.
        """
        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")
        if not os.path.exists(usd_path):
            self.skipTest(f"Test USD file not found: {usd_path}")

        builder = newton.ModelBuilder()
        result = parse_usd(builder, usd_path)
        self.assertGreater(result["actuator_count"], 0)
        model = builder.finalize()

        self.assertEqual(len(model.actuators), 2)
        clamped = next(a for a in model.actuators if a.clamping)
        delayed = next(a for a in model.actuators if a.delay is not None)

        self.assertEqual(clamped.num_actuators, 1)
        self.assertAlmostEqual(clamped.controller.kp.numpy()[0], 100.0, places=3)
        self.assertAlmostEqual(clamped.controller.kd.numpy()[0], 10.0, places=3)
        self.assertIsInstance(clamped.clamping[0], ClampingMaxEffort)
        self.assertAlmostEqual(clamped.clamping[0].max_effort.numpy()[0], 50.0, places=3)

        self.assertEqual(delayed.num_actuators, 1)
        self.assertAlmostEqual(delayed.controller.kp.numpy()[0], 200.0, places=3)
        self.assertAlmostEqual(delayed.controller.kd.numpy()[0], 20.0, places=3)
        np.testing.assert_array_equal(delayed.delay.delay_steps.numpy(), [5])
        self.assertEqual(delayed.delay.buf_depth, 5)

        stage = Usd.Stage.Open(usd_path)
        parsed = parse_actuator_prim(stage.GetPrimAtPath("/World/Robot/Joint1Actuator"))
        self.assertIsNotNone(parsed)
        self.assertIsInstance(parsed, ActuatorParsed)
        self.assertEqual(parsed.controller_class, ControllerPD)

    def test_load_from_fixture_usda_with_ignore_paths(self):
        """Actuator prims matched by ``ignore_paths`` are not registered."""
        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")

        builder = newton.ModelBuilder()
        result = parse_usd(builder, usd_path, ignore_paths=[".*Joint1Actuator"])
        self.assertEqual(result["actuator_count"], 1)

        builder2 = newton.ModelBuilder()
        result2 = parse_usd(builder2, usd_path, ignore_paths=[".*Actuator"])
        self.assertEqual(result2["actuator_count"], 0)

    def test_parse_works_when_schema_plugin_not_loaded(self):
        """``parse_actuator_prim`` falls back to raw ``apiSchemas`` metadata when
        the schema plugin isn't registered (``GetAppliedSchemas() == []``).
        """
        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")
        stage = Usd.Stage.Open(usd_path)
        prim = stage.GetPrimAtPath("/World/Robot/Joint1Actuator")

        with patch("newton._src.actuators.usd_parser.get_applied_api_schemas") as mock_schemas:
            mock_schemas.return_value = [SchemaNames.PD_CONTROL, SchemaNames.MAX_EFFORT_CLAMPING]
            parsed = parse_actuator_prim(prim)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.controller_class, ControllerPD)
        self.assertEqual(len(parsed.component_specs), 1)
        self.assertEqual(parsed.component_specs[0][0], ClampingMaxEffort)

    # ---- Per-schema programmatic parsing --------------------------------

    def run_test_parse_schema(
        self,
        components: list[type],
        attrs: dict[str, Any],
        expected_controller: type | None,
        expected_components: list[type],
        controller_kwarg_checks: dict[str, float] | None = None,
        component_kwarg_checks: list[dict[str, float]] | None = None,
    ):
        """Build a stage applying ``components`` schemas + ``attrs`` and parse it.

        Asserts the parsed controller class, the parsed component classes,
        and (optionally) that key kwargs round-trip with the expected
        numeric values.
        """
        stage = self._build_actuator_stage(components, attrs)
        prim = stage.GetPrimAtPath(self._ACT_PATH)
        parsed = parse_actuator_prim(prim)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.target_path, self._JOINT_PATH)
        if expected_controller is not None:
            self.assertEqual(parsed.controller_class, expected_controller)
            for key, val in (controller_kwarg_checks or {}).items():
                self.assertAlmostEqual(parsed.controller_kwargs[key], val, places=4)

        self.assertEqual([cls for cls, _ in parsed.component_specs], expected_components)
        for spec_kwargs, checks in zip(
            (kw for _, kw in parsed.component_specs),
            component_kwarg_checks or [],
            strict=False,
        ):
            for key, val in checks.items():
                self.assertAlmostEqual(spec_kwargs[key], val, places=4)

    def test_parse_pd_controller(self):
        self.run_test_parse_schema(
            components=[ControllerPD],
            attrs={"kp": 12.5, "kd": 1.5},
            expected_controller=ControllerPD,
            expected_components=[],
            controller_kwarg_checks={"kp": 12.5, "kd": 1.5},
        )

    def test_parse_pid_controller(self):
        self.run_test_parse_schema(
            components=[ControllerPID],
            attrs={"kp": 5.0, "ki": 0.1, "kd": 0.5, "integralMax": 10.0},
            expected_controller=ControllerPID,
            expected_components=[],
            controller_kwarg_checks={"kp": 5.0, "ki": 0.1, "kd": 0.5, "integral_max": 10.0},
        )

    @unittest.skipUnless(_HAS_ONNX, "onnx not installed")
    def test_parse_mlp_controller(self):
        """``parse_actuator_prim`` resolves ``Sdf.AssetPath`` for MLP checkpoint."""
        model_path = self._make_mlp_checkpoint(metadata={"model_type": "mlp"})
        stage = self._build_neural_stage(model_path)
        prim = stage.GetPrimAtPath(self._ACT_PATH)

        parsed = parse_actuator_prim(prim)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.controller_class, ControllerNeuralMLP)
        self.assertEqual(parsed.controller_kwargs["model_path"], model_path)
        self.assertEqual(parsed.target_path, self._JOINT_PATH)
        self.assertEqual(len(parsed.component_specs), 1)
        cls, kwargs = parsed.component_specs[0]
        self.assertEqual(cls, ClampingDCMotor)
        self.assertAlmostEqual(kwargs["saturation_effort"], 100.0)

    @unittest.skipUnless(_HAS_ONNX, "onnx not installed")
    def test_parse_lstm_controller(self):
        """``parse_actuator_prim`` resolves ``Sdf.AssetPath`` for LSTM checkpoint."""
        model_path = self._make_lstm_checkpoint(metadata={"model_type": "lstm"})
        stage = self._build_neural_stage(model_path, controller_cls=ControllerNeuralLSTM)
        prim = stage.GetPrimAtPath(self._ACT_PATH)

        parsed = parse_actuator_prim(prim)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.controller_class, ControllerNeuralLSTM)
        self.assertEqual(parsed.controller_kwargs["model_path"], model_path)

    def test_parse_max_effort_clamping(self):
        self.run_test_parse_schema(
            components=[ControllerPD, ClampingMaxEffort],
            attrs={"kp": 1.0, "kd": 0.0, "maxEffort": 42.0},
            expected_controller=ControllerPD,
            expected_components=[ClampingMaxEffort],
            component_kwarg_checks=[{"max_effort": 42.0}],
        )

    def test_parse_dc_motor_clamping(self):
        self.run_test_parse_schema(
            components=[ControllerPD, ClampingDCMotor],
            attrs={
                "kp": 1.0,
                "kd": 0.0,
                "saturationEffort": 80.0,
                "velocityLimit": 15.0,
                "maxMotorEffort": 120.0,
            },
            expected_controller=ControllerPD,
            expected_components=[ClampingDCMotor],
            component_kwarg_checks=[
                {"saturation_effort": 80.0, "velocity_limit": 15.0, "max_motor_effort": 120.0},
            ],
        )

    def test_parse_position_based_clamping(self):
        stage = self._build_actuator_stage(
            components=[ControllerPD, ClampingPositionBased],
            attrs={
                "kp": 1.0,
                "kd": 0.0,
                "lookupPositions": [0.0, 1.0, 2.0],
                "lookupEfforts": [10.0, 5.0, 1.0],
            },
        )
        prim = stage.GetPrimAtPath(self._ACT_PATH)
        parsed = parse_actuator_prim(prim)

        self.assertEqual(parsed.controller_class, ControllerPD)
        self.assertEqual(len(parsed.component_specs), 1)
        cls, kwargs = parsed.component_specs[0]
        self.assertEqual(cls, ClampingPositionBased)
        np.testing.assert_allclose(list(kwargs["lookup_positions"]), [0.0, 1.0, 2.0])
        np.testing.assert_allclose(list(kwargs["lookup_efforts"]), [10.0, 5.0, 1.0])

    def test_parse_delay_component(self):
        self.run_test_parse_schema(
            components=[ControllerPD, Delay],
            attrs={"kp": 1.0, "kd": 0.0, "delaySteps": 4},
            expected_controller=ControllerPD,
            expected_components=[Delay],
            component_kwarg_checks=[{"delay_steps": 4}],
        )

    # ---- Error paths ----------------------------------------------------

    def test_no_targets_relationship_raises(self):
        stage = self._build_actuator_stage(
            components=[ControllerPD],
            attrs={"kp": 1.0},
            target=None,
        )
        with self.assertRaisesRegex(ValueError, "newton:targets"):
            parse_actuator_prim(stage.GetPrimAtPath(self._ACT_PATH))

    def test_no_controller_schema_raises(self):
        """Actuator with only a clamping schema (no controller) is rejected."""
        stage = self._build_actuator_stage(
            components=[ClampingMaxEffort],
            attrs={"maxEffort": 50.0},
        )
        with self.assertRaisesRegex(ValueError, "no controller schema"):
            parse_actuator_prim(stage.GetPrimAtPath(self._ACT_PATH))

    def test_multiple_controllers_raises(self):
        stage = self._build_actuator_stage(
            components=[ControllerPD, ControllerPID],
            attrs={"kp": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "multiple controllers"):
            parse_actuator_prim(stage.GetPrimAtPath(self._ACT_PATH))

    def test_invalid_target_type_raises(self):
        """``newton:targets`` must point at a Revolute/Prismatic joint."""
        stage = self._build_actuator_stage(
            components=[ControllerPD],
            attrs={"kp": 1.0},
            target="base",
        )
        with self.assertRaisesRegex(ValueError, "only"):
            parse_actuator_prim(stage.GetPrimAtPath(self._ACT_PATH))

    @unittest.skipUnless(_HAS_ONNX, "onnx not installed")
    def test_neural_controller_missing_model_type_raises(self):
        """Neural checkpoint without ``model_type`` metadata is rejected."""
        model_path = self._make_mlp_checkpoint(metadata=None)
        stage = self._build_neural_stage(model_path)
        with self.assertRaisesRegex(ValueError, "model_type"):
            parse_actuator_prim(stage.GetPrimAtPath(self._ACT_PATH))

    def test_non_actuator_prim_returns_none(self):
        """Prims of types other than ``NewtonActuator`` parse to ``None``."""
        stage = self._build_actuator_stage(components=[])
        non_actuator = stage.GetPrimAtPath("/World/Robot/Base")
        self.assertIsNone(parse_actuator_prim(non_actuator))


# -------------------------------------------------------------------------
# 7. Cross-cutting error paths
# -------------------------------------------------------------------------


class TestActuatorErrorPaths(unittest.TestCase):
    """Constructors and ``step()`` should reject malformed configurations."""

    def run_test_controller_resolve_negative(self, controller_cls, key: str):
        """``resolve_arguments`` rejects a negative gain/limit for ``key``."""
        bad_args = {key: -1.0}
        with self.assertRaisesRegex(ValueError, key):
            controller_cls.resolve_arguments(bad_args)

    def test_controller_pd_negative_kp(self):
        self.run_test_controller_resolve_negative(ControllerPD, "kp")

    def test_controller_pd_negative_kd(self):
        self.run_test_controller_resolve_negative(ControllerPD, "kd")

    def test_controller_pid_negative_kp(self):
        self.run_test_controller_resolve_negative(ControllerPID, "kp")

    def test_controller_pid_negative_ki(self):
        self.run_test_controller_resolve_negative(ControllerPID, "ki")

    def test_controller_pid_negative_kd(self):
        self.run_test_controller_resolve_negative(ControllerPID, "kd")

    def test_controller_pid_negative_integral_max(self):
        self.run_test_controller_resolve_negative(ControllerPID, "integral_max")

    def run_test_pd_init_shape_mismatch(self, kp_n: int, kd_n: int, ce_n: int | None):
        """``ControllerPD.__init__`` rejects array shape mismatches."""
        kp = _wp_array([1.0] * kp_n)
        kd = _wp_array([1.0] * kd_n)
        ce = _wp_array([0.0] * ce_n) if ce_n is not None else None
        with self.assertRaises(ValueError):
            ControllerPD(kp=kp, kd=kd, const_effort=ce)

    def test_pd_init_kp_kd_shape_mismatch(self):
        self.run_test_pd_init_shape_mismatch(kp_n=3, kd_n=2, ce_n=None)

    def test_pd_init_const_effort_shape_mismatch(self):
        self.run_test_pd_init_shape_mismatch(kp_n=3, kd_n=3, ce_n=2)

    def test_clamping_max_effort_negative(self):
        with self.assertRaisesRegex(ValueError, "max_effort"):
            ClampingMaxEffort.resolve_arguments({"max_effort": -0.1})

    def test_clamping_dc_motor_negative_saturation(self):
        bad = {"saturation_effort": -1.0, "velocity_limit": 10.0, "max_motor_effort": 1.0}
        with self.assertRaisesRegex(ValueError, "saturation_effort"):
            ClampingDCMotor.resolve_arguments(bad)

    def test_clamping_dc_motor_non_positive_velocity_limit(self):
        bad = {"saturation_effort": 1.0, "velocity_limit": 0.0, "max_motor_effort": 1.0}
        with self.assertRaisesRegex(ValueError, "velocity_limit"):
            ClampingDCMotor.resolve_arguments(bad)

    def test_clamping_position_based_negative_effort_in_table(self):
        bad = {"lookup_positions": [0.0, 1.0], "lookup_efforts": [1.0, -0.5]}
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ClampingPositionBased.resolve_arguments(bad)

    def test_clamping_position_based_non_monotonic_positions(self):
        bad = {"lookup_positions": [0.0, 0.5, 0.2], "lookup_efforts": [1.0, 1.0, 1.0]}
        with self.assertRaisesRegex(ValueError, "monotonically"):
            ClampingPositionBased.resolve_arguments(bad)

    def test_delay_resolve_negative_steps(self):
        with self.assertRaisesRegex(ValueError, "delay_steps"):
            Delay.resolve_arguments({"delay_steps": -1})

    def test_delay_resolve_missing_steps(self):
        with self.assertRaisesRegex(ValueError, "delay_steps"):
            Delay.resolve_arguments({})

    def test_delay_init_zero_max_delay(self):
        with self.assertRaisesRegex(ValueError, "max_delay"):
            Delay(delay_steps=_wp_array([1], dtype=wp.int32), max_delay=0)

    def run_test_actuator_index_shape_mismatch(
        self,
        n: int,
        pos_n: int | None,
        target_n: int | None,
        effort_n: int | None,
    ):
        """``Actuator.__init__`` rejects index arrays whose shapes don't match ``indices``."""
        indices = _wp_array([0] * n, dtype=wp.uint32)
        pos = _wp_array([0] * pos_n, dtype=wp.uint32) if pos_n is not None else None
        tgt = _wp_array([0] * target_n, dtype=wp.uint32) if target_n is not None else None
        eff = _wp_array([0] * effort_n, dtype=wp.uint32) if effort_n is not None else None
        ctrl = ControllerPD(kp=_wp_array([1.0] * n), kd=_wp_array([0.0] * n))
        with self.assertRaises(ValueError):
            Actuator(
                indices=indices,
                controller=ctrl,
                pos_indices=pos,
                target_pos_indices=tgt,
                effort_indices=eff,
            )

    def test_actuator_pos_indices_mismatch(self):
        self.run_test_actuator_index_shape_mismatch(n=3, pos_n=2, target_n=None, effort_n=None)

    def test_actuator_target_pos_indices_mismatch(self):
        self.run_test_actuator_index_shape_mismatch(n=3, pos_n=None, target_n=4, effort_n=None)

    def test_actuator_effort_indices_mismatch(self):
        self.run_test_actuator_index_shape_mismatch(n=3, pos_n=None, target_n=None, effort_n=1)

    def _build_stateful_actuator(self, n: int = 2) -> Actuator:
        """Build a delayed PD actuator (stateful) without any state arrays."""
        indices = _wp_array(list(range(n)), dtype=wp.uint32)
        ctrl = ControllerPD(kp=_wp_array([1.0] * n), kd=_wp_array([0.0] * n))
        delay = Delay(delay_steps=_wp_array([1] * n, dtype=wp.int32), max_delay=2)
        return Actuator(indices=indices, controller=ctrl, delay=delay)

    def _empty_sim_namespaces(self, n: int):
        sim_state = types.SimpleNamespace(joint_q=_wp_array([0.0] * n), joint_qd=_wp_array([0.0] * n))
        sim_control = types.SimpleNamespace(
            joint_target_q=_wp_array([0.0] * n),
            joint_target_qd=_wp_array([0.0] * n),
            joint_act=None,
            joint_f=wp.zeros(n, dtype=wp.float32),
        )
        return sim_state, sim_control

    def test_actuator_stateful_step_without_states_raises(self):
        actuator = self._build_stateful_actuator()
        sim_state, sim_control = self._empty_sim_namespaces(2)
        with self.assertRaisesRegex(ValueError, "Stateful"):
            actuator.step(sim_state, sim_control, None, None, dt=0.01)

    def test_actuator_stateful_step_missing_next_state_raises(self):
        actuator = self._build_stateful_actuator()
        sim_state, sim_control = self._empty_sim_namespaces(2)
        current = actuator.state()
        with self.assertRaisesRegex(ValueError, "Stateful"):
            actuator.step(sim_state, sim_control, current, None, dt=0.01)


# -------------------------------------------------------------------------
# 8. Actuator.step integration — ModelBuilder-built and standalone
# -------------------------------------------------------------------------


class TestActuatorStep(unittest.TestCase):
    """End-to-end :meth:`Actuator.step` — built via :class:`ModelBuilder` or directly."""

    def run_actuators_step(
        self,
        actuator: Actuator,
        state,
        control,
        target_schedule,
        expected_forces,
        *,
        target_pos_attr: str = "joint_target_q",
        dofs: list[int] | None = None,
        dt: float = 0.01,
        atol: float = 1e-3,
    ):
        """Step ``actuator`` once per entry in ``target_schedule`` and assert ``joint_f`` matches.

        Works for both standalone (:class:`types.SimpleNamespace` state/control) and
        :class:`ModelBuilder`-built setups (:class:`State` / :class:`Control`).

        Args:
            actuator: Built actuator instance.
            state: Object exposing ``joint_q`` / ``joint_qd``.
            control: Object exposing ``joint_f`` and ``target_pos_attr``.
            target_schedule: Per step, either one scalar (broadcast to all DOFs) or a list with one value per ``dofs`` entry.
            expected_forces: ``expected_forces[step_i][k]`` for the k-th DOF in ``dofs``.
            target_pos_attr: Attribute on ``control`` the actuator reads targets from.
            dofs: DOF indices into ``control.joint_f``. ``None`` ⇒ ``range(actuator.num_actuators)``.
            dt: Simulation timestep.
            atol: Absolute tolerance for force comparisons.

        Returns:
            ``(s0, s1)`` — actuator state after the final step (``(None, None)`` if stateless).
        """
        if dofs is None:
            dofs = list(range(actuator.num_actuators))
        target_arr = getattr(control, target_pos_attr)
        s0, s1 = (actuator.state(), actuator.state()) if actuator.is_stateful() else (None, None)

        for step_i, tp in enumerate(target_schedule):
            tp_per_dof = list(tp) if isinstance(tp, list | tuple) else [tp] * len(dofs)
            if len(tp_per_dof) != len(dofs):
                raise ValueError(f"target_schedule[{step_i}] has {len(tp_per_dof)} values, dofs has {len(dofs)}")
            target_np = target_arr.numpy()
            for d, v in zip(dofs, tp_per_dof, strict=True):
                target_np[d] = v
            wp.copy(target_arr, wp.array(target_np, dtype=float, device=target_arr.device))

            control.joint_f.zero_()
            actuator.step(state, control, s0, s1, dt=dt)
            if s0 is not None:
                s0, s1 = s1, s0

            got = np.asarray([control.joint_f.numpy()[d] for d in dofs], dtype=np.float32)
            np.testing.assert_allclose(
                got,
                np.asarray(expected_forces[step_i], dtype=np.float32),
                atol=atol,
                err_msg=f"step={step_i} target={tp} dofs={dofs}",
            )

        return s0, s1

    def test_full_pipeline(self):
        """End-to-end ``Actuator.step``: PD + per-DOF delay + DC-motor clamp on a free + 2 revolute template.

        The free joint contributes 7 coords / 6 DOFs, so the actuator's
        ``pos_indices`` (coord layout) and ``target_pos_indices`` (DOF layout)
        differ by one. Combined with distinct per-joint targets at every step,
        a wrong-index read into ``joint_target_q`` would surface as a
        wrong-sign force on at least one joint.

        Per step we expect ``force = dc_clamp(kp*(delayed_target - q) + kd*(0 - qd), qd)``
        where ``dc_clamp`` is the DC-motor velocity-dependent torque envelope.
        Finally verifies ``num_pushes`` is clamped to ``buf_depth``.
        """
        kp, kd = 50.0, 5.0
        sat, v_lim = 80.0, 20.0
        delay_a, delay_b = 2, 3
        num_envs = 3
        qd_val = 2.0

        template = newton.ModelBuilder()
        base = template.add_link()
        free_joint = template.add_joint_free(parent=-1, child=base)
        link_a = template.add_link()
        joint_a = template.add_joint_revolute(parent=base, child=link_a, axis=newton.Axis.Z)
        link_b = template.add_link()
        joint_b = template.add_joint_revolute(parent=link_a, child=link_b, axis=newton.Axis.Z)
        template.add_articulation([free_joint, joint_a, joint_b])
        dof_a = template.joint_qd_start[joint_a]
        dof_b = template.joint_qd_start[joint_b]
        dc_args = {"saturation_effort": sat, "velocity_limit": v_lim, "max_motor_effort": 1e6}
        for dof, delay in [(dof_a, delay_a), (dof_b, delay_b)]:
            template.add_actuator(
                ControllerPD,
                index=dof,
                kp=kp,
                kd=kd,
                delay_steps=delay,
                clamping=[(ClampingDCMotor, dc_args)],
            )

        builder = newton.ModelBuilder()
        builder.replicate(template, num_envs)
        model = builder.finalize()

        self.assertEqual(len(model.actuators), 1, "all DOFs share controller+clamping type")
        actuator = model.actuators[0]
        n = actuator.num_actuators
        self.assertEqual(n, 2 * num_envs)

        pos_idx = actuator.pos_indices.numpy()
        target_pos_idx = actuator.target_pos_indices.numpy()
        self.assertFalse(
            np.array_equal(pos_idx, target_pos_idx),
            f"Free joint should give coord/DOF index mismatch; pos={pos_idx} target_pos={target_pos_idx}",
        )

        expected_delays = [delay_a, delay_b] * num_envs
        np.testing.assert_array_equal(actuator.delay.delay_steps.numpy(), expected_delays)

        state = model.state()
        dofs = actuator.indices.numpy().tolist()
        _write_dof_values(model, state.joint_qd, dofs, [qd_val] * n)

        # Joint_a gets +s, joint_b gets -s — wrong indexing would swap targets between joints.
        scalars = [10.0, 20.0, 30.0, 40.0, 50.0]
        target_schedule = [[s, -s] * num_envs for s in scalars]

        def _dc_clamp(raw: float, vel: float) -> float:
            tau_max = min(sat * (1.0 - vel / v_lim), 1e6)
            tau_min = max(sat * (-1.0 - vel / v_lim), -1e6)
            return max(min(raw, tau_max), tau_min)

        def _delayed_target(step_i: int, local_dof: int, dof_delay: int) -> float:
            if step_i == 0:
                return target_schedule[step_i][local_dof]
            lag = min(dof_delay - 1, step_i - 1)
            return target_schedule[step_i - 1 - lag][local_dof]

        expected_forces = [
            [
                _dc_clamp(kp * (_delayed_target(step_i, k, d) - 0.0) + kd * (0.0 - qd_val), qd_val)
                for k, d in enumerate(expected_delays)
            ]
            for step_i in range(len(target_schedule))
        ]

        s0, _ = self.run_actuators_step(
            actuator,
            state,
            model.control(),
            target_schedule,
            expected_forces,
            target_pos_attr="joint_target_q",
            dofs=dofs,
        )

        np.testing.assert_array_equal(
            s0.delay_state.num_pushes.numpy(),
            [min(5, actuator.delay.buf_depth)] * n,
            err_msg="num_pushes should be clamped to buf_depth",
        )

    def _make_state_control(self, n_dof: int, q: float = 0.2, qd: float = 0.0):
        """Return duck-typed ``sim_state`` / ``sim_control`` namespaces with ``n_dof`` joints."""
        sim_state = types.SimpleNamespace(
            joint_q=_wp_array([q] * n_dof),
            joint_qd=_wp_array([qd] * n_dof),
        )
        sim_control = types.SimpleNamespace(
            joint_target_q=_wp_array([0.0] * n_dof),
            joint_target_qd=_wp_array([0.0] * n_dof),
            joint_act=None,
            joint_f=wp.zeros(n_dof, dtype=wp.float32),
        )
        return sim_state, sim_control

    def test_standalone_pd_1dof_no_delay_no_clamp(self):
        kp, kd, q = 10.0, 0.5, 0.2
        targets = [1.0, 2.0]
        actuator = Actuator(
            indices=_wp_array([0], dtype=wp.uint32),
            controller=ControllerPD(kp=_wp_array([kp]), kd=_wp_array([kd])),
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        sim_state, sim_control = self._make_state_control(n_dof=1, q=q)
        # f = kp*(tp - q) + kd*(0 - 0)
        expected = [[kp * (tp - q)] for tp in targets]
        self.run_actuators_step(actuator, sim_state, sim_control, targets, expected)

    def test_standalone_pd_4dof_no_delay_no_clamp(self):
        n_dof = 4
        kp, kd, q = 10.0, 0.5, 0.2
        targets = [1.0, 2.0]
        actuator = Actuator(
            indices=_wp_array(list(range(n_dof)), dtype=wp.uint32),
            controller=ControllerPD(kp=_wp_array([kp] * n_dof), kd=_wp_array([kd] * n_dof)),
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        sim_state, sim_control = self._make_state_control(n_dof, q=q)
        expected = [[kp * (tp - q)] * n_dof for tp in targets]
        self.run_actuators_step(actuator, sim_state, sim_control, targets, expected)

    def test_standalone_pd_4dof_with_delay_no_clamp(self):
        n_dof = 4
        kp, kd, q = 10.0, 0.5, 0.2
        targets = [1.0, 2.0]
        actuator = Actuator(
            indices=_wp_array(list(range(n_dof)), dtype=wp.uint32),
            controller=ControllerPD(kp=_wp_array([kp] * n_dof), kd=_wp_array([kd] * n_dof)),
            delay=Delay(delay_steps=_wp_array([1] * n_dof, dtype=wp.int32), max_delay=1),
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        sim_state, sim_control = self._make_state_control(n_dof, q=q)
        # delay=1, max_delay=1: step 0 falls back to current, step 1 reads targets[0].
        effective = [targets[max(0, i - 1)] for i in range(len(targets))]
        expected = [[kp * (e - q)] * n_dof for e in effective]
        self.run_actuators_step(actuator, sim_state, sim_control, targets, expected)

    def test_standalone_pd_4dof_no_delay_with_clamp(self):
        n_dof = 4
        kp, kd, q, max_eff = 10.0, 0.5, 0.2, 10.0
        targets = [1.0, 2.0]
        actuator = Actuator(
            indices=_wp_array(list(range(n_dof)), dtype=wp.uint32),
            controller=ControllerPD(kp=_wp_array([kp] * n_dof), kd=_wp_array([kd] * n_dof)),
            clamping=[ClampingMaxEffort(max_effort=_wp_array([max_eff] * n_dof))],
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        sim_state, sim_control = self._make_state_control(n_dof, q=q)
        expected = [[max(-max_eff, min(max_eff, kp * (tp - q)))] * n_dof for tp in targets]
        self.run_actuators_step(actuator, sim_state, sim_control, targets, expected)

    def test_standalone_pd_4dof_with_delay_with_clamp(self):
        n_dof = 4
        kp, kd, q, max_eff = 10.0, 0.5, 0.2, 10.0
        targets = [1.0, 2.0]
        actuator = Actuator(
            indices=_wp_array(list(range(n_dof)), dtype=wp.uint32),
            controller=ControllerPD(kp=_wp_array([kp] * n_dof), kd=_wp_array([kd] * n_dof)),
            delay=Delay(delay_steps=_wp_array([1] * n_dof, dtype=wp.int32), max_delay=1),
            clamping=[ClampingMaxEffort(max_effort=_wp_array([max_eff] * n_dof))],
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        sim_state, sim_control = self._make_state_control(n_dof, q=q)
        # Delay collapses step 1's effective target to targets[0] ⇒ raw below clamp threshold.
        effective = [targets[max(0, i - 1)] for i in range(len(targets))]
        expected = [[max(-max_eff, min(max_eff, kp * (e - q)))] * n_dof for e in effective]
        self.run_actuators_step(actuator, sim_state, sim_control, targets, expected)


if __name__ == "__main__":
    unittest.main()
