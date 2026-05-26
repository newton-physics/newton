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


# ---------------------------------------------------------------------------
# 1. Controllers
# ---------------------------------------------------------------------------


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


@unittest.skipUnless(_HAS_ONNX, "onnx not installed")
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


# ---------------------------------------------------------------------------
# 2. Delay
# ---------------------------------------------------------------------------


class TestDelay(unittest.TestCase):
    """Delay unit tests — construct Delay directly, call get_delayed_targets/update_state."""

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

    def test_latency_behavior(self):
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

        # DOF 0 (delay=0): always sees current target
        self.assertAlmostEqual(history_dof0[0], 10.0, places=4, msg="dof0 step 0")
        self.assertAlmostEqual(history_dof0[1], 20.0, places=4, msg="dof0 step 1")
        self.assertAlmostEqual(history_dof0[2], 30.0, places=4, msg="dof0 step 2")
        self.assertAlmostEqual(history_dof0[3], 40.0, places=4, msg="dof0 step 3")

        # DOF 1 (delay=1): empty buffer fallback then one-step lag
        self.assertAlmostEqual(history_dof1[0], 10.0, places=4, msg="dof1 step 0: empty -> current")
        self.assertAlmostEqual(history_dof1[1], 10.0, places=4, msg="dof1 step 1: reads step 0 (10)")
        self.assertAlmostEqual(history_dof1[2], 20.0, places=4, msg="dof1 step 2: reads step 1 (20)")
        self.assertAlmostEqual(history_dof1[3], 30.0, places=4, msg="dof1 step 3: reads step 2 (30)")


# ---------------------------------------------------------------------------
# 3. Clamping
# ---------------------------------------------------------------------------


class TestClampingMaxEffort(unittest.TestCase):
    """ClampingMaxEffort: output is clamped to +/-max_effort."""

    def test_modify_forces(self):
        """Construct clamping directly and call modify_forces()."""
        max_f = 50.0
        n = 3
        clamp = ClampingMaxEffort(max_effort=wp.array([max_f] * n, dtype=wp.float32))

        src_vals = [100.0, -80.0, 30.0]
        src = wp.array(src_vals, dtype=wp.float32)
        dst = wp.zeros(n, dtype=wp.float32)
        indices = wp.array(list(range(n)), dtype=wp.uint32)

        clamp.modify_forces(src, dst, wp.zeros(n, dtype=wp.float32), wp.zeros(n, dtype=wp.float32), indices, indices)

        result = dst.numpy()
        for i, s in enumerate(src_vals):
            expected = max(min(s, max_f), -max_f)
            self.assertAlmostEqual(result[i], expected, places=5, msg=f"DOF {i}")


class TestClampingDCMotor(unittest.TestCase):
    """DC motor torque-speed curve: clamp = saturation * (1 - v/v_limit)."""

    def test_modify_forces(self):
        """Construct clamping directly and call modify_forces() at several velocity points."""
        sat, v_lim, max_f = 100.0, 10.0, 200.0
        clamp = ClampingDCMotor(
            saturation_effort=wp.array([sat], dtype=wp.float32),
            velocity_limit=wp.array([v_lim], dtype=wp.float32),
            max_motor_effort=wp.array([max_f], dtype=wp.float32),
        )
        indices = wp.array([0], dtype=wp.uint32)
        raw_force = 500.0

        for qd in [0.0, 5.0, 10.0, -5.0]:
            src = wp.array([raw_force], dtype=wp.float32)
            dst = wp.zeros(1, dtype=wp.float32)
            vel = wp.array([qd], dtype=wp.float32)

            clamp.modify_forces(src, dst, wp.zeros(1, dtype=wp.float32), vel, indices, indices)

            tau_max = min(sat * (1.0 - qd / v_lim), max_f)
            tau_min = max(sat * (-1.0 - qd / v_lim), -max_f)
            expected = max(min(raw_force, tau_max), tau_min)
            self.assertAlmostEqual(dst.numpy()[0], expected, places=3, msg=f"qd={qd}")


class TestClampingPositionBased(unittest.TestCase):
    """Position-based clamping with angle-dependent lookup table."""

    def test_modify_forces(self):
        """Construct clamping directly and verify interpolated angle-dependent limits."""
        angles = (-1.0, 0.0, 1.0)
        torques = (10.0, 30.0, 50.0)
        device = wp.get_device()
        clamp = ClampingPositionBased(lookup_positions=angles, lookup_efforts=torques)
        clamp.finalize(device, 1)

        raw_force = 999.0
        indices = wp.array([0], dtype=wp.uint32, device=device)

        for pos, expected_limit in [(-1.0, 10.0), (0.0, 30.0), (1.0, 50.0), (-0.5, 20.0), (0.5, 40.0)]:
            src = wp.array([raw_force], dtype=wp.float32, device=device)
            dst = wp.zeros(1, dtype=wp.float32, device=device)
            positions = wp.array([pos], dtype=wp.float32, device=device)

            clamp.modify_forces(
                src, dst, positions, wp.zeros(1, dtype=wp.float32, device=device), indices, indices, device=device
            )

            self.assertAlmostEqual(dst.numpy()[0], expected_limit, places=2, msg=f"pos={pos}")


# ---------------------------------------------------------------------------
# 4. Actuator pipeline — full step() integration
# ---------------------------------------------------------------------------


class TestActuatorStep(unittest.TestCase):
    """Integration test: full Actuator.step() with delay + PD + DC-motor clamping."""

    def test_full_pipeline(self):
        """Two-joint template x 3 envs, per-DOF delays (2 / 3), PD + DC motor.

        At each of 5 steps we verify:
            raw   = kp*(delayed_target - q) + kd*(0 - qd)
            τ_max = clamp(sat*(1 - qd/v_lim),  0,  max_f)
            τ_min = clamp(sat*(-1 - qd/v_lim), -max_f, 0)
            force = clamp(raw, τ_min, τ_max)
        """
        kp, kd = 50.0, 5.0
        sat, v_lim = 80.0, 20.0
        delay_a, delay_b = 2, 3
        num_envs = 3
        dt = 0.01

        template = newton.ModelBuilder()
        link_a = template.add_link()
        joint_a = template.add_joint_revolute(parent=-1, child=link_a, axis=newton.Axis.Z)
        link_b = template.add_link()
        joint_b = template.add_joint_revolute(parent=link_a, child=link_b, axis=newton.Axis.Z)
        template.add_articulation([joint_a, joint_b])
        dof_a = template.joint_qd_start[joint_a]
        dof_b = template.joint_qd_start[joint_b]
        dc_args = {"saturation_effort": sat, "velocity_limit": v_lim, "max_motor_effort": 1e6}
        template.add_actuator(
            ControllerPD,
            index=dof_a,
            kp=kp,
            kd=kd,
            delay_steps=delay_a,
            clamping=[(ClampingDCMotor, dc_args)],
        )
        template.add_actuator(
            ControllerPD,
            index=dof_b,
            kp=kp,
            kd=kd,
            delay_steps=delay_b,
            clamping=[(ClampingDCMotor, dc_args)],
        )

        builder = newton.ModelBuilder()
        builder.replicate(template, num_envs)
        model = builder.finalize()

        self.assertEqual(len(model.actuators), 1, "all DOFs share controller+clamping type")
        actuator = model.actuators[0]
        n = actuator.num_actuators
        self.assertEqual(n, 2 * num_envs)

        delays_np = actuator.delay.delay_steps.numpy()
        expected_delays = [delay_a, delay_b] * num_envs
        np.testing.assert_array_equal(delays_np, expected_delays)

        state = model.state()
        state_0 = actuator.state()
        state_1 = actuator.state()

        qd_val = 2.0
        dofs = actuator.indices.numpy().tolist()
        _write_dof_values(model, state.joint_qd, dofs, [qd_val] * n)

        target_schedule = [10.0, 20.0, 30.0, 40.0, 50.0]
        written_targets: list[float] = []

        def _dc_clamp(raw: float, vel: float) -> float:
            tau_max = min(sat * (1.0 - vel / v_lim), 1e6)
            tau_min = max(sat * (-1.0 - vel / v_lim), -1e6)
            return max(min(raw, tau_max), tau_min)

        def _delayed_target(step_i: int, dof_delay: int) -> float:
            pushes = step_i
            if pushes == 0:
                return target_schedule[step_i]
            lag = min(dof_delay - 1, pushes - 1)
            return written_targets[step_i - 1 - lag]

        control = model.control()
        for step_i in range(5):
            tgt = target_schedule[step_i]
            _write_dof_values(model, control.joint_target_pos, dofs, [tgt] * n)
            written_targets.append(tgt)

            control.joint_f.zero_()
            actuator.step(state, control, state_0, state_1, dt)
            state_0, state_1 = state_1, state_0

            forces = control.joint_f.numpy()
            for local_i in range(n):
                d = dofs[local_i]
                dof_delay = expected_delays[local_i]
                delayed_tgt = _delayed_target(step_i, dof_delay)
                raw = kp * (delayed_tgt - 0.0) + kd * (0.0 - qd_val)
                expected = _dc_clamp(raw, qd_val)
                self.assertAlmostEqual(
                    forces[d],
                    expected,
                    places=3,
                    msg=f"step={step_i} dof={local_i} delay={dof_delay} "
                    f"delayed_tgt={delayed_tgt} raw={raw} expected={expected}",
                )

        ds = state_0.delay_state
        np.testing.assert_array_equal(
            ds.num_pushes.numpy(),
            [min(5, actuator.delay.buf_depth)] * n,
            err_msg="num_pushes should be clamped to buf_depth",
        )


# ---------------------------------------------------------------------------
# 5. Builder — from USD, programmatic, and free-joint replication
# ---------------------------------------------------------------------------


class TestActuatorBuilder(unittest.TestCase):
    """ModelBuilder actuator construction — grouping, params, state, and index layouts."""

    @unittest.skipUnless(HAS_USD, "pxr not installed")
    def test_from_usd(self):
        """Load actuators from a USD stage and verify params after finalize.

        The asset has two actuators:
          Joint1Actuator: PD (kp=100, kd=10) + MaxForce(50)
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

    @unittest.skipUnless(HAS_USD, "pxr not installed")
    def test_from_usd_ignore_paths(self):
        """Actuator prims matched by ignore_paths are not registered."""
        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")

        builder = newton.ModelBuilder()
        result = parse_usd(builder, usd_path, ignore_paths=[".*Joint1Actuator"])
        self.assertEqual(result["actuator_count"], 1)

        builder2 = newton.ModelBuilder()
        result2 = parse_usd(builder2, usd_path, ignore_paths=[".*Actuator"])
        self.assertEqual(result2["actuator_count"], 0)

    @unittest.skipUnless(HAS_USD, "pxr not installed")
    def test_from_usd_schema_plugin_not_loaded(self):
        """parse_actuator_prim works when the USD schema plugin is not registered.

        Simulates the headless case where GetAppliedSchemas() returns [] because
        the Newton schema plugin failed to load, but the raw apiSchemas metadata
        is still present on the prim.
        """
        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")
        if not os.path.exists(usd_path):
            self.skipTest(f"Test USD file not found: {usd_path}")

        stage = Usd.Stage.Open(usd_path)
        prim = stage.GetPrimAtPath("/World/Robot/Joint1Actuator")

        with patch.object(type(prim), "GetAppliedSchemas", return_value=[]):
            self.assertEqual(prim.GetAppliedSchemas(), [], "patch must be active for this test to be meaningful")
            parsed = parse_actuator_prim(prim)

        self.assertIsNotNone(parsed)
        self.assertIsInstance(parsed, ActuatorParsed)
        self.assertEqual(parsed.controller_class, ControllerPD)
        self.assertAlmostEqual(parsed.controller_kwargs["kp"], 100.0)
        self.assertAlmostEqual(parsed.controller_kwargs["kd"], 10.0)

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


# ---------------------------------------------------------------------------
# 6. Legacy newton_actuators backward compatibility
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_LEGACY_ACTUATORS, "newton_actuators not installed")
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


# ---------------------------------------------------------------------------
# 7. Parameter access via ArticulationView
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 7. State reset (masked and full)
# ---------------------------------------------------------------------------


class TestStateReset(unittest.TestCase):
    """Exercise State.reset() for delay, PID, and composed Actuator.State."""

    def test_delay_masked_reset(self):
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

    def test_delay_full_reset(self):
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

    def test_pid_masked_reset(self):
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

    def test_actuator_composed_reset(self):
        """Actuator.State.reset delegates to both delay and controller sub-states."""
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
            _write_dof_values(model, control.joint_target_pos, dofs, [10.0] * n)
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


# ---------------------------------------------------------------------------
# 7b. CUDA graph capture — end-to-end with Newton solver + delayed actuator
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()),
    "CUDA graph capture requires CUDA device with memory pools",
)
class TestDelayGraphCapture(unittest.TestCase):
    """Verify delayed actuator is graph-safe with device-side write_idx.

    Captures N actuator + K physics substeps as a CUDA graph and replays
    with varying targets. With N even and N % buf_depth != 0, the test
    confirms graph replay matches eager execution — proving the write
    pointer advances correctly on-device during replay.
    """

    def test_delay_graph_n_not_multiple_matches_eager(self):
        """N=2, buf_depth=5: graph matches eager across multiple cycles.

        This configuration (N < buf_depth, N % buf_depth != 0) previously
        failed when write_idx was a host-side scalar baked into the graph.
        With device-side write_idx the kernel advances the pointer on-GPU,
        making graph replay correct for any even N.
        """
        max_delay = 5
        N = 2  # 2 % 5 != 0, N < buf_depth, N is even
        K = 2
        dt = 0.02
        warmup_target = 0.0
        cycle_targets = [2.0, -3.0, 5.0, -1.0]

        # Build a single-DOF revolute pendulum with delayed PD actuator
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

        # --- Eager ---
        solver, s0, s1, ctrl, act, act_a, act_b = _setup()
        wp.copy(ctrl.joint_target_pos, wp.full(ndof, warmup_target, dtype=wp.float32, device=device))
        s0, s1, act_a, act_b = _loop(solver, s0, s1, ctrl, act, act_a, act_b, max_delay)
        eager_results = []
        for tgt in cycle_targets:
            wp.copy(ctrl.joint_target_pos, wp.full(ndof, tgt, dtype=wp.float32, device=device))
            s0, s1, act_a, act_b = _loop(solver, s0, s1, ctrl, act, act_a, act_b, N)
            eager_results.append(s0.joint_q.numpy().copy())

        # --- Graph ---
        solver_g, s0_g, s1_g, ctrl_g, act_g, act_a_g, act_b_g = _setup()
        wp.copy(ctrl_g.joint_target_pos, wp.full(ndof, warmup_target, dtype=wp.float32, device=device))
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
            wp.copy(ctrl_g.joint_target_pos, wp.full(ndof, tgt, dtype=wp.float32, device=device))
            wp.capture_launch(graph)
            graph_results.append(s0_g.joint_q.numpy().copy())

        for ci in range(len(cycle_targets)):
            np.testing.assert_allclose(
                graph_results[ci],
                eager_results[ci],
                rtol=1e-4,
                err_msg=f"Cycle {ci}: graph should match eager with device-side write_idx",
            )


# ---------------------------------------------------------------------------
# 8. Neural controller via USD parsing (parse_actuator_prim)
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_USD and _HAS_ONNX, "pxr or onnx not installed")
class TestNeuralActuatorUsdParsing(unittest.TestCase):
    """Verify ``parse_actuator_prim`` correctly handles neural controller
    prims with asset-typed ``newton:modelPath`` attributes.

    This exercises the full USD parsing path — the same path that
    ``ModelBuilder.add_usd`` uses — rather than constructing controllers
    directly from a file path.
    """

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _make_mlp_checkpoint(self, metadata: dict | None = None) -> str:
        """Create a minimal ONNX MLP checkpoint with optional metadata."""
        path = os.path.join(self._tmp_dir, "mlp.onnx")
        weights = np.zeros((1, 2), dtype=np.float32)
        bias = np.ones((1,), dtype=np.float32)
        _build_mlp_onnx(path, weights, bias, metadata)
        return path

    def _make_lstm_checkpoint(self, metadata: dict | None = None) -> str:
        """Create a minimal ONNX LSTM checkpoint with optional metadata."""
        path = os.path.join(self._tmp_dir, "lstm.onnx")
        _build_lstm_onnx(path, hidden_size=8, num_layers=1, metadata=metadata)
        return path

    def _build_neural_stage(self, model_path: str) -> "Usd.Stage":
        """Create a minimal USD stage with a neural actuator prim.

        The stage has a two-link articulation with a single revolute
        joint and a ``NewtonActuator`` prim with ``NewtonNeuralControlAPI``
        applied, referencing *model_path* via the ``newton:modelPath``
        asset attribute.
        """
        from pxr import Sdf

        stage = Usd.Stage.CreateInMemory()
        world = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world)

        stage.DefinePrim("/World/PhysicsScene", "PhysicsScene")

        robot = stage.DefinePrim("/World/Robot", "Xform")
        schemas = Sdf.TokenListOp()
        schemas.prependedItems = ["PhysicsArticulationRootAPI"]
        robot.SetMetadata("apiSchemas", schemas)

        base = stage.DefinePrim("/World/Robot/Base", "Xform")
        base_schemas = Sdf.TokenListOp()
        base_schemas.prependedItems = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        base.SetMetadata("apiSchemas", base_schemas)
        base.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(1.0)
        base.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)

        link1 = stage.DefinePrim("/World/Robot/Link1", "Xform")
        link1_schemas = Sdf.TokenListOp()
        link1_schemas.prependedItems = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        link1.SetMetadata("apiSchemas", link1_schemas)
        link1.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(0.5)

        joint = stage.DefinePrim("/World/Robot/Joint1", "PhysicsRevoluteJoint")
        joint_schemas = Sdf.TokenListOp()
        joint_schemas.prependedItems = ["PhysicsDriveAPI:angular"]
        joint.SetMetadata("apiSchemas", joint_schemas)
        joint.CreateRelationship("physics:body0").SetTargets([Sdf.Path("/World/Robot/Base")])
        joint.CreateRelationship("physics:body1").SetTargets([Sdf.Path("/World/Robot/Link1")])
        joint.CreateAttribute("physics:axis", Sdf.ValueTypeNames.Token).Set("Z")

        act_prim = stage.DefinePrim("/World/Robot/NeuralActuator", "NewtonActuator")
        act_schemas = Sdf.TokenListOp()
        act_schemas.prependedItems = ["NewtonNeuralControlAPI", "NewtonDCMotorClampingAPI"]
        act_prim.SetMetadata("apiSchemas", act_schemas)
        act_prim.CreateRelationship("newton:targets").SetTargets([Sdf.Path("/World/Robot/Joint1")])
        act_prim.CreateAttribute("newton:modelPath", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(model_path))
        act_prim.CreateAttribute("newton:saturationEffort", Sdf.ValueTypeNames.Float).Set(100.0)
        act_prim.CreateAttribute("newton:velocityLimit", Sdf.ValueTypeNames.Float).Set(20.0)
        act_prim.CreateAttribute("newton:maxMotorEffort", Sdf.ValueTypeNames.Float).Set(200.0)

        return stage

    def test_parse_mlp_from_usd(self):
        """parse_actuator_prim resolves Sdf.AssetPath for MLP checkpoint."""
        model_path = self._make_mlp_checkpoint(metadata={"model_type": "mlp"})
        stage = self._build_neural_stage(model_path)
        prim = stage.GetPrimAtPath("/World/Robot/NeuralActuator")

        parsed = parse_actuator_prim(prim)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.controller_class, ControllerNeuralMLP)
        self.assertEqual(parsed.controller_kwargs["model_path"], model_path)
        self.assertEqual(parsed.target_path, "/World/Robot/Joint1")

        self.assertEqual(len(parsed.component_specs), 1)
        cls, kwargs = parsed.component_specs[0]
        self.assertEqual(cls, ClampingDCMotor)
        self.assertAlmostEqual(kwargs["saturation_effort"], 100.0)

    def test_parse_lstm_from_usd(self):
        """parse_actuator_prim resolves Sdf.AssetPath for LSTM checkpoint."""
        model_path = self._make_lstm_checkpoint(metadata={"model_type": "lstm"})
        stage = self._build_neural_stage(model_path)
        prim = stage.GetPrimAtPath("/World/Robot/NeuralActuator")

        parsed = parse_actuator_prim(prim)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.controller_class, ControllerNeuralLSTM)
        self.assertEqual(parsed.controller_kwargs["model_path"], model_path)


# ---------------------------------------------------------------------------
# 9. target_pos_indices separation from pos_indices
# ---------------------------------------------------------------------------


class TestTargetPosIndicesSeparation(unittest.TestCase):
    """Actuator must read joint_target_pos via target_pos_indices, not pos_indices."""

    def test_target_pos_read_from_dof_index_not_coord_index(self):
        device = wp.get_device()

        def _a(vals, dtype=wp.float32):
            return wp.array(vals, dtype=dtype, device=device)

        kp = 100.0
        actual_pos = 0.5
        correct_target = 2.0
        sentinel = 99.0  # placed at coord index 3 to catch wrong index usage

        indices = _a([1], dtype=wp.uint32)  # DOF index 1
        pos_indices = _a([3], dtype=wp.uint32)  # coord index 3 (joint_q layout)
        target_pos_indices = _a([1], dtype=wp.uint32)  # DOF index 1 (joint_target_pos layout)

        ctrl = ControllerPD(kp=_a([kp]), kd=_a([0.0]), const_effort=_a([0.0]))
        actuator = Actuator(
            indices=indices,
            controller=ctrl,
            pos_indices=pos_indices,
            target_pos_indices=target_pos_indices,
        )

        # joint_q is coord-shaped; actual position at coord index 3
        joint_q = _a([0.0, 0.0, 0.0, actual_pos])
        joint_qd = _a([0.0, 0.0])
        # joint_target_pos padded to size 4 so both index 1 (correct) and
        # index 3 (sentinel) are reachable — lets us distinguish the two code paths
        joint_target_pos = _a([0.0, correct_target, 0.0, sentinel])
        joint_target_vel = _a([0.0, 0.0, 0.0, 0.0])
        joint_f = wp.zeros(4, dtype=wp.float32, device=device)

        sim_state = types.SimpleNamespace(joint_q=joint_q, joint_qd=joint_qd)
        sim_control = types.SimpleNamespace(
            joint_target_pos=joint_target_pos,
            joint_target_vel=joint_target_vel,
            joint_act=None,
            joint_f=joint_f,
        )

        actuator.step(sim_state, sim_control, None, None, dt=0.01)

        expected = kp * (correct_target - actual_pos)  # 150.0
        wrong = kp * (sentinel - actual_pos)  # 9850.0
        got = joint_f.numpy()[1]
        self.assertAlmostEqual(
            got,
            expected,
            places=3,
            msg=(
                f"Force should be {expected} (target_pos_indices path); "
                f"got {got}. If {wrong}, pos_indices was wrongly used for target lookup."
            ),
        )


if __name__ == "__main__":
    unittest.main()
