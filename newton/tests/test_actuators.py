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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.actuators.utils import load_metadata
from newton._src.utils.import_usd import parse_usd
from newton.actuators import (
    Actuator,
    ActuatorParsed,
    ClampingBase,
    ClampingDCMotor,
    ClampingMaxEffort,
    ClampingPositionBased,
    Delay,
    DriveBase,
    DriveNeuralLSTM,
    DriveNeuralMLP,
    DrivePD,
    DrivePID,
    ResponseOracle,
    parse_actuator_prim,
)
from newton.selection import ArticulationView

try:
    from pxr import Usd

    HAS_USD = True
except ImportError:
    HAS_USD = False


_HAS_ONNX = importlib.util.find_spec("onnx") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_WARP_NN = importlib.util.find_spec("warp_nn") is not None


if _HAS_TORCH:
    import torch as _torch

    class _LSTMNet(_torch.nn.Module):
        """Minimal LSTM network for exercising the Torch checkpoint path."""

        def __init__(self, hidden: int = 8, layers: int = 1, bidirectional: bool = False):
            super().__init__()
            self.lstm = _torch.nn.LSTM(2, hidden, layers, batch_first=True, bidirectional=bidirectional)
            self.dec = _torch.nn.Linear(hidden, 1)

        def forward(
            self,
            x: _torch.Tensor,
            hc: tuple[_torch.Tensor, _torch.Tensor],
        ) -> tuple[_torch.Tensor, tuple[_torch.Tensor, _torch.Tensor]]:
            out, (h, c) = self.lstm(x, hc)
            return self.dec(out[:, -1, :]), (h, c)


def _onnx_modules():
    """Lazily import ONNX modules used by test model builders."""
    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    return onnx, TensorProto, helper, numpy_helper


def _build_mlp_onnx(
    path: str,
    weights: np.ndarray,
    bias: np.ndarray,
    metadata: dict | None = None,
    batch_dim: int | None = None,
) -> None:
    """Build a single-Gemm ONNX MLP at ``path``."""
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


def _build_elu_mlp_onnx(path: str, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> None:
    """Build a two-layer ONNX MLP with an ELU activation at ``path``."""
    onnx_mod, TensorProto, helper, numpy_helper = _onnx_modules()

    inits = [numpy_helper.from_array(a, n) for a, n in ((w1, "W1"), (b1, "b1"), (w2, "W2"), (b2, "b2"))]
    n1 = helper.make_node("Gemm", ["input", "W1", "b1"], ["hl"], alpha=1.0, beta=1.0, transB=1)
    n2 = helper.make_node("Elu", ["hl"], ["ael"], alpha=1.0)
    n3 = helper.make_node("Gemm", ["ael", "W2", "b2"], ["output"], alpha=1.0, beta=1.0, transB=1)
    x_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, int(w1.shape[1])])
    y_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, int(w2.shape[0])])
    graph = helper.make_graph([n1, n2, n3], "elu_mlp", [x_vi], [y_vi], initializer=inits)
    onnx_mod.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def _build_lstm_onnx(
    path: str,
    hidden_size: int = 8,
    num_layers: int = 1,
    metadata: dict | None = None,
    rng_seed: int = 0,
) -> None:
    """Build a small ONNX LSTM policy model for drive tests."""
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
    squeeze_axes = numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="squeeze_axes")
    initializers.append(squeeze_axes)
    sq = helper.make_node("Squeeze", ["Y", "squeeze_axes"], ["Y_2d"])
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


# Regularize the fixtures' idealized point masses without changing their
# effective dynamics from the inertia that validation previously synthesized.
_POINT_MASS_INERTIA = wp.mat33(1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6)


def _write_dof_values(
    model: newton.Model,
    array: wp.array[float],
    dof_indices: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> None:
    """Write scalar values into specific DOF positions of a Warp array."""
    arr_np = array.numpy()
    for dof, val in zip(dof_indices, values, strict=True):
        arr_np[dof] = val
    wp.copy(array, wp.array(arr_np, dtype=float, device=model.device))


def _build_pendulum(device: wp.Device, worlds: int = 1) -> newton.Model:
    """Single revolute joint with an offset COM and no gravity — one scalar DOF.

    Args:
        device: Device to finalize the model on.
        worlds: Number of identical worlds to replicate the pendulum into.
    """
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = template.add_link(com=wp.vec3(0.5, 0.0, 0.0), inertia=_POINT_MASS_INERTIA, mass=1.0)
    joint = template.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
    template.add_articulation([joint])
    if worlds == 1:
        return template.finalize(device=device)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, worlds, spacing=(0.0, 0.0, 0.0))
    return builder.finalize(device=device)


def _two_link_builder(armature: float = 0.0, dummy_body: bool = False) -> newton.ModelBuilder:
    """Builder for a two-link revolute chain — one articulation, two coupled DOFs.

    Args:
        armature: Rotor inertia added to each joint.
        dummy_body: Add a hinged body before the chain, outside the articulation.
            MuJoCo orders articulated bodies first, so ``mjc_dof_to_newton_dof``
            becomes a permutation rather than the identity.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if dummy_body:
        dummy = builder.add_link(inertia=_POINT_MASS_INERTIA, mass=1.0)
        builder.add_joint_revolute(parent=-1, child=dummy, axis=newton.Axis.Z)
    base = builder.add_link(com=wp.vec3(0.3, 0.0, 0.0), inertia=_POINT_MASS_INERTIA, mass=2.0)
    tip = builder.add_link(com=wp.vec3(0.25, 0.0, 0.0), inertia=_POINT_MASS_INERTIA, mass=1.0)
    j0 = builder.add_joint_revolute(parent=-1, child=base, axis=newton.Axis.Z, armature=armature)
    j1 = builder.add_joint_revolute(
        parent=base,
        child=tip,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.6, 0.0, 0.0), wp.quat_identity()),
        armature=armature,
    )
    builder.add_articulation([j0, j1])
    return builder


def _build_two_link(device: wp.Device, dummy_body: bool = False, worlds: int = 1) -> newton.Model:
    """Two-link revolute chain — one articulation, two inertially coupled DOFs.

    Args:
        device: Device to finalize the model on.
        dummy_body: See :func:`_two_link_builder`.
        worlds: Number of identical worlds to replicate the chain into.
    """
    template = _two_link_builder(dummy_body=dummy_body)
    if worlds == 1:
        return template.finalize(device=device)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, worlds, spacing=(0.0, 0.0, 0.0))
    return builder.finalize(device=device)


def _assert_worlds_match(
    test_case: unittest.TestCase,
    model: newton.Model,
    array: wp.array[float],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Assert every world's slice of a per-DOF array equals world 0's."""
    per_world = _arm_values(model, array).reshape(model.world_count, -1)
    for world in range(1, model.world_count):
        np.testing.assert_allclose(per_world[world], per_world[0], rtol=rtol, atol=atol)


def _response_at(model: newton.Model, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Coupled response A = inv(H) at pose *q*, evaluated on a scratch state."""
    scratch = model.state()
    scratch.joint_q.assign(np.asarray(q, dtype=np.float32))
    scratch.joint_qd.assign(np.asarray(qd, dtype=np.float32))
    newton.eval_fk(model, scratch.joint_q, scratch.joint_qd, scratch)
    n = model.joint_dof_count
    return np.linalg.inv(newton.eval_mass_matrix(model, scratch).numpy()[0, :n, :n])


def _arm_dofs(model: newton.Model) -> np.ndarray:
    """DOF indices belonging to an articulation, in order.

    Equal to ``arange(joint_dof_count)`` unless the model was built with
    ``dummy_body=True``, which prepends a standalone body that is not actuated
    and cannot be: implicit actuation rejects DOFs outside an articulation.
    """
    art_start = model.articulation_start.numpy()
    art_end = model.articulation_end.numpy()
    qd_start = model.joint_qd_start.numpy()
    spans = [np.arange(qd_start[art_start[a]], qd_start[art_end[a]]) for a in range(model.articulation_count)]
    return np.concatenate(spans).astype(np.uint32) if spans else np.empty(0, dtype=np.uint32)


def _set_arm(model: newton.Model, array: wp.array[float], values: Sequence[float] | np.ndarray) -> None:
    """Write *values* into the articulation's slice of a per-DOF array."""
    buf = array.numpy()
    buf[_arm_dofs(model).astype(np.int64)] = np.asarray(values, dtype=np.float32)
    array.assign(buf)


def _arm_values(model: newton.Model, array: wp.array[float]) -> np.ndarray:
    """Read the articulation's slice out of a per-DOF array."""
    return array.numpy()[_arm_dofs(model).astype(np.int64)]


def _mujoco_solver(test_case: unittest.TestCase, model: newton.Model) -> newton.solvers.SolverMuJoCo:
    """Build :class:`SolverMuJoCo`, tolerating the standalone-root advisory.

    Models built with ``dummy_body=True`` carry a joint outside any articulation.
    That is what makes MuJoCo reorder the DOFs, and the solver says so on
    conversion.
    """
    if model.articulation_count and len(_arm_dofs(model)) != model.joint_dof_count:
        with test_case.assertWarnsRegex(UserWarning, "standalone world roots"):
            return newton.solvers.SolverMuJoCo(model, disable_contacts=True)
    return newton.solvers.SolverMuJoCo(model, disable_contacts=True)


def _mujoco_solve(
    solver: newton.solvers.SolverMuJoCo,
) -> Callable[[wp.array2d[float], wp.array2d[float]], None]:
    """``(x, y) -> x = M^-1 y`` backed by MuJoCo's per-step factorization."""
    import mujoco_warp

    def solve_inverse(x: wp.array2d[float], y: wp.array2d[float]) -> None:
        mujoco_warp.solve_m(solver.mjw_model, solver.mjw_data, x, y)

    return solve_inverse


def _response_at_state(model: newton.Model, state: newton.State) -> np.ndarray:
    """Coupled response A = inv(H) at the pose held by *state*."""
    return _response_at(model, state.joint_q.numpy(), state.joint_qd.numpy())


def _expected_implicit_pd(
    model: newton.Model,
    state: newton.State,
    kp: float,
    kd: float,
    target: float,
    dt: float,
) -> float:
    """Closed-form single-DOF implicit PD effort from the pose held by *state*."""
    alpha = _response_at_state(model, state)[0, 0]
    q = float(state.joint_q.numpy()[0])
    qd = float(state.joint_qd.numpy()[0])
    return (kp * (target - q - dt * qd) - kd * qd) / (1.0 + alpha * dt * kd + alpha * dt * dt * kp)


def _make_implicit_actuator(
    model: newton.Model,
    device: wp.Device,
    kp: wp.array[float],
    kd: wp.array[float],
    max_effort: Sequence[float] | np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[Actuator, ResponseOracle]:
    """Build an implicit PD Actuator over all DOFs, with an optional max-effort clamp.

    Returns the actuator together with the response oracle driving its solve.
    """
    clamping = None
    if max_effort is not None:
        clamping = [ClampingMaxEffort(max_effort=wp.array(max_effort, dtype=float, device=device))]
    oracle = kwargs.setdefault("response", ResponseOracle(model))
    actuator = Actuator(
        indices=wp.array(_arm_dofs(model), dtype=wp.uint32, device=device),
        drive=DrivePD(kp=kp, kd=kd),
        clamping=clamping,
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )
    actuator.set_effort_mode_implicit(**kwargs)
    return actuator, oracle


def _refresh_and_step(
    actuator: Actuator,
    oracle: ResponseOracle,
    state: newton.State,
    control: newton.Control,
    dt: float,
) -> None:
    """Refresh the response oracle at *state*, then step the actuator — the simulation order."""
    oracle.refresh(state)
    actuator.step(state, control, dt=dt)


def _ignore_torchscript_deprecation(test_case: unittest.TestCase) -> None:
    """Tolerate torch's TorchScript-family deprecation notices for one test.

    The neural-drive tests deliberately exercise the TorchScript checkpoint
    path (``torch.jit.script``/``save``/``load``), which PyTorch now deprecates in
    favor of ``torch.export``. Ignore just those advisories, scoped to the calling
    test, so strict-warnings mode still surfaces everything else.
    """
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    test_case.addCleanup(ctx.__exit__, None, None, None)
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.jit\..* is deprecated",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Loading (TorchScript|dict) checkpoints .* is deprecated",
        category=DeprecationWarning,
    )


# ---------------------------------------------------------------------------
# 1. Drives
# ---------------------------------------------------------------------------


class TestDrivePD(unittest.TestCase):
    """PD drive: f = constant + act + kp*(target_pos - q) + kd*(target_vel - v)."""

    def test_compute(self):
        """Construct drive directly and call compute() with all terms."""
        n = 2
        kp_vals = [100.0, 200.0]
        kd_vals = [10.0, 20.0]
        const_vals = [5.0, -3.0]
        q = [0.3, -0.5]
        qd = [1.0, -2.0]
        tgt_pos = [1.0, 0.5]
        tgt_vel = [0.0, 1.0]
        ff = [3.0, -1.0]

        def _f(vals: Sequence[float]) -> wp.array[float]:
            return wp.array(vals, dtype=wp.float32)

        indices = wp.array(list(range(n)), dtype=wp.uint32)
        ctrl = DrivePD(kp=_f(kp_vals), kd=_f(kd_vals), const_effort=_f(const_vals))
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


class TestDrivePID(unittest.TestCase):
    """PID drive: f = const + act + kp*e + ki*integral + kd*de."""

    def test_compute(self):
        """Construct drive directly and call compute() over multiple steps."""
        kp, ki, kd, const = 50.0, 10.0, 5.0, 2.0
        dt = 0.01
        q, qd = [0.0], [0.0]
        tgt_pos, tgt_vel = [1.0], [0.0]
        pos_error = tgt_pos[0] - q[0]
        vel_error = tgt_vel[0] - qd[0]
        device = wp.get_device()

        def _f(vals: Sequence[float]) -> wp.array[float]:
            return wp.array(vals, dtype=wp.float32, device=device)

        indices = wp.array([0], dtype=wp.uint32, device=device)
        ctrl = DrivePID(
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


@unittest.skipUnless(_HAS_ONNX and _HAS_WARP_NN, "onnx or warp-nn not installed")
class TestDriveNeuralMLP(unittest.TestCase):
    """DriveNeuralMLP - load via model_path, call compute() directly."""

    def setUp(self):
        self.device = wp.get_device()
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _save_mlp(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        filename: str = "mlp.onnx",
        metadata: dict | None = None,
        batch_dim: int | None = None,
    ) -> str:
        path = os.path.join(self._tmp_dir, filename)
        _build_mlp_onnx(path, weights, bias, metadata, batch_dim=batch_dim)
        return path

    def test_compute(self):
        """Constant-bias network produces known output; history rolls after update_state."""
        weights = np.zeros((1, 2), dtype=np.float32)
        bias = np.array([42.0], dtype=np.float32)
        path = self._save_mlp(weights, bias)
        n = 1
        ctrl = DriveNeuralMLP(model_path=path)
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

    def test_velocity_input_is_raw_joint_velocity(self):
        """Network receives raw joint velocity, not velocity error (target_vel must not affect it)."""
        weights = np.array([[0.0, 1.0]], dtype=np.float32)  # output = velocity feature
        bias = np.zeros((1,), dtype=np.float32)
        path = self._save_mlp(weights, bias)
        n = 1
        ctrl = DriveNeuralMLP(model_path=path)
        ctrl.finalize(self.device, n)
        state_a = ctrl.state(n, self.device)
        state_b = ctrl.state(n, self.device)

        q, qd = 0.5, 2.0
        target_q, target_qd = q, 5.0  # zero pos error; target_qd must not enter the network input
        expected = weights[0, 0] * (target_q - q) + weights[0, 1] * qd + bias[0]

        indices = wp.array([0], dtype=wp.uint32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)
        ctrl.compute(
            wp.array([q], dtype=wp.float32, device=self.device),
            wp.array([qd], dtype=wp.float32, device=self.device),
            wp.array([target_q], dtype=wp.float32, device=self.device),
            wp.array([target_qd], dtype=wp.float32, device=self.device),
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
        self.assertAlmostEqual(forces.numpy()[0], expected, places=3, msg="input must be joint velocity, not vel error")

        ctrl.update_state(state_a, state_b)
        self.assertAlmostEqual(
            float(state_b.vel_history.numpy()[0, 0]),
            qd,
            places=4,
            msg="history should contain raw joint velocity from current step",
        )

    def test_metadata_scales(self):
        """Metadata effort_scale is applied to the network output."""
        weights = np.zeros((1, 2), dtype=np.float32)
        bias = np.array([10.0], dtype=np.float32)
        path = self._save_mlp(weights, bias, metadata={"effort_scale": 3.0})

        n = 1
        ctrl = DriveNeuralMLP(model_path=path)
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

    def test_corrupt_single_metadata_property_raises(self):
        """A corrupt JSON metadata blob must not silently fall back to defaults."""
        weights = np.zeros((1, 2), dtype=np.float32)
        bias = np.zeros((1,), dtype=np.float32)
        path = self._save_mlp(weights, bias, metadata={"effort_scale": 1.0})

        onnx_mod, _, _, _ = _onnx_modules()
        model = onnx_mod.load(path)
        model.metadata_props[0].value = "{"
        onnx_mod.save(model, path)

        with self.assertRaisesRegex(ValueError, "Invalid JSON.*metadata.*mlp.onnx"):
            DriveNeuralMLP(model_path=path)

    def test_non_mapping_single_metadata_property_raises(self):
        weights = np.zeros((1, 2), dtype=np.float32)
        bias = np.zeros((1,), dtype=np.float32)
        path = self._save_mlp(weights, bias, metadata={"effort_scale": 1.0})

        onnx_mod, _, _, _ = _onnx_modules()
        model = onnx_mod.load(path)
        model.metadata_props[0].value = json.dumps(["not", "a", "mapping"])
        onnx_mod.save(model, path)

        with self.assertRaisesRegex(ValueError, "mlp.onnx.*expected a JSON object"):
            DriveNeuralMLP(model_path=path)

    def test_invalid_scale_metadata_names_key_and_path(self):
        weights = np.zeros((1, 2), dtype=np.float32)
        bias = np.zeros((1,), dtype=np.float32)
        path = self._save_mlp(weights, bias, metadata={"effort_scale": None})

        with self.assertRaisesRegex(ValueError, "effort_scale.*mlp.onnx"):
            DriveNeuralMLP(model_path=path)

        path = self._save_mlp(weights, bias, filename="zero_scale.onnx", metadata={"effort_scale": 0.0})
        with self.assertRaisesRegex(ValueError, "effort_scale.*zero_scale.onnx"):
            DriveNeuralMLP(model_path=path)

    def test_finalize_fixed_batch_onnx_with_multiple_actuators(self):
        """Fixed-batch ONNX exports can still run one scalar per actuator."""
        weights = np.array([[2.0, 0.0]], dtype=np.float32)
        bias = np.array([1.0], dtype=np.float32)
        path = self._save_mlp(weights, bias, filename="fixed_batch_mlp.onnx", batch_dim=1)

        n = 3
        ctrl = DriveNeuralMLP(model_path=path)
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

    def test_neural_mlp_implicit_linear_net(self):
        """A 1-layer (linear) neural drive solves implicitly, exact.

        For a linear net (tau = w0*pos_err + w1*vel_err + b) the linearization is
        the net itself, so the solve matches the analytic Stable-PD solution with
        kp = w0, kd = w1, const = b.
        """
        device = wp.get_device()
        h = 0.01
        w0, w1, b = 400.0, 8.0, 2.5
        q0, target = 0.2, 1.0

        model = _build_pendulum(device)
        state = model.state()
        state.joint_q.assign(np.array([q0], dtype=np.float32))
        control = model.control()
        control.joint_target_q.assign(np.array([target], dtype=np.float32))

        path = self._save_mlp(
            np.array([[w0, w1]], dtype=np.float32), np.array([b], dtype=np.float32), filename="implicit_linear.onnx"
        )
        drive = DriveNeuralMLP(model_path=path)
        oracle = ResponseOracle(model)
        actuator = Actuator(
            indices=wp.array([0], dtype=wp.uint32, device=device),
            drive=drive,
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        actuator.set_effort_mode_implicit(response=oracle)
        self.assertTrue(actuator.is_graphable())

        oracle.refresh(state)
        state_a, state_b = actuator.state(), actuator.state()
        control.joint_f.zero_()
        actuator.step(state, control, state_a, state_b, dt=h)

        alpha = _response_at_state(model, state)[0, 0]
        e_q = target - q0
        expected_tau = (w0 * e_q + b) / (1.0 - alpha * h * w1 + alpha * h * h * w0)
        self.assertAlmostEqual(control.joint_f.numpy()[0], expected_tau, delta=abs(expected_tau) * 1e-4)

        # Same step under graph capture, with no warm-up: the per-step wp.Tape
        # backward and the network's gradient buffers must both be capture-safe.
        if device.is_cuda:
            fresh = DriveNeuralMLP(model_path=path)
            captured = Actuator(
                indices=wp.array([0], dtype=wp.uint32, device=device),
                drive=fresh,
                control_target_pos_attr="joint_target_q",
                control_target_vel_attr="joint_target_qd",
            )
            captured.set_effort_mode_implicit(response=oracle)
            cap_a, cap_b = captured.state(), captured.state()
            control.joint_f.zero_()
            with wp.ScopedCapture() as capture:
                captured.step(state, control, cap_a, cap_b, dt=h)
            wp.capture_launch(capture.graph)
            self.assertAlmostEqual(control.joint_f.numpy()[0], expected_tau, delta=abs(expected_tau) * 1e-4)

    def test_neural_mlp_implicit_multi_dof_uses_per_dof_response(self):
        """Each DOF must be capped with its own inverse mass, not DOF 0's.

        The Jacobian guard scales a network's slopes using that DOF's response
        ``alpha_i``. Every other implicit neural test is single-DOF, so the
        per-slot gather always reads index 0 and a wrong local index is
        invisible. Here a velocity-only net is made stiff enough that the guard
        binds on a coupled two-link chain, where the two DOFs have different
        ``alpha``: the capped slope is ``0.9 / (dt * alpha_i)``, so using the
        wrong one changes the answer.
        """
        device = wp.get_device()
        h = 0.01
        w0, w1, bias = 0.0, 400.0, 1.5
        margin = DriveNeuralMLP.IMPLICIT_JACOBIAN_MARGIN
        q0 = np.array([0.0, 0.0], dtype=np.float32)
        qd0 = np.array([3.0, -2.0], dtype=np.float32)

        model = _build_two_link(device)
        state = model.state()
        state.joint_q.assign(q0)
        state.joint_qd.assign(qd0)
        control = model.control()
        control.joint_target_q.assign(np.zeros(2, dtype=np.float32))

        path = self._save_mlp(
            np.array([[w0, w1]], dtype=np.float32),
            np.array([bias], dtype=np.float32),
            filename="implicit_multidof.onnx",
        )
        oracle = ResponseOracle(model)
        actuator = Actuator(
            indices=wp.array([0, 1], dtype=wp.uint32, device=device),
            drive=DriveNeuralMLP(model_path=path),
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        actuator.set_effort_mode_implicit(response=oracle)

        oracle.refresh(state)
        state_a, state_b = actuator.state(), actuator.state()
        control.joint_f.zero_()
        actuator.step(state, control, state_a, state_b, dt=h)

        response = _response_at_state(model, state)
        alpha = np.diag(response)
        # Guard: slope = a*dt + b is capped to (1 - margin) / (dt * alpha_i).
        slope = -w0 * h + w1
        limit = (1.0 - margin) / (h * alpha)
        self.assertTrue(np.all(slope > limit), "the guard must bind for this test to mean anything")
        self.assertGreater(abs(limit[0] - limit[1]), 1e-3 * abs(limit[0]), "the two DOFs must differ")
        b_capped = w1 * (limit / slope)

        # Affine law: (I - h*diag(b_capped) @ A) p = h*tau0, and effort = p/h.
        tau0 = w1 * qd0 + bias
        p = np.linalg.solve(np.eye(2) - h * (b_capped[:, None] * response), h * tau0)
        np.testing.assert_allclose(control.joint_f.numpy(), p / h, rtol=2e-3, atol=1e-4)

    def test_neural_mlp_implicit_nonlinear_linearized(self):
        """A nonlinear neural drive enters the solve as a linearization.

        Builds a 2-layer ELU net. Implicit actuation linearizes it once about the
        current state, ``tau ~= tau0 + a*(q-q0) + b*(qd-qd0)`` with
        ``a = d(tau)/dq``, ``b = d(tau)/dqd``, then solves the resulting linear
        Stable-PD system exactly. The solved effort must match the closed-form
        solution of that linear system.
        """
        device = wp.get_device()
        h = 0.01
        q0, qd0, target = 0.2, 0.0, 1.0

        rng = np.random.default_rng(0)
        w1 = (rng.standard_normal((4, 2)) * 3.0).astype(np.float32)
        b1 = (rng.standard_normal(4) * 0.5).astype(np.float32)
        w2 = (rng.standard_normal((1, 4)) * 4.0).astype(np.float32)
        b2 = np.array([3.0], dtype=np.float32)

        def net_np(e_q: float, e_qd: float) -> float:
            x = np.array([e_q, e_qd], dtype=np.float32)
            hl = w1 @ x + b1
            a = np.where(hl >= 0.0, hl, np.exp(hl) - 1.0)  # ELU, alpha=1
            return float((w2 @ a + b2)[0])

        def dnet_np(e_q: float, e_qd: float) -> tuple[float, float]:
            # d(net)/d(e_q), d(net)/d(e_qd); ELU'(x) = 1 (x>=0) else exp(x)
            hl = w1 @ np.array([e_q, e_qd], dtype=np.float32) + b1
            elu_p = np.where(hl >= 0.0, 1.0, np.exp(hl))
            g = w2[0] * elu_p  # (4,)
            return float(g @ w1[:, 0]), float(g @ w1[:, 1])

        model = _build_pendulum(device)
        state = model.state()
        state.joint_q.assign(np.array([q0], dtype=np.float32))
        state.joint_qd.assign(np.array([qd0], dtype=np.float32))
        control = model.control()
        control.joint_target_q.assign(np.array([target], dtype=np.float32))
        alpha = _response_at_state(model, state)[0, 0]

        # Linearize tau(q, qd) = net(target - q, qd) about (q0, qd0). The first
        # feature is a position error, so the chain rule negates its slope; the
        # second is the raw velocity, so its slope carries through unchanged:
        #   a = d(tau)/dq = -d(net)/d(feat_pos),  b = d(tau)/dqd = +d(net)/d(feat_vel).
        tau0 = net_np(target - q0, qd0)
        dneq, dneqd = dnet_np(target - q0, qd0)
        a, b = -dneq, dneqd
        # Solve p/h from p = h*(tau0 + a*(q(p)-q0) + b*(qd(p)-qd0)),
        #   qd(p) = qd0 + alpha*p, q(p) = q0 + h*qd(p).
        expected_tau = (tau0 + a * h * qd0) / (1.0 - alpha * h * (a * h + b))

        path = os.path.join(self._tmp_dir, "implicit_nonlinear.onnx")
        _build_elu_mlp_onnx(path, w1, b1, w2, b2)
        drive = DriveNeuralMLP(model_path=path)
        oracle = ResponseOracle(model)
        actuator = Actuator(
            indices=wp.array([0], dtype=wp.uint32, device=device),
            drive=drive,
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        actuator.set_effort_mode_implicit(response=oracle)
        oracle.refresh(state)
        sa, sb = actuator.state(), actuator.state()
        control.joint_f.zero_()
        actuator.step(state, control, sa, sb, dt=h)
        self.assertAlmostEqual(float(control.joint_f.numpy()[0]), expected_tau, delta=abs(expected_tau) * 3e-3)


@unittest.skipUnless(_HAS_ONNX and _HAS_WARP_NN, "onnx or warp-nn not installed")
class TestDriveNeuralLSTM(unittest.TestCase):
    """DriveNeuralLSTM - load via model_path, call compute() directly."""

    def setUp(self):
        self.device = wp.get_device()
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _save_lstm(self, filename: str = "lstm.onnx", hidden: int = 8, metadata: dict | None = None) -> str:
        path = os.path.join(self._tmp_dir, filename)
        _build_lstm_onnx(path, hidden_size=hidden, num_layers=1, metadata=metadata)
        return path

    def _run_lstm_compute(self, ctrl: DriveNeuralLSTM) -> None:
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
        path = self._save_lstm()
        ctrl = DriveNeuralLSTM(model_path=path)
        self._run_lstm_compute(ctrl)

    def test_metadata_scales(self):
        metadata = {"pos_scale": 2.0, "vel_scale": 0.5, "effort_scale": 10.0}
        path = self._save_lstm(metadata=metadata)

        ctrl = DriveNeuralLSTM(model_path=path)
        self.assertAlmostEqual(ctrl.pos_scale, 2.0)
        self.assertAlmostEqual(ctrl.vel_scale, 0.5)
        self.assertAlmostEqual(ctrl.effort_scale, 10.0)

        self._run_lstm_compute(ctrl)

    def test_invalid_scale_metadata_names_key_and_path(self):
        path = self._save_lstm(filename="invalid_lstm.onnx", metadata={"vel_scale": float("inf")})

        with self.assertRaisesRegex(ValueError, "vel_scale.*invalid_lstm.onnx"):
            DriveNeuralLSTM(model_path=path)

    def test_neural_lstm_implicit_linearized(self):
        """The LSTM enters the implicit solve as a per-step linearization.

        A pack with zero slopes is the failure to watch for: it makes the law
        the constant ``tau0``, which reduces the solve to the explicit impulse
        while still paying for the Newton loop, and no assertion on the solved
        effort alone would notice.
        """
        device = wp.get_device()
        h = 0.01
        q0, qd0, target = 0.2, 0.0, 1.0

        model = _build_pendulum(device)
        state = model.state()
        state.joint_q.assign(np.array([q0], dtype=np.float32))
        state.joint_qd.assign(np.array([qd0], dtype=np.float32))
        control = model.control()
        control.joint_target_q.assign(np.array([target], dtype=np.float32))

        path = self._save_lstm(filename="implicit_lstm.onnx", metadata={"effort_scale": 10.0})
        drive = DriveNeuralLSTM(model_path=path)
        oracle = ResponseOracle(model)
        actuator = Actuator(
            indices=wp.array([0], dtype=wp.uint32, device=device),
            drive=drive,
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )

        actuator.set_effort_mode_implicit(response=oracle)
        oracle.refresh(state)
        sa, sb = actuator.state(), actuator.state()
        control.joint_f.zero_()
        actuator.step(state, control, sa, sb, dt=h)

        pack = drive._lin_params.numpy()
        self.assertEqual(pack.shape[1], 5)  # [tau0, a, b, q0, qd0]
        tau0, a, b, pq0, pqd0 = (float(v) for v in pack[0])
        self.assertAlmostEqual(pq0, q0, delta=1e-6)
        self.assertAlmostEqual(pqd0, qd0, delta=1e-6)
        self.assertGreater(abs(a) + abs(b), 0.0)  # not a constant law

        # The solve must match the closed form of the affine law it was handed.
        alpha = _response_at_state(model, state)[0, 0]
        expected = (tau0 + a * h * pqd0) / (1.0 - alpha * h * (a * h + b))
        tau = float(control.joint_f.numpy()[0])
        self.assertTrue(np.isfinite(tau))
        self.assertAlmostEqual(tau, expected, delta=abs(expected) * 1e-3 + 1e-6)
        self.assertTrue(np.any(sb.drive_state.hidden.numpy() != 0.0))

        # Finite differences of the drive's own output pin the sign and the
        # scaling of the slopes. They are compared against the raw slopes, not
        # the packed ones, which may be scaled down to bound the Jacobian.
        actuator.set_effort_mode_explicit()

        def explicit_tau(q_val: float, qd_val: float) -> float:
            state.joint_q.assign(np.array([q_val], dtype=np.float32))
            state.joint_qd.assign(np.array([qd_val], dtype=np.float32))
            control.joint_f.zero_()
            actuator.step(state, control, actuator.state(), actuator.state(), dt=h)
            return float(control.joint_f.numpy()[0])

        eps = 1e-3
        fd_dq = (explicit_tau(q0 + eps, qd0) - explicit_tau(q0 - eps, qd0)) / (2.0 * eps)
        fd_dqd = (explicit_tau(q0, qd0 + eps) - explicit_tau(q0, qd0 - eps)) / (2.0 * eps)
        self.assertAlmostEqual(float(drive._dtau_dq.numpy()[0]), fd_dq, delta=abs(fd_dq) * 0.02 + 1e-3)
        self.assertAlmostEqual(float(drive._dtau_dqd.numpy()[0]), fd_dqd, delta=abs(fd_dqd) * 0.02 + 1e-3)


class _TorchCheckpointTestMixin:
    """Shared helpers for saving pt2 / TorchScript / dict torch checkpoints."""

    def setUp(self):
        import torch

        self.device = wp.get_device()
        if self.device.is_cuda and not torch.cuda.is_available():
            self.skipTest("Torch not compiled with CUDA support")
        self.torch = torch
        _ignore_torchscript_deprecation(self)
        self._torch_dev = torch.device(f"cuda:{self.device.ordinal}" if self.device.is_cuda else "cpu")
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _save_torchscript(self, net: Any, filename: str = "model.pt", metadata: dict | None = None) -> str:
        path = os.path.join(self._tmp_dir, filename)
        scripted = self.torch.jit.script(net)
        extra = {"metadata.json": json.dumps(metadata)} if metadata else {}
        self.torch.jit.save(scripted, path, _extra_files=extra)
        return path

    def _save_dict(self, net: Any, filename: str = "model_dict.pt", metadata: dict | None = None) -> str:
        path = os.path.join(self._tmp_dir, filename)
        self.torch.save({"model": net, "metadata": metadata or {}}, path)
        return path

    def _export_pt2(
        self,
        net: Any,
        example_inputs: tuple[Any, ...],
        dynamic_shapes: tuple[Any, ...] | None,
        filename: str,
        metadata: dict | None = None,
    ) -> str:
        path = os.path.join(self._tmp_dir, filename)
        net.eval()
        exported = self.torch.export.export(net, example_inputs, dynamic_shapes=dynamic_shapes)
        extra = {"metadata.json": json.dumps(metadata)} if metadata else None
        self.torch.export.save(exported, path, extra_files=extra)
        return path


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestDriveNeuralMLPTorchFormats(_TorchCheckpointTestMixin, unittest.TestCase):
    """DriveNeuralMLP loading from pt2, TorchScript, and dict checkpoints."""

    def _make_mlp(self, bias: float = 0.0) -> Any:
        net = self.torch.nn.Sequential(self.torch.nn.Linear(2, 1, bias=True)).to(self._torch_dev)
        with self.torch.no_grad():
            net[0].weight.fill_(0.0)
            net[0].bias.fill_(bias)
        return net

    def _save_pt2(self, net: Any, filename: str = "mlp.pt2", metadata: dict | None = None) -> str:
        example = (self.torch.randn(2, 2, device=self._torch_dev),)
        batch = self.torch.export.Dim("batch", min=1)
        return self._export_pt2(net, example, ({0: batch},), filename, metadata=metadata)

    def test_dict_checkpoint(self):
        """Load MLP from a dict checkpoint with metadata."""
        path = self._save_dict(self._make_mlp(bias=5.0), metadata={"effort_scale": 4.0})
        ctrl = DriveNeuralMLP(model_path=path)
        self.assertAlmostEqual(ctrl.effort_scale, 4.0)

    def test_pt2_checkpoint(self):
        """Load MLP from a pt2 archive with metadata and run compute."""
        path = self._save_pt2(self._make_mlp(bias=7.0), metadata={"effort_scale": 2.0})
        n = 1
        ctrl = DriveNeuralMLP(model_path=path)
        self.assertAlmostEqual(ctrl.effort_scale, 2.0)
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
        self.assertAlmostEqual(forces.numpy()[0], 14.0, places=3, msg="bias=7 * effort_scale=2 -> 14")

    def test_legacy_formats_warn(self):
        """TorchScript and dict checkpoints emit a DeprecationWarning on load."""
        ts_path = self._save_torchscript(self._make_mlp())
        dict_path = self._save_dict(self._make_mlp())

        with self.assertWarnsRegex(DeprecationWarning, "TorchScript checkpoints"):
            DriveNeuralMLP(model_path=ts_path)
        with self.assertWarnsRegex(DeprecationWarning, "dict checkpoints"):
            DriveNeuralMLP(model_path=dict_path)

    def test_deprecation_warning_points_at_caller(self):
        """The legacy-format warning is attributed to the calling code, not newton internals."""
        path = self._save_torchscript(self._make_mlp())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DriveNeuralMLP(model_path=path)
        hits = [w for w in caught if "TorchScript checkpoints" in str(w.message)]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].filename, __file__)

    def test_load_metadata_reads_zip_entry_without_warning(self):
        """Metadata-only reads do not deserialize the network or warn about legacy formats."""
        path = self._save_torchscript(self._make_mlp(), metadata={"effort_scale": 3.0})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metadata = load_metadata(path)
        self.assertEqual(metadata, {"effort_scale": 3.0})
        self.assertFalse([w for w in caught if "checkpoints" in str(w.message)])

    def test_dict_checkpoint_uses_target_pos_indices(self):
        """Verify target_pos uses target_pos_indices, not sequential or pos_indices.

        Regression test for a bug where the Torch path fell back to sequential indices
        into target_pos whenever target_pos_indices was not literally the same array
        object as pos_indices. This matters in practice for a floating-base robot: the
        free joint occupies 7 coordinate ("q", position-layout) DOFs but only 6 velocity
        ("qd") DOFs, so ``pos_indices`` (coord layout) and ``indices``/``target_pos_indices``
        (legacy DOF layout, see ``newton.use_coord_layout_targets``) are offset
        differently for the two actuated joints that follow it, and neither equals a plain
        ``arange(n)``.
        """
        self.torch.manual_seed(0)
        net = self.torch.nn.Sequential(self.torch.nn.Linear(2, 1, bias=True)).to(self._torch_dev)
        path = self._save_dict(net, metadata={"effort_scale": 1.0})
        ctrl = DriveNeuralMLP(model_path=path)
        n = 2
        ctrl.finalize(self.device, n)
        state_a = ctrl.state(n, self.device)

        # joint_q layout: 7 free-joint DOFs, then the 2 actuated revolute joints at 7, 8.
        positions = wp.array([0.0] * 7 + [0.3, -0.2], dtype=wp.float32, device=self.device)
        pos_indices = wp.array([7, 8], dtype=wp.uint32, device=self.device)
        # joint_qd / legacy target layout: 6 free-joint DOFs, then the actuated joints at 6, 7.
        velocities = wp.zeros(8, dtype=wp.float32, device=self.device)
        vel_indices = wp.array([6, 7], dtype=wp.uint32, device=self.device)
        target_pos = wp.array([0.0] * 6 + [1.0, 2.0], dtype=wp.float32, device=self.device)
        target_pos_indices = wp.array([6, 7], dtype=wp.uint32, device=self.device)
        target_vel = wp.zeros(n, dtype=wp.float32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)

        ctrl.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            None,
            pos_indices,
            vel_indices,
            target_pos_indices,
            target_pos_indices,
            forces,
            state_a,
            0.01,
            self.device,
        )

        pos_error = self.torch.tensor([0.7, 2.2], device=self._torch_dev)  # target_pos[[6, 7]] - positions[[7, 8]]
        vel = self.torch.tensor([0.0, 0.0], device=self._torch_dev)
        net_input = self.torch.stack([pos_error, vel], dim=1)
        with self.torch.inference_mode():
            expected = net(net_input).reshape(n)

        np.testing.assert_allclose(forces.numpy(), expected.cpu().numpy(), rtol=1e-4, atol=1e-6)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestDriveNeuralLSTMTorchFormats(_TorchCheckpointTestMixin, unittest.TestCase):
    """DriveNeuralLSTM loading from pt2, TorchScript, and dict checkpoints."""

    def _make_lstm(self, hidden: int = 8, layers: int = 1, bidirectional: bool = False) -> Any:
        return _LSTMNet(hidden=hidden, layers=layers, bidirectional=bidirectional).to(self._torch_dev)

    def _save_pt2(self, net: Any, filename: str = "lstm.pt2", metadata: dict | None = None) -> str:
        layers, hidden = net.lstm.num_layers, net.lstm.hidden_size
        n = 2
        x = self.torch.randn(n, 1, 2, device=self._torch_dev)
        h = self.torch.zeros(layers, n, hidden, device=self._torch_dev)
        c = self.torch.zeros(layers, n, hidden, device=self._torch_dev)
        batch = self.torch.export.Dim("batch", min=1)
        dynamic_shapes = ({0: batch}, ({1: batch}, {1: batch}))
        return self._export_pt2(net, (x, (h, c)), dynamic_shapes, filename, metadata=metadata)

    def _run_lstm_compute(self, ctrl: DriveNeuralLSTM) -> None:
        n = 1
        ctrl.finalize(self.device, n)

        state_a = ctrl.state(n, self.device)
        state_b = ctrl.state(n, self.device)
        self.assertTrue(self.torch.all(state_a.hidden == 0.0).item())

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
        self.assertFalse(self.torch.all(state_b.hidden == 0.0).item(), "hidden state should evolve")
        return forces.numpy()[0]

    def test_dict_checkpoint(self):
        """Load LSTM from a dict checkpoint with metadata."""
        path = self._save_dict(self._make_lstm(hidden=8, layers=1), metadata={"effort_scale": 5.0})
        ctrl = DriveNeuralLSTM(model_path=path)
        self.assertAlmostEqual(ctrl.effort_scale, 5.0)
        self._run_lstm_compute(ctrl)

    def test_pt2_checkpoint(self):
        """Load LSTM from a pt2 archive; layer config comes from metadata."""
        metadata = {"effort_scale": 5.0, "num_layers": 2, "hidden_size": 8}
        path = self._save_pt2(self._make_lstm(hidden=8, layers=2), metadata=metadata)
        ctrl = DriveNeuralLSTM(model_path=path)
        self.assertAlmostEqual(ctrl.effort_scale, 5.0)
        self.assertEqual(ctrl._num_layers, 2)
        self.assertEqual(ctrl._hidden_size, 8)
        self._run_lstm_compute(ctrl)

    def test_pt2_without_config_metadata_raises(self):
        """A pt2 checkpoint lacking num_layers/hidden_size fails with clear guidance."""
        path = self._save_pt2(self._make_lstm(hidden=8, layers=2), metadata={"effort_scale": 5.0})
        with self.assertRaisesRegex(ValueError, "num_layers.*hidden_size"):
            DriveNeuralLSTM(model_path=path)

    def test_pt2_metadata_config_coerced_to_int(self):
        """JSON floats for num_layers/hidden_size are coerced to int."""
        metadata = {"num_layers": 2.0, "hidden_size": 8.0}
        path = self._save_pt2(self._make_lstm(hidden=8, layers=2), metadata=metadata)
        ctrl = DriveNeuralLSTM(model_path=path)
        self.assertIsInstance(ctrl._num_layers, int)
        self.assertIsInstance(ctrl._hidden_size, int)
        self.assertEqual(ctrl._num_layers, 2)
        self.assertEqual(ctrl._hidden_size, 8)

    def test_metadata_config_mismatch_raises(self):
        """Metadata that contradicts the network's actual LSTM fails at load."""
        path = self._save_dict(self._make_lstm(hidden=8, layers=1), metadata={"num_layers": 2, "hidden_size": 8})
        with self.assertRaisesRegex(ValueError, "num_layers"):
            DriveNeuralLSTM(model_path=path)

    def test_invalid_lstm_not_masked_by_config_metadata(self):
        """Structural validation still runs when metadata provides the LSTM config."""
        net = self._make_lstm(hidden=8, layers=1, bidirectional=True)
        path = self._save_dict(net, metadata={"num_layers": 1, "hidden_size": 8})
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            DriveNeuralLSTM(model_path=path)

    def test_legacy_formats_warn(self):
        """TorchScript and dict checkpoints emit a DeprecationWarning on load."""
        ts_path = self._save_torchscript(self._make_lstm(hidden=8, layers=1))
        dict_path = self._save_dict(self._make_lstm(hidden=8, layers=1))

        with self.assertWarnsRegex(DeprecationWarning, "TorchScript checkpoints"):
            DriveNeuralLSTM(model_path=ts_path)
        with self.assertWarnsRegex(DeprecationWarning, "dict checkpoints"):
            DriveNeuralLSTM(model_path=dict_path)

    def test_dict_checkpoint_uses_target_pos_indices(self):
        """Verify target_pos uses target_pos_indices, not sequential or pos_indices.

        Regression test for a bug where the Torch path fell back to sequential indices
        into target_pos whenever target_pos_indices was not literally the same array
        object as pos_indices. This matters in practice for a floating-base robot: the
        free joint occupies 7 coordinate ("q", position-layout) DOFs but only 6 velocity
        ("qd") DOFs, so ``pos_indices`` (coord layout) and ``indices``/``target_pos_indices``
        (legacy DOF layout, see ``newton.use_coord_layout_targets``) are offset
        differently for the two actuated joints that follow it, and neither equals a plain
        ``arange(n)``.
        """
        net = self._make_lstm(hidden=4, layers=1)
        path = self._save_dict(net, metadata={"effort_scale": 1.0})
        ctrl = DriveNeuralLSTM(model_path=path)
        n = 2
        ctrl.finalize(self.device, n)
        state_a = ctrl.state(n, self.device)

        # joint_q layout: 7 free-joint DOFs, then the 2 actuated revolute joints at 7, 8.
        positions = wp.array([0.0] * 7 + [0.3, -0.2], dtype=wp.float32, device=self.device)
        pos_indices = wp.array([7, 8], dtype=wp.uint32, device=self.device)
        # joint_qd / legacy target layout: 6 free-joint DOFs, then the actuated joints at 6, 7.
        velocities = wp.zeros(8, dtype=wp.float32, device=self.device)
        vel_indices = wp.array([6, 7], dtype=wp.uint32, device=self.device)
        target_pos = wp.array([0.0] * 6 + [1.0, 2.0], dtype=wp.float32, device=self.device)
        target_pos_indices = wp.array([6, 7], dtype=wp.uint32, device=self.device)
        target_vel = wp.zeros(n, dtype=wp.float32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)

        ctrl.compute(
            positions,
            velocities,
            target_pos,
            target_vel,
            None,
            pos_indices,
            vel_indices,
            target_pos_indices,
            target_pos_indices,
            forces,
            state_a,
            0.01,
            self.device,
        )

        pos_error = self.torch.tensor([0.7, 2.2], device=self._torch_dev)  # target_pos[[6, 7]] - positions[[7, 8]]
        vel = self.torch.tensor([0.0, 0.0], device=self._torch_dev)
        net_input = self.torch.stack([pos_error, vel], dim=1).unsqueeze(1)
        h = self.torch.zeros(1, n, 4, device=self._torch_dev)
        c = self.torch.zeros(1, n, 4, device=self._torch_dev)
        with self.torch.inference_mode():
            expected, _ = net(net_input, (h, c))
        expected = expected.reshape(n)

        np.testing.assert_allclose(forces.numpy(), expected.cpu().numpy(), rtol=1e-4, atol=1e-6)

    def test_masked_reset_after_inference_mode_output(self):
        """Verify masked reset clears selected LSTM state without raising.

        Regression test: masked resets used to write in-place into the hidden/cell tensors
        produced by the network under torch.inference_mode(), which raised
        ``RuntimeError: Inplace update to inference tensor outside InferenceMode is not
        allowed``.
        """
        net = self._make_lstm(hidden=4, layers=1)
        path = self._save_dict(net, metadata={"effort_scale": 1.0})
        ctrl = DriveNeuralLSTM(model_path=path)
        n = 3
        ctrl.finalize(self.device, n)
        state_a = ctrl.state(n, self.device)
        state_b = ctrl.state(n, self.device)

        indices = wp.array(list(range(n)), dtype=wp.uint32, device=self.device)
        positions = wp.zeros(n, dtype=wp.float32, device=self.device)
        velocities = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.device)
        target_pos = wp.array([1.0, 1.0, 1.0], dtype=wp.float32, device=self.device)
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
        self.assertTrue(state_b.hidden.is_inference(), "precondition: hidden must be a network output")

        hidden_before = state_b.hidden.clone()
        cell_before = state_b.cell.clone()

        masked = [True, False, True]
        mask = wp.array(masked, dtype=wp.bool, device=self.device)
        state_b.reset(mask)  # must not raise RuntimeError

        for b, is_masked in enumerate(masked):
            if is_masked:
                self.assertTrue(self.torch.all(state_b.hidden[:, b, :] == 0.0).item(), f"actuator {b} not zeroed")
                self.assertTrue(self.torch.all(state_b.cell[:, b, :] == 0.0).item(), f"actuator {b} not zeroed")
            else:
                self.assertTrue(
                    self.torch.equal(state_b.hidden[:, b, :], hidden_before[:, b, :]), f"actuator {b} not preserved"
                )
                self.assertTrue(
                    self.torch.equal(state_b.cell[:, b, :], cell_before[:, b, :]), f"actuator {b} not preserved"
                )


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestDriveNeuralMLPLegacyTorchScript(unittest.TestCase):
    """Regression tests for the supported .pt MLP checkpoint path."""

    def setUp(self):
        self.device = wp.get_device()
        self._tmp_dir = tempfile.mkdtemp()
        _ignore_torchscript_deprecation(self)

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_finalize_legacy_torchscript_checkpoint(self):
        """.pt checkpoints keep the Torch backend and state interface."""
        import torch

        n = 1
        in_features = 2

        class _BiasOnlyMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(in_features, 1, bias=True)
                with torch.no_grad():
                    self.fc.weight.zero_()
                    self.fc.bias.fill_(7.0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        model = _BiasOnlyMLP().eval()
        scripted = torch.jit.script(model)
        path = os.path.join(self._tmp_dir, "legacy_mlp.pt")
        scripted.save(path, _extra_files={"metadata.json": json.dumps({"effort_scale": 1.0})})

        ctrl = DriveNeuralMLP(model_path=path)
        ctrl.finalize(self.device, n)

        self.assertFalse(ctrl.is_graphable())
        self.assertIsNotNone(ctrl.network)
        self.assertIsNone(ctrl._network)

        indices = wp.array([0], dtype=wp.uint32, device=self.device)
        forces = wp.zeros(n, dtype=wp.float32, device=self.device)
        state_a = ctrl.state(n, self.device)
        self.assertTrue(type(state_a.pos_error_history).__module__.startswith("torch"))
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


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TestDriveNeuralLSTMLegacyTorchScript(unittest.TestCase):
    """Regression tests for the supported .pt LSTM checkpoint path."""

    def setUp(self):
        self.device = wp.get_device()
        self._tmp_dir = tempfile.mkdtemp()
        _ignore_torchscript_deprecation(self)

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _build_legacy_lstm_checkpoint(self, path: str, hidden_size: int = 4, metadata: dict | None = None):
        import torch

        class _LegacyLSTM(torch.nn.Module):
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

    def test_synthesizes_metadata_from_torch_module(self):
        path = os.path.join(self._tmp_dir, "legacy_lstm.pt")
        hidden = 6
        self._build_legacy_lstm_checkpoint(path, hidden_size=hidden, metadata={"effort_scale": 2.5})

        ctrl = DriveNeuralLSTM(model_path=path)

        self.assertEqual(ctrl._num_layers, 1)
        self.assertEqual(ctrl._hidden_size, hidden)
        self.assertAlmostEqual(ctrl.effort_scale, 2.5)

    def test_finalize_and_compute(self):
        path = os.path.join(self._tmp_dir, "legacy_lstm.pt")
        self._build_legacy_lstm_checkpoint(path, hidden_size=4)

        ctrl = DriveNeuralLSTM(model_path=path)

        n = 1
        ctrl.finalize(self.device, n)
        self.assertFalse(ctrl.is_graphable())

        state_a = ctrl.state(n, self.device)
        state_b = ctrl.state(n, self.device)
        self.assertTrue(type(state_a.hidden).__module__.startswith("torch"))
        np.testing.assert_array_equal(state_a.hidden.detach().cpu().numpy(), 0.0)

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

        self.assertNotAlmostEqual(float(forces.numpy()[0]), 0.0, places=6)
        self.assertTrue(np.any(state_b.hidden.detach().cpu().numpy() != 0.0))

    def test_implicit_rejected_for_torch_backend(self):
        """Implicit actuation is refused for .pt checkpoints rather than degrading.

        The Torch backend runs outside Warp's tape, so there is no input adjoint
        to linearize the network with; the solve would silently fall back to the
        explicit impulse while still paying for the Newton loop.
        """
        path = os.path.join(self._tmp_dir, "legacy_lstm.pt")
        self._build_legacy_lstm_checkpoint(path, hidden_size=4)

        model = _build_pendulum(self.device)
        drive = DriveNeuralLSTM(model_path=path)
        actuator = Actuator(
            indices=wp.array([0], dtype=wp.uint32, device=self.device),
            drive=drive,
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        self.assertIsNone(drive.bind_params())
        with self.assertRaises(NotImplementedError):
            actuator.set_effort_mode_implicit(response=ResponseOracle(model))


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

    def test_dc_motor_retune_takes_effect(self):
        """Retuning the envelope must change the clamp, in both effort modes.

        ``corner_velocity`` used to be computed once at construction. After a
        retune through the live parameter views it described a different motor,
        which could invert the clamp bounds and let through more effort than
        ``max_motor_effort``. It is now derived from the live parameters
        wherever it is needed.
        """
        device = wp.get_device()
        h = 0.01
        qd0 = 10.0
        sat, vel_lim = 10.0, 5.0

        def run(max_e: float, implicit: bool) -> tuple[float, float]:
            model = _build_pendulum(device)
            state = model.state()
            state.joint_q.assign(np.array([0.0], dtype=np.float32))
            state.joint_qd.assign(np.array([qd0], dtype=np.float32))
            control = model.control()
            control.joint_target_q.assign(np.array([1.0], dtype=np.float32))
            clamp = ClampingDCMotor(
                saturation_effort=wp.array([sat], dtype=float, device=device),
                velocity_limit=wp.array([vel_lim], dtype=float, device=device),
                max_motor_effort=wp.array([20.0], dtype=float, device=device),
            )
            oracle = ResponseOracle(model)
            actuator = Actuator(
                indices=wp.array([0], dtype=wp.uint32, device=device),
                drive=DrivePD(
                    kp=wp.array([5.0e4], dtype=float, device=device),
                    kd=wp.zeros(1, dtype=float, device=device),
                ),
                clamping=[clamp],
                control_target_pos_attr="joint_target_q",
                control_target_vel_attr="joint_target_qd",
            )
            if implicit:
                actuator.set_effort_mode_implicit(response=oracle)
            # Retune through the (possibly view-backed) parameter array.
            clamp.max_motor_effort.assign(np.array([max_e], dtype=np.float32))
            control.joint_f.zero_()
            _refresh_and_step(actuator, oracle, state, control, h)
            return float(control.joint_f.numpy()[0]), float(oracle.inverse_blocks.numpy()[0, 0, 0])

        # Explicit mode clamps at the measured velocity, so the envelope is exact:
        #   corner = vel_lim * (1 + max_e/sat); vel = clip(qd0, +/-corner)
        #   effort = clip(kp*e, sat*(-1 - vel/vel_lim), sat*(1 - vel/vel_lim)) then +/-max_e
        for max_e, expected in ((20.0, -10.0), (5.0, -5.0)):
            self.assertAlmostEqual(run(max_e, implicit=False)[0], expected, places=3)

        # Implicit mode clamps at the *predicted* velocity, so the effort must be
        # the envelope evaluated at the velocity its own impulse produces.
        for max_e in (20.0, 5.0):
            effort, alpha = run(max_e, implicit=True)
            qd_pred = qd0 + alpha * h * effort
            self.assertAlmostEqual(effort, _dc_envelope(sat, vel_lim, max_e, qd_pred), places=3)

        self.assertLess(abs(run(5.0, implicit=True)[0]), abs(run(20.0, implicit=True)[0]))


def _dc_bounds(sat: float, vel_lim: float, max_e: float, qd: float) -> tuple[float, float]:
    """DC-motor effort interval at velocity *qd*, mirroring ClampingDCMotor."""
    corner = vel_lim * (1.0 + max_e / sat) if sat > 0.0 else vel_lim
    vel = float(np.clip(qd, -corner, corner))
    return (
        max(sat * (-1.0 - vel / vel_lim), -max_e),
        min(sat * (1.0 - vel / vel_lim), max_e),
    )


def _dc_envelope(sat: float, vel_lim: float, max_e: float, qd: float) -> float:
    """Upper DC-motor effort bound at *qd*; a hard-driven drive lands here."""
    return _dc_bounds(sat, vel_lim, max_e, qd)[1]


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

_MAX_EFFORT = 10.0
_DC_SATURATION, _DC_VELOCITY_LIMIT, _DC_MAX_EFFORT = 200.0, 2.0, 1.0e6
_POSITION_LOOKUP_POSITIONS = (0.0, 1.0)
_POSITION_LOOKUP_EFFORTS = (200.0, 0.0)


def _pipeline_clamping(
    clamp: str | None,
    device: wp.Device,
    n: int,
) -> tuple[list[ClampingBase] | None, Callable[[float, float], tuple[float, float]]]:
    """Return the clamping list for *clamp* and its NumPy ``(q, qd) -> (lo, hi)`` law."""
    if clamp is None:
        return None, lambda q, qd: (-np.inf, np.inf)

    def _full(value: float) -> wp.array[float]:
        return wp.array(np.full(n, value, dtype=np.float32), dtype=float, device=device)

    if clamp == "max_effort":
        clamping = [ClampingMaxEffort(max_effort=_full(_MAX_EFFORT))]
        return clamping, lambda q, qd: (-_MAX_EFFORT, _MAX_EFFORT)

    if clamp == "dc_motor":
        clamping = [
            ClampingDCMotor(
                saturation_effort=_full(_DC_SATURATION),
                velocity_limit=_full(_DC_VELOCITY_LIMIT),
                max_motor_effort=_full(_DC_MAX_EFFORT),
            )
        ]

        def limit(q: float, qd: float) -> tuple[float, float]:
            envelope = _DC_SATURATION * (1.0 - qd / _DC_VELOCITY_LIMIT)
            reverse = _DC_SATURATION * (-1.0 - qd / _DC_VELOCITY_LIMIT)
            return max(reverse, -_DC_MAX_EFFORT), min(envelope, _DC_MAX_EFFORT)

        return clamping, limit

    if clamp == "position":
        clamping = [
            ClampingPositionBased(
                lookup_positions=_POSITION_LOOKUP_POSITIONS,
                lookup_efforts=_POSITION_LOOKUP_EFFORTS,
            )
        ]

        def limit(q: float, qd: float) -> tuple[float, float]:
            effort = float(np.interp(q, _POSITION_LOOKUP_POSITIONS, _POSITION_LOOKUP_EFFORTS))
            return -effort, effort

        return clamping, limit

    if clamp in ("max_effort_then_open", "open_then_max_effort"):
        # A DC-motor clamp with bounds far outside anything the solve produces is
        # a no-op, so stacking it either side of the max-effort clamp must leave
        # the result unchanged.
        open_clamp = ClampingDCMotor(
            saturation_effort=_full(1.0e9),
            velocity_limit=_full(1.0e9),
            max_motor_effort=_full(1.0e9),
        )
        bound = ClampingMaxEffort(max_effort=_full(_MAX_EFFORT))
        order = [bound, open_clamp] if clamp == "max_effort_then_open" else [open_clamp, bound]
        return order, lambda q, qd: (-_MAX_EFFORT, _MAX_EFFORT)

    raise ValueError(f"unknown clamp kind: {clamp}")


def _solve_clamped_effort(
    residual: Callable[[float], float],
    bound: float = 1.0e12,
    iterations: int = 200,
) -> float:
    """Bisect ``residual(tau) = clamped_force(tau) - tau``, which decreases in tau.

    Fixed-point iteration diverges at stiff gains (the map's slope is
    ``-alpha*dt^2*kp``), so the reference bisects instead. Monotonicity holds
    because both the control law and the clamp bounds fall as tau rises. Only
    valid for a single DOF; the unclamped law is linear and solved directly.
    """
    lo, hi = -bound, bound
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class TestActuatorStep(unittest.TestCase):
    """Actuator.step() integration: solver step, response refresh, actuator step."""

    def run_test_actuator_pipeline(self, **config):
        """Run one pipeline configuration.

        Args:
            **config: Keywords accepted by :meth:`_run_actuator_pipeline`, plus
                an optional ``device``, defaulting to the current Warp device.
        """
        device = config.pop("device", None) or wp.get_device()
        self._run_actuator_pipeline(device, **config)

    def _run_actuator_pipeline(
        self,
        device: wp.Device,
        *,
        drive: str = "pd",
        clamp: str | None = None,
        implicit: bool = True,
        dofs: int = 1,
        kp: float | Sequence[float] = 500.0,
        kd: float | Sequence[float] = 5.0,
        ki: float | Sequence[float] = 50.0,
        integral_max: float | Sequence[float] = 1.0e9,
        q0: float | Sequence[float] = 0.2,
        qd0: float | Sequence[float] = 0.0,
        target: float | Sequence[float] = 1.0,
        dt: float = 0.01,
        steps: int = 1,
        retune: bool = False,
        expect: str | None = None,
        check_forces: bool = True,
        expect_integral_saturated: bool = False,
        worlds: int = 1,
        actuated: Sequence[int] | None = None,
    ) -> None:
        """Run the actuator pipeline for one drive / clamp / effort-mode combination.

        Each step follows the order a simulation uses: zero the forces, refresh
        the response oracle at the step-start pose, step the actuator, then step
        the solver. Every effort is compared against a NumPy reference built from
        the dense response ``A = inv(H)``. Without a clamp the law is linear, so
        the reference solves ``(I + dt*diag(dt*kp + kd)*A) tau = f0`` directly and
        covers the coupled chain; with a clamp it bisects the same scalar residual
        the kernel solves. Implicit mode evaluates the law and the clamp at the
        predicted end-of-step state, explicit mode at the current state.

        On a CUDA device with memory pools the refresh + step pair runs from a
        captured graph rather than eagerly, so every configuration doubles as a
        check that the pipeline is capturable. Correctness is still judged only
        by the reference comparison, which now covers the replayed efforts.

        Args:
            device: Device to build the model and actuator on.
            drive: ``"pd"`` or ``"pid"``.
            clamp: ``None``, ``"max_effort"``, ``"dc_motor"`` or ``"position"``.
                Only the single-DOF model supports a clamp.
            implicit: Solve the control law against the predicted end-of-step state.
            dofs: 1 for the single revolute pendulum, 2 for the coupled two-link chain.
            kp: Proportional gain [N·m/rad], scalar or per-DOF.
            kd: Derivative gain [N·m·s/rad], scalar or per-DOF.
            ki: Integral gain [N·m/(rad·s)], PID only, scalar or per-DOF.
            integral_max: Anti-windup bound on the PID integral [rad·s].
            q0: Initial joint position [rad], scalar or per-DOF.
            qd0: Initial joint velocity [rad/s], scalar or per-DOF.
            target: Position target [rad], scalar or per-DOF.
            dt: Step size [s].
            steps: Number of pipeline steps to run.
            retune: Rewrite the gains (and a max-effort limit) mid-run and
                verify the next solve picks them up.
            expect: ``"converge"`` to require the final pose near ``target``,
                ``"diverge"`` to require it to blow up (explicit stiff gains).
            check_forces: Compare each step's effort against the reference.
            expect_integral_saturated: Require the PID integral to end at
                ``integral_max``, proving anti-windup actually bound.
            worlds: Replicate the model into this many identical worlds. The
                reference is computed once and compared against every world.
            actuated: Per-world DOF indices the actuator drives; ``None`` drives
                all of them. Undriven DOFs must end up with exactly zero effort.
        """
        model = _build_pendulum(device, worlds=worlds) if dofs == 1 else _build_two_link(device, worlds=worlds)
        n = dofs
        total = n * worlds
        self.assertEqual(model.joint_dof_count, total)

        # Which DOFs the actuator drives, per world and across the whole model.
        act = np.arange(n) if actuated is None else np.asarray(sorted(actuated), dtype=int)
        driven = len(act)
        act_all = np.concatenate([act + w * n for w in range(worlds)]).astype(np.uint32)
        if clamp is not None and driven != 1:
            raise ValueError("the clamped reference bisects a scalar residual, so it needs a single driven DOF")

        clamping, limit = _pipeline_clamping(clamp, device, driven * worlds)

        def _vec(value: float | Sequence[float]) -> np.ndarray:
            return np.full(n, value, dtype=np.float32) if np.isscalar(value) else np.asarray(value, dtype=np.float32)

        def _tiled(values: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray(values), worlds)

        kp, kd, ki = _vec(kp), _vec(kd), _vec(ki)
        integral_max, q0, qd0, target = _vec(integral_max), _vec(q0), _vec(qd0), _vec(target)

        def _arr(values: np.ndarray) -> wp.array[float]:
            """Per-actuator array: select the driven DOFs, then repeat per world."""
            return wp.array(_tiled(np.asarray(values)[act]), dtype=float, device=device)

        if drive == "pd":
            control_law = DrivePD(kp=_arr(kp), kd=_arr(kd))
        elif drive == "pid":
            control_law = DrivePID(kp=_arr(kp), ki=_arr(ki), kd=_arr(kd), integral_max=_arr(integral_max))
        else:
            raise ValueError(f"unknown drive kind: {drive}")

        oracle = ResponseOracle(model)
        actuator = Actuator(
            indices=wp.array(act_all, dtype=wp.uint32, device=device),
            drive=control_law,
            clamping=clamping,
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        if implicit:
            actuator.set_effort_mode_implicit(response=oracle)
        self.assertTrue(actuator.is_graphable())

        def reference(
            q: np.ndarray,
            qd: np.ndarray,
            integral: np.ndarray,
            kp_now: np.ndarray,
            kd_now: np.ndarray,
        ) -> np.ndarray:
            """The effort the actuator should produce from state (q, qd).

            Returns one value per model DOF; undriven DOFs are zero, since the
            actuator never writes them.
            """
            feedforward = np.asarray(ki * integral if drive == "pid" else np.zeros(n), dtype=np.float64)
            out = np.zeros(n, dtype=np.float64)

            if not implicit:
                law = (kp_now * (target - q) - kd_now * qd + feedforward)[act]
                out[act] = law if clamp is None else np.clip(law, *limit(float(q[act[0]]), float(qd[act[0]])))
                return out

            # Only the driven DOFs are solved, coupled through their submatrix of
            # inv(H). Inverting first leaves the undriven DOFs free to move.
            response = _response_at(model, _tiled(q), _tiled(qd))[np.ix_(act, act)]
            if clamp is None:
                gain = dt * kp_now[act] + kd_now[act]
                jacobian = np.eye(driven) + dt * np.diag(gain) @ response
                rhs = kp_now[act] * (target[act] - q[act] - dt * qd[act]) - kd_now[act] * qd[act] + feedforward[act]
                out[act] = np.linalg.solve(jacobian, rhs)
                return out

            alpha = float(response[0, 0])
            j = act[0]

            def residual(tau: float) -> float:
                qd_pred = qd[j] + alpha * dt * tau
                q_pred = q[j] + dt * qd_pred
                lo, hi = limit(q_pred, qd_pred)
                law = kp_now[j] * (target[j] - q_pred) - kd_now[j] * qd_pred + feedforward[j]
                return float(np.clip(law, lo, hi)) - tau

            out[j] = _solve_clamped_effort(residual)
            return out

        state_in, state_out = model.state(), model.state()
        state_in.joint_q.assign(_tiled(q0))
        state_in.joint_qd.assign(_tiled(qd0))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        control = model.control()
        control.joint_target_q.assign(_tiled(target))
        solver = newton.solvers.SolverFeatherstone(model)
        act_a, act_b = actuator.state(), actuator.state()

        integral = np.zeros(n, dtype=np.float64)
        kp_now, kd_now = kp, kd
        stateful = actuator.is_stateful()
        use_graph = device.is_cuda and wp.is_mempool_enabled(device)
        graphs = {}

        if use_graph:
            # Module loading and lazy allocation have to happen before a capture.
            control.joint_f.zero_()
            oracle.refresh(state_in)
            actuator.step(state_in, control, act_a, act_b, dt=dt)
            if stateful:
                act_a.drive_state.integral.zero_()
                act_b.drive_state.integral.zero_()

        def actuate():
            """Zero the efforts, refresh the response, and step the actuator.

            The state and actuator buffers alternate with period two, so keying
            the graphs on them builds at most two and replays them thereafter.
            Parameter writes still land because a graph reads the live arrays.
            """
            if not use_graph:
                control.joint_f.zero_()
                oracle.refresh(state_in)
                actuator.step(state_in, control, act_a, act_b, dt=dt)
                return
            key = (id(state_in), id(act_a))
            if key not in graphs:
                with wp.ScopedCapture(device) as capture:
                    control.joint_f.zero_()
                    oracle.refresh(state_in)
                    actuator.step(state_in, control, act_a, act_b, dt=dt)
                graphs[key] = capture.graph
            wp.capture_launch(graphs[key])

        def check_effort(
            step_label: str,
            kp_now: np.ndarray,
            kd_now: np.ndarray,
            integral: np.ndarray,
        ) -> None:
            """Step once from the live state and compare the effort to the reference."""
            q = state_in.joint_q.numpy().astype(np.float64)[:n]
            qd = state_in.joint_qd.numpy().astype(np.float64)[:n]
            if drive == "pid":
                integral[:] = np.clip(integral + (target - q) * dt, -integral_max, integral_max)

            actuate()
            effort = control.joint_f.numpy().astype(np.float64)

            if check_forces:
                expected = reference(q, qd, integral, kp_now, kd_now)
                self.assertTrue(np.all(np.isfinite(effort)), msg=f"{step_label}: effort must stay finite")
                np.testing.assert_allclose(
                    effort[:n],
                    expected,
                    rtol=2.0e-3,
                    atol=1.0e-4,
                    err_msg=f"{step_label}: q={q} qd={qd} integral={integral}",
                )
                _assert_worlds_match(self, model, control.joint_f)
                if drive == "pid":
                    # act_b holds the integral this step just advanced to.
                    np.testing.assert_allclose(
                        act_b.drive_state.integral.numpy(),
                        _tiled(integral[act]),
                        rtol=1.0e-4,
                        atol=1.0e-9,
                        err_msg=f"{step_label}: advanced integral",
                    )
            return effort

        for step_i in range(steps):
            check_effort(f"step {step_i}", kp_now, kd_now, integral)
            act_a, act_b = act_b, act_a
            solver.step(state_in, state_out, control, None, dt=dt)
            state_in, state_out = state_out, state_in

        if retune:
            kp_now, kd_now = (4.0 * kp).astype(np.float32), (4.0 * kd).astype(np.float32)
            actuator.drive.kp.assign(_tiled(kp_now[act]))
            actuator.drive.kd.assign(_tiled(kd_now[act]))
            np.testing.assert_allclose(actuator.drive.kp.numpy(), _tiled(kp_now[act]), rtol=1e-6)
            retuned = check_effort("retune", kp_now, kd_now, integral)

            if clamp == "max_effort":
                # Tightening the limit below the solved effort must bind on the next solve.
                tight = 0.25 * np.abs(retuned[act_all.astype(np.int64)])
                actuator.clamping[0].max_effort.assign(tight.astype(np.float32))
                actuate()
                np.testing.assert_allclose(
                    np.abs(control.joint_f.numpy()[act_all.astype(np.int64)]), tight, rtol=1.0e-4, atol=1.0e-5
                )

        if expect_integral_saturated:
            # Per-step checks already tie the device integral to this one.
            np.testing.assert_allclose(integral, integral_max, rtol=1.0e-4)

        q_final = state_in.joint_q.numpy()
        if expect == "converge":
            self.assertTrue(np.all(np.isfinite(q_final)))
            np.testing.assert_allclose(q_final, _tiled(target), atol=0.05)
        elif expect == "diverge":
            self.assertFalse(
                np.all(np.isfinite(q_final)) and np.all(np.abs(q_final) < 10.0 * np.abs(_tiled(target))),
                msg=f"explicit mode should not stay bounded at these gains, got q={q_final}",
            )
        elif expect is not None:
            raise ValueError(f"unknown expectation: {expect}")

    def test_pipeline_pd_implicit(self):
        """Verify the implicit PD solve against the Stable-PD reference."""
        self.run_test_actuator_pipeline(drive="pd")

    def test_pipeline_pd_explicit(self):
        """Verify plain PD in explicit mode through the same pipeline."""
        self.run_test_actuator_pipeline(drive="pd", implicit=False)

    def test_pipeline_pd_max_effort_implicit(self):
        """Verify a max-effort clamp bounds the implicit effort exactly."""
        self.run_test_actuator_pipeline(drive="pd", clamp="max_effort")

    def test_pipeline_pd_stacked_clamps_implicit(self):
        """Stacking an open clamp with a max-effort clamp gives the same bound in either order."""
        for order in ("max_effort_then_open", "open_then_max_effort"):
            with self.subTest(order=order):
                self.run_test_actuator_pipeline(drive="pd", clamp=order)

    def test_pipeline_pd_dc_motor_implicit(self):
        """Verify the DC-motor envelope binds at the predicted end-of-step velocity."""
        self.run_test_actuator_pipeline(drive="pd", clamp="dc_motor", kp=5.0e4, kd=0.0, q0=0.0, qd0=1.0, worlds=2)

    def test_pipeline_pd_dc_motor_explicit(self):
        """Verify the DC-motor envelope binds at the current velocity in explicit mode."""
        self.run_test_actuator_pipeline(drive="pd", clamp="dc_motor", implicit=False, kp=5.0e4, kd=0.0, q0=0.0, qd0=1.0)

    def test_pipeline_pd_position_implicit(self):
        """Verify the position-based limit is read at the predicted end-of-step position."""
        self.run_test_actuator_pipeline(drive="pd", clamp="position", kp=5.0e4, kd=0.0, q0=0.2, target=2.0)

    def test_pipeline_pd_high_gain_implicit(self):
        """Verify an extreme stiffness stays finite and matches the reference."""
        self.run_test_actuator_pipeline(drive="pd", kp=1.0e8, kd=0.0, q0=0.0, target=0.5)

    def test_pipeline_pid_antiwindup_implicit(self):
        """Verify a saturating integral bound feeds the implicit solve each step.

        The error is 0.8 rad and dt is 0.01 s, so the integral grows by 0.008
        per step and anti-windup binds on step 3 of 5.
        """
        self.run_test_actuator_pipeline(
            drive="pid",
            kp=400.0,
            ki=50.0,
            kd=6.0,
            integral_max=0.02,
            steps=5,
            expect_integral_saturated=True,
            worlds=2,
        )

    def test_pipeline_pd_coupled_implicit(self):
        """Verify the implicit solve couples both DOFs of a two-link chain through inv(H)."""
        self.run_test_actuator_pipeline(
            drive="pd",
            dofs=2,
            kp=[4000.0, 3000.0],
            kd=[40.0, 30.0],
            q0=[0.3, -0.8],
            target=[0.6, 0.4],
            worlds=2,
        )

    def test_splitting_dofs_across_two_actuators(self):
        """One actuator over both DOFs versus one actuator per DOF.

        Explicit efforts match: the control law is evaluated per DOF, so it does
        not matter which actuator evaluates it. Implicit efforts do not: an
        actuator solves all of its own DOFs as one coupled system through the
        submatrix of inv(H), so splitting them leaves each actuator solving its
        DOF alone, without the cross-coupling term.
        """
        device = wp.get_device()
        h = 0.01
        kp = np.array([4000.0, 3000.0], dtype=np.float32)
        kd = np.array([40.0, 30.0], dtype=np.float32)
        q0 = np.array([0.3, -0.8], dtype=np.float32)
        target = np.array([0.6, 0.4], dtype=np.float32)

        def efforts(dof_groups: Sequence[Sequence[int]], implicit: bool) -> np.ndarray:
            model = _build_two_link(device)
            state = model.state()
            state.joint_q.assign(q0)
            control = model.control()
            control.joint_target_q.assign(target)
            actuators, oracles = [], []
            for group in dof_groups:
                dofs = np.asarray(group)
                actuator = Actuator(
                    indices=wp.array(dofs.astype(np.uint32), dtype=wp.uint32, device=device),
                    drive=DrivePD(
                        kp=wp.array(kp[dofs], dtype=float, device=device),
                        kd=wp.array(kd[dofs], dtype=float, device=device),
                    ),
                    control_target_pos_attr="joint_target_q",
                    control_target_vel_attr="joint_target_qd",
                )
                oracle = ResponseOracle(model)
                if implicit:
                    actuator.set_effort_mode_implicit(response=oracle)
                actuators.append(actuator)
                oracles.append(oracle)
            control.joint_f.zero_()
            for oracle in oracles:
                oracle.refresh(state)
            for actuator in actuators:
                actuator.step(state, control, dt=h)
            return control.joint_f.numpy().copy()

        together, apart = efforts([[0, 1]], False), efforts([[0], [1]], False)
        np.testing.assert_allclose(together, apart, rtol=1e-5, atol=1e-6)

        together, apart = efforts([[0, 1]], True), efforts([[0], [1]], True)
        self.assertFalse(
            np.allclose(together, apart, rtol=1e-3), "the coupled solve must differ from two scalar solves"
        )

    def test_pipeline_pd_partially_actuated_implicit(self):
        """Drive only the tip joint of the two-link chain."""
        self.run_test_actuator_pipeline(
            drive="pd",
            dofs=2,
            actuated=[1],
            kp=[4000.0, 3000.0],
            kd=[40.0, 30.0],
            q0=[0.3, -0.8],
            target=[0.6, 0.4],
        )

    def test_pipeline_pid_coupled_implicit(self):
        """Verify a stateful drive on the coupled chain over several steps."""
        self.run_test_actuator_pipeline(
            drive="pid",
            dofs=2,
            kp=[400.0, 300.0],
            ki=[50.0, 30.0],
            kd=[8.0, 6.0],
            q0=[0.3, -0.8],
            qd0=[0.1, -0.2],
            target=[0.6, 0.4],
            steps=3,
        )

    def test_pipeline_pd_implicit_retune(self):
        """Verify gain and clamp writes reach the installed implicit solve."""
        self.run_test_actuator_pipeline(drive="pd", clamp="max_effort", retune=True, worlds=2)

    def test_pipeline_pd_stiff_implicit_converges(self):
        """Verify stiff gains converge to the target over a long implicit run."""
        self.run_test_actuator_pipeline(
            drive="pd", kp=5.0e4, kd=300.0, q0=0.0, steps=300, expect="converge", check_forces=False
        )

    def test_pipeline_pd_stiff_explicit_diverges(self):
        """Verify explicit mode blows up where implicit mode stays bounded."""
        self.run_test_actuator_pipeline(
            drive="pd",
            kp=5.0e4,
            kd=300.0,
            q0=0.0,
            steps=300,
            implicit=False,
            expect="diverge",
            check_forces=False,
        )

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
            DrivePD,
            index=dof_a,
            kp=kp,
            kd=kd,
            delay_steps=delay_a,
            clamping=[(ClampingDCMotor, dc_args)],
        )
        template.add_actuator(
            DrivePD,
            index=dof_b,
            kp=kp,
            kd=kd,
            delay_steps=delay_b,
            clamping=[(ClampingDCMotor, dc_args)],
        )

        builder = newton.ModelBuilder()
        builder.replicate(template, num_envs)
        model = builder.finalize()

        self.assertEqual(len(model.actuators), 1, "all DOFs share drive+clamping type")
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
            _write_dof_values(model, control.joint_target_q, dofs, [tgt] * n)
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

    def test_effort_mode_switch_roundtrip(self):
        """Effort modes are interchangeable at runtime: implicit -> explicit -> implicit."""
        device = wp.get_device()
        h = 0.01
        kp_val, kd_val = 500.0, 5.0
        q0, target = 0.2, 1.0

        model = _build_pendulum(device)
        state = model.state()
        state.joint_q.assign(np.array([q0], dtype=np.float32))
        control = model.control()
        control.joint_target_q.assign(np.array([target], dtype=np.float32))

        actuator, oracle = _make_implicit_actuator(
            model,
            device,
            kp=wp.array([kp_val], dtype=float, device=device),
            kd=wp.array([kd_val], dtype=float, device=device),
        )
        control.joint_f.zero_()
        _refresh_and_step(actuator, oracle, state, control, h)
        implicit_tau = float(control.joint_f.numpy()[0])

        actuator.set_effort_mode_explicit()
        control.joint_f.zero_()
        _refresh_and_step(actuator, oracle, state, control, h)
        explicit_tau = float(control.joint_f.numpy()[0])
        self.assertAlmostEqual(explicit_tau, kp_val * (target - q0), delta=1e-3)
        self.assertLess(implicit_tau, explicit_tau)

        actuator.set_effort_mode_implicit(response=oracle)
        control.joint_f.zero_()
        _refresh_and_step(actuator, oracle, state, control, h)
        self.assertAlmostEqual(float(control.joint_f.numpy()[0]), implicit_tau, delta=abs(implicit_tau) * 1e-5)


# ---------------------------------------------------------------------------
# 5. Implicit effort mode — validation and coupled-solve internals
# ---------------------------------------------------------------------------


class TestActuatorImplicit(unittest.TestCase):
    """Implicit effort mode: construction validation and coupled-solve internals.

    Behaviour that also has an explicit-mode analogue is verified through the
    configurable runner in :class:`TestActuatorStep`; what lives here is specific
    to the implicit solve.
    """

    def test_unsupported_drive_raises(self):
        """A drive without evaluate_force is rejected at construction."""
        device = wp.get_device()

        class _NoForceDriveBase(DriveBase):
            def is_stateful(self):
                return False

            def is_graphable(self):
                return True

        model = _build_pendulum(device)
        indices = wp.array(np.arange(model.joint_dof_count, dtype=np.uint32), device=device)
        actuator = Actuator(
            indices=indices,
            drive=_NoForceDriveBase(),
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        with self.assertRaises(NotImplementedError):
            actuator.set_effort_mode_implicit(response=ResponseOracle(model))

    def test_validation_errors(self):
        """A non-ResponseOracle inverse mass and a missing dt raise clearly."""
        device = wp.get_device()
        model = _build_pendulum(device)
        indices = wp.array(np.arange(model.joint_dof_count, dtype=np.uint32), device=device)
        kp = wp.array([100.0], dtype=float, device=device)
        kd = wp.array([1.0], dtype=float, device=device)

        actuator = Actuator(
            indices=indices,
            drive=DrivePD(kp=kp, kd=kd),
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        with self.assertRaisesRegex(ValueError, "ResponseOracle"):
            actuator.set_effort_mode_implicit(response=None)

        actuator, _ = _make_implicit_actuator(model, device, kp=kp, kd=kd)
        with self.assertRaisesRegex(ValueError, "requires dt"):
            actuator.step(model.state(), model.control())

    def test_validation_rejects_bad_options(self):
        """dt, warm_start and fd_epsilon are validated rather than silently misbehaving."""
        device = wp.get_device()
        model = _build_pendulum(device)
        kp = wp.array([100.0], dtype=float, device=device)
        kd = wp.array([1.0], dtype=float, device=device)

        with self.assertRaisesRegex(ValueError, "warm_start"):
            _make_implicit_actuator(
                model, device, kp=kp, kd=kd, options=newton.actuators.Actuator.ImplicitOptions(warm_start="Zero")
            )

        with self.assertRaisesRegex(ValueError, "fd_epsilon"):
            _make_implicit_actuator(
                model, device, kp=kp, kd=kd, options=newton.actuators.Actuator.ImplicitOptions(fd_epsilon=0.0)
            )

        actuator, _ = _make_implicit_actuator(model, device, kp=kp, kd=kd)
        with self.assertRaisesRegex(ValueError, "dt > 0"):
            actuator.step(model.state(), model.control(), dt=0.0)

    def test_prediction_matches_featherstone_step(self):
        """The state the solve predicts is the state the solver actually reaches.

        The whole scheme rests on ``qd(p) = qd + A p`` and ``q(p) = q + h qd(p)``.
        With no gravity and zero initial velocity that is exactly what
        semi-implicit Euler produces, so applying the solved effort through
        Featherstone must land on the predicted velocity and position.
        """
        device = wp.get_device()
        h = 0.01
        kp = np.array([300.0, 200.0], dtype=np.float32)
        kd = np.array([3.0, 2.0], dtype=np.float32)
        q0 = np.array([0.3, -0.8], dtype=np.float32)
        target = np.array([0.6, 0.4], dtype=np.float32)

        model = _build_two_link(device)
        n = model.joint_dof_count
        state_in, state_out = model.state(), model.state()
        state_in.joint_q.assign(q0)
        control = model.control()
        control.joint_target_q.assign(target)

        actuator, oracle = _make_implicit_actuator(
            model,
            device,
            kp=wp.array(kp, dtype=float, device=device),
            kd=wp.array(kd, dtype=float, device=device),
        )
        control.joint_f.zero_()
        _refresh_and_step(actuator, oracle, state_in, control, h)
        tau = control.joint_f.numpy().copy()

        # What the solve assumed the step would do.
        A = oracle.inverse_blocks.numpy()[0, :n, :n]
        qd_pred = A @ (h * tau)
        q_pred = q0 + h * qd_pred

        # What the solver actually does with that effort.
        solver = newton.solvers.SolverFeatherstone(model)
        solver.step(state_in, state_out, control, None, dt=h)

        np.testing.assert_allclose(state_out.joint_qd.numpy(), qd_pred, rtol=2e-3, atol=1e-5)
        np.testing.assert_allclose(state_out.joint_q.numpy(), q_pred, rtol=2e-3, atol=1e-6)

    def test_coupled_solve_clamp_in_residual(self):
        """The clamp is composed into the coupled residual, not applied afterwards.

        A tight max-effort clamp on DOF 0 of a two-link block must bind exactly
        at the limit, and — because the clamp lives inside the block Newton —
        DOF 1 must re-solve against the *clamped* DOF-0 impulse through the
        off-diagonal coupling. A post-hoc clamp would leave DOF 1 at its
        unclamped value, so the test asserts DOF 1 both moves away from that
        value and matches the analytic solution of the pinned system.
        """
        device = wp.get_device()
        h = 0.01
        kp = np.array([4000.0, 3000.0], dtype=np.float32)
        kd = np.array([40.0, 30.0], dtype=np.float32)
        q0 = np.array([0.3, -0.8], dtype=np.float32)
        target = np.array([0.6, 0.4], dtype=np.float32)

        model = _build_two_link(device)
        state = model.state()
        state.joint_q.assign(q0)
        control = model.control()
        control.joint_target_q.assign(target)

        A = _response_at(model, q0, np.zeros(2, dtype=np.float32))

        # Unclamped block solve, to size a binding limit on DOF 0.
        actuator, oracle = _make_implicit_actuator(
            model,
            device,
            kp=wp.array(kp, dtype=float, device=device),
            kd=wp.array(kd, dtype=float, device=device),
        )
        control.joint_f.zero_()
        _refresh_and_step(actuator, oracle, state, control, h)
        unclamped = control.joint_f.numpy().copy()
        limit = 0.5 * abs(unclamped[0])

        clamped, _ = _make_implicit_actuator(
            model,
            device,
            kp=wp.array(kp, dtype=float, device=device),
            kd=wp.array(kd, dtype=float, device=device),
            max_effort=np.array([limit, 1.0e6], dtype=np.float32),
            response=oracle,
        )
        control.joint_f.zero_()
        _refresh_and_step(clamped, oracle, state, control, h)
        joint_f = control.joint_f.numpy()

        # DOF 0 binds exactly at the limit (same sign as the unclamped force).
        self.assertAlmostEqual(abs(joint_f[0]), limit, delta=limit * 1e-3)

        # Analytic DOF-1 solve with DOF 0 pinned at its clamped impulse p0 = h*tau0.
        # qd1(p) = A10*p0 + A11*p1, q1(p) = q0[1] + h*qd1; PD with qd0 = target_vel = 0:
        #   tau1 = [kp1*(t1 - q0[1]) - (h*kp1 + kd1)*A10*p0] / (1 + h*(h*kp1 + kd1)*A11)
        tau0_clamped = np.sign(unclamped[0]) * limit
        p0 = h * tau0_clamped
        g1 = h * kp[1] + kd[1]
        tau1_ref = (kp[1] * (target[1] - q0[1]) - g1 * A[1, 0] * p0) / (1.0 + h * g1 * A[1, 1])
        self.assertAlmostEqual(joint_f[1], tau1_ref, delta=abs(tau1_ref) * 1e-3)

        # And DOF 1 genuinely responded to DOF 0's saturation (a post-hoc clamp would not).
        self.assertGreater(abs(joint_f[1] - unclamped[1]), abs(unclamped[1]) * 1e-3)

    def test_joint_type_support_follows_the_coordinate_layout(self):
        """Joints with one coordinate per DOF are accepted; quaternion ones are not.

        The solve predicts ``q + dt*qd``, which needs a scalar coordinate per
        DOF. Revolute and D6 joints satisfy that, so both must be accepted. A
        ball joint has three DOFs but four coordinates (a quaternion), so it
        must be rejected with an error naming the joint type.
        """
        device = wp.get_device()

        def build(kind: str) -> newton.Model:
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            link = builder.add_link(mass=1.0)
            builder.add_shape_box(link, hx=0.2, hy=0.1, hz=0.1)
            if kind == "revolute":
                joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
            elif kind == "d6":
                joint = builder.add_joint_d6(
                    parent=-1,
                    child=link,
                    linear_axes=[],
                    angular_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z)],
                )
            else:
                joint = builder.add_joint_ball(parent=-1, child=link)
            builder.add_articulation([joint])
            return builder.finalize(device=device)

        def install(model: newton.Model) -> None:
            n = model.joint_dof_count
            actuator = Actuator(
                indices=wp.array(np.arange(n, dtype=np.uint32), device=device),
                drive=DrivePD(
                    kp=wp.array(np.full(n, 100.0, dtype=np.float32), dtype=float, device=device),
                    kd=wp.zeros(n, dtype=float, device=device),
                ),
                control_target_pos_attr="joint_target_q",
                control_target_vel_attr="joint_target_qd",
            )
            actuator.set_effort_mode_implicit(response=ResponseOracle(model))

        for kind in ("revolute", "d6"):
            install(build(kind))  # must not raise

        with self.assertRaisesRegex(ValueError, "BALL"):
            install(build("ball"))

    def test_newton_loop_needed_when_the_clamp_switches_branch(self):
        """The Newton loop must iterate when the solve crosses a clamp branch.

        Every built-in control law is affine in the impulse, and each branch of
        the DC-motor envelope is affine too, so a single Newton step is normally
        exact and the iteration count never matters. It starts to matter when
        the solution lies on a different branch of the envelope than the initial
        guess. Starting from zero impulse on a coupled two-link chain does that:
        one step lands far away, two steps converge, and the result agrees with
        a damped Picard iteration -- a genuinely different algorithm -- so both
        ``max_iters`` and the convergence tolerances stay covered.
        """
        device = wp.get_device()
        h = 0.01
        # Tuned so the solve crosses a clamp branch; the assertions below fail if it stops doing so.
        kp, sat, vel_lim, max_e = 5.0, 20.0, 1.0, 50.0
        q0 = np.zeros(2)
        qd0 = np.array([1.5, -1.0])
        target = np.array([1.0, 1.0])

        def run(max_iters: int) -> tuple[np.ndarray, np.ndarray]:
            model = _build_two_link(device)
            state = model.state()
            state.joint_q.assign(q0.astype(np.float32))
            state.joint_qd.assign(qd0.astype(np.float32))
            control = model.control()
            control.joint_target_q.assign(target.astype(np.float32))
            oracle = ResponseOracle(model)
            actuator = Actuator(
                indices=wp.array([0, 1], dtype=wp.uint32, device=device),
                drive=DrivePD(
                    kp=wp.array([kp, kp], dtype=float, device=device),
                    kd=wp.zeros(2, dtype=float, device=device),
                ),
                clamping=[
                    ClampingDCMotor(
                        saturation_effort=wp.array([sat, sat], dtype=float, device=device),
                        velocity_limit=wp.array([vel_lim, vel_lim], dtype=float, device=device),
                        max_motor_effort=wp.array([max_e, max_e], dtype=float, device=device),
                    )
                ],
                control_target_pos_attr="joint_target_q",
                control_target_vel_attr="joint_target_qd",
            )
            actuator.set_effort_mode_implicit(
                response=oracle,
                options=newton.actuators.Actuator.ImplicitOptions(max_iters=max_iters, warm_start="zero"),
            )
            control.joint_f.zero_()
            _refresh_and_step(actuator, oracle, state, control, h)
            return control.joint_f.numpy().copy(), _response_at_state(model, state)

        converged, response = run(8)

        # Independent reference: damped fixed point on p = h*g(q(p), qd(p)).
        p = np.zeros(2)
        for _ in range(200000):
            qd_p = qd0 + response @ p
            q_p = q0 + h * qd_p
            bounds = np.array([_dc_bounds(sat, vel_lim, max_e, v) for v in qd_p])
            f = np.clip(kp * (target - q_p), bounds[:, 0], bounds[:, 1])
            p += 0.02 * (h * f - p)
        np.testing.assert_allclose(converged, p / h, rtol=2e-3, atol=1e-3)

        # One step lands on the wrong branch; two are enough and further ones change nothing.
        single, _ = run(1)
        self.assertGreater(np.max(np.abs(single - converged)), 0.5 * np.max(np.abs(converged)))
        np.testing.assert_allclose(run(2)[0], converged, rtol=1e-6, atol=1e-6)

    def test_warm_start_zero_reaches_the_same_solution(self):
        """``warm_start="zero"`` converges to the same effort as the explicit start.

        The two warm starts only change the initial guess, so a converged solve
        must not depend on which was used. Nothing else exercises the ``"zero"``
        branch, and starting from zero also makes the Newton loop climb from a
        genuinely bad guess.
        """
        device = wp.get_device()
        h = 0.01
        kp_val, kd_val = 5.0e3, 12.0

        def run(warm_start: str) -> np.ndarray:
            model = _build_two_link(device)
            state = model.state()
            state.joint_q.assign(np.array([0.1, -0.2], dtype=np.float32))
            state.joint_qd.assign(np.array([0.5, 0.25], dtype=np.float32))
            control = model.control()
            control.joint_target_q.assign(np.array([0.8, 0.4], dtype=np.float32))
            actuator, oracle = _make_implicit_actuator(
                model,
                device,
                kp=wp.array([kp_val, kp_val], dtype=float, device=device),
                kd=wp.array([kd_val, kd_val], dtype=float, device=device),
                options=newton.actuators.Actuator.ImplicitOptions(warm_start=warm_start),
            )
            control.joint_f.zero_()
            _refresh_and_step(actuator, oracle, state, control, h)
            return control.joint_f.numpy().copy()

        np.testing.assert_allclose(run("zero"), run("explicit"), rtol=1e-5, atol=1e-6)

    def test_singular_jacobian_stays_finite(self):
        """A degenerate Jacobian must not leak Inf/NaN into the effort.

        ``derivative_floor`` bounds the pivot used by the elimination; the same
        floored value has to reach the back-substitution divide, otherwise a
        vanishing diagonal produces a non-finite impulse. With kp = kd = 0 the
        force law is identically zero, so the residual is flat and the solve
        leans on that floor. The correct effort is exactly zero, so asserting
        that (rather than only finiteness) also pins the sign of the floored
        pivot: a floor that drops the sign still returns a finite wrong answer.
        """
        device = wp.get_device()
        h = 0.01
        model = _build_pendulum(device)
        state = model.state()
        state.joint_q.assign(np.array([0.2], dtype=np.float32))
        control = model.control()
        control.joint_target_q.assign(np.array([1.0], dtype=np.float32))

        actuator, oracle = _make_implicit_actuator(
            model,
            device,
            kp=wp.zeros(model.joint_dof_count, dtype=float, device=device),
            kd=wp.zeros(model.joint_dof_count, dtype=float, device=device),
            options=newton.actuators.Actuator.ImplicitOptions(derivative_floor=1.0e-8),
        )
        control.joint_f.zero_()
        _refresh_and_step(actuator, oracle, state, control, h)
        np.testing.assert_allclose(control.joint_f.numpy(), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 6. Response oracle — the per-articulation inv(H) the implicit solve reads
# ---------------------------------------------------------------------------


class TestResponseOracle(unittest.TestCase):
    """ResponseOracle: the per-articulation inv(H) the implicit solve reads."""

    def test_singular_one_dof_mass_matrix_uses_float32_floor(self):
        """Bound the inverse of a singular one-DOF mass matrix by float32 epsilon."""
        model = _build_pendulum(wp.get_device())
        oracle = ResponseOracle(model)
        oracle._H.zero_()

        oracle._invert_blocks()

        expected = 1.0 / np.finfo(np.float32).eps
        self.assertAlmostEqual(float(oracle.inverse_blocks.numpy()[0, 0, 0]), expected)

    def test_inverse_blocks_match_dense_inverse(self):
        """refresh() fills the full per-articulation inverse mass block.

        inverse_blocks[a] must equal inv(H_a).
        """
        device = wp.get_device()
        model = _build_two_link(device, worlds=2)
        n = len(_arm_dofs(model)) // model.world_count
        q0 = np.array([0.3, -0.8], dtype=np.float32)
        state = model.state()
        _set_arm(model, state.joint_q, np.tile(q0, model.world_count))

        oracle = ResponseOracle(model)
        oracle.refresh(state)

        blocks = oracle.inverse_blocks.numpy()
        dofs = len(_arm_dofs(model))
        Hinv = _response_at(model, np.tile(q0, model.world_count), np.zeros(dofs, dtype=np.float32))
        np.testing.assert_allclose(blocks[0, :n, :n], Hinv, rtol=1e-4, atol=1e-6)
        # Each world holds an identical articulation, so their blocks must agree.
        for art in range(1, model.articulation_count):
            np.testing.assert_allclose(blocks[art, :n, :n], blocks[0, :n, :n], rtol=1e-5, atol=1e-6)

    def test_armature_enters_the_response(self):
        """Joint armature is rotor inertia the solver feels, so it must reduce alpha."""
        device = wp.get_device()
        q0 = np.array([0.3, -0.8], dtype=np.float32)

        def alpha_for(armature: float) -> np.ndarray:
            m = _two_link_builder(armature=armature).finalize(device=device)
            st = m.state()
            st.joint_q.assign(q0)
            o = ResponseOracle(m)
            o.refresh(st)
            return np.diag(o.inverse_blocks.numpy()[0, :2, :2]).copy()

        bare = alpha_for(0.0)
        with_armature = alpha_for(0.5)
        self.assertTrue(np.all(with_armature < bare))

        # alpha must match a dense inverse of (H + diag(armature)).
        m = _two_link_builder(armature=0.5).finalize(device=device)
        st = m.state()
        st.joint_q.assign(q0)
        newton.eval_fk(m, st.joint_q, st.joint_qd, st)
        H = newton.eval_mass_matrix(m, st).numpy()[0, :2, :2] + np.diag([0.5, 0.5])
        np.testing.assert_allclose(with_armature, np.diag(np.linalg.inv(H)), rtol=1e-4)

    def test_refresh_does_not_mutate_state(self):
        """``refresh()`` must not write back into the caller's state.

        It runs forward kinematics internally; doing that on the caller's state
        would overwrite ``body_q``/``body_qd``, which are authoritative for
        maximal-coordinate solvers.
        """
        device = wp.get_device()
        model = _build_two_link(device)
        state = model.state()
        state.joint_q.assign(np.array([0.3, -0.8], dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        # Move the body pose away from FK of joint_q.
        body_q = state.body_q.numpy().copy()
        body_q[:, 0] += 5.0
        state.body_q.assign(body_q)

        ResponseOracle(model).refresh(state)
        np.testing.assert_allclose(state.body_q.numpy(), body_q, rtol=0, atol=0)

    def test_multi_articulation_indexing(self):
        """Two articulations in one model solve with their own response blocks.

        Every other test uses a single articulation, so ``art_base`` is always 0
        and an articulation-local index bug would be invisible. Here the second
        articulation's DOFs start at a nonzero base and carry different gains.
        """
        device = wp.get_device()
        h = 0.01
        kp = np.array([300.0, 200.0, 500.0, 400.0], dtype=np.float32)
        kd = np.array([3.0, 2.0, 5.0, 4.0], dtype=np.float32)
        q0 = np.array([0.3, -0.8, 0.1, -0.2], dtype=np.float32)
        target = np.array([0.6, 0.4, -0.3, 0.5], dtype=np.float32)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(2):
            builder.add_builder(_two_link_builder())
        model = builder.finalize(device=device)
        self.assertEqual(model.articulation_count, 2)
        n = model.joint_dof_count
        self.assertEqual(n, 4)

        state = model.state()
        state.joint_q.assign(q0)
        control = model.control()
        control.joint_target_q.assign(target)

        actuator, oracle = _make_implicit_actuator(
            model,
            device,
            kp=wp.array(kp, dtype=float, device=device),
            kd=wp.array(kd, dtype=float, device=device),
        )
        control.joint_f.zero_()
        _refresh_and_step(actuator, oracle, state, control, h)

        # Each articulation is its own 2x2 coupled solve.
        blocks = oracle.inverse_blocks.numpy()
        expected = np.zeros(n, dtype=np.float64)
        for a in range(2):
            sl = slice(2 * a, 2 * a + 2)
            A = blocks[a, :2, :2]
            f0 = kp[sl] * (target[sl] - q0[sl])
            J = np.eye(2) + h * np.diag(h * kp[sl] + kd[sl]) @ A
            expected[sl] = np.linalg.solve(J, h * f0) / h
        np.testing.assert_allclose(_arm_values(model, control.joint_f), expected, rtol=1e-3, atol=1e-3)

    def test_refresh_from_solve_captures_without_warmup(self):
        """refresh_from_solve captures and replays, with no warm-up call first.

        Deliberately captures straight after constructing the oracle, so the whole
        path -- scratch setup, the per-column solves and the scatter -- has to be
        capture-safe. Where the device has no memory pool, allocating on a
        capturing stream would fail here.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("graph capture requires CUDA")

        model = _build_two_link(device, dummy_body=True)
        n = len(_arm_dofs(model))
        state = model.state()
        _set_arm(model, state.joint_q, [0.3, -0.8])
        solver = _mujoco_solver(self, model)
        solver.step(state, model.state(), model.control(), None, 0.01)
        solve_inverse = _mujoco_solve(solver)

        oracle = ResponseOracle(model)  # fresh: nothing allocated by a prior call
        with wp.ScopedCapture(device) as capture:
            oracle.refresh_from_solve(solve_inverse, dof_map=solver.mjc_dof_to_newton_dof)
        wp.capture_launch(capture.graph)

        reference = ResponseOracle(model)
        reference.refresh(state)
        np.testing.assert_allclose(
            oracle.inverse_blocks.numpy()[0, :n, :n],
            reference.inverse_blocks.numpy()[0, :n, :n],
            rtol=2e-3,
            atol=1e-6,
        )

    def test_response_from_mujoco_factorization(self):
        """Fill the oracle response from MuJoCo's per-step factorized inertia.

        MuJoCo refactorizes its inertia at the step-start pose every step, so --
        unlike the compile-time, diagonal-only ``dof_invweight0`` -- the recovered
        inverse tracks inertial coupling at the current configuration. Checks
        :meth:`ResponseOracle.refresh_from_solve` against a host-side
        inverse-and-remap of that inertia, against the built-in oracle, and by
        driving the coupled implicit solve with it.
        """
        device = wp.get_device()
        h = 0.01
        kp = np.array([300.0, 200.0], dtype=np.float32)
        kd = np.array([3.0, 2.0], dtype=np.float32)
        q0 = np.array([0.3, -0.8], dtype=np.float32)  # away from qpos0: alpha differs from invweight0
        target = np.array([0.6, 0.4], dtype=np.float32)

        model = _build_two_link(device, dummy_body=True, worlds=2)
        worlds = model.world_count
        state = model.state()
        _set_arm(model, state.joint_q, np.tile(q0, worlds))
        control = model.control()
        _set_arm(model, control.joint_target_q, np.tile(target, worlds))

        # One solver step populates qM at the state's pose (computed before integration).
        solver = _mujoco_solver(self, model)
        state_out = model.state()
        solver.step(state, state_out, control, None, h)

        n = len(_arm_dofs(model)) // worlds
        self.assertEqual(solver.mj_model.nv * worlds, model.joint_dof_count)

        mjc_oracle = ResponseOracle(model)
        mjc_oracle.refresh_from_solve(_mujoco_solve(solver), dof_map=solver.mjc_dof_to_newton_dof)
        response_newton = mjc_oracle.inverse_blocks.numpy()[0, :n, :n]

        # The solver's mass matrix must agree with the oracle's own dense recompute.
        oracle_ref = ResponseOracle(model)
        oracle_ref.refresh(state)
        np.testing.assert_allclose(response_newton, oracle_ref.inverse_blocks.numpy()[0, :n, :n], rtol=1e-4)

        # Drive the implicit solve with the solver-provided values (no refresh()).
        actuator, oracle = _make_implicit_actuator(
            model,
            device,
            kp=wp.array(np.tile(kp, worlds), dtype=float, device=device),
            kd=wp.array(np.tile(kd, worlds), dtype=float, device=device),
        )
        oracle.refresh_from_solve(_mujoco_solve(solver), dof_map=solver.mjc_dof_to_newton_dof)
        np.testing.assert_allclose(oracle.inverse_blocks.numpy()[0, :n, :n], response_newton, rtol=1e-4)
        control.joint_f.zero_()
        actuator.step(state, control, dt=h)

        f0 = kp * (target - q0)
        jacobian = np.eye(n) + h * np.diag(h * kp + kd) @ response_newton
        expected = np.linalg.solve(jacobian, h * f0) / h
        np.testing.assert_allclose(_arm_values(model, control.joint_f)[:n], expected, rtol=1e-3, atol=1e-3)
        _assert_worlds_match(self, model, control.joint_f, rtol=1e-4, atol=1e-4)

    def test_full_loop_response_from_mujoco_matches_refresh(self):
        """Closed-loop run with the coupled response from MuJoCo's inertia.

        Runs the same simulation twice, updating the oracle response every step
        either from the solver's own factorization (``refresh_from_solve`` --
        the "solver-owned oracle" path) or with the built-in ``oracle.refresh()``.
        The refresh is scheduled at the same one-step-stale phase as the
        factorization, so the trajectories must coincide. On CUDA the
        whole step (actuator + solver + response update) is graph-captured, which
        is what requires both update paths to be kernel-only.
        """
        device = wp.get_device()
        h = 0.005
        outer_iters = 30
        kp = np.array([300.0, 200.0], dtype=np.float32)
        kd = np.array([3.0, 2.0], dtype=np.float32)
        q_init = np.array([0.3, -0.8], dtype=np.float32)
        target = np.array([0.6, 0.4], dtype=np.float32)

        def run(use_qm: bool) -> np.ndarray:
            model = _build_two_link(device, dummy_body=True)
            states = [model.state(), model.state()]
            control = model.control()
            _set_arm(model, control.joint_target_q, target)

            solver = _mujoco_solver(self, model)
            actuator, oracle = _make_implicit_actuator(
                model,
                device,
                kp=wp.array(kp, dtype=float, device=device),
                kd=wp.array(kd, dtype=float, device=device),
            )

            self.assertEqual(solver.mj_model.nv, model.joint_dof_count)
            solve_m = _mujoco_solve(solver)

            def update_response(state_prev: newton.State) -> None:
                if use_qm:
                    # Full inverse response from the solver's factorization at the
                    # pose of the step that just ran — same staleness as refresh().
                    oracle.refresh_from_solve(solve_m, dof_map=solver.mjc_dof_to_newton_dof)
                else:
                    oracle.refresh(state_prev)

            def two_steps():
                for _ in range(2):  # even count: state buffers line up for graph replay
                    control.joint_f.zero_()
                    actuator.step(states[0], control, dt=h)
                    solver.step(states[0], states[1], control, None, h)
                    update_response(states[0])  # states[0] still holds the pre-step pose
                    states[0], states[1] = states[1], states[0]

            def reset():
                _set_arm(model, states[0].joint_q, q_init)
                states[0].joint_qd.zero_()
                newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
                oracle.refresh(states[0])  # prime the response at the initial pose

            reset()
            two_steps()  # warm-up: module loads and lazy allocations before capture

            reset()
            if device.is_cuda:
                with wp.ScopedCapture(device) as capture:
                    two_steps()
                step_fn = lambda: wp.capture_launch(capture.graph)  # noqa: E731
            else:
                step_fn = two_steps

            traj = []
            for _ in range(outer_iters):
                step_fn()
                traj.append(states[0].joint_q.numpy().copy())
            return np.array(traj)

        traj_qm = run(use_qm=True)
        traj_ref = run(use_qm=False)
        self.assertTrue(np.all(np.isfinite(traj_qm)))
        np.testing.assert_allclose(traj_qm, traj_ref, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. Builder — from USD, programmatic, and free-joint replication
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
        self.assertAlmostEqual(clamped.drive.kp.numpy()[0], 100.0, places=3)
        self.assertAlmostEqual(clamped.drive.kd.numpy()[0], 10.0, places=3)
        self.assertIsInstance(clamped.clamping[0], ClampingMaxEffort)
        self.assertAlmostEqual(clamped.clamping[0].max_effort.numpy()[0], 50.0, places=3)

        self.assertEqual(delayed.num_actuators, 1)
        self.assertAlmostEqual(delayed.drive.kp.numpy()[0], 200.0, places=3)
        self.assertAlmostEqual(delayed.drive.kd.numpy()[0], 20.0, places=3)
        np.testing.assert_array_equal(delayed.delay.delay_steps.numpy(), [5])
        self.assertEqual(delayed.delay.buf_depth, 5)

        stage = Usd.Stage.Open(usd_path)
        parsed = parse_actuator_prim(stage.GetPrimAtPath("/World/Robot/Joint1Actuator"))
        self.assertIsNotNone(parsed)
        self.assertIsInstance(parsed, ActuatorParsed)
        self.assertEqual(parsed.drive_class, DrivePD)

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
        self.assertEqual(parsed.drive_class, DrivePD)
        self.assertAlmostEqual(parsed.drive_kwargs["kp"], 100.0)
        self.assertAlmostEqual(parsed.drive_kwargs["kd"], 10.0)

    def test_programmatic(self):
        """Mixed drive types, clamping, and delays via add_actuator.

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

        builder.add_actuator(DrivePD, index=dofs[0], kp=50.0, kd=5.0, const_effort=1.0)
        builder.add_actuator(
            DrivePID,
            index=dofs[1],
            kp=100.0,
            ki=10.0,
            kd=20.0,
            clamping=[
                (ClampingDCMotor, {"saturation_effort": 80.0, "velocity_limit": 15.0, "max_motor_effort": 200.0})
            ],
        )
        builder.add_actuator(DrivePD, index=dofs[2], kp=150.0, delay_steps=4)

        model = builder.finalize()
        self.assertEqual(len(model.actuators), 3)

        pd_plain = next(a for a in model.actuators if isinstance(a.drive, DrivePD) and a.delay is None)
        pid_act = next(a for a in model.actuators if isinstance(a.drive, DrivePID))
        pd_delay = next(a for a in model.actuators if isinstance(a.drive, DrivePD) and a.delay is not None)

        self.assertEqual(pd_plain.num_actuators, 1)
        np.testing.assert_array_almost_equal(pd_plain.drive.kp.numpy(), [50.0])
        np.testing.assert_array_almost_equal(pd_plain.drive.kd.numpy(), [5.0])
        np.testing.assert_array_almost_equal(pd_plain.drive.const_effort.numpy(), [1.0])
        self.assertIsNone(pd_plain.state())

        self.assertEqual(pid_act.num_actuators, 1)
        np.testing.assert_array_almost_equal(pid_act.drive.kp.numpy(), [100.0])
        np.testing.assert_array_almost_equal(pid_act.drive.ki.numpy(), [10.0])
        np.testing.assert_array_almost_equal(pid_act.drive.kd.numpy(), [20.0])
        self.assertIsInstance(pid_act.clamping[0], ClampingDCMotor)
        self.assertAlmostEqual(pid_act.clamping[0].saturation_effort.numpy()[0], 80.0, places=3)
        self.assertAlmostEqual(pid_act.clamping[0].max_motor_effort.numpy()[0], 200.0, places=3)
        pid_state = pid_act.state()
        self.assertIsNotNone(pid_state.drive_state)
        self.assertEqual(pid_state.drive_state.integral.shape, (1,))
        np.testing.assert_array_equal(pid_state.drive_state.integral.numpy(), [0.0])

        self.assertEqual(pd_delay.num_actuators, 1)
        np.testing.assert_array_almost_equal(pd_delay.drive.kp.numpy(), [150.0])
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
            DrivePD, index=dof1, kp=100.0, kd=10.0, pos_index=template.joint_q_start[j1], delay_steps=2
        )
        template.add_actuator(
            DrivePD, index=dof2, kp=200.0, kd=20.0, pos_index=template.joint_q_start[j2], delay_steps=3
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

        np.testing.assert_array_almost_equal(act.drive.kp.numpy(), [100.0, 200.0] * num_envs)
        np.testing.assert_array_almost_equal(act.drive.kd.numpy(), [10.0, 20.0] * num_envs)

        np.testing.assert_array_equal(act.delay.delay_steps.numpy(), [2, 3] * num_envs)
        self.assertEqual(act.delay.buf_depth, 3)

        act_state = act.state()
        self.assertEqual(act_state.delay_state.buffer_pos.shape, (3, n))
        np.testing.assert_array_equal(act_state.delay_state.num_pushes.numpy(), [0] * n)


# ---------------------------------------------------------------------------
# 8. Parameter binding and access via ArticulationView
# ---------------------------------------------------------------------------


class TestActuatorSelectionAPI(unittest.TestCase):
    """Tests for actuator parameter access via ArticulationView."""

    def build_actuator_view(self):
        single_world_builder = newton.ModelBuilder()
        body = single_world_builder.add_link()
        joint = single_world_builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        single_world_builder.add_articulation([joint], label="robot")
        single_world_builder.add_actuator(
            DrivePD,
            index=single_world_builder.joint_qd_start[joint],
            kp=100.0,
        )

        builder = newton.ModelBuilder()
        builder.replicate(single_world_builder, 2)
        model = builder.finalize()
        view = ArticulationView(model, "robot")
        return model.actuators[0], view

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
            single_articulation_builder.add_actuator(DrivePD, index=dof, kp=100.0 * (i + 1))

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

        kp_values = joint_view.get_actuator_parameter(actuator, actuator.drive, "kp").numpy().copy()

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
        joint_view.set_actuator_parameter(actuator, actuator.drive, "kp", wp_kp, mask=mask)

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

        measured_kp = actuator.drive.kp.numpy()
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

    def test_set_actuator_parameter_rejects_invalid_masks_before_launch(self):
        actuator, view = self.build_actuator_view()
        values = wp.ones((view.world_count, 1), dtype=wp.float32, device=view.device)

        invalid_masks = (
            (wp.ones((view.world_count, 1), dtype=wp.bool, device=view.device), "mask shape"),
            (wp.ones(view.world_count, dtype=wp.int32, device=view.device), "Boolean mask"),
        )
        if wp.is_cuda_available():
            other_device = "cpu" if view.device.is_cuda else "cuda:0"
            invalid_masks += ((wp.ones(view.world_count, dtype=wp.bool, device=other_device), "device"),)

        for mask, message in invalid_masks:
            with self.subTest(shape=mask.shape, dtype=mask.dtype, device=mask.device):
                with patch.object(wp, "launch") as launch:
                    with self.assertRaisesRegex(ValueError, message):
                        view.set_actuator_parameter(actuator, actuator.drive, "kp", values, mask=mask)
                    launch.assert_not_called()

    def test_selection_api_updates_implicit_solve(self):
        """Gain writes reach the installed implicit solve, through either write path.

        ``set_effort_mode_implicit`` re-points the drive's parameter arrays
        at columns of a packed array. Both the masked scatter used by
        ``set_actuator_parameter`` and a direct ``.assign`` must land in that
        pack. Re-installing the mode must also reuse the same pack, otherwise
        the second bind would detach the views handed out by the first.
        """
        device = wp.get_device()
        h = 0.01
        kp1, kp2, kd_val = 500.0, 2000.0, 5.0
        q0, target = 0.2, 1.0

        model = _build_pendulum(device)
        state = model.state()
        state.joint_q.assign(np.array([q0], dtype=np.float32))
        control = model.control()
        control.joint_target_q.assign(np.array([target], dtype=np.float32))

        actuator, oracle = _make_implicit_actuator(
            model,
            device,
            kp=wp.array([kp1], dtype=float, device=device),
            kd=wp.array([kd_val], dtype=float, device=device),
        )

        view = ArticulationView(model, "*", verbose=False)
        np.testing.assert_allclose(view.get_actuator_parameter(actuator, actuator.drive, "kp").numpy(), [[kp1]])
        view.set_actuator_parameter(actuator, actuator.drive, "kp", wp.array([[kp2]], dtype=float, device=device))
        np.testing.assert_allclose(actuator.drive.kp.numpy(), [kp2])

        control.joint_f.zero_()
        _refresh_and_step(actuator, oracle, state, control, h)
        expected = _expected_implicit_pd(model, state, kp2, kd_val, target, h)
        self.assertAlmostEqual(control.joint_f.numpy()[0], expected, delta=abs(expected) * 1e-4)

        # Re-installing must keep the same pack, so a direct assign still lands.
        pack = actuator.drive._param_pack
        actuator.set_effort_mode_implicit(response=oracle)
        self.assertIs(actuator.drive._param_pack, pack)
        actuator.drive.kp.assign(np.array([kp1], dtype=np.float32))
        control.joint_f.zero_()
        _refresh_and_step(actuator, oracle, state, control, h)
        expected = _expected_implicit_pd(model, state, kp1, kd_val, target, h)
        self.assertAlmostEqual(control.joint_f.numpy()[0], expected, delta=abs(expected) * 1e-4)


# ---------------------------------------------------------------------------
# 9. State reset (masked and full)
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

        def _f(vals: Sequence[float]) -> wp.array[float]:
            return wp.array(vals, dtype=wp.float32, device=device)

        indices = wp.array(list(range(n)), dtype=wp.uint32, device=device)
        ctrl = DrivePID(
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
        with patch("newton._src.actuators.drives.drive_pid.wp.launch", wraps=wp.launch) as launch:
            state_0.reset(mask)

        launch.assert_called_once()
        self.assertEqual(launch.call_args.kwargs["device"], state_0.integral.device)

        integral_after = state_0.integral.numpy()
        self.assertAlmostEqual(integral_after[0], 0.0, places=6, msg="DOF 0 should be reset")
        self.assertAlmostEqual(integral_after[1], integral_before[1], places=6, msg="DOF 1 should be untouched")
        self.assertAlmostEqual(integral_after[2], 0.0, places=6, msg="DOF 2 should be reset")

    def test_pid_masked_reset_rejects_invalid_mask(self):
        state = DrivePID.State(integral=wp.zeros(3, dtype=wp.float32, device="cpu"))

        with self.assertRaisesRegex(ValueError, "one-dimensional Boolean array"):
            state.reset(wp.zeros(3, dtype=wp.int32, device="cpu"))
        with self.assertRaisesRegex(ValueError, "one-dimensional Boolean array"):
            state.reset(wp.zeros((1, 3), dtype=wp.bool, device="cpu"))
        with self.assertRaisesRegex(ValueError, r"mask length \(2\) must match integral length \(3\)"):
            state.reset(wp.zeros(2, dtype=wp.bool, device="cpu"))

    @unittest.skipUnless(wp.get_cuda_device_count() > 0, "CUDA device required")
    def test_pid_masked_reset_rejects_wrong_device(self):
        state = DrivePID.State(integral=wp.zeros(3, dtype=wp.float32, device="cuda:0"))
        mask = wp.zeros(3, dtype=wp.bool, device="cpu")

        with self.assertRaisesRegex(ValueError, "mask device .* must match integral device"):
            state.reset(mask)

    def test_actuator_composed_reset(self):
        """Actuator.State.reset delegates to both delay and drive sub-states."""
        num_envs = 2
        device = wp.get_device()

        template = newton.ModelBuilder()
        link = template.add_link()
        joint = template.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
        template.add_articulation([joint])
        dof = template.joint_qd_start[joint]
        template.add_actuator(DrivePID, index=dof, kp=50.0, ki=10.0, kd=5.0, delay_steps=2)

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
        self.assertTrue(all(v > 0 for v in state_0.drive_state.integral.numpy()))

        mask = wp.array([True, False], dtype=wp.bool, device=device)
        state_0.reset(mask)

        self.assertEqual(state_0.delay_state.num_pushes.numpy()[0], 0, "env 0 delay should be reset")
        self.assertGreater(state_0.delay_state.num_pushes.numpy()[1], 0, "env 1 delay should be untouched")
        self.assertAlmostEqual(
            state_0.drive_state.integral.numpy()[0], 0.0, places=6, msg="env 0 integral should be reset"
        )
        self.assertGreater(state_0.drive_state.integral.numpy()[1], 0.0, msg="env 1 integral should be untouched")


# ---------------------------------------------------------------------------
# 10. CUDA graph capture — end-to-end with Newton solver + delayed actuator
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
            DrivePD,
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

        def _loop(
            solver: newton.solvers.SolverMuJoCo,
            s0: newton.State,
            s1: newton.State,
            ctrl: newton.Control,
            act: Actuator,
            act_a: Actuator.State,
            act_b: Actuator.State,
            n: int,
        ) -> tuple[newton.State, newton.State, Actuator.State, Actuator.State]:
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
        wp.copy(ctrl.joint_target_q, wp.full(ndof, warmup_target, dtype=wp.float32, device=device))
        s0, s1, act_a, act_b = _loop(solver, s0, s1, ctrl, act, act_a, act_b, max_delay)
        eager_results = []
        for tgt in cycle_targets:
            wp.copy(ctrl.joint_target_q, wp.full(ndof, tgt, dtype=wp.float32, device=device))
            s0, s1, act_a, act_b = _loop(solver, s0, s1, ctrl, act, act_a, act_b, N)
            eager_results.append(s0.joint_q.numpy().copy())

        # --- Graph ---
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


class TestActuatorStateAssign(unittest.TestCase):
    """Cover the :meth:`Actuator.State.assign` copy contract."""

    def test_assign_copies_built_in_state(self):
        """Copy every array in the built-in delay and PID drive state."""
        device = wp.get_device()
        builder = newton.ModelBuilder()
        link = builder.add_link()
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
        builder.add_articulation([joint])
        builder.add_actuator(DrivePID, index=builder.joint_qd_start[joint], kp=0.0, ki=1.0, kd=0.0, delay_steps=2)
        actuator = builder.finalize(device=device).actuators[0]
        current, advanced = actuator.state(), actuator.state()

        delay_fields = ("buffer_pos", "buffer_vel", "buffer_act", "num_pushes", "write_idx")
        for value, name in enumerate(delay_fields, start=1):
            getattr(advanced.delay_state, name).fill_(value)
        advanced.drive_state.integral.fill_(6.0)

        current.assign(advanced)

        for value, name in enumerate(delay_fields, start=1):
            with self.subTest(name=name):
                np.testing.assert_array_equal(getattr(current.delay_state, name).numpy(), value)
        np.testing.assert_array_equal(current.drive_state.integral.numpy(), 6.0)

    def test_assign_delegates_to_custom_drive_state(self):
        """Delegate nested and slotted storage to the custom drive state."""

        class _CustomState(DriveBase.State):
            __slots__ = ("nested",)

            def __init__(self, num_actuators: int, device: wp.Device):
                self.nested = types.SimpleNamespace(value=wp.zeros(num_actuators, dtype=wp.float32, device=device))

            def assign(self, other: DriveBase.State) -> None:
                self.nested.value.assign(other.nested.value)

        device = wp.get_device()
        current, advanced = _CustomState(1, device), _CustomState(1, device)
        advanced.nested.value.fill_(1.0)

        Actuator.State(drive_state=current).assign(Actuator.State(drive_state=advanced))

        self.assertEqual(current.nested.value.numpy()[0], 1.0)
        self.assertIsNot(current.nested, advanced.nested)

    def test_assign_copies_slotted_dataclass_state(self):
        """Copy a custom slotted dataclass through the default implementation."""

        @dataclass(slots=True)
        class _CustomState(DriveBase.State):
            value: wp.array | None = None

        device = wp.get_device()
        current = _CustomState(wp.zeros(1, dtype=wp.float32, device=device))
        advanced = _CustomState(wp.ones(1, dtype=wp.float32, device=device))
        destination = current.value

        Actuator.State(drive_state=current).assign(Actuator.State(drive_state=advanced))

        self.assertIs(current.value, destination)
        self.assertEqual(current.value.numpy()[0], 1.0)

    def test_assign_copies_neural_warp_state_in_place(self):
        """Copy neural Warp arrays while preserving destination storage."""
        device = wp.get_device()
        cases = (
            (
                DriveNeuralMLP.State(
                    pos_error_history=wp.zeros((2, 1), dtype=wp.float32, device=device),
                    vel_history=wp.zeros((2, 1), dtype=wp.float32, device=device),
                ),
                DriveNeuralMLP.State(
                    pos_error_history=wp.ones((2, 1), dtype=wp.float32, device=device),
                    vel_history=wp.ones((2, 1), dtype=wp.float32, device=device),
                ),
                ("pos_error_history", "vel_history"),
            ),
            (
                DriveNeuralLSTM.State(
                    hidden=wp.zeros((1, 1, 2), dtype=wp.float32, device=device),
                    cell=wp.zeros((1, 1, 2), dtype=wp.float32, device=device),
                ),
                DriveNeuralLSTM.State(
                    hidden=wp.ones((1, 1, 2), dtype=wp.float32, device=device),
                    cell=wp.ones((1, 1, 2), dtype=wp.float32, device=device),
                ),
                ("hidden", "cell"),
            ),
        )

        for current, advanced, fields in cases:
            destinations = {name: getattr(current, name) for name in fields}
            Actuator.State(drive_state=current).assign(Actuator.State(drive_state=advanced))
            for name in fields:
                destination = destinations[name]
                self.assertIs(getattr(current, name), destination)
                np.testing.assert_array_equal(destination.numpy(), 1.0)

    def test_assign_rejects_undecorated_custom_drive_state(self):
        """Reject an undeclared custom state rather than silently skipping it."""

        class _CustomState(DriveBase.State):
            def __init__(self, device: wp.Device):
                self.value = wp.zeros(1, dtype=wp.float32, device=device)

        current = _CustomState(wp.get_device())
        advanced = _CustomState(wp.get_device())
        with self.assertRaisesRegex(NotImplementedError, "decorated with @dataclass or implement assign"):
            Actuator.State(drive_state=current).assign(Actuator.State(drive_state=advanced))

    def test_assign_rejects_undeclared_attribute(self):
        """Reject an attribute present on only one otherwise compatible state."""

        @dataclass
        class _CustomState(DriveBase.State):
            value: wp.array | None = None

        device = wp.get_device()
        current = _CustomState(wp.zeros(1, dtype=wp.float32, device=device))
        advanced = _CustomState(wp.ones(1, dtype=wp.float32, device=device))
        current.extra = None

        with self.assertRaisesRegex(ValueError, "undeclared state attributes: extra"):
            Actuator.State(drive_state=current).assign(Actuator.State(drive_state=advanced))

    @unittest.skipUnless(_HAS_TORCH, "PyTorch is required")
    def test_assign_copies_torch_state_without_aliasing(self):
        """Copy Torch-backed drive state without aliasing the source tensors."""
        with _torch.inference_mode():
            current = DriveNeuralLSTM.State(hidden=_torch.zeros((1, 1, 1)), cell=_torch.zeros((1, 1, 1)))
            advanced = DriveNeuralLSTM.State(hidden=_torch.ones((1, 1, 1)), cell=_torch.ones((1, 1, 1)))

        Actuator.State(drive_state=current).assign(Actuator.State(drive_state=advanced))

        self.assertIsNot(current.hidden, advanced.hidden)
        self.assertIsNot(current.cell, advanced.cell)
        with _torch.inference_mode():
            advanced.hidden.zero_()
            advanced.cell.zero_()
        self.assertEqual(current.hidden.item(), 1.0)
        self.assertEqual(current.cell.item(), 1.0)

    def test_assign_rejects_mismatched_components(self):
        """Raise rather than silently skip when only one state holds a component."""
        state = DrivePID.State(integral=wp.zeros(1, dtype=wp.float32, device=wp.get_device()))
        with self.assertRaisesRegex(ValueError, "drive_state"):
            Actuator.State(drive_state=state).assign(Actuator.State())


@unittest.skipUnless(
    wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()),
    "CUDA graph capture requires CUDA device with memory pools",
)
class TestDriveStateGraphCapture(unittest.TestCase):
    """Drive state must advance per replay, not only for an even captured step count.

    A PID with only ``ki`` set, held at a constant position error by a model no
    solver moves, accumulates exactly ``ki * error * dt`` per actuator step
    however the loop is chunked.  A graph cannot re-point its state buffers
    between replays, so an odd-length captured region needs either a boundary
    :meth:`Actuator.State.assign` or a second graph of the opposite parity.
    """

    DT = 0.01
    KI = 1.0
    TARGET = 1.0
    STEPS = 12

    def _integral_after_all_steps(self, implicit: bool, steps_per_graph: int | None, parity_graphs: bool) -> float:
        """Run :attr:`STEPS` actuator steps and return the resulting PID integral.

        Args:
            implicit: Solve the control law implicitly instead of explicitly.
                Both effort modes advance the integral through the same state.
            steps_per_graph: Steps per captured region, or ``None`` to run eagerly.
            parity_graphs: Capture one graph per state-buffer orientation and
                alternate them, instead of assigning at the region boundary.
        """
        device = wp.get_device()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        # Mass and inertia only matter to the implicit response oracle.
        body = builder.add_link(com=wp.vec3(0.5, 0.0, 0.0), inertia=_POINT_MASS_INERTIA, mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        builder.add_articulation([joint])
        builder.add_actuator(DrivePID, index=builder.joint_qd_start[joint], kp=0.0, ki=self.KI, kd=0.0)
        model = builder.finalize(device=device)

        actuator = model.actuators[0]
        oracle = ResponseOracle(model) if implicit else None
        if oracle is not None:
            actuator.set_effort_mode_implicit(response=oracle)
        state, control = model.state(), model.control()
        control.joint_target_q.fill_(self.TARGET)  # joint_q stays 0, so the error is constant
        s0, s1 = actuator.state(), actuator.state()

        def run(s0, s1, steps, boundary_assign=False):
            """Step the actuator, swapping state as the documented loop does."""
            for i in range(steps):
                control.joint_f.zero_()
                if oracle is not None:
                    oracle.refresh(state)
                actuator.step(state, control, s0, s1, dt=self.DT)
                if boundary_assign and steps % 2 == 1 and i == steps - 1:
                    s0.assign(s1)  # keeps a single odd-length graph correct
                else:
                    s0, s1 = s1, s0
            return s0, s1

        if steps_per_graph is None:
            s0, s1 = run(s0, s1, self.STEPS)
        else:
            # Module loading and lazy allocation have to happen before a capture.
            s0, s1 = run(s0, s1, 1)
            s0.drive_state.integral.zero_()
            s1.drive_state.integral.zero_()
            graphs = {}
            for _ in range(self.STEPS // steps_per_graph):
                # Keying on the current buffer builds one graph per orientation.
                key = id(s0) if parity_graphs else None
                if key not in graphs:
                    with wp.ScopedCapture(device) as capture:
                        after = run(s0, s1, steps_per_graph, boundary_assign=not parity_graphs)
                    graphs[key] = (capture.graph, after)
                graph, (s0, s1) = graphs[key]
                wp.capture_launch(graph)
            self.assertLessEqual(len(graphs), 2, msg="alternating parity needs at most two graphs")
        return float(s0.drive_state.integral.numpy()[0])

    def test_pid_integral_advances_per_replay_boundary_assign(self):
        """Assign at an odd-length region's boundary and match the eager integral."""
        expected = self.KI * self.TARGET * self.DT * self.STEPS
        for implicit in (False, True):
            for steps_per_graph in (None, 1, 2, 3):
                with self.subTest(implicit=implicit, steps_per_graph=steps_per_graph):
                    got = self._integral_after_all_steps(implicit, steps_per_graph, parity_graphs=False)
                    self.assertAlmostEqual(got, expected, places=6)

    def test_pid_integral_advances_per_replay_parity_graphs(self):
        """Alternate one graph per buffer orientation and match the eager integral."""
        expected = self.KI * self.TARGET * self.DT * self.STEPS
        for steps_per_graph in (1, 2, 3):
            with self.subTest(steps_per_graph=steps_per_graph):
                got = self._integral_after_all_steps(False, steps_per_graph, parity_graphs=True)
                self.assertAlmostEqual(got, expected, places=6)


# ---------------------------------------------------------------------------
# 11. Neural drive via USD parsing (parse_actuator_prim)
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_USD and _HAS_ONNX, "pxr or onnx not installed")
class TestNeuralActuatorUsdParsing(unittest.TestCase):
    """Verify ``parse_actuator_prim`` correctly handles neural drive
    prims with asset-typed ``newton:modelPath`` attributes.

    This exercises the full USD parsing path — the same path that
    ``ModelBuilder.add_usd`` uses — rather than constructing drives
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
        self.assertEqual(parsed.drive_class, DriveNeuralMLP)
        self.assertEqual(parsed.drive_kwargs["model_path"], model_path)
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
        self.assertEqual(parsed.drive_class, DriveNeuralLSTM)
        self.assertEqual(parsed.drive_kwargs["model_path"], model_path)


# ---------------------------------------------------------------------------
# 12. target_pos_indices separation from pos_indices
# ---------------------------------------------------------------------------


class TestTargetPosIndicesSeparation(unittest.TestCase):
    """Actuator must read joint_target_q via target_pos_indices, not pos_indices."""

    def test_target_pos_read_from_dof_index_not_coord_index(self):
        device = wp.get_device()

        def _a(vals: Sequence[float], dtype: type = wp.float32) -> wp.array[Any]:
            return wp.array(vals, dtype=dtype, device=device)

        kp = 100.0
        actual_pos = 0.5
        correct_target = 2.0
        sentinel = 99.0  # placed at coord index 3 to catch wrong index usage

        indices = _a([1], dtype=wp.uint32)  # DOF index 1
        pos_indices = _a([3], dtype=wp.uint32)  # coord index 3 (joint_q layout)
        target_pos_indices = _a([1], dtype=wp.uint32)  # DOF index 1 (legacy DOF target layout)

        ctrl = DrivePD(kp=_a([kp]), kd=_a([0.0]), const_effort=_a([0.0]))
        actuator = Actuator(
            indices=indices,
            drive=ctrl,
            pos_indices=pos_indices,
            target_pos_indices=target_pos_indices,
        )

        # joint_q is coord-shaped; actual position at coord index 3
        joint_q = _a([0.0, 0.0, 0.0, actual_pos])
        joint_qd = _a([0.0, 0.0])
        # joint_target_q padded to size 4 so both index 1 (correct) and
        # index 3 (sentinel) are reachable — lets us distinguish the two code paths
        joint_target_q = _a([0.0, correct_target, 0.0, sentinel])
        joint_target_qd = _a([0.0, 0.0, 0.0, 0.0])
        joint_f = wp.zeros(4, dtype=wp.float32, device=device)

        sim_state = types.SimpleNamespace(joint_q=joint_q, joint_qd=joint_qd)
        sim_control = types.SimpleNamespace(
            joint_target_q=joint_target_q,
            joint_target_qd=joint_target_qd,
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


class TestControlTargetAttrDefaults(unittest.TestCase):
    """``control_target_pos_attr`` / ``control_target_vel_attr`` accept ``None``.

    Both parameters used to default to ``None``, meaning "resolve against the
    active target layout". The layout switch removed that resolution step, but
    callers may still pass ``None`` explicitly; it must keep selecting the
    canonical names instead of reaching ``getattr()`` with a non-string.
    """

    def _actuator(self, **kwargs):
        device = wp.get_device()
        indices = wp.array([0], dtype=wp.uint32, device=device)
        drive = DrivePD(
            kp=wp.array([10.0], dtype=wp.float32, device=device),
            kd=wp.array([0.0], dtype=wp.float32, device=device),
        )
        return Actuator(indices=indices, drive=drive, **kwargs)

    def test_omitted_attrs_default_to_canonical_names(self):
        """Verify omitted attributes select canonical names."""
        actuator = self._actuator()
        self.assertEqual(actuator.control_target_pos_attr, "joint_target_q")
        self.assertEqual(actuator.control_target_vel_attr, "joint_target_qd")

    def test_explicit_none_normalizes_to_canonical_names(self):
        """Verify explicit None normalizes to canonical names."""
        actuator = self._actuator(control_target_pos_attr=None, control_target_vel_attr=None)
        self.assertEqual(actuator.control_target_pos_attr, "joint_target_q")
        self.assertEqual(actuator.control_target_vel_attr, "joint_target_qd")

    def test_explicit_none_still_steps(self):
        """Verify stepping succeeds after passing explicit None."""
        device = wp.get_device()
        actuator = self._actuator(control_target_pos_attr=None, control_target_vel_attr=None)

        def _a(vals: Sequence[float]) -> wp.array[float]:
            return wp.array(vals, dtype=wp.float32, device=device)

        sim_state = types.SimpleNamespace(joint_q=_a([0.0]), joint_qd=_a([0.0]))
        sim_control = types.SimpleNamespace(
            joint_target_q=_a([1.0]),
            joint_target_qd=_a([0.0]),
            joint_act=None,
            joint_f=wp.zeros(1, dtype=wp.float32, device=device),
        )
        actuator.step(sim_state, sim_control, dt=0.01)
        # kp * (target - pos) = 10 * (1.0 - 0.0)
        self.assertAlmostEqual(float(sim_control.joint_f.numpy()[0]), 10.0, places=4)

    def test_custom_attr_names_are_preserved(self):
        """Verify caller-supplied attribute names remain unchanged."""
        actuator = self._actuator(control_target_pos_attr="my_pos", control_target_vel_attr="my_vel")
        self.assertEqual(actuator.control_target_pos_attr, "my_pos")
        self.assertEqual(actuator.control_target_vel_attr, "my_vel")


if __name__ == "__main__":
    unittest.main()
