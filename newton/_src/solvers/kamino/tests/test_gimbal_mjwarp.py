# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MJWarp conformance tests for right- and left-handed rotational D6 joints."""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
from newton import Contacts
from newton.solvers import SolverKamino, SolverMuJoCo

_DEVICE = "cuda:0"
_DT = 1.0 / 120.0
_RH_AXES = (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
_LH_AXES = (newton.Axis.X, newton.Axis.Z, newton.Axis.Y)
_BASE_INERTIA = wp.mat33(0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0)
_LINK_INERTIA = wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4)


@dataclass(frozen=True)
class _Fixture:
    """Model and D6 layout metadata for one conformance fixture."""

    model: newton.Model
    q_start: int
    qd_start: int
    target_q_start: int


@dataclass(frozen=True)
class _Probe:
    """Raw coordinate, velocity, and effort data from a solver rollout."""

    q: np.ndarray
    qd: np.ndarray
    effort: np.ndarray
    coord_count: int
    dof_count: int


def _build_fixture(
    fixed_base: bool,
    axes: tuple[newton.Axis, newton.Axis, newton.Axis],
    *,
    stiffness: float = 0.0,
    drive_damping: float = 0.0,
    armature: float = 0.0,
    passive_damping: float = 0.0,
    lower: float = -newton.MAXVAL,
    upper: float = newton.MAXVAL,
) -> _Fixture:
    """Build a collision-free articulated rotational D6."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    base = builder.add_link(mass=2.0, inertia=_BASE_INERTIA, label="base")
    link = builder.add_link(mass=1.0, inertia=_LINK_INERTIA, label="link")
    root = (
        builder.add_joint_fixed(parent=-1, child=base, label="root")
        if fixed_base
        else builder.add_joint_free(parent=-1, child=base, label="root")
    )
    configs = [
        newton.ModelBuilder.JointDofConfig(
            axis=axis,
            target_pos=0.0,
            target_vel=0.0,
            target_ke=stiffness,
            target_kd=drive_damping,
            damping=passive_damping,
            armature=armature,
            limit_lower=lower,
            limit_upper=upper,
            limit_ke=1.0e4,
            limit_kd=100.0,
        )
        for i, axis in enumerate(axes)
    ]
    d6 = builder.add_joint_d6(base, link, angular_axes=configs, label="d6")
    builder.add_articulation([root, d6], label="d6")
    model = builder.finalize(device=_DEVICE)
    return _Fixture(
        model,
        int(model.joint_q_start.numpy()[d6]),
        int(model.joint_qd_start.numpy()[d6]),
        int(model.joint_target_q_start.numpy()[d6]),
    )


def _make_solver(backend: str, model: newton.Model) -> SolverKamino | SolverMuJoCo:
    """Create a configured collision-free conformance solver."""
    if backend == "kamino":
        config = SolverKamino.Config(
            integrator="euler",
            use_collision_detector=False,
            use_fk_solver=False,
            sparse_jacobian=True,
        )
        config.constraints.alpha = 0.0
        config.constraints.beta = 0.1
        config.padmm.max_iterations = 200
        config.padmm.primal_tolerance = 1.0e-6
        config.padmm.dual_tolerance = 1.0e-6
        config.padmm.compl_tolerance = 1.0e-6
        return SolverKamino(model, config)
    if backend == "mjwarp":
        return SolverMuJoCo(
            model, disable_contacts=True, integrator="implicitfast", iterations=100, use_mujoco_contacts=False
        )
    raise ValueError(f"Unsupported conformance backend: {backend}")


def _run(
    backend: str,
    fixed_base: bool,
    axes: tuple[newton.Axis, newton.Axis, newton.Axis],
    *,
    q: np.ndarray | None = None,
    qd: np.ndarray | None = None,
    effort: np.ndarray | None = None,
    position_target: np.ndarray | None = None,
    velocity_target: np.ndarray | None = None,
    steps: int = 1,
    **fixture_kwargs,
) -> _Probe:
    """Run a D6 rollout and retain the raw D6 coordinate trajectory."""
    fixture = _build_fixture(fixed_base, axes, **fixture_kwargs)
    state_in, state_out, control = fixture.model.state(), fixture.model.state(), fixture.model.control()
    if q is not None:
        values = state_in.joint_q.numpy()
        values[fixture.q_start : fixture.q_start + 3] = q
        state_in.joint_q.assign(values)
    if qd is not None:
        values = state_in.joint_qd.numpy()
        values[fixture.qd_start : fixture.qd_start + 3] = qd
        state_in.joint_qd.assign(values)
    if effort is not None:
        values = np.zeros(fixture.model.joint_dof_count, dtype=np.float32)
        values[fixture.qd_start : fixture.qd_start + 3] = effort
        control.joint_f.assign(values)
    if position_target is not None:
        values = control.joint_target_q.numpy()
        values[fixture.target_q_start : fixture.target_q_start + 3] = position_target
        control.joint_target_q.assign(values)
    if velocity_target is not None:
        values = control.joint_target_qd.numpy()
        values[fixture.qd_start : fixture.qd_start + 3] = velocity_target
        control.joint_target_qd.assign(values)
    newton.eval_fk(fixture.model, state_in.joint_q, state_in.joint_qd, state_in)
    solver = _make_solver(backend, fixture.model)
    contacts = Contacts(rigid_contact_max=0, soft_contact_max=0, device=_DEVICE) if backend == "mjwarp" else None
    positions = [state_in.joint_q.numpy()[fixture.q_start : fixture.q_start + 3].copy()]
    velocities = [state_in.joint_qd.numpy()[fixture.qd_start : fixture.qd_start + 3].copy()]
    for _ in range(steps):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, _DT)
        state_in, state_out = state_out, state_in
        positions.append(state_in.joint_q.numpy()[fixture.q_start : fixture.q_start + 3].copy())
        velocities.append(state_in.joint_qd.numpy()[fixture.qd_start : fixture.qd_start + 3].copy())
    return _Probe(
        np.stack(positions),
        np.stack(velocities),
        control.joint_f.numpy()[fixture.qd_start : fixture.qd_start + 3].copy(),
        fixture.model.joint_coord_count,
        fixture.model.joint_dof_count,
    )


@unittest.skipUnless(
    wp.get_cuda_device_count() and os.getenv("NEWTON_RUN_MJWARP_CONFORMANCE") == "1",
    "set NEWTON_RUN_MJWARP_CONFORMANCE=1 on a CUDA host to run MJWarp D6 conformance",
)
class TestGimbalMJWarp(unittest.TestCase):
    """Compare Kamino and MJWarp through the public Newton state layout."""

    def _assert_pair(
        self, fixed_base: bool, axes, *, scenario: str, rtol: float = 1.0e-2, atol: float = 1.0e-2, **kwargs
    ):
        """Run and compare both solvers for one D6 scenario."""
        mjwarp = _run("mjwarp", fixed_base, axes, **kwargs)
        kamino = _run("kamino", fixed_base, axes, **kwargs)
        np.testing.assert_allclose(mjwarp.q, kamino.q, rtol=rtol, atol=atol, err_msg=f"{scenario}: q")
        np.testing.assert_allclose(mjwarp.qd, kamino.qd, rtol=rtol, atol=atol, err_msg=f"{scenario}: qd")
        return mjwarp, kamino

    def test_layout_and_state_writers(self):
        """Match D6 layout and independent coordinate/rate writes."""
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base, scenario="layout"):
                    mjwarp = _run("mjwarp", fixed_base, axes, steps=0)
                    kamino = _run("kamino", fixed_base, axes, steps=0)
                    root_coords, root_dofs = (0, 0) if fixed_base else (7, 6)
                    self.assertEqual(mjwarp.coord_count, root_coords + 3)
                    self.assertEqual(mjwarp.dof_count, root_dofs + 3)
                    self.assertEqual(kamino.coord_count, root_coords + 3)
                    self.assertEqual(kamino.dof_count, root_dofs + 3)
                for axis in range(3):
                    q = np.zeros(3, dtype=np.float32)
                    qd = np.zeros(3, dtype=np.float32)
                    q[axis] = -0.1 if axis == 1 else 0.1
                    qd[axis] = -0.2 if axis == 2 else 0.2
                    with self.subTest(axes=axes, fixed_base=fixed_base, axis=axis):
                        self._assert_pair(
                            fixed_base, axes, scenario="state writers", q=q, qd=qd, steps=1, rtol=1.0e-5, atol=1.0e-5
                        )

    def test_effort_trajectories(self):
        """Match direct generalized-effort rollouts."""
        kwargs = {
            "q": np.array([0.9, -0.7, 0.5], dtype=np.float32),
            "effort": np.array([1.0, -0.75, 0.5], dtype=np.float32),
            "steps": 10,
        }
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base):
                    mjwarp, kamino = self._assert_pair(fixed_base, axes, scenario="effort", **kwargs)
                    np.testing.assert_array_equal(mjwarp.effort, kwargs["effort"])
                    np.testing.assert_array_equal(kamino.effort, kwargs["effort"])

    def test_effort_with_armature_trajectories(self):
        """Match implicit effort-with-armature rollouts."""
        kwargs = {
            "q": np.array([0.9, -0.7, 0.5], dtype=np.float32),
            "effort": np.array([1.0, -0.75, 0.5], dtype=np.float32),
            "armature": 0.5,
            "steps": 10,
        }
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base):
                    mjwarp, kamino = self._assert_pair(fixed_base, axes, scenario="effort with armature", **kwargs)
                    np.testing.assert_array_equal(mjwarp.effort, kwargs["effort"])
                    np.testing.assert_array_equal(kamino.effort, kwargs["effort"])

    def test_pd_trajectories(self):
        """Match PD, implicit PD, and unwrap rollouts."""
        scenarios = (
            (
                "large-angle-pd",
                {
                    "q": np.array([1.2, -0.9, 0.7], dtype=np.float32),
                    "qd": np.array([0.4, -0.3, 0.2], dtype=np.float32),
                    "position_target": np.array([-0.6, 0.5, -0.4], dtype=np.float32),
                    "stiffness": 20.0,
                    "drive_damping": 8.0,
                    "steps": 40,
                },
            ),
            (
                "implicit-pd",
                {
                    "position_target": np.array([0.12, -0.08, 0.05], dtype=np.float32),
                    "stiffness": 80.0,
                    "drive_damping": 12.0,
                    "steps": 20,
                },
            ),
            (
                "pd-beyond-pi",
                {
                    "q": np.array([2.6, -2.5, 2.4], dtype=np.float32),
                    "position_target": np.array([np.pi + 0.4, -np.pi - 0.3, np.pi + 0.2], dtype=np.float32),
                    "stiffness": 80.0,
                    "drive_damping": 12.0,
                    "steps": 20,
                },
            ),
        )
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                for scenario, kwargs in scenarios:
                    with self.subTest(axes=axes, fixed_base=fixed_base, scenario=scenario):
                        tolerance = {"rtol": 2.0e-2, "atol": 5.0e-3} if "pd" in scenario else {}
                        self._assert_pair(fixed_base, axes, scenario=scenario, **tolerance, **kwargs)

    def test_limits_and_passive_damping(self):
        """Match position-limit and passive-damping D6 behavior."""
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base, scenario="limits"):
                    _mjwarp, kamino = self._assert_pair(
                        fixed_base,
                        axes,
                        scenario="limits",
                        position_target=np.array([0.6, -0.6, 0.6], dtype=np.float32),
                        stiffness=100.0,
                        drive_damping=15.0,
                        lower=-0.2,
                        upper=0.2,
                        steps=120,
                    )
                    self.assertGreater(np.max(np.abs(kamino.q[-1])), 1.0e-3)
                    self.assertTrue(np.all(kamino.q[-1] <= 0.2 + 1.0e-3))
                    self.assertTrue(np.all(kamino.q[-1] >= -0.2 - 1.0e-3))
                for axis in range(3):
                    velocity = np.zeros(3, dtype=np.float32)
                    velocity[axis] = 0.5
                    with self.subTest(axes=axes, fixed_base=fixed_base, axis=axis, scenario="damping"):
                        baseline = _run("kamino", fixed_base, axes, qd=velocity)
                        damped = _run("kamino", fixed_base, axes, qd=velocity, passive_damping=2.0)
                        self.assertLess(abs(damped.qd[-1, axis]), abs(baseline.qd[-1, axis]))
                with self.subTest(axes=axes, fixed_base=fixed_base, scenario="damping-match"):
                    self._assert_pair(
                        fixed_base,
                        axes,
                        scenario="passive damping",
                        q=np.array([0.9, -0.7, 0.5], dtype=np.float32),
                        qd=np.array([0.5, -0.4, 0.3], dtype=np.float32),
                        passive_damping=2.0,
                        steps=20,
                    )
