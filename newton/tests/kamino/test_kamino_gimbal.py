# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Kamino intrinsic-Euler D6 joints."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
from newton import Contacts
from newton._src.sim import JointType
from newton._src.solvers.kamino._src.core.joints import JointDoFType
from newton._src.solvers.kamino._src.kinematics.joints import (
    gimbal_reciprocal_axes,
    gimbal_transported_axes,
    map_gimbal_angular_velocity_to_rates,
    select_gimbal_coords,
)
from newton.solvers import SolverKamino, SolverMuJoCo
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.extract import extract_cts_jacobians, extract_dofs_jacobians

_RH_AXES = (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
_LH_AXES = (newton.Axis.X, newton.Axis.Z, newton.Axis.Y)


@wp.kernel
def _evaluate_gimbal_chart(
    coords: wp.array[wp.vec3f],
    reference: wp.array[wp.vec3f],
    omega: wp.array[wp.vec3f],
    effort: wp.array[wp.vec3f],
    third_axis_sign: wp.float32,
    selected: wp.array[wp.vec3f],
    basis_product: wp.array[wp.mat33f],
    power: wp.array[wp.vec2f],
):
    """Evaluate chart selection and reciprocal-basis identities."""
    q = coords[0]
    axes = gimbal_transported_axes(q, third_axis_sign)
    rotation = (
        wp.quat_from_axis_angle(wp.vec3f(axes[:, 2]), q[2])
        * wp.quat_from_axis_angle(wp.vec3f(axes[:, 1]), q[1])
        * wp.quat_from_axis_angle(wp.vec3f(axes[:, 0]), q[0])
    )
    selected[0] = select_gimbal_coords(rotation, reference[0], third_axis_sign)
    reciprocal = gimbal_reciprocal_axes(selected[0], third_axis_sign)
    basis_product[0] = wp.transpose(reciprocal) @ gimbal_transported_axes(selected[0], third_axis_sign)
    rates = map_gimbal_angular_velocity_to_rates(selected[0], omega[0], third_axis_sign)
    power[0] = wp.vec2f(wp.dot(effort[0], rates), wp.dot(reciprocal @ effort[0], omega[0]))


def _build_rotational_d6(
    axes: tuple[newton.Axis, newton.Axis, newton.Axis],
    device: wp.DeviceLike,
    *,
    target_ke: float = 0.0,
    armature: float = 0.0,
):
    """Build a minimal articulated three-axis D6 fixture."""
    builder = newton.ModelBuilder()
    parent = builder.add_link(mass=1.0, inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    child = builder.add_link(mass=1.0, inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    root = builder.add_joint_fixed(-1, parent)
    d6 = builder.add_joint_d6(
        parent,
        child,
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(axis=axis, target_ke=target_ke, armature=armature) for axis in axes
        ],
    )
    builder.add_articulation([root, d6])
    return builder.finalize(device=device), d6


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
    device: wp.DeviceLike,
    *,
    stiffness: float = 0.0,
    drive_damping: float = 0.0,
    armature: float = 0.0,
    passive_damping: float = 0.0,
    lower: float | np.ndarray = -newton.MAXVAL,
    upper: float | np.ndarray = newton.MAXVAL,
) -> _Fixture:
    """Build a collision-free articulated rotational D6."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    base = builder.add_link(mass=2.0, inertia=wp.mat33(0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0), label="base")
    link = builder.add_link(mass=1.0, inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4), label="link")
    root = (
        builder.add_joint_fixed(parent=-1, child=base, label="root")
        if fixed_base
        else builder.add_joint_free(parent=-1, child=base, label="root")
    )
    lower_values = np.broadcast_to(lower, 3)
    upper_values = np.broadcast_to(upper, 3)
    configs = [
        newton.ModelBuilder.JointDofConfig(
            axis=axis,
            target_pos=0.0,
            target_vel=0.0,
            target_ke=stiffness,
            target_kd=drive_damping,
            damping=passive_damping,
            armature=armature,
            limit_lower=float(lower_values[i]),
            limit_upper=float(upper_values[i]),
            limit_ke=1.0e4,
            limit_kd=100.0,
        )
        for i, axis in enumerate(axes)
    ]
    d6 = builder.add_joint_d6(base, link, angular_axes=configs, label="d6")
    builder.add_articulation([root, d6], label="d6")
    model = builder.finalize(device=device)
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
    device: wp.DeviceLike,
    *,
    q: np.ndarray | None = None,
    qd: np.ndarray | None = None,
    effort: np.ndarray | None = None,
    position_target: np.ndarray | None = None,
    velocity_target: np.ndarray | None = None,
    steps: int = 1,
    record_trajectory: bool = True,
    **fixture_kwargs,
) -> _Probe:
    """Run a D6 rollout and optionally retain each coordinate sample."""
    fixture = _build_fixture(fixed_base, axes, device, **fixture_kwargs)
    model = fixture.model
    state_in, state_out, control = model.state(), model.state(), model.control()
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
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    solver = _make_solver(backend, model)
    contacts = Contacts(rigid_contact_max=0, soft_contact_max=0, device=model.device) if backend == "mjwarp" else None
    q_slice = slice(fixture.q_start, fixture.q_start + 3)
    qd_slice = slice(fixture.qd_start, fixture.qd_start + 3)
    positions = [state_in.joint_q.numpy()[q_slice].copy()] if record_trajectory else []
    velocities = [state_in.joint_qd.numpy()[qd_slice].copy()] if record_trajectory else []
    for _ in range(steps):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        state_in, state_out = state_out, state_in
        if record_trajectory:
            positions.append(state_in.joint_q.numpy()[q_slice].copy())
            velocities.append(state_in.joint_qd.numpy()[qd_slice].copy())
    if not record_trajectory:
        positions.append(state_in.joint_q.numpy()[q_slice].copy())
        velocities.append(state_in.joint_qd.numpy()[qd_slice].copy())
    return _Probe(
        np.stack(positions),
        np.stack(velocities),
        control.joint_f.numpy()[fixture.qd_start : fixture.qd_start + 3].copy(),
        model.joint_coord_count,
        model.joint_dof_count,
    )


class TestGimbal(unittest.TestCase):
    """Verify the rotational D6 representation."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def test_rotational_d6_converts_to_gimbal(self):
        """Convert a three-angular-axis D6 joint to GIMBAL."""
        model, d6 = _build_rotational_d6(_RH_AXES, self.default_device)
        solver = SolverKamino(model)
        self.assertEqual(solver._model_kamino.joints.dof_type.numpy()[d6], JointDoFType.GIMBAL)

    def test_left_handed_rotational_d6_converts_to_gimbal_left_handed(self):
        """Classify an X-Z-Y D6 joint as left-handed."""
        model, d6 = _build_rotational_d6(_LH_AXES, self.default_device)
        solver = SolverKamino(model)
        self.assertEqual(solver._model_kamino.joints.dof_type.numpy()[d6], JointDoFType.GIMBAL_LEFT_HANDED)

    def test_gimbal_rejects_nonorthogonal_axes(self):
        """Reject a gimbal whose axes are not an orthonormal basis."""
        model, d6 = _build_rotational_d6(_RH_AXES, self.default_device)
        assert model.joint_qd_start is not None
        assert model.joint_axis is not None
        qd_start = model.joint_qd_start.numpy()[d6]
        axes = model.joint_axis.numpy()
        axes[qd_start + 1] = [1.0, 0.0, 0.0]
        model.joint_axis.assign(axes)

        with self.assertRaisesRegex(ValueError, "gimbal axes must be unit length and orthogonal"):
            SolverKamino(model)

    def test_universal_rejects_nonorthogonal_axes(self):
        """Reject a universal joint whose axes are not perpendicular."""
        builder = newton.ModelBuilder()
        parent = builder.add_link(mass=1.0, inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        child = builder.add_link(mass=1.0, inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        root = builder.add_joint_fixed(-1, parent)
        universal = builder.add_joint_d6(
            parent,
            child,
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y),
            ],
        )
        builder.add_articulation([root, universal])
        model = builder.finalize(device=self.default_device)
        assert model.joint_qd_start is not None
        assert model.joint_axis is not None
        qd_start = model.joint_qd_start.numpy()[universal]
        axes = model.joint_axis.numpy()
        axes[qd_start + 1] = [1.0, 0.0, 0.0]
        model.joint_axis.assign(axes)

        with self.assertRaisesRegex(ValueError, "universal and gimbal axes must be unit length and orthogonal"):
            SolverKamino(model)

    def test_from_newton_derives_gimbal_handedness(self):
        """Derive the gimbal type from the orientation of its three axes."""
        limits = np.zeros(3, dtype=np.float32)
        right_handed = JointDoFType.from_newton(JointType.D6, 3, 3, (0, 3), limits, limits, np.eye(3, dtype=np.float32))
        left_handed = JointDoFType.from_newton(
            JointType.D6,
            3,
            3,
            (0, 3),
            limits,
            limits,
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(right_handed, JointDoFType.GIMBAL)
        self.assertEqual(left_handed, JointDoFType.GIMBAL_LEFT_HANDED)

    def test_chart_selects_nearest_equivalent_branch(self):
        """Select the equivalent Euler branch nearest the authored reference."""
        for third_axis_sign in (1.0, -1.0):
            coords = np.array([[0.4, 0.7, third_axis_sign * -0.5]], dtype=np.float32)
            alternative = np.array(
                [[coords[0, 0] + np.pi, np.pi - coords[0, 1], coords[0, 2] + third_axis_sign * np.pi]],
                dtype=np.float32,
            )
            for reference, expected in ((coords, coords), (alternative, alternative)):
                selected = wp.empty(1, dtype=wp.vec3f, device="cpu")
                product = wp.empty(1, dtype=wp.mat33f, device="cpu")
                power = wp.empty(1, dtype=wp.vec2f, device="cpu")
                wp.launch(
                    _evaluate_gimbal_chart,
                    dim=1,
                    inputs=[
                        wp.array(coords, dtype=wp.vec3f, device="cpu"),
                        wp.array(reference, dtype=wp.vec3f, device="cpu"),
                        wp.array([[0.2, -0.4, 0.3]], dtype=wp.vec3f, device="cpu"),
                        wp.array([[0.7, -0.6, 0.5]], dtype=wp.vec3f, device="cpu"),
                        third_axis_sign,
                    ],
                    outputs=[selected, product, power],
                    device="cpu",
                )
                np.testing.assert_allclose(selected.numpy(), expected, atol=1.0e-5)

    def test_reciprocal_basis_preserves_power(self):
        """Preserve dual-basis identity and generalized power away from singularity."""
        selected = wp.empty(1, dtype=wp.vec3f, device="cpu")
        product = wp.empty(1, dtype=wp.mat33f, device="cpu")
        power = wp.empty(1, dtype=wp.vec2f, device="cpu")
        coords = np.array([[0.8, np.pi / 2.0 - 0.02, 0.6]], dtype=np.float32)
        wp.launch(
            _evaluate_gimbal_chart,
            dim=1,
            inputs=[
                wp.array(coords, dtype=wp.vec3f, device="cpu"),
                wp.array(coords, dtype=wp.vec3f, device="cpu"),
                wp.array([[0.2, -0.4, 0.3]], dtype=wp.vec3f, device="cpu"),
                wp.array([[0.7, -0.6, 0.5]], dtype=wp.vec3f, device="cpu"),
                1.0,
            ],
            outputs=[selected, product, power],
            device="cpu",
        )
        np.testing.assert_allclose(product.numpy()[0], np.eye(3), atol=1.0e-5)
        np.testing.assert_allclose(power.numpy()[0, 0], power.numpy()[0, 1], atol=1.0e-5)

    def test_built_gimbal_jacobians_map_body_twists_to_rates(self):
        """Map body twists to authored rates through dense and sparse gimbal Jacobians."""
        q_expected = np.array([0.9, -0.7, 0.5], dtype=np.float32)
        qd_expected = np.array([0.4, -0.3, 0.2], dtype=np.float32)
        for axes in (_RH_AXES, _LH_AXES):
            for sparse_jacobian in (False, True):
                with self.subTest(axes=axes, sparse_jacobian=sparse_jacobian):
                    model, d6 = _build_rotational_d6(axes, self.default_device, armature=0.5)
                    solver = SolverKamino(
                        model,
                        SolverKamino.Config(use_collision_detector=False, sparse_jacobian=sparse_jacobian),
                    )
                    state = model.state()
                    q_start = model.joint_q_start.numpy()[d6]
                    qd_start = model.joint_qd_start.numpy()[d6]
                    joint_q = state.joint_q.numpy()
                    joint_qd = state.joint_qd.numpy()
                    joint_q[q_start : q_start + 3] = q_expected
                    joint_qd[qd_start : qd_start + 3] = qd_expected
                    state.joint_q.assign(joint_q)
                    state.joint_qd.assign(joint_qd)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                    model_kamino = solver._model_kamino
                    solver_kamino = solver._solver_kamino
                    data = solver_kamino._data
                    data.bodies.q_i.assign(state.body_q)
                    data.bodies.u_i.assign(state.body_qd)
                    solver_kamino._update_joints_data(q_j_p=state.joint_q)
                    solver_kamino._update_jacobians()
                    jacobians = solver_kamino._jacobians
                    self.assertEqual(model_kamino.joints.num_dynamic_cts.numpy()[d6], 3)
                    j_dofs = extract_dofs_jacobians(model_kamino, jacobians)[0]
                    j_cts = extract_cts_jacobians(model_kamino, None, None, jacobians)[0]
                    body_twist = data.bodies.u_i.numpy().reshape(-1)

                    np.testing.assert_allclose(j_dofs @ body_twist, qd_expected, atol=1.0e-5)
                    np.testing.assert_allclose(j_cts[:3] @ body_twist, qd_expected, atol=1.0e-5)
                    np.testing.assert_allclose(j_cts[:3], j_dofs, atol=1.0e-6)

    def test_gimbal_metadata(self):
        """Expose three coordinates, rates, and translational constraints."""
        self.assertEqual(JointDoFType.GIMBAL.num_coords, 3)
        self.assertEqual(JointDoFType.GIMBAL.num_dofs, 3)
        self.assertEqual(JointDoFType.GIMBAL.num_cts, 3)
        self.assertTrue(JointDoFType.GIMBAL.is_pure_three_dof_rotation)
        self.assertEqual(JointDoFType.GIMBAL_LEFT_HANDED.num_coords, 3)
        self.assertEqual(JointDoFType.GIMBAL_LEFT_HANDED.num_dofs, 3)
        self.assertEqual(JointDoFType.GIMBAL_LEFT_HANDED.num_cts, 3)
        self.assertTrue(JointDoFType.GIMBAL_LEFT_HANDED.is_pure_three_dof_rotation)

    def test_fk_reset_preserves_left_handed_coordinates_and_rates(self):
        """Reset an FK-enabled solver with authored left-handed D6 state."""
        model, d6 = _build_rotational_d6(_LH_AXES, self.default_device, target_ke=1.0)
        solver = SolverKamino(model, SolverKamino.Config(use_fk_solver=True))
        state = model.state()
        q_start = model.joint_q_start.numpy()[d6]
        qd_start = model.joint_qd_start.numpy()[d6]
        q = state.joint_q.numpy()
        qd = state.joint_qd.numpy()
        q[q_start : q_start + 3] = [0.2, -0.3, 0.4]
        qd[qd_start : qd_start + 3] = [-0.1, 0.15, -0.2]
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        solver.reset(
            state,
            config=SolverKamino.ResetConfig(
                body_poses=SolverKamino.ResetConfig.FromJointQ(state.joint_q),
                body_velocities=SolverKamino.ResetConfig.FromJointU(state.joint_qd),
            ),
        )
        np.testing.assert_allclose(state.joint_q.numpy()[q_start : q_start + 3], q[q_start : q_start + 3], atol=1.0e-5)
        np.testing.assert_allclose(
            state.joint_qd.numpy()[qd_start : qd_start + 3], qd[qd_start : qd_start + 3], atol=1.0e-5
        )

    def test_limits(self):
        """Drive each D6 coordinate to its own position limit."""
        lower = np.array([-0.15, -0.25, -0.35], dtype=np.float32)
        upper = np.array([0.2, 0.3, 0.4], dtype=np.float32)
        target = np.array([0.6, -0.6, 0.6], dtype=np.float32)
        expected = np.where(target > 0.0, upper, lower)
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base):
                    probe = _run(
                        "kamino",
                        fixed_base,
                        axes,
                        self.default_device,
                        position_target=target,
                        stiffness=100.0,
                        drive_damping=15.0,
                        lower=lower,
                        upper=upper,
                        steps=50,
                        record_trajectory=False,
                    )
                    np.testing.assert_allclose(probe.q[-1], expected, atol=1.0e-2, rtol=0.0)


@unittest.skipUnless(wp.get_cuda_device_count(), "requires CUDA device")
class TestGimbalMJWarp(unittest.TestCase):
    """Compare Kamino and MJWarp through the public Newton state layout."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _assert_pair(
        self, fixed_base: bool, axes, *, scenario: str, rtol: float = 1.0e-2, atol: float = 1.0e-2, **kwargs
    ):
        """Run and compare both solvers for one D6 scenario."""
        mjwarp = _run("mjwarp", fixed_base, axes, self.default_device, **kwargs)
        kamino = _run("kamino", fixed_base, axes, self.default_device, **kwargs)
        np.testing.assert_allclose(mjwarp.q, kamino.q, rtol=rtol, atol=atol, err_msg=f"{scenario}: q")
        np.testing.assert_allclose(mjwarp.qd, kamino.qd, rtol=rtol, atol=atol, err_msg=f"{scenario}: qd")
        return mjwarp, kamino

    def test_layout_and_state_writers(self):
        """Match D6 layout and independent coordinate/rate writes."""
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base, scenario="layout"):
                    mjwarp = _run("mjwarp", fixed_base, axes, self.default_device, steps=0)
                    kamino = _run("kamino", fixed_base, axes, self.default_device, steps=0)
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

    def test_implicit_pd_trajectories(self):
        """Match implicit PD rollouts."""
        kwargs = {
            "position_target": np.array([0.12, -0.08, 0.05], dtype=np.float32),
            "stiffness": 80.0,
            "drive_damping": 12.0,
            "steps": 20,
        }
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base):
                    self._assert_pair(
                        fixed_base,
                        axes,
                        scenario="implicit-pd",
                        # MJWarp implicitfast omits Kamino's dt**2 * kp effective-inertia term.
                        # This intentional discretization difference requires a higher tolerance.
                        rtol=6.0e-2,
                        atol=5.0e-3,
                        **kwargs,
                    )

    def test_unwrapped_pd_targets(self):
        """Match one-step PD motion toward unwrapped targets beyond 2*pi."""
        kwargs = {
            "q": np.array([0.2, -0.5, 0.3], dtype=np.float32),
            "position_target": np.array([2.0 * np.pi + 0.4, -2.0 * np.pi - 0.3, 2.0 * np.pi + 0.2], dtype=np.float32),
            "stiffness": 5.0,
            "drive_damping": 12.0,
            "steps": 1,
        }
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                with self.subTest(axes=axes, fixed_base=fixed_base):
                    self._assert_pair(
                        fixed_base,
                        axes,
                        scenario="unwrapped-pd-target",
                        rtol=2.5e-2,
                        atol=5.0e-3,
                        **kwargs,
                    )

    def test_passive_damping(self):
        """Match passive-damping D6 behavior."""
        for axes in (_RH_AXES, _LH_AXES):
            for fixed_base in (True, False):
                for axis in range(3):
                    velocity = np.zeros(3, dtype=np.float32)
                    velocity[axis] = 0.5
                    with self.subTest(axes=axes, fixed_base=fixed_base, axis=axis):
                        baseline = _run("kamino", fixed_base, axes, self.default_device, qd=velocity)
                        damped = _run("kamino", fixed_base, axes, self.default_device, qd=velocity, passive_damping=2.0)
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


if __name__ == "__main__":
    setup_tests()
    unittest.main()
