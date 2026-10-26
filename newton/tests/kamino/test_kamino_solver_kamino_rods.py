# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the rod joints of the Kamino dynamics backends."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context

_TIME_STEP = 0.01
_ROD_BACKENDS = ("lox", "dvi")


def _build_rod_model(
    *,
    binary: bool = False,
    explicit_world: bool = True,
    stretch_stiffness: float = 0.0,
    stretch_damping: float = 0.0,
    shear_stiffness: float = 0.0,
    shear_damping: float = 0.0,
    bend_stiffness: float = 0.0,
    bend_damping: float = 0.0,
    twist_stiffness: float = 0.0,
    twist_damping: float = 0.0,
    child_xform: wp.transformf | None = None,
    device: wp.DeviceLike = None,
) -> tuple[newton.Model, int, int]:
    """Build a gravity-free world or binary rod with coincident rest anchors."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    if explicit_world:
        builder.begin_world()
    inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    parent = -1
    if binary:
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    if child_xform is None:
        child_xform = wp.transform_identity(dtype=wp.float32)
    child = builder.add_link(xform=child_xform, mass=1.0, inertia=inertia, lock_inertia=True)
    joint = builder.add_joint_rod(
        parent,
        child,
        stretch_stiffness=stretch_stiffness,
        stretch_damping=stretch_damping,
        shear_stiffness=shear_stiffness,
        shear_damping=shear_damping,
        bend_stiffness=bend_stiffness,
        bend_damping=bend_damping,
        twist_stiffness=twist_stiffness,
        twist_damping=twist_damping,
    )
    builder.add_articulation([joint])
    if explicit_world:
        builder.end_world()
    return builder.finalize(device=device), parent, child


def _principal_angle(angle: float) -> float:
    """Return an angle in the principal branch ``(-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _linearly_implicit_twist_reference(
    *,
    initial_strain: float,
    initial_speed: float,
    stiffness: float,
    damping: float,
    time_step: float,
) -> float:
    """Return the unit-inertia twist speed of one step with the elastic strain linearized at its start."""
    elastic = stiffness * _principal_angle(initial_strain)
    return (initial_speed - time_step * elastic) / (1.0 + time_step * damping + time_step * time_step * stiffness)


def _make_config(backend: str) -> SolverKamino.Config:
    config = SolverKamino.Config(dynamics_solver=backend, use_collision_detector=False)
    if backend == "lox":
        config.lox.max_iterations = 80
    return config


def _build_rod_cases(cases: list[dict[str, float | str]], device: wp.DeviceLike) -> tuple[newton.Model, list[int]]:
    """Build one world-attached unit-inertia rod per case, each in its own world."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    bodies = []
    inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    for case in cases:
        builder.begin_world()
        bend = case.get("mode", "twist") == "bend"
        axis = wp.vec3f(1.0, 0.0, 0.0) if bend else wp.vec3f(0.0, 0.0, 1.0)
        body = builder.add_link(
            xform=wp.transformf(wp.vec3f(0.0), wp.quat_from_axis_angle(axis, float(case.get("rest_angle", 0.0)))),
            mass=1.0,
            inertia=inertia,
            lock_inertia=True,
        )
        stiffness = float(case.get("stiffness", 0.0))
        damping = float(case.get("damping", 0.0))
        joint = builder.add_joint_rod(
            -1,
            body,
            bend_stiffness=stiffness if bend else 0.0,
            bend_damping=damping if bend else 0.0,
            twist_stiffness=0.0 if bend else stiffness,
            twist_damping=0.0 if bend else damping,
        )
        builder.add_articulation([joint])
        builder.end_world()
        bodies.append(body)
    return builder.finalize(device=device), bodies


def _initial_rod_cases_state(
    model: newton.Model, bodies: list[int], cases: list[dict[str, float | str]]
) -> newton.State:
    state = model.state()
    poses = state.body_q.numpy()
    velocities = np.zeros((model.body_count, 6), dtype=np.float32)
    for body, case in zip(bodies, cases, strict=True):
        bend = case.get("mode", "twist") == "bend"
        axis = wp.vec3f(1.0, 0.0, 0.0) if bend else wp.vec3f(0.0, 0.0, 1.0)
        angle = float(case.get("rest_angle", 0.0)) + float(case.get("initial_strain", 0.0))
        poses[body, 3:] = np.asarray(wp.quat_from_axis_angle(axis, angle))
        velocities[body, 3 if bend else 5] = float(case.get("initial_speed", 0.0))
    state.body_q.assign(poses)
    state.body_qd.assign(velocities)
    return state


def _run_rod_cases(cases: list[dict[str, float | str]], device: wp.DeviceLike, backend: str) -> np.ndarray:
    """Run one step of the rod cases and return the body velocities."""
    model, bodies = _build_rod_cases(cases, device)
    state_previous = _initial_rod_cases_state(model, bodies, cases)
    state_next = model.state()
    solver = SolverKamino(model, config=_make_config(backend))
    solver.step(state_previous, state_next, model.control(), contacts=None, dt=_TIME_STEP)
    return state_next.body_qd.numpy()


class TestSolverKaminoRods(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _step_rod(
        self, model: newton.Model, backend: str, body_q=None, body_qd=None, control: newton.Control | None = None
    ) -> np.ndarray:
        solver = SolverKamino(model, config=_make_config(backend))
        state_previous = model.state()
        state_next = model.state()
        if body_q is not None:
            state_previous.body_q.assign(body_q)
        if body_qd is not None:
            state_previous.body_qd.assign(body_qd)
        control = model.control() if control is None else control
        solver.step(state_previous, state_next, control, contacts=None, dt=_TIME_STEP)
        return state_next.body_qd.numpy()

    def test_rod_world_parent_stretch_and_damping(self):
        """Restore stretch and reduce the axial speed through rod damping."""
        for backend in _ROD_BACKENDS:
            with self.subTest(backend=backend):
                model, _parent, child = _build_rod_model(stretch_stiffness=100.0, device=self.default_device)
                velocity = self._step_rod(
                    model, backend, body_q=[wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32))]
                )
                self.assertLess(float(velocity[child, 2]), 0.0)

                speeds = []
                for damping in (0.0, 10.0):
                    model, _parent, child = _build_rod_model(stretch_damping=damping, device=self.default_device)
                    velocity = self._step_rod(
                        model, backend, body_qd=[wp.spatial_vectorf(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)]
                    )
                    speeds.append(float(velocity[child, 2]))
                self.assertLess(speeds[1], speeds[0])

    def test_rod_accepts_implicit_single_world(self):
        """Accept a rod without an explicit world in a single-world model."""
        for backend in _ROD_BACKENDS:
            with self.subTest(backend=backend):
                model, _parent, child = _build_rod_model(
                    explicit_world=False, stretch_stiffness=100.0, device=self.default_device
                )
                velocity = self._step_rod(
                    model, backend, body_q=[wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32))]
                )
                self.assertLess(float(velocity[child, 2]), 0.0)

    def test_rod_bend_and_twist_restore_rotation(self):
        """Restore isolated bend and twist rotations, and keep a pre-curved rest pose at rest."""
        cases = (
            ("bend", wp.vec3f(1.0, 0.0, 0.0), {"bend_stiffness": 20.0}, 3),
            ("twist", wp.vec3f(0.0, 0.0, 1.0), {"twist_stiffness": 20.0}, 5),
        )
        for backend in _ROD_BACKENDS:
            for name, axis, coefficients, velocity_index in cases:
                with self.subTest(backend=backend, mode=name):
                    model, _parent, child = _build_rod_model(device=self.default_device, **coefficients)
                    velocity = self._step_rod(
                        model, backend, body_q=[wp.transformf(wp.vec3f(0.0), wp.quat_from_axis_angle(axis, 0.1))]
                    )
                    self.assertLess(float(velocity[child, velocity_index]), 0.0)
                    self.assertTrue(np.isfinite(velocity).all())

            with self.subTest(backend=backend, mode="pre-curved rest"):
                model, _parent, child = _build_rod_model(
                    bend_stiffness=20.0,
                    twist_stiffness=10.0,
                    child_xform=wp.transformf(wp.vec3f(0.0), wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), 0.2)),
                    device=self.default_device,
                )
                velocity = self._step_rod(model, backend)
                np.testing.assert_allclose(velocity[child], 0.0, atol=1.0e-6)

    def test_rod_rest_follows_control_target(self):
        """Twist the rod towards the rest rotation of the control joint target."""
        for backend in _ROD_BACKENDS:
            with self.subTest(backend=backend):
                model, _parent, child = _build_rod_model(twist_stiffness=20.0, device=self.default_device)
                control = model.control()
                target = control.joint_target_q.numpy()
                target[3:7] = np.asarray(wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), 0.1))
                control.joint_target_q.assign(target)
                velocity = self._step_rod(model, backend, control=control)
                self.assertGreater(float(velocity[child, 5]), 0.0)

    def test_binary_rod_balances_wrenches(self):
        """Balance the wrenches of a binary rod."""
        for backend in _ROD_BACKENDS:
            with self.subTest(backend=backend):
                model, parent, child = _build_rod_model(binary=True, shear_stiffness=50.0, device=self.default_device)
                velocity = self._step_rod(
                    model,
                    backend,
                    body_q=[
                        wp.transform_identity(dtype=wp.float32),
                        wp.transformf(wp.vec3f(0.1, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
                    ],
                )
                self.assertGreater(float(velocity[parent, 0]), 0.0)
                self.assertLess(float(velocity[child, 0]), 0.0)
                np.testing.assert_allclose(velocity[parent, :3] + velocity[child, :3], 0.0, atol=1.0e-6)

    def test_rod_rejects_unsupported_configurations(self):
        """Reject rod joints with the PADMM backend and with joint armature."""
        model, _parent, _child = _build_rod_model(stretch_stiffness=100.0, device=self.default_device)
        with self.assertRaisesRegex(ValueError, "ROD"):
            SolverKamino(model, config=_make_config("padmm"))

        armature = model.joint_armature.numpy()
        armature[:] = 0.1
        model.joint_armature.assign(armature)
        with self.assertRaisesRegex(ValueError, "joint_armature"):
            SolverKamino(model, config=_make_config("lox"))

    def test_twist_damping_crosses_both_principal_branches(self):
        """Damp twist velocities that cross the principal angle branch in both directions."""
        cases = [
            {"initial_strain": 3.1, "initial_speed": 10.0, "damping": 100.0},
            {"initial_strain": -3.1, "initial_speed": -10.0, "damping": 100.0},
            {"initial_strain": 0.5, "initial_speed": 10.0, "damping": 100.0},
        ]
        for backend in _ROD_BACKENDS:
            with self.subTest(backend=backend):
                velocity = _run_rod_cases(cases, self.default_device, backend)
                np.testing.assert_allclose(velocity[:, 5], [5.0, -5.0, 5.0], rtol=0.0, atol=1.0e-4)

    def test_mixed_twist_matches_linearly_implicit_reference(self):
        """Match a linearly implicit elastic and damped twist step, with rest twists across the angle branch."""
        cases = [
            {"rest_angle": -0.35, "initial_strain": strain, "initial_speed": speed, "stiffness": 37.0, "damping": 13.0}
            for strain, speed in ((0.8, 7.0), (3.1, 7.0), (-3.1, -7.0))
        ]
        expected = [
            _linearly_implicit_twist_reference(
                initial_strain=float(case["initial_strain"]),
                initial_speed=float(case["initial_speed"]),
                stiffness=float(case["stiffness"]),
                damping=float(case["damping"]),
                time_step=_TIME_STEP,
            )
            for case in cases
        ]
        for backend in _ROD_BACKENDS:
            with self.subTest(backend=backend):
                velocity = _run_rod_cases(cases, self.default_device, backend)
                np.testing.assert_allclose(velocity[:, 5], expected, rtol=0.0, atol=2.0e-4)

    def test_zero_twist_coefficients_leave_velocity_unchanged(self):
        """Leave the twist velocity unchanged without twist stiffness and damping."""
        for backend in _ROD_BACKENDS:
            with self.subTest(backend=backend):
                velocity = _run_rod_cases(
                    [{"initial_strain": 3.1, "initial_speed": 10.0}], self.default_device, backend
                )
                np.testing.assert_allclose(velocity[0], [0.0, 0.0, 0.0, 0.0, 0.0, 10.0], rtol=0.0, atol=1.0e-6)

    def test_binary_rod_keeps_common_rotation(self):
        """Keep a large common rotation of a damped binary rod, which leaves its strains unchanged."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        child = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        joint = builder.add_joint_rod(parent, child, bend_damping=100.0, twist_damping=100.0)
        builder.add_articulation([joint])
        builder.end_world()
        model = builder.finalize(device=self.default_device)
        for backend in _ROD_BACKENDS:
            with self.subTest(backend=backend):
                velocity = self._step_rod(
                    model, backend, body_qd=np.tile(np.asarray([0.0, 0.0, 0.0, 400.0, 0.0, 0.0], np.float32), (2, 1))
                )
                np.testing.assert_allclose(velocity[:, 3], 400.0, rtol=1.0e-4)

    def test_finite_near_fold_and_large_rod_updates(self):
        """Keep finite velocities for near-fold bends and large angular updates."""
        cases = [
            {"mode": "bend", "initial_strain": 3.1, "initial_speed": 10.0, "damping": 10000.0},
            {"mode": "bend", "initial_strain": 0.5, "initial_speed": 600.0, "damping": 100.0},
            {"mode": "twist", "initial_strain": 0.5, "initial_speed": 600.0, "damping": 1.0},
        ]
        for backend in _ROD_BACKENDS:
            with self.subTest(backend=backend):
                velocity = _run_rod_cases(cases, self.default_device, backend)
                self.assertTrue(np.isfinite(velocity).all())

    def test_cuda_graph_replays_twist_branch_crossing(self):
        """Replay a captured rod step across a twist branch crossing."""
        if not self.default_device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        cases = [{"initial_strain": 3.1, "initial_speed": 10.0, "damping": 100.0}]
        for backend in _ROD_BACKENDS:
            with self.subTest(backend=backend):
                model, bodies = _build_rod_cases(cases, self.default_device)
                state_previous = _initial_rod_cases_state(model, bodies, cases)
                state_next = model.state()
                solver = SolverKamino(model, config=_make_config(backend))
                with wp.ScopedCapture() as capture:
                    solver.step(state_previous, state_next, model.control(), contacts=None, dt=_TIME_STEP)
                wp.capture_launch(capture.graph)
                self.assertAlmostEqual(float(state_next.body_qd.numpy()[0, 5]), 5.0, delta=1.0e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
