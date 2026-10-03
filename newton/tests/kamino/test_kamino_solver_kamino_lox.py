# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the LOX Kamino dynamics backend."""

import math
import unittest

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.detector import CollisionDetector
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.utils.basics import (
    build_box_on_plane,
    build_cartpole,
)


def _build_revolute_dynamics_model(
    *,
    damping: float,
    friction: float,
    velocity: float,
    armature: float = 0.0,
    target_ke: float = 0.0,
    target_kd: float = 0.0,
    effort_limit: float = math.inf,
    actuator_mode: newton.JointTargetMode | None = None,
    device: wp.DeviceLike = None,
) -> newton.Model:
    """Build a gravity-free world-to-body hinge with consistent initial velocity."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Y,
        damping=damping,
        friction=friction,
        armature=armature,
        target_ke=target_ke,
        target_kd=target_kd,
        effort_limit=effort_limit,
        actuator_mode=actuator_mode,
    )
    builder.add_articulation([joint])
    builder.body_qd[body] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, velocity, 0.0)
    builder.joint_qd[builder.joint_qd_start[joint]] = velocity
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_partially_dynamic_d6_model(*, binary: bool, device: wp.DeviceLike = None) -> newton.Model:
    """Build a two-axis rotational joint with armature on its first axis only.

    The joint anchor is the origin of the world, and the spinning child link hangs 0.5 m below it.
    With ``binary``, the joint's parent is a link on a revolute joint about the same anchor.
    """
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    inertia = wp.mat33f(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)
    joints = []
    parent = -1
    if binary:
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        joints.append(builder.add_joint_revolute(parent=-1, child=parent, axis=newton.Axis.Z, armature=0.1))
    link = builder.add_link(
        xform=wp.transformf(wp.vec3f(0.0, 0.0, -0.5), wp.quat_identity(dtype=wp.float32)),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    axis = newton.ModelBuilder.JointDofConfig
    joints.append(
        builder.add_joint_d6(
            parent=parent,
            child=link,
            angular_axes=[axis(axis=newton.Axis.X, armature=0.1), axis(axis=newton.Axis.Y, armature=0.0)],
            child_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.5), wp.quat_identity(dtype=wp.float32)),
        )
    )
    builder.add_articulation(joints)
    builder.body_qd[link] = wp.spatial_vectorf(0.0, 0.0, 0.0, 2.0, 1.0, 0.0)
    builder.end_world()
    return builder.finalize(device=device)


def _build_massless_fixed_child_drive_model(
    *,
    include_massless_fixed_child: bool = True,
    device: wp.DeviceLike = None,
) -> newton.Model:
    """Build a driven link with an optional massless fixed child."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    inertia = wp.mat33f(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    driven = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    revolute = builder.add_joint_revolute(
        parent=-1,
        child=driven,
        axis=newton.Axis.Z,
        armature=0.1,
        target_ke=650.0,
        target_kd=100.0,
        effort_limit=math.inf,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    joints = [revolute]
    if include_massless_fixed_child:
        child = builder.add_link(
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.1), wp.quat_identity(dtype=wp.float32)),
            mass=0.0,
            inertia=wp.mat33f(),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_fixed(
                parent=driven,
                child=child,
                parent_xform=wp.transformf(
                    wp.vec3f(0.0, 0.0, 0.1),
                    wp.quat_identity(dtype=wp.float32),
                ),
            )
        )
    builder.add_articulation(joints)
    builder.end_world()
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_driven_link_with_grounded_fixed_base(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int]:
    """Build a driven link whose prescribed base overlaps the ground."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    inertia = wp.mat33f(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    base = builder.add_link(
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.05), wp.quat_identity(dtype=wp.float32)),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_box(
        base,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    driven = builder.add_link(
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.25), wp.quat_identity(dtype=wp.float32)),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    fixed = builder.add_joint_fixed(
        parent=-1,
        child=base,
        parent_xform=wp.transformf(wp.vec3f(0.0, 0.0, 0.05), wp.quat_identity(dtype=wp.float32)),
    )
    revolute = builder.add_joint_revolute(
        parent=base,
        child=driven,
        axis=newton.Axis.Z,
        armature=0.1,
        target_ke=650.0,
        target_kd=100.0,
        effort_limit=math.inf,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([fixed, revolute])
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    builder.end_world()
    model = builder.finalize(device=device)
    model.rigid_contact_max = 16
    return model, driven


def _build_zero_mass_prismatic_anchor_model(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int, int]:
    """Build a zero-mass anchor connected to a dynamic prismatic child."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    anchor = builder.add_link()
    child = builder.add_link()
    builder.add_shape_box(anchor, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    builder.add_shape_box(child)
    fixed = builder.add_joint_fixed(parent=-1, child=anchor)
    prismatic = builder.add_joint_prismatic(parent=anchor, child=child, axis=newton.Axis.Z)
    builder.add_articulation([fixed, prismatic])
    builder.end_world()
    return builder.finalize(device=device), anchor, child


def _build_prescribed_only_model(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int]:
    """Build a world containing only one zero-mass body."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link()
    builder.add_shape_box(body, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    fixed = builder.add_joint_fixed(parent=-1, child=body)
    builder.add_articulation([fixed])
    builder.end_world()
    return builder.finalize(device=device), body


def _build_flagged_kinematic_model(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int]:
    """Build a massive free body whose motion is prescribed by its body flag."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    builder.end_world()
    return builder.finalize(device=device), body


class TestSolverKaminoLOX(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def make_config(self, compute_solution_metrics: bool = False) -> SolverKamino.Config:
        return SolverKamino.Config(
            dynamics_solver="lox",
            compute_solution_metrics=compute_solution_metrics,
            lox=kamino_config.LOXSolverConfig(
                max_iterations=25,
                projection_iterations=5,
            ),
        )

    def test_moreau_free_fall_uses_midpoint_configuration(self):
        """Compose LOX forward dynamics with Kamino's Moreau integrator."""
        model = ModelKamino.from_newton(build_box_on_plane(ground=False).finalize(device=self.default_device))
        config = self.make_config()
        config.integrator = "moreau"
        config.use_collision_detector = True
        solver = SolverKaminoImpl(model=model, config=config)
        state_previous = model.state()
        state_next = model.state()
        time_step = 0.01

        solver.step(state_previous, state_next, model.control(), dt=time_step)

        gravity = model.gravity.vector.numpy()[0]
        expected_velocity = time_step * gravity
        np.testing.assert_allclose(state_next.u_i.numpy()[0, :3], expected_velocity, rtol=0.0, atol=5.0e-4)
        expected_height = state_previous.q_i.numpy()[0, 2] + 0.5 * time_step * expected_velocity[2]
        self.assertAlmostEqual(float(state_next.q_i.numpy()[0, 2]), float(expected_height), places=6)

    def test_free_fall_uses_each_world_time_step(self):
        """Advance each rigid world with its configured device-side time step."""
        builder = build_box_on_plane(ground=False)
        build_box_on_plane(builder=builder, ground=False)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        time_step = np.asarray([0.01, 0.025], dtype=np.float32)
        model.time.dt.assign(time_step)
        model.time.inv_dt.assign(1.0 / time_step)
        # Seeding the joint penalty with a uniform time step keeps the time step of each world
        solver.solver_fd.joint_penalty_scale_seed(0.01)

        solver.step(state_previous, state_next, model.control(), dt=None)

        gravity = model.gravity.vector.numpy()[: model.size.num_worlds]
        body_world = model.bodies.wid.numpy()
        expected_velocity = time_step[body_world, None] * gravity[body_world]
        np.testing.assert_allclose(state_next.u_i.numpy()[:, :3], expected_velocity, rtol=0.0, atol=5.0e-4)
        expected_height = state_previous.q_i.numpy()[:, 2] + time_step[body_world] * expected_velocity[:, 2]
        np.testing.assert_allclose(state_next.q_i.numpy()[:, 2], expected_height, rtol=0.0, atol=1.0e-6)

    def test_free_rotation_preserves_angular_momentum_magnitude(self):
        """Keep |I ω| of a torque-free body through the LOX solve and each Kamino integrator."""
        # Principal moments that satisfy the triangle inequality, which the model builder enforces
        inertia = np.diag([2.0, 3.0, 4.0])
        omega = np.array([3.0, -7.0, 11.0])
        for integrator in ("euler", "moreau"):
            with self.subTest(integrator=integrator):
                builder = newton.ModelBuilder()
                SolverKamino.register_custom_attributes(builder)
                body = builder.add_body(mass=1.0, inertia=wp.mat33f(*inertia.flatten()), lock_inertia=True)
                builder.body_qd[body] = wp.spatial_vectorf(0.0, 0.0, 0.0, *omega)
                model = builder.finalize(device=self.default_device)
                model.set_gravity((0.0, 0.0, 0.0))
                config = self.make_config()
                config.integrator = integrator
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.02)

                # The body starts at the identity orientation, so its world inertia is the body inertia
                next_omega = state_next.body_qd.numpy()[0, 3:]
                self.assertAlmostEqual(
                    float(np.linalg.norm(inertia @ next_omega) / np.linalg.norm(inertia @ omega)), 1.0, delta=1.0e-5
                )

    def test_spinning_hinge_keeps_its_axis(self):
        """Integrate the LOX velocity of a body spinning about a non-principal hinge axis along that axis."""
        axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
        speed = 10.0
        for integrator in ("euler", "moreau"):
            with self.subTest(integrator=integrator):
                builder = newton.ModelBuilder()
                SolverKamino.register_custom_attributes(builder)
                body = builder.add_link(
                    mass=1.0, inertia=wp.mat33f(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0), lock_inertia=True
                )
                joint = builder.add_joint_revolute(parent=-1, child=body, axis=wp.vec3f(*axis))
                builder.add_articulation([joint])
                builder.body_qd[body] = wp.spatial_vectorf(0.0, 0.0, 0.0, *(speed * axis))
                builder.joint_qd[builder.joint_qd_start[joint]] = speed
                model = builder.finalize(device=self.default_device)
                model.set_gravity((0.0, 0.0, 0.0))
                config = self.make_config()
                config.integrator = integrator
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.02)

                # The hinge absorbs the gyroscopic torque, which the integrator must apply as LOX did
                omega = state_next.body_qd.numpy()[0, 3:]
                self.assertLess(float(np.linalg.norm(omega - np.dot(omega, axis) * axis)), 1.0e-5 * speed)

    def test_prescribed_bodies_keep_their_velocity(self):
        """Keep the velocity of zero-mass and kinematic bodies, and ignore joints between prescribed bodies."""
        anchor_model, anchor, child = _build_zero_mass_prismatic_anchor_model(device=self.default_device)
        friction_model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=1.0,
            velocity=0.25,
            device=self.default_device,
        )
        effort_model = _build_revolute_dynamics_model(
            damping=0.0,
            friction=0.0,
            velocity=0.25,
            target_ke=100.0,
            effort_limit=1.0,
            actuator_mode=newton.JointTargetMode.POSITION,
            device=self.default_device,
        )
        for model in (friction_model, effort_model):
            model.body_flags.fill_(int(newton.BodyFlags.KINEMATIC))
        cases = {
            "zero_mass_anchor": (anchor_model, anchor),
            "prescribed_only_world": _build_prescribed_only_model(device=self.default_device),
            "kinematic_flag": _build_flagged_kinematic_model(device=self.default_device),
            "kinematic_joint_friction": (friction_model, 0),
            "kinematic_joint_effort": (effort_model, 0),
        }
        for name, (model, body) in cases.items():
            with self.subTest(case=name):
                config = self.make_config()
                config.use_collision_detector = False
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                velocity = state_previous.body_qd.numpy()
                if not velocity[body].any():
                    velocity[body, 0] = 0.25
                    state_previous.body_qd.assign(velocity)

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

                np.testing.assert_array_equal(state_next.body_qd.numpy()[body], velocity[body])
                self.assertTrue(np.isfinite(state_next.body_q.numpy()).all())
                self.assertTrue(np.isfinite(state_next.body_qd.numpy()).all())
                if name == "zero_mass_anchor":
                    # The dynamic child slides with the anchor and falls along the prismatic axis
                    self.assertAlmostEqual(float(state_next.body_qd.numpy()[child, 0]), 0.25, places=4)
                    self.assertLess(float(state_next.body_qd.numpy()[child, 2]), -1.0e-3)

    def test_joint_with_partially_dynamic_dofs_keeps_its_anchor(self):
        """Keep the anchor of a joint whose armature acts on only some of its DOFs."""
        for binary in (False, True):
            with self.subTest(binary=binary):
                model = _build_partially_dynamic_d6_model(binary=binary, device=self.default_device)
                config = self.make_config()
                config.use_collision_detector = False
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                control = model.control()

                for _ in range(50):
                    solver.step(state_previous, state_next, control, contacts=None, dt=1.0 / 240.0)
                    state_previous, state_next = state_next, state_previous

                pose = state_previous.body_q.numpy()[-1]
                anchor = pose[:3] + np.asarray(wp.quat_rotate(wp.quatf(*pose[3:]), wp.vec3f(0.0, 0.0, 0.5)))
                np.testing.assert_allclose(anchor, 0.0, rtol=0.0, atol=1.0e-3)

    def test_kinematic_flag_change_requires_solver_rebuild(self):
        """Reject a runtime immovability change after Kamino has culled constraints."""
        model, _body = _build_flagged_kinematic_model(device=self.default_device)
        solver = SolverKamino(model, config=self.make_config())

        model.body_flags.fill_(int(newton.BodyFlags.DYNAMIC))
        with self.assertRaisesRegex(RuntimeError, "immovability.*recreate SolverKamino"):
            solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)

    def test_joint_damping_is_implicit(self):
        """Apply joint damping implicitly."""
        time_step = 0.1
        damping = 4.0
        initial_velocity = 2.0
        model = _build_revolute_dynamics_model(damping=damping, friction=0.0, velocity=initial_velocity)
        solver = SolverKamino(model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=time_step)

        expected = initial_velocity / (1.0 + time_step * damping)
        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), expected, places=4)

    def test_joint_friction_on_some_dofs(self):
        """Slow only the frictional axis of a two-axis joint by its friction impulse."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        inertia = 0.1
        link = builder.add_link(
            mass=1.0,
            inertia=wp.mat33f(inertia, 0.0, 0.0, 0.0, inertia, 0.0, 0.0, 0.0, inertia),
            lock_inertia=True,
        )
        axis = newton.ModelBuilder.JointDofConfig
        joint = builder.add_joint_d6(
            parent=-1,
            child=link,
            angular_axes=[axis(axis=newton.Axis.X, friction=0.5), axis(axis=newton.Axis.Y, friction=0.0)],
        )
        builder.add_articulation([joint])
        builder.body_qd[link] = wp.spatial_vectorf(0.0, 0.0, 0.0, 1.0, 1.0, 0.0)
        builder.joint_qd[builder.joint_qd_start[joint] : builder.joint_qd_start[joint] + 2] = [1.0, 1.0]
        builder.end_world()
        model = builder.finalize(device=self.default_device)
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        state_next = model.state()
        time_step = 0.01

        solver.step(model.state(), state_next, model.control(), contacts=None, dt=time_step)

        expected = [1.0 - 0.5 * time_step / inertia, 1.0]
        np.testing.assert_allclose(state_next.joint_qd.numpy(), expected, rtol=0.0, atol=1.0e-4)

    def test_relaxed_joint_proximal_preserves_nonlinear_fixed_point(self):
        """Converge a relaxed joint correction to the exact candidate-pose constraint."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        inertia = wp.mat33f(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4)
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        child = builder.add_link(
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            mass=1.0,
            inertia=inertia,
            lock_inertia=True,
        )
        root = builder.add_joint_revolute(parent=-1, child=parent, axis=newton.Axis.Y)
        fixed = builder.add_joint_fixed(
            parent=parent,
            child=child,
            parent_xform=wp.transformf(
                wp.vec3f(0.0, 0.0, 1.0),
                wp.quat_identity(dtype=wp.float32),
            ),
        )
        builder.add_articulation([root, fixed])
        builder.end_world()
        model = builder.finalize(device=self.default_device)

        config = self.make_config()
        config.use_collision_detector = False
        config.lox.fixed_iterations = True
        config.lox.max_iterations = 20
        config.lox.joint_proximal_relaxation = 0.5
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        velocity = np.zeros((model.body_count, 6), dtype=np.float32)
        velocity[parent] = (0.0, 0.0, 0.0, 0.0, 5.0, 0.0)
        velocity[child] = (8.0, -5.0, 3.0, 18.0, -12.0, 15.0)
        state_previous.body_qd.assign(velocity)

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.05)

        poses = state_next.body_q.numpy()
        parent_position = poses[parent, :3]
        parent_axis = poses[parent, 3:6]
        parent_scalar = poses[parent, 6]
        joint_offset = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        rotated_offset = joint_offset + 2.0 * (
            parent_scalar * np.cross(parent_axis, joint_offset)
            + np.cross(parent_axis, np.cross(parent_axis, joint_offset))
        )
        position_error = poses[child, :3] - parent_position - rotated_offset
        np.testing.assert_allclose(position_error, 0.0, atol=1.0e-5)

    def test_position_drive_with_massless_fixed_child(self):
        """Track a target when the driven body has a massless fixed child."""
        cases = ((False, "euler"), (True, "euler"), (True, "moreau"))
        for include_massless_fixed_child, integrator in cases:
            with self.subTest(
                include_massless_fixed_child=include_massless_fixed_child,
                integrator=integrator,
            ):
                model = _build_massless_fixed_child_drive_model(
                    include_massless_fixed_child=include_massless_fixed_child,
                    device=self.default_device,
                )
                config = self.make_config()
                config.integrator = integrator
                config.use_collision_detector = integrator == "moreau"
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                control = model.control()
                control.joint_target_q.assign([0.2])

                for _ in range(240):
                    solver.step(state_previous, state_next, control, contacts=None, dt=1.0 / 600.0)
                    state_previous, state_next = state_next, state_previous

                self.assertGreater(float(state_previous.joint_q.numpy()[0]), 0.1)
                if include_massless_fixed_child:
                    body_q = state_previous.body_q.numpy()
                    body_qd = state_previous.body_qd.numpy()
                    np.testing.assert_allclose(body_qd[1], body_qd[0], rtol=0.0, atol=1.0e-4)
                    np.testing.assert_allclose(body_q[1, 3:], body_q[0, 3:], rtol=0.0, atol=1.0e-5)
                    np.testing.assert_allclose(body_q[1, :2], body_q[0, :2], rtol=0.0, atol=1.0e-5)
                    self.assertAlmostEqual(float(body_q[1, 2] - body_q[0, 2]), 0.1, places=5)

    def test_joint_penalty_seed_supports_massless_bodies(self):
        """Seed the joint penalty of a drive whose link carries a massless fixed child, then track its target."""
        model = _build_massless_fixed_child_drive_model(device=self.default_device)
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        time_step = 1.0 / 600.0

        seed = solver.lox_joint_penalty_scale_seed(time_step).numpy()

        self.assertTrue(np.all(np.isfinite(seed)))
        self.assertTrue(np.all(seed > 0.0))
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.joint_target_q.assign([0.2])
        for _ in range(240):
            solver.step(state_previous, state_next, control, contacts=None, dt=time_step)
            state_previous, state_next = state_next, state_previous
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
        self.assertGreater(float(state_previous.joint_q.numpy()[0]), 0.1)

    def test_prescribed_contact_does_not_reject_dynamic_world(self):
        """Ignore contacts whose two incident bodies are prescribed."""
        model, _driven = _build_driven_link_with_grounded_fixed_base(device=self.default_device)
        shape_pairs = wp.array([(0, 1)], dtype=wp.vec2i, device=self.default_device)
        collision_pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=shape_pairs,
        )
        contacts = collision_pipeline.contacts()
        config = self.make_config()
        config.use_collision_detector = False
        config.lox.gauss_seidel_max_colors = 4
        solver = SolverKamino(model, config=config)
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.joint_target_q.assign([0.2])
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_previous)

        collision_pipeline.collide(state_previous, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        solver.step(state_previous, state_next, control, contacts, dt=1.0 / 600.0)
        state_previous, state_next = state_next, state_previous

        for _ in range(119):
            collision_pipeline.collide(state_previous, contacts)
            solver.step(state_previous, state_next, control, contacts, dt=1.0 / 600.0)
            state_previous, state_next = state_next, state_previous

        self.assertGreater(float(state_previous.joint_q.numpy()[0]), 0.1)

    def test_joint_effort_limit_bounds_implicit_drive(self):
        """Clamp an implicit position drive at its effort limit, excluding external joint forces."""
        # The drive of stiffness 100 N m/rad saturates towards its target; the body and armature
        # inertias about the hinge are 1 kg m^2 each
        cases = (
            # (armature, effort limit [N m], target [rad], external force [N m], expected velocity [rad/s])
            (1.0, 1.0, 1.0, 0.0, 0.05),
            (0.0, 1.0, 1.0, 0.0, 0.1),
            (1.0, 0.5, 0.0, 20.0, 0.975),
        )
        for armature, effort_limit, target, external_force, expected in cases:
            for max_iterations in (1, 40):
                with self.subTest(armature=armature, external_force=external_force, max_iterations=max_iterations):
                    model = _build_revolute_dynamics_model(
                        damping=0.0,
                        friction=0.0,
                        velocity=0.0,
                        armature=armature,
                        target_ke=100.0,
                        effort_limit=effort_limit,
                        actuator_mode=newton.JointTargetMode.POSITION,
                        device=self.default_device,
                    )
                    config = self.make_config()
                    config.lox.max_iterations = max_iterations
                    solver = SolverKamino(model, config=config)
                    state_next = model.state()
                    control = model.control()
                    control.joint_target_q.assign([target])
                    control.joint_f.assign([external_force])

                    solver.step(model.state(), state_next, control, contacts=None, dt=0.1)

                    # The reported effort is within its bound even before convergence
                    effort = float(solver._solver_kamino.data.joints.lambda_tau_j.numpy()[0])
                    self.assertAlmostEqual(abs(effort), effort_limit, places=5)
                    if max_iterations > 1:
                        self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), expected, places=4)

        # An inactive finite bound retains the unbounded implicit drive
        results = []
        for effort_limit in (math.inf, 1000.0):
            model = _build_revolute_dynamics_model(
                damping=0.0,
                friction=0.0,
                velocity=0.0,
                armature=1.0,
                target_ke=100.0,
                effort_limit=effort_limit,
                actuator_mode=newton.JointTargetMode.POSITION,
                device=self.default_device,
            )
            config = self.make_config()
            config.lox.max_iterations = 40
            solver = SolverKamino(model, config=config)
            state_next = model.state()
            control = model.control()
            control.joint_target_q.assign([1.0])

            solver.step(model.state(), state_next, control, contacts=None, dt=0.1)
            results.append(float(state_next.joint_qd.numpy()[0]))

        self.assertGreater(results[0], 1.0)
        self.assertAlmostEqual(results[1], results[0], places=4)

    def test_sliding_box_stops_at_coulomb_distance(self):
        """Converge the friction of a sliding box to its discrete Coulomb stopping distance."""
        speed, friction, gravity, time_step = 2.0, 0.5, 9.81, 1.0 / 240.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
        SolverKamino.register_custom_attributes(builder)
        shape = newton.ModelBuilder.ShapeConfig(mu=friction, restitution=0.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=shape)
        builder.add_ground_plane(cfg=shape)
        builder.body_qd[body] = wp.spatial_vector(speed, 0.0, 0.0, 0.0, 0.0, 0.0)
        model = builder.finalize(device=self.default_device)
        solver = SolverKamino(model, config=SolverKamino.Config(dynamics_solver="lox"))
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()
        start = float(state_in.body_q.numpy()[0, 0])

        for _ in range(int(0.5 / time_step)):
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, time_step)
            state_in, state_out = state_out, state_in

        # Friction decelerates the box by a constant step per time step until it stops. The friction
        # impulse must converge at the velocity tolerance, not at the looser position tolerance of
        # the body-space residuals
        deceleration = friction * gravity
        steps = np.floor(speed / (deceleration * time_step) + 1.0e-6)
        expected = time_step * (steps * speed - deceleration * time_step * steps * (steps + 1.0) / 2.0)
        distance = float(state_in.body_q.numpy()[0, 0]) - start
        self.assertAlmostEqual(float(state_in.body_qd.numpy()[0, 0]), 0.0, delta=1.0e-3)
        self.assertAlmostEqual(distance, expected, delta=5.0e-3 * expected)

    def test_light_sliding_box_converges_in_any_unit_system(self):
        """Converge the friction of an 8 g sliding box as fast as the same box expressed in grams."""
        for density in (1.0e3, 1.0e6):
            with self.subTest(density=density):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
                SolverKamino.register_custom_attributes(builder)
                shape = newton.ModelBuilder.ShapeConfig(density=density, margin=0.0, gap=0.01, mu=0.5)
                body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity()))
                builder.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01, cfg=shape)
                builder.add_ground_plane(cfg=shape)
                builder.body_qd[body] = wp.spatial_vector(0.3, 0.0, 0.0, 0.0, 0.0, 0.0)
                model = builder.finalize(device=self.default_device)
                config = SolverKamino.Config(dynamics_solver="lox", use_collision_detector=False)
                config.lox.gauss_seidel_max_colors = 4
                solver = SolverKamino(model, config=config)
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                state_in, state_out, control = model.state(), model.state(), model.control()
                converged = []

                for _ in range(100):
                    pipeline.collide(state_in, contacts)
                    solver.step(state_in, state_out, control, contacts, 0.002)
                    state_in, state_out = state_out, state_in
                    converged.append(int(solver.status.numpy()["converged"][0]))

                self.assertGreaterEqual(np.mean(converged), 0.9)

    def test_box_on_plane_projects_detected_contact(self):
        """Project detected contacts to keep a box above the plane with every projection schedule."""
        schedules = ((1, "none"), (4, "none"), (1, "apgd"), (1, "anderson"), (4, "anderson"))
        for max_colors, acceleration in schedules:
            with self.subTest(gauss_seidel_max_colors=max_colors, projection_acceleration=acceleration):
                model = ModelKamino.from_newton(build_box_on_plane(ground=True).finalize(device=self.default_device))
                detector = CollisionDetector(
                    model,
                    config=kamino_config.CollisionDetectorConfig(pipeline="unified"),
                )
                contacts = detector.contacts
                config = self.make_config(compute_solution_metrics=True)
                config.lox.gauss_seidel_max_colors = max_colors
                config.lox.projection_acceleration = acceleration
                solver = SolverKaminoImpl(model=model, contacts=contacts, config=config)
                state_previous = model.state()
                state_next = model.state()

                solver.step(
                    state_previous,
                    state_next,
                    model.control(),
                    contacts=contacts,
                    detector=detector,
                    dt=0.01,
                )

                self.assertGreater(int(contacts.model_active_contacts.numpy()[0]), 0)
                self.assertGreaterEqual(float(contacts.reaction.numpy()[0, 2]), 0.0)
                self.assertGreaterEqual(float(contacts.velocity.numpy()[0, 2]), -2.0e-4)
                self.assertGreaterEqual(float(state_next.u_i.numpy()[0, 2]), -2.0e-4)
                self.assertGreater(float(state_next.w_i.numpy()[0, 2]), 0.0)
                self.assertGreaterEqual(int(contacts.mode.numpy()[0]), 0)
                self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
                self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())
                status = solver.solver_status.numpy()
                np.testing.assert_array_equal(status["failed"], [0])
                self.assertLessEqual(float(status["r_c"][0]), 1.0e-8)
                metrics = solver.metrics.data
                for name in ("r_eom", "r_v_plus", "r_ncp_primal", "r_ncp_dual", "r_ncp_compl", "r_vi_natmap"):
                    self.assertTrue(np.isfinite(getattr(metrics, name).numpy()).all(), name)
                self.assertLess(float(metrics.r_eom.numpy()[0]), 1.0e-3)
                self.assertLess(float(metrics.r_ncp_primal.numpy()[0]), 1.0e-5)

    def test_cartpole_projects_detected_joint_limit(self):
        """Project detected joint limits for a cartpole."""
        model = ModelKamino.from_newton(build_cartpole(ground=False, limits=True).finalize(device=self.default_device))
        solver = SolverKaminoImpl(
            model=model,
            config=self.make_config(compute_solution_metrics=True),
        )
        state_previous = model.state()
        state_next = model.state()
        pose = state_previous.q_i.numpy()
        pose[:, 1] += 4.1
        state_previous.q_i.assign(pose)

        solver.step(state_previous, state_next, model.control(), dt=0.01)

        limits = solver._limits
        self.assertGreater(int(limits.model_active_limits.numpy()[0]), 0)
        self.assertGreater(float(limits.reaction.numpy()[0]), 0.0)
        self.assertGreaterEqual(float(limits.velocity.numpy()[0]), -2.0e-4)
        self.assertLess(float(state_next.dq_j.numpy()[0]), -0.09)
        self.assertTrue(np.isfinite(state_next.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_next.u_i.numpy()).all())
        status = solver.solver_status.numpy()
        np.testing.assert_array_equal(status["failed"], [0])
        self.assertLessEqual(float(status["r_c"][0]), 1.0e-8)
        metrics = solver.metrics.data
        for name in ("r_eom", "r_v_plus", "r_ncp_primal", "r_ncp_dual", "r_ncp_compl", "r_vi_natmap"):
            self.assertTrue(np.isfinite(getattr(metrics, name).numpy()).all(), name)
        self.assertLess(float(metrics.r_eom.numpy()[0]), 2.0e-2)
        self.assertLess(float(metrics.r_ncp_primal.numpy()[0]), 1.0e-5)

    def test_cartpole_sustained_joint_force_remains_bounded(self):
        """Keep cartpole motion bounded under sustained joint force."""
        model = ModelKamino.from_newton(build_cartpole(ground=False, limits=True).finalize(device=self.default_device))
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        control.tau_j.assign(np.asarray([10.0, 0.0], dtype=np.float32))

        for _ in range(300):
            solver.step(state_previous, state_next, control, dt=0.001)
            state_previous, state_next = state_next, state_previous

        body_velocity = state_previous.u_i.numpy()
        joint_residual = solver.data.joints.r_j.numpy()
        self.assertTrue(np.isfinite(body_velocity).all())
        self.assertTrue(np.isfinite(joint_residual).all())
        self.assertLess(float(np.max(np.abs(body_velocity))), 100.0)
        self.assertLess(float(np.max(np.abs(joint_residual))), 1.0e-2)
        np.testing.assert_array_equal(solver.solver_status.numpy()["failed"], [0])

    def test_cuda_graph_capture_replays_step(self):
        """Replay a captured step with and without CUDA graph conditional nodes."""
        if not self.default_device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        for use_graph_conditionals in (True, False):
            with self.subTest(use_graph_conditionals=use_graph_conditionals):
                if use_graph_conditionals and not wp.is_conditional_graph_supported():
                    self.skipTest("CUDA conditional graph nodes require CUDA 12.4 or newer.")
                model = build_box_on_plane(ground=True).finalize(device=self.default_device)
                config = self.make_config()
                config.lox.use_graph_conditionals = use_graph_conditionals
                solver = SolverKamino(model, config=config)
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                state_in, state_out, control = model.state(), model.state(), model.control()
                pipeline.collide(state_in, contacts)
                self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                solver.step(state_in, state_out, control, contacts, dt=0.01)
                expected = state_out.body_qd.numpy()

                with wp.ScopedCapture(device=self.default_device) as capture:
                    solver.step(state_in, state_out, control, contacts, dt=0.01)
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                    np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=0.0, atol=2.0e-4)
                    np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
