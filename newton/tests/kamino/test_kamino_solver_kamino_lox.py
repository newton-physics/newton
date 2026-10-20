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
    build_boxes_fourbar,
    build_boxes_hinged,
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


def _build_lone_massless_model(*, device: wp.DeviceLike = None) -> tuple[newton.Model, int]:
    """Build a massless body whose only joint is a free joint to the world."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link()
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
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


def _build_driven_chain_model(*, world_count: int, device: wp.DeviceLike = None) -> newton.Model:
    """Build worlds of an effort-limited driven hinge carrying a fixed child under gravity."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    inertia = wp.mat33f(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4)
    offset = wp.transformf(wp.vec3f(1.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32))
    for _ in range(world_count):
        builder.begin_world()
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        child = builder.add_link(xform=offset, mass=1.0, inertia=inertia, lock_inertia=True)
        root = builder.add_joint_revolute(
            parent=-1,
            child=parent,
            axis=newton.Axis.Y,
            target_ke=100.0,
            effort_limit=1.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        fixed = builder.add_joint_fixed(parent=parent, child=child, parent_xform=offset)
        builder.add_articulation([root, fixed])
        builder.end_world()
    return builder.finalize(device=device)


def _build_prescribed_rotating_joint_model(
    *,
    angle: float,
    joint_kind: str = "ball",
    nonidentity_frames: bool = False,
    device: wp.DeviceLike = None,
) -> tuple[newton.Model, int, int, wp.transformf, wp.transformf]:
    """Build a prescribed parent rotating by ``angle`` [rad] in 0.01 s with an offset dynamic child."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True, is_kinematic=True)
    child = builder.add_link(
        xform=wp.transformf(wp.vec3f(1.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32)),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    root = builder.add_joint_free(parent=-1, child=parent)
    frame_orientation = (
        wp.quat_from_axis_angle(wp.normalize(wp.vec3f(1.0, 2.0, -1.0)), 0.7)
        if nonidentity_frames
        else wp.quat_identity(dtype=wp.float32)
    )
    parent_xform = wp.transformf(wp.vec3f(1.0, 0.0, 0.0), frame_orientation)
    child_xform = wp.transformf(wp.vec3f(0.0), frame_orientation)
    if joint_kind == "ball":
        joint = builder.add_joint_ball(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform)
    elif joint_kind == "revolute":
        joint = builder.add_joint_revolute(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=newton.Axis.X,
        )
    else:
        raise ValueError(f"Unsupported joint kind: {joint_kind}")
    builder.add_articulation([root, joint])
    builder.body_qd[parent] = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, angle / 0.01)
    builder.end_world()
    return builder.finalize(device=device), parent, child, parent_xform, child_xform


def _build_two_body_joint_model(
    *,
    joint_kind: str,
    mass_ratio: float,
    reverse_body_order: bool,
    device: wp.DeviceLike = None,
) -> tuple[newton.Model, int, int, wp.transformf, wp.transformf]:
    """Build an offset two-body fixed or revolute joint below a world hinge."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    frame_orientation = wp.quat_from_axis_angle(wp.normalize(wp.vec3f(1.0, -2.0, 0.5)), 0.6)
    parent_xform = wp.transformf(wp.vec3f(0.3, -0.2, 0.7), frame_orientation)
    child_xform = wp.transformf(wp.vec3f(-0.1, 0.2, -0.3), frame_orientation)
    child_position = wp.vec3f(0.4, -0.4, 1.0)
    parent_inertia = wp.mat33f(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4)
    child_inertia = parent_inertia / mass_ratio

    def add_parent() -> int:
        return builder.add_link(mass=1.0, inertia=parent_inertia, lock_inertia=True)

    def add_child() -> int:
        return builder.add_link(
            xform=wp.transformf(child_position, wp.quat_identity(dtype=wp.float32)),
            mass=1.0 / mass_ratio,
            inertia=child_inertia,
            lock_inertia=True,
        )

    if reverse_body_order:
        child = add_child()
        parent = add_parent()
    else:
        parent = add_parent()
        child = add_child()
    root = builder.add_joint_revolute(parent=-1, child=parent, axis=newton.Axis.Y)
    if joint_kind == "fixed":
        joint = builder.add_joint_fixed(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform)
    elif joint_kind == "revolute":
        joint = builder.add_joint_revolute(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=newton.Axis.X,
        )
    else:
        raise ValueError(f"Unsupported joint kind: {joint_kind}")
    builder.add_articulation([root, joint])
    builder.end_world()
    return builder.finalize(device=device), parent, child, parent_xform, child_xform


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by an xyzw quaternion."""
    axis = quaternion[:3]
    scalar = quaternion[3]
    return vector + 2.0 * (scalar * np.cross(axis, vector) + np.cross(axis, np.cross(axis, vector)))


def _quaternion_product(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Multiply xyzw quaternions."""
    first_axis = first[:3]
    second_axis = second[:3]
    return np.concatenate(
        (
            first[3] * second_axis + second[3] * first_axis + np.cross(first_axis, second_axis),
            np.asarray([first[3] * second[3] - np.dot(first_axis, second_axis)]),
        )
    )


def _joint_anchor_error(
    body_pose: np.ndarray,
    parent: int,
    child: int,
    parent_xform: wp.transformf,
    child_xform: wp.transformf,
) -> np.ndarray:
    """Return the child-minus-parent joint anchor separation in world coordinates [m]."""
    parent_anchor = body_pose[parent, :3] + _rotate_vector(body_pose[parent, 3:], np.asarray(parent_xform.p))
    child_anchor = body_pose[child, :3] + _rotate_vector(body_pose[child, 3:], np.asarray(child_xform.p))
    return child_anchor - parent_anchor


def _joint_angular_error(
    body_pose: np.ndarray,
    parent: int,
    child: int,
    parent_xform: wp.transformf,
    child_xform: wp.transformf,
    joint_kind: str,
) -> float:
    """Return the constrained angular error of a fixed or X-axis revolute joint [rad]."""
    parent_frame = _quaternion_product(body_pose[parent, 3:], np.asarray(parent_xform.q))
    child_frame = _quaternion_product(body_pose[child, 3:], np.asarray(child_xform.q))
    parent_inverse = parent_frame * np.asarray((-1.0, -1.0, -1.0, 1.0))
    relative = _quaternion_product(parent_inverse, child_frame)
    relative /= np.linalg.norm(relative)
    if joint_kind == "fixed":
        return float(2.0 * np.arccos(np.clip(abs(relative[3]), 0.0, 1.0)))
    follower_axis = _rotate_vector(relative, np.asarray((1.0, 0.0, 0.0)))
    return float(np.linalg.norm(np.cross(np.asarray((1.0, 0.0, 0.0)), follower_axis)))


class _LOXTestCase(unittest.TestCase):
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


class TestSolverKaminoLOX(_LOXTestCase):
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
        """Keep |I ω| of a torque-free body through the LOX solve and the integrator."""
        # Principal moments that satisfy the triangle inequality, which the model builder enforces
        inertia = np.diag([2.0, 3.0, 4.0])
        omega = np.array([3.0, -7.0, 11.0])
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        body = builder.add_body(mass=1.0, inertia=wp.mat33f(*inertia.flatten()), lock_inertia=True)
        builder.body_qd[body] = wp.spatial_vectorf(0.0, 0.0, 0.0, *omega)
        model = builder.finalize(device=self.default_device)
        model.set_gravity((0.0, 0.0, 0.0))
        solver = SolverKamino(model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()

        solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.02)

        # The body starts at the identity orientation, so its world inertia is the body inertia
        next_omega = state_next.body_qd.numpy()[0, 3:]
        self.assertAlmostEqual(
            float(np.linalg.norm(inertia @ next_omega) / np.linalg.norm(inertia @ omega)), 1.0, delta=1.0e-5
        )

    def test_joint_violation_correction_is_bounded(self):
        """Correct only the part of a joint translation beyond the position tolerance, and of a joint
        rotation beyond the rotation tolerance, by at most ``joint_max_correction`` per step."""
        time_step = 0.01
        tolerance = 1.0e-4
        # The other tolerance stays far smaller, so each row must use the tolerance of its own kind
        small_tolerance = 1.0e-7
        max_correction = self.make_config().lox.joint_max_correction
        cases = (
            # (kind, offset, corrected offset, accuracy of the corrected offset)
            ("translation", 0.5 * tolerance, 0.0, 0.05 * tolerance),
            ("translation", 3.0 * tolerance, 2.0 * tolerance, 0.05 * tolerance),
            ("rotation", 3.0 * tolerance, 2.0 * tolerance, 0.05 * tolerance),
            ("translation", 100.0 * max_correction, max_correction, 0.05 * max_correction),
        )
        for kind, offset, corrected, accuracy in cases:
            with self.subTest(kind=kind, offset=offset):
                model = _build_revolute_dynamics_model(damping=0.0, friction=0.0, velocity=0.0)
                config = self.make_config()
                config.lox.position_tolerance = tolerance if kind == "translation" else small_tolerance
                config.lox.rotation_tolerance = tolerance if kind == "rotation" else small_tolerance
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                body_q = state_previous.body_q.numpy()
                if kind == "translation":
                    # Pull the link off its hinge anchor along x
                    body_q[0, 0] += offset
                else:
                    # Tilt the link about x, across its hinge axis along y
                    body_q[0, 3:7] = (math.sin(0.5 * offset), 0.0, 0.0, math.cos(0.5 * offset))
                state_previous.body_q.assign(body_q)

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=time_step)

                velocity = state_next.body_qd.numpy()[0]
                component = velocity[0] if kind == "translation" else velocity[3]
                self.assertAlmostEqual(float(-component), corrected / time_step, delta=accuracy / time_step)

    def test_spinning_hinge_keeps_its_axis(self):
        """Integrate the LOX velocity of a body spinning about a non-principal hinge axis along that axis."""
        axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
        speed = 10.0
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
        solver = SolverKamino(model, config=self.make_config())
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
            "lone_massless_free_body": _build_lone_massless_model(device=self.default_device),
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


class TestSolverKaminoLOXJoints(_LOXTestCase):
    def test_joint_penalty_seed_follows_notified_mass_changes(self):
        """Seed the joint penalty again when the masses change, and keep stepping a graph captured before."""
        time_step = 0.01

        def build(model):
            config = self.make_config()
            config.lox.joint_penalty_seed_time_step = time_step
            return SolverKamino(model, config=config)

        model = _build_driven_chain_model(world_count=2, device=self.default_device)
        solver = build(model)
        state_previous, state_next, control = model.state(), model.state(), model.control()
        with wp.ScopedCapture(device=self.default_device) as capture:
            solver.step(state_previous, state_next, control, contacts=None, dt=time_step)
        seed = solver._solver_kamino.solver_fd._data.joint_penalty_scale
        initial_seed = seed.numpy().copy()

        # Make the child of each world heavier
        mass = model.body_mass.numpy()
        inertia = model.body_inertia.numpy()
        mass[1::2] *= 50.0
        inertia[1::2] *= 50.0
        model.body_mass.assign(mass)
        model.body_inv_mass.assign(1.0 / mass)
        model.body_inertia.assign(inertia)
        model.body_inv_inertia.assign(np.linalg.inv(inertia))
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

        fresh_seed = build(model)._solver_kamino.solver_fd._data.joint_penalty_scale.numpy()
        self.assertGreater(float(fresh_seed[0]), 2.0 * float(initial_seed[0]))
        np.testing.assert_allclose(seed.numpy(), fresh_seed, rtol=1.0e-4)
        for _ in range(5):
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0, 0])
        np.testing.assert_allclose(seed.numpy(), fresh_seed, rtol=1.0e-4)

    def test_massless_link_needs_inertia_on_its_free_joint_dof(self):
        """Reject singular links on joint DOFs without armature, and step a massless link once its DOF has armature."""
        for armature in (0.0, 0.01):
            with self.subTest(armature=armature):
                builder = newton.ModelBuilder()
                SolverKamino.register_custom_attributes(builder)
                base = builder.add_link(
                    mass=1.0, inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), lock_inertia=True
                )
                link = builder.add_link()
                hinge = builder.add_joint_revolute(parent=-1, child=base, axis=newton.Axis.Y)
                joint = builder.add_joint_revolute(
                    parent=base,
                    child=link,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform((0.5, 0.0, 0.0), wp.quat_identity()),
                    armature=armature,
                )
                builder.add_articulation([hinge, joint])
                model = builder.finalize(device=self.default_device)
                config = self.make_config()
                if armature == 0.0:
                    with self.assertRaisesRegex(ValueError, "joint DOFs without armature"):
                        SolverKamino(model, config=config)
                    continue
                solver = SolverKamino(model, config=config)
                state_previous, state_next = model.state(), model.state()
                for _ in range(10):
                    solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)
                    state_previous, state_next = state_next, state_previous
                np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])

        # A free body with mass but a singular inverse inertia leaves its rotation without inertia
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.add_body(mass=1.0, inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        model = builder.finalize(device=self.default_device)
        model.body_inv_inertia.zero_()
        with self.assertRaisesRegex(ValueError, "joint DOFs without armature"):
            SolverKamino(model, config=self.make_config())

    def test_joint_proximal_uses_frozen_frame_for_full_position_blocks(self):
        """Constrain offset anchors while a prescribed parent rotates substantially."""
        # The relaxation test covers a ball joint rotating by 0.5 rad with offset frames
        cases = (
            ("ball", 0.1, False),
            ("ball", 1.2, True),
            ("revolute", 0.5, True),
        )
        for joint_kind, angle, nonidentity_frames in cases:
            with self.subTest(joint_kind=joint_kind, angle=angle, nonidentity_frames=nonidentity_frames):
                model, parent, child, parent_xform, child_xform = _build_prescribed_rotating_joint_model(
                    angle=angle,
                    joint_kind=joint_kind,
                    nonidentity_frames=nonidentity_frames,
                    device=self.default_device,
                )
                config = self.make_config()
                config.use_collision_detector = False
                config.lox.fixed_iterations = True
                config.lox.joint_proximal_relaxation = 1.0
                solver = SolverKamino(model, config=config)
                state_next = model.state()

                solver.step(model.state(), state_next, model.control(), contacts=None, dt=0.01)

                anchor_error = _joint_anchor_error(state_next.body_q.numpy(), parent, child, parent_xform, child_xform)
                self.assertTrue(np.isfinite(anchor_error).all())
                np.testing.assert_allclose(anchor_error, 0.0, rtol=0.0, atol=1.0e-6)

    def test_joint_proximal_relaxations_share_position_fixed_point(self):
        """Reach the same anchor constraint for every positive relaxation."""
        child_poses = []
        for relaxation in (0.25, 0.5, 1.0):
            with self.subTest(relaxation=relaxation):
                model, parent, child, parent_xform, child_xform = _build_prescribed_rotating_joint_model(
                    angle=0.5,
                    nonidentity_frames=True,
                    device=self.default_device,
                )
                config = self.make_config()
                config.use_collision_detector = False
                config.lox.fixed_iterations = True
                config.lox.max_iterations = 80
                config.lox.joint_proximal_relaxation = relaxation
                solver = SolverKamino(model, config=config)
                state_next = model.state()

                solver.step(model.state(), state_next, model.control(), contacts=None, dt=0.01)

                poses = state_next.body_q.numpy()
                anchor_error = _joint_anchor_error(poses, parent, child, parent_xform, child_xform)
                np.testing.assert_allclose(anchor_error, 0.0, rtol=0.0, atol=1.0e-6)
                child_poses.append(poses[child])
        child_poses = np.asarray(child_poses)
        np.testing.assert_allclose(child_poses, np.broadcast_to(child_poses[0], child_poses.shape), atol=2.0e-6)

    def test_joint_proximal_handles_two_body_frames_and_orderings(self):
        """Keep fixed and revolute joint errors bounded across mass ratios, body storage orders, and relaxations."""
        cases = (
            ("fixed", 1.0, False, 0.5),
            ("fixed", 100.0, True, 1.0),
            ("revolute", 100.0, False, 1.0),
            ("revolute", 1.0, True, 1.0),
        )
        for joint_kind, mass_ratio, reverse_body_order, relaxation in cases:
            with self.subTest(
                joint_kind=joint_kind,
                mass_ratio=mass_ratio,
                reverse_body_order=reverse_body_order,
                relaxation=relaxation,
            ):
                model, parent, child, parent_xform, child_xform = _build_two_body_joint_model(
                    joint_kind=joint_kind,
                    mass_ratio=mass_ratio,
                    reverse_body_order=reverse_body_order,
                    device=self.default_device,
                )
                config = self.make_config()
                config.use_collision_detector = False
                config.lox.fixed_iterations = True
                config.lox.joint_proximal_relaxation = relaxation
                solver = SolverKamino(model, config=config)
                state_previous = model.state()
                state_next = model.state()
                velocity = np.zeros((model.body_count, 6), dtype=np.float32)
                velocity[parent] = (2.0, -1.0, 0.5, 0.0, 50.0, 0.0)
                velocity[child] = (-3.0, 2.0, -1.0, 20.0, -10.0, 15.0)
                state_previous.body_qd.assign(velocity)

                solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)

                poses = state_next.body_q.numpy()
                position_error = np.linalg.norm(_joint_anchor_error(poses, parent, child, parent_xform, child_xform))
                angular_error = _joint_angular_error(poses, parent, child, parent_xform, child_xform, joint_kind)
                self.assertTrue(math.isfinite(position_error))
                self.assertTrue(math.isfinite(angular_error))
                self.assertLess(position_error, 1.0e-4)
                self.assertLess(angular_error, 0.1)

    def test_joint_proximal_accepts_finite_root_translation_transient(self):
        """Accept a large finite first trial of a relaxed fixed joint."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        inertia = wp.mat33f(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2)
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        child = builder.add_link(
            xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
            mass=1.0,
            inertia=inertia,
            lock_inertia=True,
        )
        root = builder.add_joint_revolute(parent=-1, child=parent, axis=newton.Axis.Y)
        parent_xform = wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32))
        child_xform = wp.transformf(wp.vec3f(0.0), wp.quat_identity(dtype=wp.float32))
        joint = builder.add_joint_fixed(parent=parent, child=child, parent_xform=parent_xform, child_xform=child_xform)
        builder.add_articulation([root, joint])
        builder.end_world()
        model = builder.finalize(device=self.default_device)
        state_in = model.state()
        state_in.body_qd.assign(np.random.default_rng(1).normal(size=(2, 6)).astype(np.float32) / 0.01)
        state_out = model.state()
        config = self.make_config()
        config.use_collision_detector = False
        config.lox.fixed_iterations = True
        config.lox.max_iterations = 80
        config.lox.joint_proximal_relaxation = 1.0
        solver = SolverKamino(model, config=config)

        solver.step(state_in, state_out, model.control(), contacts=None, dt=0.01)

        self.assertEqual(int(solver.status.numpy()["failed"][0]), 0)
        anchor_error = _joint_anchor_error(state_out.body_q.numpy(), parent, child, parent_xform, child_xform)
        self.assertTrue(np.isfinite(anchor_error).all())
        self.assertLess(float(np.linalg.norm(anchor_error)), 1.0e-5)

    def test_joint_proximal_rotates_poses_velocities_and_wrenches(self):
        """Rotate the whole scene and obtain the rotated solution."""
        rotation = np.asarray(wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, -2.0, 0.7)), 1.1))
        results = []
        for rotated in (False, True):
            model, parent, child, _, _ = _build_two_body_joint_model(
                joint_kind="revolute",
                mass_ratio=100.0,
                reverse_body_order=True,
                device=self.default_device,
            )
            velocity = np.zeros((model.body_count, 6), dtype=np.float32)
            velocity[parent] = (2.0, -1.0, 0.5, 0.0, 50.0, 0.0)
            velocity[child] = (-3.0, 2.0, -1.0, 20.0, -10.0, 15.0)
            if rotated:
                poses = model.body_q.numpy()
                for pose in poses:
                    pose[:3] = _rotate_vector(rotation, pose[:3])
                    pose[3:] = _quaternion_product(rotation, pose[3:])
                model.body_q.assign(poses)
                root_frames = model.joint_X_p.numpy()
                for joint in np.flatnonzero(model.joint_parent.numpy() < 0):
                    root_frames[joint, :3] = _rotate_vector(rotation, root_frames[joint, :3])
                    root_frames[joint, 3:] = _quaternion_product(rotation, root_frames[joint, 3:])
                model.joint_X_p.assign(root_frames)
                for body in range(model.body_count):
                    velocity[body, :3] = _rotate_vector(rotation, velocity[body, :3])
                    velocity[body, 3:] = _rotate_vector(rotation, velocity[body, 3:])
            config = self.make_config()
            config.use_collision_detector = False
            config.lox.fixed_iterations = True
            config.lox.joint_proximal_relaxation = 1.0
            solver = SolverKamino(model, config=config)
            state_previous, state_next = model.state(), model.state()
            state_previous.body_qd.assign(velocity)
            solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)
            results.append(
                (
                    state_next.body_q.numpy(),
                    state_next.body_qd.numpy(),
                    solver._solver_kamino.data.bodies.w_j_i.numpy(),
                )
            )
        original, transformed = results
        for body in range(len(original[0])):
            np.testing.assert_allclose(
                transformed[0][body, :3], _rotate_vector(rotation, original[0][body, :3]), atol=2.0e-5
            )
            expected_q = _quaternion_product(rotation, original[0][body, 3:])
            actual_q = transformed[0][body, 3:]
            if np.dot(expected_q, actual_q) < 0.0:
                actual_q = -actual_q
            np.testing.assert_allclose(actual_q, expected_q, atol=2.0e-5)
            for quantity in (1, 2):
                for begin in (0, 3):
                    np.testing.assert_allclose(
                        transformed[quantity][body, begin : begin + 3],
                        _rotate_vector(rotation, original[quantity][body, begin : begin + 3]),
                        rtol=5.0e-4,
                        atol=2.0e-3,
                    )

    def test_failed_jointed_world_stays_failed_until_reset(self):
        """Zero the outputs of a failed world, keep it failed, and recover it after a reset."""
        model = _build_driven_chain_model(world_count=2, device=self.default_device)
        solver = SolverKamino(model, config=self.make_config())
        state_in, state_out, control = model.state(), model.state(), model.control()
        solver.step(state_in, state_out, control, contacts=None, dt=0.01)
        state_in, state_out = state_out, state_in
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0, 0])
        joints = solver._solver_kamino.data.joints
        bodies = solver._solver_kamino.data.bodies
        half_rows = joints.lambda_kin_j.shape[0] // 2
        half_bodies = model.body_count // 2
        self.assertGreater(float(np.abs(joints.lambda_kin_j.numpy()[half_rows:]).max()), 0.0)

        # On the CPU, the NumPy view aliases the state
        finite = state_in.body_qd.numpy().copy()
        velocity = finite.copy()
        velocity[half_bodies:] = np.nan
        state_in.body_qd.assign(velocity)
        solver.step(state_in, state_out, control, contacts=None, dt=0.01)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0, 1])
        np.testing.assert_array_equal(joints.lambda_tau_j.numpy()[1], 0.0)
        np.testing.assert_array_equal(bodies.w_j_i.numpy()[half_bodies:], 0.0)
        self.assertTrue(np.isfinite(joints.lambda_kin_j.numpy()[:half_rows]).all())

        # The failed world stays failed with a finite input
        state_in.body_qd.assign(finite)
        solver.step(state_in, state_out, control, contacts=None, dt=0.01)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0, 1])
        np.testing.assert_array_equal(bodies.w_j_i.numpy()[half_bodies:], 0.0)

        # A reset of the failed world clears its failure and its joint warm starts
        solver.reset(state_in, world_mask=wp.array([False, True, False], dtype=wp.bool, device=self.default_device))
        np.testing.assert_array_equal(joints.lambda_kin_j.numpy()[half_rows:], 0.0)
        solver.step(state_in, state_out, control, contacts=None, dt=0.01)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0, 0])
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())

    def test_position_drive_with_massless_fixed_child(self):
        """Seed the joint penalty of a drive whose link carries a massless fixed child, then track its target."""
        time_step = 1.0 / 600.0
        for integrator in ("euler", "moreau"):
            with self.subTest(integrator=integrator):
                model = _build_massless_fixed_child_drive_model(device=self.default_device)
                config = self.make_config()
                config.integrator = integrator
                config.use_collision_detector = integrator == "moreau"
                config.lox.joint_penalty_seed_time_step = time_step
                solver = SolverKamino(model, config=config)
                seed = solver._solver_kamino.solver_fd._data.joint_penalty_scale.numpy()
                self.assertTrue(np.all(np.isfinite(seed)))
                self.assertTrue(np.all(seed > 0.0))
                state_previous = model.state()
                state_next = model.state()
                control = model.control()
                control.joint_target_q.assign([0.2])

                for _ in range(120):
                    solver.step(state_previous, state_next, control, contacts=None, dt=time_step)
                    state_previous, state_next = state_next, state_previous

                np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
                self.assertGreater(float(state_previous.joint_q.numpy()[0]), 0.1)
                body_q = state_previous.body_q.numpy()
                body_qd = state_previous.body_qd.numpy()
                np.testing.assert_allclose(body_qd[1], body_qd[0], rtol=0.0, atol=1.0e-4)
                np.testing.assert_allclose(body_q[1, 3:], body_q[0, 3:], rtol=0.0, atol=1.0e-5)
                np.testing.assert_allclose(body_q[1, :2], body_q[0, :2], rtol=0.0, atol=1.0e-5)
                self.assertAlmostEqual(float(body_q[1, 2] - body_q[0, 2]), 0.1, places=5)

    def test_bodies_welded_to_moving_kinematic_body_follow_it(self):
        """Carry a massive and a massless welded body rigidly with a translating and rotating kinematic body."""
        inertia = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, -1.0)), 0.7)
        for eliminate in (True, False):
            with self.subTest(eliminate_fixed_world_islands=eliminate):
                builder = newton.ModelBuilder()
                SolverKamino.register_custom_attributes(builder)
                base = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True, is_kinematic=True)
                child = builder.add_link(mass=1.0, inertia=inertia, com=(0.1, 0.0, 0.2), lock_inertia=True)
                grandchild = builder.add_link()
                root = builder.add_joint_free(parent=-1, child=base)
                child_weld = builder.add_joint_fixed(
                    parent=base, child=child, parent_xform=wp.transform((0.5, 0.0, 0.0), rotation)
                )
                grandchild_weld = builder.add_joint_fixed(
                    parent=child,
                    child=grandchild,
                    parent_xform=wp.transform((0.0, 0.3, 0.0), wp.quat_identity()),
                    child_xform=wp.transform((0.0, 0.0, 0.1), rotation),
                )
                builder.add_articulation([root, child_weld, grandchild_weld])
                builder.body_qd[base] = wp.spatial_vectorf(0.3, 0.0, 0.0, 0.0, 0.0, 1.0)
                model = builder.finalize(device=self.default_device)
                config = self.make_config()
                config.use_collision_detector = False
                config.lox.eliminate_fixed_world_islands = eliminate
                solver = SolverKamino(model, config=config)
                # The welded bodies follow joint frames changed after construction
                joint_x_p = model.joint_X_p.numpy()
                joint_x_p[child_weld, :3] = (0.6, 0.0, 0.1)
                model.joint_X_p.assign(joint_x_p)
                solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
                state_previous, state_next = model.state(), model.state()

                for _ in range(50):
                    solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)
                    state_previous, state_next = state_next, state_previous

                np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
                body_q = state_previous.body_q.numpy()
                body_qd = state_previous.body_qd.numpy()
                body_com = model.body_com.numpy()
                joint_x_p = model.joint_X_p.numpy()
                joint_x_c = model.joint_X_c.numpy()
                expected_pose = wp.transform(*body_q[base])
                for joint, body in ((child_weld, child), (grandchild_weld, grandchild)):
                    expected_pose = wp.transform_multiply(
                        expected_pose,
                        wp.transform_multiply(
                            wp.transform(*joint_x_p[joint]), wp.transform_inverse(wp.transform(*joint_x_c[joint]))
                        ),
                    )
                    np.testing.assert_allclose(body_q[body], np.array(expected_pose), rtol=0.0, atol=1.0e-5)
                    lever = np.array(wp.transform_point(expected_pose, wp.vec3(*body_com[body]))) - body_q[base, :3]
                    transported = body_qd[base, :3] + np.cross(body_qd[base, 3:], lever)
                    np.testing.assert_allclose(body_qd[body, :3], transported, rtol=0.0, atol=1.0e-5)
                    np.testing.assert_allclose(body_qd[body, 3:], body_qd[base, 3:], rtol=0.0, atol=1.0e-6)

    def test_massless_body_welded_to_world_holds_its_island(self):
        """Hold a massive body welded to the world through a massless body, with world islands kept dynamic."""
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        base = builder.add_link()
        link = builder.add_link(
            xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            lock_inertia=True,
        )
        world_weld = builder.add_joint_fixed(parent=-1, child=base)
        weld = builder.add_joint_fixed(
            parent=base, child=link, parent_xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity())
        )
        builder.add_articulation([world_weld, weld])
        model = builder.finalize(device=self.default_device)
        config = self.make_config()
        config.use_collision_detector = False
        config.lox.eliminate_fixed_world_islands = False
        solver = SolverKamino(model, config=config)
        state_previous, state_next = model.state(), model.state()

        for _ in range(100):
            solver.step(state_previous, state_next, model.control(), contacts=None, dt=0.01)
            state_previous, state_next = state_next, state_previous

        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
        np.testing.assert_allclose(state_previous.body_q.numpy(), model.body_q.numpy(), rtol=0.0, atol=1.0e-3)

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
        for _ in range(2):
            collision_pipeline.collide(state_previous, contacts)
            solver.step(state_previous, state_next, control, contacts, dt=1.0 / 600.0)
            state_previous, state_next = state_next, state_previous

        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0])
        self.assertGreater(float(state_previous.joint_qd.numpy()[0]), 0.0)

    def test_joint_effort_limit_bounds_implicit_drive(self):
        """Clamp an implicit position drive at its effort limit, excluding external joint forces."""
        # The drive of stiffness 100 N m/rad saturates towards its target; the body and armature
        # inertias about the hinge are 1 kg m^2 each
        cases = (
            # (armature, effort limit [N m], target [rad], external force [N m], expected velocity [rad/s], saturated)
            (1.0, 1.0, 1.0, 0.0, 0.05, True),
            (0.0, 1.0, 1.0, 0.0, 0.1, True),
            (1.0, 0.5, 0.0, 20.0, 0.975, True),
            # An inactive finite bound retains the implicit drive velocity dt k / (I + dt^2 k) = 10 / 3 rad/s
            (1.0, 1000.0, 1.0, 0.0, 10.0 / 3.0, False),
        )
        # The first case also runs a single iteration, whose reported effort is already within its bound
        runs = [(case, 40) for case in cases] + [(cases[0], 1)]
        for (armature, effort_limit, target, external_force, expected, saturated), max_iterations in runs:
            with self.subTest(
                armature=armature,
                effort_limit=effort_limit,
                external_force=external_force,
                max_iterations=max_iterations,
            ):
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

                if saturated:
                    effort = float(solver._solver_kamino.data.joints.lambda_tau_j.numpy()[0])
                    self.assertAlmostEqual(abs(effort), effort_limit, places=5)
                if max_iterations > 1:
                    self.assertAlmostEqual(float(state_next.joint_qd.numpy()[0]), expected, places=4)


class TestSolverKaminoLOXConstraints(_LOXTestCase):
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

                # The box slides for 30 steps
                for _ in range(20):
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
        """Keep cartpole motion bounded under a sustained joint force against the rail limit."""
        model = ModelKamino.from_newton(build_cartpole(ground=False, limits=True).finalize(device=self.default_device))
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        # Start past the upper rail limit, pushed further along the rail
        pose = state_previous.q_i.numpy()
        pose[:, 1] += 4.1
        state_previous.q_i.assign(pose)
        control = model.control()
        control.tau_j.assign(np.asarray([10.0, 0.0], dtype=np.float32))

        for _ in range(50):
            solver.step(state_previous, state_next, control, dt=0.001)
            state_previous, state_next = state_next, state_previous

        body_velocity = state_previous.u_i.numpy()
        joint_residual = solver.data.joints.r_j.numpy()
        self.assertTrue(np.isfinite(body_velocity).all())
        self.assertTrue(np.isfinite(joint_residual).all())
        self.assertLess(float(np.max(np.abs(body_velocity))), 100.0)
        self.assertLess(float(np.max(np.abs(joint_residual))), 1.0e-2)
        np.testing.assert_array_equal(solver.solver_status.numpy()["failed"], [0])

    def test_weight_beta_caps_geometric_mean_weights(self):
        """Keep each body weight between its free patch complement and weight_beta times it, and below the mean."""
        model = build_boxes_fourbar(floatingbase=True, ground=False).finalize(device=self.default_device)
        config = self.make_config()
        config.use_collision_detector = False
        solver = SolverKamino(model, config=config)
        solver.step(model.state(), model.state(), model.control(), contacts=None, dt=0.01)
        lox = solver._solver_kamino._solver_fd
        problem = lox._problem
        # Weigh every body, and evaluate each cap on the same smooth matrix
        problem.data.body_weight_enabled.fill_(1)

        def weights(beta: float, sigma: float) -> np.ndarray:
            problem.build_smooth_system(lox._model.time.dt, joint_penalty_scale=lox._data.joint_penalty_scale)
            problem.compute_patch_schur_weights(beta, sigma)
            return problem.system.data.weight.numpy().astype(np.float64)

        def ratios(weight: np.ndarray, metric: np.ndarray) -> np.ndarray:
            """Eigenvalues of each weight relative to each metric."""
            inverse = np.linalg.inv(np.linalg.cholesky(metric))
            return np.linalg.eigvalsh(inverse @ weight @ np.swapaxes(inverse, -1, -2))

        mean = weights(0.0, 0.0)
        free = weights(1.0, 0.0)
        mean_ratios = ratios(mean, free)
        self.assertGreater(float(mean_ratios.min()), 1.0 - 1.0e-3)
        # A cap between one and the largest ratio of the mean binds on some bodies
        beta = 0.5 * (1.0 + float(mean_ratios.max()))
        self.assertGreater(beta, 1.1)
        capped = weights(beta, 0.0)
        capped_ratios = ratios(capped, free)
        self.assertGreater(float(capped_ratios.min()), 1.0 - 1.0e-3)
        self.assertAlmostEqual(float(capped_ratios.max()), beta, delta=1.0e-3 * beta)
        self.assertLess(float(ratios(capped, mean).max()), 1.0 + 1.0e-3)
        # A unit floor keeps the mean under any cap
        np.testing.assert_allclose(weights(1.0, 1.0), mean, rtol=0.0, atol=1.0e-5 * np.abs(mean).max())

    def test_hinged_boxes_drop_settles_with_bounded_joint_error(self):
        """Settle dropped hinged boxes with bounded joint error."""
        model = ModelKamino.from_newton(
            build_boxes_hinged(z_offset=0.1, ground=True).finalize(device=self.default_device)
        )
        detector = CollisionDetector(model, config=kamino_config.CollisionDetectorConfig(pipeline="unified"))
        contacts = detector.contacts
        solver = SolverKaminoImpl(model=model, contacts=contacts, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()

        for _ in range(150):
            solver.step(state_previous, state_next, control, contacts=contacts, detector=detector, dt=0.005)
            state_previous, state_next = state_next, state_previous

        # A failed world stays failed, so the final status covers every step
        np.testing.assert_array_equal(solver.solver_status.numpy()["failed"], [0])
        self.assertGreater(int(contacts.model_active_contacts.numpy()[0]), 0)
        self.assertLess(float(np.max(np.abs(solver.data.joints.r_j.numpy()), initial=0.0)), 5.0e-3)
        self.assertLess(float(np.max(np.linalg.norm(state_previous.u_i.numpy(), axis=1))), 1.0e-1)
        np.testing.assert_allclose(state_previous.q_i.numpy()[:, 2], 0.05, rtol=0.0, atol=1.0e-2)


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
