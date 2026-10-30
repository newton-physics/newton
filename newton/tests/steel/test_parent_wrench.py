# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""BODY_PARENT_F of SolverSteel: joint wrenches accounted from the solved step."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.kernels import accumulate_connect_wrenches_from_impulses, populate_connect_J_for_size
from newton.exceptions import NewtonWarning
from newton.solvers import SolverBase, SolverSteel
from newton.tests.steel._fixtures import (
    _build_carried_load,
    _build_four_bar,
    _parent_wrench_scene,
    set_torsion_radius,
)
from newton.tests.steel.test_articulation import _prescribe_carrier
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

FLAGS = {SolverBase.ObservableKind.BODY_PARENT_F}
DT_parent_wrench = 1.0 / 120.0


def test_parent_wrench_eager_step_before_first_replay(test, device):
    """Step eagerly between capturing the first BODY_PARENT_F step and its first replay, and match an eager twin.

    Each run first steps once without the observable, so the collision pipeline's lazily allocated buffers exist.
    """
    colored = {"enable_joint_velocity_limits": True, "articulated_contact_response": "propagation_colored"}
    for options in ({"enable_joint_velocity_limits": False}, {"enable_joint_velocity_limits": True}, colored):
        with test.subTest(**options):
            runs = []
            for _ in range(2):
                model, pipeline = _parent_wrench_scene(device)
                solver = SolverSteel(
                    model,
                    solve_mode="matrix_free",
                    friction_anchor_beta=0.0,
                    articulation_constraint_max=128,
                    **options,
                )
                contacts = pipeline.contacts()
                observables = solver.observables(FLAGS)
                states = [model.state(), model.state()]
                runs.append((pipeline, contacts, solver, observables, states, model.control()))

            def step(run, observe=True):
                pipeline, contacts, solver, observables, states, control = run
                for i in (0, 1):
                    pipeline.collide(states[i], contacts)
                    solver.step(
                        states[i],
                        states[1 - i],
                        control,
                        contacts,
                        DT_parent_wrench,
                        observables=observables if observe else None,
                    )

            for run in runs:
                step(run, observe=False)
            with wp.ScopedCapture(device) as capture:
                step(runs[0])
            step(runs[0])
            step(runs[1])
            wp.synchronize_device(device)
            for _ in range(3):
                wp.capture_launch(capture.graph)
                step(runs[1])
            captured, eager = (run[3].body_parent_f.numpy() for run in runs)
            test.assertGreater(np.abs(eager).max(), 1.0)
            np.testing.assert_allclose(captured, eager, rtol=1.0e-5, atol=1.0e-5)


def _branched_chains(device, world_count=2):
    """Two-branch revolute trees on a fixed base plus a free box, with gravity, revolute joint forces and velocities.

    The branches have three and four links, so the tree and the free box form two articulation-size groups.
    """
    template = newton.ModelBuilder()
    root = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    template.add_shape_box(root, hx=0.05, hy=0.05, hz=0.05)
    joints = [template.add_joint_fixed(parent=-1, child=root)]
    for side, levels in ((-1.0, 3), (1.0, 4)):
        parent = root
        for level in range(levels):
            child = template.add_link(mass=0.5 + 0.1 * level, inertia=wp.mat33(np.eye(3) * 0.005))
            template.add_shape_capsule(child, radius=0.02, half_height=0.08)
            axis = (0.0, 1.0, 0.0) if level % 2 == 0 else (1.0, 0.0, 0.0)
            joints.append(
                template.add_joint_revolute(
                    parent=parent,
                    child=child,
                    axis=axis,
                    parent_xform=wp.transform(wp.vec3(side * 0.1, 0.0, -0.05 - 0.15 * level), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.08), wp.quat_identity()),
                )
            )
            parent = child
    template.add_articulation(joints)
    free = template.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 2.0), wp.quat_identity()))
    template.add_shape_box(free, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    for _ in range(world_count):
        builder.add_world(template)
    model = builder.finalize(device=device)
    state = model.state()
    rng = np.random.default_rng(3)
    joint_q = model.joint_q.numpy()
    joint_qd = model.joint_qd.numpy()
    revolute = model.joint_type.numpy() == int(newton.JointType.REVOLUTE)
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
    for joint in np.flatnonzero(revolute):
        joint_q[q_start[joint]] = rng.uniform(-0.6, 0.6)
        joint_qd[qd_start[joint]] = rng.uniform(-1.0, 1.0)
        # Free-joint forces enter Featherstone's and Steel's root wrench differently.
        joint_f[qd_start[joint]] = rng.uniform(-0.5, 0.5)
    state.joint_q.assign(joint_q)
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    control.joint_f.assign(joint_f)
    return model, state, control


def _first_step(model, state, control, solver):
    observables = solver.observables(FLAGS)
    state_out = model.state()
    solver.step(state, state_out, control, None, DT_parent_wrench, observables=observables)
    return observables.body_parent_f.numpy(), state_out


def _revolute_wrench_torques(model, state, wrench):
    """Project each revolute joint's published wrench onto its axis about its anchor."""
    body_q = state.body_q.numpy()
    body_com = model.body_com.numpy()
    joint_X_c = model.joint_X_c.numpy()
    joint_axis = model.joint_axis.numpy()
    torques = {}
    for joint in np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE)):
        child = model.joint_child.numpy()[joint]
        X_wc = wp.transform(*body_q[child])
        X_wj = X_wc * wp.transform(*joint_X_c[joint])
        anchor = np.array(wp.transform_get_translation(X_wj))
        axis = np.array(
            wp.quat_rotate(wp.transform_get_rotation(X_wj), wp.vec3(*joint_axis[model.joint_qd_start.numpy()[joint]]))
        )
        com = np.array(wp.transform_point(X_wc, wp.vec3(*body_com[child])))
        force, torque_com = wrench[child, :3], wrench[child, 3:]
        torques[joint] = float(np.dot(axis, torque_com + np.cross(com - anchor, force)))
    return torques


def test_parent_wrench_matches_inverse_dynamics_on_every_path(test, device):
    """Publish the joint wrench of the solved motion, whichever inverse-dynamics path a configuration selects.

    Every revolute wrench projects onto its axis as exactly the applied joint torque, and a free body in flight
    transmits nothing.
    """
    model, state, control = _branched_chains(device)
    configurations = [{"solve_mode": "split"}]
    if wp.get_device(device).is_cuda:
        configurations += [
            {"solve_mode": "matrix_free"},
            {"solve_mode": "matrix_free", "parallel_tree": True},
            {"solve_mode": "matrix_free", "use_parallel_streams": True},
            {"solve_mode": "matrix_free", "double_buffer": True},
        ]
    joint_f = control.joint_f.numpy()
    qd_start = model.joint_qd_start.numpy()
    free_bodies = model.joint_child.numpy()[model.joint_type.numpy() == int(newton.JointType.FREE)]
    reference = None
    for options in configurations:
        with test.subTest(**options):
            solver = SolverSteel(model, **options)
            if options.get("use_parallel_streams"):
                test.assertGreater(len(solver._size_groups), 1)
            solver._rigid_body_angular_damping.zero_()
            wrench, _ = _first_step(model, state, control, solver)
            test.assertGreater(np.abs(wrench).max(), 1.0)
            for joint, torque in _revolute_wrench_torques(model, state, wrench).items():
                test.assertAlmostEqual(torque, joint_f[qd_start[joint]], delta=1.0e-3)
            np.testing.assert_allclose(wrench[free_bodies], 0.0, atol=1.0e-3)
            if reference is None:
                reference = wrench
            np.testing.assert_allclose(wrench, reference, rtol=1.0e-4, atol=1.0e-4)


G = 9.81
CONTACT_DT = 1.0 / 480.0
M_CARRIAGE = 1.0
M_GRIP = 0.5
M_FINGER = 0.1
M_BOX = 0.5


def _gripper_scene(device, *, lift_z=0.5, lift_drive=True, flange_drive=True, load_box_x=None, pinch=False):
    """A held tray gripper: a fixed prismatic lift, a revolute flange about z and a 0.2 m square tray.

    Optionally a free box rests on the tray at ``load_box_x``, or two prismatic fingers pinch a free box below it.
    The tray's bottom face is 0.13 m below the lift's origin.
    """
    builder = newton.ModelBuilder()
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=1.0))
    carriage = builder.add_link(mass=M_CARRIAGE, inertia=wp.mat33(np.eye(3) * 0.01))
    ke, kd = (1.0e5, 1.0e3) if lift_drive else (0.0, 0.0)
    lift = builder.add_joint_prismatic(
        -1,
        carriage,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, lift_z), wp.quat_identity()),
        target_ke=ke,
        target_kd=kd,
    )
    grip = builder.add_link(
        mass=M_GRIP, com=(0.0, 0.0, -0.02), inertia=wp.mat33(np.diag([0.002, 0.002, 0.003])), lock_inertia=True
    )
    builder.add_shape_box(
        grip, xform=wp.transform((0.0, 0.0, -0.02), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.01, cfg=shape_cfg
    )
    flange = builder.add_joint_revolute(
        carriage,
        grip,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, -0.1), wp.quat_identity()),
        target_ke=1.0e4 if flange_drive else 0.0,
        target_kd=1.0e2 if flange_drive else 0.0,
    )
    joints = [lift, flange]
    if pinch:
        for side in (-1.0, 1.0):
            finger = builder.add_link(mass=M_FINGER, inertia=wp.mat33(np.eye(3) * 1.0e-4))
            builder.add_shape_box(
                finger,
                xform=wp.transform((0.0, 0.0, -0.08), wp.quat_identity()),
                hx=0.01,
                hy=0.03,
                hz=0.05,
                cfg=shape_cfg,
            )
            joints.append(
                builder.add_joint_prismatic(
                    grip,
                    finger,
                    axis=(side, 0.0, 0.0),
                    parent_xform=wp.transform((side * 0.06, 0.0, -0.03), wp.quat_identity()),
                    target_pos=-0.03,
                    target_ke=2000.0,
                    target_kd=20.0,
                    limit_lower=-0.05,
                    limit_upper=0.05,
                )
            )
    builder.add_articulation(joints)
    box = None
    box_z = None
    if load_box_x is not None:
        box_x, box_z = load_box_x, lift_z - 0.07
    elif pinch:
        box_x, box_z = 0.0, lift_z - 0.21
    if box_z is not None:
        box = builder.add_body(
            xform=wp.transform((box_x, 0.0, box_z), wp.quat_identity()),
            mass=M_BOX,
            inertia=wp.mat33(np.eye(3) * M_BOX * 0.0032 / 6.0),
            lock_inertia=True,
        )
        builder.add_shape_box(box, hx=0.04, hy=0.04, hz=0.04, cfg=shape_cfg)
    model = builder.finalize(device=device)
    return model, {"grip": grip, "lift": lift, "flange": flange, "box": box}


_CONTACT_CONFIGURATIONS = {
    # Patch friction, shared anchors, spin rows and sleeping together.
    "patch_spin_sleep": {
        "solve_mode": "matrix_free",
        "iterations": 32,
        "contact_regularization": 0.01,
        "contact_shared_anchor": True,
        "contact_friction_shared_anchor": True,
        "friction_anchor_beta": 0.2,
        "contact_friction_gap_max": 0.001,
        "enable_joint_velocity_limits": True,
        "torsion_radius": 0.01,
        "contact_torsion_device": True,
        "enable_sleeping": True,
        "articulation_constraint_max": 256,
        "free_body_constraint_max": 256,
    },
    "point_friction": {"solve_mode": "matrix_free", "friction_anchor_beta": 0.0, "articulation_constraint_max": 256},
    "split": {"solve_mode": "split", "articulation_constraint_max": 64},
    "propagation": {
        "solve_mode": "matrix_free",
        "articulated_contact_response": "propagation",
        "articulation_constraint_max": 256,
    },
    "velocity_pass": {"solve_mode": "matrix_free", "velocity_iterations": 4, "articulation_constraint_max": 256},
    "propagation_spin": {
        "solve_mode": "matrix_free",
        "articulated_contact_response": "propagation",
        "torsion_radius": 0.01,
        "contact_torsion_device": True,
        "articulation_constraint_max": 256,
    },
}


def _settled_wrench(test, model, ids, configuration, steps, *, joint_f=None, capture=False):
    """Average the published wrenches over the last 30 of ``steps`` steps."""
    options = dict(_CONTACT_CONFIGURATIONS[configuration])
    set_torsion_radius(model, options.pop("torsion_radius", 0.0))
    if options.get("contact_shared_anchor"):
        with test.assertWarnsRegex(NewtonWarning, "Patch friction selects"):
            solver = SolverSteel(model, **options)
    else:
        solver = SolverSteel(model, **options)
    solver._rigid_body_angular_damping.zero_()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, contact_matching="sticky")
    observables = solver.observables(FLAGS)
    states, control = [model.state(), model.state()], model.control()
    if joint_f is not None:
        values = np.zeros(model.joint_dof_count, dtype=np.float32)
        for joint, value in joint_f.items():
            values[model.joint_qd_start.numpy()[ids[joint]]] = value
        control.joint_f.assign(values)
    contacts = pipeline.contacts()

    def step():
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, CONTACT_DT, observables=observables)
        states.reverse()

    samples = []
    for i in range(steps):
        if capture and i == steps - 31:
            # Two steps per graph keep the state ping-pong in place across replays.
            with wp.ScopedCapture(model.device) as recording:
                step()
                step()
            for _ in range(15):
                wp.capture_launch(recording.graph)
                solver.check_constraint_capacity()
                samples.append(observables.body_parent_f.numpy())
            break
        step()
        if i >= steps - 30:
            samples.append(observables.body_parent_f.numpy())
    return np.mean(samples, axis=0)


def test_parent_wrench_carries_a_load_resting_on_the_gripper(test, device):
    """A box resting off-centre on the held tray reaches the flange as weight plus its moment."""
    box_x = 0.03
    model, ids = _gripper_scene(device, load_box_x=box_x)
    for configuration in _CONTACT_CONFIGURATIONS:
        with test.subTest(configuration=configuration):
            wrench = _settled_wrench(test, model, ids, configuration, 480)
            expected = [0.0, 0.0, (M_GRIP + M_BOX) * G, 0.0, -box_x * M_BOX * G, 0.0]
            np.testing.assert_allclose(wrench[ids["grip"]], expected, atol=2.0e-3)
            np.testing.assert_allclose(wrench[ids["box"]], 0.0, atol=2.0e-3)


def test_parent_wrench_reports_a_commanded_push(test, device):
    """A lift force pressing the tray into the ground reaches the flange as that force plus the carriage weight."""
    push = 20.0
    model, ids = _gripper_scene(device, lift_z=0.1305, lift_drive=False)
    for configuration in ("patch_spin_sleep", "split"):
        with test.subTest(configuration=configuration):
            wrench = _settled_wrench(
                test,
                model,
                ids,
                configuration,
                480,
                joint_f={"lift": -push},
                capture=configuration == "patch_spin_sleep",
            )
            # The four tray corners share the load indeterminately, so only the force is fixed.
            np.testing.assert_allclose(wrench[ids["grip"], :3], [0.0, 0.0, -(push + M_CARRIAGE * G)], atol=2.0e-3)


def test_parent_wrench_carries_a_pinched_load(test, device):
    """Friction-held payload weight reaches the flange; the squeeze stays inside the gripper subtree."""
    model, ids = _gripper_scene(device, pinch=True)
    for configuration in ("point_friction", "propagation"):
        with test.subTest(configuration=configuration):
            wrench = _settled_wrench(test, model, ids, configuration, 720)
            expected = [0.0, 0.0, (M_GRIP + 2.0 * M_FINGER + M_BOX) * G, 0.0, 0.0, 0.0]
            np.testing.assert_allclose(wrench[ids["grip"]], expected, atol=1.0e-2)
            np.testing.assert_allclose(wrench[ids["box"]], 0.0, atol=1.0e-2)


def test_parent_wrench_includes_spin_rows(test, device):
    """A box riding a spinning tray transmits nothing through its free joint once spin-row torques are counted."""
    model, ids = _gripper_scene(device, flange_drive=False, load_box_x=0.0)
    for configuration in ("patch_spin_sleep", "propagation_spin"):
        with test.subTest(configuration=configuration):
            wrench = _settled_wrench(test, model, ids, configuration, 240, joint_f={"flange": 0.3})
            np.testing.assert_allclose(wrench[ids["box"]], 0.0, atol=1.0e-3)
            test.assertAlmostEqual(wrench[ids["grip"], 5], 0.3, delta=1.0e-3)


def test_parent_wrench_excludes_velocity_limit_prescaling(test, device):
    """A disk spun past its velocity limit is rescaled before the dynamics; the rescale transmits no wrench."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    hub = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    hub_joint = builder.add_joint_revolute(-1, hub, axis=(0.0, 0.0, 1.0))
    disk = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag([0.015, 0.015, 0.02])), lock_inertia=True)
    disk_joint = builder.add_joint_revolute(
        hub,
        disk,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.3, 0.0, 0.0), wp.quat_identity()),
        velocity_limit=1.0,
    )
    builder.add_articulation([hub_joint, disk_joint])
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", enable_joint_velocity_limits=True)
    solver._rigid_body_angular_damping.zero_()
    observables = solver.observables(FLAGS)
    state_in, state_out = model.state(), model.state()
    state_in.joint_qd.assign(np.array([0.0, 10.0], dtype=np.float32))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    solver.step(state_in, state_out, None, None, CONTACT_DT, observables=observables)
    test.assertAlmostEqual(float(state_out.joint_qd.numpy()[1]), 1.0, places=4)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), 0.0, atol=1.0e-4)


def test_parent_wrench_includes_loop_closure_reactions(test, device):
    """Loop-closure forces reach both closure links: an undriven joint in a four-bar transmits no axis torque."""
    builder = _build_four_bar()
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1)
    solver._rigid_body_angular_damping.zero_()
    observables = solver.observables(FLAGS)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    for step in range(120):
        state_in = state_0
        solver.step(state_0, state_1, control, None, 1.0 / 240.0, observables=observables)
        state_0, state_1 = state_1, state_0
        if step % 20 == 19:
            wrench = observables.body_parent_f.numpy()
            test.assertGreater(np.abs(wrench).max(), 0.5)
            torques = _revolute_wrench_torques(model, state_in, wrench)
            # Joint 0 is the driven crank; the coupler and rocker joints are undriven.
            for joint in (1, 2):
                test.assertAlmostEqual(torques[joint], 0.0, delta=2.0e-3)


def test_parent_wrench_of_a_load_held_by_closures(test, device):
    """A FREE load hung from the world or a moving kinematic carrier by closures transmits nothing through its free joint."""
    for world_parent in (True, False):
        with test.subTest(world_parent=world_parent):
            builder, carrier, load, _joints = _build_carried_load(world_parent=world_parent)
            model = builder.finalize(device=device)
            solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
            observables = solver.observables(FLAGS)
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            articulation = None
            if not world_parent:
                articulation = wp.array(
                    [int(solver._body_to_articulation.numpy()[carrier])], dtype=wp.int32, device=device
                )
            dt = 1.0 / 240.0
            for i in range(120):
                if articulation is not None:
                    _prescribe_carrier(
                        model,
                        state_0,
                        articulation,
                        i * dt,
                        np.array([0.3, 0.0, 0.1]),
                        np.array([0.0, 0.0, 1.5]),
                    )
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, dt, observables=observables)
                state_0, state_1 = state_1, state_0
            np.testing.assert_allclose(observables.body_parent_f.numpy()[load], 0.0, atol=2.0e-3)


NEAR_REDUNDANT_LEVERS = (0.02, 0.001, 0.0002, 0.00015)


def test_closure_force_uses_the_row_builder_norm(test, device):
    """A closure row whose shared-ancestor terms cancel maps its impulse back with the builder's stored norm."""

    def ints(values):
        return wp.array(values, dtype=wp.int32, device=device)

    for lever in NEAR_REDUNDANT_LEVERS:
        with test.subTest(lever=lever):
            zero = ints([0])
            anchor_xform = wp.array(
                [wp.transform((1.0, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform, device=device
            )
            body_q = wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device)
            # A shared root hinge plus a child hinge whose closure anchor sits `lever` from its axis.
            motion = wp.array(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, -(1.0 - lever), 0.0, 0.0, 0.0, 1.0]],
                dtype=wp.spatial_vector,
                device=device,
            )
            body_to_joint, ancestor, qd_start = ints([0, 1]), ints([-1, 0]), ints([0, 1, 2])
            origin = wp.zeros(1, dtype=wp.vec3, device=device)
            J = wp.zeros((1, 3, 2), dtype=float, device=device)
            row_type = wp.zeros((1, 3), dtype=wp.int32, device=device)
            row_parent = wp.zeros((1, 3), dtype=wp.int32, device=device)
            row_fields = [wp.zeros((1, 3), dtype=float, device=device) for _ in range(3)]
            inv_norm = wp.zeros((1, 3), dtype=float, device=device)
            wp.launch(
                populate_connect_J_for_size,
                dim=1,
                inputs=[
                    zero,  # articulation_dof_start
                    zero,  # art_to_world
                    zero,  # group_to_art
                    2,
                    zero,  # connect_slot
                    zero,  # connect_art
                    zero,  # connect_joint
                    zero,  # connect_body_p
                    ints([1]),  # connect_body_c
                    anchor_xform,
                    anchor_xform,
                    zero,  # connect_parent_prescribed
                    body_q,
                    wp.zeros(2, dtype=wp.spatial_vector, device=device),
                    wp.zeros(2, dtype=wp.vec3, device=device),
                    body_to_joint,
                    ancestor,
                    qd_start,
                    motion,
                    origin,
                ],
                outputs=[J, row_type, row_parent, *row_fields, inv_norm],
                device=device,
            )
            wrench = wp.zeros(2, dtype=wp.spatial_vector, device=device)
            impulses = wp.array([[0.0, 1.0, 0.0]], dtype=wp.float32, device=device)
            wp.launch(
                accumulate_connect_wrenches_from_impulses,
                dim=1,
                inputs=[
                    zero,  # connect_slot
                    zero,  # connect_art
                    zero,  # connect_body_p
                    ints([1]),  # connect_body_c
                    zero,  # connect_joint
                    anchor_xform,
                    anchor_xform,
                    inv_norm,
                    zero,  # art_to_world
                    body_q,
                    ints([0, 0]),  # body_to_articulation
                    origin,
                    ints([3]),  # world_constraint_count
                    row_type,
                    impulses,
                    1.0,
                ],
                outputs=[wrench],
                device=device,
            )
            np.testing.assert_allclose(J.numpy()[0, 1], [0.0, -1.0], atol=1.0e-6)
            float32_lever = 1.0 + float(motion.numpy()[1, 1])
            force = wrench.numpy()[:, 1]
            np.testing.assert_allclose(force, [1.0 / float32_lever, -1.0 / float32_lever], rtol=1.0e-5)


def test_parent_wrench_of_a_near_redundant_closure(test, device):
    """A child hinge braced by a closure anchored just off its axis reports the torque applied to it."""
    torque = 0.01
    for lever in NEAR_REDUNDANT_LEVERS:
        with test.subTest(lever=lever):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            root_joint = builder.add_joint_revolute(-1, root, axis=(0.0, 0.0, 1.0), armature=0.0)
            carrier = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            carrier_joint = builder.add_joint_fixed(root, carrier)
            child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            child_joint = builder.add_joint_revolute(
                carrier,
                child,
                axis=(0.0, 0.0, 1.0),
                parent_xform=wp.transform((1.0 - lever, 0.0, 0.0), wp.quat_identity()),
                armature=0.0,
            )
            builder.add_articulation([root_joint, carrier_joint, child_joint])
            builder.add_joint_ball(
                root,
                child,
                parent_xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform((lever, 0.0, 0.0), wp.quat_identity()),
            )
            model = builder.finalize(device=device)
            solver = SolverSteel(model, solve_mode="matrix_free", iterations=64, cfm=0.0)
            solver._rigid_body_angular_damping.zero_()
            state = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            control = model.control()
            joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
            joint_f[model.joint_qd_start.numpy()[child_joint]] = torque
            control.joint_f.assign(joint_f)
            wrench, _state_out = _first_step(model, state, control, solver)
            test.assertFalse(bool(solver.constraint_overflow.numpy().any()))
            test.assertAlmostEqual(_revolute_wrench_torques(model, state, wrench)[child_joint], torque, delta=1.0e-5)


def test_parent_wrench_warns_without_preeliminated_closures(test, device):
    """Pre-eliminated loop closures have no stored impulses, so requesting the wrench warns once."""
    model = _build_four_bar().finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", enable_bilateral_preelimination=True)
    observables = solver.observables(FLAGS)
    state_0, state_1 = model.state(), model.state()
    with test.assertWarnsRegex(NewtonWarning, "pre-eliminated loop-closure"):
        solver.step(state_0, state_1, None, None, 1.0 / 240.0, observables=observables)
    # Sleeping keeps a wrench history, which is no request.
    sleeping = SolverSteel(model, solve_mode="matrix_free", enable_bilateral_preelimination=True, enable_sleeping=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NewtonWarning)
        sleeping.step(state_0, state_1, None, None, 1.0 / 240.0)


def test_unattached_body_reports_zero(test, device):
    """A body outside every articulation overwrites its row with zero."""
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    builder.add_articulation([builder.add_joint_revolute(-1, root, axis=(0.0, 1.0, 0.0))])
    builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode="split")
    solver._rigid_body_angular_damping.zero_()
    observables = solver.observables(FLAGS)
    observables.body_parent_f.fill_(wp.spatial_vector(123.0, 123.0, 123.0, 123.0, 123.0, 123.0))
    solver.step(model.state(), model.state(), None, None, DT_parent_wrench, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy()[1], 0.0)


def _sphere_on_ground(device, armature):
    """A 1 kg sphere on a free joint with the given FREE armature, resting on frictionless ground."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01)
    )
    builder.add_articulation([builder.add_joint_free(body)])
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
    builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    model = builder.finalize(device=device)
    model.joint_armature.fill_(armature)
    return model


def test_jointless_model_reports_zero(test, device):
    """A model without joints overwrites the requested wrench rows with zero."""
    builder = newton.ModelBuilder()
    body = builder.add_link(is_kinematic=True)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode="split")
    observables = solver.observables(FLAGS)
    observables.body_parent_f.fill_(wp.spatial_vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    solver.step(model.state(), model.state(), None, None, DT_parent_wrench, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy(), 0.0)


ARMATURE_DT = 0.01


def _armature_scene(device, armature):
    """A FREE body, a revolute carrying a welded child, a prismatic and a ball, each joint with the given armature.

    The FREE body starts at rest at its center of mass, so its joint coordinates are its world-frame motion.
    """
    builder = newton.ModelBuilder()
    free = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3) * 0.2)
    )
    builder.add_articulation([builder.add_joint_free(free)])
    arm = builder.add_link(mass=1.0, com=wp.vec3(0.1, 0.0, 0.0), inertia=wp.mat33(np.diag([0.15, 0.2, 0.3])))
    welded = builder.add_link(mass=0.5, com=wp.vec3(0.0, 0.05, 0.0), inertia=wp.mat33(np.eye(3) * 0.01))
    builder.add_articulation(
        [
            builder.add_joint_revolute(
                -1, arm, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform(wp.vec3(1.0, 0.0, 1.0), wp.quat_identity())
            ),
            builder.add_joint_fixed(arm, welded, parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity())),
        ]
    )
    slider = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.1))
    builder.add_articulation(
        [
            builder.add_joint_prismatic(
                -1, slider, axis=(0.0, 0.0, 1.0), parent_xform=wp.transform(wp.vec3(2.0, 0.0, 1.0), wp.quat_identity())
            )
        ]
    )
    pendulum = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, -0.2), inertia=wp.mat33(np.eye(3) * 0.05))
    builder.add_articulation(
        [builder.add_joint_ball(-1, pendulum, parent_xform=wp.transform(wp.vec3(3.0, 0.0, 1.0), wp.quat_identity()))]
    )
    model = builder.finalize(device=device)
    model.joint_armature.fill_(armature)
    state = model.state()
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[6:] = (0.7, -0.2, 0.2, -0.1, 0.3)
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_f = np.zeros((model.body_count, 6), dtype=np.float32)
    body_f[free] = (1.0, -2.0, 10.0, 0.5, -0.3, 0.2)
    state.body_f.assign(body_f)
    control = model.control()
    control.joint_f.assign(np.array([0.0] * 6 + [10.0, 5.0, 1.0, -2.0, 0.5], dtype=np.float32))
    return model, state, control


def _step_armature_scene(model, state, control, solver):
    """Step once; return the published wrenches and the joint accelerations of the step."""
    observables = solver.observables(FLAGS)
    state_out = model.state()
    solver.step(state, state_out, control, None, ARMATURE_DT, observables=observables)
    qdd = (state_out.joint_qd.numpy() - state.joint_qd.numpy()) / ARMATURE_DT
    return observables.body_parent_f.numpy(), qdd


def test_parent_wrench_excludes_joint_armature(test, device):
    """Reconstruct every joint type's wrench without its armature.

    Projected onto its motion subspace, every wrench is the applied joint force less ``armature * qdd``. The FREE
    body has no ``joint_f``, so it reports ``-armature * qdd``, and the weld reports its child's rigid-body wrench.
    """
    configurations = [{"solve_mode": "split"}]
    if wp.get_device(device).is_cuda:
        configurations.append({"solve_mode": "matrix_free"})
    for armature in (0.0, 9.0):
        model, state, control = _armature_scene(device, armature)
        joint_f = control.joint_f.numpy()
        com_world = [
            np.array(wp.transform_point(wp.transform(*X), wp.vec3(*com)))
            for X, com in zip(state.body_q.numpy(), model.body_com.numpy(), strict=True)
        ]
        for options in configurations:
            with test.subTest(armature=armature, **options):
                solver = SolverSteel(model, **options)
                solver._rigid_body_angular_damping.zero_()
                wrench, qdd = _step_armature_scene(model, state, control, solver)
                expected = joint_f - armature * qdd
                np.testing.assert_allclose(wrench[0], expected[:6], rtol=1.0e-4, atol=1.0e-4)
                if armature:
                    test.assertGreater(np.abs(wrench[0]).max(), 1.0)
                hinge = np.array([1.0, 0.0, 1.0])
                hinge_torque = wrench[1, 3:] + np.cross(com_world[1] - hinge, wrench[1, :3])
                test.assertAlmostEqual(
                    float(hinge_torque[1]), float(expected[6]), delta=1.0e-4 * abs(expected[6]) + 1.0e-4
                )
                test.assertAlmostEqual(
                    float(wrench[3, 2]), float(expected[7]), delta=1.0e-4 * abs(expected[7]) + 1.0e-4
                )
                socket = np.array([3.0, 0.0, 1.0])
                ball_torque = wrench[4, 3:] + np.cross(com_world[4] - socket, wrench[4, :3])
                np.testing.assert_allclose(ball_torque, expected[8:11], rtol=1.0e-4, atol=1.0e-4)
                # The weld carries its child's mass times its acceleration from the hinge's start velocity and qdd.
                omega = np.array([0.0, state.joint_qd.numpy()[6], 0.0])
                alpha = np.array([0.0, qdd[6], 0.0])
                lever = com_world[2] - hinge
                accel = np.cross(alpha, lever) + np.cross(omega, np.cross(omega, lever))
                weld_force = model.body_mass.numpy()[2] * (accel - np.array([0.0, 0.0, -9.81]))
                np.testing.assert_allclose(wrench[2, :3], weld_force, rtol=1.0e-4, atol=1.0e-4)
                np.testing.assert_allclose(wrench[2, 3:], 0.01 * alpha, rtol=1.0e-4, atol=1.0e-4)


def test_parent_wrench_of_a_contact_stop_with_armature(test, device, solve_mode="split"):
    """Report a contact-stopped FREE body's armature share of the impact as its joint wrench.

    The contact impulse stops the armature-augmented inertia; the body's own mass takes ``m * dv / dt`` of it and
    the FREE joint reports the remaining ``-armature * dv / dt``.
    """
    for armature in (0.0, 9.0):
        with test.subTest(armature=armature):
            model = _sphere_on_ground(device, armature)
            solver = SolverSteel(model, solve_mode=solve_mode, cfm=0.0)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            observables = solver.observables(FLAGS | {SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            state_in, state_out = model.state(), model.state()
            qd = state_in.joint_qd.numpy()
            qd[2] = -1.0
            state_in.joint_qd.assign(qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, ARMATURE_DT, observables=observables)
            count = int(contacts.rigid_contact_count.numpy()[0])
            contact_z = abs(float(observables.contact_f.numpy()[:count, 2].sum()))
            accel_z = float(state_out.joint_qd.numpy()[2] - qd[2]) / ARMATURE_DT
            wrench = observables.body_parent_f.numpy()[0]
            test.assertGreater(accel_z, 90.0)
            test.assertAlmostEqual(float(wrench[2]), -armature * accel_z, delta=1.0e-4 * armature * accel_z + 1.0e-3)
            np.testing.assert_allclose(wrench[[0, 1, 3, 4, 5]], 0.0, atol=1.0e-3)
            mass = float(model.body_mass.numpy()[0])
            test.assertAlmostEqual(contact_z + float(wrench[2]), mass * accel_z, delta=1.0e-3 * mass * accel_z)


def test_parent_wrench_of_a_commanded_free_joint(test, device):
    """Report a FREE joint's own generalized command less its armature reaction.

    A 1 kg, unit-inertia body at rest driven through ``joint_f`` by 10 N and 10 N·m accelerates at
    ``10 / (1 + armature)``, so it reports ``10 - armature * qdd``: 1 with armature 9 and 10 without.
    """
    for armature, expected in ((0.0, 10.0), (9.0, 1.0)):
        with test.subTest(armature=armature):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.add_articulation([builder.add_joint_free(body)])
            model = builder.finalize(device=device)
            model.joint_armature.fill_(armature)
            state = model.state()
            control = model.control()
            control.joint_f.assign(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32))
            solver = SolverSteel(model, solve_mode="split")
            solver._rigid_body_angular_damping.zero_()
            wrench, qdd = _step_armature_scene(model, state, control, solver)
            np.testing.assert_allclose(qdd[[0, 5]], 10.0 / (1.0 + armature), rtol=1.0e-5)
            np.testing.assert_allclose(wrench[0], [expected, 0.0, 0.0, 0.0, 0.0, expected], rtol=1.0e-4, atol=1.0e-4)


def test_parent_wrench_of_a_held_kinematic_joint(test, device):
    """Report the support load of a held kinematic hinge, whatever its armature.

    The 1 kg hinge body sits on the axis and carries a welded 1 kg child 1 m out, so the hinge holds 2 g of
    weight and the child's -g moment about its axis, with zero command and zero acceleration.
    """
    for armature in (0.0, 9.0):
        with test.subTest(armature=armature):
            builder = newton.ModelBuilder()
            hinge = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=True)
            child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.add_articulation(
                [
                    builder.add_joint_revolute(-1, hinge, axis=(0.0, 1.0, 0.0)),
                    builder.add_joint_fixed(
                        hinge, child, parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())
                    ),
                ]
            )
            model = builder.finalize(device=device)
            model.joint_armature.fill_(armature)
            solver = SolverSteel(model, solve_mode="split")
            solver._rigid_body_angular_damping.zero_()
            wrench, qdd = _step_armature_scene(model, model.state(), model.control(), solver)
            test.assertEqual(float(qdd[0]), 0.0)
            np.testing.assert_allclose(wrench[0], [0.0, 0.0, 2.0 * 9.81, 0.0, -9.81, 0.0], rtol=1.0e-5, atol=1.0e-4)


def test_parent_wrench_matches_mujoco_with_armature(test, device):
    """Publish the same joint wrenches as SolverMuJoCo on one step of the armature scene, with and without armature."""
    for armature in (0.0, 9.0):
        with test.subTest(armature=armature):
            model, state, control = _armature_scene(device, armature)
            solver = SolverSteel(model, solve_mode="matrix_free")
            solver._rigid_body_angular_damping.zero_()
            steel, _ = _step_armature_scene(model, state, control, solver)
            mujoco, _ = _step_armature_scene(
                model, state, control, newton.solvers.SolverMuJoCo(model, integrator="euler")
            )
            test.assertGreater(np.abs(mujoco).max(), 1.0)
            np.testing.assert_allclose(steel, mujoco, rtol=1.0e-4, atol=1.0e-4)


def test_parent_wrench_of_an_arm_resting_on_a_joint_limit(test, device):
    """A horizontal arm held up by its hinge's limit row reports its full weight, and no axis torque about the COM."""
    builder = newton.ModelBuilder()
    arm = builder.add_link(mass=1.0, com=wp.vec3(0.5, 0.0, 0.0), inertia=wp.mat33(np.eye(3) * 0.01))
    builder.add_articulation(
        [builder.add_joint_revolute(-1, arm, axis=(0.0, 1.0, 0.0), limit_lower=-1.0, limit_upper=0.0, armature=0.0)]
    )
    model = builder.finalize(device=device)
    solve_mode = "matrix_free" if wp.get_device(device).is_cuda else "split"
    solver = SolverSteel(model, solve_mode=solve_mode, enable_joint_limits=True, iterations=32)
    solver._rigid_body_angular_damping.zero_()
    observables = solver.observables(FLAGS)
    states, control = [model.state(), model.state()], model.control()
    for _ in range(60):
        solver.step(states[0], states[1], control, None, DT_parent_wrench, observables=observables)
        states.reverse()
    test.assertAlmostEqual(float(states[0].joint_q.numpy()[0]), 0.0, delta=1.0e-3)
    np.testing.assert_allclose(observables.body_parent_f.numpy()[arm], [0.0, 0.0, 9.81, 0.0, 0.0, 0.0], atol=2.0e-3)


class TestSteelParentWrench(unittest.TestCase):
    pass


add_function_test(
    TestSteelParentWrench,
    "test_parent_wrench_eager_step_before_first_replay",
    test_parent_wrench_eager_step_before_first_replay,
    devices=get_cuda_test_devices(),
)
for _name in (
    "test_unattached_body_reports_zero",
    "test_jointless_model_reports_zero",
    "test_closure_force_uses_the_row_builder_norm",
    "test_parent_wrench_matches_inverse_dynamics_on_every_path",
    "test_parent_wrench_excludes_joint_armature",
    "test_parent_wrench_of_a_contact_stop_with_armature",
    "test_parent_wrench_of_a_commanded_free_joint",
    "test_parent_wrench_of_a_held_kinematic_joint",
    "test_parent_wrench_of_an_arm_resting_on_a_joint_limit",
):
    add_function_test(TestSteelParentWrench, _name, globals()[_name], devices=get_test_devices())
add_function_test(
    TestSteelParentWrench,
    "test_parent_wrench_of_a_contact_stop_with_armature_matrix_free",
    test_parent_wrench_of_a_contact_stop_with_armature,
    devices=get_cuda_test_devices(),
    solve_mode="matrix_free",
)
for _name in (
    "test_parent_wrench_of_a_near_redundant_closure",
    "test_parent_wrench_matches_mujoco_with_armature",
    "test_parent_wrench_excludes_velocity_limit_prescaling",
    "test_parent_wrench_includes_loop_closure_reactions",
    "test_parent_wrench_of_a_load_held_by_closures",
    "test_parent_wrench_warns_without_preeliminated_closures",
):
    add_function_test(TestSteelParentWrench, _name, globals()[_name], devices=get_cuda_test_devices())


class TestSteelParentWrenchGripper(unittest.TestCase):
    pass


for _name in (
    "test_parent_wrench_carries_a_load_resting_on_the_gripper",
    "test_parent_wrench_reports_a_commanded_push",
    "test_parent_wrench_carries_a_pinched_load",
    "test_parent_wrench_includes_spin_rows",
):
    add_function_test(TestSteelParentWrenchGripper, _name, globals()[_name], devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
