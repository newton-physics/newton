# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
import hashlib
import os
import re
import subprocess
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.tests.unittest_utils
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    _get_tree_fk_kernel,
    _get_tree_tau_kernel,
    crba_fill_par_dof,
)
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import _build_four_bar
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_cuda_test_devices, get_test_devices

RUN_TO_RUN = wp.DeterministicMode.RUN_TO_RUN

# Free root, then two ball joints: every joint has several DOFs.
JOINT_DOF_DIMS = ((3, 3), (0, 3), (0, 3))
CRBA_ARTICULATIONS = 256
CRBA_LAUNCHES = 8

# Drive, joint-limit and velocity-limit rows of two floating ants per world, plus free boxes.
MATRIX_FREE_OPTIONS = {
    "drive_mode": "rows",
    "enable_joint_limits": True,
    "enable_joint_velocity_limits": True,
    "iterations": 8,
    "articulation_constraint_max": 256,
    "free_body_constraint_max": 256,
    # Rows that do not fit are dropped deterministically too, so overflow stays in scope.
    "warn_constraint_overflow": False,
}

SPLIT_OPTIONS = {
    "solve_mode": "split",
    "enable_joint_limits": True,
    "articulation_constraint_max": 128,
    "free_body_constraint_max": 256,
    "warn_constraint_overflow": False,
}

# Device-prepared contact torsion rows follow the contact rows of their world.
TORSION_OPTIONS = {**MATRIX_FREE_OPTIONS, "contact_torsion_device": True, "articulation_constraint_max": 512}

# Joint-friction rows alongside the contact, drive, joint-limit and velocity-limit rows.
FRICTION_OPTIONS = {**MATRIX_FREE_OPTIONS, "enable_joint_friction": True}

# Warm-started impulses are matched through contact matching and carried across steps.
WARMSTART_OPTIONS = {**MATRIX_FREE_OPTIONS, "contact_warmstart": True, "velocity_iterations": 2}

# Ants alone select the sparse mass factors and their one-warp joint-limit builder.
SPARSE_OPTIONS = {
    "iterations": 8,
    "enable_joint_limits": True,
    "articulation_constraint_max": 256,
    "friction_anchor_beta": 0.0,
    "warn_constraint_overflow": False,
}


def test_crba_fill_par_dof_is_batch_invariant(test, device):
    """Give identical articulations bitwise-identical, exactly symmetric mass matrices on every launch."""
    joint_count = len(JOINT_DOF_DIMS)
    dof_count = sum(lin + ang for lin, ang in JOINT_DOF_DIMS)
    rng = np.random.default_rng(7)
    # Generic motion subspaces and inertias, so S_r.(I S_c) and S_c.(I S_r) round differently.
    motion = rng.normal(size=(dof_count, 6)).astype(np.float32)
    inertia = rng.normal(size=(joint_count, 6, 6))
    inertia = (inertia + np.swapaxes(inertia, 1, 2)).astype(np.float32)

    n = CRBA_ARTICULATIONS
    joint_qd_start = np.concatenate(([0], np.cumsum([lin + ang for lin, ang in JOINT_DOF_DIMS] * n))).astype(np.int32)
    joint_ancestor = np.array([j - 1 if j % joint_count else -1 for j in range(n * joint_count)], dtype=np.int32)
    inputs = [
        wp.array(np.arange(n + 1, dtype=np.int32) * joint_count, dtype=wp.int32, device=device),
        wp.array(np.arange(n, dtype=np.int32) * dof_count, dtype=wp.int32, device=device),
        wp.ones(n, dtype=wp.int32, device=device),
        wp.array(joint_ancestor, dtype=wp.int32, device=device),
        wp.array(np.arange(n * joint_count, dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(joint_qd_start, dtype=wp.int32, device=device),
        wp.array(np.tile(np.array(JOINT_DOF_DIMS, dtype=np.int32), (n, 1)), dtype=wp.int32, device=device),
        wp.array(np.tile(motion, (n, 1)), dtype=wp.spatial_vector, device=device),
        wp.array(np.tile(inertia, (n, 1, 1)), dtype=wp.spatial_matrix, device=device),
        wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=device),
        dof_count,
    ]

    reference = None
    for _launch in range(CRBA_LAUNCHES):
        H = wp.zeros((n, dof_count, dof_count), dtype=wp.float32, device=device)
        wp.launch(crba_fill_par_dof, dim=n * dof_count, inputs=inputs, outputs=[H], device=device, block_dim=128)
        H = H.numpy()
        np.testing.assert_array_equal(H, np.swapaxes(H, 1, 2), err_msg="H is not exactly symmetric")
        np.testing.assert_array_equal(H, np.broadcast_to(H[0], H.shape), err_msg="identical articulations differ")
        if reference is None:
            reference = H
        np.testing.assert_array_equal(H, reference, err_msg="H differs between launches")


def _build_pendulum_model(device):
    """Three-link pendulum in two worlds: its mass matrix changes every step."""
    env = newton.ModelBuilder()
    parent = -1
    joints = []
    for link_index in range(3):
        link = env.add_link()
        env.add_shape_box(link, hx=0.2, hy=0.04, hz=0.04)
        joints.append(
            env.add_joint_revolute(
                parent,
                link,
                parent_xform=wp.transform(wp.vec3(0.2 if parent >= 0 else 0.0, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.Y if link_index != 1 else newton.Axis.Z,
            )
        )
        parent = link
    env.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(env, 2)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.tile(np.array([0.7, -0.4, 0.3], dtype=np.float32), 2))
    return model


def _simulate(model, solver, steps):
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
    return state_0.body_q.numpy(), state_0.body_qd.numpy()


def test_full_reset_matches_fresh_solver(test, device):
    """Replay a fresh solver's trajectory bit for bit with eager steps after a full reset."""
    for interval in (1, 3):
        with test.subTest(update_mass_matrix_interval=interval):
            model = _build_pendulum_model(device)
            options = {"solve_mode": "split", "update_mass_matrix_interval": interval}
            reused = SolverSteel(model, **options)
            _simulate(model, reused, 5)
            reused.reset(model.state())
            body_q, body_qd = _simulate(model, reused, 7)

            fresh_body_q, fresh_body_qd = _simulate(model, SolverSteel(model, **options), 7)
            np.testing.assert_array_equal(body_q, fresh_body_q)
            np.testing.assert_array_equal(body_qd, fresh_body_qd)


def _build_contact_scene(device, worlds=4, boxes=6, ants=2, torsion_radius=0.0, joint_friction=0.0):
    """Floating ants dropped onto each other and onto a pile of free boxes in every world.

    A positive ``joint_friction`` adds a floating prismatic, ball and D6 chain and sets it on every non-free DOF.
    """
    env = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(env)
    for ant in range(ants):
        env.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            xform=wp.transform(wp.vec3(0.1 * ant, 0.05 * ant, 0.6 + 0.5 * ant), wp.quat_identity()),
            floating=True,
            ignore_names=["floor"],
        )
    rng = np.random.default_rng(3)
    for box in range(boxes):
        body = env.add_body(
            xform=wp.transform(
                wp.vec3(
                    0.13 * (box % 3) + 0.01 * rng.standard_normal(), 0.13 * ((box // 3) % 2), 0.07 + 0.13 * (box // 6)
                ),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(0.0, 3.0))),
            )
        )
        env.add_shape_box(body, hx=0.06, hy=0.05, hz=0.06, cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.7))
    if joint_friction > 0.0:
        _add_mixed_chain(env, wp.vec3(-0.2, -1.0, 0.12))
    builder = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(builder)
    builder.replicate(env, worlds)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.joint_velocity_limit.fill_(8.0)
    if torsion_radius > 0.0:
        model.steel.min_torsional_patch_radius.fill_(torsion_radius)
    if joint_friction > 0.0:
        friction = np.zeros(model.joint_dof_count, dtype=np.float32)
        joint_dofs = np.repeat(model.joint_type.numpy(), np.diff(model.joint_qd_start.numpy()))
        friction[joint_dofs != newton.JointType.FREE] = joint_friction
        model.joint_friction.assign(friction)
    return model


def _add_mixed_chain(builder, position):
    """Add a floating chain of prismatic, ball and D6 joints with limits on its linear axes."""
    dof = newton.ModelBuilder.JointDofConfig
    links = []
    for index in range(4):
        link = builder.add_link(xform=wp.transform(position + wp.vec3(0.12 * index, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.05, hy=0.04, hz=0.03, cfg=newton.ModelBuilder.ShapeConfig(density=400.0))
        links.append(link)
    offset = wp.transform(wp.vec3(0.12, 0.0, 0.0), wp.quat_identity())
    joints = [
        builder.add_joint_free(links[0]),
        builder.add_joint_prismatic(
            links[0], links[1], parent_xform=offset, axis=newton.Axis.X, limit_lower=-0.04, limit_upper=0.04
        ),
        builder.add_joint_ball(links[1], links[2], parent_xform=offset),
        builder.add_joint_d6(
            links[2],
            links[3],
            parent_xform=offset,
            linear_axes=[dof(axis=newton.Axis.Z, limit_lower=-0.03, limit_upper=0.03)],
            angular_axes=[dof(axis=newton.Axis.X), dof(axis=newton.Axis.Y)],
        ),
    ]
    builder.add_articulation(joints)


def _actuate(model, control):
    torques = np.zeros(model.joint_dof_count, dtype=np.float32)
    torques[np.arange(model.joint_dof_count) % 14 >= 6] = 30.0
    # The mixed chain is unactuated.
    dof_counts = np.diff(model.joint_qd_start.numpy())
    dof_type = np.repeat(model.joint_type.numpy(), dof_counts)
    dof_articulation = np.repeat(model.joint_articulation.numpy(), dof_counts)
    chain = (newton.JointType.PRISMATIC, newton.JointType.BALL, newton.JointType.D6)
    torques[np.isin(dof_articulation, dof_articulation[np.isin(dof_type, chain)])] = 0.0
    control.joint_f.assign(torques)


def _sample(state):
    """Hash of the state and whether every value in it is finite."""
    digest = hashlib.sha256()
    finite = True
    for array in (state.body_q, state.body_qd, state.joint_q, state.joint_qd):
        values = np.ascontiguousarray(array.numpy())
        digest.update(values.tobytes())
        finite = finite and bool(np.isfinite(values).all())
    return digest.hexdigest(), finite


def _make_pipeline(model, options):
    if options.get("contact_warmstart"):
        return newton.CollisionPipeline(model, contact_matching="latest")
    return newton.CollisionPipeline(model, deterministic=True)


def _run_hashes(model, options, steps, graph=False):
    """Simulate with deterministic collision and solver; return (hash, finite) per step or per graph replay."""
    solver = SolverSteel(model, deterministic=RUN_TO_RUN, **options)
    pipeline = _make_pipeline(model, options)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    _actuate(model, control)

    def step(state_in, state_out):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 0.005)

    hashes = []
    if graph:
        with wp.ScopedCapture(device=model.device) as capture:
            step(state_0, state_1)
            step(state_1, state_0)
        for _ in range(steps // 2):
            wp.capture_launch(capture.graph)
            hashes.append(_sample(state_0))
        return hashes
    for _ in range(steps):
        step(state_0, state_1)
        state_0, state_1 = state_1, state_0
        hashes.append(_sample(state_0))
    return hashes


def test_deterministic_runs_are_bitwise_identical(test, device):
    """Produce identical finite states in two runs at every eager step and every two-step CUDA graph boundary."""
    model = _build_contact_scene(device)
    configurations = [("split", model, SPLIT_OPTIONS, False)]
    if wp.get_device(device).is_cuda:
        torsion_model = _build_contact_scene(device, torsion_radius=0.02)
        friction_model = _build_contact_scene(device, joint_friction=0.4)
        configurations += [
            ("split_graph", model, SPLIT_OPTIONS, True),
            ("matrix_free", model, MATRIX_FREE_OPTIONS, False),
            ("matrix_free_graph", model, MATRIX_FREE_OPTIONS, True),
            ("torsion_graph", torsion_model, TORSION_OPTIONS, True),
            ("warmstart_graph", model, WARMSTART_OPTIONS, True),
            ("joint_friction", friction_model, FRICTION_OPTIONS, False),
            ("joint_friction_graph", friction_model, FRICTION_OPTIONS, True),
        ]
    for name, scene, options, graph in configurations:
        with test.subTest(name):
            first = _run_hashes(scene, options, 60, graph=graph)
            second = _run_hashes(scene, options, 60, graph=graph)
            test.assertEqual(first, second)
            test.assertTrue(all(finite for _, finite in first))


def test_warm_started_full_reset_matches_fresh_solver(test, device):
    """Replay a fresh warm-started solver's states and contact forces bit for bit after a full reset."""
    model = _build_contact_scene(device, worlds=2, boxes=3, ants=1)

    def simulate(solver, steps):
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        observables = solver.observables({SolverSteel.ObservableKind.CONTACT_F}, contacts=contacts)
        state_0, state_1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        _actuate(model, control)
        for _ in range(steps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 0.005, observables=observables)
            state_0, state_1 = state_1, state_0
        return _sample(state_0), observables.contact_f.numpy()

    # A refresh interval that does not divide the pre-reset step count exposes a stale step cadence.
    options = {**WARMSTART_OPTIONS, "update_mass_matrix_interval": 3}
    reused = SolverSteel(model, deterministic=RUN_TO_RUN, **options)
    simulate(reused, 5)
    reused.reset(model.state())
    (state_hash, finite), contact_f = simulate(reused, 7)
    (fresh_hash, _), fresh_contact_f = simulate(SolverSteel(model, deterministic=RUN_TO_RUN, **options), 7)
    test.assertTrue(finite)
    test.assertEqual(state_hash, fresh_hash)
    test.assertGreater(np.abs(contact_f).max(), 0.0)
    np.testing.assert_array_equal(contact_f, fresh_contact_f)


def _struct_arrays(value):
    """Arrays passed directly or as fields of a Warp struct."""
    if isinstance(value, wp.array):
        return [value]
    fields = getattr(getattr(value, "_cls", None), "vars", {})
    return [getattr(value, name) for name in fields if isinstance(getattr(value, name), wp.array)]


def _replaying(original, replays, varied):
    """Wrap a launch function: rerun every launch from a snapshot of its arrays and record kernels whose outputs vary."""

    def launch(kernel, dim, inputs=(), outputs=(), **kwargs):
        arrays = [a for v in (*(inputs or ()), *(outputs or ())) for a in _struct_arrays(v) if a.size > 0 and a.ptr]
        wp.synchronize()
        snapshot = [wp.clone(a) for a in arrays]
        reference = None
        result = None
        for replay in range(replays):
            if replay:
                for array, saved in zip(arrays, snapshot, strict=True):
                    wp.copy(array, saved)
            result = original(kernel, dim=dim, inputs=inputs, outputs=outputs, **kwargs)
            wp.synchronize()
            values = [a.numpy().tobytes() for a in arrays]
            if reference is None:
                reference = values
            elif values != reference:
                varied.add(kernel.key)
        return result

    return launch


def test_deterministic_launches_replay_identically(test, device):
    """Reproduce the outputs of every solver launch bitwise when it is rerun from the same inputs.

    The contact radix sort runs outside this interception.
    """
    model = _build_contact_scene(device, worlds=8)
    torsion_model = _build_contact_scene(device, worlds=8, torsion_radius=0.02)
    ant_model = _build_contact_scene(device, worlds=8, boxes=0)
    friction_model = _build_contact_scene(device, worlds=8, joint_friction=0.4)
    for name, scene, options in (
        ("matrix_free", model, MATRIX_FREE_OPTIONS),
        ("split", model, SPLIT_OPTIONS),
        ("torsion", torsion_model, TORSION_OPTIONS),
        ("warmstart", model, WARMSTART_OPTIONS),
        ("sparse", ant_model, SPARSE_OPTIONS),
        ("joint_friction", friction_model, FRICTION_OPTIONS),
    ):
        with test.subTest(name):
            solver = SolverSteel(scene, deterministic=RUN_TO_RUN, **options)
            test.assertEqual(solver._sparse_mass_matrix_size is not None, name == "sparse")
            pipeline = _make_pipeline(scene, options)
            contacts = pipeline.contacts()
            state_0, state_1, control = scene.state(), scene.state(), scene.control()
            newton.eval_fk(scene, scene.joint_q, scene.joint_qd, state_0)
            _actuate(scene, control)
            varied = set()
            for _ in range(12):
                state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                with (
                    mock.patch.object(wp, "launch", _replaying(wp.launch, 3, varied)),
                    mock.patch.object(wp, "launch_tiled", _replaying(wp.launch_tiled, 3, varied)),
                ):
                    solver.step(state_0, state_1, control, contacts, 0.005)
                state_0, state_1 = state_1, state_0
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            test.assertEqual(varied, set())


def _route_contacts(model, state, contacts, **options):
    """Step once with ``options`` and return the contact routing and capacity status."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    solver = SolverSteel(model, deterministic=RUN_TO_RUN, warn_constraint_overflow=False, **options)
    solver.step(state, model.state(), model.control(), contacts, 0.005)
    return (
        solver._contact_world.numpy()[:count],
        solver._contact_slot.numpy()[:count],
        solver._contact_path.numpy()[:count],
        solver._contact_slots_needed.numpy()[:count],
        solver._row_dropped_mf.numpy(),
        solver.constraint_overflow.numpy(),
    )


def test_deterministic_overflow_keeps_earliest_contacts(test, device):
    """Keep a contact-order prefix of each overflowing world's rows and report the rest as dropped."""
    model = _build_contact_scene(device, worlds=4, boxes=12, ants=0)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline.collide(state, contacts)

    world, _, path, needed, _, _ = _route_contacts(
        model, state, contacts, solve_mode="split", free_body_constraint_max=4096
    )
    small_world, slot, small_path, _, dropped, overflow = _route_contacts(
        model, state, contacts, solve_mode="split", free_body_constraint_max=24
    )
    for w in range(model.world_count):
        routed = np.nonzero((path == 1) & (world == w))[0]
        kept = np.nonzero((small_path == 1) & (small_world == w))[0]
        test.assertGreater(len(routed), len(kept))
        np.testing.assert_array_equal(kept, routed[: len(kept)])
        np.testing.assert_array_equal(slot[kept], np.concatenate(([0], np.cumsum(needed[kept])[:-1])))
        test.assertEqual(dropped[w], int(needed[routed[len(kept) :]].sum()))
        test.assertTrue(overflow[w])


def test_overflow_warning_names_deterministic_mode(test, device):
    """Print one row-overflow warning line; in deterministic mode it ends with the drop rule, which holds."""
    model = _build_contact_scene(device, worlds=2, boxes=12, ants=0)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline.collide(state, contacts)
    warning = r"Warning: Steel matrix-free constraint-row overflow in world \d+: requested \d+ rows, "
    warning += r"limit 24; dropped \d+ contact/friction rows\. Increase free_body_constraint_max\."
    note = (
        " Deterministic mode: each world keeps the rows of its earliest contacts in contact-buffer order. "
        "While its contacts and joint rows are unchanged it drops the same contacts every step; otherwise it drops "
        "the tail of the new contact order."
    )
    # Warp CPU printf capture is unreliable on Windows; the overflow status and routing are still checked there.
    check_output = not (wp.get_device(device).is_cpu and sys.platform == "win32")
    for deterministic in (wp.DeterministicMode.NOT_GUARANTEED, RUN_TO_RUN):
        with test.subTest(deterministic=deterministic.name):
            solver = SolverSteel(model, deterministic=deterministic, solve_mode="split", free_body_constraint_max=24)
            capture = StdOutCapture()
            if check_output:
                capture.begin()
            try:
                paths = []
                for _ in range(2):
                    solver.step(state, model.state(), model.control(), contacts, 0.005)
                    paths.append(solver._contact_path.numpy())
                wp.synchronize_device(device)
            finally:
                output = capture.end() if check_output else ""
            if check_output:
                line = warning + (re.escape(note) if deterministic == RUN_TO_RUN else "") + r"\n"
                test.assertEqual(len(re.findall(line, output)), 1, output)
                test.assertEqual(output.count(note), int(deterministic == RUN_TO_RUN), output)
            test.assertTrue(solver.constraint_overflow.numpy()[: model.world_count].all())
            if deterministic == RUN_TO_RUN:
                np.testing.assert_array_equal(paths[0], paths[1], err_msg="same contacts dropped differently")


def test_deterministic_joint_rows_follow_articulation_order(test, device):
    """Give a world's drive, joint-limit, joint-friction and velocity-limit rows their slots in articulation order."""
    arts_per_world = 40
    env = newton.ModelBuilder()
    for art in range(arts_per_world):
        link = env.add_link(xform=wp.transform(wp.vec3(0.5 * art, 0.0, 1.0), wp.quat_identity()))
        env.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
        joint = env.add_joint_revolute(
            -1,
            link,
            parent_xform=wp.transform(wp.vec3(0.5 * art, 0.0, 1.0), wp.quat_identity()),
            axis=newton.Axis.Y,
            target_ke=10.0,
            limit_lower=-0.1,
            limit_upper=0.1,
            velocity_limit=5.0,
        )
        env.add_articulation([joint])
    builder = newton.ModelBuilder()
    builder.replicate(env, 2)
    model = builder.finalize(device=device)
    # Every joint violates its upper limit by a different amount, so limit rows are told apart by phi.
    q = 0.2 + 0.001 * np.arange(model.joint_coord_count, dtype=np.float32)
    model.joint_q.assign(q)
    # Distinct coefficients tell the friction rows apart by their bound.
    friction = 1.0 + 0.01 * np.arange(model.joint_dof_count, dtype=np.float32)
    model.joint_friction.assign(friction)
    solver = SolverSteel(
        model,
        deterministic=RUN_TO_RUN,
        drive_mode="rows",
        enable_joint_limits=True,
        joint_limit_activation_gap=0.0,
        enable_joint_velocity_limits=True,
        fuse_joint_velocity_limits=False,
        enable_joint_friction=True,
        articulation_constraint_max=5 * arts_per_world,
    )
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver.step(state, model.state(), model.control(), None, 0.005)

    order = np.arange(arts_per_world)
    drive_slot = solver._drive_slot.numpy().reshape(2, arts_per_world)
    velocity_limit_slot = solver._velocity_limit_slot.numpy().reshape(2, arts_per_world, 2)
    row_type = solver._row_type.numpy()
    phi = solver._phi.numpy()
    row_mu = solver._row_mu.numpy()
    for world in range(2):
        np.testing.assert_array_equal(drive_slot[world], order)
        limit_rows = np.nonzero(row_type[world] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)[0]
        np.testing.assert_array_equal(limit_rows, arts_per_world + order)
        np.testing.assert_allclose(phi[world, limit_rows], 0.1 - q.reshape(2, arts_per_world)[world], atol=1.0e-6)
        friction_rows = np.nonzero(row_type[world] == PGS_CONSTRAINT_TYPE_JOINT_FRICTION)[0]
        np.testing.assert_array_equal(friction_rows, 2 * arts_per_world + order)
        np.testing.assert_allclose(
            row_mu[world, friction_rows], 0.005 * friction.reshape(2, arts_per_world)[world], rtol=1.0e-6
        )
        np.testing.assert_array_equal(
            velocity_limit_slot[world].reshape(-1), 3 * arts_per_world + np.arange(2 * arts_per_world)
        )


def test_deterministic_joint_friction_rows_follow_dof_order(test, device):
    """Give a world's joint-friction rows of revolute, prismatic, ball and D6 joints their slots in DOF order."""
    arts_per_world = 48
    dof = newton.ModelBuilder.JointDofConfig
    env = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for art in range(arts_per_world):
        link = env.add_link(xform=wp.transform(wp.vec3(0.5 * art, 0.0, 1.0), wp.quat_identity()))
        env.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
        anchor = wp.transform(wp.vec3(0.5 * art, 0.0, 1.0), wp.quat_identity())
        kind = art % 4
        if kind == 0:
            joint = env.add_joint_revolute(-1, link, parent_xform=anchor, axis=newton.Axis.Y)
        elif kind == 1:
            joint = env.add_joint_prismatic(-1, link, parent_xform=anchor, axis=newton.Axis.X)
        elif kind == 2:
            joint = env.add_joint_ball(-1, link, parent_xform=anchor)
        else:
            joint = env.add_joint_d6(
                -1,
                link,
                parent_xform=anchor,
                linear_axes=[dof(axis=newton.Axis.Z)],
                angular_axes=[dof(axis=newton.Axis.X)],
            )
        env.add_articulation([joint])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(env, 2)
    model = builder.finalize(device=device)
    friction = 1.0 + 0.01 * np.arange(model.joint_dof_count, dtype=np.float32)
    model.joint_friction.assign(friction)
    solver = SolverSteel(model, deterministic=RUN_TO_RUN, enable_joint_friction=True, articulation_constraint_max=128)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver.step(state, model.state(), model.control(), None, 0.005)

    world_dofs = model.joint_dof_count // 2
    row_type = solver._row_type.numpy()
    row_mu = solver._row_mu.numpy()
    for world in range(2):
        friction_rows = np.nonzero(row_type[world] == PGS_CONSTRAINT_TYPE_JOINT_FRICTION)[0]
        np.testing.assert_array_equal(friction_rows, np.arange(world_dofs))
        expected = 0.005 * friction[world * world_dofs : (world + 1) * world_dofs]
        np.testing.assert_allclose(row_mu[world, friction_rows], expected, rtol=1.0e-6)


def test_deterministic_bilateral_rows_follow_element_order(test, device):
    """Give a world's mimic rows, then its loop-closure rows, their slots in element order."""
    mimic_count = 512
    builder = _build_four_bar()
    # Many rows on one follower need the deprecated API; joint-owned mimics allow one per joint.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for _ in range(mimic_count):
            builder.add_constraint_mimic(joint0=1, joint1=0, coef0=0.0, coef1=1.0)
    model = builder.finalize(device=device)
    solver = SolverSteel(model, deterministic=RUN_TO_RUN, articulation_constraint_max=mimic_count + 64)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver.step(state, model.state(), model.control(), None, 0.005)

    mimic_slot = solver._mimic_slot.numpy()
    np.testing.assert_array_equal(mimic_slot, mimic_slot[0] + np.arange(mimic_count))
    np.testing.assert_array_equal(solver._connect_slot.numpy(), [mimic_slot[0] + mimic_count])


@contextlib.contextmanager
def _warp_lowering_off():
    saved = wp.config.deterministic
    wp.config.deterministic = wp.DeterministicMode.NOT_GUARANTEED
    try:
        yield
    finally:
        wp.config.deterministic = saved


def _holds_steel_kernel(module):
    return any(kernel.func.__module__.startswith("newton._src.solvers.steel") for kernel in module.kernels.values())


def _check_inherited_global_mode(device):
    """Run Steel in a process that set ``wp.config.deterministic`` to RUN_TO_RUN before importing Newton."""
    from warp._src.context import Module, user_modules  # noqa: PLC0415

    assert wp.config.deterministic == RUN_TO_RUN
    # Newton's collision kernels do not support Warp's deterministic lowering; only Steel's modules keep the global.
    for module in list(user_modules.values()):
        if not _holds_steel_kernel(module):
            wp.set_module_options({"deterministic": wp.DeterministicMode.NOT_GUARANTEED}, module=module)
    model = _build_contact_scene(device)

    def run(deterministic):
        solver = SolverSteel(model, deterministic=deterministic, **MATRIX_FREE_OPTIONS)
        with _warp_lowering_off():
            pipeline = newton.CollisionPipeline(model, deterministic=True)
            contacts = pipeline.contacts()
        state_0, state_1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        _actuate(model, control)

        def step(state_in, state_out):
            state_in.clear_forces()
            with _warp_lowering_off():
                pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, 0.005)

        hashes = []
        for _ in range(40):
            step(state_0, state_1)
            state_0, state_1 = state_1, state_0
            hashes.append(_sample(state_0))
        with wp.ScopedCapture(device=device) as capture:
            step(state_0, state_1)
            step(state_1, state_0)
        for _ in range(10):
            wp.capture_launch(capture.graph)
            hashes.append(_sample(state_0))
        return solver, hashes

    solver, first = run(None)
    assert solver._ordered_rows
    _, second = run(None)
    assert first == second, "inherited RUN_TO_RUN runs differ"
    assert all(finite for _, finite in first)
    _, unordered = run(wp.DeterministicMode.NOT_GUARANTEED)
    assert all(finite for _, finite in unordered)
    # Lazy module factories, and explicitly constructed modules, live outside Warp's module registry.
    _get_tree_fk_kernel(2, "id")
    _get_tree_fk_kernel(2, "public")
    _get_tree_tau_kernel(2)
    modules = [m for m in gc.get_objects() if isinstance(m, Module) and _holds_steel_kernel(m)]
    assert len(modules) > len([m for m in user_modules.values() if _holds_steel_kernel(m)])
    lowered = [m.name for m in modules if m.options["deterministic"] != wp.DeterministicMode.NOT_GUARANTEED]
    assert not lowered, f"Steel modules use Warp's deterministic lowering: {lowered}"


def test_inherited_global_mode_keeps_steel_ordering(test, device):
    """Steel keeps its own ordering, and repeats bitwise, when RUN_TO_RUN is set process-wide before import."""
    code = (
        "import warp as wp; wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN; "
        "from newton.tests.steel.test_determinism import _check_inherited_global_mode; "
        f"_check_inherited_global_mode({str(device)!r})"
    )
    if newton.tests.unittest_utils.strict_warnings:
        policy = "import warnings; warnings.filterwarnings('error', module=r'newton(\\.|$)'); "
        for message in newton.tests.unittest_utils.allowed_deprecation_warnings:
            policy += (
                f"warnings.filterwarnings('default', message={re.escape(message)!r}, category=DeprecationWarning); "
            )
        code = policy + code
    env = os.environ.copy()
    env.pop("PYTHONWARNINGS", None)
    result = subprocess.run(
        [sys.executable, *newton.tests.unittest_utils.get_strict_warning_args(), "-c", code],
        capture_output=True,
        env=env,
        text=True,
        timeout=900,
        check=False,
    )
    test.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


class TestSteelDeterminism(unittest.TestCase):
    pass


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()
for _fn in (
    test_crba_fill_par_dof_is_batch_invariant,
    test_full_reset_matches_fresh_solver,
    test_deterministic_runs_are_bitwise_identical,
    test_deterministic_overflow_keeps_earliest_contacts,
):
    add_function_test(TestSteelDeterminism, _fn.__name__, _fn, devices=devices)
add_function_test(
    TestSteelDeterminism,
    "test_overflow_warning_names_deterministic_mode",
    test_overflow_warning_names_deterministic_mode,
    devices=devices,
    check_output=False,
)
for _fn in (
    test_warm_started_full_reset_matches_fresh_solver,
    test_deterministic_launches_replay_identically,
    test_deterministic_joint_rows_follow_articulation_order,
    test_deterministic_joint_friction_rows_follow_dof_order,
    test_deterministic_bilateral_rows_follow_element_order,
    test_inherited_global_mode_keeps_steel_ordering,
):
    add_function_test(TestSteelDeterminism, _fn.__name__, _fn, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
