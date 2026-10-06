# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for FeatherPGS speculative-contact controls."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    allocate_world_contact_slots,
    apply_world_contact_restitution,
    compute_mf_effective_mass_and_rhs,
    compute_world_contact_bias,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    prepare_world_contact_rows,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

PATH_DENSE = 0
PATH_MATRIX_FREE = 1
_UNBOUNDED = 2**31 - 1


def _disabled_patches(device):
    """Return the disabled friction-patch view: every contact gets point friction rows."""
    patches = FrictionPatches()
    patches.enabled = 0
    patches.weight = wp.zeros(0, dtype=float, device=device)
    patches.next_contact = wp.zeros(0, dtype=int, device=device)
    patches.point_a = wp.zeros(0, dtype=wp.vec3, device=device)
    patches.point_b = wp.zeros(0, dtype=wp.vec3, device=device)
    patches.phi = wp.zeros(1, dtype=wp.vec2, device=device)
    return patches


def _friction_threshold(enable_friction: bool, friction_gap: float) -> float:
    """Contacts always get friction rows below the threshold; ``-inf`` gives none."""
    return friction_gap if enable_friction else -float("inf")


def _launch_contact_allocator(
    device,
    *,
    route: int,
    gap: float,
    gate: float,
    responsive: bool = True,
    scoped_gate: float = 0.0,
    pair_gate: float = 0.0,
    enable_friction: bool = False,
    friction_gap: float = float("inf"),
    friction_pairs_only: bool = False,
):
    """Allocate one contact of a one-DOF body against the ground and return its routing and counters."""
    outputs = {
        "world": wp.full((1,), -9, dtype=wp.int32, device=device),
        "slot": wp.full((1,), -9, dtype=wp.int32, device=device),
        "art_a": wp.full((1,), -9, dtype=wp.int32, device=device),
        "art_b": wp.full((1,), -9, dtype=wp.int32, device=device),
        "slots_needed": wp.full((1,), -9, dtype=wp.int32, device=device),
        "path": wp.full((1,), -9, dtype=wp.int32, device=device),
    }
    counters = {
        "dense_count": wp.zeros((1,), dtype=wp.int32, device=device),
        "mf_count": wp.zeros((1,), dtype=wp.int32, device=device),
        "dense_dropped": wp.zeros((1,), dtype=wp.int32, device=device),
        "mf_dropped": wp.zeros((1,), dtype=wp.int32, device=device),
    }
    is_free = route == PATH_MATRIX_FREE
    wp.launch(
        allocate_world_contact_slots,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device=device),
            1,
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([-1], dtype=wp.int32, device=device),
            wp.array([wp.vec3(gap, 0.0, 0.0)], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(0.0)], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(-1.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.array([int(responsive)], dtype=wp.int32, device=device),
            wp.array([int(is_free)], dtype=wp.int32, device=device),
            1,
            0,
            0,
            0,
            8,
            8,
            8,
            wp.array([0], dtype=wp.int32, device=device),
            gate,
            scoped_gate,
            pair_gate,
            _friction_threshold(enable_friction, friction_gap),
            int(friction_pairs_only),
            _disabled_patches(device),
        ],
        outputs=[
            outputs["world"],
            outputs["slot"],
            outputs["art_a"],
            outputs["art_b"],
            outputs["slots_needed"],
            counters["dense_count"],
            outputs["path"],
            counters["mf_count"],
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            counters["dense_dropped"],
            counters["mf_dropped"],
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.zeros((2,), dtype=wp.int32, device=device),
        ],
        device=device,
    )
    return {name: int(array.numpy()[0]) for name, array in (outputs | counters).items()}


def _launch_articulation_pair_contact_allocator(
    device,
    *,
    gap: float,
    scoped_gate: float = 0.0,
    pair_gate: float = 0.0,
    cross_articulation: bool = False,
    enable_friction: bool = False,
    friction_gap: float = float("inf"),
    friction_pairs_only: bool = False,
):
    """Allocate one contact between two non-free articulated links; return ``(slot, path, slots_needed)``."""
    body_to_articulation = [0, 1] if cross_articulation else [0, 0]
    articulation_count = 2 if cross_articulation else 1
    contact_slot = wp.full((1,), -9, dtype=wp.int32, device=device)
    contact_path = wp.full((1,), -9, dtype=wp.int32, device=device)
    contact_slots_needed = wp.full((1,), -9, dtype=wp.int32, device=device)
    wp.launch(
        allocate_world_contact_slots,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device=device),
            1,
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([wp.vec3(gap, 0.0, 0.0)], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(0.0)], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(-1.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array(body_to_articulation, dtype=wp.int32, device=device),
            wp.array([0] * articulation_count, dtype=wp.int32, device=device),
            wp.array([1] * articulation_count, dtype=wp.int32, device=device),
            wp.zeros((2,), dtype=wp.int32, device=device),
            wp.ones((2,), dtype=wp.int32, device=device),
            wp.zeros((articulation_count,), dtype=wp.int32, device=device),
            0,
            0,
            0,
            0,
            8,
            8,
            8,
            wp.zeros((articulation_count,), dtype=wp.int32, device=device),
            0.0,
            scoped_gate,
            pair_gate,
            _friction_threshold(enable_friction, friction_gap),
            int(friction_pairs_only),
            _disabled_patches(device),
        ],
        outputs=[
            wp.full((1,), -9, dtype=wp.int32, device=device),
            contact_slot,
            wp.full((1,), -9, dtype=wp.int32, device=device),
            wp.full((1,), -9, dtype=wp.int32, device=device),
            contact_slots_needed,
            wp.zeros((1,), dtype=wp.int32, device=device),
            contact_path,
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.zeros((2,), dtype=wp.int32, device=device),
        ],
        device=device,
    )
    slots_needed = max(0, int(contact_slots_needed.numpy()[0]))
    return int(contact_slot.numpy()[0]), int(contact_path.numpy()[0]), slots_needed


def _dense_speculative_rhs(device, scale: float) -> float:
    """Return the dense positive-gap right-hand side for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device=device)
    wp.launch(
        compute_world_contact_bias,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([[1.0]], dtype=wp.float32, device=device),
            wp.array([[PGS_CONSTRAINT_TYPE_CONTACT]], dtype=wp.int32, device=device),
            wp.zeros((1, 1), dtype=wp.float32, device=device),
            0.2,
            0.2,
            scale,
            1.0,
            0.5,
        ],
        outputs=[rhs, wp.zeros((1, 1), dtype=wp.float32, device=device)],
        device=device,
    )
    return float(rhs.numpy()[0, 0])


def _mf_speculative_rhs(device, scale: float) -> float:
    """Return the free-body setup right-hand side for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device=device)
    wp.launch(
        compute_mf_effective_mass_and_rhs,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([[-1]], dtype=wp.int32, device=device),
            wp.array([[-1]], dtype=wp.int32, device=device),
            wp.zeros((1, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((1, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.spatial_matrix, device=device),
            wp.array([[1.0]], dtype=wp.float32, device=device),
            wp.array([[PGS_CONSTRAINT_TYPE_CONTACT]], dtype=wp.int32, device=device),
            wp.zeros((1, 1), dtype=wp.float32, device=device),
            wp.zeros((1, 1), dtype=wp.float32, device=device),
            0,
            wp.array([-1], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.array([float("inf")], dtype=wp.float32, device=device),
            1.0,
            0.2,
            1.0,
            0.5,
            scale,
            0.5,
            1,
        ],
        outputs=[
            wp.zeros((1, 1), dtype=wp.float32, device=device),
            wp.zeros((1, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((1, 1, 6), dtype=wp.float32, device=device),
            rhs,
            wp.zeros((1, 1), dtype=wp.float32, device=device),
        ],
        device=device,
    )
    return float(rhs.numpy()[0, 0])


def _dense_restitution_rhs(device, scale: float) -> float:
    """Build the scaled position bias of an impacting dense contact, then apply restitution.

    The row has gap 0.5 m, restitution 0.5 and incident normal velocity -3 m/s over
    ``dt = 0.5`` s, so the rebound fires and replaces the bias by ``e * u = -1.5``.
    """
    phi = wp.array([[0.5]], dtype=wp.float32, device=device)
    row_type = wp.array([[PGS_CONSTRAINT_TYPE_CONTACT]], dtype=wp.int32, device=device)
    target = wp.zeros((1, 1), dtype=wp.float32, device=device)
    rhs = wp.zeros((1, 1), dtype=wp.float32, device=device)
    row_w = wp.ones((1, 1), dtype=wp.float32, device=device)
    wp.launch(
        compute_world_contact_bias,
        dim=1,
        inputs=[wp.array([1], dtype=wp.int32, device=device), phi, row_type, target, 0.2, 0.2, scale, 1.0, 0.5],
        outputs=[rhs, row_w],
        device=device,
    )
    wp.launch(
        apply_world_contact_restitution,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device=device),
            1,
            wp.array([1], dtype=wp.int32, device=device),
            phi,
            row_type,
            target,
            wp.array([[0.5]], dtype=wp.float32, device=device),
            wp.array([-3.0], dtype=wp.float32, device=device),
            wp.array([[0]], dtype=wp.int32, device=device),
            wp.ones((1, 1, 1), dtype=wp.float32, device=device),
            0.5,
            0.5,
            0,
        ],
        outputs=[rhs, row_w],
        device=device,
    )
    return float(rhs.numpy()[0, 0])


# Body-frame witness points and thicknesses of the dense same-articulation fixture contact.
_DENSE_POINT0 = (0.2, 0.1, -0.05)
_DENSE_POINT1 = (-0.1, 0.05, 0.2)
_DENSE_THICKNESS = (0.01, 0.02)
# World positions of the fixture's two contact bodies (shape 0 on body 2, shape 1 on body 1).
_DENSE_BODY_POSITIONS = ((0.0, 0.0, 0.0), (-0.2, 0.3, 0.1), (0.4, -0.1, 0.2))


def _launch_dense_contact_builders(
    device, *, anchors=(0, 0), point0=_DENSE_POINT0, point1=_DENSE_POINT1, thickness=_DENSE_THICKNESS
):
    """Build one same-articulation contact's Jacobian through the tree-walk and compact paths."""
    contact_count = wp.array([1], dtype=wp.int32, device=device)
    point0 = wp.array([wp.vec3(*point0)], dtype=wp.vec3, device=device)
    point1 = wp.array([wp.vec3(*point1)], dtype=wp.vec3, device=device)
    normal = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3, device=device)
    shape0 = wp.array([0], dtype=wp.int32, device=device)
    shape1 = wp.array([1], dtype=wp.int32, device=device)
    thickness0 = wp.array([thickness[0]], dtype=wp.float32, device=device)
    thickness1 = wp.array([thickness[1]], dtype=wp.float32, device=device)
    contact_world = wp.array([0], dtype=wp.int32, device=device)
    contact_slot = wp.array([0], dtype=wp.int32, device=device)
    contact_art_a = wp.array([0], dtype=wp.int32, device=device)
    contact_art_b = wp.array([0], dtype=wp.int32, device=device)
    contact_path = wp.array([PATH_DENSE], dtype=wp.int32, device=device)
    contact_slots_needed = wp.array([3], dtype=wp.int32, device=device)
    response_dof_count = wp.array([3], dtype=wp.int32, device=device)
    art_group_idx = wp.array([0], dtype=wp.int32, device=device)
    art_dof_start = wp.array([0], dtype=wp.int32, device=device)
    articulation_origin = wp.array([wp.vec3(0.1, -0.2, 0.3)], dtype=wp.vec3, device=device)
    body_to_joint = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    joint_ancestor = wp.array([-1, 0, 1], dtype=wp.int32, device=device)
    joint_qd_start = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
    joint_s = wp.array(
        [
            wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.5),
            wp.spatial_vector(0.0, 1.0, 0.0, 0.25, 0.0, 0.0),
            wp.spatial_vector(0.0, 0.0, 1.0, 0.0, -0.5, 0.0),
        ],
        dtype=wp.spatial_vector,
        device=device,
    )
    shape_body = wp.array([2, 1], dtype=wp.int32, device=device)
    body_q = wp.array(
        [wp.transform(wp.vec3(*position), wp.quat_identity()) for position in _DENSE_BODY_POSITIONS],
        dtype=wp.transform,
        device=device,
    )
    geometry = [point0, point1, normal, shape0, shape1, thickness0, thickness1]
    patches = _disabled_patches(device)

    serial = wp.zeros((1, 8, 3), dtype=wp.float32, device=device)
    wp.launch(
        populate_world_J_for_size,
        dim=1,
        inputs=[
            contact_count,
            1,
            *geometry,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
            contact_slots_needed,
            3,
            response_dof_count,
            art_group_idx,
            art_dof_start,
            articulation_origin,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            joint_s,
            shape_body,
            body_q,
            patches,
            *anchors,
        ],
        outputs=[serial],
        device=device,
    )
    compact = wp.zeros((1, 8, 3), dtype=wp.float32, device=device)
    wp.launch(
        populate_world_J_for_compact_size,
        dim=(1, 32),
        inputs=[
            contact_count,
            1,
            *geometry,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
            contact_slots_needed,
            3,
            response_dof_count,
            art_group_idx,
            art_dof_start,
            articulation_origin,
            wp.array([1, 3, 7], dtype=wp.uint32, device=device),
            joint_s,
            shape_body,
            body_q,
            patches,
            *anchors,
        ],
        outputs=[compact],
        device=device,
    )
    metadata = {
        "row_type": wp.full((1, 8), -9, dtype=wp.int32, device=device),
        "row_parent": wp.full((1, 8), -9, dtype=wp.int32, device=device),
        "row_mu": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "phi": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "target_velocity": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "row_restitution": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
    }
    wp.launch(
        prepare_world_contact_rows,
        dim=1,
        inputs=[
            contact_count,
            1,
            *geometry,
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
            contact_slots_needed,
            shape_body,
            body_q,
            wp.zeros((3,), dtype=wp.spatial_vector, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            articulation_origin,
            wp.array([0.6, 0.8], dtype=wp.float32, device=device),
            wp.array([0.2, 0.4], dtype=wp.float32, device=device),
            patches,
            *anchors,
        ],
        outputs=list(metadata.values()),
        device=device,
    )
    return serial.numpy(), compact.numpy(), {name: array.numpy() for name, array in metadata.items()}


def _free_body_model(device):
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize(device=device)


def test_compact_contact_builder_matches_tree_walk(test, device):
    """Match the dense Jacobians of the compact and tree-walk builders for a same-articulation contact."""
    serial, compact, metadata = _launch_dense_contact_builders(device)
    test.assertGreater(float(np.abs(serial[0, :3]).max()), 0.0)
    np.testing.assert_allclose(compact, serial, rtol=0.0, atol=1.0e-6)
    np.testing.assert_array_equal(
        metadata["row_type"][0, :3],
        [PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION],
    )
    np.testing.assert_array_equal(metadata["row_parent"][0, :3], [-1, 0, 0])
    np.testing.assert_allclose(metadata["row_mu"][0, :3], 0.7, rtol=1.0e-6)
    np.testing.assert_allclose(metadata["row_restitution"][0, :3], [0.3, 0.0, 0.0], rtol=1.0e-6)


def test_shared_anchors_move_the_row_points_to_the_witness_midpoint(test, device):
    """Apply shared-anchor rows at the witness midpoint while ``phi`` keeps the witness points."""
    # The same contact with both points at the world midpoint and no thickness.
    normal = np.array([0.0, 0.0, -1.0])
    witness_a = np.add(_DENSE_BODY_POSITIONS[2], _DENSE_POINT0) + _DENSE_THICKNESS[0] * normal
    witness_b = np.add(_DENSE_BODY_POSITIONS[1], _DENSE_POINT1) - _DENSE_THICKNESS[1] * normal
    midpoint = 0.5 * (witness_a + witness_b)
    midpoint_rows = _launch_dense_contact_builders(
        device,
        point0=tuple(midpoint - _DENSE_BODY_POSITIONS[2]),
        point1=tuple(midpoint - _DENSE_BODY_POSITIONS[1]),
        thickness=(0.0, 0.0),
    )
    witness_rows = _launch_dense_contact_builders(device)
    for anchors, normal_source, friction_source in (
        ((1, 0), midpoint_rows, midpoint_rows),
        ((1, 1), midpoint_rows, midpoint_rows),
        ((0, 1), witness_rows, midpoint_rows),
    ):
        with test.subTest(anchors=anchors):
            serial, compact, metadata = _launch_dense_contact_builders(device, anchors=anchors)
            for jacobian in (serial, compact):
                np.testing.assert_allclose(jacobian[0, 0], normal_source[0][0, 0], rtol=0.0, atol=1.0e-6)
                np.testing.assert_allclose(jacobian[0, 1:3], friction_source[0][0, 1:3], rtol=0.0, atol=1.0e-6)
            np.testing.assert_allclose(metadata["phi"][0, 0], witness_rows[2]["phi"][0, 0], rtol=0.0, atol=1.0e-7)
    # The fixture's witness points are separated along and across the normal, so the rows differ.
    test.assertGreater(float(np.abs(midpoint_rows[0][0, :3] - witness_rows[0][0, :3]).max()), 1.0e-3)


def test_shared_anchors_warn_with_patch_friction(test, device):
    """Explain that patch anchors keep their friction points when shared anchors are requested."""
    model = _free_body_model(device)
    for flag in ("contact_shared_anchor", "contact_friction_shared_anchor"):
        with test.subTest(flag=flag), test.assertWarnsRegex(UserWarning, "friction_anchor_beta=0"):
            SolverFeatherPGS(model, **{flag: True})
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.0, **{flag: True})
        test.assertTrue(getattr(solver, flag))


def test_shared_anchors_step_on_every_route(test, device):
    """Rest a box and a two-link arm on the ground with shared anchors in every solve and response."""
    builder = newton.ModelBuilder()
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    parent = -1
    joints = []
    for _ in range(2):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.08, hy=0.05, hz=0.05)
        xform = wp.vec3(0.5, 0.0, 0.05) if parent == -1 else wp.vec3(0.2, 0.0, 0.0)
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(xform, wp.quat_identity()),
                child_xform=wp.transform_identity(),
            )
        )
        parent = link
    builder.add_articulation(joints)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    routes = (
        {"pgs_mode": "matrix_free"},
        {"pgs_mode": "split"},
        {"articulated_contact_response": "propagation"},
        {"articulated_contact_response": "propagation-fused"},
    )

    def run(route, shared):
        solver = SolverFeatherPGS(
            model,
            friction_anchor_beta=0.0,
            contact_shared_anchor=shared,
            contact_friction_shared_anchor=shared,
            dense_max_constraints=128,
            mf_max_constraints=128,
            **route,
        )
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        for _ in range(120):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        solver.check_constraint_capacity()
        return state_0.body_q.numpy(), state_0.body_qd.numpy()

    for route in routes:
        with test.subTest(**route):
            body_q, body_qd = run(route, True)
            reference_q, _ = run(route, False)
            test.assertTrue(np.isfinite(body_q).all())
            # Resting contacts have coincident witness points, so the rest pose is unchanged.
            np.testing.assert_allclose(body_q, reference_q, rtol=0.0, atol=2.0e-3)
            test.assertLess(float(np.abs(body_qd).max()), 0.05)


def test_solver_exposes_documented_defaults(test, device):
    """Expose the documented contact-control defaults, with friction patches on."""
    model = _free_body_model(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free")
    test.assertEqual(solver.friction_anchor_beta, 0.2)
    test.assertTrue(solver._friction_anchors_enabled)
    test.assertEqual(solver.contact_speculative_scale, 1.0)
    test.assertEqual(solver.contact_gap_gate, 0.0)
    test.assertEqual(solver.same_articulation_contact_gap_gate, 0.0)
    test.assertEqual(solver.articulation_pair_contact_gap_gate, 0.0)
    test.assertEqual(solver.contact_friction_gap_threshold, float("inf"))
    test.assertFalse(solver.contact_friction_articulation_pairs_only)
    test.assertEqual(solver.pgs_contact_regularization, 0.0)
    test.assertEqual(solver.pgs_velocity_iterations, 0)
    test.assertFalse(solver.pgs_warmstart)
    test.assertEqual(solver.restitution_velocity_threshold, 0.5)
    test.assertTrue(solver.warn_constraint_overflow)
    test.assertIsNotNone(solver._row_overflow_warning_emitted)
    quiet_solver = SolverFeatherPGS(model, pgs_mode="matrix_free", warn_constraint_overflow=False)
    test.assertFalse(quiet_solver.warn_constraint_overflow)
    test.assertIsNone(quiet_solver._row_overflow_warning_emitted)
    point = SolverFeatherPGS(model, pgs_mode="matrix_free", friction_anchor_beta=0.0)
    test.assertFalse(point._friction_anchors_enabled)


def test_solver_validates_and_stores_contact_controls(test, device):
    """Accept finite non-negative controls and reject malformed values."""
    model = _free_body_model(device)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        contact_speculative_scale=0.0,
        contact_gap_gate=0.001,
        same_articulation_contact_gap_gate=0.002,
        articulation_pair_contact_gap_gate=0.003,
        contact_friction_articulation_pairs_only=True,
    )
    test.assertEqual(solver.contact_speculative_scale, 0.0)
    test.assertEqual(solver.contact_gap_gate, 0.001)
    test.assertEqual(solver.same_articulation_contact_gap_gate, 0.002)
    test.assertEqual(solver.articulation_pair_contact_gap_gate, 0.003)
    test.assertTrue(solver.contact_friction_articulation_pairs_only)

    for name in (
        "contact_speculative_scale",
        "contact_gap_gate",
        "same_articulation_contact_gap_gate",
        "articulation_pair_contact_gap_gate",
        "friction_anchor_beta",
        "restitution_velocity_threshold",
    ):
        for value in (-0.1, float("nan"), float("inf"), "invalid"):
            with test.subTest(name=name, value=value):
                with test.assertRaisesRegex(ValueError, name):
                    SolverFeatherPGS(model, pgs_mode="matrix_free", **{name: value})
    with test.assertRaisesRegex(ValueError, "contact_friction_gap_threshold"):
        SolverFeatherPGS(model, pgs_mode="matrix_free", contact_friction_gap_threshold=float("nan"))


def test_scoped_gap_gate_only_drops_distant_same_articulation_contact(test, device):
    """Retain near self-contact under the scoped gate while bounding its speculative tail."""
    test.assertEqual(
        _launch_articulation_pair_contact_allocator(device, gap=0.002, scoped_gate=0.003),
        (0, PATH_DENSE, 1),
    )
    test.assertEqual(
        _launch_articulation_pair_contact_allocator(device, gap=0.004, scoped_gate=0.003),
        (-1, -1, 0),
    )
    test.assertEqual(
        _launch_articulation_pair_contact_allocator(device, gap=0.004, scoped_gate=0.0),
        (0, PATH_DENSE, 1),
    )


def test_scoped_gap_gate_preserves_other_contact_routes(test, device):
    """Do not shorten predictive contacts for ground, free-body, or cross-articulation rows."""
    for route, counter in ((PATH_DENSE, "dense_count"), (PATH_MATRIX_FREE, "mf_count")):
        with test.subTest(route=route):
            result = _launch_contact_allocator(device, route=route, gap=0.04, gate=0.0, scoped_gate=0.003)
            test.assertEqual(result["path"], route)
            test.assertEqual(result[counter], 1)


def test_articulation_pair_gap_gate_drops_distant_pair_contact(test, device):
    """Gate same- and cross-articulation contacts with the pair gate, leaving free bodies untouched."""
    for cross_articulation in (False, True):
        with test.subTest(cross_articulation=cross_articulation):
            test.assertEqual(
                _launch_articulation_pair_contact_allocator(
                    device, gap=0.004, scoped_gate=0.0, pair_gate=0.003, cross_articulation=cross_articulation
                ),
                (-1, -1, 0),
            )
    result = _launch_contact_allocator(device, route=PATH_MATRIX_FREE, gap=0.04, gate=0.0, pair_gate=0.003)
    test.assertEqual(result["path"], PATH_MATRIX_FREE)
    test.assertEqual(result["mf_count"], 1)


def test_articulation_pair_friction_filter_preserves_free_and_ground_rows(test, device):
    """Scope tight friction controls to articulated pairs, not free-body or ground routes."""
    for route in (PATH_DENSE, PATH_MATRIX_FREE):
        with test.subTest(route=route):
            result = _launch_contact_allocator(
                device,
                route=route,
                gap=0.004,
                gate=0.0,
                enable_friction=True,
                friction_gap=0.002,
                friction_pairs_only=True,
            )
            test.assertEqual(result["path"], route)
            test.assertEqual(result["slots_needed"], 3)


def test_articulation_pair_friction_filter_reduces_pair_rows(test, device):
    """Apply the configured friction gap to same- and cross-articulation contacts."""
    for cross_articulation in (False, True):
        with test.subTest(cross_articulation=cross_articulation):
            result = _launch_articulation_pair_contact_allocator(
                device,
                gap=0.004,
                cross_articulation=cross_articulation,
                enable_friction=True,
                friction_gap=0.002,
                friction_pairs_only=True,
            )
            test.assertEqual(result, (0, PATH_DENSE, 1))


def test_negative_scoped_friction_gap_delays_only_articulation_pair_tangents(test, device):
    """Keep pair normals while delaying friction, without changing free-body friction."""
    shallow_pair = _launch_articulation_pair_contact_allocator(
        device, gap=-0.0005, enable_friction=True, friction_gap=-0.001, friction_pairs_only=True
    )
    deep_pair = _launch_articulation_pair_contact_allocator(
        device, gap=-0.002, enable_friction=True, friction_gap=-0.001, friction_pairs_only=True
    )
    free_body = _launch_contact_allocator(
        device,
        route=PATH_MATRIX_FREE,
        gap=0.004,
        gate=0.0,
        enable_friction=True,
        friction_gap=-1.0,
        friction_pairs_only=True,
    )
    test.assertEqual(shallow_pair, (0, PATH_DENSE, 1))
    test.assertEqual(deep_pair, (0, PATH_DENSE, 3))
    test.assertEqual(free_body["slots_needed"], 3)


def test_speculative_scale_controls_every_position_rhs_family(test, device):
    """Scale the positive-gap position bias of dense and free-body rows."""
    for name, compute_rhs in (("dense", _dense_speculative_rhs), ("free_body", _mf_speculative_rhs)):
        with test.subTest(family=name):
            test.assertEqual(compute_rhs(device, 0.0), 0.0)
            test.assertAlmostEqual(compute_rhs(device, 1.0), 2.0, places=6)


def test_dense_restitution_removes_the_scaled_position_bias(test, device):
    """Replace the position bias with the rebound target, independently of the speculative scale."""
    test.assertAlmostEqual(_dense_restitution_rhs(device, 0.0), -1.5, places=6)
    test.assertAlmostEqual(_dense_restitution_rhs(device, 1.0), -1.5, places=6)


def test_gap_gate_prevents_all_route_allocations(test, device):
    """Drop positive gaps above the gate before any route reserves a slot."""
    for route in (PATH_DENSE, PATH_MATRIX_FREE):
        with test.subTest(route=route):
            allocated = _launch_contact_allocator(device, route=route, gap=0.002, gate=0.0)
            test.assertEqual(allocated["path"], route)
            test.assertEqual(allocated["slot"], 0)
            test.assertEqual(allocated["world"], 0)
            test.assertEqual(allocated["art_a"], 0)
            test.assertEqual(allocated["art_b"], -1)
            test.assertEqual(allocated["slots_needed"], 1)
            test.assertEqual(
                (allocated["dense_count"], allocated["mf_count"]),
                (int(route == PATH_DENSE), int(route == PATH_MATRIX_FREE)),
            )

            dropped = _launch_contact_allocator(device, route=route, gap=0.002, gate=0.001)
            test.assertEqual(dropped["slot"], -1)
            test.assertEqual(dropped["path"], -1)
            test.assertEqual(dropped["slots_needed"], 0)
            test.assertEqual(dropped["dense_count"], 0)
            test.assertEqual(dropped["mf_count"], 0)


def test_gap_gate_keeps_contacts_at_threshold(test, device):
    """Keep a contact whose gap equals the positive gate exactly."""
    for route in (PATH_DENSE, PATH_MATRIX_FREE):
        with test.subTest(route=route):
            result = _launch_contact_allocator(device, route=route, gap=0.001, gate=0.001)
            test.assertEqual(result["path"], route)
            test.assertEqual(result["slots_needed"], 1)


def test_allocator_skips_contacts_without_response_dofs(test, device):
    """Do not allocate a row when neither contact body can change velocity."""
    result = _launch_contact_allocator(device, route=PATH_DENSE, gap=-0.001, gate=0.0, responsive=False)
    test.assertEqual(result["slot"], -1)
    test.assertEqual(result["path"], -1)
    test.assertEqual(result["dense_count"], 0)


class TestFeatherPGSContactControls(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _fn in (
    test_compact_contact_builder_matches_tree_walk,
    test_shared_anchors_move_the_row_points_to_the_witness_midpoint,
    test_shared_anchors_warn_with_patch_friction,
    test_shared_anchors_step_on_every_route,
    test_solver_exposes_documented_defaults,
    test_solver_validates_and_stores_contact_controls,
    test_scoped_gap_gate_only_drops_distant_same_articulation_contact,
    test_scoped_gap_gate_preserves_other_contact_routes,
    test_articulation_pair_gap_gate_drops_distant_pair_contact,
    test_articulation_pair_friction_filter_preserves_free_and_ground_rows,
    test_articulation_pair_friction_filter_reduces_pair_rows,
    test_negative_scoped_friction_gap_delays_only_articulation_pair_tangents,
    test_speculative_scale_controls_every_position_rhs_family,
    test_dense_restitution_removes_the_scaled_position_bias,
    test_gap_gate_prevents_all_route_allocations,
    test_gap_gate_keeps_contacts_at_threshold,
    test_allocator_skips_contacts_without_response_dofs,
):
    add_function_test(TestFeatherPGSContactControls, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
