# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact controls, contact forces, restitution and paired responses of SolverSteel."""

import itertools
import unittest
import warnings

import numpy as np
import warp as wp

import newton
import newton._src.solvers.steel.kernels
from newton._src.geometry.sdf_hydroelastic import HydroelasticSDF
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.friction import contact_tangent_basis
from newton._src.solvers.steel.friction_patches import FrictionPatches
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    allocate_world_contact_slots,
    apply_world_contact_restitution,
    compute_mf_effective_mass_and_rhs,
    compute_mf_velocity_rhs,
    compute_world_contact_bias,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    prepare_world_contact_rows,
)
from newton.solvers import SolverBase, SolverSteel
from newton.tests.steel._fixtures import (
    RADIUS,
    _build_plane_model,
    _build_sparse_contact_friction_model,
    _diagonal_inertia,
    _reset_state,
    _shape_cfg,
    _sphere_inertia,
    set_torsion_radius,
)
from newton.tests.unittest_utils import (
    StdOutCapture,
    add_function_test,
    get_cuda_test_devices,
    get_selected_cuda_test_devices,
    get_test_devices,
)

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
    gate: float | None = None,
    responsive: bool = True,
    enable_friction: bool = False,
    friction_gap: float = float("inf"),
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
            float("inf") if gate is None else gate,
            _friction_threshold(enable_friction, friction_gap),
            _disabled_patches(device),
            0,
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
    gate: float | None = None,
    cross_articulation: bool = False,
    enable_friction: bool = False,
    friction_gap: float = float("inf"),
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
            float("inf") if gate is None else gate,
            _friction_threshold(enable_friction, friction_gap),
            _disabled_patches(device),
            0,
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
        with test.subTest(flag=flag), test.assertWarnsRegex(newton.exceptions.NewtonWarning, "friction_anchor_beta=0"):
            SolverSteel(model, **{flag: True})
        solver = SolverSteel(model, friction_anchor_beta=0.0, **{flag: True})
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
        {"solve_mode": "matrix_free"},
        {"solve_mode": "split"},
        {"articulated_contact_response": "propagation"},
    )

    def run(route, shared):
        solver = SolverSteel(
            model,
            friction_anchor_beta=0.0,
            contact_shared_anchor=shared,
            contact_friction_shared_anchor=shared,
            articulation_constraint_max=128,
            free_body_constraint_max=128,
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
    solver = SolverSteel(model, solve_mode="matrix_free")
    test.assertEqual(solver.friction_anchor_beta, 0.2)
    test.assertTrue(solver._friction_anchors_enabled)
    test.assertEqual(solver.contact_speculative_scale, 1.0)
    test.assertIsNone(solver.contact_gap_gate)
    test.assertIsNone(solver.contact_friction_gap_max)
    test.assertEqual(solver.contact_regularization, 0.0)
    test.assertEqual(solver.velocity_iterations, 0)
    test.assertFalse(solver.contact_warmstart)
    test.assertEqual(solver.restitution_velocity_threshold, 0.5)
    test.assertTrue(solver.warn_constraint_overflow)
    test.assertIsNotNone(solver._row_overflow_warning_emitted)
    quiet_solver = SolverSteel(model, solve_mode="matrix_free", warn_constraint_overflow=False)
    test.assertFalse(quiet_solver.warn_constraint_overflow)
    test.assertIsNotNone(quiet_solver._row_overflow_warning_emitted)
    point = SolverSteel(model, solve_mode="matrix_free", friction_anchor_beta=0.0)
    test.assertFalse(point._friction_anchors_enabled)


def test_solver_validates_and_stores_contact_controls(test, device):
    """Accept finite non-negative controls and reject malformed values."""
    model = _free_body_model(device)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        contact_speculative_scale=0.0,
        contact_gap_gate=0.001,
        contact_friction_gap_max=-0.5,
    )
    test.assertEqual(solver.contact_speculative_scale, 0.0)
    test.assertEqual(solver.contact_gap_gate, 0.001)
    test.assertEqual(solver.contact_friction_gap_max, -0.5)

    for name in (
        "contact_speculative_scale",
        "contact_gap_gate",
        "friction_anchor_beta",
        "restitution_velocity_threshold",
    ):
        for value in (-0.1, float("nan"), float("inf"), "invalid"):
            with test.subTest(name=name, value=value):
                with test.assertRaisesRegex(ValueError, name):
                    SolverSteel(model, solve_mode="matrix_free", **{name: value})
    with test.assertRaisesRegex(ValueError, "contact_friction_gap_max"):
        SolverSteel(model, solve_mode="matrix_free", contact_friction_gap_max=float("nan"))


def test_zero_gap_gate_keeps_touching_contacts_only(test, device):
    """Keep touching and penetrating contacts with a zero gate, and drop any positive gap on every route."""
    for route, counter in ((PATH_DENSE, "dense_count"), (PATH_MATRIX_FREE, "mf_count")):
        with test.subTest(route=route):
            for gap, kept in ((-0.001, True), (0.0, True), (1.0e-4, False)):
                result = _launch_contact_allocator(device, route=route, gap=gap, gate=0.0)
                test.assertEqual(result[counter], int(kept), f"gap={gap}")
            test.assertEqual(_launch_contact_allocator(device, route=route, gap=0.04)[counter], 1)
    for cross_articulation in (False, True):
        with test.subTest(cross_articulation=cross_articulation):
            test.assertEqual(
                _launch_articulation_pair_contact_allocator(
                    device, gap=0.004, gate=0.003, cross_articulation=cross_articulation
                ),
                (-1, -1, 0),
            )


def test_friction_gap_max_applies_to_every_route(test, device):
    """Give contacts beyond the friction gap a normal row only, on every route, also with a negative max."""
    for route in (PATH_DENSE, PATH_MATRIX_FREE):
        with test.subTest(route=route):
            near = _launch_contact_allocator(device, route=route, gap=0.001, enable_friction=True, friction_gap=0.002)
            far = _launch_contact_allocator(device, route=route, gap=0.004, enable_friction=True, friction_gap=0.002)
            test.assertEqual((near["slots_needed"], far["slots_needed"]), (3, 1))
    shallow = _launch_articulation_pair_contact_allocator(
        device, gap=-0.0005, enable_friction=True, friction_gap=-0.001
    )
    deep = _launch_articulation_pair_contact_allocator(device, gap=-0.002, enable_friction=True, friction_gap=-0.001)
    test.assertEqual(shallow, (0, PATH_DENSE, 1))
    test.assertEqual(deep, (0, PATH_DENSE, 3))


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
            allocated = _launch_contact_allocator(device, route=route, gap=0.002)
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
    result = _launch_contact_allocator(device, route=PATH_DENSE, gap=-0.001, responsive=False)
    test.assertEqual(result["slot"], -1)
    test.assertEqual(result["path"], -1)
    test.assertEqual(result["dense_count"], 0)


def test_friction_gap_max_keeps_the_constructed_row_layout(test, device):
    """Reject a switch to or from ``None`` after construction, and match a fresh solver after an allowed edit."""
    model = _build_sparse_contact_friction_model(device=device)
    # Tilted boxes make the triple and normal-only schedules give different motion.
    transforms = model.shape_transform.numpy()
    transforms[model.shape_body.numpy() >= 0, 3:] = np.asarray(wp.quat_rpy(0.2, 0.3, 0.1))
    model.shape_transform.assign(transforms)
    options = {
        "solve_mode": "matrix_free",
        "iterations": 8,
        "enable_joint_limits": True,
        "articulation_constraint_max": 96,
        "free_body_constraint_max": 32,
        "use_parallel_streams": True,
        "friction_anchor_beta": 0.0,
    }
    every_contact = SolverSteel(model, **options)
    for value in (-float("inf"), -1.0, 0.01):
        with test.subTest(value=value), test.assertRaisesRegex(ValueError, "contact_friction_gap_max"):
            every_contact.contact_friction_gap_max = value
    every_contact.contact_friction_gap_max = float("inf")
    gated = SolverSteel(model, **options, contact_friction_gap_max=0.01)
    with test.assertRaisesRegex(ValueError, "contact_friction_gap_max"):
        gated.contact_friction_gap_max = None
    gated.contact_friction_gap_max = -float("inf")
    fresh = SolverSteel(model, **options, contact_friction_gap_max=-float("inf"))
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
    contacts = pipeline.contacts()
    trajectories = []
    for solver in (gated, fresh):
        state, output = model.state(), model.state()
        qd = state.joint_qd.numpy()
        qd[:] = -0.3
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        trajectory = []
        for _ in range(4):
            pipeline.collide(state, contacts)
            solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
            solver.check_constraint_capacity()
            state, output = output, state
            trajectory.append(np.concatenate([state.joint_q.numpy(), state.joint_qd.numpy()]))
        trajectories.append(np.stack(trajectory))
    np.testing.assert_array_equal(trajectories[0], trajectories[1])


def test_overflow_warning_enabled_after_construction_latches(test, device):
    """Latch the dense-row overflow warning when warnings are enabled after a quiet overflowing step."""
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity())
    body = scene.add_link(xform=xform)
    scene.add_shape_box(body, hx=0.15, hy=0.1, hz=0.05)
    scene.add_articulation([scene.add_joint_prismatic(-1, body, axis=newton.Axis.Z, parent_xform=xform)])
    model = scene.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
    state, output = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver = SolverSteel(model, solve_mode="split", articulation_constraint_max=1, warn_constraint_overflow=False)
    solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
    test.assertTrue(solver.constraint_overflow.numpy().any())
    test.assertEqual(solver._row_overflow_warning_emitted.numpy().tolist(), [0, 0, 0, 0])
    solver.warn_constraint_overflow = True
    capture = StdOutCapture()
    capture.begin()
    try:
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
    finally:
        try:
            wp.synchronize()
        finally:
            printed = capture.end()
    test.assertRegex(printed, "constraint-row overflow")
    test.assertEqual(int(solver._row_overflow_warning_emitted.numpy()[0]), 1)
    test.assertTrue(np.isfinite(output.joint_q.numpy()).all())


def test_restitution_threshold_cannot_enable_split_restitution(test, device):
    """Reject positive restitution in the split solve at any threshold, so later threshold edits stay supported."""
    model, _ = _build_plane_model(device, separation=1.0e-3, restitution=0.5, scene="articulated")
    with test.assertRaisesRegex(NotImplementedError, "restitution"):
        SolverSteel(model, solve_mode="split", restitution_velocity_threshold=float(np.finfo(np.float32).max))
    solver = SolverSteel(model, solve_mode="split", enable_restitution=False)
    solver.restitution_velocity_threshold = 0.5
    test.assertEqual(solver._rebound_velocity_threshold, float(np.finfo(np.float32).max))


class TestSteelContactControls(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


for _fn in (
    test_compact_contact_builder_matches_tree_walk,
    test_shared_anchors_move_the_row_points_to_the_witness_midpoint,
    test_shared_anchors_warn_with_patch_friction,
    test_shared_anchors_step_on_every_route,
    test_solver_exposes_documented_defaults,
    test_solver_validates_and_stores_contact_controls,
    test_zero_gap_gate_keeps_touching_contacts_only,
    test_friction_gap_max_applies_to_every_route,
    test_speculative_scale_controls_every_position_rhs_family,
    test_dense_restitution_removes_the_scaled_position_bias,
    test_gap_gate_prevents_all_route_allocations,
    test_gap_gate_keeps_contacts_at_threshold,
    test_allocator_skips_contacts_without_response_dofs,
    test_friction_gap_max_keeps_the_constructed_row_layout,
    test_restitution_threshold_cannot_enable_split_restitution,
):
    add_function_test(TestSteelContactControls, _fn.__name__, _fn, devices=devices)
# The test captures the device warning it expects.
add_function_test(
    TestSteelContactControls,
    "test_overflow_warning_enabled_after_construction_latches",
    test_overflow_warning_enabled_after_construction_latches,
    devices=devices,
    check_output=False,
)


DT = 1.0 / 120.0


BOX_MASS = 2.0


def _box_on_ground(device, gravity=None, mu=None):
    """Box resting on the ground."""
    builder = newton.ModelBuilder() if gravity is None else newton.ModelBuilder(gravity=gravity)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    if mu is not None:
        shape_cfg.mu = mu
    # Solid-cube inertia for a 0.2 m box of mass BOX_MASS.
    inertia = wp.mat33(np.eye(3) * BOX_MASS * 0.04 / 6.0)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), mass=BOX_MASS, inertia=inertia
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=shape_cfg.mu))
    return builder.finalize(device=device)


def test_resting_box_reports_weight_as_linear_force(test, device, solve_mode="matrix_free", response="immediate"):
    """Report the resting box's weight through the linear contact force and leave the torque zero."""
    model = _box_on_ground(device)
    solver = SolverSteel(model, solve_mode=solve_mode, iterations=32, articulated_contact_response=response)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(120):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT, observables=observables)
        state_0, state_1 = state_1, state_0

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    # Force on shape0's body by shape1's; orient every record as the force on the box.
    box_shape = int(np.flatnonzero(model.shape_body.numpy() >= 0)[0])
    sign = np.where(shape0 == box_shape, 1.0, -1.0)[:, None]
    wrench = observables.contact_f.numpy()[:count]
    linear = wrench[:, :3] * sign
    weight = BOX_MASS * float(np.linalg.norm(model.gravity.numpy()[0]))
    np.testing.assert_allclose(linear.sum(axis=0), [0.0, 0.0, weight], rtol=2.0e-2, atol=2.0e-2 * weight)

    np.testing.assert_array_equal(wrench[:, 3:], np.zeros((count, 3)))


def _tilted_gravity_box_velocity(device, slope_angle, mu, steps, solve_mode="matrix_free", response="immediate"):
    """Box on the ground under gravity tilted by ``slope_angle``, as on an incline; return its x velocity."""
    g = 9.81
    gravity = (g * np.sin(slope_angle), 0.0, -g * np.cos(slope_angle))
    model = _box_on_ground(device, gravity=gravity, mu=mu)
    solver = SolverSteel(model, solve_mode=solve_mode, iterations=32, articulated_contact_response=response)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return float(state_0.body_qd.numpy()[0, 0])


def test_friction_holds_inside_the_cone(test, device, solve_mode="matrix_free", response="immediate"):
    """Hold the box at rest when the tangential load is inside the friction cone."""
    # tan(0.3) = 0.31 < mu = 0.5.
    velocity = _tilted_gravity_box_velocity(
        device, slope_angle=0.3, mu=0.5, steps=120, solve_mode=solve_mode, response=response
    )
    test.assertLess(abs(velocity), 1.0e-3)


def test_friction_slides_at_the_coulomb_bound(test, device, solve_mode="matrix_free", response="immediate"):
    """Accelerate the box at ``g (sin a - mu cos a)`` when the load exceeds the friction cone."""
    slope_angle, mu, steps = 0.6, 0.3, 120
    velocity = _tilted_gravity_box_velocity(
        device, slope_angle=slope_angle, mu=mu, steps=steps, solve_mode=solve_mode, response=response
    )
    expected = 9.81 * (np.sin(slope_angle) - mu * np.cos(slope_angle)) * steps * DT
    test.assertAlmostEqual(velocity, expected, delta=0.03 * expected)


ARMATURE = (9.0, 4.0, 2.0, 0.5, 1.5, 3.0)


def _sphere_on_ground(device, armature, dense, com=(0.0, 0.0, 0.0), mu=0.0):
    """A 1 kg sphere on a free joint, optionally with a massless fixed child that routes it to dense rows."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(com),
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    joints = [builder.add_joint_free(body)]
    if dense:
        child = builder.add_link(mass=0.0, inertia=wp.mat33(0.0))
        joints.append(builder.add_joint_fixed(body, child))
    builder.add_articulation(joints)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu)
    builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    model = builder.finalize(device=device)
    joint_armature = model.joint_armature.numpy()
    joint_armature[:6] = armature
    model.joint_armature.assign(joint_armature)
    return model


def _impact(model, solver, joint_qd):
    """Step once from the given free-joint velocity; return the new velocity and summed contact force."""
    state_in, state_out = model.state(), model.state()
    qd = state_in.joint_qd.numpy()
    qd[:6] = joint_qd
    state_in.joint_qd.assign(qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    solver.step(state_in, state_out, model.control(), contacts, 0.01, observables=observables)
    count = int(contacts.rigid_contact_count.numpy()[0])
    return state_out.joint_qd.numpy()[:6], observables.contact_f.numpy()[:count, :3].sum(axis=0)


def test_free_body_contact_response_includes_armature(test, device, solve_mode="matrix_free", response="immediate"):
    """Respond on free-body rows with the same armature-augmented inertia as on articulated rows."""
    stop = (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    for armature, expected in ((0.0, 100.0), (9.0, 1000.0)):
        with test.subTest(armature=armature):
            model = _sphere_on_ground(device, armature, dense=False)
            solver = SolverSteel(model, solve_mode=solve_mode, cfm=0.0, articulated_contact_response=response)
            qd, force = _impact(model, solver, stop)
            # The sphere starts 0.1 mm deep, so the depenetration bias adds 0.2 %.
            test.assertAlmostEqual(abs(float(force[2])), expected, delta=5.0e-3 * expected)
            test.assertAlmostEqual(float(qd[2]), 0.0, delta=5.0e-3)

    # An oblique frictional impact of an offset-COM body with per-axis armature.
    oblique = (0.3, -0.2, -1.0, 0.4, 0.5, -0.3)
    results = {}
    for dense in (False, True):
        model = _sphere_on_ground(device, ARMATURE, dense=dense, com=(0.03, -0.02, 0.01), mu=0.5)
        solver = SolverSteel(
            model, solve_mode=solve_mode, cfm=0.0, iterations=64, articulated_contact_response=response
        )
        results[dense] = _impact(model, solver, oblique)
    np.testing.assert_allclose(results[False][0], results[True][0], rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(results[False][1], results[True][1], rtol=1.0e-3, atol=1.0e-2)

    # A notified armature change reaches the free-body response.
    model = _sphere_on_ground(device, 0.0, dense=False)
    solver = SolverSteel(model, solve_mode=solve_mode, cfm=0.0, articulated_contact_response=response)
    _impact(model, solver, stop)
    joint_armature = model.joint_armature.numpy()
    joint_armature[:6] = 9.0
    model.joint_armature.assign(joint_armature)
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    _, force = _impact(model, solver, stop)
    test.assertAlmostEqual(abs(float(force[2])), 1000.0, delta=5.0)


def test_free_body_impact_shares_armature_momentum(test, device, solve_mode="matrix_free", response="immediate"):
    """Conserve the armature-augmented momentum in a frictionless plastic impact between free bodies."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
    for x in (0.0, 0.1999):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01)
        )
        builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    model = builder.finalize(device=device)
    joint_armature = model.joint_armature.numpy()
    joint_armature[:6] = 9.0
    model.joint_armature.assign(joint_armature)
    solver = SolverSteel(model, solve_mode=solve_mode, cfm=0.0, iterations=64, articulated_contact_response=response)
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[0] = 1.0
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    solver.step(state_in, state_out, model.control(), contacts, 0.01)
    qd = state_out.joint_qd.numpy()
    # Effective masses 10 and 1: both move at 10 / 11 m/s (plus a small depenetration bias).
    test.assertAlmostEqual(float(qd[0]), 10.0 / 11.0, delta=2.0e-3)
    test.assertAlmostEqual(float(qd[6]), 10.0 / 11.0, delta=1.5e-2)
    test.assertAlmostEqual(float(10.0 * qd[0] + qd[6]), 10.0, delta=1.0e-3)


ANCHOR_ROTATION = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, 0.2)), 0.25 * np.pi)


def _anchored_sphere(device, anchor_rotation, gravity=None, inertia=(0.003, 0.004, 0.006), max_speed=None):
    """An offset-COM sphere on the ground whose FREE joint has the given parent-anchor rotation."""
    builder = newton.ModelBuilder() if gravity is None else newton.ModelBuilder(gravity=gravity)
    SolverSteel.register_custom_attributes(builder)
    pose = wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity())
    limits = {}
    if max_speed is not None:
        limits = {
            "steel:rigid_body_max_linear_velocity": max_speed[0],
            "steel:rigid_body_max_angular_velocity": max_speed[1],
        }
    body = builder.add_link(
        xform=pose,
        mass=1.0,
        com=wp.vec3(0.03, -0.02, 0.01),
        inertia=wp.mat33(np.diag(inertia)),
        custom_attributes=limits,
    )
    builder.add_articulation([builder.add_joint_free(body, parent_xform=wp.transform_identity())])
    builder.add_shape_sphere(body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    model = builder.finalize(device=device)
    _set_free_anchor(model, anchor_rotation, (0.0,) * 6)
    return model


def _set_free_anchor(model, anchor_rotation, world_twist):
    """Move the FREE joint's parent anchor, keeping the body's world pose and giving it ``world_twist``."""
    pose = wp.transform(*model.body_q.numpy()[0])
    anchor = wp.transform(wp.vec3(0.2, -0.1, 0.3), anchor_rotation)
    joint_X_p = model.joint_X_p.numpy()
    joint_X_p[0] = [*anchor.p, *anchor.q]
    model.joint_X_p.assign(joint_X_p)
    local = wp.transform_inverse(anchor) * pose
    joint_q = model.joint_q.numpy()
    joint_q[:7] = [*local.p, *local.q]
    model.joint_q.assign(joint_q)
    # FREE joint velocities are expressed in the parent anchor frame.
    to_anchor = np.array(wp.quat_to_matrix(anchor_rotation)).reshape(3, 3).T
    joint_qd = model.joint_qd.numpy()
    joint_qd[:3] = to_anchor @ np.asarray(world_twist[:3])
    joint_qd[3:6] = to_anchor @ np.asarray(world_twist[3:])
    model.joint_qd.assign(joint_qd)


def _run_anchored_sphere(model, solver, steps, dt):
    """Step from the model's joint state and return the body's world twist."""
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, model.control(), contacts, dt)
        state_0, state_1 = state_1, state_0
    return state_0.body_qd.numpy()[0]


def _anchored_sphere_cases():
    """An oblique frictional impact, the body rolling off its offset COM under gravity, and velocity limits."""
    anisotropic, isotropic = (0.003, 0.004, 0.006), (0.004, 0.004, 0.004)
    return {
        "impact": ((0.3, -0.2, -1.0, 0.4, 0.5, -0.3), wp.vec3(0.0), 1, 0.01, None, anisotropic),
        "gravity": ((0.0,) * 6, None, 60, DT, None, anisotropic),
        # Box velocity limits act on world axes, and so does their activation gate.
        "velocity_limits": ((0.3, -0.2, -1.0, 0.4, 0.5, -0.3), wp.vec3(0.0), 3, 0.01, (0.35, 0.3), isotropic),
    }


def test_rotated_anchor_free_body_matches_world_aligned(test, device, solve_mode="matrix_free", response="immediate"):
    """Give a FREE body the same world motion whatever the rotation of its parent anchor."""
    for case, (world_twist, gravity, steps, dt, max_speed, inertia) in _anchored_sphere_cases().items():
        twists = []
        for rotation in (wp.quat_identity(), ANCHOR_ROTATION):
            model = _anchored_sphere(device, rotation, gravity, inertia, max_speed)
            _set_free_anchor(model, rotation, world_twist)
            solver = SolverSteel(
                model,
                solve_mode=solve_mode,
                cfm=0.0,
                iterations=64,
                articulated_contact_response=response,
                velocity_limit_activation_fraction=0.5,
            )
            twists.append(_run_anchored_sphere(model, solver, steps, dt))
        with test.subTest(case=case):
            np.testing.assert_allclose(twists[1], twists[0], rtol=0.0, atol=2.0e-4)


def test_rotated_anchor_free_body_on_articulation_matches_world_aligned(
    test, device, solve_mode="matrix_free", response="immediate"
):
    """Give a FREE body rolling on a driven articulated plate the same world motion whatever its anchor rotation."""
    twists = []
    for rotation in (wp.quat_identity(), ANCHOR_ROTATION):
        builder = newton.ModelBuilder()
        plate = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), mass=5.0)
        builder.add_shape_box(plate, hx=0.5, hy=0.5, hz=0.05)
        hinge = builder.add_joint_revolute(
            parent=-1,
            child=plate,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
            target_ke=2.0e3,
            target_kd=50.0,
        )
        builder.add_articulation([hinge])
        pose = wp.transform(wp.vec3(0.1, 0.0, 0.1999), wp.quat_identity())
        ball = builder.add_link(
            xform=pose, mass=1.0, com=wp.vec3(0.03, -0.02, 0.01), inertia=wp.mat33(np.diag([0.003, 0.004, 0.006]))
        )
        builder.add_shape_sphere(ball, radius=0.1)
        anchor = wp.transform(wp.vec3(0.2, -0.1, 0.3), rotation)
        builder.add_articulation([builder.add_joint_free(ball, parent_xform=anchor)])
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        local = wp.transform_inverse(anchor) * pose
        joint_q = model.joint_q.numpy()
        joint_q[1:8] = [*local.p, *local.q]
        model.joint_q.assign(joint_q)
        to_anchor = np.array(wp.quat_to_matrix(rotation)).reshape(3, 3).T
        joint_qd = model.joint_qd.numpy()
        joint_qd[1:4] = to_anchor @ np.array([0.4, 0.2, -0.3])
        joint_qd[4:7] = to_anchor @ np.array([1.0, -2.0, 0.5])
        model.joint_qd.assign(joint_qd)
        solver = SolverSteel(model, solve_mode=solve_mode, iterations=64, articulated_contact_response=response)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        for _ in range(60):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, model.control(), contacts, DT)
            state_0, state_1 = state_1, state_0
        twists.append(state_0.body_qd.numpy())
    np.testing.assert_allclose(twists[1], twists[0], rtol=0.0, atol=1.0e-4)


def test_free_anchor_rotated_after_construction(test, device, solve_mode="matrix_free", response="immediate"):
    """Follow a FREE body's parent anchor rotated after construction, eagerly and in a replayed graph."""
    world_twist, gravity, steps, dt, _, _ = _anchored_sphere_cases()["gravity"]
    rotation = ANCHOR_ROTATION
    # The world-aligned anchor gives the expected world motion.
    model = _anchored_sphere(device, wp.quat_identity(), gravity)
    solver = SolverSteel(model, solve_mode=solve_mode, iterations=64, articulated_contact_response=response)
    expected = _run_anchored_sphere(model, solver, steps, dt)

    model = _anchored_sphere(device, wp.quat_identity(), gravity)
    solver = SolverSteel(model, solve_mode=solve_mode, iterations=64, articulated_contact_response=response)
    _run_anchored_sphere(model, solver, 1, dt)
    _set_free_anchor(model, rotation, world_twist)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    np.testing.assert_allclose(_run_anchored_sphere(model, solver, steps, dt), expected, rtol=0.0, atol=2.0e-4)

    if not wp.get_device(device).is_cuda:
        return
    model = _anchored_sphere(device, wp.quat_identity(), gravity)
    solver = SolverSteel(model, solve_mode=solve_mode, iterations=64, articulated_contact_response=response)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    def step_pair():
        for _ in range(2):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0.assign(state_1)

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    step_pair()
    with wp.ScopedCapture(device) as capture:
        step_pair()
    _set_free_anchor(model, rotation, world_twist)
    state_0.joint_q.assign(model.joint_q)
    state_0.joint_qd.assign(model.joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    with wp.ScopedCapture(device) as notify:
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    wp.capture_launch(notify.graph)
    for _ in range(steps // 2):
        wp.capture_launch(capture.graph)
    np.testing.assert_allclose(state_0.body_qd.numpy()[0], expected, rtol=0.0, atol=2.0e-4)


class TestSteelContactForce(unittest.TestCase):
    pass


split_devices = get_test_devices()


for _name in (
    "test_resting_box_reports_weight_as_linear_force",
    "test_friction_holds_inside_the_cone",
    "test_friction_slides_at_the_coulomb_bound",
    "test_free_body_contact_response_includes_armature",
    "test_free_body_impact_shares_armature_momentum",
    "test_rotated_anchor_free_body_matches_world_aligned",
    "test_rotated_anchor_free_body_on_articulation_matches_world_aligned",
    "test_free_anchor_rotated_after_construction",
):
    add_function_test(TestSteelContactForce, _name, globals()[_name], devices=devices)
    add_function_test(
        TestSteelContactForce, f"{_name}_split", globals()[_name], devices=split_devices, solve_mode="split"
    )
    # The propagation response solves these contacts as body-space rows.
    add_function_test(
        TestSteelContactForce, f"{_name}_propagation", globals()[_name], devices=devices, response="propagation"
    )


# Colored propagation builds the same body-space rows as propagation and solves them differently.
for _name in (
    "test_rotated_anchor_free_body_matches_world_aligned",
    "test_rotated_anchor_free_body_on_articulation_matches_world_aligned",
):
    add_function_test(
        TestSteelContactForce,
        f"{_name}_propagation_colored",
        globals()[_name],
        devices=devices,
        response="propagation_colored",
    )


LOWER_MASS = 2.0


UPPER_MASS = 0.5


FILL = -7.0


def _stacked_boxes(device):
    """Two free boxes stacked on the ground: ground-box and box-box contacts."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for z, half, mass in ((0.1, 0.1, LOWER_MASS), (0.25, 0.05, UPPER_MASS)):
        inertia = wp.mat33(np.eye(3) * mass * (2.0 * half) ** 2 / 6.0)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), mass=mass, inertia=inertia
        )
        builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    return model, pipeline


def _net_body_forces(model, contacts, contact_f):
    """Sum each body's contact forces from the per-contact rows (force on shape 0's body)."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
    body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
    linear = contact_f[:count, :3]
    net = np.zeros((model.body_count, 3))
    for body in range(model.body_count):
        net[body] = linear[body0 == body].sum(axis=0) - linear[body1 == body].sum(axis=0)
    return net


def _check_rows(test, contacts, contact_f):
    """Live rows hold a finite linear force and no torque; every other row is zero."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    test.assertTrue(np.isfinite(contact_f[:count, :3]).all())
    np.testing.assert_array_equal(contact_f[:count, 3:], 0.0)
    np.testing.assert_array_equal(contact_f[count:], 0.0)


def test_contact_f_reports_each_box_weight(test, device, steps=150, torsion_radius=None, **options):
    """Report linear-only per-contact forces that sum to each box's weight at rest."""
    model, pipeline = _stacked_boxes(device)
    if torsion_radius is not None:
        set_torsion_radius(model, torsion_radius)
    solver = SolverSteel(model, iterations=32, **options)
    contacts = pipeline.contacts()
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    test.assertEqual(observables.contact_f.shape[0], model.rigid_contact_max + model.soft_contact_max)
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    states = [model.state(), model.state()]
    control = model.control()
    for _ in range(steps):
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        states.reverse()
    contact_f = observables.contact_f.numpy()
    _check_rows(test, contacts, contact_f)
    if options.get("free_body_constraint_max") is not None:
        # Rows beyond the capacity were dropped; those contacts report zero.
        count = int(contacts.rigid_contact_count.numpy()[0])
        dropped = solver._contact_slot.numpy()[:count] < 0
        test.assertTrue(dropped.any())
        np.testing.assert_array_equal(contact_f[:count][dropped], 0.0)
    else:
        weight = np.linalg.norm(model.gravity.numpy()[0])
        net = _net_body_forces(model, contacts, contact_f)
        np.testing.assert_allclose(net[0], [0.0, 0.0, LOWER_MASS * weight], atol=0.03 * LOWER_MASS * weight)
        np.testing.assert_allclose(net[1], [0.0, 0.0, UPPER_MASS * weight], atol=0.03 * UPPER_MASS * weight)


def test_unrequested_contact_f_is_left_untouched(test, device):
    """Write nothing when the observable is allocated but not selected for the step."""
    model, pipeline = _stacked_boxes(device)
    solver = SolverSteel(model, solve_mode="matrix_free" if wp.get_device(device).is_cuda else "split")
    contacts = pipeline.contacts()
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DT, observables=observables.select(set()))
    np.testing.assert_array_equal(observables.contact_f.numpy(), FILL)
    solver.step(state_in, state_out, model.control(), contacts, DT, observables=observables)
    test.assertFalse(np.any(observables.contact_f.numpy() == FILL))


def test_contact_f_in_a_captured_graph(test, device):
    """Write the observable from the replayed step's impulses in a captured graph."""
    model, pipeline = _stacked_boxes(device)
    solver = SolverSteel(model, iterations=32)
    contacts = pipeline.contacts()
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    states = [model.state(), model.state()]
    control = model.control()

    def step():
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        pipeline.collide(states[1], contacts)
        solver.step(states[1], states[0], control, contacts, DT, observables=observables)

    step()
    with wp.ScopedCapture(device) as recorded:
        step()
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    for _ in range(60):
        wp.capture_launch(recorded.graph)
    _check_rows(test, contacts, observables.contact_f.numpy())
    weight = np.linalg.norm(model.gravity.numpy()[0])
    net = _net_body_forces(model, contacts, observables.contact_f.numpy())
    np.testing.assert_allclose(net[1, 2], UPPER_MASS * weight, rtol=0.03)


def test_contact_f_of_sleeping_islands(test, device, skip_constraints=True):
    """Report a sleeping island's contact forces: zero with skipped rows, else the resting force."""
    model, pipeline = _stacked_boxes(device)
    solver = SolverSteel(
        model,
        iterations=32,
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        sleep_skip_constraints=skip_constraints,
        friction_anchor_beta=0.0,
    )
    contacts = pipeline.contacts()
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    states = [model.state(), model.state()]
    control = model.control()
    for step in range(150):
        if step == 149:
            observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        states.reverse()
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    contact_f = observables.contact_f.numpy()
    _check_rows(test, contacts, contact_f)
    net = _net_body_forces(model, contacts, contact_f)
    if skip_constraints:
        np.testing.assert_array_equal(net, 0.0)
    else:
        weight = np.linalg.norm(model.gravity.numpy()[0])
        np.testing.assert_allclose(net[0, 2], LOWER_MASS * weight, rtol=0.03)
        np.testing.assert_allclose(net[1, 2], UPPER_MASS * weight, rtol=0.03)


def test_contact_f_of_compliant_hydroelastic_rows(test, device):
    """Report the force of hydroelastic contacts solved with contact compliance."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.5,
            is_hydroelastic=True,
            sdf_max_resolution=32,
            sdf_narrow_band_range=(-0.01, 0.01),
            sdf_padding=0.006,
            gap=0.005,
            kh=1e7,
        )
        builder.add_shape_box(
            body=-1, hx=0.1, hy=0.1, hz=0.025, xform=wp.transform(wp.vec3(0, 0, -0.025), wp.quat_identity())
        )
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.048), wp.quat_identity()))
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 0.3 / (4 / 3 * np.pi * 0.05**3)
        builder.add_shape_sphere(body, radius=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            rigid_contact_max=512,
            deterministic=True,
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                reduce_contacts=True, moment_matching=True, anchor_contact=True, buffer_fraction=1.0
            ),
        )
        solver = SolverSteel(
            model, contact_compliance=True, friction_anchor_beta=0.0, iterations=64, free_body_constraint_max=2048
        )
        contacts = pipeline.contacts()
        observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
        states = [model.state(), model.state()]
        newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
        for _ in range(5):
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], model.control(), contacts, 0.0025, observables=observables)
            states.reverse()
        test.assertGreater(solver.compliance_contact_count, 0)
        contact_f = observables.contact_f.numpy()
        _check_rows(test, contacts, contact_f)
        test.assertGreater(np.abs(contact_f[:, 2]).sum(), 0.0)


def test_overflowing_contact_count_stays_in_the_rigid_rows(test, device, capture=False):
    """Bound rigid rows by the capacity when the contact count overflows it, leaving soft rows zero."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    # Built before the pipeline, the solver's contact scratch is larger than the contact buffer.
    solve_mode = "matrix_free" if wp.get_device(device).is_cuda else "split"
    solver = SolverSteel(model, solve_mode=solve_mode, warn_constraint_overflow=False)
    rigid_capacity, soft_capacity = 4, 2
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=rigid_capacity, soft_contact_max=soft_capacity)
    contacts = pipeline.contacts()
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    test.assertEqual(observables.contact_f.shape[0], rigid_capacity + soft_capacity)
    state_in, state_out, control = model.state(), model.state(), model.control()
    overflow_count = wp.array([rigid_capacity + soft_capacity + 3], dtype=wp.int32, device=device)

    def step():
        pipeline.collide(state_in, contacts)
        # The collision counter keeps counting past the capacity when the buffer overflows.
        wp.copy(contacts.rigid_contact_count, overflow_count)
        solver.step(state_in, state_out, control, contacts, DT, observables=observables)

    pipeline.collide(state_in, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), rigid_capacity)
    step()
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    if capture:
        with wp.ScopedCapture(device) as recorded:
            step()
        observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
        wp.capture_launch(recorded.graph)
    else:
        step()
    contact_f = observables.contact_f.numpy()
    test.assertTrue(np.isfinite(contact_f[:rigid_capacity, :3]).all())
    test.assertFalse(np.any(contact_f[:rigid_capacity] == FILL))
    np.testing.assert_array_equal(contact_f[:rigid_capacity, 3:], 0.0)
    np.testing.assert_array_equal(contact_f[rigid_capacity:], 0.0)


def test_contact_force_kernel_reads_only_rigid_rows(test, device):
    """Read no rigid contact data past the capacity when the count overflows into the soft rows."""
    rigid_capacity, soft_capacity = 1, 2
    rows = rigid_capacity + soft_capacity
    backing = []

    def rigid_view(values, dtype):
        # Valid storage past the view makes a read beyond the rigid capacity deterministic.
        full = wp.array(values, dtype=dtype, device=device)
        backing.append(full)
        return wp.array(ptr=full.ptr, shape=(rigid_capacity,), dtype=dtype, device=device)

    impulses = wp.array([[3.0]], dtype=wp.float32, device=device)
    row_count = wp.array([1], dtype=wp.int32, device=device)
    contact_f = wp.full(rows, wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL), device=device)
    wp.launch(
        newton._src.solvers.steel.kernels.compute_contact_spatial_force_from_impulses,
        dim=rows,
        inputs=[
            wp.array([rows + 4], dtype=wp.int32, device=device),
            rigid_view([[0.0, 0.0, -1.0]] * rows, wp.vec3),
            rigid_view([0] * rows, wp.int32),
            rigid_view([0] * rows, wp.int32),
            rigid_view([0] * rows, wp.int32),
            rigid_view([1] * rows, wp.int32),
            impulses,
            impulses,
            impulses,
            row_count,
            row_count,
            row_count,
            10.0,
            rigid_capacity,
        ],
        outputs=[contact_f],
        device=device,
    )
    np.testing.assert_allclose(contact_f.numpy()[0], [0.0, 0.0, 30.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(contact_f.numpy()[rigid_capacity:], 0.0)


class TestSteelContactObservable(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()


all_devices = get_test_devices()


add_function_test(
    TestSteelContactObservable,
    "test_contact_f_reports_each_box_weight",
    test_contact_f_reports_each_box_weight,
    devices=cuda_devices,
)


add_function_test(
    TestSteelContactObservable,
    "test_contact_f_reports_each_box_weight_split",
    test_contact_f_reports_each_box_weight,
    devices=all_devices,
    solve_mode="split",
)


for _suffix, _options in (
    ("propagation", {"articulated_contact_response": "propagation"}),
    ("point_friction", {"friction_anchor_beta": 0.0}),
    ("torsion", {"torsion_radius": 0.01}),
    ("device_torsion", {"torsion_radius": 0.01, "contact_torsion_device": True}),
    ("double_buffer", {"double_buffer": True}),
    ("row_overflow", {"free_body_constraint_max": 6, "warn_constraint_overflow": False}),
):
    add_function_test(
        TestSteelContactObservable,
        f"test_contact_f_reports_each_box_weight_{_suffix}",
        test_contact_f_reports_each_box_weight,
        devices=cuda_devices,
        **_options,
    )


add_function_test(
    TestSteelContactObservable,
    "test_unrequested_contact_f_is_left_untouched",
    test_unrequested_contact_f_is_left_untouched,
    devices=all_devices,
)


add_function_test(
    TestSteelContactObservable,
    "test_overflowing_contact_count_stays_in_the_rigid_rows",
    test_overflowing_contact_count_stays_in_the_rigid_rows,
    devices=all_devices,
)


add_function_test(
    TestSteelContactObservable,
    "test_overflowing_contact_count_stays_in_the_rigid_rows_captured",
    test_overflowing_contact_count_stays_in_the_rigid_rows,
    devices=cuda_devices,
    capture=True,
)


add_function_test(
    TestSteelContactObservable,
    "test_contact_force_kernel_reads_only_rigid_rows",
    test_contact_force_kernel_reads_only_rigid_rows,
    devices=all_devices,
)


add_function_test(
    TestSteelContactObservable,
    "test_contact_f_in_a_captured_graph",
    test_contact_f_in_a_captured_graph,
    devices=cuda_devices,
)


add_function_test(
    TestSteelContactObservable,
    "test_contact_f_of_sleeping_islands",
    test_contact_f_of_sleeping_islands,
    devices=cuda_devices,
)


add_function_test(
    TestSteelContactObservable,
    "test_contact_f_of_sleeping_islands_without_skipped_rows",
    test_contact_f_of_sleeping_islands,
    devices=cuda_devices,
    skip_constraints=False,
)


add_function_test(
    TestSteelContactObservable,
    "test_contact_f_of_compliant_hydroelastic_rows",
    test_contact_f_of_compliant_hydroelastic_rows,
    devices=cuda_devices,
)


_DEVICE = "cuda:0"


def _cylinder_foot(builder, pos, mass=1.0, num_cyl=7, radius=0.02, half_height=0.015):
    """Add one free body whose collision is ``num_cyl`` small cylinders in a row.

    All cylinders reach the ground at nearly the same height, so a plane contact
    produces a multiple of ``num_cyl`` contacts for a single region.
    """
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass)
    for i in range(num_cyl):
        x = (i - (num_cyl - 1) / 2.0) * (2.2 * radius)
        builder.add_shape_cylinder(
            body,
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            radius=radius,
            half_height=half_height,
        )
    return body


@unittest.skipUnless(wp.is_cuda_available(), "SolverSteel requires CUDA")
class TestSteelPersistentContacts(unittest.TestCase):
    def test_combined_history_reset_replays_with_changed_world_mask(self):
        """Capture contact-matching and solver resets with a live device mask."""
        template = newton.ModelBuilder()
        _cylinder_foot(template, wp.vec3(0.0, 0.0, 0.015))
        builder = newton.ModelBuilder()
        builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize(device=_DEVICE)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        solver = SolverSteel(model, solve_mode="matrix_free", contact_warmstart=True)
        state, output = model.state(), model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        # The solver and the pipeline share the (world_count + 1,) reset-mask layout.
        mask = wp.array([False, False, False], dtype=wp.bool, device=model.device)
        # Compile both reset paths before capture.
        solver.reset(state, mask)
        pipeline.reset_contact_matching(mask)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.reset(state, mask)
            pipeline.reset_contact_matching(mask)
            pipeline.collide(state, contacts)
        for selected in (0, 1):
            values = np.zeros(3, dtype=bool)
            values[selected] = True
            mask.assign(values)
            solver._ws_prev_mf_impulses.fill_(1.0)
            wp.capture_launch(capture.graph)
            previous = solver._ws_prev_mf_impulses.numpy()
            self.assertTrue(np.all(previous[selected] == 0.0))
            self.assertTrue(np.all(previous[1 - selected] == 1.0))
            count = int(contacts.rigid_contact_count.numpy()[0])
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            worlds = np.maximum(model.shape_world.numpy()[shape0], model.shape_world.numpy()[shape1])
            matches = contacts.rigid_contact_match_index.numpy()[:count]
            self.assertTrue(np.any(worlds == selected) and np.any(worlds == 1 - selected))
            self.assertTrue(np.all(matches[worlds == selected] < 0))
            self.assertTrue(np.all(matches[worlds == 1 - selected] >= 0))

    def _seed_rotating_contact(self, friction, *, oblique=False, rotated=False, device=_DEVICE):
        """Prepare a cached friction impulse and move its normal across a basis seam."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.X, gravity=wp.vec3(0.0))
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        model.shape_material_mu.fill_(1.0)
        solver = SolverSteel(model, solve_mode="matrix_free", contact_warmstart=True, iterations=0)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        pipeline.collide(state_in, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        contacts.rigid_contact_count.fill_(1)
        old_normal = wp.normalize(wp.vec3(-1.0, 0.001, 0.0))
        new_normal = wp.normalize(wp.vec3(-1.0, -0.001, 0.0))
        if oblique:
            old_normal = wp.normalize(wp.vec3(0.2, 0.3, 0.93))
            new_normal = wp.normalize(wp.vec3(0.3, 0.2, 0.93))
        if rotated:
            old_normal = wp.vec3(0.0, 0.0, 1.0)
            new_normal = wp.vec3(0.5 / np.sqrt(2.0), 0.5 / np.sqrt(2.0), np.sqrt(3.0) / 2.0)
        normals = contacts.rigid_contact_normal.numpy()
        normals[0] = old_normal
        contacts.rigid_contact_normal.assign(normals)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        # Material edits reach the solver through a notification, which discards the carried
        # contact impulses; keep their row layout so both carriers are seeded below.
        row_type = solver._ws_prev_mf_row_type.numpy()
        row_parent = solver._ws_prev_mf_row_parent.numpy()
        model.shape_material_mu.fill_(friction)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        solver._ws_prev_mf_row_type.assign(row_type)
        solver._ws_prev_mf_row_parent.assign(row_parent)
        slot = int(solver._contact_slot.numpy()[0])
        self.assertGreaterEqual(slot, 0)
        jacobian = solver._mf_J_a if int(solver._mf_body_a.numpy()[0, slot]) >= 0 else solver._mf_J_b
        old_direction = jacobian.numpy()[0, slot + 1, :3].copy()
        previous = solver._ws_prev_mf_impulses.numpy()
        previous[0, slot : slot + 3] = (1.0, 0.5, 0.0)
        solver._ws_prev_mf_impulses.assign(previous)
        # Seed both owners of the previous solve's tangent impulse. Persistent
        # anchors take precedence over the contact-matched cache when carried.
        patches = solver._friction_patches
        tangent, _ = contact_tangent_basis(-old_normal)
        tangent *= 0.5
        if patches.current.flipped.numpy()[0]:
            tangent = -tangent
        anchor_body = int(patches.current.body_a.numpy()[0])
        if anchor_body >= 0:
            pose = state_in.body_q.numpy()[anchor_body]
            transform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
            tangent = wp.transform_vector(wp.transform_inverse(transform), tangent)
        cached = patches.previous.tangent_impulse.numpy()
        cached[0] = tangent
        patches.previous.tangent_impulse.assign(cached)
        normals[0] = new_normal
        contacts.rigid_contact_normal.assign(normals)
        contacts.rigid_contact_match_index.fill_(-1)
        indices = contacts.rigid_contact_match_index.numpy()
        indices[0] = 0
        contacts.rigid_contact_match_index.assign(indices)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        slot = int(solver._contact_slot.numpy()[0])
        impulses = solver._mf_impulses.numpy()[0, slot : slot + 3]
        tangent_rows = jacobian.numpy()[0, slot + 1 : slot + 3, :3]
        world_tangent = impulses[1:] @ tangent_rows
        # Independent world-space projection; solver rows use B-to-A normals.
        n = np.asarray(new_normal)
        projected_direction = old_direction - n * np.dot(n, old_direction)
        return impulses, world_tangent, projected_direction

    def test_cached_friction_preserves_world_direction(self):
        """Transport a cached tangent impulse through a discontinuous tangent basis."""
        _, world_tangent, old_direction = self._seed_rotating_contact(1.0)
        np.testing.assert_allclose(world_tangent, 0.5 * old_direction, atol=1.0e-6)

    def test_cached_friction_uses_solver_normal_convention(self):
        """Project friction in the frame used by the actual contact Jacobians."""
        _, world_tangent, projected_direction = self._seed_rotating_contact(1.0, oblique=True)
        np.testing.assert_allclose(world_tangent, 0.5 * projected_direction, atol=1.0e-6)

    def test_cached_friction_obeys_changed_material(self):
        """Clamp carried friction to the current material before applying velocity."""
        impulses, world_tangent, old_direction = self._seed_rotating_contact(0.1)
        self.assertLessEqual(float(np.linalg.norm(impulses[1:])), 0.1 * impulses[0] + 1.0e-6)
        np.testing.assert_allclose(world_tangent, 0.1 * old_direction, atol=1.0e-6)

    def test_rotating_contact_clamps_the_projected_friction_cone(self):
        """Project through a 30-degree normal change and saturate the current cone."""
        impulses, tangent, projected = self._seed_rotating_contact(0.1, rotated=True)
        expected = 0.1 * projected / np.linalg.norm(projected)
        np.testing.assert_allclose(tangent, expected, atol=1.0e-6)
        self.assertAlmostEqual(float(np.linalg.norm(impulses[1:])), 0.1 * impulses[0], delta=1.0e-6)


def test_matching_retains_world_distance_policy(test, device):
    """Cold-start contacts that move beyond the world-distance threshold."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for z in (0.1, 0.29):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    poses = state.body_q.numpy()
    poses[:, 0] += 1.0
    state.body_q.assign(poses)
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), count)
    test.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] < 0))


add_function_test(
    TestSteelPersistentContacts,
    "test_matching_retains_world_distance_policy",
    test_matching_retains_world_distance_policy,
    devices=get_test_devices(),
)


DEFAULT_DT = 1.0e-3


IMPACT_FRACTION = 0.4


# Values written into ``SolverSteel.contact_path``.
PATH_DENSE = 0


# The route each scene takes: a lone free body uses the free-body rows, and a
# contact on an articulated root uses the dense rows.
EXPECTED_PATHS = {"free": {PATH_MATRIX_FREE}, "articulated": {PATH_DENSE}}


def _make_solver(
    model,
    *,
    velocity_iterations=0,
    restitution_velocity_threshold=0.0,
    iterations=16,
    contact_warmstart=False,
    enable_restitution=True,
):
    """Construct a deterministic frictionless Steel test solver without angular damping."""
    kwargs = {
        "iterations": iterations,
        "velocity_iterations": velocity_iterations,
        "cfm": 0.0,
        "relaxation": 1.0,
        "contact_warmstart": contact_warmstart,
        "enable_restitution": enable_restitution,
    }
    if restitution_velocity_threshold is not None:
        kwargs["restitution_velocity_threshold"] = restitution_velocity_threshold
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **kwargs)
    solver._rigid_body_angular_damping.zero_()
    return solver


def _step_plane(
    device,
    *,
    separation,
    speed,
    restitution,
    dt=DEFAULT_DT,
    mass=1.0,
    plane_restitution=None,
    gravity=(0.0, 0.0, 0.0),
    initial_vertical_velocity=None,
    scene="free",
    velocity_iterations=0,
    restitution_velocity_threshold=0.0,
    iterations=16,
    enable_restitution=True,
):
    """Run one sphere-plane step and return observable contact and state data."""
    model, body = _build_plane_model(
        device,
        separation=separation,
        restitution=restitution,
        plane_restitution=plane_restitution,
        mass=mass,
        gravity=gravity,
        scene=scene,
    )
    solver = _make_solver(
        model,
        velocity_iterations=velocity_iterations,
        restitution_velocity_threshold=restitution_velocity_threshold,
        iterations=iterations,
        enable_restitution=enable_restitution,
    )
    state_in, state_out = model.state(), model.state()
    vz = -speed if initial_vertical_velocity is None else initial_vertical_velocity
    _reset_state(model, state_in, vz)
    before = float(state_in.body_qd.numpy()[body][2])

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, dt)

    count = int(contacts.rigid_contact_count.numpy()[0])
    active_paths = {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0}
    qd = state_out.body_qd.numpy()[body].astype(np.float64)
    q = state_out.body_q.numpy()[body].astype(np.float64)
    return {
        "before": before,
        "after": float(qd[2]),
        "body_qd": qd,
        "end_gap": float(q[2] - RADIUS),
        "contact_count": count,
        "paths": active_paths,
    }


def _assert_velocity(test, actual, expected, scale, label, *, rtol=2.0e-3, atol=2.0e-4):
    """Assert a scalar velocity with a scale-aware float32 tolerance."""
    test.assertAlmostEqual(
        actual,
        expected,
        delta=atol + rtol * max(abs(scale), abs(expected)),
        msg=f"{label}: got {actual:+.7f} m/s, expected {expected:+.7f} m/s",
    )


def _assert_crossing_setup(test, result, separation, speed, dt, label):
    """Prove that a test generated a positive-gap contact crossing this step."""
    test.assertGreater(separation, 0.0, f"{label}: contact did not start separated")
    test.assertLess(separation, speed * dt, f"{label}: contact could not reach the surface this step")
    test.assertEqual(result["contact_count"], 1, f"{label}: expected exactly one contact")
    _assert_velocity(test, result["before"], -speed, speed, f"{label}, incident state", rtol=1.0e-6, atol=1.0e-6)


def _assert_restitution_row_position(test, result, separation, speed, restitution, dt, label):
    """Assert the single-state PGS position law and its known time-of-impact error."""
    expected_gap = separation + restitution * speed * dt
    test.assertAlmostEqual(
        result["end_gap"],
        expected_gap,
        delta=max(5.0e-6, 3.0e-3 * expected_gap),
        msg=f"{label}: ordinary restitution row used an unexpected position trajectory",
    )
    physical_gap = restitution * (speed * dt - separation)
    expected_error = (1.0 + restitution) * separation
    test.assertAlmostEqual(
        result["end_gap"] - physical_gap,
        expected_error,
        delta=max(8.0e-6, 4.0e-3 * expected_error),
        msg=f"{label}: phase-dependent position error changed unexpectedly",
    )


def test_one_step_restitution_law_over_coefficient_speed_and_mass(test, device):
    """Enforce Newton's impact law across coefficient, speed, and mass scales."""
    cases = []
    cases.extend((f"e={e}", e, 4.0, 1.0) for e in (0.25, 0.6, 1.0))
    cases.extend((f"speed={speed}", 0.6, speed, 1.0) for speed in (0.05, 2.0, 20.0))
    cases.extend((f"mass={mass}", 0.6, 4.0, mass) for mass in (1.0e-6, 1.0, 1.0e6))

    for label, restitution, speed, mass in cases:
        with test.subTest(case=label):
            separation = IMPACT_FRACTION * speed * DEFAULT_DT
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                mass=mass,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
            expected = restitution * speed
            _assert_velocity(test, result["after"], expected, speed, label)
            _assert_restitution_row_position(test, result, separation, speed, restitution, DEFAULT_DT, label)
            energy_ratio = (result["after"] / speed) ** 2
            test.assertAlmostEqual(
                energy_ratio,
                restitution * restitution,
                delta=5.0e-3,
                msg=f"{label}: kinetic-energy ratio was {energy_ratio:.6f}",
            )


def test_crossing_result_is_invariant_to_dt_and_time_of_impact(test, device):
    """Use incident speed rather than gap-clipped velocity at every impact time."""
    restitution = 0.65
    speed = 3.0
    cases = (
        # At impact_fraction -> 0 the single-state position law coincides with
        # the analytical rebound (both e*speed*dt), so the first case anchors
        # the position assertions to exact ground truth; the rest pin the law
        # and its known (1+e)*separation deviation across the step phase.
        (1.0e-3, 0.01),
        (2.5e-4, 0.1),
        (2.5e-4, 0.9),
        (1.0e-3, 0.5),
        (1.0e-2, 0.1),
        (1.0e-2, 0.9),
    )
    for iterations in (1, 2, 8):
        for dt, impact_fraction in cases:
            label = f"iters={iterations}, dt={dt:g}, impact/dt={impact_fraction:g}"
            with test.subTest(iterations=iterations, dt=dt, impact_fraction=impact_fraction):
                separation = speed * dt * impact_fraction
                result = _step_plane(
                    device,
                    separation=separation,
                    speed=speed,
                    restitution=restitution,
                    dt=dt,
                    iterations=iterations,
                )
                _assert_crossing_setup(test, result, separation, speed, dt, label)
                _assert_velocity(test, result["after"], restitution * speed, speed, label)
                _assert_restitution_row_position(test, result, separation, speed, restitution, dt, label)


def test_penetrating_impact_replaces_baumgarte_with_restitution(test, device):
    """Keep penetration correction out of the qualifying restitution target."""
    restitution = 0.5
    speed = 2.0
    for penetration in (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2):
        label = f"penetration={penetration:g}"
        with test.subTest(penetration=penetration):
            result = _step_plane(
                device,
                separation=-penetration,
                speed=speed,
                restitution=restitution,
                velocity_iterations=0,
                iterations=8,
            )
            test.assertEqual(result["contact_count"], 1, f"{label}: expected exactly one contact")
            test.assertEqual(result["paths"], {PATH_MATRIX_FREE}, f"{label}: wrong contact route")
            _assert_velocity(test, result["before"], -speed, speed, f"{label}, incident state")
            _assert_velocity(test, result["after"], restitution * speed, speed, label)


def test_restitution_uses_the_post_force_incident_predictor(test, device):
    """Include this step's known gravity impulse in the frozen incident speed."""
    restitution = 0.6
    initial_speed = 2.0
    gravity = 9.81
    dt = 1.0e-2
    incident_speed = initial_speed + gravity * dt
    separation = IMPACT_FRACTION * incident_speed * dt
    result = _step_plane(
        device,
        separation=separation,
        speed=initial_speed,
        restitution=restitution,
        dt=dt,
        gravity=(0.0, 0.0, -gravity),
    )

    test.assertEqual(result["contact_count"], 1, "gravity case generated no unique contact")
    _assert_velocity(test, result["before"], -initial_speed, initial_speed, "gravity, authored velocity")
    _assert_restitution_row_position(
        test, result, separation, incident_speed, restitution, dt, "gravity, post-force incident predictor"
    )
    _assert_velocity(
        test,
        result["after"],
        restitution * incident_speed,
        incident_speed,
        "gravity, post-force incident predictor",
    )


def test_non_crossing_speculative_contact_does_not_bounce(test, device):
    """Preserve free approach when a speculative contact cannot impact this step."""
    restitution = 1.0
    speed = 3.0
    separation = 1.25 * speed * DEFAULT_DT
    result = _step_plane(
        device,
        separation=separation,
        speed=speed,
        restitution=restitution,
    )
    test.assertGreater(result["contact_count"], 0, "the speculative contact was never generated")
    test.assertGreater(separation - speed * DEFAULT_DT, 0.0, "the setup unexpectedly crosses the surface")
    _assert_velocity(test, result["after"], -speed, speed, "non-crossing contact")
    test.assertGreater(result["end_gap"], 0.0, "the non-crossing body reached or crossed the plane")


def test_shape_restitution_uses_symmetric_arithmetic_average(test, device):
    """Average sanitized shape coefficients symmetrically, including authored zero."""
    speed = 4.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    cases = (
        (0.0, 0.8, 0.4),
        (0.2, 0.8, 0.5),
        (0.8, 0.2, 0.5),
        (2.0, 0.8, 0.9),
        (-0.2, 0.8, 0.4),
        (float("nan"), 0.8, 0.4),
    )
    for sphere_e, plane_e, mixed in cases:
        label = f"sphere={sphere_e}, plane={plane_e}"
        with test.subTest(sphere=sphere_e, plane=plane_e):
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=sphere_e,
                plane_restitution=plane_e,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
            _assert_velocity(test, result["after"], mixed * speed, speed, label)


def test_restitution_velocity_threshold_uses_incident_relative_speed(test, device):
    """Suppress low-speed bounce and activate immediately above the threshold."""
    threshold = 0.5
    restitution = 0.8
    for speed in (0.25, 0.49, 0.51, 2.0):
        label = f"speed={speed}, threshold={threshold}"
        with test.subTest(speed=speed):
            separation = IMPACT_FRACTION * speed * DEFAULT_DT
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                restitution_velocity_threshold=threshold,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
            expected = restitution * speed if speed > threshold else -separation / DEFAULT_DT
            _assert_velocity(test, result["after"], expected, speed, label)
            if speed > threshold:
                _assert_restitution_row_position(test, result, separation, speed, restitution, DEFAULT_DT, label)
            else:
                test.assertLess(abs(result["end_gap"]), 5.0e-6, f"{label}: speculative row missed the surface")

    # Both bodies move quickly in the world frame, but their closing speed is
    # below the threshold. This catches a gate applied to either body's
    # absolute speed instead of the contact's relative normal speed.
    out_a, out_b, _distance, count, _paths = _step_two_spheres(
        device,
        mass_a=1.0,
        mass_b=1.0,
        velocity_a=10.0,
        velocity_b=9.6,
        restitution=restitution,
        restitution_velocity_threshold=threshold,
    )
    test.assertEqual(count, 1, "relative-threshold test generated no unique contact")
    # Restitution is suppressed, but the pair still closes the remaining gap
    # under the speculative allowance and arrives touching with relative speed
    # separation/dt, the impulse shared equally between the equal masses -- the
    # same rule the single-body sub-cases assert above. Expecting a common 9.8
    # would freeze the bodies mid-air while still separated.
    residual_closing = IMPACT_FRACTION * (10.0 - 9.6)
    _assert_velocity(test, out_a, 9.8 + 0.5 * residual_closing, 0.4, "relative threshold, body A")
    _assert_velocity(test, out_b, 9.8 - 0.5 * residual_closing, 0.4, "relative threshold, body B")
    test.assertAlmostEqual(out_a + out_b, 19.6, delta=1.0e-4, msg="relative threshold: momentum drifted")


def test_disabled_restitution_needs_no_material_edits(test, device):
    """Give the zero-restitution result with enable_restitution=False, leaving materials unchanged.

    Zeroing one shape's coefficient still leaves the average.
    """
    speed = 2.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    for scene in ("free", "articulated"):
        with test.subTest(scene=scene):
            common = {"separation": separation, "speed": speed, "scene": scene}
            bouncing = _step_plane(device, restitution=0.8, plane_restitution=0.8, **common)
            one_zero = _step_plane(device, restitution=0.0, plane_restitution=0.8, **common)
            inelastic = _step_plane(device, restitution=0.0, plane_restitution=0.0, **common)
            disabled = _step_plane(device, restitution=0.8, plane_restitution=0.8, enable_restitution=False, **common)
            _assert_velocity(test, bouncing["after"], 0.8 * speed, speed, f"{scene}: authored rebound")
            _assert_velocity(test, one_zero["after"], 0.4 * speed, speed, f"{scene}: averaged rebound")
            test.assertAlmostEqual(disabled["after"], inelastic["after"], delta=1.0e-6 * speed)
            test.assertAlmostEqual(disabled["end_gap"], inelastic["end_gap"], delta=1.0e-7)


def test_bouncing_ball_settles_to_rest_under_gravity(test, device):
    """Settle a bouncing ball on the surface without perpetual micro-bounce or sinking.

    Restitution shares the contact row with the speculative/Baumgarte law, so
    the long-run interaction with resting contact is a distinct failure mode
    from any single-impact assertion: a bounce target that keeps firing at
    resting speeds produces a Zeno micro-bounce, while an over-eager gate
    lets the row sink. Impacts decay 2.0 -> 1.0 -> 0.5 m/s, the third landing
    falls at the 0.5 m/s threshold and is suppressed, and the tail of the
    rollout must be quiescent on the surface.
    """
    restitution = 0.5
    speed = 2.0
    gravity = 9.81
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    model, body = _build_plane_model(
        device,
        separation=separation,
        restitution=restitution,
        gravity=(0.0, 0.0, -gravity),
    )
    solver = _make_solver(model, restitution_velocity_threshold=0.5)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    _reset_state(model, state_in, -speed)

    heights, speeds = [], []
    for _ in range(800):
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)
        state_in, state_out = state_out, state_in
        heights.append(float(state_in.body_q.numpy()[body][2]))
        speeds.append(float(state_in.body_qd.numpy()[body][2]))
    heights = np.array(heights)
    speeds = np.array(speeds)

    # Non-vacuity: the first impact must actually rebound at ~e*speed.
    test.assertGreater(float(speeds[:100].max()), 0.9 * restitution * speed, "the ball never bounced")
    test.assertLess(float(heights.min()), RADIUS + 1.0e-3, "the ball never reached the surface")
    test.assertGreater(float(heights.min()), RADIUS - 2.0e-3, "the ball sank through the plane")
    # Quiescence: by 0.6 s every impact is below the threshold, so the tail
    # must rest on the surface instead of bouncing or creeping.
    tail_heights = heights[-200:]
    tail_speeds = speeds[-200:]
    test.assertLess(float(np.abs(tail_speeds).max()), 0.1, "resting contact kept bouncing")
    test.assertLess(float(tail_heights.max()), RADIUS + 1.5e-3, "resting contact hovered above the surface")
    test.assertGreater(float(tail_heights.min()), RADIUS - 1.5e-3, "resting contact sank into the plane")


def _build_two_sphere_model(device, *, separation, restitution, mass_a, mass_b):
    """Build two free spheres closing along their line of centres."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = _shape_cfg(restitution)
    body_a = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=mass_a,
        inertia=_sphere_inertia(mass_a),
        lock_inertia=True,
    )
    body_b = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * RADIUS + separation), wp.quat_identity()),
        mass=mass_b,
        inertia=_sphere_inertia(mass_b),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body_a, radius=RADIUS, cfg=cfg)
    builder.add_shape_sphere(body_b, radius=RADIUS, cfg=cfg)
    return builder.finalize(device=device), body_a, body_b


def _step_two_spheres(
    device,
    *,
    mass_a,
    mass_b,
    velocity_a,
    velocity_b,
    restitution,
    restitution_velocity_threshold=0.0,
):
    """Run one two-body impact and return the final normal velocities."""
    closing_speed = velocity_a - velocity_b
    separation = IMPACT_FRACTION * closing_speed * DEFAULT_DT
    model, body_a, body_b = _build_two_sphere_model(
        device,
        separation=separation,
        restitution=restitution,
        mass_a=mass_a,
        mass_b=mass_b,
    )
    solver = _make_solver(model, restitution_velocity_threshold=restitution_velocity_threshold)
    state_in, state_out = model.state(), model.state()
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[2] = velocity_a
    joint_qd[8] = velocity_b
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)
    qd = state_out.body_qd.numpy()
    q = state_out.body_q.numpy()
    count = int(contacts.rigid_contact_count.numpy()[0])
    paths = {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0}
    return float(qd[body_a][2]), float(qd[body_b][2]), float(q[body_b][2] - q[body_a][2]), count, paths


def test_two_body_impact_conserves_momentum_and_reverses_relative_speed(test, device):
    """Match the exact two-body impact solution over mass ratio and frame shift."""
    restitution = 0.6
    for mass_a, mass_b, frame_speed in ((1.0, 1.0, 0.0), (1.0, 4.0, 0.0), (4.0, 1.0, 0.0), (1.0, 4.0, 3.0)):
        velocity_a = 2.0 + frame_speed
        velocity_b = -1.0 + frame_speed
        label = f"m=({mass_a},{mass_b}), frame={frame_speed}"
        with test.subTest(mass_a=mass_a, mass_b=mass_b, frame_speed=frame_speed):
            out_a, out_b, distance, count, paths = _step_two_spheres(
                device,
                mass_a=mass_a,
                mass_b=mass_b,
                velocity_a=velocity_a,
                velocity_b=velocity_b,
                restitution=restitution,
            )
            total_mass = mass_a + mass_b
            expected_a = (
                (mass_a - restitution * mass_b) * velocity_a + (1.0 + restitution) * mass_b * velocity_b
            ) / total_mass
            expected_b = (
                (1.0 + restitution) * mass_a * velocity_a + (mass_b - restitution * mass_a) * velocity_b
            ) / total_mass
            test.assertEqual(count, 1, f"{label}: expected exactly one sphere-sphere contact")
            test.assertEqual(paths, {PATH_MATRIX_FREE}, f"{label}: unexpected contact route {paths}")
            closing_speed = velocity_a - velocity_b
            separation = IMPACT_FRACTION * closing_speed * DEFAULT_DT
            expected_distance = 2.0 * RADIUS + separation + restitution * closing_speed * DEFAULT_DT
            test.assertAlmostEqual(distance, expected_distance, delta=2.0e-4, msg=f"{label}: wrong end separation")
            _assert_velocity(test, out_a, expected_a, velocity_a - velocity_b, f"{label}, body A")
            _assert_velocity(test, out_b, expected_b, velocity_a - velocity_b, f"{label}, body B")

            momentum_before = mass_a * velocity_a + mass_b * velocity_b
            momentum_after = mass_a * out_a + mass_b * out_b
            test.assertAlmostEqual(
                momentum_after,
                momentum_before,
                delta=2.0e-3 * max(1.0, abs(momentum_before)),
                msg=f"{label}: normal momentum changed",
            )
            test.assertAlmostEqual(
                out_b - out_a,
                -restitution * (velocity_b - velocity_a),
                delta=5.0e-3,
                msg=f"{label}: relative impact law failed",
            )
            reduced_mass = mass_a * mass_b / total_mass
            expected_energy_loss = 0.5 * reduced_mass * (1.0 - restitution**2) * (velocity_a - velocity_b) ** 2
            energy_before = 0.5 * mass_a * velocity_a**2 + 0.5 * mass_b * velocity_b**2
            energy_after = 0.5 * mass_a * out_a**2 + 0.5 * mass_b * out_b**2
            test.assertAlmostEqual(
                energy_before - energy_after,
                expected_energy_loss,
                delta=5.0e-3 * max(1.0, energy_before),
                msg=f"{label}: collision energy loss was inconsistent with e",
            )


def test_restitution_is_relative_to_a_moving_kinematic_surface(test, device):
    """Measure rebound relative to prescribed motion on the free-body rows."""
    restitution = 0.6
    closing_speed = 3.0
    separation = IMPACT_FRACTION * closing_speed * DEFAULT_DT
    for frame_speed in (0.0, 7.0):
        label = f"frame={frame_speed}"
        with test.subTest(frame_speed=frame_speed):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            cfg = _shape_cfg(restitution)
            dynamic_body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=_sphere_inertia(1.0),
                lock_inertia=True,
            )
            kinematic_body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * RADIUS + separation), wp.quat_identity()),
                mass=1.0,
                inertia=_sphere_inertia(1.0),
                lock_inertia=True,
                is_kinematic=True,
            )
            builder.add_shape_sphere(dynamic_body, radius=RADIUS, cfg=cfg)
            builder.add_shape_sphere(kinematic_body, radius=RADIUS, cfg=cfg)
            model = builder.finalize(device=device)
            solver = _make_solver(model)
            state_in, state_out = model.state(), model.state()
            velocity_dynamic = frame_speed + 2.0
            velocity_surface = frame_speed - 1.0
            joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
            joint_qd[2] = velocity_dynamic
            joint_qd[8] = velocity_surface
            state_in.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
            contacts = pipeline.contacts()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

            count = int(contacts.rigid_contact_count.numpy()[0])
            paths = {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0}
            velocity_after = float(state_out.body_qd.numpy()[dynamic_body][2])
            expected_after = velocity_surface - restitution * closing_speed
            test.assertEqual(count, 1, f"{label}: expected one sphere contact")
            test.assertEqual(paths, {PATH_MATRIX_FREE}, f"{label}: wrong contact route")
            _assert_velocity(test, velocity_after, expected_after, closing_speed, label, rtol=5.0e-3)
            test.assertAlmostEqual(
                velocity_surface - velocity_after,
                restitution * closing_speed,
                delta=1.5e-2,
                msg=f"{label}: restitution was not measured relative to the moving surface",
            )


def test_every_contact_response_route_enforces_restitution(test, device):
    """Enforce the same impact law on the free-body and the dense contact rows."""
    restitution = 0.65
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    for scene in ("free", "articulated"):
        with test.subTest(scene=scene):
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                scene=scene,
                velocity_iterations=0,
                iterations=16,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, scene)
            test.assertEqual(result["paths"], EXPECTED_PATHS[scene], f"{scene}: wrong route")
            _assert_velocity(test, result["after"], restitution * speed, speed, scene, rtol=5.0e-3)


def test_multiworld_restitution_keeps_case_data_isolated(test, device):
    """Apply each world's coefficient and crossing state without cross-world leakage."""
    # (e, speed, mass, toi/dt); the third case is deliberately non-crossing.
    cases = ((0.25, 2.0, 1.0, 0.4), (0.75, 4.0, 10.0, 0.7), (1.0, 3.0, 0.5, 1.25), (0.0, 5.0, 2.0, 0.2))
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for restitution, speed, mass, impact_fraction in cases:
        world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        separation = speed * DEFAULT_DT * impact_fraction
        body = world.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + separation), wp.quat_identity()),
            mass=mass,
            inertia=_sphere_inertia(mass),
            lock_inertia=True,
        )
        cfg = _shape_cfg(restitution)
        world.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        world.add_ground_plane(cfg=cfg)
        scene.add_world(world)
    model = scene.finalize(device=device)
    solver = _make_solver(model)
    state_in, state_out = model.state(), model.state()
    joint_qd = np.zeros((len(cases), 6), dtype=np.float32)
    joint_qd[:, 2] = [-case[1] for case in cases]
    state_in.joint_qd.assign(joint_qd.reshape(-1))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

    count = int(contacts.rigid_contact_count.numpy()[0])
    after = state_out.body_qd.numpy()[:, 2]
    end_gap = state_out.body_q.numpy()[:, 2] - RADIUS
    paths = solver._contact_path.numpy()[:count]
    test.assertEqual(count, len(cases), "expected exactly one contact in every world")
    test.assertTrue(np.all(paths == PATH_MATRIX_FREE), f"multiworld contacts used paths {paths.tolist()}")
    for world, (restitution, speed, _mass, impact_fraction) in enumerate(cases):
        with test.subTest(world=world):
            crossing = impact_fraction < 1.0
            if crossing and restitution > 0.0:
                expected = restitution * speed
            elif crossing:
                expected = -impact_fraction * speed
            else:
                expected = -speed
            _assert_velocity(test, float(after[world]), expected, speed, f"world {world}")
            if crossing and restitution > 0.0:
                expected_gap = impact_fraction * speed * DEFAULT_DT + restitution * speed * DEFAULT_DT
                test.assertAlmostEqual(
                    float(end_gap[world]), expected_gap, delta=2.0e-4, msg=f"world {world} wrong row trajectory"
                )
            elif crossing:
                test.assertLess(abs(float(end_gap[world])), 2.0e-4, f"world {world} missed the surface")
            else:
                test.assertGreater(float(end_gap[world]), 0.0, f"world {world} unexpectedly reached the surface")


def test_restitution_uses_ordinary_rows_and_validates_configuration(test, device):
    """Keep velocity iterations explicit while provisioning live restitution rows."""
    positive_model, _ = _build_plane_model(
        device,
        separation=1.0e-3,
        restitution=0.6,
    )
    solver = _make_solver(positive_model, velocity_iterations=0)
    test.assertEqual(solver.velocity_iterations, 0)
    test.assertFalse(hasattr(solver, "_velocity_post_iterations"), "zero iterations were silently overridden")
    test.assertEqual(solver._row_restitution.shape, (solver._world_count, solver.articulation_constraint_max))
    test.assertEqual(solver._mf_row_restitution.shape, (solver._world_count, solver.free_body_constraint_max))
    for invalid_threshold in (-1.0, float("nan"), float("inf")):
        with test.subTest(invalid_threshold=invalid_threshold):
            with test.assertRaisesRegex(ValueError, "restitution_velocity_threshold"):
                _make_solver(positive_model, restitution_velocity_threshold=invalid_threshold)

    zero_model, _ = _build_plane_model(
        device,
        separation=1.0e-3,
        restitution=0.0,
    )
    zero_solver = _make_solver(zero_model, velocity_iterations=0)
    test.assertEqual(zero_solver.velocity_iterations, 0)

    zero_model.shape_material_restitution.fill_(0.6)
    zero_solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    default_solver = _make_solver(positive_model, velocity_iterations=0, restitution_velocity_threshold=None)
    test.assertAlmostEqual(default_solver.restitution_velocity_threshold, 0.5)


def test_velocity_iteration_counts_reach_the_same_single_row_solution(test, device):
    """Keep restitution correct while optional velocity iterations refine the row."""
    restitution = 0.7
    speed = 2.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    for iterations in (0, 1, 4, 12):
        with test.subTest(velocity_iterations=iterations):
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                velocity_iterations=iterations,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, f"iterations={iterations}")
            _assert_velocity(test, result["after"], restitution * speed, speed, f"iterations={iterations}")


def test_warm_start_history_does_not_change_or_repeat_restitution(test, device):
    """Use the current incident velocity once under the contact warm start of both row families."""
    restitution = 0.6
    # Both incident speeds used below must cross from this same authored pose.
    separation = IMPACT_FRACTION * 2.0 * DEFAULT_DT
    cases = (
        ("free-body rows", "free", PATH_MATRIX_FREE),
        ("dense rows", "articulated", PATH_DENSE),
    )
    for label, scene, expected_path in cases:
        with test.subTest(warmstart=label):
            model, body = _build_plane_model(
                device,
                separation=separation,
                restitution=restitution,
                scene=scene,
            )
            solver = _make_solver(
                model,
                velocity_iterations=0,
                contact_warmstart=True,
            )
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
            contacts = pipeline.contacts()
            state_in, state_out = model.state(), model.state()

            for speed in (2.0, 5.0):
                _reset_state(model, state_in, -speed)
                contacts.clear()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)
                count = int(contacts.rigid_contact_count.numpy()[0])
                test.assertEqual(count, 1, f"{label}: reset impact generated {count} contacts")
                test.assertEqual(
                    {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0},
                    {expected_path},
                    f"{label}: reset impact used the wrong route",
                )
                after = float(state_out.body_qd.numpy()[body][2])
                _assert_velocity(test, after, restitution * speed, speed, f"{label}, speed={speed}", rtol=5.0e-3)

            # The second impact ends on the plane and separates. A persistent
            # matched contact on the next step must not apply restitution a
            # second time from either warm-start impulse history or stale u^-.
            contacts.clear()
            pipeline.collide(state_out, contacts)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0, f"{label}: contact did not persist")
            solver.step(state_out, state_in, model.control(), contacts, DEFAULT_DT)
            separating = float(state_in.body_qd.numpy()[body][2])
            _assert_velocity(test, separating, restitution * 5.0, 5.0, f"{label}, separating step", rtol=1.0e-2)


def test_redundant_box_contacts_do_not_multiply_restitution_energy(test, device):
    """Bounce a flat box without row-count-dependent speed, torque, or energy gain."""
    hx, hy, hz = 0.1, 0.08, 0.05
    mass = 2.0
    restitution = 0.6
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    ix = mass * (hy * hy + hz * hz) / 3.0
    iy = mass * (hx * hx + hz * hz) / 3.0
    iz = mass * (hx * hx + hy * hy) / 3.0
    inertia = wp.mat33(ix, 0.0, 0.0, 0.0, iy, 0.0, 0.0, 0.0, iz)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, hz + separation), wp.quat_identity()),
        mass=mass,
        inertia=inertia,
        lock_inertia=True,
    )
    cfg = _shape_cfg(restitution)
    builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device)
    solver = _make_solver(model, velocity_iterations=32, iterations=32)
    state_in, state_out = model.state(), model.state()
    _reset_state(model, state_in, -speed)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

    count = int(contacts.rigid_contact_count.numpy()[0])
    qd = state_out.body_qd.numpy()[body].astype(np.float64)
    end_gap = float(state_out.body_q.numpy()[body][2] - hz)
    test.assertGreaterEqual(count, 4, f"box test generated only {count} contacts")
    _assert_velocity(test, float(qd[2]), restitution * speed, speed, "flat box", rtol=1.0e-2)
    test.assertLess(np.linalg.norm(qd[:2]), 1.0e-2, f"flat box gained tangential speed {qd[:2]}")
    test.assertLess(np.linalg.norm(qd[3:]), 2.0e-2, f"flat box gained angular speed {qd[3:]}")
    expected_gap = separation + restitution * speed * DEFAULT_DT
    test.assertAlmostEqual(end_gap, expected_gap, delta=3.0e-4, msg=f"flat box ended at gap {end_gap:+.3e} m")
    energy_before = 0.5 * mass * speed * speed
    energy_after = 0.5 * mass * float(np.dot(qd[:3], qd[:3])) + 0.5 * (
        ix * qd[3] ** 2 + iy * qd[4] ** 2 + iz * qd[5] ** 2
    )
    test.assertAlmostEqual(
        energy_after / energy_before,
        restitution * restitution,
        delta=2.0e-2,
        msg=f"flat-box energy ratio was {energy_after / energy_before:.6f}",
    )


def test_articulated_redundant_manifold_preserves_symmetric_rebound(test, device):
    """Bounce an articulated four-point foot without spurious spin or energy gain."""
    mass = 2.0
    child_mass = 0.25
    half = 0.08
    sphere_radius = 0.02
    foot_offset = 0.03
    restitution = 0.6
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    root_height = foot_offset + sphere_radius + separation
    inertia = _diagonal_inertia(0.02)

    for joint_kind in ("fixed", "revolute"):
        with test.subTest(joint=joint_kind):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            xform = wp.transform(wp.vec3(0.0, 0.0, root_height), wp.quat_identity())
            root = builder.add_link(xform=xform, mass=mass, inertia=inertia, lock_inertia=True)
            child = builder.add_link(
                xform=xform,
                mass=child_mass,
                inertia=_diagonal_inertia(0.002),
                lock_inertia=True,
            )
            root_joint = builder.add_joint_free(child=root)
            if joint_kind == "fixed":
                child_joint = builder.add_joint_fixed(parent=root, child=child)
            else:
                child_joint = builder.add_joint_revolute(
                    parent=root,
                    child=child,
                    axis=wp.vec3(0.0, 0.0, 1.0),
                )
            builder.add_articulation([root_joint, child_joint])
            cfg = _shape_cfg(restitution)
            for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)):
                builder.add_shape_sphere(
                    root,
                    xform=wp.transform(wp.vec3(sx * half, sy * half, -foot_offset), wp.quat_identity()),
                    radius=sphere_radius,
                    cfg=cfg,
                )
            builder.add_ground_plane(cfg=cfg)
            model = builder.finalize(device=device)
            solver = _make_solver(model, velocity_iterations=0, iterations=16)
            state_in, state_out = model.state(), model.state()
            _reset_state(model, state_in, -speed)
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
            contacts = pipeline.contacts()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

            count = int(contacts.rigid_contact_count.numpy()[0])
            paths = {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0}
            qd = state_out.body_qd.numpy()[root].astype(np.float64)
            expected_path = PATH_DENSE
            label = joint_kind
            test.assertGreaterEqual(count, 4, f"{label}: manifold had only {count} contacts")
            test.assertEqual(paths, {expected_path}, f"{label}: wrong contact route")
            _assert_velocity(test, float(qd[2]), restitution * speed, speed, label, rtol=7.5e-3)
            test.assertLess(np.linalg.norm(qd[3:]), 3.0e-2, f"{label}: symmetric foot spun at {qd[3:]}")


def test_cuda_graph_replay_reads_current_restitution_and_matches_eager(test, device):
    """Replay collide plus restitution with stable pointers and device-read material data."""
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT

    def make_fixture():
        model, body = _build_plane_model(
            device,
            separation=separation,
            restitution=0.0,
        )
        solver = _make_solver(model, velocity_iterations=0)
        state_in, state_out = model.state(), model.state()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        control = model.control()

        def one_step():
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DEFAULT_DT)
            # Keep addresses stable for graph replay.
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
            wp.copy(state_in.joint_q, state_out.joint_q)
            wp.copy(state_in.joint_qd, state_out.joint_qd)

        return model, solver, body, state_in, contacts, one_step

    graph_model, graph_solver, graph_body, graph_state, graph_contacts, graph_step = make_fixture()
    eager_model, eager_solver, eager_body, eager_state, eager_contacts, eager_step = make_fixture()

    # Warm both fixtures through the same history, which also compiles the kernels.
    _reset_state(graph_model, graph_state, -speed)
    _reset_state(eager_model, eager_state, -speed)
    graph_step()
    eager_step()
    _reset_state(graph_model, graph_state, -speed)
    _reset_state(eager_model, eager_state, -speed)

    with wp.ScopedCapture(device) as capture:
        graph_step()

    # Change restitution from zero after capture. Row storage is independent
    # of construction-time values, and the coefficient must come from the
    # device array rather than a captured host scalar.
    for model, solver in ((graph_model, graph_solver), (eager_model, eager_solver)):
        model.shape_material_restitution.fill_(0.75)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    wp.capture_launch(capture.graph)
    eager_step()

    graph_velocity = float(graph_state.body_qd.numpy()[graph_body][2])
    eager_velocity = float(eager_state.body_qd.numpy()[eager_body][2])
    graph_count = int(graph_contacts.rigid_contact_count.numpy()[0])
    eager_count = int(eager_contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(graph_count, 1, "captured collision generated no unique contact")
    test.assertEqual(eager_count, 1, "eager collision generated no unique contact")
    _assert_velocity(test, graph_velocity, 0.75 * speed, speed, "captured step")
    test.assertAlmostEqual(graph_velocity, eager_velocity, delta=1.0e-6, msg="captured and eager steps diverged")


def test_notify_reads_replaced_and_in_place_restitution(test, device):
    """Match a fresh solver after a SHAPE_PROPERTIES notify, eagerly and in a captured graph."""
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    for scene, old, replace, capture in itertools.product(
        ("free", "articulated"), (0.0, 1.0), (False, True), (False, True)
    ):
        new = 1.0 - old
        label = f"{scene} e={old}->{new} {'replaced' if replace else 'in place'} {'captured' if capture else 'eager'}"
        with test.subTest(label):
            model, body = _build_plane_model(device, separation=separation, restitution=old, scene=scene)
            solver = _make_solver(model)
            state_in, state_out = model.state(), model.state()
            _reset_state(model, state_in, -speed)
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
            contacts = pipeline.contacts()
            control = model.control()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DEFAULT_DT)
            if capture:
                with wp.ScopedCapture(device) as graph:
                    pipeline.collide(state_in, contacts)
                    solver.step(state_in, state_out, control, contacts, DEFAULT_DT)
            if replace:
                model.shape_material_restitution = wp.full(model.shape_count, new, dtype=float, device=device)
            else:
                model.shape_material_restitution.fill_(new)
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, DEFAULT_DT)

            reference = model.state()
            _make_solver(model).step(state_in, reference, control, contacts, DEFAULT_DT)
            count = int(contacts.rigid_contact_count.numpy()[0])
            paths = {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0}
            test.assertEqual(count, 1, f"{label}: expected one contact")
            test.assertEqual(paths, EXPECTED_PATHS[scene], f"{label}: wrong route")
            expected = float(reference.body_qd.numpy()[body][2])
            _assert_velocity(test, expected, new * speed if new else -separation / DEFAULT_DT, speed, label)
            test.assertAlmostEqual(float(state_out.body_qd.numpy()[body][2]), expected, delta=2.0e-5, msg=label)


devices_restitution = get_selected_cuda_test_devices()


class TestSteelRestitution(unittest.TestCase):
    pass


for _fn in (
    test_one_step_restitution_law_over_coefficient_speed_and_mass,
    test_crossing_result_is_invariant_to_dt_and_time_of_impact,
    test_penetrating_impact_replaces_baumgarte_with_restitution,
    test_restitution_uses_the_post_force_incident_predictor,
    test_non_crossing_speculative_contact_does_not_bounce,
    test_shape_restitution_uses_symmetric_arithmetic_average,
    test_restitution_velocity_threshold_uses_incident_relative_speed,
    test_disabled_restitution_needs_no_material_edits,
    test_bouncing_ball_settles_to_rest_under_gravity,
    test_two_body_impact_conserves_momentum_and_reverses_relative_speed,
    test_restitution_is_relative_to_a_moving_kinematic_surface,
    test_every_contact_response_route_enforces_restitution,
    test_multiworld_restitution_keeps_case_data_isolated,
    test_restitution_uses_ordinary_rows_and_validates_configuration,
    test_velocity_iteration_counts_reach_the_same_single_row_solution,
    test_warm_start_history_does_not_change_or_repeat_restitution,
    test_redundant_box_contacts_do_not_multiply_restitution_energy,
    test_articulated_redundant_manifold_preserves_symmetric_rebound,
    test_cuda_graph_replay_reads_current_restitution_and_matches_eager,
    test_notify_reads_replaced_and_in_place_restitution,
):
    add_function_test(TestSteelRestitution, _fn.__name__, _fn, devices=devices_restitution)


SIM_DT = 1.0 / 2000.0


DROP = 0.12


# The route each scene takes, measured rather than assumed: a lone free body uses the
# free-body rows, and a contact on an articulated root uses the dense rows.
EXPECTED_PATHS_velocity_pass_speculative = {
    "free": {PATH_MATRIX_FREE},
    "articulated": {PATH_DENSE},
}


def _make_solver_velocity_pass_speculative(model, **kwargs):
    """Create the solver without angular damping, which would mask the contact response."""
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", **kwargs)
    solver._rigid_body_angular_damping.zero_()
    return solver


def _build(device, scene, mass=1.0):
    """Free sphere, or a two-link articulation whose root carries the contact."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    if scene == "free":
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + DROP), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    else:
        # add_link + explicit joints + add_articulation: add_body would create a
        # standalone free articulation per call, so appending a revolute to two
        # of them yields two articulations and an unowned loop joint rather than
        # the single two-link chain this fixture is meant to exercise.
        body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + DROP), wp.quat_identity()), mass=mass)
        builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        link = builder.add_link(xform=wp.transform(wp.vec3(0.12, 0.0, RADIUS + DROP), wp.quat_identity()), mass=mass)
        builder.add_shape_sphere(link, radius=RADIUS, cfg=cfg)
        root_joint = builder.add_joint_free(child=body)
        hinge = builder.add_joint_revolute(
            parent=body,
            child=link,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.12, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([root_joint, hinge])
    builder.add_ground_plane(cfg=cfg)
    return builder.finalize(device=device), body


def _drop(device, velocity_iterations, steps=620, scene="free", *, contact_warmstart=False):
    """Drop a body onto a plane; return heights, velocities, contact counts, routes."""
    model, body = _build(device, scene)
    solver = _make_solver_velocity_pass_speculative(
        model, velocity_iterations=velocity_iterations, contact_warmstart=contact_warmstart
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    heights, velocities, counts, routes = [], [], [], set()
    pipeline = (
        newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        if contact_warmstart
        else newton.CollisionPipeline(model)
    )
    contacts = pipeline.contacts()
    for _ in range(steps):
        contacts.clear()
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, SIM_DT)
        state_0, state_1 = state_1, state_0
        heights.append(float(state_0.body_q.numpy()[body][2]) - RADIUS)
        velocities.append(float(state_0.body_qd.numpy()[body][2]))
        n = int(contacts.rigid_contact_count.numpy()[0])
        counts.append(n)
        if n:
            routes.update(int(v) for v in solver._contact_path.numpy()[:n] if v >= 0)
    return np.asarray(heights), np.asarray(velocities), np.asarray(counts), routes


def _assert_landed(test, heights, velocities, counts, label):
    """Assert the body came to rest ON the plane, not through it and not above it."""
    final_h, final_v = float(heights[-1]), float(velocities[-1])
    # Bounded on BOTH sides: a tunnelling body has negative heights and would
    # satisfy any upper bound by itself.
    test.assertLess(abs(final_h), 0.005, f"{label}: rested at {final_h:+.4f} m instead of on the surface")
    test.assertLess(abs(final_v), 0.02, f"{label}: still moving at {final_v:+.4f} m/s")
    test.assertGreater(float(heights[:80].min()), 0.05, f"{label}: body was not still falling early in the run")
    test.assertGreater(int(counts.max()), 0, f"{label}: no contact was ever generated")
    settled = heights[-150:]
    test.assertLess(float(settled.max() - settled.min()), 0.002, f"{label}: resting height was not stable")


def _single_step(device, gap, approach_speed, velocity_iterations=4, mass=1.0):
    """One step of a body ``gap`` above the plane closing at ``approach_speed``.

    Returns the velocity before and after, the contact count, the position-pass
    impulse, the start gap, and the linearized end gap on positive-gap rows, so
    a test can confirm which side of the velocity-pass classifier it exercised.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))  # no gravity: isolate the constraint
    # density=0 so the body's mass is exactly the requested value: the point of
    # the light-body case is that the position impulse scales with it.
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, density=0.0)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + gap), wp.quat_identity()),
        mass=mass,
        inertia=wp.mat33(np.eye(3) * 0.4 * mass * RADIUS**2),
    )
    builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    # The light-body case sits below the inertia floor on purpose.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Inertia validation corrected", category=UserWarning)
        model = builder.finalize(device=device)
    solver = _make_solver_velocity_pass_speculative(model, velocity_iterations=velocity_iterations)
    # The position solve does not depend on the velocity-only iterations, so a twin
    # solver without them exposes the position-solve rows, impulses and velocity.
    position_solver = _make_solver_velocity_pass_speculative(model)
    state_0, state_1 = model.state(), model.state()
    # A free body is driven through joint_qd; writing body_qd alone is discarded
    # by the forward kinematics the solver runs from joint state.
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[2] = -abs(approach_speed)
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    before = float(state_0.body_qd.numpy()[body][2])
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    state_0.clear_forces()
    position_state = model.state()
    position_solver.step(state_0, position_state, model.control(), contacts, SIM_DT)
    solver.step(state_0, state_1, model.control(), contacts, SIM_DT)
    n = int(contacts.rigid_contact_count.numpy()[0])
    rows = int(position_solver._mf_constraint_count.numpy()[0])
    phi = position_solver._mf_phi.numpy()[0][:rows]
    lam = position_solver._mf_impulses.numpy()[0][:rows]
    row_type = position_solver._mf_row_type.numpy()[0][:rows]
    contact_rows = np.flatnonzero((row_type == 0) & (phi > 0.0))
    position_impulse = float(lam[contact_rows].max()) if contact_rows.size else 0.0
    max_phi = float(phi[contact_rows].max()) if contact_rows.size else 0.0
    v_position = position_solver._v_out.numpy()
    J_a = position_solver._mf_J_a.numpy()[0]
    J_b = position_solver._mf_J_b.numpy()[0]
    dof_a = position_solver._mf_dof_a.numpy()[0]
    dof_b = position_solver._mf_dof_b.numpy()[0]
    world_dofs = position_solver._world_dof_indices.numpy()[0]
    end_gaps = []
    for row in contact_rows:
        jv = 0.0
        if dof_a[row] >= 0:
            indices = world_dofs[dof_a[row] : dof_a[row] + 6]
            jv += float(np.dot(J_a[row][indices >= 0], v_position[indices[indices >= 0]]))
        if dof_b[row] >= 0:
            indices = world_dofs[dof_b[row] : dof_b[row] + 6]
            jv += float(np.dot(J_b[row][indices >= 0], v_position[indices[indices >= 0]]))
        end_gaps.append(float(phi[row] + SIM_DT * jv))
    max_end_gap = max(end_gaps, default=0.0)
    return before, float(state_1.body_qd.numpy()[body][2]), n, position_impulse, max_phi, max_end_gap


def test_rhs_classifies_exact_end_gap_without_impulse(test, device):
    """Treat exactly and numerically reached surfaces as active without consulting impulse."""
    count = wp.array([1], dtype=wp.int32, device=device)
    dof_a = wp.array([[0]], dtype=wp.int32, device=device)
    dof_b = wp.array([[-1]], dtype=wp.int32, device=device)
    J_a = wp.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=wp.float32, device=device)
    J_b = wp.zeros((1, 1, 6), dtype=wp.float32, device=device)
    world_dofs = wp.array([[0, 1, 2, 3, 4, 5]], dtype=wp.int32, device=device)
    phi = wp.array([[1.0]], dtype=wp.float32, device=device)
    row_type = wp.array([[0]], dtype=wp.int32, device=device)
    target_velocity = wp.zeros((1, 1), dtype=wp.float32, device=device)
    row_restitution = wp.zeros((1, 1), dtype=wp.float32, device=device)
    position_velocity = wp.zeros((6,), dtype=wp.float32, device=device)
    rhs = wp.zeros((1, 1), dtype=wp.float32, device=device)

    def classify(normal_velocity, target=0.0, has_target=0):
        position_velocity.assign(np.array([normal_velocity, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        target_velocity.assign(np.array([[target]], dtype=np.float32))
        wp.launch(
            compute_mf_velocity_rhs,
            dim=1,
            inputs=[
                count,
                dof_a,
                dof_b,
                J_a,
                J_b,
                world_dofs,
                phi,
                row_type,
                target_velocity,
                row_restitution,
                has_target,
                0.5,
                position_velocity,
                position_velocity,
                0.5,
                1,
            ],
            outputs=[rhs],
            device=device,
        )
        return float(rhs.numpy()[0, 0])

    outside_velocity = np.float32(-1.999997)
    outside_end_gap = np.float32(1.0) + np.float32(0.5) * outside_velocity
    near_velocity = np.float32(-1.999999)
    near_end_gap = np.float32(1.0) + np.float32(0.5) * near_velocity

    test.assertGreater(float(outside_end_gap), 1.0e-6, "outside test case did not clear the activation slop")
    test.assertGreater(float(near_end_gap), 0.0, "near-boundary test case did not retain a positive residual")
    test.assertLessEqual(float(near_end_gap), 1.0e-6, "near-boundary test case exceeded the activation slop")

    test.assertEqual(classify(-1.0), 2.0, "a row ending 0.5 m away lost its speculative allowance")
    test.assertEqual(
        classify(float(outside_velocity)),
        2.0,
        "a row ending beyond the numerical boundary lost its speculative allowance",
    )
    test.assertEqual(
        classify(float(near_velocity)),
        0.0,
        "a binding row with a small positive residual retained its speculative allowance",
    )
    test.assertEqual(classify(-2.0), 0.0, "a row reaching the surface retained its speculative allowance")
    test.assertEqual(
        classify(-1.0, target=1.0, has_target=1),
        -1.0,
        "prescribed target velocity was omitted or used with the wrong sign",
    )


def test_velocity_iterations_do_not_halt_a_falling_body(test, device):
    """Land a falling body on the plane with the velocity pass enabled."""
    for iterations in (2, 8):
        h, v, n, _r = _drop(device, iterations)
        _assert_landed(test, h, v, n, f"velocity_iterations={iterations}")


def test_velocity_iterations_match_the_position_only_landing(test, device):
    """Land at the same height with and without velocity iterations."""
    baseline, base_v, base_n, _r0 = _drop(device, 0)
    with_velocity, vel_v, vel_n, _r4 = _drop(device, 4)
    _assert_landed(test, baseline, base_v, base_n, "velocity_iterations=0")
    _assert_landed(test, with_velocity, vel_v, vel_n, "velocity_iterations=4")
    test.assertAlmostEqual(
        float(with_velocity[-1]),
        float(baseline[-1]),
        delta=0.002,
        msg=(
            f"resting height moved from {float(baseline[-1]):.4f} m to "
            f"{float(with_velocity[-1]):.4f} m when velocity iterations were enabled"
        ),
    )


def test_every_contact_response_route_lands(test, device):
    """Land correctly on the free-body and the dense contact rows.

    Asserts the exact ``contact_path`` each scene took, so the suite cannot claim
    coverage of a route the scene never reaches.
    """
    for scene in ("free", "articulated"):
        h, v, n, routes = _drop(device, 4, scene=scene)
        _assert_landed(test, h, v, n, scene)
        test.assertEqual(
            routes,
            EXPECTED_PATHS_velocity_pass_speculative[scene],
            f"{scene}: exercised contact_path {sorted(routes)}, expected {sorted(EXPECTED_PATHS_velocity_pass_speculative[scene])}",
        )


def test_non_crossing_row_keeps_its_speculative_allowance(test, device):
    """Leave the velocity untouched when the body cannot reach the surface.

    With ``phi + h*u > 0`` the position trajectory remains separated. If
    the velocity pass dropped its ``phi/h`` allowance it would forbid any
    approach and brake a body still far from contact.
    """
    gap = 0.02
    before, after, n, impulse, phi, end_gap = _single_step(device, gap, 0.5 * gap / SIM_DT)
    test.assertLess(before, -1.0, "setup failed to give the body an approach velocity")
    test.assertGreater(n, 0, "no speculative contact was generated, so the branch was never reached")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    test.assertEqual(impulse, 0.0, f"a non-crossing row took a position impulse of {impulse:.3e}")
    test.assertGreater(end_gap, 0.0, f"non-crossing row ended at gap {end_gap:+.3e} m")
    test.assertAlmostEqual(
        after, before, delta=0.02 * abs(before), msg=f"free approach was braked: {before:.3f} -> {after:.3f} m/s"
    )


def test_crossing_row_loses_its_speculative_allowance(test, device):
    """Arrest a body that would cross the surface within the step.

    A blanket "always retain phi/h" implementation leaves it closing at the gap
    rate instead, which this bound catches.
    """
    gap = 0.002
    before, after, n, impulse, phi, end_gap = _single_step(device, gap, 5.0 * gap / SIM_DT)
    test.assertLess(before, -1.0, "setup failed to give the body an approach velocity")
    test.assertGreater(n, 0, "no contact was generated")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    test.assertGreater(impulse, 0.0, "a crossing row took no position impulse")
    test.assertLessEqual(end_gap, 1.0e-6, f"crossing row still ended at gap {end_gap:+.3e} m")
    # Magnitude, not a signed bound: "after <= 0.05*|before|" is satisfied by any
    # negative value, so a row that merely slowed from -20 to -4 m/s would pass.
    test.assertLess(
        abs(after),
        0.05 * abs(before),
        f"crossing contact kept approaching: {before:.3f} -> {after:.3f} m/s",
    )


def test_light_body_crossing_uses_end_gap_not_impulse_scale(test, device):
    """Arrest a very light crossing body independently of impulse scale.

    The impulse is effective mass times the velocity change, so it can be made
    arbitrarily small without changing the physics of the impact. At this mass it
    is ~2e-11, below the 1e-9 absolute threshold an impulse classifier once
    used. End-gap classification is invariant to that mass scaling. The mass is
    deliberately unphysical: the point is the classifier, not the scenario.
    """
    gap = 0.002
    before, after, n, impulse, phi, end_gap = _single_step(device, gap, 5.0 * gap / SIM_DT, mass=1.0e-12)
    test.assertGreater(n, 0, "no contact was generated")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    # Straddles the 1e-9 cutoff this implementation once used: positive, so the
    # row reaches the surface, yet small enough that an absolute threshold misses it.
    test.assertGreater(impulse, 0.0, "light crossing row took no position impulse")
    test.assertLess(impulse, 1.0e-9, f"position impulse {impulse:.3e} does not straddle the old 1e-9 cutoff")
    test.assertLessEqual(end_gap, 1.0e-6, f"light crossing row still ended at gap {end_gap:+.3e} m")
    test.assertLess(
        abs(after),
        0.05 * abs(before),
        f"light crossing body kept approaching: {before:.3f} -> {after:.3f} m/s (position impulse {impulse:.3e})",
    )


def test_multiworld_end_gap_uses_each_world_position_velocity(test, device):
    """Classify speculative rows from the corresponding world's velocity."""
    gap = 0.002
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0)
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + gap), wp.quat_identity()))
    template.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    template.add_ground_plane(cfg=cfg)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2)
    model = builder.finalize(device=device)
    solver = _make_solver_velocity_pass_speculative(model, velocity_iterations=4)
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy().reshape(2, 6)
    joint_qd[0, 2] = -0.5 * gap / SIM_DT
    joint_qd[1, 2] = -5.0 * gap / SIM_DT
    state_0.joint_qd.assign(joint_qd.reshape(-1))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    state_0.clear_forces()
    solver.step(state_0, state_1, model.control(), contacts, SIM_DT)
    after = state_1.body_qd.numpy()[:, 2]
    test.assertAlmostEqual(
        float(after[0]),
        float(joint_qd[0, 2]),
        delta=0.02 * abs(float(joint_qd[0, 2])),
        msg=f"non-crossing world was braked to {after[0]:.3f} m/s",
    )
    test.assertLess(
        abs(float(after[1])),
        0.05 * abs(float(joint_qd[1, 2])),
        f"crossing world kept approaching at {after[1]:.3f} m/s",
    )


def test_warm_start_with_velocity_iterations_is_supported(test, device):
    """Combine the contact warm start with end-gap classification on both row families."""
    # Exercise the dense and free-body identity warm starts through the velocity
    # iterations, not just their construction.
    h, v, n, routes = _drop(device, 4, scene="articulated", contact_warmstart=True)
    _assert_landed(test, h, v, n, "dense warm start")
    test.assertEqual(routes, {PATH_DENSE})

    h, v, n, routes = _drop(device, 4, contact_warmstart=True)
    _assert_landed(test, h, v, n, "free-body warm start")
    test.assertEqual(routes, {PATH_MATRIX_FREE})


def test_velocity_iterations_keep_angular_damping(test, device):
    """Apply the same root angular damping to the published velocity for every iteration count."""
    dt = 0.01
    spin = 10.0
    for contact, damping, iterations in itertools.product((False, True), (0.0, 0.05, 2.0), (0, 1, 4)):
        label = f"contact={contact} damping={damping} iterations={iterations}"
        with test.subTest(label):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if contact else 0.0))
            inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            xform = wp.transform(wp.vec3(0.0, 0.0, RADIUS), wp.quat_identity())
            body = builder.add_body(xform=xform, mass=1.0, inertia=inertia, lock_inertia=True)
            if contact:
                # Frictionless contact through the centre leaves the spin to damping alone.
                cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
                builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
                builder.add_ground_plane(cfg=cfg)
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", velocity_iterations=iterations)
            solver._rigid_body_angular_damping.fill_(damping)
            state_in, state_out = model.state(), model.state()
            state_in.joint_qd.assign(np.array([0.0, 0.0, 0.0, 0.0, 0.0, spin], dtype=np.float32))
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            contacts = None
            if contact:
                pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
                contacts = pipeline.contacts()
                pipeline.collide(state_in, contacts)
                test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0, f"{label}: no contact")
            solver.step(state_in, state_out, model.control(), contacts, dt)

            expected = spin * (1.0 - damping * dt)
            test.assertAlmostEqual(float(state_out.joint_qd.numpy()[5]), expected, delta=1.0e-5, msg=label)
            test.assertAlmostEqual(float(state_out.body_qd.numpy()[body][5]), expected, delta=1.0e-5, msg=label)


devices_velocity_pass_speculative = get_selected_cuda_test_devices()


class TestSteelVelocityPassSpeculative(unittest.TestCase):
    pass


for _fn in (
    test_rhs_classifies_exact_end_gap_without_impulse,
    test_velocity_iterations_do_not_halt_a_falling_body,
    test_velocity_iterations_match_the_position_only_landing,
    test_every_contact_response_route_lands,
    test_non_crossing_row_keeps_its_speculative_allowance,
    test_crossing_row_loses_its_speculative_allowance,
    test_light_body_crossing_uses_end_gap_not_impulse_scale,
    test_multiworld_end_gap_uses_each_world_position_velocity,
    test_warm_start_with_velocity_iterations_is_supported,
    test_velocity_iterations_keep_angular_damping,
):
    add_function_test(
        TestSteelVelocityPassSpeculative,
        _fn.__name__,
        _fn,
        devices=devices_velocity_pass_speculative,
    )


def _build_mixed_response_model(
    device,
    world_count=1,
    *,
    dof_count=13,
    friction=0.0,
    restitution=0.0,
    static_plane=False,
    static_support=False,
    free_body_first=False,
    free_body_velocity_limit=None,
):
    """Build one serial articulation contacting one free rigid body.

    ``static_support`` rests the free body on a static ledge that clears the arm, so the free body also
    produces matrix-free contact rows. ``free_body_first`` builds the free body before the articulation,
    which packs the world DOFs free-body first. ``free_body_velocity_limit`` caps the free body's linear
    and angular velocity [m/s, rad/s], producing matrix-free velocity-limit rows once exceeded.
    ``static_plane`` adds a ground plane under the free body.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if free_body_velocity_limit is not None:
        SolverSteel.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.restitution = restitution
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    def add_free_body():
        custom_attributes = None
        if free_body_velocity_limit is not None:
            custom_attributes = {
                "steel:rigid_body_max_linear_velocity": free_body_velocity_limit,
                "steel:rigid_body_max_angular_velocity": free_body_velocity_limit,
            }
        box = builder.add_link(
            xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()), custom_attributes=custom_attributes
        )
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
        if static_support:
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.7, 0.11, 0.47), wp.quat_identity()),
                hx=0.05,
                hy=0.05,
                hz=0.05,
            )

    if free_body_first:
        add_free_body()
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=arm,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
        )
    ]
    parent = arm
    for index in range(dof_count - 1):
        child = builder.add_link(
            mass=0.05,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-3),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[index % 3],
            )
        )
        parent = child
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0])

    if not free_body_first:
        add_free_body()
    if static_plane:
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, -0.53))
    if world_count == 1:
        return builder.finalize(device=device)
    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(builder, world_count, spacing=(3.0, 0.0, 0.0))
    return replicated.finalize(device=device)


def _run_mixed_response(
    kernel,
    *,
    warmstart,
    preelimination,
    dof_count=13,
    articulation_constraint_max=32,
    inactive_joint_limit_capacity=False,
    friction=0.0,
    restitution=0.0,
    tangential_velocity=0.0,
    static_plane=False,
    contact_regularization=0.0,
    friction_anchor_beta=None,
    model_kwargs=None,
):
    """Run a short mixed-contact trajectory with one H-inverse implementation."""
    model = _build_mixed_response_model(
        "cuda:0",
        dof_count=dof_count,
        friction=friction,
        restitution=restitution,
        static_plane=static_plane,
        **(model_kwargs or {}),
    )
    with kernel_overrides({"hinv_jt_kernel": kernel}):
        solver = SolverSteel(
            model,
            solve_mode="matrix_free",
            contact_warmstart=warmstart,
            enable_bilateral_preelimination=preelimination,
            enable_joint_limits=inactive_joint_limit_capacity,
            joint_limit_activation_gap=0.0,
            iterations=8,
            contact_regularization=contact_regularization,
            friction_anchor_beta=friction_anchor_beta,
            articulation_constraint_max=articulation_constraint_max,
            free_body_constraint_max=32,
        )
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    free_articulation = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
    free_dof_start = int(solver._model_plan.articulation_dof_start[free_articulation])
    joint_qd[free_dof_start] = tangential_velocity
    joint_qd[free_dof_start + 2] = -3.0
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        reduce_contacts=False,
        contact_matching="latest",
    )
    contacts = pipeline.contacts()
    control = model.control()
    samples = []
    for _ in range(4):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        constraint_count = int(solver._constraint_count.numpy()[0])
        mf_count = int(solver._mf_constraint_count.numpy()[0])
        samples.append(
            (
                constraint_count,
                solver._diag.numpy()[0, :constraint_count].copy(),
                solver._impulses.numpy()[0, :constraint_count].copy(),
                state_out.joint_q.numpy().copy(),
                state_out.joint_qd.numpy().copy(),
                0,
                solver._row_type.numpy()[0, :constraint_count].copy(),
                mf_count,
                solver._mf_impulses.numpy()[0, :mf_count].copy(),
                solver._mf_row_type.numpy()[0, :mf_count].copy(),
                solver._mf_row_parent.numpy()[0, :mf_count].copy(),
                solver._mf_J_a.numpy()[0, :mf_count].copy(),
                solver._mf_J_b.numpy()[0, :mf_count].copy(),
                solver._mf_MiJt_a.numpy()[0, :mf_count].copy(),
                solver._mf_MiJt_b.numpy()[0, :mf_count].copy(),
            )
        )
        state_in, state_out = state_out, state_in
    return solver, samples


@unittest.skipUnless(wp.is_cuda_available(), "graph capture requires CUDA")
class TestSteelJointlessKinematicBody(unittest.TestCase):
    PRESCRIBED_QD = (0.3, -0.2, 0.0, 0.0, 0.0, 0.5)

    def _published(self, response, capture):
        """Step a free box on a jointless kinematic floor and return the floor's state in both buffers."""
        builder = newton.ModelBuilder()
        floor = builder.add_link(is_kinematic=True)
        builder.add_shape_box(floor, hx=0.5, hy=0.5, hz=0.02, xform=wp.transform(wp.vec3(0.0, 0.0, -0.02)))
        box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.021)))
        builder.add_shape_box(box, hx=0.02, hy=0.02, hz=0.02)
        model = builder.finalize(device=wp.get_cuda_device())
        # Fill freed device memory with NaN so unwritten scratch the solver publishes is likely to read as NaN.
        nan_fill = wp.full(1 << 22, float("nan"), dtype=wp.float32, device=model.device)
        del nan_fill
        solver = SolverSteel(model, articulated_contact_response=response)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        states = (model.state(), model.state())
        body_q, body_qd = states[0].body_q.numpy(), states[0].body_qd.numpy()
        body_q[floor, 0] += 0.1
        body_qd[floor] = self.PRESCRIBED_QD
        states[0].body_q.assign(body_q)
        states[0].body_qd.assign(body_qd)
        stale_q, stale_qd = states[1].body_q.numpy(), states[1].body_qd.numpy()
        stale_q[floor] = np.nan
        stale_qd[floor] = np.nan
        states[1].body_q.assign(stale_q)
        states[1].body_qd.assign(stale_qd)
        control = model.control()

        def two_steps():
            for state_in, state_out in (states, states[::-1]):
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, DEFAULT_DT)

        if capture:
            with wp.ScopedCapture(model.device) as scope:
                two_steps()
            for _ in range(2):
                wp.capture_launch(scope.graph)
        else:
            for _ in range(2):
                two_steps()
        self.assertTrue(np.isfinite(states[0].body_q.numpy()[box]).all())
        return body_q[floor], [(s.body_q.numpy()[floor], s.body_qd.numpy()[floor]) for s in states]

    def test_jointless_kinematic_body_publishes_its_prescribed_state(self):
        """Pass a jointless kinematic body's input pose and velocity through every step, eager and captured."""
        for response in ("immediate", "propagation"):
            for capture in (False, True):
                with self.subTest(response=response, capture=capture):
                    prescribed_q, published = self._published(response, capture)
                    for body_q, body_qd in published:
                        np.testing.assert_array_equal(body_q, prescribed_q)
                        np.testing.assert_array_equal(body_qd, np.array(self.PRESCRIBED_QD, dtype=np.float32))


class TestSteelJointlessOnlyModel(unittest.TestCase):
    def test_model_without_joints_publishes_its_prescribed_state(self):
        """Pass a jointless kinematic body's state through a step of a model that has no joints."""
        devices = [("cpu", "split")]
        if wp.is_cuda_available():
            devices.append((wp.get_cuda_device(), "matrix_free"))
        for device, solve_mode in devices:
            with self.subTest(device=str(device)):
                builder = newton.ModelBuilder()
                body = builder.add_link(is_kinematic=True)
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                model = builder.finalize(device=device)
                self.assertEqual(model.joint_count, 0)
                solver = SolverSteel(model, solve_mode=solve_mode)
                state_in, state_out = model.state(), model.state()
                body_q, body_qd = state_in.body_q.numpy(), state_in.body_qd.numpy()
                body_q[body, 0] = 0.25
                body_qd[body] = (0.3, -0.2, 0.0, 0.0, 0.0, 0.5)
                state_in.body_q.assign(body_q)
                state_in.body_qd.assign(body_qd)
                solver.step(state_in, state_out, None, None, DEFAULT_DT)
                np.testing.assert_array_equal(state_out.body_q.numpy()[body], body_q[body])
                np.testing.assert_array_equal(state_out.body_qd.numpy()[body], body_qd[body])


class TestSteelPairedResponse(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_response_matches_general_23_dof_trajectory(self):
        """Match the general response when one warp owns a robot/free-body pair, with patch and point friction."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "articulation_constraint_max": 96,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
        }
        # Patch friction rows are not uniform triples, so the generic factor path pools the patch load; point
        # friction keeps the contact-triple layout.
        for label, friction_anchor_beta, triples in (("patch", None, False), ("point", 0.0, True)):
            with self.subTest(friction=label):
                reference_solver, reference = _run_mixed_response(
                    "par_row", friction_anchor_beta=friction_anchor_beta, **run_kwargs
                )
                paired_solver, paired = _run_mixed_response(
                    "auto", friction_anchor_beta=friction_anchor_beta, **run_kwargs
                )

                self.assertIsNone(reference_solver._paired_response_primary_size)
                self.assertEqual(paired_solver._friction_anchors_enabled, label == "patch")
                self.assertEqual(paired_solver._paired_response_primary_size, 23)
                self.assertEqual(paired_solver._paired_response_secondary_size, 6)
                self.assertIsNotNone(paired_solver._paired_response_kernel)
                self.assertIsNotNone(paired_solver._paired_factor_solve_kernel)
                self.assertTrue(paired_solver._paired_factor_coordinates)
                self.assertEqual(paired_solver._factor_coordinate_contact_triples, triples)
                self.assertGreater(reference[0][0], 0, "mixed scene generated no dense constraint rows")
                self.assertTrue(
                    any(np.any(sample[6] == PGS_CONSTRAINT_TYPE_FRICTION) for sample in paired),
                    "mixed scene generated no friction rows",
                )
                for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
                    self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
                    np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
                    for label_value, expected_value, actual_value in zip(
                        ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
                    ):
                        np.testing.assert_allclose(
                            actual_value,
                            expected_value,
                            rtol=5.0e-4,
                            atol=1.0e-5,
                            err_msg=f"{label_value} differed at step {step}",
                        )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response fallback requires CUDA")
    def test_paired_response_falls_back_for_matrix_free_rows(self):
        """Keep mixed dense/matrix-free worlds in physical coordinates."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "articulation_constraint_max": 96,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "static_plane": True,
        }
        reference_solver, reference = _run_mixed_response("par_row", **run_kwargs)
        paired_solver, paired = _run_mixed_response("auto", **run_kwargs)

        self.assertIsNone(reference_solver._paired_response_primary_size)
        self.assertTrue(paired_solver._paired_factor_coordinates)
        self.assertTrue(any(sample[7] > 0 for sample in paired), "static contact generated no matrix-free rows")
        for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=1.0e-5,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_factor_coordinates_follow_patch_friction_and_torsion(self):
        """Keep factor coordinates with default patch friction and yield them to the general owner for torsion."""
        model = _build_mixed_response_model("cuda:0", dof_count=23, friction=0.7)
        solvers = {}
        with kernel_overrides({"hinv_jt_kernel": "auto"}):
            for radius in (0.0, 0.01):
                set_torsion_radius(model, radius)
                solvers[radius] = SolverSteel(
                    model,
                    solve_mode="matrix_free",
                    enable_joint_limits=True,
                    joint_limit_activation_gap=0.0,
                    iterations=8,
                    articulation_constraint_max=96,
                    free_body_constraint_max=32,
                )
            set_torsion_radius(model, 0.0)
            point = SolverSteel(
                model,
                solve_mode="matrix_free",
                enable_joint_limits=True,
                joint_limit_activation_gap=0.0,
                iterations=8,
                articulation_constraint_max=96,
                free_body_constraint_max=32,
                friction_anchor_beta=0.0,
            )
        self.assertTrue(solvers[0.0]._friction_anchors_enabled)
        self.assertTrue(solvers[0.0]._paired_factor_coordinates)
        # Patch rows are not uniform triples; the generic factor path pools the patch load instead.
        self.assertFalse(solvers[0.0]._factor_coordinate_contact_triples)
        self.assertFalse(point._friction_anchors_enabled)
        self.assertTrue(point._paired_factor_coordinates)
        self.assertTrue(point._factor_coordinate_contact_triples)
        self.assertTrue(solvers[0.01]._contact_torsion_enabled)
        self.assertIsNotNone(solvers[0.01]._paired_response_primary_size)
        self.assertFalse(solvers[0.01]._paired_factor_coordinates)
        self.assertFalse(solvers[0.01]._factor_coordinate_contact_triples)

    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_response_without_factor_coordinates_matches_general(self):
        """Match the general response with the explicit inverses that velocity limits keep in physical coordinates."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "articulation_constraint_max": 96,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "friction_anchor_beta": 0.0,
            "contact_regularization": 0.01,
        }
        reference_solver, reference = _run_mixed_response("par_row", **run_kwargs)
        paired_solver, paired = _run_mixed_response("auto", **run_kwargs)
        self.assertIsNone(reference_solver._paired_response_primary_size)
        self.assertEqual(paired_solver._paired_response_primary_size, 23)
        self.assertFalse(paired_solver._paired_factor_coordinates)
        self.assertIsNone(paired_solver._paired_factor_solve_kernel)
        self.assertIsNotNone(paired_solver._paired_cholesky_inverse_kernel)
        for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value, expected_value, rtol=5.0e-4, atol=1.0e-5, err_msg=f"{label} differed at step {step}"
                )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response selection requires CUDA")
    def test_paired_response_selection(self):
        """Select the pair only for one 17-26 DOF articulation and one free body per world, in the matrix-free solve."""
        for dof_count, expected in ((23, 23), (12, None), (27, None)):
            with self.subTest(dof_count=dof_count):
                model = _build_mixed_response_model("cuda:0", dof_count=dof_count)
                solver = SolverSteel(model, articulation_constraint_max=96, free_body_constraint_max=32)
                self.assertEqual(solver._paired_response_primary_size, expected)
        model = _build_mixed_response_model("cuda:0", dof_count=23)
        self.assertTrue(
            SolverSteel(model, articulation_constraint_max=96, free_body_constraint_max=32)._paired_factor_coordinates
        )
        # The tiled response is chunked only when the row capacity exceeds one chunk.
        self.assertIsNone(SolverSteel(model, articulation_constraint_max=16)._paired_response_primary_size)
        for options in ({"enable_bilateral_preelimination": True},):
            with self.subTest(**options):
                solver = SolverSteel(model, articulation_constraint_max=96, free_body_constraint_max=32, **options)
                self.assertIsNone(solver._paired_response_primary_size)
        for options in (
            {"enable_joint_velocity_limits": True},
            {"contact_regularization": 0.01},
            {"row_schedule": "phased"},
            {"drive_mode": "rows"},
            {"enable_joint_friction": True},
        ):
            with self.subTest(**options):
                solver = SolverSteel(model, articulation_constraint_max=96, free_body_constraint_max=32, **options)
                self.assertEqual(solver._paired_response_primary_size, 23)
                self.assertFalse(solver._paired_factor_coordinates)

    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_factor_coordinates_with_contact_compliance_match_general(self):
        """Solve compliant contacts in factor coordinates like the general response."""
        results = {}
        for kernel in ("par_row", "auto"):
            model = _build_mixed_response_model("cuda:0", dof_count=23, friction=0.5)
            with kernel_overrides({"hinv_jt_kernel": kernel}):
                solver = SolverSteel(
                    model,
                    friction_anchor_beta=0.0,
                    contact_compliance=True,
                    iterations=16,
                    articulation_constraint_max=96,
                    free_body_constraint_max=32,
                )
            state_0, state_1 = model.state(), model.state()
            joint_qd = state_0.joint_qd.numpy()
            free = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
            joint_qd[int(solver._model_plan.articulation_dof_start[free]) + 2] = -3.0
            state_0.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=64)
            contacts = pipeline.contacts()
            for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
                setattr(contacts, name, wp.zeros(64, dtype=float, device="cuda:0"))
            compliant = 0
            for _ in range(20):
                pipeline.collide(state_0, contacts)
                contacts.rigid_contact_stiffness.fill_(3000.0)
                contacts.rigid_contact_damping.fill_(20.0)
                contacts.rigid_contact_friction.fill_(1.0)
                solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
                state_0, state_1 = state_1, state_0
                compliant = max(compliant, solver.compliance_contact_count)
            results[kernel] = (solver, compliant, state_0.joint_qd.numpy())
        paired, compliant, joint_qd = results["auto"]
        self.assertTrue(paired._paired_factor_coordinates)
        self.assertGreater(compliant, 0)
        np.testing.assert_allclose(joint_qd, results["par_row"][2], rtol=0.0, atol=1.0e-4)


if __name__ == "__main__":
    unittest.main()
