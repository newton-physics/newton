# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for FeatherPGS speculative-contact controls."""

import inspect
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton._src.solvers.feather_pgs.kernels import (
    allocate_world_contact_slots,
    apply_world_contact_restitution_accumulated,
    compute_mf_effective_mass_and_rhs,
    compute_propagation_effective_mass_and_rhs,
    compute_world_contact_bias,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    prepare_world_contact_rows,
)
from newton.solvers import SolverFeatherPGS

PATH_DENSE = 0
PATH_MATRIX_FREE = 1
PATH_PROPAGATION = 2


def _disabled_patches(device):
    patches = FrictionPatches()
    patches.phi = wp.zeros(1, dtype=wp.vec2, device=device)
    return patches


def _launch_contact_allocator(
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
    """Allocate one contact and return its route metadata and counters."""
    device = "cpu"
    outputs = {
        "world": wp.full((1,), -9, dtype=wp.int32, device=device),
        "slot": wp.full((1,), -9, dtype=wp.int32, device=device),
        "art_a": wp.full((1,), -9, dtype=wp.int32, device=device),
        "art_b": wp.full((1,), -9, dtype=wp.int32, device=device),
        "path": wp.full((1,), -9, dtype=wp.int32, device=device),
        "slots_needed": wp.full((1,), -9, dtype=wp.int32, device=device),
    }
    counters = {
        "dense_count": wp.zeros((1,), dtype=wp.int32, device=device),
        "mf_count": wp.zeros((1,), dtype=wp.int32, device=device),
        "propagation_count": wp.zeros((1,), dtype=wp.int32, device=device),
        "dense_world_flag": wp.zeros((1,), dtype=wp.int32, device=device),
        "dense_dropped": wp.zeros((1,), dtype=wp.int32, device=device),
        "mf_dropped": wp.zeros((1,), dtype=wp.int32, device=device),
        "propagation_dropped": wp.zeros((1,), dtype=wp.int32, device=device),
        "dense_first_rejected": wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
        "mf_first_rejected": wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
        "propagation_first_rejected": wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
    }
    is_free = route == PATH_MATRIX_FREE
    propagation_enabled = route == PATH_PROPAGATION
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
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.array([int(responsive)], dtype=wp.int32, device=device),
            wp.array([int(is_free)], dtype=wp.int32, device=device),
            1,
            int(propagation_enabled),
            0,
            0,
            gate,
            scoped_gate,
            pair_gate,
            8,
            8,
            8,
            int(enable_friction),
            friction_gap,
            int(friction_pairs_only),
            0,
            FrictionPatches(),
        ],
        outputs=[
            outputs["world"],
            outputs["slot"],
            outputs["art_a"],
            outputs["art_b"],
            counters["dense_count"],
            outputs["path"],
            counters["mf_count"],
            counters["propagation_count"],
            counters["dense_world_flag"],
            outputs["slots_needed"],
            counters["dense_dropped"],
            counters["mf_dropped"],
            counters["propagation_dropped"],
            counters["dense_first_rejected"],
            counters["mf_first_rejected"],
            counters["propagation_first_rejected"],
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return {name: int(array.numpy()[0]) for name, array in (outputs | counters).items()}


def _launch_articulation_pair_contact_allocator(
    *,
    gap: float,
    scoped_gate: float = 0.0,
    pair_gate: float = 0.0,
    cross_articulation: bool = False,
    enable_friction: bool = False,
    friction_gap: float = float("inf"),
    friction_pairs_only: bool = False,
):
    """Allocate one contact between two non-free articulated links."""
    device = "cpu"
    body_to_articulation = [0, 1] if cross_articulation else [0, 0]
    articulation_count = 2 if cross_articulation else 1
    contact_slot = wp.full((1,), -9, dtype=wp.int32, device=device)
    contact_path = wp.full((1,), -9, dtype=wp.int32, device=device)
    contact_slots_needed = wp.full((1,), -9, dtype=wp.int32, device=device)
    dense_count = wp.zeros((1,), dtype=wp.int32, device=device)
    dense_dropped = wp.zeros((1,), dtype=wp.int32, device=device)
    mf_dropped = wp.zeros((1,), dtype=wp.int32, device=device)
    propagation_dropped = wp.zeros((1,), dtype=wp.int32, device=device)
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
            wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array(body_to_articulation, dtype=wp.int32, device=device),
            wp.array([0] * articulation_count, dtype=wp.int32, device=device),
            wp.array([1] * articulation_count, dtype=wp.int32, device=device),
            wp.zeros((2,), dtype=wp.int32, device=device),
            wp.ones((2,), dtype=wp.int32, device=device),
            wp.zeros((2,), dtype=wp.int32, device=device),
            0,
            0,
            0,
            0,
            0.0,
            scoped_gate,
            pair_gate,
            8,
            8,
            8,
            int(enable_friction),
            friction_gap,
            int(friction_pairs_only),
            0,
            FrictionPatches(),
        ],
        outputs=[
            wp.full((1,), -9, dtype=wp.int32, device=device),
            contact_slot,
            wp.full((1,), -9, dtype=wp.int32, device=device),
            wp.full((1,), -9, dtype=wp.int32, device=device),
            dense_count,
            contact_path,
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            contact_slots_needed,
            dense_dropped,
            mf_dropped,
            propagation_dropped,
            wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
            wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
            wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    wp.synchronize_device(device)
    slots_needed = max(0, int(contact_slots_needed.numpy()[0]))
    return int(contact_slot.numpy()[0]), int(contact_path.numpy()[0]), slots_needed


def _dense_speculative_rhs(scale: float) -> float:
    """Return the dense positive-gap RHS for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    wp.launch(
        compute_world_contact_bias,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device="cpu"),
            1,
            wp.array([[1.0]], dtype=wp.float32, device="cpu"),
            wp.array([[0.2]], dtype=wp.float32, device="cpu"),
            wp.array([[0]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            0.5,
            1.0,
            scale,
            1.0,
            1.0,
        ],
        outputs=[rhs, wp.zeros((1, 1), dtype=wp.float32, device="cpu")],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


def _mf_speculative_rhs(scale: float) -> float:
    """Return the MF setup RHS for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    wp.launch(
        compute_mf_effective_mass_and_rhs,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([[-1]], dtype=wp.int32, device="cpu"),
            wp.array([[-1]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1,), dtype=wp.spatial_matrix, device="cpu"),
            wp.array([[1.0]], dtype=wp.float32, device="cpu"),
            wp.array([[0]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            0,
            wp.array([-1], dtype=wp.int32, device="cpu"),
            wp.array([0], dtype=wp.int32, device="cpu"),
            wp.zeros((1,), dtype=wp.float32, device="cpu"),
            wp.array([float("inf")], dtype=wp.float32, device="cpu"),
            1.0,
            0.2,
            1.0,
            0.5,
            scale,
            0.5,
            1,
        ],
        outputs=[
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            rhs,
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
        ],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


def _propagation_speculative_rhs(scale: float) -> float:
    """Return the propagation setup RHS for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    wp.launch(
        compute_propagation_effective_mass_and_rhs,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([[-1]], dtype=wp.int32, device="cpu"),
            wp.array([[-1]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 6, 6), dtype=wp.float32, device="cpu"),
            wp.array([[1.0]], dtype=wp.float32, device="cpu"),
            wp.array([[0]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 6), dtype=wp.float32, device="cpu"),
            wp.array([float("inf")], dtype=wp.float32, device="cpu"),
            1.0,
            0.2,
            1.0,
            0.5,
            scale,
            0.5,
            1,
        ],
        outputs=[
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            rhs,
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
        ],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


def _dense_restitution_rhs(scale: float) -> float:
    """Apply dense restitution after accumulating the scaled position RHS."""
    rhs = wp.array([[-2.0 + scale]], dtype=wp.float32, device="cpu")
    wp.launch(
        apply_world_contact_restitution_accumulated,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device="cpu"),
            1,
            wp.array([[1.0]], dtype=wp.float32, device="cpu"),
            wp.array([[0.2]], dtype=wp.float32, device="cpu"),
            wp.array([[0]], dtype=wp.int32, device="cpu"),
            wp.array([[0.5]], dtype=wp.float32, device="cpu"),
            1.0,
            scale,
            0.0,
            0,
        ],
        outputs=[rhs, wp.zeros((1, 1), dtype=wp.float32, device="cpu")],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


def _contact_row_outputs(device: str) -> dict[str, wp.array]:
    """Allocate dense-contact Jacobian and metadata outputs."""
    return {
        "J": wp.zeros((1, 8, 3), dtype=wp.float32, device=device),
        "row_type": wp.full((1, 8), -9, dtype=wp.int32, device=device),
        "row_parent": wp.full((1, 8), -9, dtype=wp.int32, device=device),
        "row_mu": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "row_beta": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "row_cfm": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "phi": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "target_velocity": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "row_restitution": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
    }


def _launch_dense_contact_builders(device: str = "cpu") -> tuple[dict[str, wp.array], dict[str, wp.array]]:
    """Build one same-articulation contact through serial and compact paths."""
    contact_count = wp.array([1], dtype=wp.int32, device=device)
    point0 = wp.array([wp.vec3(0.2, 0.1, -0.05)], dtype=wp.vec3, device=device)
    point1 = wp.array([wp.vec3(-0.1, 0.05, 0.2)], dtype=wp.vec3, device=device)
    normal = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3, device=device)
    shape0 = wp.array([0], dtype=wp.int32, device=device)
    shape1 = wp.array([1], dtype=wp.int32, device=device)
    thickness0 = wp.array([0.01], dtype=wp.float32, device=device)
    thickness1 = wp.array([0.02], dtype=wp.float32, device=device)
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
        [
            wp.transform(wp.vec3(0.0), wp.quat_identity()),
            wp.transform(wp.vec3(-0.2, 0.3, 0.1), wp.quat_identity()),
            wp.transform(wp.vec3(0.4, -0.1, 0.2), wp.quat_identity()),
        ],
        dtype=wp.transform,
        device=device,
    )
    body_v = wp.zeros((3,), dtype=wp.spatial_vector, device=device)
    prescribed = wp.zeros((1,), dtype=wp.int32, device=device)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    material_mu = wp.array([0.6, 0.8], dtype=wp.float32, device=device)
    material_restitution = wp.array([0.2, 0.4], dtype=wp.float32, device=device)

    common_geometry = [
        contact_count,
        point0,
        point1,
        normal,
        shape0,
        shape1,
        thickness0,
        thickness1,
    ]
    serial = _contact_row_outputs(device)
    wp.launch(
        populate_world_J_for_size,
        dim=1,
        inputs=[
            contact_count,
            1,
            *common_geometry[1:],
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
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
            body_v,
            prescribed,
            shape_transform,
            material_mu,
            material_restitution,
            1,
            1.0,
            1,
            0,
            wp.zeros((1,), dtype=wp.int32, device=device),
            1.0,
            0,
            0.05,
            1.0e-6,
            _disabled_patches(device),  # patch friction off
            0.0,  # friction_anchor_beta
        ],
        outputs=list(serial.values()),
        device=device,
    )

    compact = _contact_row_outputs(device)
    wp.launch(
        prepare_world_contact_rows,
        dim=1,
        inputs=[
            contact_count,
            1,
            *common_geometry[1:],
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
            contact_slots_needed,
            shape_body,
            body_q,
            body_v,
            prescribed,
            articulation_origin,
            material_mu,
            material_restitution,
            1,
            1.0,
            0,
            0.05,
            1.0e-6,
            _disabled_patches(device),  # patch friction off
            0.0,  # friction_anchor_beta
        ],
        outputs=list(compact.values())[1:],
        device=device,
    )
    wp.launch(
        populate_world_J_for_compact_size,
        dim=(1, 32),
        inputs=[
            contact_count,
            1,
            *common_geometry[1:],
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
            1,
            FrictionPatches(),
            0,
        ],
        outputs=[compact["J"]],
        device=device,
    )
    wp.synchronize_device(device)
    return serial, compact


class TestFeatherPGSContactControls(unittest.TestCase):
    def test_compact_contact_builder_matches_tree_walk(self):
        """Match dense Jacobians and metadata for a same-articulation contact."""
        serial, compact = _launch_dense_contact_builders()

        for name in serial:
            with self.subTest(output=name):
                np.testing.assert_allclose(compact[name].numpy(), serial[name].numpy(), rtol=0.0, atol=1.0e-6)

    def test_solver_exposes_documented_defaults(self):
        """Expose legacy-preserving defaults through the public constructor."""
        solver = SolverFeatherPGS(newton.ModelBuilder().finalize(device="cpu"))

        self.assertEqual(solver.contact_speculative_scale, 1.0)
        self.assertEqual(solver.contact_gap_gate, 0.0)
        self.assertEqual(solver.same_articulation_contact_gap_gate, 0.0)
        self.assertEqual(solver.articulation_pair_contact_gap_gate, 0.0)
        self.assertFalse(solver.contact_friction_articulation_pairs_only)
        self.assertTrue(solver.warn_constraint_overflow)
        self.assertTrue(solver._track_row_capacity)
        self.assertFalse(solver._row_watermark)
        quiet_solver = SolverFeatherPGS(
            newton.ModelBuilder().finalize(device="cpu"),
            warn_constraint_overflow=False,
        )
        self.assertFalse(quiet_solver.warn_constraint_overflow)
        self.assertFalse(quiet_solver._track_row_capacity)
        self.assertIsNone(quiet_solver._row_overflow_warning_emitted)
        parameters = tuple(inspect.signature(SolverFeatherPGS).parameters)
        self.assertIn("same_articulation_contact_gap_gate", parameters)
        self.assertIn("contact_friction_articulation_pairs_only", parameters)

    def test_new_contact_controls_preserve_legacy_positional_layout(self):
        """Append new controls without shifting established positional arguments."""
        parameters = tuple(inspect.signature(SolverFeatherPGS).parameters)
        for sequence in (
            ("contact_friction_anchor_limit", "contact_friction_scale", "contact_shared_anchor"),
            ("row_watermark", "restitution_velocity_threshold", "contact_speculative_scale", "contact_gap_gate"),
        ):
            start = parameters.index(sequence[0])
            self.assertEqual(parameters[start : start + len(sequence)], sequence)

        legacy_tail = parameters.index("contact_gap_gate")
        for name in (
            "contact_friction_articulation_pairs_only",
            "enable_restitution",
            "same_articulation_contact_gap_gate",
            "articulation_pair_contact_gap_gate",
            "warn_constraint_overflow",
        ):
            self.assertGreater(parameters.index(name), legacy_tail)

    def test_solver_validates_and_stores_contact_controls(self):
        """Accept finite non-negative controls and reject malformed values."""
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = SolverFeatherPGS(
            model,
            contact_speculative_scale=0.0,
            contact_gap_gate=0.001,
            same_articulation_contact_gap_gate=0.002,
            articulation_pair_contact_gap_gate=0.003,
            contact_friction_articulation_pairs_only=True,
        )
        self.assertEqual(solver.contact_speculative_scale, 0.0)
        self.assertEqual(solver.contact_gap_gate, 0.001)
        self.assertEqual(solver.same_articulation_contact_gap_gate, 0.002)
        self.assertEqual(solver.articulation_pair_contact_gap_gate, 0.003)
        self.assertTrue(solver.contact_friction_articulation_pairs_only)

        for name in (
            "contact_speculative_scale",
            "contact_gap_gate",
            "same_articulation_contact_gap_gate",
            "articulation_pair_contact_gap_gate",
        ):
            for value in (-0.1, float("nan"), float("inf"), "invalid"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        SolverFeatherPGS(model, **{name: value})

    def test_scoped_gap_gate_only_drops_distant_same_articulation_contact(self):
        """The scoped gate retains near self-contact while bounding its speculative tail."""
        self.assertEqual(
            _launch_articulation_pair_contact_allocator(gap=0.002, scoped_gate=0.003),
            (0, PATH_DENSE, 1),
        )
        self.assertEqual(
            _launch_articulation_pair_contact_allocator(gap=0.004, scoped_gate=0.003),
            (-1, -1, 0),
        )
        self.assertEqual(
            _launch_articulation_pair_contact_allocator(gap=0.004, scoped_gate=0.0),
            (0, PATH_DENSE, 1),
        )

    def test_scoped_gap_gate_preserves_other_contact_routes(self):
        """Do not shorten predictive contacts for ground, free-body, or cross-articulation rows."""
        for route, counter in (
            (PATH_DENSE, "dense_count"),
            (PATH_MATRIX_FREE, "mf_count"),
            (PATH_PROPAGATION, "propagation_count"),
        ):
            with self.subTest(route=route):
                result = _launch_contact_allocator(
                    route=route,
                    gap=0.04,
                    gate=0.0,
                    scoped_gate=0.003,
                )
                self.assertEqual(result["path"], route)
                self.assertEqual(result[counter], 1)

    def test_articulation_pair_gap_gate_drops_distant_pair_contact(self):
        """The pair gate includes same/cross-articulation contact without touching free bodies."""
        for cross_articulation in (False, True):
            with self.subTest(cross_articulation=cross_articulation):
                self.assertEqual(
                    _launch_articulation_pair_contact_allocator(
                        gap=0.004,
                        scoped_gate=0.0,
                        pair_gate=0.003,
                        cross_articulation=cross_articulation,
                    ),
                    (-1, -1, 0),
                )
        result = _launch_contact_allocator(
            route=PATH_MATRIX_FREE,
            gap=0.04,
            gate=0.0,
            pair_gate=0.003,
        )
        self.assertEqual(result["path"], PATH_MATRIX_FREE)
        self.assertEqual(result["mf_count"], 1)

    def test_articulation_pair_friction_filter_preserves_free_and_ground_rows(self):
        """Scope tight friction controls to articulated pairs, not ball/ground routes."""
        for route in (PATH_DENSE, PATH_MATRIX_FREE, PATH_PROPAGATION):
            with self.subTest(route=route):
                result = _launch_contact_allocator(
                    route=route,
                    gap=0.004,
                    gate=0.0,
                    enable_friction=True,
                    friction_gap=0.002,
                    friction_pairs_only=True,
                )
                self.assertEqual(result["path"], route)
                self.assertEqual(result["slots_needed"], 3)

    def test_articulation_pair_friction_filter_reduces_pair_rows(self):
        """Apply the configured friction gap to same- and cross-articulation contacts."""
        for cross_articulation in (False, True):
            with self.subTest(cross_articulation=cross_articulation):
                slot, path, slots_needed = _launch_articulation_pair_contact_allocator(
                    gap=0.004,
                    cross_articulation=cross_articulation,
                    enable_friction=True,
                    friction_gap=0.002,
                    friction_pairs_only=True,
                )
                self.assertEqual((slot, path, slots_needed), (0, PATH_DENSE, 1))

    def test_negative_scoped_friction_gap_delays_only_articulation_pair_tangents(self):
        """Keep pair normals while delaying friction, without changing free-body friction."""
        shallow_pair = _launch_articulation_pair_contact_allocator(
            gap=-0.0005,
            enable_friction=True,
            friction_gap=-0.001,
            friction_pairs_only=True,
        )
        deep_pair = _launch_articulation_pair_contact_allocator(
            gap=-0.002,
            enable_friction=True,
            friction_gap=-0.001,
            friction_pairs_only=True,
        )
        free_body = _launch_contact_allocator(
            route=PATH_MATRIX_FREE,
            gap=0.004,
            gate=0.0,
            enable_friction=True,
            friction_gap=-1.0,
            friction_pairs_only=True,
        )
        self.assertEqual(shallow_pair, (0, PATH_DENSE, 1))
        self.assertEqual(deep_pair, (0, PATH_DENSE, 3))
        self.assertEqual(free_body["slots_needed"], 3)

    def test_speculative_scale_controls_every_position_rhs_family(self):
        """Scale positive-gap position bias on dense, MF, and propagation rows."""
        rhs_families = {
            "dense": _dense_speculative_rhs,
            "matrix_free": _mf_speculative_rhs,
            "propagation": _propagation_speculative_rhs,
        }
        for name, compute_rhs in rhs_families.items():
            with self.subTest(family=name):
                self.assertEqual(compute_rhs(0.0), 0.0)
                self.assertAlmostEqual(compute_rhs(1.0), 2.0, places=6)

    def test_dense_restitution_removes_the_scaled_position_bias(self):
        """Recover the same incident speed before restitution at scales zero and one."""
        self.assertEqual(_dense_restitution_rhs(0.0), -3.0)
        self.assertEqual(_dense_restitution_rhs(1.0), -3.0)

    def test_gap_gate_prevents_all_route_allocations(self):
        """Drop positive gaps above the gate before any route reserves a slot."""
        for route in (PATH_DENSE, PATH_MATRIX_FREE, PATH_PROPAGATION):
            with self.subTest(route=route):
                allocated = _launch_contact_allocator(route=route, gap=0.002, gate=0.0)
                self.assertEqual(allocated["path"], route)
                self.assertEqual(allocated["slot"], 0)
                self.assertEqual(allocated["world"], 0)
                self.assertEqual(allocated["art_a"], 0)
                self.assertEqual(allocated["art_b"], -1)
                self.assertEqual(allocated["slots_needed"], 1)
                self.assertEqual(
                    (
                        allocated["dense_count"],
                        allocated["mf_count"],
                        allocated["propagation_count"],
                    ),
                    (
                        int(route == PATH_DENSE),
                        int(route == PATH_MATRIX_FREE),
                        int(route == PATH_PROPAGATION),
                    ),
                )

                dropped = _launch_contact_allocator(route=route, gap=0.002, gate=0.001)
                self.assertEqual(dropped["slot"], -1)
                self.assertEqual(dropped["path"], -1)
                self.assertEqual(dropped["dense_count"], 0)
                self.assertEqual(dropped["mf_count"], 0)
                self.assertEqual(dropped["propagation_count"], 0)
                self.assertEqual(dropped["dense_world_flag"], 0)

    def test_gap_gate_keeps_contacts_at_threshold(self):
        """Keep a contact whose gap equals the positive gate exactly."""
        for route in (PATH_DENSE, PATH_MATRIX_FREE, PATH_PROPAGATION):
            with self.subTest(route=route):
                result = _launch_contact_allocator(route=route, gap=0.001, gate=0.001)
                self.assertEqual(result["path"], route)
                self.assertEqual(result["slots_needed"], 1)

    def test_allocator_skips_contacts_without_response_dofs(self):
        """Do not allocate a row when neither contact body can change velocity."""
        result = _launch_contact_allocator(route=PATH_DENSE, gap=-0.001, gate=0.0, responsive=False)

        self.assertEqual(result["slot"], -1)
        self.assertEqual(result["path"], -1)
        self.assertEqual(result["dense_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
