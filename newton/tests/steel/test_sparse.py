# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse contact, mass-matrix and articulation-local solves of SolverSteel."""

import unittest
from itertools import product
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.friction import friction_pair_candidate
from newton._src.solvers.steel.friction_patches import FrictionPatches
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
    ROW_SLOTS_ATOMIC,
    classify_local_solve_worlds,
    compact_local_pair_candidates,
    populate_world_J_for_compact_size,
)
from newton._src.solvers.steel.solver_steel import (
    _get_build_independent_sparse_contact_groups_kernel,
    _get_mark_independent_sparse_contact_candidates_kernel,
    _get_pgs_solve_local_owned_kernel,
    _get_pgs_solve_sparse_diagonal_kernel,
)
from newton._src.solvers.steel.sparse_contact import (
    _get_sparse_contact_response_kernel,
    apply_sparse_factor_velocity,
    apply_sparse_free_velocity,
    build_sparse_joint_limit_rows,
)
from newton._src.solvers.steel.sparse_mass_matrix import (
    _get_crba_sparse_factor_kernel,
    _SparseMassMatrixPlan,
    solve_sparse_mass_matrix,
)
from newton._src.solvers.steel.sparse_pgs import _get_pgs_solve_sparse_kernel
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import (
    _build_mixed_response_model,
    _build_sparse_contact_friction_model,
    set_torsion_radius,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _point_friction_patches(device):
    """Return the disabled friction-patch view: every contact gets its two friction rows."""
    patches = FrictionPatches()
    patches.enabled = 0
    patches.weight = wp.zeros(0, dtype=float, device=device)
    patches.next_contact = wp.zeros(0, dtype=int, device=device)
    patches.point_a = wp.zeros(0, dtype=wp.vec3, device=device)
    patches.point_b = wp.zeros(0, dtype=wp.vec3, device=device)
    patches.phi = wp.zeros(1, dtype=wp.vec2, device=device)
    return patches


def _contact_directions(normal):
    """Return the normal and the two friction directions of a contact row triple."""
    tangent0 = np.cross(normal, (1.0, 0.0, 0.0))
    if tangent0 @ tangent0 < 1.0e-12:
        tangent0 = np.cross(normal, (0.0, 1.0, 0.0))
    tangent0 /= np.linalg.norm(tangent0)
    tangent1 = np.cross(normal, tangent0)
    tangent1 /= np.linalg.norm(tangent1)
    return normal, tangent0, tangent1


def _array(value, dtype=float):
    return wp.array(value, dtype=dtype, device="cpu")


def _full(shape, value, dtype=float):
    return wp.full(shape, value, dtype=dtype, device="cpu")


def _fixture(device="cpu", joint_dof_count=(3, 2, 1, 0)):
    """Make a two-branch articulation with a multi-DOF root and fixed endpoint."""

    def array(value, dtype=float):
        return wp.array(value, dtype=dtype, device=device)

    plan = _SparseMassMatrixPlan.build([-1, 0, 0, 1], joint_dof_count)
    dofs = plan.dof_count
    rng = np.random.default_rng(916)
    lower = np.zeros((dofs, dofs))
    lower[plan.entry_rows, plan.columns] = rng.normal(scale=0.3, size=plan.nonzero_count)
    lower[np.diag_indices(dofs)] = 2.0
    inverse = np.linalg.inv(lower)
    normals = rng.normal(size=(3, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    physical_masks = [sum(1 << int(plan.permutation[i]) for i in plan.endpoint_support(j)) for j in range(4)]
    return {
        "plan": plan,
        "indices": plan.to_device(device),
        "inverse": inverse,
        "packed": array(inverse[plan.entry_rows, plan.columns][None, :]),
        "count": array([3], int),
        "point0": array(rng.normal(size=(3, 3)), wp.vec3),
        "point1": array(rng.normal(size=(3, 3)), wp.vec3),
        "normal": array(normals, wp.vec3),
        "shape0": array([0, 0, 2], int),
        "shape1": array([3, 1, 3], int),
        "thickness0": array([0.01, 0.02, 0.03]),
        "thickness1": array([0.03, 0.01, 0.02]),
        "world": array([0, 0, 0], int),
        "slot": array([0, 3, 6], int),
        "art_a": array([0, 0, 0], int),
        "art_b": array([-1, 0, -1], int),
        "path": array([0, 0, 0], int),
        "group": array([0], int),
        "start": array([0], int),
        "origin": array([[0.3, -0.4, 0.7]], wp.vec3),
        "sparse_mask": array(plan.joint_ancestor_mask, wp.uint64),
        "physical_mask": array(physical_masks, wp.uint32 if dofs <= 32 else wp.uint64),
        "motion": array(rng.normal(size=(dofs, 6)), wp.spatial_vector),
        "shape_body": array([1, 2, 3, -1], int),
        "body_q": array([wp.transform(wp.vec3(*p), wp.quat_identity()) for p in rng.normal(size=(4, 3))], wp.transform),
        "velocity": array(rng.normal(size=dofs)),
    }


class TestSparseContacts(unittest.TestCase):
    def test_mixed_contact_response_matches_independent_mass_oracle(self):
        """Match physical mixed responses with arbitrary offsets and wide ancestry support."""
        for counts in ((1, 1, 1, 0), (3, 2, 1, 0), (6, 19, 18, 0), (6, 29, 29, 0)):
            with self.subTest(joint_dof_count=counts):
                self._check_mixed_contact_response("cpu", 1, counts)

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative contact construction requires CUDA")
    def test_mixed_contact_response_cuda(self):
        """Preserve mixed responses across partial warps, group widths and grid strides."""
        for lanes in (8, 16, 32):
            counts_to_check = (
                ((1, 1, 1, 0), (3, 2, 1, 0), (6, 19, 18, 0), (6, 29, 29, 0))
                if lanes == 8
                else ((3, 2, 1, 0), (6, 29, 29, 0))
            )
            for counts in counts_to_check:
                for workers in (1, 2, 7) if lanes == 8 else (2,):
                    with self.subTest(joint_dof_count=counts, workers=workers, lanes=lanes):
                        self._check_mixed_contact_response("cuda:0", workers, counts, lanes=lanes)

    def _check_mixed_contact_response(self, device, workers, joint_dof_count=(3, 2, 1, 0), *, lanes=8):
        """Match robot/robot, robot/object, object/object, ground and self-contact at distinct offsets."""

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def full(shape, value, dtype=float):
            return wp.full(shape, value, dtype=dtype, device=device)

        f = _fixture(device, joint_dof_count)
        p, indices = f["plan"], f["indices"]
        dofs = p.dof_count
        rng = np.random.default_rng(500)
        art_a = np.array([1, 0, 0, 2, 1, 1, 3, 3], dtype=np.int32)
        art_b = np.array([0, 1, 2, -1, 1, 3, 1, -1], dtype=np.int32)
        body_a = np.array([1, 4, 4, 5, 1, 1, 8, 9], dtype=np.int32)
        body_b = np.array([4, 1, 5, -1, 2, 8, 1, -1], dtype=np.int32)
        contact_count = len(art_a)
        articulated = np.array([False, True, False, True])
        start = np.array([1, 8, 8 + dofs + 6, 8 + dofs + 16])
        offset, group = np.array([0, 6, 6 + dofs, 12 + dofs]), np.array([1, 0, 0, 1])
        world_dofs = 2 * dofs + 12
        global_dofs = int(start[3] + dofs)
        second_lower = np.zeros((dofs, dofs))
        second_lower[p.entry_rows, p.columns] = rng.normal(scale=0.2, size=p.nonzero_count)
        second_lower[np.diag_indices(dofs)] = 1.7
        articulated_inverse = np.stack((f["inverse"], np.linalg.inv(second_lower)))
        free_lower = np.tril(rng.normal(scale=0.1, size=(2, 6, 6))) + 1.4 * np.eye(6)
        free_inverse = np.linalg.inv(free_lower)
        origin = rng.normal(size=(4, 3)).astype(np.float32)
        motion = rng.normal(size=(global_dofs, 6)).astype(np.float32)
        velocity = rng.normal(size=global_dofs).astype(np.float32)
        point_a, point_b = rng.normal(size=(2, contact_count, 3)).astype(np.float32)
        normal = rng.normal(size=(contact_count, 3))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        normal = normal.astype(np.float32)
        # Reversing either a mixed or two-articulation contact preserves its row.
        for first, reverse in ((0, 1), (5, 6)):
            point_a[reverse], point_b[reverse], normal[reverse] = point_b[first], point_a[first], -normal[first]
        transforms = [
            wp.transform(wp.vec3(*position), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(angle)))
            for position, angle in zip(rng.normal(size=(10, 3)), np.linspace(-0.6, 0.8, 10), strict=True)
        ]
        masks = np.concatenate((p.joint_ancestor_mask, np.array([63, 63], dtype=np.uint64), p.joint_ancestor_mask))
        # Every dense contact owns a normal row and two friction rows.
        row_count = 3 * contact_count
        row_shape = (1, row_count, max(2 * dofs, 12))
        row_dof, row_factor = full(row_shape, -9, int), full(row_shape, -9.0)
        row_free_response = full(row_shape, -9.0)
        incident, diagonal = full(row_shape[:2], -9.0), full(row_shape[:2], -9.0)
        wp.launch(
            _get_sparse_contact_response_kernel(dofs, lanes_per_contact=lanes),
            dim=workers * lanes,
            inputs=[
                array([contact_count], int),
                workers,
                array(point_a, wp.vec3),
                array(point_b, wp.vec3),
                array(normal, wp.vec3),
                array(body_a, int),
                array(np.where(body_b < 0, len(transforms), body_b), int),
                full(contact_count, 0.0),
                full(contact_count, 0.0),
                full(contact_count, 0, int),
                array(3 * np.arange(contact_count), int),
                array(art_a, int),
                array(art_b, int),
                full(contact_count, 0, int),
                array(group, int),
                array(np.where(articulated, dofs, 6), int),
                array(~articulated, int),
                array(offset, int),
                array(start, int),
                array(origin, wp.vec3),
                array(masks, wp.uint64),
                array(motion, wp.spatial_vector),
                array([*range(len(transforms)), -1], int),
                array(transforms, wp.transform),
                0,
                0,
                indices.permutation,
                indices.row_offsets,
                array(articulated_inverse[:, p.entry_rows, p.columns]),
                array(free_inverse),
                array(velocity),
            ],
            outputs=[row_dof, row_factor, row_free_response, incident, diagonal],
            block_dim=128,
            device=device,
        )
        mass, transform = np.zeros((world_dofs, world_dofs)), np.zeros((world_dofs, world_dofs))
        compact_velocity = np.zeros(world_dofs)
        free_coordinates = np.zeros(world_dofs, dtype=bool)
        for art, is_articulated in enumerate(articulated):
            width = dofs if is_articulated else 6
            block = slice(offset[art], offset[art] + width)
            permutation = p.permutation if is_articulated else np.arange(6)
            inverse = articulated_inverse[group[art]] if is_articulated else free_inverse[group[art]]
            local_transform = inverse @ np.eye(width)[permutation]
            transform[block, block] = local_transform if is_articulated else np.eye(6)
            local_decode = np.linalg.inv(local_transform)
            mass[block, block] = local_decode @ local_decode.T
            compact_velocity[block] = velocity[start[art] : start[art] + width]
            free_coordinates[block] = not is_articulated
        jacobian = np.zeros((row_count, world_dofs))
        for contact in range(contact_count):
            directions = _contact_directions(-normal[contact].astype(np.float64))
            for body, art, point, sign in (
                (body_a[contact], art_a[contact], point_a[contact], 1.0),
                (body_b[contact], art_b[contact], point_b[contact], -1.0),
            ):
                if body < 0:
                    continue
                position = np.asarray(wp.transform_point(transforms[body], wp.vec3(*point)))
                permutation = p.permutation if articulated[art] else np.arange(6)
                for node, physical in enumerate(permutation):
                    if not (int(masks[body]) & (1 << node)):
                        continue
                    axis = motion[start[art] + physical]
                    point_velocity = axis[:3] + np.cross(axis[3:], position - origin[art])
                    for k, direction in enumerate(directions):
                        jacobian[3 * contact + k, offset[art] + physical] += sign * (direction @ point_velocity)
        actual = np.zeros((row_count, world_dofs))
        actual_response = np.zeros_like(actual)
        for row, support in enumerate(row_dof.numpy()[0]):
            valid = support >= 0
            self.assertEqual(np.unique(support[valid]).size, np.count_nonzero(valid))
            actual[row, support[valid]] = row_factor.numpy()[0, row, valid]
            for entry, coordinate in enumerate(support):
                if coordinate < 0:
                    continue
                actual_response[row, coordinate] = (
                    row_free_response.numpy()[0, row, entry]
                    if free_coordinates[coordinate]
                    else row_factor.numpy()[0, row, entry]
                )
        expected = jacobian @ transform.T
        np.testing.assert_allclose(actual, expected, atol=3.0e-6, rtol=3.0e-6)
        # Reversing a contact flips its normal, so its normal row is unchanged.
        np.testing.assert_allclose(actual[0], actual[3], atol=3.0e-6)
        np.testing.assert_allclose(actual[15], actual[18], atol=3.0e-6)
        np.testing.assert_allclose(incident.numpy()[0], jacobian @ compact_velocity, atol=6.0e-6, rtol=3.0e-6)
        gram = jacobian @ np.linalg.solve(mass, jacobian.T)
        expected_response = np.linalg.solve(mass, jacobian.T).T @ np.linalg.inv(transform)
        np.testing.assert_allclose(actual_response, expected_response, atol=3.0e-6, rtol=3.0e-6)
        np.testing.assert_allclose(actual @ actual_response.T, gram, atol=1.0e-5, rtol=4.0e-6)
        np.testing.assert_allclose(diagonal.numpy()[0], np.diag(gram), atol=1.0e-5, rtol=4.0e-6)

    def test_free_velocity_uses_world_offsets(self):
        """Publish physical free-body deltas without overwriting robot velocities."""
        rng = np.random.default_rng(502)
        group_to_art = np.array([4, 0, 3, 2], dtype=np.int32)
        world = np.array([1, 0, 0, 1, 0], dtype=np.int32)
        offset = np.array([9, 0, 9, 15, 15], dtype=np.int32)
        start = np.array([0, 6, 15, 21, 27], dtype=np.int32)
        delta = rng.normal(size=(2, 21)).astype(np.float32)
        velocity = rng.normal(size=33).astype(np.float32)
        out = _array(velocity)
        wp.launch(
            apply_sparse_free_velocity,
            dim=(4, 6),
            inputs=[
                _array(group_to_art, int),
                _array(world, int),
                _array(offset, int),
                _array(start, int),
                _array(delta),
                _array(velocity),
            ],
            outputs=[out],
            device="cpu",
        )
        expected = velocity.copy()
        for art in group_to_art:
            expected[start[art] : start[art] + 6] += delta[world[art], offset[art] : offset[art] + 6]
        np.testing.assert_allclose(out.numpy(), expected, atol=3.0e-7, rtol=2.0e-6)
        np.testing.assert_array_equal(out.numpy()[6:15], velocity[6:15])

    def test_contact_response_matches_dense_jacobian(self):
        """Match dense contact rows for static, same-articulation, and fixed endpoints."""
        for anchors in ((0, 0), (1, 0), (0, 1)):
            with self.subTest(anchors=anchors):
                self._check_contact_response("cpu", 1, anchors=anchors)

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative contact construction requires CUDA")
    def test_contact_response_cuda(self):
        """Match the dense contact rows with cooperative writes across lane groups and grid strides."""
        for lanes in (8, 16, 32):
            for workers in (1, 2, 5) if lanes == 8 else (2,):
                with self.subTest(workers=workers, lanes=lanes):
                    self._check_contact_response("cuda:0", workers, lanes=lanes)
        with self.subTest(anchors=(1, 1)):
            self._check_contact_response("cuda:0", 2, anchors=(1, 1))

    def _check_contact_response(self, device, workers, *, lanes=8, anchors=(0, 0)):
        """Match dense contact rows for static, same-articulation, and fixed endpoints."""

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def full(shape, value, dtype=float):
            return wp.full(shape, value, dtype=dtype, device=device)

        f = _fixture(device)
        p = f["plan"]
        indices = f["indices"]
        geometry = [f[name] for name in ("point0", "point1", "normal", "shape0", "shape1", "thickness0", "thickness1")]
        dense = full((1, 9, 6), 0.0)
        row_dof = full((1, 9, 6), -9, int)
        row_factor = full((1, 9, 6), -9.0)
        incident = full((1, 9), -9.0)
        diagonal = full((1, 9), -9.0)
        wp.launch(
            populate_world_J_for_compact_size,
            dim=(1, 32),
            inputs=[
                f["count"],
                1,
                *geometry,
                f["slot"],
                f["art_a"],
                f["art_b"],
                f["path"],
                array([3, 3, 3], int),
                6,
                array([6], int),
                f["group"],
                f["start"],
                f["origin"],
                f["physical_mask"],
                f["motion"],
                f["shape_body"],
                f["body_q"],
                _point_friction_patches(device),
                *anchors,
            ],
            outputs=[dense],
            device=device,
        )
        wp.launch(
            _get_sparse_contact_response_kernel(6, lanes_per_contact=lanes),
            dim=workers * lanes,
            inputs=[
                f["count"],
                workers,
                *geometry,
                f["world"],
                f["slot"],
                f["art_a"],
                f["art_b"],
                f["path"],
                f["group"],
                array([6], int),
                array([0], int),
                array([0], int),
                f["start"],
                f["origin"],
                f["sparse_mask"],
                f["motion"],
                f["shape_body"],
                f["body_q"],
                *anchors,
                indices.permutation,
                indices.row_offsets,
                f["packed"],
                array(np.eye(6)[None]),
                f["velocity"],
            ],
            outputs=[row_dof, row_factor, full((1, 9, 12), -9.0), incident, diagonal],
            block_dim=128,
            device=device,
        )
        jacobian = dense.numpy()[0]
        expected = jacobian[:, p.permutation] @ f["inverse"].T
        actual = np.zeros_like(expected)
        for row in range(9):
            dofs = row_dof.numpy()[0, row]
            valid = dofs >= 0
            self.assertEqual(len(set(dofs[valid])), int(np.count_nonzero(valid)))
            actual[row, dofs[valid]] = row_factor.numpy()[0, row, valid]
        np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(incident.numpy()[0], jacobian @ f["velocity"].numpy(), rtol=3.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(diagonal.numpy()[0], np.sum(expected**2, axis=1), rtol=3.0e-6, atol=2.0e-6)

    def test_joint_limit_rows_and_capacity(self):
        """Match signed inverse-factor columns and preserve limit allocation order."""
        f = _fixture()
        plan, indices = f["plan"], f["indices"]
        expected_rows = [(0, 1.0, -0.02), (1, -1.0, 0.01), (3, -1.0, -0.2), (4, 1.0, 0.0), (4, -1.0, 0.0)]
        for capacity in (6, 3):
            with self.subTest(capacity=capacity):
                count = _array([1], int)
                row_type, parent = [_full((1, capacity), -9, int) for _ in range(2)]
                mu, phi, target, incident, diagonal = [_full((1, capacity), -9.0) for _ in range(5)]
                dof = _full((1, capacity, 6), -9, int)
                factor = _full((1, capacity, 6), -9.0)
                wp.launch(
                    build_sparse_joint_limit_rows,
                    dim=32,
                    inputs=[
                        f["group"],
                        _array([0], int),
                        _array([0], int),
                        f["start"],
                        _array([0, 1, -1, 3, 4, 5], int),
                        _array([-1.0, -1.0, -np.inf, -0.5, 0.0, -2.0]),
                        _array([1.0, 1.0, np.inf, 0.5, 0.0, 2.0]),
                        _array([-1.02, 0.99, 0.0, 0.7, 0.0, -1.0]),
                        0.02,
                        indices.ancestor_mask,
                        indices.inverse_permutation,
                        indices.lookup,
                        f["packed"],
                        f["velocity"],
                    ],
                    outputs=[
                        count,
                        row_type,
                        parent,
                        mu,
                        phi,
                        target,
                        dof,
                        factor,
                        incident,
                        diagonal,
                        ROW_SLOTS_ATOMIC,
                        wp.zeros(1, dtype=wp.int32, device="cpu"),
                    ],
                    block_dim=128,
                    device="cpu",
                )
                self.assertEqual(count.numpy()[0], 6)
                self.assertEqual(row_type.numpy()[0, 0], -9)
                for row, (physical, sign, gap) in enumerate(expected_rows[: capacity - 1], 1):
                    expected = sign * f["inverse"][:, plan.inverse_permutation[physical]]
                    actual = np.zeros(6)
                    active = dof.numpy()[0, row] >= 0
                    actual[dof.numpy()[0, row, active]] = factor.numpy()[0, row, active]
                    np.testing.assert_allclose(actual, expected, atol=1.0e-7)
                    self.assertEqual(row_type.numpy()[0, row], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
                    self.assertEqual(parent.numpy()[0, row], -1)
                    self.assertAlmostEqual(float(phi.numpy()[0, row]), gap, places=6)
                    self.assertAlmostEqual(
                        float(incident.numpy()[0, row]), sign * f["velocity"].numpy()[physical], places=6
                    )
                    self.assertAlmostEqual(float(diagonal.numpy()[0, row]), float(expected @ expected), places=6)

    def _check_joint_limit_parallel_scan(self, device, capacity):
        plan = _SparseMassMatrixPlan.build([-1, 0, 0], [2, 31, 31])
        dofs, worlds = plan.dof_count, 3
        rng = np.random.default_rng(923)
        lower_factor = np.zeros((dofs, dofs))
        lower_factor[plan.entry_rows, plan.columns] = rng.normal(scale=0.02, size=plan.nonzero_count)
        lower_factor[np.diag_indices(dofs)] = 1.5
        inverse = np.linalg.inv(lower_factor)
        packed = np.stack([(1.0 + group * 0.1) * inverse[plan.entry_rows, plan.columns] for group in range(worlds)])
        group_to_art = np.array([2, 0, 1], dtype=np.int32)
        art_to_world = np.array([1, 2, 0], dtype=np.int32)
        start = np.arange(worlds, dtype=np.int32) * dofs
        q_index = np.arange(worlds * dofs, dtype=np.int32)
        lower = np.full((worlds, dofs), -1.0, dtype=np.float32)
        upper = np.full((worlds, dofs), 1.0, dtype=np.float32)
        position = np.zeros((worlds, dofs), dtype=np.float32)
        # Articulation zero has no active rows. The others cross warp and 64-bit mask boundaries.
        position[1, [0, 15, 32]] = -1.0
        position[1, [16, 47]] = 1.0
        lower[1, [31, 63]] = upper[1, [31, 63]] = 0.0
        lower[2] = upper[2] = 0.0
        q_index[2 * dofs + 5] = -1
        lower[2, 7] = -np.inf
        upper[2, 8] = np.inf
        lower[2, 9] = np.nan
        velocity = rng.normal(size=worlds * dofs).astype(np.float32)
        initial_count = np.array([1, 2, 3], dtype=np.int32)

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def full(shape, value, dtype=float):
            return wp.full(shape, value, dtype=dtype, device=device)

        indices = plan.to_device(device)
        counter = array(initial_count, int)
        row_type, parent = [full((worlds, capacity), -9, int) for _ in range(2)]
        mu, phi, target, incident, diagonal = [full((worlds, capacity), -9.0) for _ in range(5)]
        row_dof = full((worlds, capacity, 34), -9, int)
        row_factor = full((worlds, capacity, 34), -9.0)
        outputs = [counter, row_type, parent, mu, phi, target, row_dof, row_factor, incident, diagonal]
        wp.launch(
            build_sparse_joint_limit_rows,
            dim=worlds * 32,
            inputs=[
                array(group_to_art, int),
                array(art_to_world, int),
                array([0, 0, 0], int),
                array(start, int),
                array(q_index, int),
                array(lower.reshape(-1)),
                array(upper.reshape(-1)),
                array(position.reshape(-1)),
                0.01,
                indices.ancestor_mask,
                indices.inverse_permutation,
                indices.lookup,
                array(packed),
                array(velocity),
            ],
            outputs=[*outputs, ROW_SLOTS_ATOMIC, wp.zeros(1, dtype=wp.int32, device=device)],
            block_dim=128,
            device=device,
        )
        actual = [value.numpy() for value in outputs]
        for group, art in enumerate(group_to_art):
            world = art_to_world[art]
            first = initial_count[world]
            expected = []
            for local_dof in range(dofs):
                if q_index[start[art] + local_dof] < 0:
                    continue
                for side in (0, 1):
                    bound = (lower if side == 0 else upper)[art, local_dof]
                    q = position[art, local_dof]
                    active = q <= bound + 0.01 if side == 0 else q >= bound - 0.01
                    if np.isfinite(bound) and active:
                        expected.append((local_dof, 1.0 if side == 0 else -1.0, q - bound))
            self.assertEqual(actual[0][world], first + len(expected))
            for output in actual[1:]:
                np.testing.assert_array_equal(output[world, :first], -9)
                np.testing.assert_array_equal(output[world, first + len(expected) :], -9)
            for row, (local_dof, sign, gap) in enumerate(expected[: capacity - first], first):
                support = np.flatnonzero([int(plan.ancestor_mask[local_dof]) & (1 << bit) for bit in range(dofs)])
                np.testing.assert_array_equal(actual[6][world, row, : len(support)], support)
                np.testing.assert_array_equal(actual[6][world, row, len(support) :], -1)
                np.testing.assert_array_equal(actual[7][world, row, len(support) :], 0.0)
                expected_factor = sign * (1.0 + group * 0.1) * inverse[support, plan.inverse_permutation[local_dof]]
                np.testing.assert_allclose(actual[7][world, row, : len(support)], expected_factor, atol=1.0e-7)
                self.assertEqual(actual[1][world, row], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
                self.assertEqual(actual[2][world, row], -1)
                self.assertEqual(actual[3][world, row], 0.0)
                self.assertAlmostEqual(actual[4][world, row], sign * gap, places=7)
                self.assertEqual(actual[5][world, row], 0.0)
                self.assertAlmostEqual(actual[8][world, row], sign * velocity[start[art] + local_dof], places=7)
                self.assertAlmostEqual(actual[9][world, row], expected_factor @ expected_factor, places=6)

    def test_joint_limit_parallel_scan_cpu(self):
        """Retain ordered rows, full overflow demand and zero-row worlds across 64 DOFs."""
        for capacity in (8, 140):
            with self.subTest(capacity=capacity):
                self._check_joint_limit_parallel_scan("cpu", capacity)

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative limit construction requires CUDA")
    def test_joint_limit_parallel_scan_cuda(self):
        """Match cooperative ballot ordering and responses to the scalar CPU oracle."""
        for capacity in (8, 140):
            with self.subTest(capacity=capacity):
                self._check_joint_limit_parallel_scan("cuda:0", capacity)

    def test_decode_factor_velocity(self):
        """Decode accumulated factor impulses into the original physical DOF order."""
        f = _fixture()
        delta = np.arange(6, dtype=np.float32) * 0.17 - 0.4
        out = _full(6, -9.0)
        wp.launch(
            apply_sparse_factor_velocity,
            dim=(1, 6),
            inputs=[
                f["group"],
                _array([0], int),
                _array([0], int),
                f["start"],
                f["indices"].permutation,
                f["indices"].lookup,
                f["packed"],
                _array(delta[None]),
                f["velocity"],
            ],
            outputs=[out],
            device="cpu",
        )
        expected = f["velocity"].numpy().copy()
        expected[f["plan"].permutation] += f["inverse"].T @ delta
        np.testing.assert_allclose(out.numpy(), expected, rtol=2.0e-6, atol=2.0e-7)


_PAIR_SOLVER = {
    "solve_mode": "matrix_free",
    "articulation_constraint_max": 64,
    "free_body_constraint_max": 32,
    "iterations": 8,
    "enable_joint_limits": True,
}


def _build_sparse_diagonal_pair_model(
    num_branches=16, num_worlds=2, *, device=None, revolute_branches=False, with_free_body=False
):
    """Build one independent fixed-base star and one compact serial chain per world.

    ``revolute_branches`` hinges the star branches about Y instead of sliding them along Z, so the branch
    response depends on the link inertia and center of mass rather than on the mass alone.
    """
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    star_joints = [scene.add_joint_fixed(parent=-1, child=base)]
    add_branch_joint = scene.add_joint_revolute if revolute_branches else scene.add_joint_prismatic
    for branch in range(num_branches):
        child = scene.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        star_joints.append(
            add_branch_joint(
                parent=base,
                child=child,
                axis=newton.Axis.Y if revolute_branches else newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.03 * branch, 0.0, 0.0), wp.quat_identity()),
                limit_lower=-0.1,
                limit_upper=0.1,
            )
        )
    scene.add_articulation(star_joints)

    chain_joints = []
    parent = -1
    for _link_index in range(3):
        child = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain_joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            )
        )
        parent = child
    scene.add_articulation(chain_joints)
    if with_free_body:
        body = scene.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
        scene.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)

    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(scene, num_worlds, spacing=(3.0, 3.0, 0.0))
    return replicated.finalize(device=device)


def test_sparse_diagonal_gs_matches_scalar_reference(test, device):
    """Match a scalar PGS reference with coupled limits and friction."""
    device = wp.get_device(device)
    max_constraints, world_dofs, dense_dofs = 8, 4, 2
    cfm, omega, iterations = 1.0e-6, 1.0, 8
    inverse_mass = np.array((0.5, 0.25, 0.75, 1.0), dtype=np.float32)
    initial_velocity = np.array((-0.4, 0.3, -0.2, 0.1), dtype=np.float32)
    jacobian = np.array(
        ((0.6, -0.4, 0.2, 0.0), (0.1, 0.5, -0.3, 0.4), (-0.2, 0.25, 0.1, -0.35)),
        dtype=np.float32,
    )
    response = jacobian * inverse_mass
    rhs = np.zeros((1, max_constraints), dtype=np.float32)
    rhs[0, :3] = (-0.1, 0.0, 0.0)
    diag = np.zeros_like(rhs)
    diag[0, :3] = np.sum(jacobian * response, axis=1) + cfm
    row_type = np.zeros((1, max_constraints), dtype=np.int32)
    row_type[0, :3] = (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION)
    row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
    row_parent[0, 1:3] = 0
    row_mu = np.zeros((1, max_constraints), dtype=np.float32)
    row_mu[0, 1:3] = 0.7
    dense_j = np.zeros((1, max_constraints, dense_dofs), dtype=np.float32)
    dense_y = np.zeros_like(dense_j)
    dense_j[0, :3] = jacobian[:, :dense_dofs]
    dense_y[0, :3] = response[:, :dense_dofs]
    sparse_dof = np.full((1, max_constraints, 2), -1, dtype=np.int32)
    sparse_jy = np.zeros((1, max_constraints, 4), dtype=np.float32)
    sparse_dof[0, :3] = (2, 3)
    sparse_jy[0, :3, 0] = jacobian[:, 2]
    sparse_jy[0, :3, 1] = response[:, 2]
    sparse_jy[0, :3, 2] = jacobian[:, 3]
    sparse_jy[0, :3, 3] = response[:, 3]
    limit_active = np.zeros((1, world_dofs), dtype=np.int32)
    limit_active[0, 2] = 1
    limit_lower_rhs = np.zeros((1, world_dofs), dtype=np.float32)
    limit_lower_rhs[0, 2] = -0.15

    expected_velocity = initial_velocity.copy()
    expected_impulses = np.zeros(3, dtype=np.float32)
    expected_limit_lambda = np.float32(0.0)
    for _ in range(iterations):
        old_limit = expected_limit_lambda
        residual = expected_velocity[2] + limit_lower_rhs[0, 2]
        expected_limit_lambda = np.maximum(0.0, old_limit - omega * residual / (inverse_mass[2] + cfm))
        expected_velocity[2] += inverse_mass[2] * (expected_limit_lambda - old_limit)
        old_impulse = expected_impulses[0]
        new_impulse = old_impulse - omega * (jacobian[0] @ expected_velocity + rhs[0, 0]) / diag[0, 0]
        new_impulse = max(new_impulse, 0.0)
        expected_impulses[0] = new_impulse
        expected_velocity += response[0] * (new_impulse - old_impulse)
        # The general owner solves both tangent rows together on the friction disk at the current normal
        # load (friction_pair_candidate); the second tangent visit is a no-op.
        radius = max(row_mu[0, 1] * expected_impulses[0], 0.0)
        pair = friction_pair_candidate(
            float(diag[0, 1]),
            float(jacobian[1] @ response[2]),
            float(diag[0, 2]),
            wp.vec2(
                float(jacobian[1] @ expected_velocity + rhs[0, 1]),
                float(jacobian[2] @ expected_velocity + rhs[0, 2]),
            ),
            wp.vec2(float(expected_impulses[1]), float(expected_impulses[2])),
            float(radius),
            omega,
        )
        magnitude = np.sqrt(pair[0] * pair[0] + pair[1] * pair[1])
        scale = radius / magnitude if magnitude > radius else 1.0
        for tangent, value in ((1, pair[0] * scale), (2, pair[1] * scale)):
            expected_velocity += response[tangent] * (value - expected_impulses[tangent])
            expected_impulses[tangent] = np.float32(value)

    impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    lower_lambda = wp.zeros((1, world_dofs), dtype=wp.float32, device=device)
    upper_lambda = wp.zeros_like(lower_lambda)
    velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
    kernel = _get_pgs_solve_sparse_diagonal_kernel(
        max_constraints, world_dofs, dense_dofs, str(device.arch), contact_triples=True
    )
    wp.launch_tiled(
        kernel,
        dim=[1],
        inputs=[
            wp.array((3,), dtype=wp.int32, device=device),
            wp.array(np.arange(world_dofs, dtype=np.int32)[None, :], device=device),
            wp.array(rhs, device=device),
            wp.array(diag, device=device),
            impulses,
            wp.array(row_type, device=device),
            wp.array(row_parent, device=device),
            wp.array(row_mu, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.empty((1, world_dofs), dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.empty((1, (max_constraints + 2) // 3), dtype=wp.int32, device=device),
            wp.array((0,), dtype=wp.int32, device=device),
            wp.array((0,), dtype=wp.int32, device=device),
            wp.array(dense_j, device=device),
            wp.array(dense_y, device=device),
            wp.array(sparse_dof, device=device),
            wp.array(sparse_jy, device=device),
            wp.array(limit_active, device=device),
            wp.array(limit_lower_rhs, device=device),
            wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
            wp.array(inverse_mass, device=device),
            cfm,
            iterations,
            omega,
            0,
            0,
        ],
        outputs=[lower_lambda, upper_lambda, velocity],
        block_dim=32,
        device=device,
    )

    np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(impulses.numpy()[0, :3], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(lower_lambda.numpy()[0, 2], expected_limit_lambda, rtol=2.0e-5, atol=2.0e-6)
    test.assertEqual(float(upper_lambda.numpy()[0, 2]), 0.0)


def test_speculative_sparse_contact_batches_match_serial_reference(test, device):
    """Skip exact no-op prefixes without changing later serial contact updates."""
    device = wp.get_device(device)
    max_constraints, world_dofs, dense_dofs = 12, 8, 6
    cfm, omega, iterations = 1.0e-6, 1.0, 2
    inverse_mass = np.array((0.5, 0.25, 0.75, 1.0, 0.4, 0.6, 0.75, 0.8), dtype=np.float32)
    initial_velocity = np.array((0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.5, 0.25), dtype=np.float32)
    jacobian = np.zeros((max_constraints, world_dofs), dtype=np.float32)
    jacobian[0, 6] = 1.0
    jacobian[3, 7] = 1.0
    jacobian[6, (0, 6)] = (-0.2, -1.0)
    jacobian[9, (0, 6)] = (0.3, 1.0)
    response = jacobian * inverse_mass
    rhs = np.zeros((1, max_constraints), dtype=np.float32)
    rhs[0, 6] = -0.1
    rhs[0, 9] = -0.2
    diag = np.zeros_like(rhs)
    normal_rows = np.array((0, 3, 6, 9), dtype=np.int32)
    diag[0, normal_rows] = np.sum(jacobian[normal_rows] * response[normal_rows], axis=1) + cfm
    row_type = np.full((1, max_constraints), PGS_CONSTRAINT_TYPE_FRICTION, dtype=np.int32)
    row_type[0, normal_rows] = PGS_CONSTRAINT_TYPE_CONTACT
    row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
    for normal in normal_rows:
        row_parent[0, normal + 1 : normal + 3] = normal
    row_mu = np.zeros((1, max_constraints), dtype=np.float32)
    dense_j = jacobian[:, :dense_dofs][None, ...]
    dense_y = response[:, :dense_dofs][None, ...]
    sparse_dof = np.empty((1, max_constraints, 2), dtype=np.int32)
    sparse_dof[:, :, 0] = 6
    sparse_dof[:, :, 1] = 7
    sparse_jy = np.zeros((1, max_constraints, 4), dtype=np.float32)
    sparse_jy[0, :, 0] = jacobian[:, 6]
    sparse_jy[0, :, 1] = response[:, 6]
    sparse_jy[0, :, 2] = jacobian[:, 7]
    sparse_jy[0, :, 3] = response[:, 7]

    expected_velocity = initial_velocity.copy()
    expected_impulses = np.zeros(max_constraints, dtype=np.float32)
    for _ in range(iterations):
        for row in normal_rows:
            old_impulse = expected_impulses[row]
            residual = jacobian[row] @ expected_velocity + rhs[0, row]
            new_impulse = max(old_impulse - omega * residual / diag[0, row], 0.0)
            expected_impulses[row] = new_impulse
            expected_velocity += response[row] * (new_impulse - old_impulse)

    impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
    kernel = _get_pgs_solve_sparse_diagonal_kernel(
        max_constraints,
        world_dofs,
        dense_dofs,
        str(device.arch),
        contact_triples=True,
        speculative_contact_batches=True,
    )
    wp.launch_tiled(
        kernel,
        dim=[1],
        inputs=[
            wp.array((max_constraints,), dtype=wp.int32, device=device),
            wp.array(np.arange(world_dofs, dtype=np.int32)[None, :], device=device),
            wp.array(rhs, device=device),
            wp.array(diag, device=device),
            impulses,
            wp.array(row_type, device=device),
            wp.array(row_parent, device=device),
            wp.array(row_mu, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.empty((1, world_dofs), dtype=wp.int32, device=device),
            wp.array((len(normal_rows),), dtype=wp.int32, device=device),
            wp.array(normal_rows[None, :], device=device),
            wp.array((0,), dtype=wp.int32, device=device),
            wp.array((0,), dtype=wp.int32, device=device),
            wp.array(dense_j, device=device),
            wp.array(dense_y, device=device),
            wp.array(sparse_dof, device=device),
            wp.array(sparse_jy, device=device),
            wp.zeros((1, world_dofs), dtype=wp.int32, device=device),
            wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
            wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
            wp.array(inverse_mass, device=device),
            cfm,
            iterations,
            omega,
            0,
            0,
        ],
        outputs=[
            wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
            wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
            velocity,
        ],
        block_dim=32,
        device=device,
    )

    np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(impulses.numpy()[0], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_array_equal(impulses.numpy()[0, :6], np.zeros(6, dtype=np.float32))


def test_independent_sparse_contact_groups_exclude_coupled_coordinates(test, device):
    """Keep scalar contacts serial when a coupled contact shares their coordinate."""
    device = wp.get_device(device)
    max_constraints, max_world_dofs, contact_count = 12, 4, 4
    sparse_response_dofs = 108
    sparse_row_dof_np = np.full((1, max_constraints, 2), -1, dtype=np.int32)
    sparse_row_dof_np[0, 0:3, 0] = 2
    sparse_row_dof_np[0, 3:6, 0] = 3
    sparse_row_dof_np[0, 6:9, 0] = 3
    sparse_row_dof_np[0, 9:12, 0] = 2
    sparse_row_dof = wp.array(sparse_row_dof_np, device=device)
    group_count = wp.zeros(1, dtype=wp.int32, device=device)
    group_heads = wp.empty((1, max_world_dofs), dtype=wp.int32, device=device)
    serial_count = wp.zeros(1, dtype=wp.int32, device=device)
    serial_normals = wp.empty((1, (max_constraints + 2) // 3), dtype=wp.int32, device=device)

    marker = _get_mark_independent_sparse_contact_candidates_kernel(sparse_response_dofs, str(device.arch))
    wp.launch(
        marker,
        dim=contact_count,
        inputs=[
            wp.array((contact_count,), dtype=wp.int32, device=device),
            contact_count,
            wp.zeros(contact_count, dtype=wp.int32, device=device),
            wp.array((0, 3, 6, 9), dtype=wp.int32, device=device),
            wp.zeros(contact_count, dtype=wp.int32, device=device),
            wp.array((-1, 1, -1, -1), dtype=wp.int32, device=device),
            wp.zeros(contact_count, dtype=wp.int32, device=device),
            wp.full(contact_count, 3, dtype=wp.int32, device=device),
            wp.array((sparse_response_dofs, 6), dtype=wp.int32, device=device),
        ],
        outputs=[sparse_row_dof],
        device=device,
    )
    schedule = _get_build_independent_sparse_contact_groups_kernel(
        max_constraints, max_world_dofs, str(device.arch), build_serial_contacts=True
    )
    wp.launch(
        schedule,
        dim=32,
        inputs=[
            wp.array((max_constraints,), dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
        ],
        outputs=[sparse_row_dof, group_count, group_heads, serial_count, serial_normals],
        device=device,
    )
    wp.synchronize_device(device)

    test.assertEqual(int(group_count.numpy()[0]), 1)
    test.assertEqual(int(group_heads.numpy()[0, 0]), 0)
    test.assertEqual(int(serial_count.numpy()[0]), 2)
    np.testing.assert_array_equal(serial_normals.numpy()[0, :2], np.array((3, 6), dtype=np.int32))
    normal_links = sparse_row_dof.numpy()[0, ::3, 1]
    np.testing.assert_array_equal(normal_links, np.array((-12, -1, -1, -2), dtype=np.int32))


def test_sparse_diagonal_response_matches_dense_joint_limits(test, device):
    """Match dense position-limit rows while removing the diagonal articulation's dense response storage."""
    model = _build_sparse_diagonal_pair_model(device=device)
    # Point friction keeps the contact-triple schedule under test; patch rows are not uniform triples.
    optimized = SolverSteel(model, use_parallel_streams=True, friction_anchor_beta=0.0, **_PAIR_SOLVER)
    reference = SolverSteel(model, friction_anchor_beta=0.0, **_PAIR_SOLVER)
    test.assertTrue(optimized._sparse_diagonal_contact_solve)
    test.assertTrue(optimized._sparse_diagonal_contact_triples)
    test.assertTrue(optimized._sparse_diagonal_speculative_contact_batches)
    test.assertEqual(optimized._sparse_diagonal_response_size, 16)
    test.assertEqual(optimized._sparse_diagonal_dense_size, 3)
    test.assertEqual(optimized._H_by_size[16].shape, (1, 1, 1))
    test.assertEqual(optimized._J_by_size[16].shape, (1, 1, 1))
    test.assertIsNone(optimized._pgs_solve_mf_gs_kernel)
    test.assertFalse(reference._sparse_diagonal_contact_solve)

    # Alternate branches start past the lower and the upper limit, moving further out.
    side = np.where(np.arange(16) % 2 == 0, -1.0, 1.0)
    q = np.tile(np.concatenate((0.105 * side, np.full(3, 0.205))).astype(np.float32), 2)
    qd = np.tile(np.concatenate((0.2 * side, np.full(3, 0.1))).astype(np.float32), 2)
    trajectories = []
    for solver in (optimized, reference):
        state_in, state_out = model.state(), model.state()
        state_in.joint_q.assign(q)
        state_in.joint_qd.assign(qd)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        control = model.control()
        history = []
        for _ in range(3):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, None, 1.0 / 120.0)
            state_in, state_out = state_out, state_in
            history.append((state_in.joint_q.numpy().copy(), state_in.joint_qd.numpy().copy()))
        trajectories.append(history)

    first_q, first_qd = trajectories[0][0]
    # Each branch recovers baumgarte of its violation in one step, on both sides.
    np.testing.assert_allclose(first_q.reshape(2, 19)[:, :16], np.tile(0.104 * side, (2, 1)), rtol=0.0, atol=2.0e-6)
    np.testing.assert_allclose(first_qd.reshape(2, 19)[:, :16], np.tile(-0.12 * side, (2, 1)), rtol=0.0, atol=2.0e-6)
    for sparse, dense in zip(*trajectories, strict=True):
        np.testing.assert_allclose(sparse[0], dense[0], rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(sparse[1], dense[1], rtol=2.0e-5, atol=2.0e-6)


def test_sparse_diagonal_contact_friction_matches_general_owner(test, device):
    """Match the general sweep's paired friction on sparse contacts with patch and point friction."""
    model = _build_sparse_contact_friction_model(device=device)

    def make_solver(**kwargs):
        options = dict(_PAIR_SOLVER, articulation_constraint_max=96)
        options.update(kwargs)
        return SolverSteel(model, **options)

    def run(solver):
        state_in, state_out = model.state(), model.state()
        joint_qd = state_in.joint_qd.numpy()
        for art in np.flatnonzero(solver._model_plan.response_dof_count == 16):
            start = int(solver._model_plan.articulation_dof_start[art])
            joint_qd[start : start + 16] = -0.3
        state_in.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
        contacts = pipeline.contacts()
        control = model.control()
        history = []
        for _ in range(4):
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
            history.append((state_in.joint_q.numpy().copy(), state_in.joint_qd.numpy().copy()))
        return history

    # Normal rows only: the frictionless reference.
    slick = run(make_solver(use_parallel_streams=True, contact_friction_gap_max=-float("inf")))
    for label, friction_kwargs, triples in (("patch", {}, False), ("point", {"friction_anchor_beta": 0.0}, True)):
        with test.subTest(friction=label):
            optimized = make_solver(use_parallel_streams=True, **friction_kwargs)
            reference = make_solver(**friction_kwargs)
            test.assertTrue(optimized._sparse_diagonal_contact_solve)
            test.assertEqual(optimized._friction_anchors_enabled, label == "patch")
            test.assertEqual(optimized._sparse_diagonal_contact_triples, triples)
            test.assertEqual(optimized._sparse_diagonal_speculative_contact_batches, triples)
            test.assertFalse(reference._sparse_diagonal_contact_solve)
            sparse = run(optimized)
            general = run(reference)
            test.assertGreater(int(optimized._constraint_count.numpy().max()), 0, "no contact rows were generated")
            test.assertGreater(
                float(np.abs(sparse[-1][1] - slick[-1][1]).max()), 1.0e-3, "friction did not change the sparse response"
            )
            for step, ((sparse_q, sparse_qd), (general_q, general_qd)) in enumerate(zip(sparse, general, strict=True)):
                np.testing.assert_allclose(
                    sparse_q, general_q, rtol=2.0e-5, atol=2.0e-6, err_msg=f"joint_q step {step}"
                )
                np.testing.assert_allclose(
                    sparse_qd, general_qd, rtol=2.0e-5, atol=2.0e-6, err_msg=f"joint_qd step {step}"
                )


def test_sparse_diagonal_contact_compliance_matches_general_owner(test, device):
    """Solve compliant contacts on the sparse rows like the general sweep."""
    model = _build_sparse_contact_friction_model(device=device)
    results = {}
    for label, streams, compliance in (("sparse", True, True), ("general", False, True), ("rigid", False, False)):
        solver = SolverSteel(
            model,
            **dict(_PAIR_SOLVER, articulation_constraint_max=96, iterations=16),
            use_parallel_streams=streams,
            friction_anchor_beta=0.0,
            contact_compliance=compliance,
        )
        state_in, state_out = model.state(), model.state()
        joint_qd = state_in.joint_qd.numpy()
        for art in np.flatnonzero(solver._model_plan.response_dof_count == 16):
            start = int(solver._model_plan.articulation_dof_start[art])
            joint_qd[start : start + 16] = -0.3
        state_in.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=256)
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(256, dtype=float, device=device))
        compliant = 0
        for _ in range(40):
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            contacts.rigid_contact_stiffness.fill_(3000.0)
            contacts.rigid_contact_damping.fill_(20.0)
            contacts.rigid_contact_friction.fill_(1.0)
            solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
            compliant = max(compliant, solver.compliance_contact_count)
        results[label] = (solver, compliant, state_in.joint_qd.numpy())
    sparse, compliant, joint_qd = results["sparse"]
    test.assertTrue(sparse._sparse_diagonal_contact_solve)
    test.assertFalse(results["general"][0]._sparse_diagonal_contact_solve)
    test.assertGreater(compliant, 0)
    test.assertGreater(float(np.abs(results["rigid"][2] - results["general"][2]).max()), 1.0e-3)
    np.testing.assert_allclose(joint_qd, results["general"][2], rtol=0.0, atol=1.0e-6)


def test_sparse_diagonal_is_selected_only_where_supported(test, device):
    """Keep the general sweep without streams, on CPU, with a free body or with options it does not solve."""
    model = _build_sparse_diagonal_pair_model(device=device)
    if not wp.get_device(device).is_cuda:
        solver = SolverSteel(
            model,
            solve_mode="split",
            enable_joint_limits=True,
            articulation_constraint_max=64,
            use_parallel_streams=True,
        )
        test.assertFalse(solver._sparse_diagonal_contact_solve)
        return
    options = dict(_PAIR_SOLVER, use_parallel_streams=True)
    selected = SolverSteel(model, **options)
    test.assertTrue(selected._sparse_diagonal_contact_solve)
    test.assertTrue(selected._fused_diagonal_joint_limit_sizes)
    for label, overrides in (
        ("no streams", {"use_parallel_streams": False}),
        ("joint friction", {"enable_joint_friction": True}),
        ("joint limits off", {"enable_joint_limits": False}),
        ("regularization", {"contact_regularization": 0.02}),
        ("velocity iterations", {"velocity_iterations": 1}),
        ("warm start", {"contact_warmstart": True}),
        ("schedule", {"row_schedule": "phased"}),
        ("drive rows", {"drive_mode": "rows"}),
        ("friction mode", {"friction_mode": "bisection", "friction_anchor_beta": 0.0}),
    ):
        with test.subTest(label):
            solver = SolverSteel(model, **dict(options, **overrides))
            test.assertFalse(solver._sparse_diagonal_contact_solve)
            test.assertFalse(solver._sparse_diagonal_contact_triples)
            test.assertFalse(solver._fused_diagonal_joint_limit_sizes)
    with_free_body = _build_sparse_diagonal_pair_model(device=device, with_free_body=True)
    test.assertFalse(SolverSteel(with_free_body, **options)._sparse_diagonal_contact_solve)


class TestSteelSparseDiagonal(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()


for _fn in (
    test_sparse_diagonal_gs_matches_scalar_reference,
    test_speculative_sparse_contact_batches_match_serial_reference,
    test_independent_sparse_contact_groups_exclude_coupled_coordinates,
    test_sparse_diagonal_response_matches_dense_joint_limits,
    test_sparse_diagonal_contact_friction_matches_general_owner,
    test_sparse_diagonal_contact_compliance_matches_general_owner,
):
    add_function_test(TestSteelSparseDiagonal, _fn.__name__, _fn, devices=cuda_devices)


add_function_test(
    TestSteelSparseDiagonal,
    "test_sparse_diagonal_is_selected_only_where_supported",
    test_sparse_diagonal_is_selected_only_where_supported,
    devices=get_test_devices(),
)


def _branched_humanoid_topology():
    """Return 43 DOFs with a free base, legs, spine, arms, and branching fingers."""
    parents, counts = [-1], [6]

    def chain(parent, length):
        for _ in range(length):
            parents.append(parent)
            counts.append(1)
            parent = len(parents) - 1
        return parent

    for _ in range(2):
        chain(0, 6)
    spine = chain(0, 3)
    for _ in range(2):
        hand = chain(spine, 7)
        for _ in range(4):
            chain(hand, 1)
    return parents, counts


class TestSparseMassMatrix(unittest.TestCase):
    def test_factor_schedules_follow_dependencies(self):
        """Cover each packed factor entry once and schedule independent branch pivots together."""
        for parents, counts in [([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2]), _branched_humanoid_topology()]:
            plan = _SparseMassMatrixPlan.build(parents, counts)
            levels = np.empty(plan.dof_count, dtype=int)
            for level in range(plan.factor_level_count):
                begin, end = plan.factor_level_offsets[level : level + 2]
                levels[plan.factor_level_columns[begin:end]] = level
            np.testing.assert_array_equal(np.sort(plan.factor_level_columns), np.arange(plan.dof_count))
            np.testing.assert_array_equal(np.sort(plan.column_entries), np.arange(plan.nonzero_count))
            np.testing.assert_array_equal(
                np.sort(plan.factor_level_entries), np.flatnonzero(plan.entry_rows != plan.columns)
            )
            for row in range(plan.dof_count):
                begin, end = plan.row_offsets[row : row + 2]
                first = row - (end - begin) + 1
                np.testing.assert_array_equal(plan.columns[begin:end], np.arange(first, row + 1))
                self.assertTrue(np.all(levels[first:row] < levels[row]))
                begin, end = plan.column_offsets[row : row + 2]
                entries = plan.column_entries[begin:end]
                np.testing.assert_array_equal(plan.columns[entries], row)
                self.assertTrue(np.all(np.diff(plan.entry_rows[entries]) > 0))
            if plan.dof_count == 43:
                self.assertLess(plan.factor_level_count, plan.dof_count // 2)

    def test_forest_fixed_multidof_order(self):
        """Derive fill-free support for unordered forests and fixed joints."""
        plan = _SparseMassMatrixPlan.build([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2])
        np.testing.assert_array_equal(np.sort(plan.permutation), np.arange(9))
        np.testing.assert_array_equal(plan.inverse_permutation[plan.permutation], np.arange(9))
        self.assertEqual(plan.endpoint_support(2).size, 3)
        self.assertEqual(plan.endpoint_support(0, 5).size, 8)
        for joint in range(6):
            support = plan.endpoint_support(joint)
            mask = sum(1 << int(i) for i in support)
            self.assertEqual(int(plan.joint_ancestor_mask[joint]), mask)
        rng = np.random.default_rng(54)
        lower = np.zeros((9, 9))
        lower[plan.entry_rows, plan.columns] = rng.normal(size=plan.nonzero_count)
        lower[np.diag_indices(9)] = 2.0
        inverse = np.linalg.inv(lower)
        np.testing.assert_allclose(inverse[plan.lookup < 0], 0.0, atol=1.0e-14)
        matrix = lower @ lower.T
        structural = (plan.lookup >= 0) | (plan.lookup.T >= 0)
        np.testing.assert_allclose(matrix[~structural], 0.0, atol=1.0e-14)

    def test_invalid_topology(self):
        """Reject cycles, invalid parents, counts, and unsupported mask widths."""
        for parents, counts in [
            ([1, 0], [1, 1]),
            ([2], [1]),
            ([-2], [1]),
            ([-1], [-1]),
            ([-1], [1.5]),
            ([-1], [65]),
            ([-1, 0], [40, 40]),
            ([-1], [0]),
            ([-1], []),
            ([], []),
        ]:
            with self.subTest(parents=parents, counts=counts), self.assertRaises(ValueError):
                _SparseMassMatrixPlan.build(parents, counts)

    def _factor(self, parents, counts, *, fused_drive=True, mask=None, armature=0.4, device="cpu"):
        plan = _SparseMassMatrixPlan.build(parents, counts)
        n, j = plan.dof_count, len(parents)
        rng = np.random.default_rng(742)
        motion = rng.normal(size=(2 * n, 6)).astype(np.float32)
        body_inertia = np.empty((2 * j, 6, 6), dtype=np.float32)
        composite = np.zeros_like(body_inertia)
        dense = np.zeros((2, n, n))
        for art in range(2):
            for joint in range(j):
                q = rng.normal(size=(6, 6))
                body_inertia[art * j + joint] = q @ q.T + np.eye(6)
                physical_support = plan.permutation[plan.endpoint_support(joint)]
                jacobian = np.zeros((6, n))
                jacobian[:, physical_support] = motion[art * n + physical_support].T
                dense[art] += jacobian.T @ body_inertia[art * j + joint] @ jacobian
                ancestor = joint
                while ancestor != -1:
                    composite[art * j + ancestor] += body_inertia[art * j + joint]
                    ancestor = parents[ancestor]
        drive_map = np.arange(2 * n, dtype=np.int32)
        drive_map[::3] = -1
        stiffness = rng.uniform(-0.1, 0.5, size=2 * n).astype(np.float32)
        regularizer = np.full((2, n), armature, dtype=np.float32)
        mapping = np.array([1, 0], dtype=np.int32)
        for group, art in enumerate(mapping):
            diagonal = regularizer[group].astype(np.float64)
            if fused_drive:
                diagonal += np.where(
                    drive_map[art * n : (art + 1) * n] >= 0, np.maximum(stiffness[art * n : (art + 1) * n], 0.0), 0.0
                )
            dense[art] += np.diag(diagonal)

        # Implicit drive term per DOF; DOFs without a drive or with a nonpositive term add nothing.
        drive_dof_K = np.zeros(2 * n, dtype=np.float32)
        if fused_drive:
            drive_dof_K = np.where(drive_map >= 0, np.maximum(stiffness, 0.0), 0.0).astype(np.float32)

        def array(value, dtype):
            return wp.array(value, dtype=dtype, device=device)

        indices = plan.to_device(device)
        inverse = wp.full((2, plan.nonzero_count), -98.0, dtype=float, device=device)
        status = wp.full(2, 17, dtype=int, device=device)
        if mask is None:
            mask = [1, 1]
        joint_child = rng.permutation(2 * j).astype(np.int32)
        reordered_composite = np.empty_like(composite)
        reordered_composite[joint_child] = composite
        inputs = [
            array(mapping, int),
            array(mask, int),
            array([0, j, 2 * j], int),
            array([0, n, 2 * n], int),
            array(joint_child, int),
            array(motion, wp.spatial_vector),
            array(reordered_composite, wp.spatial_matrix),
            array(regularizer, float),
            array(drive_dof_K, float),
            indices,
        ]
        wp.launch(
            _get_crba_sparse_factor_kernel(n, plan.nonzero_count),
            dim=64,
            inputs=inputs,
            outputs=[inverse, status],
            device=device,
            block_dim=128,
        )
        return plan, dense, inverse, status, inputs

    def test_crba_factor_solve_and_endpoint_response(self):
        """Match dense dynamics and endpoint response for branched multi-DOF forests."""
        for parents, counts in [([-1, 0, 0, 1], [6, 1, 3, 0]), ([2, -1, 1, 2, -1, 4], [2, 3, 0, 1, 1, 2])]:
            for fused_drive in (False, True):
                with self.subTest(parents=parents, fused_drive=fused_drive):
                    plan, dense, inverse, status, inputs = self._factor(parents, counts, fused_drive=fused_drive)
                    np.testing.assert_array_equal(status.numpy(), [0, 0])
                    n = plan.dof_count
                    rng = np.random.default_rng(66)
                    tau = rng.normal(size=2 * n).astype(np.float32)
                    qdd = wp.zeros(2 * n, dtype=float, device="cpu")
                    scratch = wp.zeros((2, n), dtype=float, device="cpu")
                    wp.launch(
                        solve_sparse_mass_matrix,
                        dim=64,
                        inputs=[
                            inputs[0],
                            inputs[3],
                            inputs[-1],
                            inverse,
                            wp.array(tau, device="cpu"),
                            wp.ones(2, dtype=int, device="cpu"),
                            scratch,
                        ],
                        outputs=[qdd],
                        device="cpu",
                        block_dim=128,
                    )
                    for group, art in enumerate([1, 0]):
                        whiten = np.zeros((n, n))
                        whiten[plan.entry_rows, plan.columns] = inverse.numpy()[group]
                        reordered = dense[art][np.ix_(plan.permutation, plan.permutation)]
                        np.testing.assert_allclose(
                            whiten @ np.linalg.cholesky(reordered), np.eye(n), rtol=2.0e-5, atol=2.0e-5
                        )
                        np.testing.assert_allclose(
                            qdd.numpy()[art * n : (art + 1) * n],
                            np.linalg.solve(dense[art], tau[art * n : (art + 1) * n]),
                            rtol=1.0e-4,
                            atol=1.0e-5,
                        )
                        support = plan.endpoint_support(0, len(parents) - 1)
                        row = np.zeros(n)
                        row[plan.permutation[support]] = rng.normal(size=len(support))
                        z = whiten @ row[plan.permutation]
                        np.testing.assert_allclose(np.delete(z, support), 0.0, atol=1.0e-12)
                        np.testing.assert_allclose(
                            z @ z, row @ np.linalg.solve(dense[art], row), rtol=3.0e-5, atol=1.0e-6
                        )

    def test_mask_bit_63_and_fixed_root(self):
        """Preserve the highest mask bit and fixed-root endpoint support."""
        plan = _SparseMassMatrixPlan.build([-1, 0], [0, 64])
        self.assertEqual(plan.endpoint_support(0).size, 0)
        self.assertEqual(int(plan.joint_ancestor_mask[1]), (1 << 64) - 1)
        self.assertEqual(int(plan.ancestor_mask[0]), (1 << 64) - 1)
        with self.assertRaises(ValueError):
            plan.endpoint_support(2)

    def _check_branched_factor(self, device):
        """Compare a broad articulated tree with the independent dense oracle."""
        plan, dense, inverse, status, inputs = self._factor(*_branched_humanoid_topology(), armature=2.0, device=device)
        np.testing.assert_array_equal(status.numpy(), [0, 0])
        n = plan.dof_count
        tau = np.random.default_rng(521).normal(size=2 * n).astype(np.float32)
        qdd = wp.zeros(2 * n, device=device)
        scratch = wp.zeros((2, n), device=device)
        wp.launch(
            solve_sparse_mass_matrix,
            dim=64,
            inputs=[
                inputs[0],
                inputs[3],
                inputs[-1],
                inverse,
                wp.array(tau, device=device),
                wp.ones(2, dtype=int, device=device),
                scratch,
            ],
            outputs=[qdd],
            device=device,
            block_dim=128,
        )
        for group, art in enumerate([1, 0]):
            whiten = np.zeros((n, n))
            whiten[plan.entry_rows, plan.columns] = inverse.numpy()[group]
            expected = dense[art][np.ix_(plan.permutation, plan.permutation)]
            np.testing.assert_allclose(whiten @ np.linalg.cholesky(expected), np.eye(n), atol=1.0e-4, rtol=1.0e-4)
            np.testing.assert_allclose(
                qdd.numpy()[art * n : (art + 1) * n],
                np.linalg.solve(dense[art], tau[art * n : (art + 1) * n]),
                atol=2.0e-5,
                rtol=2.0e-4,
            )

    def test_branched_43_dof_factor_and_predictor(self):
        """Match dense dynamics for a broad 43-DOF articulated tree on CPU."""
        self._check_branched_factor("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse factor CUDA schedule requires CUDA")
    def test_branched_43_dof_factor_and_predictor_cuda(self):
        """Match the dense oracle with parallel factor dependencies and inverse columns."""
        self._check_branched_factor("cuda:0")

    def test_mask_preserves_factor(self):
        """Retain the inverse factor and status when an articulation is not refreshed."""
        _, _, inverse, status, _ = self._factor([-1, 0, 0], [2, 1, 1], mask=[0, 1])
        np.testing.assert_array_equal(inverse.numpy()[1], -98.0)
        np.testing.assert_array_equal(status.numpy(), [0, 17])

    def test_invalid_pivot_is_not_regularized(self):
        """Report an invalid factor without adding a new pivot floor."""
        _, _, inverse, status, _ = self._factor([-1], [1], armature=-1.0e6)
        np.testing.assert_array_equal(status.numpy(), [1, 1])
        self.assertTrue(np.isnan(inverse.numpy()).all())


_SWEEP_CASES = (1.0, 0.7, 1.2)


def _friction_pair_reference(a, c, d, residual, old, radius, omega):
    """Host reference of the coupled tangent solve in ``FRICTION_PAIR_CUDA``."""
    if radius <= 0.0:
        return np.zeros(2)
    scale = max(a, d, 1.0e-20)
    a, c, d = a / scale, c / scale, d / scale
    largest = 0.5 * (a + d + np.sqrt((a - d) ** 2 + 4.0 * c * c))
    smallest = max((a * d - c * c) / largest, 0.0) if largest > 0.0 else 0.0
    vx, vy = (largest - d, c) if a >= d else (c, largest - a)
    norm = np.hypot(vx, vy)
    vx, vy = (vx / norm, vy / norm) if norm > 0.0 else (1.0, 0.0)
    r0, r1 = residual
    old0, old1 = old
    r0_rotated = (vx * r0 + vy * r1) / scale
    r1_rotated = (-vy * r0 + vx * r1) / scale
    result0, result1 = old0, old1
    sticking = False
    if largest > 0.0 and (smallest > 0.0 or r1_rotated == 0.0):
        dx = r0_rotated / largest
        dy = r1_rotated / smallest if smallest > 0.0 else 0.0
        result0 = old0 - (vx * dx - vy * dy)
        result1 = old1 - (vy * dx + vx * dy)
        sticking = np.hypot(result0, result1) <= radius
    if not sticking:
        old_norm = np.hypot(old0, old1)
        residual_norm = np.hypot(r0, r1)
        if (
            0.0 < old_norm <= radius
            and radius - old_norm <= 2.0e-7 * radius
            and r0 * old0 + r1 * old1 <= 0.0
            and abs(r0 * old1 - r1 * old0) <= 2.0e-7 * old_norm * residual_norm
        ):
            return np.array((old0, old1))
        old_x = vx * old0 + vy * old1
        old_y = -vy * old0 + vx * old1
        b0 = largest * old_x - r0_rotated
        b1 = smallest * old_y - r1_rotated
        x = y = 0.0
        hi = np.hypot(b0, b1) / radius
        if hi > 0.0:
            lo = max(hi - largest, abs(b1) / radius - smallest, 0.0)
            alpha = lo
            converged = False
            for _ in range(8):
                inv0 = 1.0 / (largest + alpha)
                inv1 = 1.0 / (smallest + alpha) if smallest + alpha > 0.0 else 0.0
                tx = old_x - (r0_rotated + alpha * old_x) * inv0
                ty = old_y - (r1_rotated + alpha * old_y) * inv1
                norm = np.hypot(tx, ty)
                if abs(norm - radius) <= 2.0e-7 * radius or (alpha == 0.0 and norm <= radius):
                    x, y, converged = tx, ty, True
                    break
                if norm > radius:
                    lo = alpha
                else:
                    hi = alpha
                slope = tx * tx * inv0 + ty * ty * inv1
                candidate = alpha + (norm / radius - 1.0) * norm * norm / slope if slope > 0.0 else alpha
                alpha = candidate if lo < candidate < hi else 0.5 * (lo + hi)
            if not converged:
                for _ in range(24):
                    alpha = 0.5 * (lo + hi)
                    if np.hypot(b0 / (largest + alpha), b1 / (smallest + alpha)) > radius:
                        lo = alpha
                    else:
                        hi = alpha
                x = b0 / (largest + hi)
                y = b1 / (smallest + hi)
        result0 = old0 + vx * (x - old_x) - vy * (y - old_y)
        result1 = old1 + vy * (x - old_x) + vx * (y - old_y)
    return np.array((old0 + omega * (result0 - old0), old1 + omega * (result1 - old1)))


def _sweep(
    jacobian,
    response,
    velocity,
    bias,
    diagonal,
    types,
    parents,
    mu,
    impulses,
    *,
    omega=1.0,
):
    velocity, impulses = velocity.copy(), impulses.copy()
    for _iteration in range(8):
        changed = False
        for row, kind in enumerate(types):
            if kind == PGS_CONSTRAINT_TYPE_FRICTION and row != parents[row] + 1:
                continue
            residual = jacobian[row] @ velocity + bias[row]
            old = impulses[row]
            if kind == PGS_CONSTRAINT_TYPE_FRICTION:
                parent, sibling = parents[row], row + 1
                radius = max(mu[row] * impulses[parent], 0.0)
                pair = _friction_pair_reference(
                    float(diagonal[row]),
                    float(jacobian[row] @ response[sibling]),
                    float(diagonal[sibling]),
                    (float(residual), float(jacobian[sibling] @ velocity + bias[sibling])),
                    (float(old), float(impulses[sibling])),
                    float(radius),
                    omega,
                )
                magnitude = np.linalg.norm(pair)
                if magnitude > radius:
                    pair *= radius / magnitude
                changed |= pair[1] != impulses[sibling]
                velocity += response[sibling] * (pair[1] - impulses[sibling])
                impulses[sibling] = pair[1]
                value = pair[0]
            else:
                value = old - omega * residual / diagonal[row]
                if kind in (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT):
                    value = max(value, 0.0)
            changed |= value != old
            velocity += response[row] * (value - old)
            impulses[row] = value
        # The solve stops after an exactly stationary sweep.
        if not changed:
            break
    return velocity, impulses


def _problem(dofs=43, support=18, seeded=False):
    rng = np.random.default_rng(471)
    factor = np.tril(rng.normal(scale=0.04, size=(dofs, dofs))) + np.eye(dofs)
    indices = np.sort(rng.choice(dofs, support, replace=False))
    rows = np.zeros((8, dofs))
    rows[:, indices] = rng.normal(scale=0.3, size=(8, support))
    rows[0] = 0.0
    rows[0, indices[0]] = 0.8
    initial = rng.normal(scale=0.2, size=dofs)
    bias = np.array((-0.2, -0.1, 0.04, -0.03, -0.12, -0.15, 0.02, 0.04))
    types = np.array(
        (
            PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
        ),
        dtype=np.int32,
    )
    parents = np.array((-1, -1, 1, 1, -1, -1, 5, 5), dtype=np.int32)
    mu = np.array((0, 0, 0.6, 0.6, 0, 0, 0.4, 0.4), dtype=np.float64)
    impulses = np.array((0, 0.4, 0.02, -0.01, 0.3, 0.2, -0.03, 0.01)) if seeded else np.zeros(8)
    jacobian = rows @ factor.T
    response = np.linalg.solve(factor.T, rows.T).T
    incident = jacobian @ initial
    diagonal = np.sum(rows * rows, axis=1) + 1.0e-6
    return factor, indices, rows, initial, bias, types, parents, mu, impulses, jacobian, response, incident, diagonal


def _mixed_problem():
    """Build independently coupled robot/free and free/free rows in physical coordinates."""
    rng = np.random.default_rng(472)
    factor = np.zeros((21, 21))
    for first, count in ((0, 9), (9, 6), (15, 6)):
        block = slice(first, first + count)
        factor[block, block] = np.tril(rng.normal(scale=0.08, size=(count, count))) + np.eye(count)
    jacobian = np.zeros((14, 21))
    supports = [
        [0],
        [0, 2, 6, 8, *range(9, 15)],
        [0, 2, 6, 8, *range(9, 15)],
        [0, 2, 6, 8, *range(9, 15)],
        [0, 1, 4],
        [1, 3, 5, *range(15, 21)],
        [1, 3, 5, *range(15, 21)],
        [1, 3, 5, *range(15, 21)],
        *[list(range(9, 21))] * 3,
        *[list(range(15, 21))] * 3,
    ]
    for row, support in enumerate(supports):
        jacobian[row, support] = rng.normal(scale=0.3, size=len(support))
    response = np.linalg.solve(factor @ factor.T, jacobian.T).T
    # Preserve a supplied MF response that differs from the cached factor's
    # inverse; substituting z for r must fail this fixture.
    response[8:] *= np.linspace(0.7, 1.3, 21)
    # Only the robot changes coordinates; free bodies retain physical deltas.
    factor[9:, 9:] = np.eye(12)
    z = np.linalg.solve(factor, jacobian.T).T
    r = response @ factor
    initial = rng.normal(scale=0.2, size=21)
    bias = np.array((-0.2, -0.1, 0.04, -0.03, -0.12, -0.15, 0.02, 0.04, -0.13, 0.04, -0.02, -0.18, 0.02, 0.01))
    types = np.array(
        [
            PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
            *[PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION],
            PGS_CONSTRAINT_TYPE_CONTACT,
            *[PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION] * 3,
        ],
        dtype=np.int32,
    )
    parents = np.array((-1, -1, 1, 1, -1, -1, 5, 5, -1, 8, 8, -1, 11, 11), dtype=np.int32)
    mu = np.where(types == PGS_CONSTRAINT_TYPE_FRICTION, 0.6, 0.0)
    diagonal = np.einsum("ij,ij->i", jacobian, response) + 1.0e-6
    return factor, z, r, jacobian, response, initial, bias, types, parents, mu, diagonal


class TestSteelSparsePGS(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_free_velocity_limit_is_stateless(self):
        """Apply free-body velocity limits without undoing old impulses or applying relaxation."""
        device = "cuda:0"

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        meta = np.zeros((2, 4), dtype=np.int32)
        meta[:, 0] = 0xFFFF  # Active endpoint at offset zero; static endpoint -1.
        meta[:, 1] = np.float32(0.5).view(np.int32)
        meta[:, 3] = PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT | (-1 << 16)
        rows = np.zeros((2, 1, 6), dtype=np.float32)
        rows[:, 0, 0] = 1.0
        velocity = np.zeros((2, 6), dtype=np.float32)
        velocity[:, 0] = [-2.0, 2.0]
        impulses = array([[3.0], [3.0]])
        delta = wp.zeros((2, 6), device=device)
        wp.launch_tiled(
            _get_pgs_solve_sparse_kernel(1, 6, 6, free_body_constraint_max=1),
            dim=[1],
            inputs=[
                2,
                array([0, 0], int),
                array([[0.0], [0.0]]),
                array([[1.0], [1.0]]),
                array([[0.0], [0.0]]),
                array(np.full((2, 1, 6), -1), int),
                array(np.zeros((2, 1, 6))),
                array(np.zeros((2, 1, 12))),
                array([[0.0], [0.0]]),
                array([[0], [0]], int),
                array([[-1], [-1]], int),
                array([[0.0], [0.0]]),
                array(np.ones((2, 6)), int),
                array(np.arange(12).reshape(2, 6), int),
                array(velocity.reshape(-1)),
                array([1, 1], int),
                array(meta, int),
                impulses,
                array(rows),
                array(np.zeros_like(rows)),
                array(rows),
                array(np.zeros_like(rows)),
                array([[0.0], [0.0]]),
                1,
                0.4,
            ],
            outputs=[delta],
            block_dim=64,
            device=device,
        )
        np.testing.assert_array_equal(impulses.numpy(), [[1.0], [0.0]])
        expected = np.zeros((2, 6))
        expected[0, 0] = 1.0
        np.testing.assert_array_equal(delta.numpy(), expected)

    def test_mixed_factor_sweeps_preserve_supplied_response(self):
        """Preserve dense-then-MF row order and distinct free-body responses."""
        factor, z, r, jacobian, response, initial, bias, types, parents, mu, diagonal = _mixed_problem()
        impulses = np.zeros(len(types))
        np.testing.assert_allclose(z @ r.T, jacobian @ response.T, atol=1.0e-12)
        self.assertGreater(np.linalg.norm(z[8:] - r[8:]), 0.1)
        for omega in _SWEEP_CASES:
            with self.subTest(omega=omega):
                settings = {"omega": omega}
                expected, expected_impulses = _sweep(
                    jacobian, response, initial, bias, diagonal, types, parents, mu, impulses, **settings
                )
                delta, actual_impulses = _sweep(
                    z,
                    r,
                    np.zeros_like(initial),
                    jacobian @ initial + bias,
                    diagonal,
                    types,
                    parents,
                    mu,
                    impulses,
                    **settings,
                )
                np.testing.assert_allclose(
                    initial + np.linalg.solve(factor.T, delta), expected, atol=3.0e-6, rtol=3.0e-5
                )
                np.testing.assert_allclose(actual_impulses, expected_impulses, atol=3.0e-6, rtol=3.0e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_mixed_sweeps_match_physical_reference(self):
        """Preserve mixed rows, capacities and relaxation across empty worlds."""
        for include_mf, omega in product((False, True), _SWEEP_CASES):
            with self.subTest(include_mf=include_mf, omega=omega):
                self._check_cuda_mixed_sweeps(include_mf, omega)

    def _check_cuda_mixed_sweeps(self, include_mf, omega):
        """Compare physical output while independently toggling the native free/free family."""
        factor, z, r, jacobian, response, initial, bias, types, parents, mu, diagonal = _mixed_problem()
        count = 14 if include_mf else 8
        expected, expected_impulses = _sweep(
            jacobian[:count],
            response[:count],
            initial,
            bias[:count],
            diagonal[:count],
            types[:count],
            parents[:count],
            mu[:count],
            np.zeros(count),
            omega=omega,
        )
        device, worlds, capacity, mf_capacity, support = "cuda:0", 3, 10, 8, 21

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def padded(value, cap=capacity, dtype=np.float32):
            result = np.zeros((worlds, cap, *value.shape[1:]), dtype=dtype)
            result[:, : len(value)] = value
            return array(result, int if dtype == np.int32 else float)

        row_dof = np.full((worlds, capacity, support), -1, dtype=np.int32)
        row_factor = np.zeros((worlds, capacity, support), dtype=np.float32)
        row_free_response = np.zeros((worlds, capacity, 12), dtype=np.float32)
        for row in range(8):
            indices = np.flatnonzero(
                np.any(z[1:4] != 0, axis=0)
                if 1 <= row < 4
                else np.any(z[5:8] != 0, axis=0)
                if 5 <= row < 8
                else z[row] != 0
            )
            indices = np.concatenate((indices[indices >= 9], indices[indices < 9]))
            row_dof[:, row, : len(indices)] = indices
            row_factor[:, row, : len(indices)] = z[row, indices]
            free_indices = indices[indices >= 9]
            row_free_response[:, row, : len(free_indices)] = r[row, free_indices]
        mf_z, mf_r = np.zeros((6, 12)), np.zeros((6, 12))
        mf_z[:3], mf_r[:3] = z[8:11, 9:21], r[8:11, 9:21]
        mf_z[3:, :6], mf_r[3:, :6] = z[11:, 15:21], r[11:, 15:21]
        meta = np.zeros((worlds, 4 * mf_capacity), dtype=np.int32)
        for row in range(6):
            first, second = (9, 15) if row < 3 else (15, -1)
            meta[:, 4 * row] = (first << 16) | (second & 0xFFFF)
            meta[:, 4 * row + 1] = np.float32(1.0 / diagonal[8 + row]).view(np.int32)
            meta[:, 4 * row + 2] = np.float32(bias[8 + row]).view(np.int32)
            parent = -1 if row % 3 == 0 else row - row % 3
            meta[:, 4 * row + 3] = int(types[8 + row]) | (parent << 16)
        dense_impulses = padded(np.zeros(8))
        mf_impulses = padded(np.zeros(6), mf_capacity)
        delta = wp.full((worlds, 21), float("nan"), device=device)
        wp.launch_tiled(
            _get_pgs_solve_sparse_kernel(
                capacity, 21, support, free_body_constraint_max=mf_capacity if include_mf else 0, free_row_dofs=6
            ),
            dim=[2],
            inputs=[
                worlds,
                array([8, 0, 8], int),
                padded(bias[:8]),
                padded(diagonal[:8]),
                dense_impulses,
                array(row_dof, int),
                array(row_factor),
                array(row_free_response),
                padded((jacobian @ initial)[:8]),
                padded(types[:8], dtype=np.int32),
                padded(parents[:8], dtype=np.int32),
                padded(mu[:8]),
                array(np.tile(np.arange(21) >= 9, (worlds, 1)), int),
                array(np.arange(worlds * 21).reshape(worlds, 21), int),
                array(np.tile(initial, worlds)),
                array([6, 0, 6] if include_mf else [0, 0, 0], int),
                array(meta, int),
                mf_impulses,
                padded(mf_z[:, :6], mf_capacity),
                padded(mf_z[:, 6:], mf_capacity),
                padded(mf_r[:, :6], mf_capacity),
                padded(mf_r[:, 6:], mf_capacity),
                padded(mu[8:], mf_capacity),
                8,
                omega,
            ],
            outputs=[delta],
            block_dim=64,
            device=device,
        )
        actual_delta = delta.numpy()
        np.testing.assert_array_equal(actual_delta[1], 0.0)
        for world in (0, 2):
            actual = initial + np.linalg.solve(factor.T, actual_delta[world])
            np.testing.assert_allclose(actual, expected, atol=3.0e-5, rtol=3.0e-4)
            np.testing.assert_allclose(
                dense_impulses.numpy()[world, :8], expected_impulses[:8], atol=3.0e-5, rtol=3.0e-4
            )
            if include_mf:
                np.testing.assert_allclose(
                    mf_impulses.numpy()[world, :6], expected_impulses[8:], atol=3.0e-5, rtol=3.0e-4
                )

    def test_factor_coordinate_residual_and_cross_term(self):
        """Preserve physical residuals and paired-tangent effective mass."""
        for dofs, support in ((7, 6), (43, 18), (75, 39)):
            with self.subTest(dofs=dofs, support=support):
                factor, _, rows, initial, _, _, _, _, _, jacobian, response, incident, _ = _problem(dofs, support)
                delta = np.linspace(-0.3, 0.2, dofs)
                physical = initial + np.linalg.solve(factor.T, delta)
                np.testing.assert_allclose(jacobian @ physical, incident + rows @ delta, atol=1.0e-12)
                np.testing.assert_allclose(jacobian @ response.T, rows @ rows.T, atol=1.0e-12)

    def test_factor_sweeps_match_physical_sweeps(self):
        """Preserve joint limits, friction loads and seeded-impulse semantics."""
        for seeded, omega in product((False, True), _SWEEP_CASES):
            with self.subTest(seeded=seeded, omega=omega):
                factor, _, rows, initial, bias, types, parents, mu, impulses, jacobian, response, incident, diag = (
                    _problem(seeded=seeded)
                )
                settings = {"omega": omega}
                expected, expected_impulses = _sweep(
                    jacobian, response, initial, bias, diag, types, parents, mu, impulses, **settings
                )
                delta, actual_impulses = _sweep(
                    rows, rows, np.zeros_like(initial), incident + bias, diag, types, parents, mu, impulses, **settings
                )
                actual = initial + np.linalg.solve(factor.T, delta)
                np.testing.assert_allclose(actual, expected, atol=2.0e-6, rtol=2.0e-5)
                np.testing.assert_allclose(actual_impulses, expected_impulses, atol=2.0e-6, rtol=2.0e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_sparse_sweeps_match_reference(self):
        """Match relaxed serial sweeps, including padding and zero-row worlds."""
        device = wp.get_device("cuda:0")
        for dofs, support in ((43, 18), (75, 39)):
            for seeded, omega in product((False, True), _SWEEP_CASES):
                with self.subTest(dofs=dofs, support=support, seeded=seeded, omega=omega):
                    _, indices, rows, initial, bias, types, parents, mu, impulses, _, _, incident, diag = _problem(
                        dofs, support, seeded
                    )
                    expected, expected_impulses = _sweep(
                        rows,
                        rows,
                        np.zeros_like(initial),
                        incident + bias,
                        diag,
                        types,
                        parents,
                        mu,
                        impulses,
                        omega=omega,
                    )
                    worlds, capacity = 3, 10
                    row_dof = np.full((worlds, capacity, support), -1, dtype=np.int32)
                    row_dof[:, :8] = indices
                    row_dof[:, 0, 1:] = -1
                    row_factor = np.zeros((worlds, capacity, support), dtype=np.float32)
                    row_factor[:, :8] = rows[:, indices]

                    def padded(array, dtype=np.float32, shape=(worlds, capacity)):
                        result = np.zeros(shape, dtype=dtype)
                        result[:, :8] = array
                        return wp.array(result, device=device)

                    actual_impulses = padded(impulses)
                    delta = wp.full((worlds, dofs), float("nan"), device=device)
                    kernel = _get_pgs_solve_sparse_kernel(capacity, dofs, support)
                    wp.launch_tiled(
                        kernel,
                        dim=[2],
                        inputs=[
                            worlds,
                            wp.array([8, 0, 8], dtype=int, device=device),
                            padded(bias),
                            padded(diag),
                            actual_impulses,
                            wp.array(row_dof, device=device),
                            wp.array(row_factor, device=device),
                            wp.zeros((worlds, capacity, 12), device=device),
                            padded(incident),
                            padded(types, np.int32),
                            padded(parents, np.int32),
                            padded(mu),
                            wp.zeros((worlds, dofs), dtype=int, device=device),
                            wp.array(np.arange(worlds * dofs).reshape(worlds, dofs), dtype=int, device=device),
                            wp.zeros(worlds * dofs, device=device),
                            wp.zeros(worlds, dtype=int, device=device),
                            wp.zeros((worlds, 4), dtype=int, device=device),
                            wp.zeros((worlds, 1), device=device),
                            wp.zeros((worlds, 1, 6), device=device),
                            wp.zeros((worlds, 1, 6), device=device),
                            wp.zeros((worlds, 1, 6), device=device),
                            wp.zeros((worlds, 1, 6), device=device),
                            wp.zeros((worlds, 1), device=device),
                            8,
                            omega,
                        ],
                        outputs=[delta],
                        block_dim=64,
                        device=device,
                    )
                    actual = delta.numpy()
                    result_impulses = actual_impulses.numpy()
                    np.testing.assert_array_equal(actual[1], 0.0)
                    for world in (0, 2):
                        np.testing.assert_allclose(actual[world], expected, atol=2.0e-5, rtol=2.0e-4)
                        np.testing.assert_allclose(
                            result_impulses[world, :8], expected_impulses, atol=2.0e-5, rtol=2.0e-4
                        )


DT = 1.0 / 400.0


STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")


def _build_chain(device, *, links=6):
    """Build an unbranched chain, whose mass factor has no structural zeros, above a ground sphere."""
    builder = newton.ModelBuilder()
    parent = -1
    joints = []
    for index in range(links):
        body = builder.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.01))
        builder.add_shape_sphere(body, radius=0.05)
        joints.append(
            builder.add_joint_revolute(
                parent,
                body,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.2 if index == 0 else -0.12), wp.quat_identity()),
            )
        )
        parent = body
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _build_model(*, free_count=0, free_first=False, robot_count=1, prescribed=False, heterogeneous=False, condim=None):
    """Replicate a contacting tripod, optionally surrounded by separate free bodies with a ``steel:condim``."""
    env = newton.ModelBuilder()
    if prescribed:
        support = env.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.02), wp.quat_identity()), is_kinematic=True)
        env.add_shape_box(support, hx=0.6, hy=0.4, hz=0.02)
        env.joint_qd[:6] = [0.2, 0.0, 0.04, 0.0, 0.0, 0.15]

    def add_objects():
        for index in range(free_count):
            body = env.add_link(
                xform=wp.transform(
                    wp.vec3(0.26 + index * 0.095, 0.0, 0.05),
                    wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.4, 0.5)), 0.6 + index * 0.2),
                ),
                mass=0.4 + index * 0.2,
                com=wp.vec3(0.01, -0.015, 0.007),
                inertia=wp.mat33(np.diag([0.003, 0.004, 0.005])),
            )
            cfg = env.default_shape_cfg.copy()
            cfg.density, cfg.mu = 0.0, 0.6
            env.add_shape_sphere(body, radius=0.05, cfg=cfg)
            env.add_articulation([env.add_joint_free(body)])

    if free_first:
        add_objects()
    for robot in range(robot_count):
        robot_q_start = len(env.joint_q)
        root = env.add_link(
            xform=wp.transform(wp.vec3(robot * 0.105, 0, 0.16), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.eye(3) * 0.03),
        )
        joints = [env.add_joint_free(root)]
        for x, y in ((0.16, 0.0), (-0.08, 0.14), (-0.08, -0.14)):
            child = env.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.002))
            cfg = env.default_shape_cfg.copy()
            cfg.density, cfg.mu = 0.0, 0.6
            env.add_shape_sphere(child, radius=0.055, cfg=cfg)
            joints.append(
                env.add_joint_revolute(
                    root,
                    child,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(x, y, -0.11), wp.quat_identity()),
                    target_pos=0.05,
                    target_ke=15.0,
                    target_kd=1.0,
                    armature=0.05,
                    limit_lower=-0.2,
                    limit_upper=0.2,
                    effort_limit=10.0,
                )
            )
        env.add_articulation(joints)
        env.joint_q[robot_q_start + 7] = 0.23  # Exercise an actual unilateral limit on the first step.
    if not free_first:
        add_objects()
    builder = newton.ModelBuilder()
    if condim is not None:
        SolverSteel.register_custom_attributes(builder)
    if heterogeneous:
        builder.add_world(env)
        free_world = newton.ModelBuilder()
        body = free_world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
        free_world.add_shape_sphere(body, radius=0.05)
        builder.add_world(free_world)
    else:
        builder.replicate(env, 2)
    if not prescribed:
        builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    if condim is not None:
        model.steel.condim.fill_(condim)
    return model


def _make_solver(model, sparse, **overrides):
    """Change only the representation selection between comparison solvers."""
    options = {
        "iterations": 32,
        "enable_joint_limits": True,
        "enable_joint_velocity_limits": False,
        "articulation_constraint_max": 32,
        "update_mass_matrix_interval": 4,
        # Sparse factors solve point friction; persistent friction patches keep dense factors.
        "friction_anchor_beta": 0.0,
    }
    options.update(overrides)
    with kernel_overrides({"sparse_mass_matrix": sparse}):
        return SolverSteel(model, solve_mode="matrix_free", **options)


def _case(sparse, *, free_count=0, free_first=False, robot_count=1, prescribed=False, heterogeneous=False):
    """Allocate a stable input state for eager and captured cache reuse."""
    model = _build_model(
        free_count=free_count,
        free_first=free_first,
        robot_count=robot_count,
        prescribed=prescribed,
        heterogeneous=heterogeneous,
    )
    solver = _make_solver(
        model,
        sparse,
        **(
            {"articulation_constraint_max": 96 if robot_count > 1 else 64, "free_body_constraint_max": 64}
            if free_count or robot_count > 1
            else {}
        ),
    )
    state, out = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128 if free_count else 64)
    return model, solver, state, out, model.control(), pipeline, pipeline.contacts()


def _step(case):
    """Advance without swapping the input object used by the dynamics cache."""
    _, solver, state, out, control, pipeline, contacts = case
    state.clear_forces()
    pipeline.collide(state, contacts)
    solver.step(state, out, control, contacts, DT)
    for name in STATE_FIELDS:
        wp.copy(getattr(state, name), getattr(out, name))


@unittest.skipUnless(wp.is_cuda_available(), "Sparse integrated solve requires CUDA")
class TestSteelSparseSolver(unittest.TestCase):
    def test_heterogeneous_worlds_do_not_invent_joint_support(self):
        """Admit a robot-only world and a separate free-only world without oversized row support."""
        dense, sparse = _case(False, heterogeneous=True), _case(True, heterogeneous=True)
        self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
        np.testing.assert_array_equal(sparse[1]._world_dof_count.numpy(), [9, 6])
        self.assertLessEqual(sparse[1]._sparse_row_dof.shape[2], sparse[1]._max_world_dofs)
        for _ in range(4):
            for case in (dense, sparse):
                _step(case)
            self.assert_states_close(dense[2], sparse[2])
        self.assert_sparse_state(sparse)

    def test_moving_prescribed_support_and_reset(self):
        """Preserve nonzero prescribed contact targets, support velocity and masked reset."""
        dense = _case(False, free_count=1, prescribed=True)
        sparse = _case(True, free_count=1, prescribed=True)
        self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
        self.assertTrue(sparse[1]._has_prescribed_response)
        targets_seen = [False, False]
        for step in range(8):
            if step == 4:
                for model, solver, state, *_ in (dense, sparse):
                    q = state.joint_q.numpy()
                    q[2] += 0.002
                    state.joint_q.assign(q)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    solver.reset(state, wp.array([True, False, False], dtype=bool, device=model.device))
            for case in (dense, sparse):
                _step(case)
            self.assert_states_close(dense[2], sparse[2])
            for family, (counter, target) in enumerate(
                (
                    ("_constraint_count", "_target_velocity"),
                    ("_mf_constraint_count", "_mf_target_velocity"),
                )
            ):
                counts = getattr(sparse[1], counter).numpy()
                for world, count in enumerate(counts):
                    actual = getattr(sparse[1], target).numpy()[world, :count]
                    expected = getattr(dense[1], target).numpy()[world, :count]
                    np.testing.assert_allclose(actual, expected, atol=2.0e-5, rtol=2.0e-5)
                    targets_seen[family] |= bool(np.any(np.abs(actual) > 0.01))
            prescribed_bodies = np.flatnonzero((sparse[0].body_flags.numpy() & int(newton.BodyFlags.KINEMATIC)) != 0)
            np.testing.assert_allclose(
                sparse[2].body_qd.numpy()[prescribed_bodies],
                np.tile([0.2, 0.0, 0.04, 0.0, 0.0, 0.15], (len(prescribed_bodies), 1)),
                atol=1.0e-6,
            )
        self.assertEqual(targets_seen, [True, True], "Exercise prescribed targets on both contact families")
        self.assert_sparse_state(sparse)

    def assert_states_close(self, first, second):
        """Allow float32 factor-order roundoff, not a changed physics recipe."""
        for name in STATE_FIELDS:
            # Independent elimination/reduction orders accumulate over 32 contact steps.
            tolerance = 2.0e-3 if name.endswith("qd") else 2.0e-4
            np.testing.assert_allclose(
                getattr(first, name).numpy(), getattr(second, name).numpy(), rtol=2.0e-4, atol=tolerance, err_msg=name
            )

    def assert_sparse_state(self, case):
        """Check storage elimination, row support, impulses, and residual coordinates."""
        _, solver, state, *_ = case
        size = solver._sparse_mass_matrix_size
        self.assertEqual(size, 9)
        for array in (
            solver._J_world,
            solver._Y_world,
            solver._J_by_size[size],
            solver._Y_by_size[size],
            solver._H_by_size[size],
            solver._L_by_size[size],
        ):
            self.assertEqual(array.shape, (1, 1, 1))  # Only argument stand-ins remain.
        for name in ("_tau_by_size", "_qdd_by_size"):
            self.assertNotIn(size, getattr(solver, name))
        for name in ("_cholesky_kernels_by_size", "_triangular_solve_kernels_by_size", "_hinv_jt_kernels_by_size"):
            self.assertIsNone(getattr(solver, name)[size])
        np.testing.assert_array_equal(solver._sparse_mass_matrix_status.numpy(), 0)
        self.assertFalse(solver.constraint_overflow.numpy().any())
        for name in STATE_FIELDS:
            self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
        counts, impulses = solver._constraint_count.numpy(), solver._impulses.numpy()
        incidence, rhs = solver._sparse_row_incident.numpy(), solver._rhs.numpy()
        v_hat, v_out = solver._v_hat.numpy(), solver._v_out.numpy()
        for world in range(solver._world_count):
            count = int(counts[world])
            self.assertLessEqual(count, solver.articulation_constraint_max)
            self.assertTrue(np.isfinite(impulses[world, :count]).all())
            kinds = solver._row_type.numpy()[world, :count]
            unilateral = (kinds == PGS_CONSTRAINT_TYPE_CONTACT) | (kinds == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
            self.assertTrue(np.all(impulses[world, :count][unilateral] >= -1.0e-7))
            z, physical_j = self._sparse_rows_physical(solver, world, count)
            velocity = solver._world_dof_indices.numpy()[world]
            np.testing.assert_allclose(physical_j @ v_hat[velocity], incidence[world, :count], atol=2.0e-5, rtol=2.0e-5)
            factor_residual = (
                z @ solver._sparse_factor_velocity_delta.numpy()[world] + incidence[world, :count] + rhs[world, :count]
            )
            np.testing.assert_allclose(
                physical_j @ v_out[velocity] + rhs[world, :count], factor_residual, atol=3.0e-5, rtol=3.0e-5
            )

    def _sparse_rows_physical(self, solver, world, count):
        """Recover physical Jacobians independently from the stored inverse factors."""
        plan = solver._sparse_mass_matrix_plan
        size = plan.dof_count
        row_dof, values = solver._sparse_row_dof.numpy(), solver._sparse_row_factor.numpy()
        z = np.zeros((count, solver._max_world_dofs))
        for row in range(count):
            support = row_dof[world, row]
            valid = support >= 0
            self.assertTrue(np.all(support[valid] < solver._max_world_dofs))
            self.assertEqual(len(set(support[valid])), int(valid.sum()))
            z[row, support[valid]] = values[world, row, valid]
        physical_j = np.zeros_like(z)
        for group, art in enumerate(solver._group_to_art[size].numpy()):
            if solver._art_to_world.numpy()[art] != world:
                continue
            inverse = np.zeros((size, size))
            inverse[plan.entry_rows, plan.columns] = solver._sparse_Linv.numpy()[group]
            offset = int(solver._articulation_world_dof_offset.numpy()[art])
            physical_j[:, offset + plan.permutation] = np.linalg.solve(inverse, z[:, offset : offset + size].T).T
        if solver._has_free_rigid_bodies:
            for art in solver._group_to_art[6].numpy():
                if solver._art_to_world.numpy()[art] == world:
                    offset = int(solver._articulation_world_dof_offset.numpy()[art])
                    physical_j[:, offset : offset + 6] = z[:, offset : offset + 6]
        return z, physical_j

    def _order_multi_articulation_rows(self, case):
        """Give this fixture identical semantic row order without changing production allocation."""
        _, solver, _, _, _, _, contacts = case
        sparse = solver._sparse_mass_matrix_size is not None
        names = [
            "rhs",
            "diag",
            "impulses",
            "row_type",
            "row_parent",
            "row_mu",
            "phi",
            "target_velocity",
        ]
        names += ["_sparse_row_dof", "_sparse_row_factor", "_sparse_row_incident"] if sparse else ["J_world", "Y_world"]
        arrays = {name: getattr(solver, f"_{name}" if not name.startswith("_") else name).numpy() for name in names}
        worlds, slots = (getattr(solver, name).numpy() for name in ("_contact_world", "_contact_slot"))
        shape_a, shape_b = contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy()
        identities = []
        for world, count in enumerate(solver._constraint_count.numpy()):
            jacobian = (
                self._sparse_rows_physical(solver, world, count)[1] if sparse else arrays["J_world"][world, :count]
            )
            bundles = []
            for row in np.flatnonzero(arrays["row_type"][world, :count] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT):
                dof = int(np.argmax(np.abs(jacobian[row])))
                bundles.append(((0, dof, int(np.sign(jacobian[row, dof]))), [row]))
            active = [
                c for c in range(int(contacts.rigid_contact_count.numpy()[0])) if worlds[c] == world and slots[c] >= 0
            ]
            for c in active:
                bundles.append(((1, int(shape_a[c]), int(shape_b[c])), list(range(slots[c], slots[c] + 3))))
            bundles.sort(key=lambda item: item[0])
            keys = [key for key, _ in bundles]
            self.assertEqual(len(keys), len(set(keys)), "Fixture must have unique limits and shape-pair contacts")
            order = np.array([row for _, rows in bundles for row in rows], dtype=int)
            np.testing.assert_array_equal(np.sort(order), np.arange(count))
            inverse = np.argsort(order)
            for values in arrays.values():
                values[world, :count] = values[world, order]
            parent = arrays["row_parent"][world, :count]
            parent[parent >= 0] = inverse[parent[parent >= 0]]
            slots[active] = inverse[slots[active]]
            identities.append([(key, len(rows)) for key, rows in bundles])
        for name, values in arrays.items():
            getattr(solver, f"_{name}" if not name.startswith("_") else name).assign(values)
        solver._contact_slot.assign(slots)
        return identities

    def test_two_articulations_share_contact_solve(self):
        """Distinguish factor groups from worlds when two branched robots contact each other."""
        dense, sparse = _case(False, robot_count=2), _case(True, robot_count=2)
        self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
        self.assertEqual(sparse[1]._sparse_Linv.shape[0], 4)
        self.assertEqual(sparse[1]._world_count, 2)
        contact_seen = False
        for _ in range(16):
            row_orders = []
            for case in (dense, sparse):
                solve = case[1]._launch_pgs_solve

                def ordered_solve(*args, _case=case, _solve=solve, _orders=row_orders, **kwargs):
                    # Both production GPU paths solve the same semantic order;
                    # atomic allocation order is not a physics-equivalence invariant.
                    _orders.append(self._order_multi_articulation_rows(_case))
                    return _solve(*args, **kwargs)

                with mock.patch.object(case[1], "_launch_pgs_solve", side_effect=ordered_solve):
                    _step(case)
            self.assertEqual(len(row_orders), 2)
            self.assertEqual(*row_orders)
            self.assert_states_close(dense[2], sparse[2])
            model, solver, _, _, _, _, contacts = sparse
            active = int(contacts.rigid_contact_count.numpy()[0])
            shape_body, body_art = model.shape_body.numpy(), solver._body_to_articulation.numpy()
            for shape_a, shape_b in zip(
                contacts.rigid_contact_shape0.numpy()[:active],
                contacts.rigid_contact_shape1.numpy()[:active],
                strict=True,
            ):
                body_a, body_b = shape_body[shape_a], shape_body[shape_b]
                contact_seen |= body_a >= 0 and body_b >= 0 and body_art[body_a] != body_art[body_b]
        self.assertTrue(contact_seen, "Fixture must couple separate responsive articulations")
        self.assert_sparse_state(sparse)

    def test_mixed_contacts_match_existing_solver(self):
        """Preserve robot/object, free/ground and free/free impulses in either articulation order."""
        for free_first in (False, True):
            with self.subTest(free_first=free_first):
                dense = _case(False, free_count=2, free_first=free_first)
                sparse = _case(True, free_count=2, free_first=free_first)
                self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
                self.assertEqual(sparse[0].articulation_count, 6)
                for case in (dense, sparse):
                    _step(case)
                self.assert_states_close(dense[2], sparse[2])
                for name in ("_constraint_count", "_mf_constraint_count"):
                    np.testing.assert_array_equal(getattr(dense[1], name).numpy(), getattr(sparse[1], name).numpy())
                for count_name, impulse_name in (
                    ("_constraint_count", "_impulses"),
                    ("_mf_constraint_count", "_mf_impulses"),
                ):
                    counts = getattr(sparse[1], count_name).numpy()
                    self.assertTrue(np.all(counts > 0), count_name)
                    for world, count in enumerate(counts):
                        np.testing.assert_allclose(
                            getattr(sparse[1], impulse_name).numpy()[world, :count],
                            getattr(dense[1], impulse_name).numpy()[world, :count],
                            atol=2.0e-4,
                            rtol=2.0e-3,
                        )
                contacts = sparse[-1]
                active = int(contacts.rigid_contact_count.numpy()[0])
                shape_body = sparse[0].shape_body.numpy()
                body_art = sparse[1]._body_to_articulation.numpy()
                free = sparse[1]._is_free_rigid.numpy()
                seen = set()
                for shape_a, shape_b in zip(
                    contacts.rigid_contact_shape0.numpy()[:active],
                    contacts.rigid_contact_shape1.numpy()[:active],
                    strict=True,
                ):
                    bodies = (shape_body[shape_a], shape_body[shape_b])
                    kinds = tuple(
                        sorted("static" if body < 0 else "free" if free[body_art[body]] else "robot" for body in bodies)
                    )
                    seen.add(kinds)
                self.assertTrue({("free", "robot"), ("free", "static"), ("free", "free")} <= seen, seen)
                for _ in range(15):
                    for case in (dense, sparse):
                        _step(case)
                    self.assert_states_close(dense[2], sparse[2])
                self.assert_sparse_state(sparse)

    def test_mixed_graph_replay_after_masked_reset(self):
        """Refresh a reset robot/free world and replay its complete shared solve graph."""
        eager = _case(True, free_count=2, free_first=True)
        graph = _case(True, free_count=2, free_first=True)
        for case in (eager, graph):
            _step(case)
            model, solver, state, *_ = case
            robot_art = int(solver._group_to_art[9].numpy()[0])
            root_joint = int(model.articulation_start.numpy()[robot_art])
            q_start = int(model.joint_q_start.numpy()[root_joint])
            q = state.joint_q.numpy()
            q[q_start + 2] += 0.01
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False, False], dtype=bool, device=model.device))
        held = graph[1]._sparse_Linv.numpy().copy()
        for case in (eager, graph):
            _step(case)
        np.testing.assert_array_equal(graph[1]._sparse_Linv.numpy()[1], held[1])
        with wp.ScopedCapture("cuda:0") as capture:
            for _ in range(4):
                _step(graph)
        for _ in range(4):
            wp.capture_launch(capture.graph)
            for _ in range(4):
                _step(eager)
        self.assert_states_close(eager[2], graph[2])
        self.assert_sparse_state(graph)

    def test_trajectory_masked_reset_and_inertial_notify(self):
        """Match 32 contact steps and refresh only the requested cached factors."""
        dense, sparse = _case(False), _case(True)
        self.assertIsNone(dense[1]._sparse_mass_matrix_size)
        self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
        contact_seen = limit_seen = False
        for _ in range(32):
            for case in (dense, sparse):
                _step(case)
            contact_seen |= bool(sparse[-1].rigid_contact_count.numpy()[0])
            limit_seen |= bool(np.any(sparse[1]._row_type.numpy() == PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
            self.assert_states_close(dense[2], sparse[2])
        self.assertTrue(contact_seen and limit_seen, "Fixture must exercise contacts and joint limits")
        self.assert_sparse_state(sparse)
        for case in (dense, sparse):
            _step(case)  # Step32 refreshes globally; step33 is a held-mass step.
        held = sparse[1]._sparse_Linv.numpy().copy()
        for model, solver, state, *_ in (dense, sparse):
            q = state.joint_q.numpy()
            q[2] += 0.02
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False, False], dtype=bool, device=model.device))
        for case in (dense, sparse):
            _step(case)
        np.testing.assert_array_equal(sparse[1]._mass_update_mask.numpy(), [1, 0])
        np.testing.assert_array_equal(sparse[1]._sparse_Linv.numpy()[1], held[1])
        self.assert_states_close(dense[2], sparse[2])
        for model, solver, *_ in (dense, sparse):
            mass = model.body_mass.numpy()
            mass[0] *= 1.1
            model.body_mass.assign(mass)
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        for case in (dense, sparse):
            _step(case)
        np.testing.assert_array_equal(sparse[1]._mass_update_mask.numpy(), [1, 1])
        self.assertFalse(np.array_equal(sparse[1]._sparse_Linv.numpy()[0], held[0]))
        self.assert_states_close(dense[2], sparse[2])
        self.assert_sparse_state(sparse)

    def test_graph_replay_preserves_sparse_problem(self):
        """Replay a complete four-step mass-refresh cadence without dense scratch."""
        eager, graph = _case(True), _case(True)
        for case in (eager, graph):
            _step(case)  # Finish allocation/compilation outside capture.
        with wp.ScopedCapture("cuda:0") as capture:
            for _ in range(4):
                _step(graph)
        for _ in range(8):
            wp.capture_launch(capture.graph)
            for _ in range(4):
                _step(eager)
        self.assert_states_close(eager[2], graph[2])
        self.assert_sparse_state(graph)

    def test_default_selects_sparse_factors_for_branched_articulations(self):
        """Select sparse factors for branched trees with point friction and keep dense factors for chains."""
        point = {"solve_mode": "matrix_free", "friction_anchor_beta": 0.0}
        self.assertEqual(SolverSteel(_build_model(), **point)._sparse_mass_matrix_size, 9)
        self.assertEqual(SolverSteel(_build_model(free_count=1), **point)._sparse_mass_matrix_size, 9)
        self.assertIsNone(SolverSteel(_build_chain("cuda:0"), **point)._sparse_mass_matrix_size)
        # The sparse kernels do not implement friction patches, the default friction.
        self.assertIsNone(SolverSteel(_build_model(), solve_mode="matrix_free")._sparse_mass_matrix_size)
        # The split solve uses dense factors.
        self.assertIsNone(
            SolverSteel(_build_model(), solve_mode="split", friction_anchor_beta=0.0)._sparse_mass_matrix_size
        )

    def test_free_body_angular_rows_keep_dense_factors(self):
        """Keep dense factors, and solve the angular rows, when a free body has ``steel:condim`` 6."""
        model = _build_model(free_count=1, condim=6)
        solver = SolverSteel(model, solve_mode="matrix_free", friction_anchor_beta=0.0)
        self.assertTrue(solver._free_rigid_angular_rows)
        self.assertIsNone(solver._sparse_mass_matrix_size)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1, control = model.state(), model.state(), model.control()
        for _ in range(4):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertIn(6, solver._contact_slots_needed.numpy()[:count].tolist())
        self.assertTrue(np.all(np.isfinite(state_0.body_q.numpy())))

    def test_unsupported_configurations_keep_existing_path(self):
        """Keep dense factors for full-support chains, velocity-limit rows, no iterations and large row capacities."""
        chain = _build_chain("cuda:0")
        self.assertIsNone(_make_solver(chain, True)._sparse_mass_matrix_size)
        model = _build_model()
        self.assertEqual(_make_solver(model, True)._sparse_mass_matrix_size, 9)
        for options in (
            {"enable_joint_velocity_limits": True},
            {"iterations": 0},
            {"articulation_constraint_max": 2048},
            {"friction_anchor_beta": 0.2},
            {"contact_warmstart": True},
            {"contact_regularization": 0.01},
            {"velocity_iterations": 1},
            {"contact_friction_gap_max": 0.01},
            {"enable_joint_friction": True},
        ):
            with self.subTest(options=options):
                solver = _make_solver(model, True, **options)
                self.assertIsNone(solver._sparse_mass_matrix_size)
                self.assertGreater(solver._J_by_size[9].size, 1)


def _build_y_tree(device, *, mimic=False, legacy_mimic=False, closure=False, closure_enabled=True):
    """Build a fixed-base Y tree (a root link with two child links), optionally coupled.

    The tree's mass factor has structural zeros, so it selects sparse factors on its own.
    The couplings are a joint-owned or legacy mimic between the two children, or a BALL
    loop-closing joint between their tips.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(np.eye(3) * 0.01)
    root = builder.add_link(mass=1.0, inertia=inertia)
    joints = [builder.add_joint_revolute(-1, root, axis=newton.Axis.Z)]
    children = []
    for side in (-1.0, 1.0):
        child = builder.add_link(mass=0.5, inertia=inertia)
        joints.append(
            builder.add_joint_revolute(
                root,
                child,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(side * 0.2, 0.0, 0.0), wp.quat_identity()),
            )
        )
        children.append(child)
    builder.add_articulation(joints)
    if mimic:
        builder.set_joint_mimic(joints[2], joints[1], coeffs=(0.0, 1.0))
    if legacy_mimic:
        builder.add_constraint_mimic(joints[2], joints[1], coef1=1.0)
    if closure:
        builder.add_joint_ball(
            children[0],
            children[1],
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
            enabled=closure_enabled,
        )
    return builder.finalize(device=device)


class TestSteelSparseSelectionGuard(unittest.TestCase):
    """Keep dense factors wherever the solver builds dense bilateral rows."""

    def test_bilateral_constraints_are_detected_on_the_model(self):
        """Detect mimic relationships and loop-closing joints, including disabled closures."""
        import warnings  # noqa: PLC0415

        from newton._src.solvers.steel.solver_steel import (  # noqa: PLC0415
            _model_has_bilateral_constraints,
        )

        self.assertFalse(_model_has_bilateral_constraints(_build_y_tree("cpu")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = _build_y_tree("cpu", legacy_mimic=True)
        for label, model in (
            ("joint mimic", _build_y_tree("cpu", mimic=True)),
            ("legacy mimic", legacy),
            ("closure", _build_y_tree("cpu", closure=True)),
            ("disabled closure", _build_y_tree("cpu", closure=True, closure_enabled=False)),
        ):
            with self.subTest(label):
                self.assertTrue(_model_has_bilateral_constraints(model))

        # An unowned world-root joint of a body outside every articulation is not a closure.
        builder = newton.ModelBuilder()
        link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_revolute(-1, link)])
        standalone = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_joint_revolute(-1, standalone)
        self.assertFalse(_model_has_bilateral_constraints(builder.finalize(device="cpu")))

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse integrated solve requires CUDA")
    def test_y_tree_selects_sparse_factors(self):
        """Select sparse factors for the uncoupled Y tree used by the bilateral guard."""
        self.assertEqual(
            SolverSteel(
                _build_y_tree("cuda:0"), solve_mode="matrix_free", friction_anchor_beta=0.0
            )._sparse_mass_matrix_size,
            3,
        )

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse integrated solve requires CUDA")
    def test_bilateral_rows_keep_dense_factors(self):
        """Keep dense factors, and their response storage, when the solver builds bilateral rows."""
        for label, kwargs in (
            ("mimic", {"mimic": True}),
            ("closure", {"closure": True}),
            ("disabled closure", {"closure": True, "closure_enabled": False}),
        ):
            with self.subTest(label):
                solver = SolverSteel(_build_y_tree("cuda:0", **kwargs), solve_mode="matrix_free")
                self.assertIsNone(solver._sparse_mass_matrix_size)
                self.assertGreater(solver._J_by_size[3].size, 1)


class TestSteelLocalInternalSolve(unittest.TestCase):
    @staticmethod
    def _empty_mf_inputs(world_count: int, device: str | wp.Device) -> list:
        return [
            wp.zeros(world_count, dtype=wp.int32, device=device),
            wp.zeros((world_count, 4), dtype=wp.int32, device=device),
            wp.zeros((world_count, 1), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1), dtype=wp.float32, device=device),
            0,
            wp.zeros((world_count, 1), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1), dtype=wp.float32, device=device),
        ]

    def test_local_owner_classifier_rejects_unsupported_worlds(self):
        """Route worlds the local solve cannot own to the general owner, and compact only active pair candidates."""
        device = "cpu"
        constraint_count = wp.array([5, 8, 21, 8, 8, 8, 0, 41], dtype=wp.int32, device=device)
        # Rows before each world's bound are internal rows, the rest contact rows.
        internal_rows = [5, 2, 2, 2, 2, 8, 0, 2]
        row_type_host = np.full((8, 41), PGS_CONSTRAINT_TYPE_CONTACT, dtype=np.int32)
        for world, end in enumerate(internal_rows):
            row_type_host[world, :end] = PGS_CONSTRAINT_TYPE_MIMIC
        row_type = wp.array(row_type_host, dtype=wp.int32, device=device)
        mf_count = wp.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=wp.int32, device=device)
        mf_body_a = wp.full((8, 1), -1, dtype=wp.int32, device=device)
        mf_body_b = wp.array([[-1], [-1], [-1], [0], [1], [-1], [-1], [-1]], dtype=wp.int32, device=device)
        mf_row_type = wp.full((8, 1), PGS_CONSTRAINT_TYPE_CONTACT, dtype=wp.int32, device=device)
        body_to_articulation = wp.array([12, 99], dtype=wp.int32, device=device)
        articulation_dof_count_host = np.full(16, 20, dtype=np.int32)
        articulation_dof_count_host[5] = 7
        articulation_dof_count = wp.array(articulation_dof_count_host, device=device)
        primary = wp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=wp.int32, device=device)
        pair = wp.array([-1, 10, 11, 12, 13, -1, 14, 15], dtype=wp.int32, device=device)
        residual_pair = wp.array([-1, 10, 11, 12, 13, -1, 14, 15], dtype=wp.int32, device=device)
        owner = wp.empty(8, dtype=wp.int32, device=device)
        general_count = wp.zeros(1, dtype=wp.int32, device=device)
        general_worlds = wp.empty(8, dtype=wp.int32, device=device)

        wp.launch(
            classify_local_solve_worlds,
            dim=8,
            inputs=[
                constraint_count,
                row_type,
                mf_count,
                mf_body_a,
                mf_body_b,
                mf_row_type,
                body_to_articulation,
                articulation_dof_count,
                primary,
                pair,
                residual_pair,
                20,
                40,
                12,
            ],
            outputs=[owner, general_count, general_worlds],
            device=device,
        )

        np.testing.assert_array_equal(
            owner.numpy(),
            [
                PGS_LOCAL_SOLVE_OWNER_SINGLE,
                PGS_LOCAL_SOLVE_OWNER_PAIR,
                PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
                PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
            ],
        )
        self.assertEqual(int(general_count.numpy()[0]), 3)
        np.testing.assert_array_equal(general_worlds.numpy()[:3], [4, 5, 7])

        active_count = wp.zeros(1, dtype=wp.int32, device=device)
        active_primary = wp.empty(6, dtype=wp.int32, device=device)
        active_secondary = wp.empty(6, dtype=wp.int32, device=device)
        wp.launch(
            compact_local_pair_candidates,
            dim=8,
            inputs=[
                primary,
                pair,
                wp.array(np.arange(16), dtype=wp.int32, device=device),
                owner,
                PGS_LOCAL_SOLVE_OWNER_PAIR,
            ],
            outputs=[active_count, active_primary, active_secondary],
            device=device,
        )
        self.assertEqual(int(active_count.numpy()[0]), 1)
        np.testing.assert_array_equal(active_primary.numpy()[:1], [1])
        np.testing.assert_array_equal(active_secondary.numpy()[:1], [10])

    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_kernel_matches_sequential_pgs_and_respects_world_ownership(self):
        """Match a sequential PGS reference in every kernel variant, leaving unowned worlds unchanged."""
        device = wp.get_cuda_device()
        dof_count = 3
        max_constraints = 4
        local_max_constraints = 2
        iterations = 4
        omega = 1.0

        lower = np.array([[2.0, 0.0, 0.0], [0.25, 1.5, 0.0], [-0.1, 0.2, 1.25]], dtype=np.float32)
        jacobian = np.zeros((2, max_constraints, dof_count), dtype=np.float32)
        jacobian[0, 0] = [1.0, 0.5, -0.25]
        jacobian[0, 1] = [-0.4, 0.3, 0.8]
        jacobian[1, 0] = [0.2, -0.1, 0.7]
        velocity = np.array([0.3, -0.5, 0.1, 9.0, 8.0, 7.0], dtype=np.float32)
        rhs = np.zeros((2, max_constraints), dtype=np.float32)
        rhs[0, :2] = [-0.2, 0.15]
        cfm = np.zeros((2, max_constraints), dtype=np.float32)
        cfm[0, :2] = [0.01, 0.02]
        row_type = np.zeros((2, max_constraints), dtype=np.int32)
        row_type[0, :2] = [PGS_CONSTRAINT_TYPE_MIMIC, PGS_CONSTRAINT_TYPE_JOINT_LIMIT]
        row_type[1, 0] = PGS_CONSTRAINT_TYPE_MIMIC

        impulses = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
        diagonal_out = wp.array(cfm, dtype=wp.float32, device=device)
        velocity_out = wp.array(velocity, dtype=wp.float32, device=device)
        candidate_articulations = wp.array([0, 1], dtype=wp.int32, device=device)
        lower_group = wp.array(np.stack([lower, np.eye(dof_count, dtype=np.float32)]), dtype=wp.float32, device=device)
        jacobian_group = wp.array(jacobian, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_owned_kernel(max_constraints, local_max_constraints, dof_count, device.arch)
        wp.launch_tiled(
            kernel,
            dim=[2],
            inputs=[
                candidate_articulations,
                candidate_articulations,
                1,
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 3], dtype=wp.int32, device=device),
                wp.array([1, 0], dtype=wp.int32, device=device),
                wp.array([2, 1], dtype=wp.int32, device=device),
                lower_group,
                jacobian_group,
                lower_group,
                jacobian_group,
                wp.array(rhs, dtype=wp.float32, device=device),
                wp.array(row_type, dtype=wp.int32, device=device),
                wp.full((2, max_constraints), -1, dtype=wp.int32, device=device),
                wp.zeros((2, max_constraints), dtype=wp.float32, device=device),
                *self._empty_mf_inputs(2, device),
                iterations,
                omega,
                0,
                0,
            ],
            outputs=[diagonal_out, impulses, velocity_out],
            block_dim=32,
            device=device,
        )
        wp.synchronize_device(device)

        mass = lower @ lower.T
        response = np.stack([np.linalg.solve(mass, row) for row in jacobian[0, :2]])
        diagonal = np.sum(jacobian[0, :2] * response, axis=1) + cfm[0, :2]
        expected_velocity = velocity[:dof_count].copy()
        expected_impulses = np.zeros(2, dtype=np.float32)
        for _ in range(iterations):
            for row in range(2):
                delta = -(jacobian[0, row] @ expected_velocity + rhs[0, row]) / diagonal[row]
                new_impulse = expected_impulses[row] + omega * delta
                if row_type[0, row] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
                    new_impulse = max(new_impulse, 0.0)
                expected_velocity += response[row] * (new_impulse - expected_impulses[row])
                expected_impulses[row] = new_impulse

        np.testing.assert_allclose(velocity_out.numpy()[:dof_count], expected_velocity, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0, :2], expected_impulses, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(diagonal_out.numpy()[0, :2], diagonal, rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(diagonal_out.numpy()[1], cfm[1])
        np.testing.assert_array_equal(velocity_out.numpy()[dof_count:], velocity[dof_count:])
        np.testing.assert_array_equal(impulses.numpy()[1], np.zeros(max_constraints, dtype=np.float32))

        def launch_variant(
            warps_per_block: int,
            *,
            lanes_per_world: int = 32,
            contact_capable: bool = True,
            persistent_queue: bool = False,
            active_count: int = 2,
            dense_response_matrix: bool = False,
        ):
            variant_diagonal = wp.array(cfm, dtype=wp.float32, device=device)
            variant_impulses = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
            variant_velocity = wp.array(velocity, dtype=wp.float32, device=device)
            variant_kernel = _get_pgs_solve_local_owned_kernel(
                max_constraints,
                local_max_constraints,
                dof_count,
                device.arch,
                persistent_queue=persistent_queue,
                warps_per_block=warps_per_block,
                lanes_per_world=lanes_per_world,
                contact_capable=contact_capable,
                dense_response_matrix=dense_response_matrix,
            )
            inputs = [candidate_articulations, candidate_articulations]
            if persistent_queue:
                inputs.extend([wp.array([active_count], dtype=wp.int32, device=device), 2])
            inputs.extend(
                [
                    PGS_LOCAL_SOLVE_OWNER_SINGLE,
                    wp.array([0, 1], dtype=wp.int32, device=device),
                    wp.array([0, 1], dtype=wp.int32, device=device),
                    wp.array([0, 3], dtype=wp.int32, device=device),
                    wp.array([PGS_LOCAL_SOLVE_OWNER_SINGLE] * 2, dtype=wp.int32, device=device),
                    wp.array([2, 1], dtype=wp.int32, device=device),
                    lower_group,
                    jacobian_group,
                    lower_group,
                    jacobian_group,
                    wp.array(rhs, dtype=wp.float32, device=device),
                    wp.array(row_type, dtype=wp.int32, device=device),
                    wp.full((2, max_constraints), -1, dtype=wp.int32, device=device),
                    wp.zeros((2, max_constraints), dtype=wp.float32, device=device),
                    *self._empty_mf_inputs(2, device),
                    iterations,
                    omega,
                    0,
                    0,
                ]
            )
            worlds_per_block = warps_per_block * 32 // lanes_per_world
            block_count = 2 // worlds_per_block
            if persistent_queue:
                block_count = max(block_count, 1)
            wp.launch_tiled(
                variant_kernel,
                dim=[block_count],
                inputs=inputs,
                outputs=[variant_diagonal, variant_impulses, variant_velocity],
                block_dim=32 * warps_per_block,
                device=device,
            )
            wp.synchronize_device(device)
            return variant_diagonal.numpy(), variant_impulses.numpy(), variant_velocity.numpy()

        one_warp_outputs = launch_variant(1)
        exact_variants = [launch_variant(2)]
        numerical_variants = [
            launch_variant(1, contact_capable=False),
            launch_variant(2, contact_capable=False),
        ]
        matrix_variants = [
            launch_variant(1, contact_capable=False, dense_response_matrix=True),
            launch_variant(
                1,
                lanes_per_world=16,
                contact_capable=False,
                dense_response_matrix=True,
            ),
            launch_variant(
                1,
                lanes_per_world=8,
                contact_capable=False,
                persistent_queue=True,
                dense_response_matrix=True,
            ),
        ]
        for variant_outputs in exact_variants:
            for one_warp, variant in zip(one_warp_outputs, variant_outputs, strict=True):
                np.testing.assert_array_equal(variant, one_warp)
        for variant_outputs in numerical_variants + matrix_variants:
            for one_warp, variant in zip(one_warp_outputs, variant_outputs, strict=True):
                np.testing.assert_allclose(variant, one_warp, rtol=0.0, atol=2.0e-6)

        one_warp_queue = launch_variant(1, persistent_queue=True, active_count=1)
        two_warp_queue = launch_variant(2, persistent_queue=True, active_count=1)
        for one_warp, two_warp in zip(one_warp_queue, two_warp_queue, strict=True):
            np.testing.assert_array_equal(two_warp, one_warp)

    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_kernel_waits_for_delayed_friction_activation(self):
        """Solve delayed friction rows even when the normal rows converge before friction starts."""
        device = wp.get_cuda_device()
        dof_count = 3
        max_constraints = 3
        candidate = wp.array([0], dtype=wp.int32, device=device)
        lower = wp.array(np.eye(dof_count, dtype=np.float32)[None], dtype=wp.float32, device=device)
        jacobian = wp.array(np.eye(dof_count, dtype=np.float32)[None], dtype=wp.float32, device=device)
        impulses = wp.array([[1.0, 0.0, 0.0]], dtype=wp.float32, device=device)
        diagonal = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        velocity = wp.array([0.0, 1.0, 0.0], dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_owned_kernel(max_constraints, max_constraints, dof_count, device.arch)
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                candidate,
                candidate,
                1,
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                wp.array([max_constraints], dtype=wp.int32, device=device),
                lower,
                jacobian,
                lower,
                jacobian,
                wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
                wp.array(
                    [[PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]],
                    dtype=wp.int32,
                    device=device,
                ),
                wp.array([[-1, 0, 0]], dtype=wp.int32, device=device),
                wp.array([[0.0, 0.5, 0.5]], dtype=wp.float32, device=device),
                *self._empty_mf_inputs(1, device),
                3,
                1.0,
                2,
                0,
            ],
            outputs=[diagonal, impulses, velocity],
            block_dim=32,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(diagonal.numpy()[0], np.ones(max_constraints), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0], [1.0, -0.5, 0.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(velocity.numpy(), [0.0, 0.5, 0.0], rtol=0.0, atol=1.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_kernel_uses_patch_load_and_coupled_tangent_response(self):
        """Match the pooled patch load and two-row tangent solve used by the general owner."""
        device = wp.get_cuda_device()
        dof_count = 2
        max_constraints = 4
        candidate = wp.array([0], dtype=wp.int32, device=device)
        lower = wp.array(np.eye(dof_count, dtype=np.float32)[None], dtype=wp.float32, device=device)
        jacobian = wp.array(
            [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0]]],
            dtype=wp.float32,
            device=device,
        )
        diagonal = wp.array([[1.0, 0.0, 0.0, 1.0]], dtype=wp.float32, device=device)
        impulses = wp.array([[0.0, 0.0, 0.0, 4.0]], dtype=wp.float32, device=device)
        velocity = wp.zeros(dof_count, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_owned_kernel(
            max_constraints,
            max_constraints,
            dof_count,
            device.arch,
            dense_response_matrix=True,
        )
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                candidate,
                candidate,
                PGS_LOCAL_SOLVE_OWNER_SINGLE,
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([PGS_LOCAL_SOLVE_OWNER_SINGLE], dtype=wp.int32, device=device),
                wp.array([max_constraints], dtype=wp.int32, device=device),
                lower,
                jacobian,
                lower,
                jacobian,
                wp.array([[0.0, -1.0, -2.0, 0.0]], dtype=wp.float32, device=device),
                wp.array(
                    [
                        [
                            PGS_CONSTRAINT_TYPE_CONTACT,
                            PGS_CONSTRAINT_TYPE_FRICTION,
                            PGS_CONSTRAINT_TYPE_FRICTION,
                            PGS_CONSTRAINT_TYPE_CONTACT,
                        ]
                    ],
                    dtype=wp.int32,
                    device=device,
                ),
                # Normal rows 0 -> 3 -> 0 pool the load. Tangent rows point to row 0.
                wp.array([[3, 0, 0, 0]], dtype=wp.int32, device=device),
                wp.array([[0.0, 0.5, 0.5, 0.0]], dtype=wp.float32, device=device),
                *self._empty_mf_inputs(1, device),
                1,
                1.0,
                0,
                0,
            ],
            outputs=[diagonal, impulses, velocity],
            block_dim=32,
            device=device,
        )
        wp.synchronize_device(device)

        # The pooled normal impulse gives radius 2. The unconstrained coupled
        # tangent solution of [[1, .5], [.5, 1.25]] x = [1, 2] is [.25, 1.5].
        np.testing.assert_allclose(impulses.numpy()[0], [0.0, 0.25, 1.5, 4.0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(velocity.numpy(), [1.0, 1.5], rtol=0.0, atol=2.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_residual_queue_matches_sequential_dense_and_matrix_free_pgs(self):
        """Match a sequential dense and matrix-free PGS reference with the residual pair queue."""
        device = wp.get_cuda_device()
        primary_dofs = 3
        secondary_dofs = 6
        max_constraints = 4
        iterations = 4

        primary_jacobian = np.zeros((2, max_constraints, primary_dofs), dtype=np.float32)
        secondary_jacobian = np.zeros((2, max_constraints, secondary_dofs), dtype=np.float32)
        primary_jacobian[0, 0] = [0.5, -0.25, 0.1]
        secondary_jacobian[1, 0, 0] = 0.4
        dense_rhs = np.zeros((1, max_constraints), dtype=np.float32)
        dense_rhs[0, 0] = 0.2
        dense_cfm = np.zeros((1, max_constraints), dtype=np.float32)
        dense_cfm[0, 0] = 0.02
        initial_velocity = np.array([0.25, -0.1, 0.05, -0.3, 0.2, -0.1, 0.0, 0.0, 0.0], dtype=np.float32)

        mf_count = 3
        mf_jacobian = np.zeros((1, max_constraints, 6), dtype=np.float32)
        mf_response = np.zeros_like(mf_jacobian)
        mf_jacobian[0, 0, 0] = 1.0
        mf_jacobian[0, 1, 1] = 1.0
        mf_jacobian[0, 2, 2] = 1.0
        mf_response[:] = mf_jacobian
        mf_rhs = np.array([-0.1, 0.0, 0.0], dtype=np.float32)
        mf_types = np.array(
            [PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION],
            dtype=np.int32,
        )
        mf_mu = np.array([0.0, 0.5, 0.5], dtype=np.float32)

        def float_bits(value: float) -> np.int32:
            return np.asarray(value, dtype=np.float32).view(np.int32).item()

        packed_meta = np.zeros((1, max_constraints * 4), dtype=np.int32)
        packed_dofs = np.asarray((0xFFFF << 16) | primary_dofs, dtype=np.uint32).view(np.int32).item()
        for row in range(mf_count):
            packed_meta[0, row * 4 : row * 4 + 4] = [
                packed_dofs,
                float_bits(1.0),
                float_bits(mf_rhs[row]),
                int(mf_types[row]) | (0 << 16),
            ]

        dense_impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        mf_impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        diagonal = wp.array(dense_cfm, dtype=wp.float32, device=device)
        velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_owned_kernel(
            max_constraints,
            2,
            primary_dofs,
            device.arch,
            paired_dof_count=secondary_dofs,
            persistent_queue=True,
            warps_per_block=2,
            free_body_constraint_max=max_constraints,
            local_mf_max_constraints=max_constraints,
            dense_response_matrix=True,
        )
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                2,
                PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 0], dtype=wp.int32, device=device),
                wp.array([0, primary_dofs], dtype=wp.int32, device=device),
                wp.array([PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                wp.array(np.stack([np.eye(primary_dofs, dtype=np.float32)] * 2), device=device),
                wp.array(primary_jacobian, device=device),
                wp.array(np.stack([np.eye(secondary_dofs, dtype=np.float32)] * 2), device=device),
                wp.array(secondary_jacobian, device=device),
                wp.array(dense_rhs, device=device),
                wp.array([[PGS_CONSTRAINT_TYPE_MIMIC, 0, 0, 0]], dtype=wp.int32, device=device),
                wp.full((1, max_constraints), -1, dtype=wp.int32, device=device),
                wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
                wp.array([mf_count], dtype=wp.int32, device=device),
                wp.array(packed_meta, device=device),
                mf_impulses,
                wp.zeros_like(wp.array(mf_jacobian, device=device)),
                wp.array(mf_jacobian, device=device),
                wp.zeros_like(wp.array(mf_response, device=device)),
                wp.array(mf_response, device=device),
                wp.array([np.pad(mf_mu, (0, 1))], device=device),
                0,
                wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
                wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
                iterations,
                1.0,
                0,
                0,
            ],
            outputs=[diagonal, dense_impulses, velocity],
            block_dim=64,
            device=device,
        )
        wp.synchronize_device(device)

        dense_jacobian = np.concatenate([primary_jacobian[0, 0], secondary_jacobian[1, 0]])
        dense_response = dense_jacobian.copy()
        dense_denominator = float(dense_jacobian @ dense_response + dense_cfm[0, 0])
        expected_velocity = initial_velocity.copy()
        expected_dense_impulse = np.float32(0.0)
        expected_mf_impulses = np.zeros(mf_count, dtype=np.float32)
        for _ in range(iterations):
            dense_delta = -(dense_jacobian @ expected_velocity + dense_rhs[0, 0]) / dense_denominator
            dense_new = expected_dense_impulse + dense_delta
            expected_velocity += dense_response * (dense_new - expected_dense_impulse)
            expected_dense_impulse = dense_new

            for row in range(mf_count):
                old_impulse = expected_mf_impulses[row]
                new_impulse = old_impulse - (expected_velocity[primary_dofs + row] + mf_rhs[row])
                if mf_types[row] == PGS_CONSTRAINT_TYPE_CONTACT:
                    new_impulse = max(new_impulse, 0.0)
                else:
                    radius = max(mf_mu[row] * expected_mf_impulses[0], 0.0)
                    sibling = 2 if row == 1 else 1
                    trial = np.array([new_impulse, expected_mf_impulses[sibling]], dtype=np.float32)
                    magnitude = float(np.linalg.norm(trial))
                    if radius <= 0.0:
                        new_impulse = 0.0
                    elif magnitude > radius:
                        trial *= radius / magnitude
                        new_impulse = trial[0]
                        sibling_delta = trial[1] - expected_mf_impulses[sibling]
                        expected_mf_impulses[sibling] = trial[1]
                        expected_velocity[primary_dofs + sibling] += sibling_delta
                expected_mf_impulses[row] = new_impulse
                expected_velocity[primary_dofs + row] += new_impulse - old_impulse

        np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=0.0, atol=3.0e-6)
        np.testing.assert_allclose(dense_impulses.numpy()[0, 0], expected_dense_impulse, rtol=0.0, atol=3.0e-6)
        np.testing.assert_allclose(mf_impulses.numpy()[0, :mf_count], expected_mf_impulses, rtol=0.0, atol=3.0e-6)


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
                int(solver._local_solve_owner.numpy()[0]),
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


def _assert_owner_parity(test, general, local):
    """Assert the local owner trajectory matches the general owner row for row."""
    test.assertEqual(len(general), len(local))
    test.assertGreater(general[0][0], 0, "mixed scene generated no dense constraint rows")
    for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
        test.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
        test.assertEqual(actual[7], expected[7], f"matrix-free row count differed at step {step}")
        np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
        np.testing.assert_array_equal(actual[9], expected[9], err_msg=f"matrix-free row types differed at step {step}")
        for label, expected_value, actual_value in zip(
            ("diagonal", "impulses", "joint_q", "joint_qd", "mf_impulses"),
            (*expected[1:5], expected[8]),
            (*actual[1:5], actual[8]),
            strict=True,
        ):
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                rtol=5.0e-4,
                atol=2.0e-6,
                err_msg=f"{label} differed at step {step}",
            )


class TestSteelLocalOwners(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_internal_mixed_world_matches_general_response(self):
        """Match the general response when an articulation contacts a free body."""
        general_solver, general = _run_mixed_response(
            "tiled", warmstart=False, preelimination=False, inactive_joint_limit_capacity=True
        )
        local_solver, local = _run_mixed_response(
            "par_row", warmstart=False, preelimination=False, inactive_joint_limit_capacity=True
        )

        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        self.assertEqual(len(general), len(local))
        self.assertGreater(general[0][0], 0, "mixed scene generated no dense constraint rows")
        self.assertTrue(
            any(sample[5] == PGS_LOCAL_SOLVE_OWNER_PAIR for sample in local),
            "mixed contact never selected the paired local owner",
        )

        free_articulation = int(np.flatnonzero(general_solver._model_plan.is_free_rigid)[0])
        free_dof_start = int(general_solver._model_plan.articulation_dof_start[free_articulation])
        self.assertGreater(
            abs(float(general[0][4][free_dof_start + 2]) + 3.0),
            1.0e-4,
            "general response did not couple the dense contact to the free body",
        )

        for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=2.0e-6,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_pair_matches_general_friction_and_restitution(self):
        """Preserve paired friction projection and restitution response."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        pair_samples = [sample for sample in local if sample[5] == PGS_LOCAL_SOLVE_OWNER_PAIR]
        self.assertTrue(pair_samples, "mixed friction contact never selected the paired local owner")
        self.assertTrue(
            any(
                np.any(np.abs(sample[2][sample[6] == PGS_CONSTRAINT_TYPE_FRICTION]) > 1.0e-5) for sample in pair_samples
            ),
            "paired solve produced no nonzero friction impulse",
        )

        for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=2.0e-6,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owners_match_general_contact_regularization(self):
        """Apply the general owner's per-row regularization weights in the local owners."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "tangential_velocity": 2.0,
            "contact_regularization": 1.0,
            "model_kwargs": {"static_support": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertTrue(general_solver._regularization_enabled)
        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        owners = [sample[5] for sample in local]
        self.assertIn(
            PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL, owners, "regularized mixed contact never selected the residual owner"
        )
        self.assertTrue(
            any(sample[7] > 0 and sample[5] != PGS_LOCAL_SOLVE_OWNER_GENERAL for sample in local),
            "no local owner solved matrix-free rows",
        )
        # Penetrating contact rows carry a weight below one in both the dense and matrix-free rows.
        constraint_count, mf_count = local[-1][0], local[-1][7]
        self.assertTrue((local_solver._row_w.numpy()[0, :constraint_count] < 1.0).any())
        self.assertTrue((local_solver._mf_row_w.numpy()[0, :mf_count] < 1.0).any())
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owner_matches_general_persistent_patch_friction(self):
        """Preserve pooled normal load and coupled tangent response under local ownership."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "model_kwargs": {"static_support": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertTrue(general_solver._friction_anchors_enabled)
        self.assertTrue(local_solver._friction_anchors_enabled)
        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        self.assertIn(PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL, [sample[5] for sample in local])

        self.assertTrue(
            any(
                sample[10][row] >= 0 and sample[10][row] != row
                for sample in local
                for row in np.flatnonzero(sample[9] == PGS_CONSTRAINT_TYPE_CONTACT)
            ),
            "scene generated no linked multi-normal patch",
        )

        tangent_cross_terms = []
        for sample in local:
            row_type, row_parent = sample[9], sample[10]
            jacobian_a, jacobian_b = sample[11], sample[12]
            response_a, response_b = sample[13], sample[14]
            for row in np.flatnonzero(row_type == PGS_CONSTRAINT_TYPE_FRICTION):
                parent = row_parent[row]
                if row != parent + 1:
                    continue
                sibling = parent + 2
                tangent_cross_terms.append(
                    float(jacobian_a[row] @ response_a[sibling] + jacobian_b[row] @ response_b[sibling])
                )
        self.assertTrue(tangent_cross_terms, "scene generated no tangent pair")
        self.assertGreater(max(abs(value) for value in tangent_cross_terms), 1.0e-3)
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_free_body_velocity_limit_rows_stay_on_general_owner(self):
        """Keep worlds with free-body velocity-limit rows on the general owner."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "model_kwargs": {"free_body_velocity_limit": 1.0},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        limited = [sample for sample in local if np.any(sample[9] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)]
        self.assertTrue(limited, "free body never produced a velocity-limit row")
        # The residual owner is eligible for this world; only the velocity-limit rows keep it general.
        self.assertGreaterEqual(int(local_solver._local_residual_pair_articulation.numpy()[0]), 0)
        for sample in limited:
            self.assertEqual(sample[5], PGS_LOCAL_SOLVE_OWNER_GENERAL, "velocity-limit rows left the general owner")
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_free_body_first_world_layout_matches_general(self):
        """Reject the residual owner when the world packs the free body before the articulation."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "model_kwargs": {"static_support": True, "free_body_first": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        free_articulation = int(np.flatnonzero(local_solver._model_plan.is_free_rigid)[0])
        self.assertEqual(int(local_solver._model_plan.articulation_dof_start[free_articulation]), 0)
        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        np.testing.assert_array_equal(local_solver._local_residual_pair_articulation.numpy(), [-1])
        self.assertGreaterEqual(int(local_solver._local_pair_articulation.numpy()[0]), 0)
        self.assertTrue(any(sample[7] > 0 for sample in local), "free body generated no matrix-free rows")
        for sample in local:
            if sample[7] > 0:
                self.assertEqual(sample[5], PGS_LOCAL_SOLVE_OWNER_GENERAL, "matrix-free rows left the general owner")
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owners_stay_general_with_contact_torsion(self):
        """Keep torsion-eligible worlds on the general owner, which solves the appended torsion row."""
        model = _build_mixed_response_model("cuda:0", friction=0.7)
        solvers = {}
        with kernel_overrides({"hinv_jt_kernel": "par_row"}):
            for radius in (0.0, 0.01):
                set_torsion_radius(model, radius)
                solvers[radius] = SolverSteel(
                    model,
                    solve_mode="matrix_free",
                    enable_joint_limits=True,
                    joint_limit_activation_gap=0.0,
                    iterations=8,
                    articulation_constraint_max=32,
                    free_body_constraint_max=32,
                )
        self.assertTrue(solvers[0.0]._local_internal_fast_path)
        self.assertTrue(solvers[0.01]._contact_torsion_enabled)
        self.assertFalse(solvers[0.01]._local_internal_fast_path)


if __name__ == "__main__":
    unittest.main()
