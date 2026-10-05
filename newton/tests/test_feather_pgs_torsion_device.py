# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare device torsion preparation against the retained host oracle."""

import unittest
from inspect import signature
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.solvers.feather_pgs import contact_torsion as host
from newton._src.solvers.feather_pgs import solver_feather_pgs
from newton._src.solvers.feather_pgs.contact_torsion_device import (
    DeviceTorsionPreparation,
    _host_dot3,
    _host_transform_point,
    enable_device_torsion,
)
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from newton.tests.test_feather_pgs_connect import _build_four_bar, _loop_anchor_gap
from newton.tests.test_feather_pgs_contact_torsion import PATCH_OPTIONS, fixture


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def _rounding_samples(
    poses: wp.array[wp.transform],
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    transformed: wp.array[wp.vec3],
    dotted: wp.array[float],
):
    i = wp.tid()
    transformed[i] = _host_transform_point(poses[i], points[i])
    dotted[i] = _host_dot3(points[i], directions[i])


def synthetic_fixture(*, worlds=3, witnesses=5, patches=True, seed=4, device="cpu", padding=7):
    """Create ordered/reordered contact worlds with unequal articulation DOFs."""
    rng = np.random.default_rng(seed)
    n = worlds * witnesses
    capacity = n + padding
    dense = witnesses * 3 + 8
    records = [(world, local) for world in range(worlds) for local in range(witnesses)]
    rng.shuffle(records)

    def arr(value, dtype=None):
        return wp.array(np.asarray(value), dtype=dtype, device=device)

    def zero(shape, dtype=float):
        return wp.zeros(shape, dtype=dtype, device=device)

    def full(shape, value, dtype=int):
        return wp.full(shape, value, dtype=dtype, device=device)

    count = np.full(worlds, witnesses * 3, np.int32)
    row_types = np.zeros((worlds, dense), np.int32)
    parents = np.full((worlds, dense), -1, np.int32)
    mu = np.zeros((worlds, dense), np.float32)
    world_ids = np.zeros(capacity, np.int32)
    slots = np.full(capacity, -1, np.int32)
    anchors = np.zeros(capacity, np.int32)
    sa = np.zeros(capacity, np.int32)
    sb = np.zeros(capacity, np.int32)
    owner = np.full(capacity, -1, np.int32)
    normals = np.zeros((capacity, 3), np.float32)
    points = np.zeros((capacity, 3), np.float32)
    region_seeds = {}
    for c, (world, local) in enumerate(records):
        slot = local * 3
        region = local % 2
        region_seeds.setdefault((world, region), c)
        world_ids[c], slots[c], anchors[c] = world, slot, 3
        sa[c], sb[c] = world * 4 + region, world * 4 + 2 + region
        owner[c] = region_seeds[world, region]
        normals[c] = (0.0, 0.0, -1.0)
        points[c] = (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.0)
        row_types[world, slot : slot + 3] = (
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
        )
        parents[world, slot : slot + 3] = (slot, slot, slot)
        mu[world, slot + 1 : slot + 3] = 0.7
    if patches:
        for world in range(worlds):
            for region in range(2):
                members = [c for c, (w, local) in enumerate(records) if w == world and local % 2 == region]
                for i, c in enumerate(members):
                    parents[world, slots[c]] = slots[members[(i + 1) % len(members)]]
                    mu[world, slots[c] + 1 : slots[c] + 3] /= len(members)
    shape_body = np.repeat(np.arange(worlds * 2, dtype=np.int32), 2)
    response_sizes = np.tile([6, 3], worlds).astype(np.int32)
    starts = np.concatenate(([0], np.cumsum(response_sizes))).astype(np.int32)
    total_dofs = int(starts[-1])
    motion = rng.normal(size=(total_dofs, 6)).astype(np.float32)
    body_velocity = rng.normal(size=(worlds * 2, 6)).astype(np.float32)
    prescribed = np.zeros(worlds * 2, np.int32)
    if worlds > 1:
        prescribed[-1] = 1
    model = SimpleNamespace(
        device=wp.get_device(device),
        shape_count=worlds * 4,
        shape_body=arr(shape_body, int),
        shape_type=full(worlds * 4, int(GeoType.BOX)),
        joint_ancestor=full(worlds * 2, -1),
        joint_qd_start=arr(starts, int),
    )
    state = SimpleNamespace(
        body_q=arr(np.tile([0, 0, 0, 0, 0, 0, 1], (worlds * 2, 1)).astype(np.float32), wp.transform)
    )
    augmented = SimpleNamespace(
        joint_S_s=arr(motion, wp.spatial_vector), body_v_s=arr(body_velocity, wp.spatial_vector)
    )
    contacts = SimpleNamespace(
        rigid_contact_count=arr([n], int),
        rigid_contact_max=capacity,
        rigid_contact_stiffness=None,
        rigid_contact_shape0=arr(sa, int),
        rigid_contact_shape1=arr(sb, int),
        rigid_contact_normal=arr(normals, wp.vec3),
        rigid_contact_point0=arr(points, wp.vec3),
        rigid_contact_point1=arr(points, wp.vec3),
        rigid_contact_margin0=zero(capacity),
        rigid_contact_margin1=zero(capacity),
    )
    solver = SimpleNamespace(
        model=model,
        world_count=worlds,
        dense_max_constraints=dense,
        _max_contacts_alloc=capacity,
        _contact_torsion_shape_set=None,
        _friction_anchors_enabled=patches,
        _friction_patches=SimpleNamespace(current=SimpleNamespace(owner=arr(owner, int))),
        contact_path=zero(capacity, int),
        contact_slot=arr(slots, int),
        contact_world=arr(world_ids, int),
        contact_slots_needed=arr(anchors, int),
        constraint_count=arr(count, int),
        slot_counter=arr(count, int),
        row_type=arr(row_types, int),
        row_parent=arr(parents, int),
        row_mu=arr(mu, float),
        _row_dropped_dense=zero(worlds, int),
        _contact_torsion_group=full((worlds, dense), -1),
        pgs_cfm=0.003,
        contact_torsion_radius=0.01,
        body_to_joint=arr(np.arange(worlds * 2, dtype=np.int32), int),
        body_to_articulation=arr(np.arange(worlds * 2, dtype=np.int32), int),
        articulation_dof_start=arr(starts[:-1], int),
        articulation_response_dof_count=arr(response_sizes, int),
        art_group_idx=arr(np.repeat(np.arange(worlds, dtype=np.int32), 2), int),
        art_to_world=arr(np.repeat(np.arange(worlds, dtype=np.int32), 2), int),
        _prescribed_articulation=arr(prescribed, int),
        group_to_art={size: arr(np.flatnonzero(response_sizes == size).astype(np.int32), int) for size in (3, 6)},
        J_by_size={size: zero((worlds, dense, size)) for size in (3, 6)},
    )
    for name in ("row_beta", "row_cfm", "phi", "target_velocity", "row_restitution"):
        setattr(solver, name, zero((worlds, dense)))
    return solver, state, augmented, contacts


def run_oracle(solver, state, augmented, contacts):
    """Invoke only the host preparation oracle, not a CPU solver backend."""
    with patch.object(host, "validate_torsion_step", return_value=None):
        host.prepare_torsion_rows(solver, state, augmented, contacts)


class TestDeviceTorsionPreparation(unittest.TestCase):
    """Exercise portable preparation kernels independently of a full CUDA solve."""

    def test_public_device_option_is_default_off(self):
        """Keep public device preparation opt-in and expose explicit graph validation."""
        self.assertIs(
            signature(solver_feather_pgs.SolverFeatherPGS).parameters["contact_torsion_device"].default, False
        )
        self.assertTrue(callable(getattr(solver_feather_pgs.SolverFeatherPGS, "prepare_contact_torsion_capture", None)))
        self.assertTrue(callable(getattr(solver_feather_pgs.SolverFeatherPGS, "validate_contact_torsion", None)))

    def compare(self, *, device="cpu", modify=None, **options):
        """Compare every row field, group membership and size-specific Jacobian."""
        expected, state, augmented, contacts = synthetic_fixture(device=device, **options)
        actual, state2, augmented2, contacts2 = synthetic_fixture(device=device, **options)
        if modify is not None:
            modify(expected, contacts)
            modify(actual, contacts2)
        run_oracle(expected, state, augmented, contacts)
        preparer = DeviceTorsionPreparation(actual)
        preparer.prepare(state2, augmented2, contacts2)
        for name in (
            "constraint_count",
            "slot_counter",
            "row_type",
            "row_parent",
            "row_mu",
            "row_beta",
            "row_cfm",
            "phi",
            "target_velocity",
            "row_restitution",
            "_contact_torsion_group",
        ):
            np.testing.assert_allclose(
                getattr(actual, name).numpy(), getattr(expected, name).numpy(), rtol=2e-6, atol=2e-6, err_msg=name
            )
        for size in expected.J_by_size:
            np.testing.assert_allclose(
                actual.J_by_size[size].numpy(),
                expected.J_by_size[size].numpy(),
                rtol=2e-6,
                atol=2e-6,
                err_msg=f"J{size}",
            )
        stats = preparer.read_stats()
        self.assertEqual(stats["rows"], expected._torsion_stats["rows"])
        actual_groups = sorted(stats["groups"], key=lambda group: (group["world"], group["row"]))
        expected_groups = sorted(expected._torsion_stats["groups"], key=lambda group: (group["world"], group["row"]))
        for actual_group, expected_group in zip(actual_groups, expected_groups, strict=True):
            for name in expected_group:
                np.testing.assert_allclose(actual_group[name], expected_group[name], rtol=2e-6, atol=2e-6)
        return actual, preparer

    def test_randomized_patch_and_point_oracle(self):
        """Preserve contact-order semantics across mixed shapes and unequal DOFs."""
        for patches in (False, True):
            for seed in range(5):
                with self.subTest(patches=patches, seed=seed):
                    self.compare(patches=patches, seed=seed)

    def test_host_math_rounding(self):
        """Match explicit dot and quaternion-transform rounding on CPU and CUDA."""
        rng = np.random.default_rng(84)
        poses = rng.normal(size=(1024, 7)).astype(np.float32)
        poses[:, 3:] /= np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
        points = rng.normal(size=(1024, 3)).astype(np.float32)
        directions = rng.normal(size=points.shape).astype(np.float32)
        expected_points = np.array([host._transform_point(q, p) for q, p in zip(poses, points, strict=True)])
        # Independent scalar reference: float32 products, sequential float64
        # sum, one float32 output rounding. Do not consult platform BLAS.
        expected_dot = np.array(
            [sum(float(np.float32(a[i] * b[i])) for i in range(3)) for a, b in zip(points, directions, strict=True)],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            np.array([host._dot3(a, b) for a, b in zip(points, directions, strict=True)]), expected_dot
        )
        for device in ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]:
            transformed = wp.empty(1024, dtype=wp.vec3, device=device)
            dotted = wp.empty(1024, dtype=float, device=device)
            wp.launch(
                _rounding_samples,
                dim=1024,
                inputs=[
                    wp.array(poses, dtype=wp.transform, device=device),
                    wp.array(points, dtype=wp.vec3, device=device),
                    wp.array(directions, dtype=wp.vec3, device=device),
                    transformed,
                    dotted,
                ],
                device=device,
            )
            np.testing.assert_array_equal(transformed.numpy(), expected_points)
            np.testing.assert_array_equal(dotted.numpy(), expected_dot)

    def test_grouping_and_jacobians_do_not_depend_on_blas(self):
        """Build host groups and rows without platform-dependent NumPy dot."""
        for patches in (False, True):
            with self.subTest(patches=patches):
                with patch.object(host.np, "dot", side_effect=AssertionError("BLAS dot must not decide torsion rows")):
                    self.compare(patches=patches)

    def test_empty_and_larger_preallocated_buffer(self):
        """Keep zero contacts valid and distinguish configured from input capacity."""
        solver, state, augmented, contacts = synthetic_fixture()
        solver._max_contacts_alloc += 5
        device = DeviceTorsionPreparation(solver)
        device.prepare(state, augmented, contacts)
        self.assertEqual(int(np.count_nonzero(device.work.spin_row.numpy() >= 0)), 6)
        contacts.rigid_contact_count.zero_()
        device.prepare(state, augmented, contacts)
        self.assertTrue(np.all(solver._contact_torsion_group.numpy() == -1))

    def test_ring_and_capacity_errors_are_latched(self):
        """Reject incomplete rings and capacity failures before row writes."""
        for failure in ("ring", "dense", "input", "spin"):
            solver, state, augmented, contacts = synthetic_fixture()
            before = solver.row_type.numpy().copy()
            if failure == "ring":
                parents = solver.row_parent.numpy()
                parents[0, 0] = solver.dense_max_constraints - 1
                solver.row_parent.assign(parents)
            elif failure == "dense":
                solver._row_dropped_dense.fill_(1)
            elif failure == "input":
                contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
            else:
                solver.constraint_count.fill_(solver.dense_max_constraints)
            device = DeviceTorsionPreparation(solver)
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                device.prepare(state, augmented, contacts)
            np.testing.assert_array_equal(before, solver.row_type.numpy())
            self.assertNotEqual(int(device.work.status.numpy()[0]), 0)
            with self.assertRaises(RuntimeError):
                device.prepare(state, augmented, None)

    def test_patch_segments_match_world_scan(self):
        """Match the per-world patch scan exactly under selection, rejection and negative owners."""
        devices = ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]
        for device in devices:
            for seed in range(8):
                results = []
                for patch_segments in (False, True):
                    solver, state, augmented, contacts = synthetic_fixture(
                        worlds=4, witnesses=12, seed=seed, device=device
                    )
                    rng = np.random.default_rng(seed)
                    solver._contact_torsion_shape_set = set(np.flatnonzero(rng.random(solver.model.shape_count) < 0.6))
                    anchors = solver.contact_slots_needed.numpy()
                    anchors[rng.random(anchors.size) < 0.3] = 1
                    solver.contact_slots_needed.assign(anchors)
                    points = contacts.rigid_contact_point0.numpy()
                    points[rng.random(len(points)) < 0.3, 2] = 0.002
                    contacts.rigid_contact_point0.assign(points)
                    if seed % 2:
                        owner = solver._friction_patches.current.owner.numpy()
                        owner[owner == owner.max()] = -1
                        solver._friction_patches.current.owner.assign(owner)
                    preparer = DeviceTorsionPreparation(solver, patch_segments=patch_segments)
                    try:
                        preparer.prepare(state, augmented, contacts)
                        error = None
                    except RuntimeError as exc:
                        error = str(exc)
                    names = ("constraint_count", "row_type", "row_parent", "row_mu", "target_velocity")
                    arrays = [getattr(solver, name).numpy() for name in (*names, "_contact_torsion_group")]
                    arrays += [preparer.work.status.numpy(), preparer.work.spin_row.numpy()]
                    results.append((error, arrays))
                with self.subTest(device=device, seed=seed):
                    self.assertEqual(results[0][0], results[1][0])
                    for legacy, segmented in zip(results[0][1], results[1][1], strict=True):
                        np.testing.assert_array_equal(segmented, legacy)

    def test_patch_eligibility_and_point_cluster_edges(self):
        """Match mixed-anchor patches, selector rejection and nontransitive clusters."""
        for scenario in ("nonanchor", "positive_gap", "selector", "nontransitive"):

            def modify(solver, contacts, scenario=scenario):
                if scenario == "nonanchor":
                    anchors = solver.contact_slots_needed.numpy()
                    anchors[::2] = 1
                    solver.contact_slots_needed.assign(anchors)
                elif scenario == "positive_gap":
                    points = contacts.rigid_contact_point0.numpy()
                    points[:, 2] = 0.002
                    contacts.rigid_contact_point0.assign(points)
                elif scenario == "selector":
                    solver._contact_torsion_shape_set = {0}
                else:
                    # Adjacent normals match, but the endpoints do not. The
                    # greedy oracle tests every member, not only its seed.
                    angles = (np.arange(contacts.rigid_contact_max) % 3) * 0.035
                    normals = np.column_stack((np.sin(angles), np.zeros_like(angles), -np.cos(angles)))
                    contacts.rigid_contact_normal.assign(normals.astype(np.float32))
                    contacts.rigid_contact_point0.zero_()
                    contacts.rigid_contact_point1.zero_()

            for patches in (False, True):
                with self.subTest(scenario=scenario, patches=patches):
                    self.compare(patches=patches, modify=modify)

    def test_touching_threshold_preserves_host_rounding(self):
        """Keep admission identical when cancelling dot products straddle the gap gate."""

        def modify(_solver, contacts):
            rng = np.random.default_rng(18)
            shape = (contacts.rigid_contact_max, 3)
            normal = rng.normal(size=shape).astype(np.float32)
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            point = rng.normal(size=shape).astype(np.float32) * 0.1
            point -= np.sum(normal * point, axis=1, keepdims=True) * normal
            point += np.float32(1e-5) * normal
            contacts.rigid_contact_normal.assign(-normal)
            contacts.rigid_contact_point0.assign(point)
            contacts.rigid_contact_point1.zero_()

        for device in ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]:
            for patches in (False, True):
                with self.subTest(device=device, patches=patches):
                    self.compare(device=device, worlds=512, witnesses=1, patches=patches, modify=modify)

    def test_rejected_region_and_normal_clusters(self):
        """Preserve whole-region rejection and all-member coplanar clustering."""
        for patches in (False, True):
            pairs = [synthetic_fixture(patches=patches) for _ in range(2)]
            for solver, _state, _augmented, contacts in pairs:
                types = solver.model.shape_type.numpy()
                types[0] = int(GeoType.MESH)
                solver.model.shape_type.assign(types)
                normals = contacts.rigid_contact_normal.numpy()
                normals[3] = (1, 0, 0)
                contacts.rigid_contact_normal.assign(normals)
                points = contacts.rigid_contact_point0.numpy()
                points[6, 2] = 0.001
                contacts.rigid_contact_point0.assign(points)
            expected, state, augmented, contacts = pairs[0]
            actual, state2, augmented2, contacts2 = pairs[1]
            run_oracle(expected, state, augmented, contacts)
            device = DeviceTorsionPreparation(actual)
            device.prepare(state2, augmented2, contacts2)
            np.testing.assert_array_equal(
                actual._contact_torsion_group.numpy(), expected._contact_torsion_group.numpy()
            )
            np.testing.assert_array_equal(actual.row_parent.numpy(), expected.row_parent.numpy())

    def test_more_than_host_global_capacity(self):
        """Retain over 4096 witnesses against a decomposed host grouping oracle."""
        solver, state, augmented, contacts = synthetic_fixture(worlds=822, witnesses=5)
        with self.assertRaisesRegex(RuntimeError, "host grouping capacity"):
            host._contact_groups(solver, state, contacts)
        worlds = solver.contact_world.numpy()
        expected = set()
        for first, last in ((0, 411), (411, 822)):
            solver.contact_path.assign(np.where((worlds >= first) & (worlds < last), 0, -1).astype(np.int32))
            for group in host._contact_groups(solver, state, contacts):
                expected.add((group[0].world, tuple(sorted(w.slot for w in group))))
        solver.contact_path.zero_()
        device = DeviceTorsionPreparation(solver)
        device.prepare(state, augmented, contacts)
        membership = solver._contact_torsion_group.numpy()
        actual = set()
        for world in range(solver.world_count):
            for spin in np.unique(membership[world]):
                if spin >= 0:
                    actual.add((world, tuple(np.flatnonzero(membership[world] == spin))))
        self.assertEqual(expected, actual)
        self.assertEqual(len(actual), 1644)

    def test_deferred_failure_is_not_partial_success(self):
        """Latch failure, suppress row solves and restore public dynamic state."""
        solver, state, augmented, contacts = synthetic_fixture()
        out = SimpleNamespace(body_q=wp.clone(state.body_q))
        device = DeviceTorsionPreparation(solver, deferred_errors=True)
        device.begin_step(state, out)
        out.body_q.zero_()
        solver._row_dropped_dense.fill_(1)
        device.prepare(state, augmented, contacts)
        device.end_step(out)
        np.testing.assert_array_equal(out.body_q.numpy(), state.body_q.numpy())
        self.assertTrue(np.all(solver.constraint_count.numpy() == 0))
        self.assertTrue(np.all(solver._contact_torsion_group.numpy() == -1))
        with self.assertRaisesRegex(RuntimeError, "Dense contact row overflow"):
            device.validate()
        solver._row_dropped_dense.zero_()
        device.prepare(state, augmented, contacts)
        with self.assertRaises(RuntimeError):
            device.validate()

    def test_deferred_preparation_has_no_host_array_transfers(self):
        """Reject accidental full or scalar host transfers in deferred preparation."""
        solver, state, augmented, contacts = synthetic_fixture()
        device = DeviceTorsionPreparation(solver, deferred_errors=True)
        with (
            patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected device-to-host transfer")),
            patch.object(wp.array, "assign", side_effect=AssertionError("Unexpected host assignment")),
        ):
            device.prepare(state, augmented, contacts)
        device.validate()

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_oracle(self):
        """Repeat the full row/Jacobian oracle comparison on CUDA."""
        for patches in (False, True):
            for seed in range(3):
                self.compare(device="cuda:0", patches=patches, seed=seed)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_cancelling_angular_projection_matches_host_rounding(self):
        """Preserve float32 host rounding for nearly cancelling angular rows."""
        pairs = [synthetic_fixture(device="cuda:0") for _ in range(2)]
        normal = np.array([0.8962938785552979, 0.44297733902931213, 0.020699456334114075], dtype=np.float32)
        angular = np.array(
            [
                [0.16465608775615692, -0.29055461287498474, -0.9425849914550781],
                [-0.411504328250885, 0.8481746912002563, -0.33356380462646484],
                [0.41150417923927307, -0.8481748104095459, 0.3335646986961365],
                [0.4004249572753906, 0.6479430198669434, -0.647942841053009],
            ],
            dtype=np.float32,
        )
        for _solver, _state, augmented, contacts in pairs:
            contacts.rigid_contact_normal.assign(np.tile(-normal, (contacts.rigid_contact_max, 1)))
            motion = augmented.joint_S_s.numpy()
            motion[:4, 3:] = angular
            augmented.joint_S_s.assign(motion)
        expected, state, augmented, contacts = pairs[0]
        actual, state2, augmented2, contacts2 = pairs[1]
        run_oracle(expected, state, augmented, contacts)
        DeviceTorsionPreparation(actual).prepare(state2, augmented2, contacts2)
        np.testing.assert_array_equal(actual.J_by_size[6].numpy()[0, :, :4], expected.J_by_size[6].numpy()[0, :, :4])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_loaded_closed_linkage(self):
        """Preserve loaded loop-closure behavior with active spin rows in both profiles."""
        for velocity, regularization in ((16, 0.0), (0, 0.01)):
            results = []
            for device_preparation in (False, True):
                builder = _build_four_bar()
                root_half = float(np.sqrt(0.5))
                builder.add_shape_plane(plane=(0.0, root_half, root_half, -0.362 * root_half))
                model = builder.finalize(device="cuda:0")
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    pgs_iterations=64,
                    pgs_velocity_iterations=velocity,
                    pgs_contact_regularization=regularization,
                    contact_torsion_radius=0.01,
                    enable_bilateral_preelimination=False,
                    dense_max_constraints=128,
                    mf_max_constraints=64,
                    row_watermark=True,
                    **PATCH_OPTIONS,
                )
                self.assertGreater(solver._connect_count, 0)
                if device_preparation:
                    enable_device_torsion(solver)
                a, z = model.state(), model.state()
                newton.eval_fk(model, a.joint_q, a.joint_qd, a)
                control = model.control()
                target = model.joint_target_q.numpy()
                target[0] = 0.03
                control.joint_target_q.assign(target)
                pipeline = newton.CollisionPipeline(
                    model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
                )
                contacts = pipeline.contacts()
                spins = 0
                load = 0.0
                for _ in range(32):
                    a.clear_forces()
                    pipeline.collide(a, contacts)
                    solver.step(a, z, control, contacts, 0.0025)
                    types = solver.row_type.numpy()
                    spins = max(spins, int(np.count_nonzero(types == PGS_CONSTRAINT_TYPE_TORSION)))
                    load = max(
                        load, float(solver.impulses.numpy()[types == PGS_CONSTRAINT_TYPE_CONTACT].max(initial=0))
                    )
                    a, z = z, a
                self.assertGreater(spins, 0)
                self.assertGreater(load, 1e-8)
                self.assertTrue(np.isfinite(a.body_q.numpy()).all())
                results.append((a.body_q.numpy(), a.joint_q.numpy(), a.joint_qd.numpy(), _loop_anchor_gap(model, a)))
            for expected, actual in zip(*results, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_solver_profiles(self):
        """Match the host's solved state and impulses for both production profiles."""

        def device_prepare(solver, state, augmented, contacts):
            enable_device_torsion(solver).prepare(state, augmented, contacts)

        for patches in (False, True):
            for velocity, regularization in ((16, 0.0), (0, 0.01)):
                options = {
                    "pgs_iterations": 16,
                    "pgs_velocity_iterations": velocity,
                    "pgs_contact_regularization": regularization,
                    "enable_bilateral_preelimination": False,
                    "sliding": 0.03,
                    "spin": 0.4,
                }
                if patches:
                    options.update(PATCH_OPTIONS)
                expected, *_ = fixture(0.01, **options)
                with patch.object(solver_feather_pgs, "prepare_torsion_rows", device_prepare):
                    actual, *_ = fixture(0.01, **options)
                for key in expected:
                    np.testing.assert_allclose(
                        actual[key], expected[key], rtol=3e-5, atol=3e-6, err_msg=f"{patches=} {velocity=} {key=}"
                    )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_preparation_graph_changes_and_failure(self):
        """Replay changing groups and fail-stop invalid rings without host reads."""
        solver, state, augmented, contacts = synthetic_fixture(device="cuda:0")
        device = DeviceTorsionPreparation(solver, deferred_errors=True)
        initial_count = wp.clone(solver.constraint_count)
        initial_type = wp.clone(solver.row_type)
        initial_parent = wp.clone(solver.row_parent)
        output = SimpleNamespace(body_q=wp.clone(state.body_q))
        device.begin_step(state, output)

        def operation():
            wp.copy(solver.constraint_count, initial_count)
            wp.copy(solver.row_type, initial_type)
            wp.copy(solver.row_parent, initial_parent)
            device.begin_step(state, output)
            output.body_q.zero_()
            device.prepare(state, augmented, contacts)
            device.end_step(output)

        operation()
        with wp.ScopedCapture(device="cuda:0") as capture:
            operation()
        for count in (0, 15, 0, 15):
            contacts.rigid_contact_count.fill_(count)
            wp.capture_launch(capture.graph)
            device.validate()
            self.assertEqual(np.count_nonzero(device.work.spin_row.numpy() >= 0), 0 if count == 0 else 6)
        bad = initial_parent.numpy()
        bad[0, 0] = solver.dense_max_constraints - 1
        initial_parent.assign(bad)
        wp.capture_launch(capture.graph)
        with self.assertRaisesRegex(RuntimeError, "load ring"):
            device.validate()
        np.testing.assert_array_equal(output.body_q.numpy(), state.body_q.numpy())
        self.assertTrue(np.all(solver.constraint_count.numpy() == 0))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_captured_capacity_errors(self):
        """Graph replay must not turn an input, dense or spin overflow into success."""
        for failure in ("input", "dense", "spin"):
            solver, state, augmented, contacts = synthetic_fixture(device="cuda:0")
            device = DeviceTorsionPreparation(solver, deferred_errors=True)
            output = SimpleNamespace(body_q=wp.clone(state.body_q))
            initial_count = wp.clone(solver.constraint_count)
            device.begin_step(state, output)

            def operation(
                solver=solver,
                state=state,
                augmented=augmented,
                contacts=contacts,
                device=device,
                output=output,
                initial_count=initial_count,
            ):
                wp.copy(solver.constraint_count, initial_count)
                device.begin_step(state, output)
                output.body_q.zero_()
                device.prepare(state, augmented, contacts)
                device.end_step(output)

            operation()
            with wp.ScopedCapture(device="cuda:0") as capture:
                operation()
            if failure == "input":
                contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
            elif failure == "dense":
                solver._row_dropped_dense.fill_(1)
            else:
                initial_count.fill_(solver.dense_max_constraints)
            for _ in range(2):
                wp.capture_launch(capture.graph)
                with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                    device.validate()
                np.testing.assert_array_equal(output.body_q.numpy(), state.body_q.numpy())
                self.assertTrue(np.all(solver.constraint_count.numpy() == 0))
                self.assertTrue(np.all(solver._contact_torsion_group.numpy() == -1))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_public_api_noop_and_fail_stop(self):
        """No radius means no new state; active host torsion cannot be captured."""
        _, plain, model, state, _ = fixture(0.0, contact_torsion_device=True, pgs_iterations=16)
        plain.prepare_contact_torsion_capture(state, model.state())
        plain.validate_contact_torsion()
        self.assertIsNone(getattr(plain, "_device_torsion", None))

        _, reference, model, state, _ = fixture(0.01, pgs_iterations=16)
        with self.assertRaisesRegex(RuntimeError, "requires contact_torsion_device"):
            reference.prepare_contact_torsion_capture(state, model.state())

        _, solver, model, state, contacts = fixture(0.01, contact_torsion_device=True, pgs_iterations=16)
        output = model.state()
        solver.prepare_contact_torsion_capture(state, output)
        solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "load ring"):
                solver.step(state, output, model.control(), contacts, 0.0025)
            with self.assertRaisesRegex(RuntimeError, "load ring"):
                solver.validate_contact_torsion()
            np.testing.assert_array_equal(output.joint_q.numpy(), state.joint_q.numpy())
            np.testing.assert_array_equal(output.joint_qd.numpy(), state.joint_qd.numpy())

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_full_solver_graph(self):
        """Match captured and eager solves through the default-off public device API."""

        for velocity, regularization in ((16, 0.0), (0, 0.01)):
            results = []
            for captured in (False, True):
                _, solver, model, a, contacts = fixture(
                    0.01,
                    contact_torsion_device=True,
                    pgs_iterations=16,
                    pgs_velocity_iterations=velocity,
                    pgs_contact_regularization=regularization,
                    enable_bilateral_preelimination=False,
                    sliding=0.01,
                    spin=0.4,
                    **PATCH_OPTIONS,
                )
                z = model.state()
                solver.prepare_contact_torsion_capture(a, z)
                control = model.control()
                pipeline = newton.CollisionPipeline(
                    model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
                )

                def pair(a=a, z=z, pipeline=pipeline, solver=solver, control=control, contacts=contacts):
                    a.clear_forces()
                    pipeline.collide(a, contacts)
                    solver.step(a, z, control, contacts, 0.0025)
                    z.clear_forces()
                    pipeline.collide(z, contacts)
                    solver.step(z, a, control, contacts, 0.0025)

                pair()
                if captured:
                    with wp.ScopedCapture(device=model.device) as capture:
                        solver.seed_double_buffer_events()
                        pair()
                    for _ in range(5):
                        wp.capture_launch(capture.graph)
                        solver.validate_contact_torsion()
                else:
                    for _ in range(5):
                        pair()
                        solver.validate_contact_torsion()
                results.append((a.joint_q.numpy(), a.joint_qd.numpy(), a.body_q.numpy()))
            for eager, graph in zip(*results, strict=True):
                np.testing.assert_allclose(graph, eager, rtol=3e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
