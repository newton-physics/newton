# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.flags import MeshSignMethod
from newton._src.geometry.sdf_contact import (
    _create_sdf_contact_funcs,
    _recover_flat_texture_contact,
    _sdf_rsqrt_rn,
    compute_block_counts_from_weights,
    compute_mesh_boundary_flags,
    mesh_sdf_contact_search_precision,
)
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.utils.heightfield import HeightfieldData
from newton.sensors import SensorContact
from newton.tests.unittest_utils import (
    MUJOCO_MODULES,
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
    requires,
)


@wp.func
def _two_sphere_distance(sdf: TextureSDFData, p: wp.vec3) -> float:
    scale = sdf.voxel_radius
    return wp.min(wp.length(p) - 0.2 * scale, wp.length(p - wp.vec3(0.55 * scale, 0.0, 0.0)) - 0.05 * scale)


@wp.func
def _two_sphere_pair(sdf: TextureSDFData, p: wp.vec3, q: wp.vec3) -> wp.vec2:
    return wp.vec2(_two_sphere_distance(sdf, p), _two_sphere_distance(sdf, q))


def _make_nonconvex_edge_kernel(paired):
    search = _create_sdf_contact_funcs(False, paired, _two_sphere_distance, _two_sphere_pair)

    @wp.kernel(enable_backward=False)
    def kernel(
        scales: wp.array[float],
        distances: wp.array[float],
        endpoints: wp.array[int],
        boundary: wp.array[wp.uint8],
        elevation: wp.array[float],
    ):
        i = wp.tid()
        scale = scales[i // 2]
        a = wp.vec3(0.0)
        b = wp.vec3(scale, 0.0, 0.0)
        if i % 2 == 1:
            a, b = b, a
        sdf = TextureSDFData()
        sdf.voxel_radius = scale
        distance, _point, endpoint = search(
            sdf,
            wp.uint64(0),
            a,
            b,
            _two_sphere_distance(sdf, (a + b) * 0.5),
            False,
            0,
            boundary,
            -1,
            False,
            HeightfieldData(),
            elevation,
            0.0,
        )
        distances[i] = distance
        endpoints[i] = endpoint

    return kernel


_nonconvex_edge_scalar = _make_nonconvex_edge_kernel(False)
_nonconvex_edge_paired = _make_nonconvex_edge_kernel(True)


def test_mesh_mesh_contained_bvh_contacts(test, device):
    """Retain deep interior edges after bounded closest-point misses."""
    with wp.ScopedDevice(device):
        outer = newton.Mesh.create_box(1.0, 1.0, 1.0)
        inner = newton.Mesh.create_box(0.02, 0.02, 0.02)
        for scale in (0.01, 1.0, 100.0):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, margin=0.0, gap=0.0)
            # Exercise the scaled shape-local bounds under a world transform.
            rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.37)
            outer_pose = wp.transform(wp.vec3(3.0 * scale, -2.0 * scale, scale), rotation)
            inner_pose = wp.transform_multiply(outer_pose, wp.transform(wp.vec3(0.2 * scale, 0.0, 0.0)))
            builder.add_shape_mesh(-1, mesh=outer, scale=(scale, 2.0 * scale, 3.0 * scale), xform=outer_pose, cfg=cfg)
            body = builder.add_body(xform=inner_pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.add_shape_mesh(body, mesh=inner, scale=(scale, scale, scale), cfg=cfg)
            model = builder.finalize()
            state = model.state()
            for reduced in (False, True):
                with test.subTest(scale=scale, reduced=reduced):
                    pipeline = newton.CollisionPipeline(model, reduce_contacts=reduced)
                    contacts = pipeline.contacts()
                    state.body_q.assign([inner_pose])
                    pipeline.collide(state, contacts)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    test.assertGreater(count, 0)
                    # The +x surface is the nearest exit, about 0.8 scaled units away.
                    normals = contacts.rigid_contact_normal.numpy()[:count]
                    test.assertTrue(np.all(np.isfinite(normals)))
                    exit_direction = np.array(wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0)))
                    np.testing.assert_allclose(normals @ exit_direction, 1.0, atol=2.0e-5)
                    points0 = contacts.rigid_contact_point0.numpy()[:count]
                    points1 = np.array(
                        [
                            wp.transform_point(inner_pose, wp.vec3(p))
                            for p in contacts.rigid_contact_point1.numpy()[:count]
                        ]
                    )
                    depths = -np.sum((points1 - points0) * normals, axis=1) / scale
                    test.assertTrue(np.all((depths > 0.77) & (depths < 0.83)), depths)
                    # Reuse the pipeline after moving outside the shape bounds.
                    exterior = wp.transform_multiply(outer_pose, wp.transform(wp.vec3(4.0 * scale, 0.0, 0.0)))
                    state.body_q.assign([exterior])
                    pipeline.collide(state, contacts)
                    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


@wp.kernel(enable_backward=False)
def _mesh_sdf_contact_search_precision_kernel(out: wp.array[wp.float32]):
    out[0] = mesh_sdf_contact_search_precision(0.0, 1.0, 0.001, True)
    out[1] = mesh_sdf_contact_search_precision(0.01, 1.0, 0.001, True)
    out[2] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.1, True)
    out[3] = mesh_sdf_contact_search_precision(0.01, 2.0, 0.001, False)


@wp.kernel(enable_backward=False)
def _sdf_rsqrt_rn_kernel(values: wp.array[wp.float32], out: wp.array[wp.float32]):
    tid = wp.tid()
    out[tid] = _sdf_rsqrt_rn(values[tid])


def test_mesh_mesh_sdf_surface_witnesses(test, device, box_sdf, *, probe_sdf=False):
    """Keep SDF-based mesh-mesh witnesses on both surfaces with reduction and speculation."""
    # Parallel faces give exact witness planes. A common shift of both witnesses
    # preserves their separation, so checking only contact distance misses the bug.
    cases = (
        ("direct", False, -0.1),
        ("predictive", False, 0.1),
        ("reduced", True, -0.1),
        ("reduced_predictive", True, 0.1),
    )
    with wp.ScopedDevice(device):
        if probe_sdf:
            # A closed probe supplies the second SDF and exercises texture-only kernels.
            probe = newton.Mesh.create_box(0.2, 0.2, 0.2)
            probe.build_sdf(max_resolution=32, device=device)
        else:
            probe = newton.Mesh(
                np.array([[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32),
                np.array([0, 2, 1], dtype=np.int32),
                compute_inertia=False,
            )
        probe_half_height = 0.2 if probe_sdf else 0.0
        box = newton.Mesh.create_box(1.0, 1.0, 0.5)
        if box_sdf:
            box.build_sdf(max_resolution=32, device=device)
        # Allow texture quantization error, well below the original 0.05 m shift.
        sdf_tolerance = 1.0e-3 if box_sdf else 2.0e-5
        for probe_first in (True, False):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, margin=0.0, gap=0.0)
            for is_probe in (probe_first, not probe_first):
                builder.add_shape_mesh(
                    body if is_probe else -1,
                    mesh=probe if is_probe else box,
                    cfg=cfg,
                )
            # Each shape order shares one model across all collision-only cases.
            model = builder.finalize(device=device)
            state = model.state()
            state.body_qd.assign([[0.0, 0.0, -2.0, 0.0, 0.0, 0.0]])
            for case, reduce_contacts, distance in cases:
                with test.subTest(probe_first=probe_first, case=case):
                    state.body_q.assign([wp.transform(wp.vec3(0.0, 0.0, 0.5 + distance + probe_half_height))])
                    pipeline = newton.CollisionPipeline(
                        model,
                        broad_phase="nxn",
                        reduce_contacts=reduce_contacts,
                        speculative_contact_gap_max=0.2 if distance > 0.0 else None,
                    )
                    if probe_sdf:
                        test.assertTrue(pipeline.narrow_phase.mesh_sdf_texture_only)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts, dt=0.1)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    test.assertGreater(count, 0)
                    points0 = contacts.rigid_contact_point0.numpy()[:count]
                    points1 = contacts.rigid_contact_point1.numpy()[:count]
                    probe_points, box_points = (points0, points1) if probe_first else (points1, points0)
                    # Dynamic-body witnesses are body-local; static ones are world-space.
                    if probe_sdf:
                        # Side edges can also produce contacts; accept any probe-box face.
                        np.testing.assert_allclose(np.max(np.abs(probe_points), axis=1), 0.2, atol=sdf_tolerance)
                    else:
                        np.testing.assert_allclose(probe_points[:, 2], 0.0, atol=2.0e-5)
                    np.testing.assert_allclose(box_points[:, 2], 0.5, atol=sdf_tolerance)


def test_mesh_mesh_contact_at_pair_gap(test, device):
    """Retain mesh contacts at the inclusive authored pair gap."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(gap=0.0625, margin=2.0**-28, density=0.0)
        slab = newton.Mesh.create_box(0.0625, 0.0625, 0.015625)
        slab.vertices[:, 2] -= 0.015625
        builder.add_shape_mesh(-1, mesh=slab, cfg=cfg)
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        probe = newton.Mesh.create_box(0.03125, 0.03125, 0.015625)
        probe.vertices[:, 2] += 0.015625
        builder.add_shape_mesh(body, mesh=probe, cfg=cfg)
        model = builder.finalize()
        state = model.state()

        # Dyadic distances keep the face separation and threshold exact in float32.
        for reduce_contacts in (False, True):
            for speculative in (False, True):
                pipeline = newton.CollisionPipeline(
                    model,
                    broad_phase="nxn",
                    reduce_contacts=reduce_contacts,
                    speculative_contact_gap_max=0.25 if speculative else None,
                )
                contacts = pipeline.contacts()
                # A closing speed of 1.5 m/s reaches 0.1875 m in one update,
                # beyond the authored 0.125 m pair gap.
                dt = 0.125 if speculative else 0.0
                state.body_qd.assign([[0.0, 0.0, -1.5, 0.0, 0.0, 0.0]])
                for separation in (0.0625, 0.125, 0.125 + 2.0**-26, 0.125 + 2.0**-25, 0.1875, 0.25):
                    with test.subTest(reduce_contacts=reduce_contacts, speculative=speculative, separation=separation):
                        state.body_q.assign([wp.transform(wp.vec3(0.0, 0.0, separation))])
                        pipeline.collide(state, contacts, dt=dt)
                        count = int(contacts.rigid_contact_count.numpy()[0])
                        surface_distance = np.float32(separation) - np.float32(2.0 * cfg.margin)
                        test.assertEqual(count > 0, surface_distance <= (0.1875 if speculative else 0.125))


def test_mesh_sdf_normal_at_slab_midplane(test, device, *, reverse_winding=False, probe_sdf=False, coarse=False):
    """Recover toward a slab face when the stored distance has a flat gradient."""
    with wp.ScopedDevice(device):
        half_thickness = 0.0146484375 if coarse else 0.0009765625
        slab = newton.Mesh.create_box(half_thickness, 0.03125, 0.03125, duplicate_vertices=False)
        if reverse_winding:
            slab.indices = slab.indices.reshape(-1, 3)[:, [0, 2, 1]].flatten()
        # A narrow band leaves the thick slab's midplane on the coarse grid.
        narrow_band = 0.000244140625 if coarse else 0.001953125
        slab.build_sdf(
            target_voxel_size=0.000244140625,
            narrow_band_range=(-narrow_band, narrow_band),
            margin=0.001953125,
        )
        # A triangle on the medial plane lies well away from the slab edges.
        # An origin-relative fallback points along the slab instead of out.
        vertices = [[0.0, 0.0078125, 0.00390625], [0.0, 0.01171875, 0.00390625], [0.0, 0.0078125, 0.01171875]]
        indices = [0, 1, 2]
        if probe_sdf:
            # Close the probe on the far side of the medial plane to
            # exercise the texture-only specialization with the same edges.
            vertices.append([0.00390625, 0.0078125, 0.00390625])
            indices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
        probe = newton.Mesh(
            np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32), compute_inertia=False
        )
        if probe_sdf:
            probe.build_sdf(
                target_voxel_size=0.000244140625, narrow_band_range=(-0.001953125, 0.001953125), margin=0.001953125
            )
        scales = ((1.0, 1.0, 1.0),) if reverse_winding or probe_sdf or coarse else ((1.0, 1.0, 1.0), (2.0, 0.75, 1.25))
        for scale in scales:
            builder = newton.ModelBuilder()
            cfg = builder.ShapeConfig(density=0.0, margin=0.0, gap=0.00025)
            builder.add_shape_mesh(-1, mesh=slab, scale=scale, cfg=cfg)
            body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.add_shape_mesh(body, mesh=probe, cfg=cfg)
            model = builder.finalize()
            state = model.state()

            for reduce_contacts in (False, True):
                with test.subTest(scale=scale, reduce_contacts=reduce_contacts):
                    pipeline = newton.CollisionPipeline(model, reduce_contacts=reduce_contacts)
                    test.assertEqual(pipeline.narrow_phase.mesh_sdf_texture_only, probe_sdf)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts)
                    if probe_sdf:
                        with wp.ScopedCapture(device=device) as capture:
                            pipeline.collide(state, contacts)
                        contacts.rigid_contact_count.zero_()
                        contacts.rigid_contact_normal.fill_(wp.vec3(float("nan")))
                        contacts.rigid_contact_point0.fill_(wp.vec3(float("nan")))
                        if reduce_contacts:
                            pipeline.narrow_phase._flat_mesh_blocks.zero_()
                        wp.capture_launch(capture.graph)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    test.assertGreater(count, 0)
                    normals = contacts.rigid_contact_normal.numpy()[:count]
                    slab_points = contacts.rigid_contact_point0.numpy()[:count]
                    np.testing.assert_allclose(np.abs(normals[:, 0]), 1.0, atol=2.0e-5)
                    # Ordinary texture contacts can use a conservative
                    # distance under nonuniform scale. A recovered witness
                    # must reach the actual mesh face.
                    face_error = np.abs(np.abs(slab_points[:, 0]) - scale[0] * half_thickness)
                    if coarse:
                        np.testing.assert_allclose(face_error, 0.0, atol=2.0e-6)
                    else:
                        test.assertLess(float(face_error.min()), 2.0e-6)


@wp.kernel(enable_backward=False)
def _flat_witness(
    mesh: wp.uint64,
    sdf: TextureSDFData,
    scale: wp.vec3,
    depth: float,
    sign_method: int,
    boundary: wp.array[wp.uint8],
    offset: wp.array[wp.vec3],
):
    witness = _recover_flat_texture_contact(mesh, sdf, wp.vec3(), scale, -depth, sign_method, boundary, 0)
    offset[0] = witness


def test_flat_witness_metric(test, device):
    """Validate flat texture witnesses in the baked distance metric, including open boundaries."""
    sdf = TextureSDFData()
    sdf.coarse_texture = wp.Texture3D(np.zeros((2, 2, 2), dtype=np.float32), device=device)
    sdf.subgrid_start_slots = wp.zeros((1, 1, 1), dtype=wp.uint32, device=device)
    sdf.subgrid_size_f = 1.0
    sdf.fine_to_coarse = 1.0
    sdf.voxel_radius = 0.03125
    sdf.scale_baked = True
    mesh = newton.Mesh.create_box(0.125, 0.25, 0.5, duplicate_vertices=False)
    # The nearest local face considers the origin inside, while the nearer
    # face after applying scale considers it outside.
    sheets = newton.Mesh(
        np.array([[0.125, -1, -1], [0.125, 1, -1], [0.125, 0, 1], [-1, 0.25, -1], [1, 0.25, -1], [0, 0.25, 1]]),
        np.arange(6, dtype=np.int32),
        compute_inertia=False,
    )
    inward_sheets = newton.Mesh(sheets.vertices, np.array([0, 1, 2, 3, 5, 4], dtype=np.int32), compute_inertia=False)
    reversed_sheets = newton.Mesh(sheets.vertices, np.array([0, 2, 1, 3, 5, 4], dtype=np.int32), compute_inertia=False)
    seam = newton.Mesh(
        np.array([[0.125, -1, -1], [0.125, 1, -1], [0.125, 1, 1], [0.125, -1, 1]]),
        np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        compute_inertia=False,
    )
    scaled_seam = newton.Mesh(
        np.vstack([sheets.vertices[:3], [[-1, 0.25, -1], [-1, 0.25, 1], [1, 0.25, 1], [1, 0.25, -1]]]),
        np.array([0, 1, 2, 3, 4, 5, 3, 5, 6], dtype=np.int32),
        compute_inertia=False,
    )
    # Move the same patch past the sample so its nearest edge is an open boundary.
    shifted_vertices = scaled_seam.vertices.copy()
    shifted_vertices[3:, 0] += 1.01
    open_edge = newton.Mesh(shifted_vertices, scaled_seam.indices, compute_inertia=False)
    # Both triangles meet along Z. Their baked-space normals disagree
    # individually, but the shared-edge pseudo-normal faces the witness.
    crease_vertices = np.array([[1, -0.25, -1], [1, -0.25, 1], [2, -0.25, 0], [2, 0.75, 0]]) / [2, 1, 1]
    crease = newton.Mesh(crease_vertices, np.array([0, 1, 2, 1, 0, 3]), compute_inertia=False)
    crease_reordered = newton.Mesh(crease_vertices, np.array([1, 0, 3, 0, 1, 2]), compute_inertia=False)
    # At this vertex, weighting by each corner angle reverses the sign of
    # the unweighted sum of the three incident normals.
    vertex_fan = newton.Mesh(
        np.array([[1, 0, 0], [2, 0, 1], [2, 1, 0], [2, 3, -1]]) / [2, 1, 1],
        np.array([0, 1, 2, 0, 2, 3, 0, 3, 1]),
        compute_inertia=False,
    )
    output = wp.empty(1, dtype=wp.vec3, device=device)
    for source, scale, depth, sign, expected in (
        (mesh, (4, 1, 1), 0.25, MeshSignMethod.PARITY, (0, 0.25, 0)),
        (mesh, (1, 1, 1), 0.09375, MeshSignMethod.PARITY, (0.125, 0, 0)),
        (mesh, (1, 1, 1), 0.093, MeshSignMethod.PARITY, (0, 0, 0)),
        (sheets, (4, 1, 1), 0.25, MeshSignMethod.NORMAL, (0, 0, 0)),
        (inward_sheets, (-4, 1, 1), 0.25, MeshSignMethod.NORMAL, (0, 0.25, 0)),
        (reversed_sheets, (4, 1, 1), 0.25, MeshSignMethod.NORMAL, (0, 0.25, 0)),
        (seam, (1, 2, 1), 0.125, MeshSignMethod.NORMAL, (0.125, 0, 0)),
        (scaled_seam, (4, 1, 1), 0.25, MeshSignMethod.NORMAL, (0, 0.25, 0)),
        (open_edge, (4, 1, 1), 0.25, MeshSignMethod.NORMAL, (0, 0, 0)),
        (crease, (2, 1, 1), np.sqrt(1.0625), MeshSignMethod.NORMAL, (1, -0.25, 0)),
        (crease_reordered, (2, 1, 1), np.sqrt(1.0625), MeshSignMethod.NORMAL, (1, -0.25, 0)),
        (vertex_fan, (2, 1, 1), 1.0, MeshSignMethod.NORMAL, (1, 0, 0)),
    ):
        with test.subTest(scale=scale, depth=depth, sign=sign):
            mesh_id = source.finalize(device)
            boundary = wp.array(compute_mesh_boundary_flags(source.mesh), dtype=wp.uint8, device=device)
            wp.launch(
                _flat_witness,
                1,
                inputs=[mesh_id, sdf, wp.vec3(*scale), depth, int(sign), boundary, output],
                device=device,
            )
            witness = output.numpy()[0]
            np.testing.assert_allclose(np.abs(witness) if source is mesh else witness, expected, atol=1.0e-6)


def test_planar_mesh_recovery_under_load(test, device, *, reduce_contacts=True, mujoco=False, vbd_stale=False):
    """Expose the outward contact and recover from a loaded thin-wall overlap."""
    with wp.ScopedDevice(device):
        wall, probe = test.loaded_wall_and_probe(device)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        cfg = builder.ShapeConfig(gap=0.0002, margin=0.0, density=0.0, ke=1.0e6, kd=150.0)
        if not reduce_contacts:
            cfg.restitution = 1.0
        builder.add_shape_mesh(-1, mesh=wall, xform=wp.transform(wp.vec3(0.0005, 0.0, 0.0)), cfg=cfg)
        body = builder.add_body(mass=0.05, inertia=wp.mat33(np.eye(3) * 1.0e-6))
        builder.add_shape_mesh(body, mesh=probe, cfg=cfg)
        if not reduce_contacts:
            floor = builder.add_ground_plane(height=-0.00395, cfg=cfg)
        if vbd_stale:
            builder.color()
        model = builder.finalize()
        # Reversed duplicate candidates must still publish one recovery row.
        pairs = None if reduce_contacts else wp.array([(0, 1), (1, 0)], dtype=wp.vec2i, device=device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn" if pairs is None else "explicit",
            shape_pairs_filtered=pairs,
            deterministic=True,
            contact_matching="sticky" if vbd_stale else "disabled",
            reduce_contacts=reduce_contacts,
            rigid_contact_max=128,
        )
        contacts = pipeline.contacts()
        state, next_state = model.state(), model.state()
        if mujoco:
            solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)
        elif vbd_stale:
            # The ordinary manifold's full stabilization must still allow
            # a bounded correction of the verified deep overlap.
            slot = newton.solvers.SolverBase.CollisionSlot.RIGID
            frequency = newton.solvers.SolverBase.CollisionFrequencyType
            solver = newton.solvers.SolverVBD(
                model,
                iterations=4,
                rigid_avbd_contact_alpha=1.0,
            )
            # Late-bound pipeline storage must also support mid-step refresh.
            solver.collision_pipeline = pipeline
            solver.contacts = contacts
            solver.set_collision_frequency(collision_frequency_type={slot: frequency.PRE_POST_INIT})
        else:
            solver = newton.solvers.SolverXPBD(model, iterations=4)
        control = model.control()
        sensor = SensorContact(model, sensing_bodies=[body], request_contact_attributes=False)
        observables = solver.observables(contacts=contacts, kinds=sensor.solver_observable_kinds)
        dt = 1.0 / 1440.0

        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertEqual(count, 1)
        normal = contacts.rigid_contact_normal.numpy()[0]
        if contacts.rigid_contact_shape0.numpy()[0] != 0:
            normal = -normal
        np.testing.assert_allclose(normal, [-1.0, 0.0, 0.0], atol=1.0e-5)
        # At the initial identity body pose, both witnesses are in world coordinates.
        wall_points, probe_points = contacts.rigid_contact_point0, contacts.rigid_contact_point1
        if contacts.rigid_contact_shape0.numpy()[0] != 0:
            wall_points, probe_points = probe_points, wall_points
        np.testing.assert_allclose(wall_points.numpy()[0], [0, 0, 0], atol=1.0e-6)
        np.testing.assert_allclose(probe_points.numpy()[0], [0.004, 0, 0], atol=1.0e-6)
        if vbd_stale:
            # Sticky matches must retain the freshly certified wall witness after a small slide.
            state.body_q.assign([wp.transform(wp.vec3(0.0, 0.00005, 0.0))])
            pipeline.collide(state, contacts)
            test.assertEqual(int(contacts.rigid_contact_match_index.numpy()[0]), 0)
            test.assertAlmostEqual(float(wall_points.numpy()[0, 1]), 0.00005, delta=1.0e-6)

        if not reduce_contacts:
            # Both penalty solvers share the force kernel. A 4 mm deep contact
            # must bound its elastic force using the 1 mm wall thickness.
            expected_speed = -cfg.ke * (0.001 / 32.0) * dt / 0.05
            for penalty_solver in (newton.solvers.SolverSemiImplicit, newton.solvers.SolverFeatherstone):
                with test.subTest(solver=penalty_solver.__name__):
                    penalty_solver(model).step(model.state(), next_state, control, contacts, dt)
                    test.assertAlmostEqual(float(next_state.body_qd.numpy()[body, 0]), expected_speed, delta=1.0e-4)
            # A separating recovery row must leave the ordinary floor correction at full weight.
            floor_pipeline = newton.CollisionPipeline(
                model,
                broad_phase="explicit",
                shape_pairs_filtered=wp.array([(0, 1), (1, floor)], dtype=wp.vec2i),
                reduce_contacts=False,
            )
            floor_contacts = floor_pipeline.contacts()
            separating = model.state()
            separating.body_qd.fill_(wp.spatial_vector(-0.2, 0.0, 0.0, 0.0, 0.0, 0.0))
            model.shape_material_mu.zero_()
            floor_pipeline.collide(separating, floor_contacts)
            count = int(floor_contacts.rigid_contact_count.numpy()[0])
            shape0 = floor_contacts.rigid_contact_shape0.numpy()
            shape1 = floor_contacts.rigid_contact_shape1.numpy()
            wall_rows = np.where((shape0[:count] == 0) | (shape1[:count] == 0))[0]
            test.assertEqual(len(wall_rows), 1)
            floor_solver = newton.solvers.SolverXPBD(model, iterations=1)
            # Hit the floor and the embedded wall simultaneously. Only the
            # ordinary floor contact should receive a restitution impulse.
            impact = model.state()
            impact.body_qd.fill_(wp.spatial_vector(0.1, 0.0, -0.1, 0.0, 0.0, 0.0))
            floor_solver.step(impact, next_state, control, floor_contacts, dt)
            inelastic = next_state.body_qd.numpy()[body]
            elastic_solver = newton.solvers.SolverXPBD(model, iterations=1, enable_restitution=True)
            elastic_solver.step(impact, next_state, control, floor_contacts, dt)
            elastic = next_state.body_qd.numpy()[body]
            test.assertGreater(elastic[2], 0.0)
            test.assertGreater(elastic[2] - inelastic[2], 0.01)
            test.assertGreater(elastic[0], 0.0)  # Still approaching the wall.
            floor_solver.step(separating, next_state, control, floor_contacts, dt)
            with_recovery = float(next_state.body_qd.numpy()[body, 2])
            test.assertGreater(with_recovery, 0.0)
            shape1[wall_rows] = shape0[wall_rows]
            floor_contacts.rigid_contact_shape1.assign(shape1)
            floor_solver.step(separating, next_state, control, floor_contacts, dt)
            test.assertAlmostEqual(float(next_state.body_qd.numpy()[body, 2]), with_recovery, delta=1.0e-6)
            # Keep a small tangential predictor displacement inside the swept footprint.
            state.body_qd.fill_(wp.spatial_vector(0.0, 0.002, 0.0, 0.0, 0.0, 0.0))

        if vbd_stale:
            # First project a live row, then hold it as the probe slides past
            # the verified footprint. The deep public separation persists.
            model.shape_material_mu.zero_()
            state.body_qd.fill_(wp.spatial_vector(0.0, 0.1, 0.0, 0.0, 0.0, 0.0))
            solver.step(state, next_state, control, None, dt, observables=observables)
            first_speed = float(next_state.body_qd.numpy()[body, 0])
            test.assertLess(first_speed, -0.01)
            test.assertLessEqual(-first_speed * dt, 0.001 / 32.0 + 1.0e-6)
            test.assertGreater(float(solver.body_body_contact_normal_rho.numpy()[0]), 0.0)
            test.assertGreater(np.linalg.norm(solver.body_body_contact_lambda.numpy()[0]), 0.0)
            # The new detection is valid at step start. The predictor moves
            # past its certified footprint before the first primal solve.
            solver.set_collision_frequency(collision_frequency_type={slot: frequency.NONE})
            pipeline.collide(next_state, contacts)
            next_state.body_qd.fill_(wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
            previous_q = wp.clone(solver.body_q_prev)
            solver.step(next_state, state, control, None, dt, observables=observables)
            test.assertAlmostEqual(float(state.body_qd.numpy()[body, 0]), 0.0, delta=1.0e-5)
            np.testing.assert_array_equal(solver.body_body_contact_lambda.numpy()[0], 0.0)
            np.testing.assert_array_equal(observables.contact_f.numpy()[0], 0.0)
            _, _, point0, point1, force, _ = solver.collect_rigid_contact_forces(state.body_q, previous_q, contacts, dt)
            pose = state.body_q.numpy()[body]
            for points, local, offsets, shape in (
                (point0, contacts.rigid_contact_point0, contacts.rigid_contact_offset0, contacts.rigid_contact_shape0),
                (point1, contacts.rigid_contact_point1, contacts.rigid_contact_offset1, contacts.rigid_contact_shape1),
            ):
                expected = local.numpy()[0] + offsets.numpy()[0]
                if shape.numpy()[0] == 1:
                    expected = wp.transform_point(wp.transform(pose[:3], pose[3:]), wp.vec3(expected))
                np.testing.assert_allclose(points.numpy()[0], expected, atol=1.0e-7)
            np.testing.assert_array_equal(force.numpy()[0], 0.0)

            # Moving beyond the entry face also releases the held row.
            state.body_q.fill_(wp.transform(wp.vec3(-0.005, previous_q.numpy()[body, 1], 0.0), wp.quat_identity()))
            state.body_qd.zero_()
            solver.step(state, next_state, control, None, dt, observables=observables)
            np.testing.assert_array_equal(solver.body_body_contact_lambda.numpy()[0], 0.0)
            np.testing.assert_array_equal(observables.contact_f.numpy()[0], 0.0)

            solver.set_collision_frequency(collision_frequency_type={slot: frequency.PRE_INIT})
            model.shape_material_ke.fill_(1.0e5)
            model.shape_material_kd.zero_()
            # At 20 N these unconstrained predictors travel 0.193 and 0.434 mm,
            # respectively inside and outside the 0.25 mm inward interval.
            for dt in (1.0 / 1440.0, 1.0 / 960.0):
                with test.subTest(dt=dt):
                    state, next_state = model.state(), model.state()
                    solver.reset(state)
                    for _ in range(4):
                        state.body_f.fill_(wp.spatial_vector(20.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                        previous_x = float(state.body_q.numpy()[body, 0])
                        solver.step(state, next_state, control, None, dt, observables=observables)
                        displacement = float(next_state.body_q.numpy()[body, 0]) - previous_x
                        test.assertLess(displacement, 0.0)
                        test.assertGreater(displacement, -0.001 / 32.0 - 1.0e-6)
                        state, next_state = next_state, state
            return

        if mujoco:
            # Reuse one compacted slot across substeps as its certificate
            # becomes stale and valid again, without refreshing the manifold.
            for lateral, active in ((0.0, True), (0.02, False), (0.0, True)):
                state.body_q.fill_(wp.transform(wp.vec3(0.0, lateral, 0.0), wp.quat_identity()))
                solver._convert_contacts_to_mjwarp(model, state, contacts)
                test.assertEqual(int(solver.mjw_data.nacon.numpy()[0]), 1)
                test.assertEqual(bool(solver.mjw_data.contact.type.numpy()[0]), active)

        # Give the bounded outward correction time to reach the entry face.
        for i in range(200 if mujoco else 136):
            if i:
                pipeline.collide(state, contacts)
            state.body_f.fill_(wp.spatial_vector(20.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            solver.step(state, next_state, control, contacts, dt, observables=observables)
            state, next_state = next_state, state
            if not i:
                # One step must begin recovery without turning a four-mm
                # geometric correction into a high outward velocity.
                speed = float(state.body_qd.numpy()[body, 0])
                test.assertGreater(speed, -0.1)
                test.assertLess(speed, 0.0)
                sensor.update(state, contacts, observables=observables)
                test.assertLess(float(sensor.total_force.numpy()[0, 0]), -1.0)
                np.testing.assert_allclose(sensor.total_force_friction.numpy(), 0.0, atol=1.0e-4)
        depth = float(state.body_q.numpy()[body, 0]) + 0.004
        test.assertLess(abs(depth), 0.0005)


def test_mujoco_stale_contact_at_capacity(test, device):
    """A stale wall row must leave the one solver slot available to the sphere."""
    with wp.ScopedDevice(device):
        wall, probe = test.loaded_wall_and_probe(device)
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(gap=0.0002, margin=0.0)
        builder.add_shape_mesh(-1, mesh=wall, xform=wp.transform(wp.vec3(0.0005, 0.0, 0.0)), cfg=cfg)
        probe_body = builder.add_body(mass=0.05)
        builder.add_shape_mesh(probe_body, mesh=probe, cfg=cfg)
        newton.solvers.mujoco.add_actuator_motor(builder, newton.solvers.mujoco.ActuatorTarget.body(probe_body))
        plane = builder.add_ground_plane()
        ball = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.049)))
        sphere = builder.add_shape_sphere(ball, radius=0.05)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=wp.array([(0, 1), (plane, sphere)], dtype=wp.vec2i),
            deterministic=True,
            rigid_contact_max=8,
        )
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, nconmax=1)
        obs = solver.observables({solver.ObservableKind.OVERFLOW})
        contacts, state, next_state = pipeline.contacts(), model.state(), model.state()
        pipeline.collide(state, contacts)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 2)
        test.assertIn(0, (contacts.rigid_contact_shape0.numpy()[0], contacts.rigid_contact_shape1.numpy()[0]))
        poses = state.body_q.numpy()
        for lateral in (0.02, 0.0):
            poses[probe_body, 1] = lateral
            state.body_q.assign(poses)
            solver.step(state, next_state, None, contacts, 1.0 / 1440.0, observables=obs)
            test.assertEqual(int(solver.mjw_data.nacon.numpy()[0]), 1)
            test.assertEqual(bool(int(obs.overflow.numpy()[-1]) & solver.OverflowBits.SOLVER_NCONMAX), lateral == 0.0)
            if lateral:
                np.testing.assert_array_equal(solver._contact_tid_to_cid.numpy()[:2], [-1, 0])


def test_planar_recovery_overflow(test, device):
    """Track a deep pair only if its row fits, and return shallow pairs to ordinary contacts."""
    with wp.ScopedDevice(device):
        wall, probe = test.loaded_wall_and_probe(device)
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig(gap=0.0002, margin=0.0)
        builder.add_shape_mesh(-1, mesh=wall, xform=wp.transform(wp.vec3(0.0005, 0.0, 0.0)), cfg=cfg)
        body = builder.add_body()
        for _ in range(2):
            builder.add_shape_mesh(body, mesh=probe, cfg=cfg)
        model = builder.finalize()
        pairs = wp.array([(0, 1), (0, 2)], dtype=wp.vec2i)
        # Deliberately overfill one output slot and verify the overflow count below.
        pipeline = newton.CollisionPipeline(
            model, broad_phase="explicit", shape_pairs_filtered=pairs, rigid_contact_max=1, verify_buffers=False
        )
        contacts = pipeline.contacts()
        state = model.state()
        pipeline.collide(state, contacts)
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
        stored = int(contacts.rigid_contact_shape1.numpy()[0])
        recovery = contacts._mesh_recovery_contacts
        test.assertEqual({1, 2}.intersection(recovery.keys.numpy()), {stored})

        # Entering a marginal overlap directly or from a deep overlap must
        # produce the same number of recovery rows, including on overflow.
        state.body_q.fill_(wp.transform(wp.vec3(-0.0039, 0.0, 0.0), wp.quat_identity()))
        fresh = pipeline.contacts()
        pipeline.collide(state, fresh)
        pipeline.collide(state, contacts)
        for data in (recovery, fresh._mesh_recovery_contacts):
            test.assertEqual(len({1, 2}.intersection(data.keys.numpy())), 1)

        # Leave 10 micrometres of overlap against the 1 mm wall.
        state.body_q.fill_(wp.transform(wp.vec3(-0.00399, 0.0, 0.0), wp.quat_identity()))
        pipeline.collide(state, contacts)
        test.assertEqual(int(recovery.active_slots.numpy()[-1]), 0)
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_planar_recovery_eligibility(test, device):
    """Keep disconnected probes and meshes without textures on the ordinary contact path."""
    with wp.ScopedDevice(device):
        wall, probe = test.loaded_wall_and_probe(device)
        small = newton.Mesh.create_capsule(0.0005, 0.0, segments=12)
        split = newton.Mesh(
            np.vstack((probe.vertices, small.vertices + np.array([0.0057, 0, 0]))),
            np.concatenate((probe.indices, small.indices + len(probe.vertices))),
        )
        split.build_sdf(target_voxel_size=0.000125, narrow_band_range=(-0.002, 0.002), margin=0.002)
        legacy_wall, legacy_probe = wall.copy(), probe.copy()
        volume = wp.Volume.load_from_numpy(np.ones((8, 8, 8), dtype=np.float32), voxel_size=0.001, device=device)
        legacy_wall.sdf = legacy_probe.sdf = newton.SDF.create_from_data(sparse_volume=volume)
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(-1, mesh=wall)
        body = builder.add_body()
        for mesh, x in ((probe, -0.0005), (split, -0.004), (legacy_probe, -0.0005)):
            builder.add_shape_mesh(body, mesh=mesh, xform=wp.transform(wp.vec3(x, 0.0, 0.0)))
        builder.add_shape_mesh(-1, mesh=legacy_wall)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=wp.array([(0, 1), (0, 2), (0, 3), (4, 1)], dtype=wp.vec2i),
        )
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        recovery = contacts._mesh_recovery_contacts
        slots = recovery.active_slots.numpy()
        np.testing.assert_array_equal(recovery.keys.numpy()[slots[: slots[-1]]], [1])


class TestSDFContact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._loaded_meshes = {}
        cls.addClassCleanup(cls._loaded_meshes.clear)

    @classmethod
    def loaded_wall_and_probe(cls, device):
        if device.alias not in cls._loaded_meshes:
            wall = newton.Mesh.create_box(0.0005, 0.015, 0.015, duplicate_vertices=False)
            probe = newton.Mesh.create_capsule(0.004, 0.0, segments=16)
            for mesh in (wall, probe):
                mesh.build_sdf(target_voxel_size=0.000125, narrow_band_range=(-0.002, 0.002), margin=0.002)
            cls._loaded_meshes[device.alias] = wall, probe
        return cls._loaded_meshes[device.alias]

    def test_nonconvex_edge_keeps_endpoint_minima(self):
        """Keep deeper endpoint minima despite a shallower interior well."""
        for device in get_test_devices():
            for kernel in (_nonconvex_edge_scalar, _nonconvex_edge_paired):
                with self.subTest(device=device, kernel=kernel.key):
                    scales = wp.array([0.001, 1.0, 1000.0], dtype=float, device=device)
                    distances = wp.empty(6, dtype=float, device=device)
                    endpoints = wp.empty(6, dtype=int, device=device)
                    wp.launch(
                        kernel,
                        dim=6,
                        inputs=[
                            scales,
                            distances,
                            endpoints,
                            wp.empty(0, dtype=wp.uint8, device=device),
                            wp.empty(0, dtype=float, device=device),
                        ],
                        device=device,
                    )
                    np.testing.assert_allclose(distances.numpy(), np.repeat(-0.2 * scales.numpy(), 2), rtol=1e-6)
                    np.testing.assert_array_equal(endpoints.numpy(), [1, 2, 1, 2, 1, 2])

    def test_block_count_scan_ignores_inactive_tail(self) -> None:
        """Keep active block offsets independent of stale inactive slots."""
        for device in get_test_devices():
            with self.subTest(device=device):
                weights = wp.array([1024, 2048, 1024, 99, 99, 99, 99, 99], dtype=wp.int32, device=device)
                pair_count = wp.array([3], dtype=wp.int32, device=device)
                total_weight = wp.array([4096], dtype=wp.int32, device=device)
                block_counts = wp.full(8, 77, dtype=wp.int32, device=device)
                offsets = wp.empty_like(block_counts)

                wp.launch(
                    compute_block_counts_from_weights,
                    dim=2,
                    inputs=[total_weight, weights, pair_count, len(weights), 4, block_counts, 2],
                    device=device,
                )
                wp.utils.array_scan(block_counts, offsets, inclusive=False)
                np.testing.assert_array_equal(offsets.numpy()[:4], [0, 1, 3, 4])

                pair_count.assign([1])
                total_weight.assign([1024])
                wp.launch(
                    compute_block_counts_from_weights,
                    dim=2,
                    inputs=[total_weight, weights, pair_count, len(weights), 4, block_counts, 2],
                    device=device,
                )
                wp.utils.array_scan(block_counts, offsets, inclusive=False)
                np.testing.assert_array_equal(offsets.numpy()[:2], [0, 4])

    def test_mesh_sdf_contact_search_precision_uses_inner_envelope(self) -> None:
        device = wp.get_preferred_device()
        values = wp.empty(4, dtype=wp.float32, device=device)

        wp.launch(_mesh_sdf_contact_search_precision_kernel, dim=1, inputs=[values], device=device)

        np.testing.assert_allclose(values.numpy(), np.array([0.0, 0.001, 0.005, 0.005], dtype=np.float32))

    def test_sdf_rsqrt_rn_on_all_devices(self) -> None:
        """Verify the native reciprocal square root on CPU and CUDA."""
        values_np = np.array([0.25, 1.0, 2.0, 4.0, 100.0], dtype=np.float32)
        expected = np.float32(1.0) / np.sqrt(values_np)

        for device in get_test_devices():
            with self.subTest(device=device):
                values = wp.array(values_np, device=device)
                result = wp.empty_like(values)
                wp.launch(_sdf_rsqrt_rn_kernel, dim=len(values_np), inputs=[values, result], device=device)

                np.testing.assert_allclose(result.numpy(), expected, rtol=np.finfo(np.float32).eps, atol=0.0)


add_function_test(
    TestSDFContact,
    "test_mesh_mesh_contained_bvh_contacts",
    test_mesh_mesh_contained_bvh_contacts,
    devices=get_test_devices(),
)

add_function_test(
    TestSDFContact,
    "test_mesh_mesh_contact_at_pair_gap",
    test_mesh_mesh_contact_at_pair_gap,
    devices=get_test_devices(),
)
add_function_test(
    TestSDFContact,
    "test_mesh_sdf_normal_at_slab_midplane",
    test_mesh_sdf_normal_at_slab_midplane,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSDFContact,
    "test_mesh_sdf_normal_at_coarse_slab_midplane",
    test_mesh_sdf_normal_at_slab_midplane,
    devices=get_cuda_test_devices(),
    coarse=True,
)
add_function_test(
    TestSDFContact,
    "test_mesh_sdf_normal_at_slab_midplane_texture_only",
    test_mesh_sdf_normal_at_slab_midplane,
    devices=get_cuda_test_devices(),
    probe_sdf=True,
)
add_function_test(
    TestSDFContact,
    "test_mesh_sdf_normal_at_slab_midplane_reversed_winding",
    test_mesh_sdf_normal_at_slab_midplane,
    devices=get_cuda_test_devices(),
    reverse_winding=True,
)
add_function_test(
    TestSDFContact,
    "test_planar_recovery_overflow",
    test_planar_recovery_overflow,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSDFContact,
    "test_planar_recovery_eligibility",
    test_planar_recovery_eligibility,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestSDFContact,
    "test_planar_mesh_recovery_with_xpbd_unreduced",
    test_planar_mesh_recovery_under_load,
    devices=get_cuda_test_devices(),
    reduce_contacts=False,
)
add_function_test(
    TestSDFContact,
    "test_held_deep_contact_retires_in_vbd",
    test_planar_mesh_recovery_under_load,
    devices=get_cuda_test_devices(),
    vbd_stale=True,
)
add_function_test(
    TestSDFContact,
    "test_mujoco_stale_contact_at_capacity",
    requires(*MUJOCO_MODULES)(test_mujoco_stale_contact_at_capacity),
    devices=get_cuda_test_devices(),
    check_output=False,  # The second step intentionally reports a solver-capacity overflow.
)
add_function_test(
    TestSDFContact,
    "test_planar_mesh_recovery_with_mujoco_warp",
    requires(*MUJOCO_MODULES)(test_planar_mesh_recovery_under_load),
    devices=get_cuda_test_devices(),
    mujoco=True,
)
add_function_test(
    TestSDFContact,
    "test_mesh_mesh_sdf_surface_witnesses_bvh_bvh",
    test_mesh_mesh_sdf_surface_witnesses,
    devices=get_test_devices(),
    box_sdf=False,
)
add_function_test(
    TestSDFContact,
    "test_mesh_mesh_sdf_surface_witnesses_bvh_texture",
    test_mesh_mesh_sdf_surface_witnesses,
    devices=get_cuda_test_devices(),
    box_sdf=True,
)
add_function_test(
    TestSDFContact,
    "test_mesh_mesh_sdf_surface_witnesses_texture_texture",
    test_mesh_mesh_sdf_surface_witnesses,
    devices=get_cuda_test_devices(),
    box_sdf=True,
    probe_sdf=True,
)


add_function_test(TestSDFContact, "test_flat_witness_metric", test_flat_witness_metric, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
