# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check conservative terrain rejection against the original triangle pipeline."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.utils.heightfield import (
    HeightfieldData,
    _heightfield_cell_below_query,
    heightfield_vs_convex_midphase,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _create_heightfield_midphase(*, upstream_default):
    """Exercise the upstream call contract or an unculled reference at the launch boundary."""

    @wp.kernel(enable_backward=False)
    def midphase(
        shape_types: wp.array[int],
        shape_transform: wp.array[wp.transform],
        shape_source: wp.array[wp.uint64],
        shape_gap: wp.array[float],
        shape_data: wp.array[wp.vec4],
        shape_collision_radius: wp.array[float],
        shape_collision_aabb_lower: wp.array[wp.vec3],
        shape_collision_aabb_upper: wp.array[wp.vec3],
        shape_heightfield_index: wp.array[wp.int32],
        heightfield_data: wp.array[HeightfieldData],
        heightfield_elevations: wp.array[wp.float32],
        shape_pairs_mesh: wp.array[wp.vec2i],
        shape_pairs_mesh_count: wp.array[int],
        total_num_threads: int,
        triangle_pairs: wp.array[wp.vec3i],
        triangle_pairs_count: wp.array[int],
    ):
        tid, lane = wp.tid()
        if lane != 0:
            return
        for i in range(tid, shape_pairs_mesh_count[0], total_num_threads):
            pair = shape_pairs_mesh[i]
            hfd = heightfield_data[shape_heightfield_index[pair[0]]]
            if wp.static(upstream_default):
                heightfield_vs_convex_midphase(
                    pair[0],
                    pair[1],
                    hfd,
                    heightfield_elevations,
                    shape_transform,
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                    shape_data,
                    shape_gap,
                    triangle_pairs,
                    triangle_pairs_count,
                )
            else:
                heightfield_vs_convex_midphase(
                    pair[0],
                    pair[1],
                    hfd,
                    heightfield_elevations,
                    shape_transform,
                    shape_collision_aabb_lower,
                    shape_collision_aabb_upper,
                    shape_data,
                    shape_gap,
                    triangle_pairs,
                    triangle_pairs_count,
                    False,
                )

    return midphase


_unculled_heightfield_midphase = _create_heightfield_midphase(upstream_default=False)
_default_heightfield_midphase = _create_heightfield_midphase(upstream_default=True)


@wp.kernel
def _check_cells(lower: wp.array[float], heights: wp.array[float], result: wp.array[int]):
    i = wp.tid()
    terrain = HeightfieldData()
    terrain.data_offset = i * 4
    terrain.nrow = 2
    terrain.ncol = 2
    terrain.min_z = -2.0
    terrain.max_z = 2.0
    result[i] = int(_heightfield_cell_below_query(lower[i], 0.001, terrain, heights, 0, 0))


def _model(
    *,
    device="cpu",
    z=0.2,
    rotation=None,
    reverse=False,
    shape="box",
    scale=(1.0, 1.0, 1.0),
    shape_rotation=None,
    mixed_mesh=False,
):
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
    heights = np.zeros((9, 9), dtype=np.float32)
    heights[0, 0] = 1.0  # Keep the broad-phase terrain box above the distant shape.
    terrain = newton.Heightfield(data=heights, nrow=9, ncol=9, hx=0.4, hy=0.4)
    q = wp.quat_identity() if rotation is None else rotation
    terrain_pose = wp.transform((20.0, -40.0, 0.0), q)
    body_pose = terrain_pose * wp.transform(
        (0.0, 0.0, z), wp.quat_identity() if shape_rotation is None else shape_rotation
    )

    def add_terrain():
        builder.add_shape_heightfield(heightfield=terrain, xform=terrain_pose, scale=scale, cfg=cfg)

    def add_shape():
        body = builder.add_body(xform=body_pose)
        if shape == "box":
            builder.add_shape_box(body, hx=0.10155461, hy=0.03273462, hz=0.00925394, cfg=cfg)
        elif shape == "sphere":
            builder.add_shape_sphere(body, radius=0.035, cfg=cfg)
        elif shape == "capsule":
            builder.add_shape_capsule(body, radius=0.025, half_height=0.04, cfg=cfg)
        else:
            builder.add_shape_cylinder(body, radius=0.035, half_height=0.015, cfg=cfg)

    for add in (add_shape, add_terrain) if reverse else (add_terrain, add_shape):
        add()
    if mixed_mesh:
        mesh = newton.Mesh(
            vertices=np.array([[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
        )
        builder.add_shape_mesh(-1, mesh=mesh, xform=wp.transform((30.0, -40.0, 10.0), wp.quat_identity()))
    return builder.finalize(device=device)


def _collide(model, *, legacy=False, speculative=False, midphase=None):
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=False,
        rigid_contact_max=2048,
        max_triangle_pairs=4096,
        speculative_contact_gap_max=0.5 if speculative else None,
    )
    if legacy:
        midphase = _unculled_heightfield_midphase
    contacts = pipeline.contacts()
    state = model.state()
    if speculative:
        velocity = np.zeros((model.body_count, 6), dtype=np.float32)
        velocity[:, 2] = -20.0
        state.body_qd.assign(velocity)
    override = (
        patch("newton._src.geometry.narrow_phase.narrow_phase_find_mesh_triangle_overlaps_kernel", midphase)
        if midphase is not None
        else nullcontext()
    )
    with override:
        pipeline.collide(state, contacts, dt=0.01 if speculative else None)
    count = int(contacts.rigid_contact_count.numpy()[0])
    assert count <= contacts.rigid_contact_max
    assert int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0]) <= pipeline.narrow_phase.max_triangle_pairs
    distance = wp.empty(2048, dtype=float, device=model.device)
    point = wp.empty(2048, dtype=wp.vec3, device=model.device)
    newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance, out_point0_world=point)
    geometry = (distance.numpy()[:count], contacts.rigid_contact_normal.numpy()[:count], point.numpy()[:count])
    return pipeline, geometry


def _assert_geometry_equal(test, before, after):
    """Compare complete physical contact geometry independently of row ordering."""
    test.assertTrue(all(np.isfinite(a).all() for values in (before, after) for a in values))
    test.assertEqual(len(before[0]), len(after[0]))
    for source, target in ((before, after), (after, before)):
        error = np.maximum.reduce(
            (
                np.abs(source[0][:, None] - target[0][None, :]) / 2.0e-4,
                np.linalg.norm(source[1][:, None] - target[1][None, :], axis=2) / 2.0e-3,
                np.linalg.norm(source[2][:, None] - target[2][None, :], axis=2) / 2.0e-4,
            )
        )
        test.assertLess(float(np.max(np.min(error, axis=1))), 1.0)


def test_remove_separated_triangle_work(test, device):
    """Reject airborne cells even while the terrain broad-phase AABB overlaps."""
    model = _model(device=device)
    before, a = _collide(model, legacy=True)
    after, b = _collide(model)
    test.assertGreater(int(before.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
    test.assertEqual(int(after.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
    test.assertEqual(len(a[0]), 0)
    test.assertEqual(len(b[0]), 0)


def test_upstream_default_call_contract(test, device):
    """Compile the upstream eleven-argument call and retain default cell rejection."""
    model = _model(device=device)
    reference, _ = _collide(model, legacy=True)
    default, geometry = _collide(model, midphase=_default_heightfield_midphase)
    test.assertGreater(int(reference.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
    test.assertEqual(int(default.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
    test.assertEqual(len(geometry[0]), 0)


def test_near_below_transformed_and_scaled(test, device):
    """Keep margin contacts, downward prisms, scaled terrain and reversed endpoints."""
    for z, rotation, reverse, scale in (
        (0.035, None, False, (1.0, 1.0, 1.0)),
        (-0.02, None, True, (1.0, 1.0, 1.0)),
        (0.005, wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45), True, (1.5, 0.7, 2.0)),
    ):
        with test.subTest(z=z):
            model = _model(
                device=device,
                z=z,
                rotation=rotation,
                reverse=reverse,
                scale=scale,
                shape_rotation=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3) if rotation is not None else None,
            )
            before, a = _collide(model, legacy=True)
            after, b = _collide(model)
            test.assertGreater(len(a[0]), 0)
            _assert_geometry_equal(test, a, b)
            test.assertEqual(
                int(before.narrow_phase.triangle_pairs_count.numpy()[0]),
                int(after.narrow_phase.triangle_pairs_count.numpy()[0]),
            )


def test_primitive_contact_geometry(test, device):
    """Preserve contacts for sphere, capsule and cylinder terrain queries."""
    for shape in ("sphere", "capsule", "cylinder"):
        with test.subTest(shape=shape):
            model = _model(device=device, z=0.02, shape=shape)
            _, a = _collide(model, legacy=True)
            _, b = _collide(model)
            test.assertGreater(len(a[0]), 0)
            _assert_geometry_equal(test, a, b)


def test_speculative_search_gap(test, device):
    """Keep approaching contacts admitted by the current expanded search gap."""
    model = _model(device=device, z=0.15)
    _, a = _collide(model, legacy=True, speculative=True)
    pipeline, b = _collide(model, speculative=True)
    test.assertGreater(float(pipeline._shape_search_gap.numpy().max()), float(model.shape_gap.numpy().max()))
    test.assertGreater(len(a[0]), 0)
    _assert_geometry_equal(test, a, b)


def test_sphere_contacts_at_gap_boundary(test, device):
    """Keep sphere contacts the narrow phase admits just above the cell's highest corner."""
    for delta in (1.0e-5, 5.0e-5, 9.0e-5):
        with test.subTest(delta=delta):
            builder = newton.ModelBuilder()
            cfg = builder.ShapeConfig(margin=0.01, gap=0.01)
            heights = np.zeros((9, 9), dtype=np.float32)
            heights[0, 0] = 1.0
            terrain_pose = wp.transform((20.0, -40.0, 0.0), wp.quat_identity())
            builder.add_shape_heightfield(
                heightfield=newton.Heightfield(data=heights, nrow=9, ncol=9, hx=0.4, hy=0.4),
                xform=terrain_pose,
                cfg=cfg,
            )
            # Sphere bottom sits delta above the margin + gap search bound of the flat cells.
            body = builder.add_body(
                xform=terrain_pose * wp.transform((0.1, 0.1, 0.035 + 0.04 + delta), wp.quat_identity())
            )
            builder.add_shape_sphere(body, radius=0.035, cfg=cfg)
            model = builder.finalize(device=device)
            _, a = _collide(model, legacy=True)
            _, b = _collide(model)
            test.assertGreater(len(a[0]), 0)
            _assert_geometry_equal(test, a, b)


def test_large_vertical_plane(test, device):
    """Retain plane candidates whose cached unit AABB does not bound the surface."""
    builder = newton.ModelBuilder()
    terrain = newton.Heightfield(data=np.zeros((5, 5), dtype=np.float32), nrow=5, ncol=5, hx=0.4, hy=0.4)
    builder.add_shape_heightfield(heightfield=terrain)
    plane_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_shape_plane(
        body=plane_body,
        xform=wp.transform((0.0, 0.0, 2.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2)),
        width=10.0,
        length=10.0,
    )
    model = builder.finalize(device=device)
    before, a = _collide(model, legacy=True)
    test.assertGreater(int(before.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
    test.assertGreater(len(a[0]), 0)
    after, b = _collide(model)
    test.assertEqual(
        int(before.narrow_phase.triangle_pairs_count.numpy()[0]),
        int(after.narrow_phase.triangle_pairs_count.numpy()[0]),
    )
    _assert_geometry_equal(test, a, b)


def test_current_height_updates(test, device):
    """Read current elevations instead of retaining a cached empty-cell decision."""
    model = _model(device=device, z=0.2)
    before, _ = _collide(model)
    test.assertEqual(int(before.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
    values = model.heightfield_elevations.numpy()
    values.reshape(9, 9)[2:7, 2:7] = 0.2
    model.heightfield_elevations.assign(values)
    after, geometry = _collide(model)
    test.assertGreater(int(after.narrow_phase.triangle_pairs_count.numpy()[0]), 0)
    test.assertGreater(len(geometry[0]), 0)


def test_replay_height_withdrawal_and_regrowth(test, device):
    """Refresh live heights in one pipeline through captured withdrawal and regrowth."""
    model = _model(device=device, z=0.2)
    pipeline = newton.CollisionPipeline(model, reduce_contacts=False, rigid_contact_max=2048, max_triangle_pairs=4096)
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)
    graph = None
    if model.device.is_cuda:
        with wp.ScopedCapture(device=model.device) as capture:
            pipeline.collide(state, contacts)
        graph = capture.graph
    initial = model.heightfield_elevations.numpy().copy()
    for elevated in (True, False, True, False):
        values = initial.copy()
        if elevated:
            values.reshape(9, 9)[2:7, 2:7] = 0.2
        model.heightfield_elevations.assign(values)
        if graph is None:
            pipeline.collide(state, contacts)
        else:
            wp.capture_launch(graph)
        triangle_count = int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0])
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertLessEqual(triangle_count, pipeline.narrow_phase.max_triangle_pairs)
        test.assertLessEqual(contact_count, contacts.rigid_contact_max)
        test.assertEqual(triangle_count > 0, elevated)
        test.assertEqual(contact_count > 0, elevated)


def test_negative_heights_ties_nonfinite_and_current_corners(test, device):
    """Retain boundary ties, deep prisms and unknown corners at negative heights."""
    lower = wp.array([-1.999, -1.998, -10.0, 0.5, 0.5, 0.5], dtype=float, device=device)
    values = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.65],
            [0.0, np.nan, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    heights = wp.array(values.ravel(), dtype=float, device=device)
    result = wp.empty(6, dtype=int, device=device)
    wp.launch(_check_cells, 6, inputs=[lower, heights, result], device=device)
    np.testing.assert_array_equal(result.numpy(), [0, 1, 0, 0, 0, 1])
    values[5, 3] = 0.7
    heights.assign(values.ravel())
    wp.launch(_check_cells, 6, inputs=[lower, heights, result], device=device)
    test.assertEqual(int(result.numpy()[5]), 0)


class TestHeightfieldCellReject(unittest.TestCase):
    pass


devices = get_test_devices()
for _test in (
    test_remove_separated_triangle_work,
    test_upstream_default_call_contract,
    test_near_below_transformed_and_scaled,
    test_primitive_contact_geometry,
    test_speculative_search_gap,
    test_sphere_contacts_at_gap_boundary,
    test_large_vertical_plane,
    test_current_height_updates,
    test_replay_height_withdrawal_and_regrowth,
    test_negative_heights_ties_nonfinite_and_current_corners,
):
    add_function_test(TestHeightfieldCellReject, _test.__name__, _test, devices=devices)


if __name__ == "__main__":
    unittest.main()
