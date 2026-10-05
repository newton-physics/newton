# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regress positive convex gaps smaller than the distance convergence tolerance."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.geometry.simplex_solver import create_solve_closest_distance
from newton._src.geometry.support_function import GenericShapeData, SupportMapDataProvider, support_map
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _query_box_gap(gap: float, direction: float, output: wp.array[float]):
    """Query two aligned boxes with a known signed surface gap."""
    a = GenericShapeData()
    a.shape_type = int(GeoType.BOX)
    a.scale = wp.vec3(0.01, 0.02, 0.03)
    b = a
    separated, pa, pb, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a,
        b,
        wp.quat_identity(),
        wp.vec3(direction * (0.02 + gap), 0.0, 0.0),
        0.0,
        SupportMapDataProvider(),
    )
    output[0] = float(separated)
    output[1] = distance
    for axis in range(3):
        output[2 + axis] = normal[axis]
        output[5 + axis] = pb[axis] - pa[axis]


class TestGJKNearContact(unittest.TestCase):
    """Preserve geometric separation independently of convergence tolerance."""

    def test_early_exit_preserves_separated_sphere_witnesses(self):
        """Return surface witnesses even when tiny distant spheres converge immediately."""
        for device in wp.get_devices():
            for cutoff in (0.0, 1.0):
                with self.subTest(device=str(device), cutoff=cutoff):
                    output = wp.zeros(11, dtype=float, device=device)
                    wp.launch(
                        _query_distant_spheres_with_cutoff, dim=1, inputs=[cutoff], outputs=[output], device=device
                    )
                    actual = output.numpy()
                    self.assertEqual(actual[0], 1.0)
                    np.testing.assert_allclose(actual[1:4], [0.001, 0.0, 0.0], atol=8e-6, rtol=0.0)
                    np.testing.assert_allclose(actual[4:7], [99.999, 0.0, 0.0], atol=8e-6, rtol=0.0)
                    np.testing.assert_allclose(actual[7:10], [1.0, 0.0, 0.0], atol=1e-6)
                    self.assertAlmostEqual(float(actual[10]), 99.998, delta=8e-6)
                    self.assertAlmostEqual(
                        float(actual[10]), float(np.linalg.norm(actual[4:7] - actual[1:4])), delta=8e-6
                    )

    def test_positive_sub_tolerance_gap(self):
        """Return actual positive gaps and oriented unit normals below 100 micrometers."""
        for device in wp.get_devices():
            for gap in (1e-3, 1.1e-4, 9.5e-5, 5e-5, 1e-5, 1e-6, 1e-7):
                for direction in (-1.0, 1.0):
                    with self.subTest(device=str(device), gap=gap, direction=direction):
                        out = wp.zeros(8, dtype=float, device=device)
                        wp.launch(_query_box_gap, dim=1, inputs=[gap, direction], outputs=[out], device=device)
                        actual = out.numpy()
                        self.assertEqual(actual[0], 1.0)
                        self.assertAlmostEqual(float(actual[1]), gap, delta=3e-9)
                        np.testing.assert_allclose(actual[2:5], [direction, 0.0, 0.0], atol=1e-6)
                        np.testing.assert_allclose(actual[5:8], [direction * gap, 0.0, 0.0], atol=3e-9)

    def test_true_overlap(self):
        """Keep the distance query's overlap classification for penetrating boxes."""
        for device in wp.get_devices():
            for gap in (-1e-3, -1e-5, 0.0):
                with self.subTest(device=str(device), gap=gap):
                    out = wp.zeros(8, dtype=float, device=device)
                    wp.launch(_query_box_gap, dim=1, inputs=[gap, 1.0], outputs=[out], device=device)
                    actual = out.numpy()
                    self.assertEqual(actual[0], 0.0)
                    self.assertEqual(actual[1], 0.0)

    def test_rotated_box_above_large_face(self):
        """Refine sub-tolerance gaps without cancellation along a large face."""
        for device in wp.get_devices():
            output = wp.zeros((1000, 6), dtype=float, device=device)
            wp.launch(_query_rotated_box, dim=1000, outputs=[output], device=device)
            actual = output.numpy()
            np.testing.assert_array_equal(actual[:, 0], 1.0)
            np.testing.assert_allclose(actual[:, 1], actual[:, 2], atol=2e-7, rtol=0.0)
            np.testing.assert_allclose(np.linalg.norm(actual[:, 3:6], axis=1), 1.0, atol=1e-6)
            np.testing.assert_allclose(actual[:900, 3:6], np.tile([0.0, 0.0, 1.0], (900, 1)), atol=1e-5)
            # The 100-micrometer boundary can round outside the near-contact
            # branch and retains the pre-existing witness-difference normal.
            np.testing.assert_allclose(actual[900:, 3:6], np.tile([0.0, 0.0, 1.0], (100, 1)), atol=2e-3)

    def test_millimeter_gap_converges_relative_to_distance(self):
        """Converge millimeter gaps to the relative tolerance, not an absolute 0.1 mm."""
        radius, half = 0.01, np.array([0.02, 0.015, 0.01])
        rng = np.random.default_rng(3)
        directions = rng.normal(size=(500, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        centers, expected = [], []
        for direction, gap in zip(directions, 10.0 ** rng.uniform(-4.0, -2.5, 500), strict=True):
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                offset = direction * mid - np.clip(direction * mid, -half, half)
                lo, hi = (mid, hi) if np.linalg.norm(offset) < radius + gap else (lo, mid)
            offset = direction * hi - np.clip(direction * hi, -half, half)
            centers.append(direction * hi)
            expected.append([np.linalg.norm(offset) - radius, *(offset / np.linalg.norm(offset))])
        expected = np.array(expected)
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                output = wp.zeros((500, 4), dtype=float, device=device)
                wp.launch(
                    _query_sphere_near_box,
                    dim=500,
                    inputs=[wp.array(np.array(centers), dtype=wp.vec3, device=device)],
                    outputs=[output],
                    device=device,
                )
                actual = output.numpy()
                np.testing.assert_allclose(actual[:, 0], expected[:, 0], atol=2e-6, rtol=0.0)
                cosine = np.clip(np.sum(actual[:, 1:] * expected[:, 1:], axis=1), -1.0, 1.0)
                self.assertLess(np.degrees(np.arccos(cosine)).max(), 1.0)

    def test_distant_small_shapes_return_witnesses_on_both_shapes(self):
        """Populate the simplex before accepting convergence, so far-apart witnesses lie on the shapes."""
        radius, center_b = 0.001, np.array([100.0, 0.0, 0.0])
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                output = wp.zeros(7, dtype=float, device=device)
                wp.launch(
                    _query_distant_spheres,
                    dim=1,
                    inputs=[radius, wp.vec3(*center_b)],
                    outputs=[output],
                    device=device,
                )
                actual = output.numpy()
                self.assertAlmostEqual(float(actual[0]), 100.0 - 2.0 * radius, delta=2e-5)
                self.assertAlmostEqual(float(np.linalg.norm(actual[1:4])), radius, delta=1e-5)
                self.assertAlmostEqual(float(np.linalg.norm(actual[4:7] - center_b)), radius, delta=1e-5)


@wp.kernel
def _query_distant_spheres(radius: float, center_b: wp.vec3, output: wp.array[float]):
    """Query two small spheres far enough apart that the center offset alone satisfies the relative gap."""
    a = GenericShapeData()
    a.shape_type = int(GeoType.SPHERE)
    a.scale = wp.vec3(radius, 0.0, 0.0)
    b = GenericShapeData()
    b.shape_type = int(GeoType.SPHERE)
    b.scale = wp.vec3(radius, 0.0, 0.0)
    _separated, point_a, point_b, _normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a, b, wp.quat_identity(), center_b, 0.0, SupportMapDataProvider()
    )
    output[0] = distance
    for axis in range(3):
        output[1 + axis] = point_a[axis]
        output[4 + axis] = point_b[axis]


@wp.kernel
def _query_distant_spheres_with_cutoff(cutoff: float, output: wp.array[float]):
    """Query tiny spheres separated by much more than their radius."""
    a = GenericShapeData()
    a.shape_type = int(GeoType.SPHERE)
    a.scale = wp.vec3(0.001, 0.0, 0.0)
    separated, pa, pb, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a, a, wp.quat_identity(), wp.vec3(100.0, 0.0, 0.0), 0.0, SupportMapDataProvider(), max_dist=cutoff
    )
    output[0] = float(separated)
    for axis in range(3):
        output[1 + axis] = pa[axis]
        output[4 + axis] = pb[axis]
        output[7 + axis] = normal[axis]
    output[10] = distance


@wp.kernel
def _query_rotated_box(output: wp.array2d[float]):
    """Place a small rotated box at a known gap above a much larger box."""
    i = wp.tid()
    a = GenericShapeData()
    a.shape_type = int(GeoType.BOX)
    a.scale = wp.vec3(0.8, 0.6, 0.02)
    b = GenericShapeData()
    b.shape_type = int(GeoType.BOX)
    b.scale = wp.vec3(0.03, 0.04, 0.07)
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.7, 0.2)), float(i % 100) * 0.03)
    height = (
        wp.abs(wp.quat_rotate(rotation, wp.vec3(0.03, 0.0, 0.0))[2])
        + wp.abs(wp.quat_rotate(rotation, wp.vec3(0.0, 0.04, 0.0))[2])
        + wp.abs(wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 0.07))[2])
    )
    gap = float(i // 100 + 1) * 1e-5
    separated, _, _, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a, b, rotation, wp.vec3(0.17, 0.13, 0.02 + height + gap), 0.0, SupportMapDataProvider()
    )
    output[i, 0] = float(separated)
    output[i, 1] = distance
    output[i, 2] = gap
    for axis in range(3):
        output[i, 3 + axis] = normal[axis]


@wp.kernel
def _query_sphere_near_box(center: wp.array[wp.vec3], output: wp.array2d[float]):
    """Query a sphere whose closest box feature is a face, edge, or corner."""
    i = wp.tid()
    a = GenericShapeData()
    a.shape_type = int(GeoType.BOX)
    a.scale = wp.vec3(0.02, 0.015, 0.01)
    b = GenericShapeData()
    b.shape_type = int(GeoType.SPHERE)
    b.scale = wp.vec3(0.01, 0.0, 0.0)
    _separated, _point_a, _point_b, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        a, b, wp.quat_identity(), center[i], 0.0, SupportMapDataProvider()
    )
    output[i, 0] = distance
    for axis in range(3):
        output[i, 1 + axis] = normal[axis]


@wp.func
def _write_query(
    output: wp.array2d[float],
    row: int,
    separated: bool,
    point_a: wp.vec3,
    point_b: wp.vec3,
    normal: wp.vec3,
    distance: float,
):
    output[row, 0] = float(separated)
    output[row, 1] = distance
    for axis in range(3):
        output[row, 2 + axis] = point_a[axis]
        output[row, 5 + axis] = point_b[axis]
        output[row, 8 + axis] = normal[axis]


@wp.kernel(module="unique")
def _query_positional_arguments(cutoff: float, output: wp.array2d[float]):
    """Bind the trailing optional arguments positionally and by keyword."""
    a = GenericShapeData()
    a.shape_type = int(GeoType.BOX)
    a.scale = wp.vec3(0.1, 0.2, 0.3)
    b = GenericShapeData()
    b.shape_type = int(GeoType.BOX)
    b.scale = wp.vec3(0.3, 0.1, 0.2)
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, 0.5)), 0.7)
    position = wp.vec3(0.9, 0.4, 0.2)
    provider = SupportMapDataProvider()
    gjk = wp.static(create_solve_closest_distance(support_map).core)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotation, position, 0.0, provider)
    _write_query(output, 0, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotation, position, 0.0, provider, 30, 1e-4)
    _write_query(output, 1, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotation, position, 0.0, provider, 1, 1e-4)
    _write_query(output, 2, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(
        a, b, rotation, position, 0.0, provider, MAX_ITER=1, COLLIDE_EPSILON=1e-4
    )
    _write_query(output, 3, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotation, position, 0.0, provider, 30, 1e-4, cutoff)
    _write_query(output, 4, separated, point_a, point_b, normal, distance)
    separated, point_a, point_b, normal, distance = gjk(a, b, rotation, position, 0.0, provider, max_dist=cutoff)
    _write_query(output, 5, separated, point_a, point_b, normal, distance)


@wp.kernel(module="unique")
def _query_pairs_at_cutoffs(
    shape_types: wp.array[int],
    scales: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    positions: wp.array[wp.vec3],
    cutoffs: wp.array2d[float],
    exact: wp.array2d[float],
    cut: wp.array3d[float],
):
    """Query each pair without the cutoff and with each of its cutoffs."""
    i, j = wp.tid()
    a = GenericShapeData()
    a.shape_type = shape_types[2 * i]
    a.scale = scales[2 * i]
    b = GenericShapeData()
    b.shape_type = shape_types[2 * i + 1]
    b.scale = scales[2 * i + 1]
    gjk = wp.static(create_solve_closest_distance(support_map).core)
    separated, point_a, point_b, normal, distance = gjk(
        a, b, rotations[i], positions[i], 0.0, SupportMapDataProvider(), max_dist=cutoffs[i, j]
    )
    cut[i, j, 0] = float(separated)
    cut[i, j, 1] = distance
    for axis in range(3):
        cut[i, j, 2 + axis] = point_a[axis]
        cut[i, j, 5 + axis] = point_b[axis]
        cut[i, j, 8 + axis] = normal[axis]
    if j == 0:
        separated, point_a, point_b, normal, distance = gjk(
            a, b, rotations[i], positions[i], 0.0, SupportMapDataProvider()
        )
        _write_query(exact, i, separated, point_a, point_b, normal, distance)


def test_positional_iteration_arguments_keep_their_meaning(test, device):
    """Bind positional MAX_ITER and COLLIDE_EPSILON as before the cutoff argument was added."""
    output = wp.zeros((6, 11), dtype=float, device=device)
    wp.launch(_query_positional_arguments, dim=1, inputs=[0.05], outputs=[output], device=device)
    default, positional, one_iter, one_iter_keyword, positional_cutoff, keyword_cutoff = output.numpy()
    np.testing.assert_array_equal(positional, default)
    np.testing.assert_array_equal(one_iter, one_iter_keyword)
    np.testing.assert_array_equal(positional_cutoff, keyword_cutoff)
    # One iteration leaves an unrefined, larger distance; the default converges.
    test.assertGreater(float(one_iter[1]), float(default[1]) + 1e-3)


# Pairs whose float32 exact distance sits just below a valid support-plane bound
# (shape A type and scale, shape B type and scale, rotation xyzw, position).
_BOUNDARY_PAIRS = (
    (
        GeoType.CAPSULE,
        (0.1682592475890331, 0.14003996945867525, 0.0),
        GeoType.CONE,
        (0.09754981982265762, 0.17272756345704407, 0.0),
        (-0.6322464146167568, -0.5302427686633309, -0.3201867819836226, 0.46538962400065786),
        (-0.2018848055252058, -0.265893102759429, 0.3215784286871637),
    ),
    (
        GeoType.CONE,
        (0.19294660619128537, 0.038428061765413746, 0.0),
        GeoType.CONE,
        (0.17865204331701115, 0.06043683914515466, 0.0),
        (0.4539830414887907, -0.049389047964049845, -0.06681424272939178, 0.8871279371941172),
        (-0.041619640305926596, -0.3983849064680972, -0.16667230457387375),
    ),
    (
        GeoType.CYLINDER,
        (0.16696513803893834, 0.13955253763697376, 0.0),
        GeoType.CYLINDER,
        (0.19116954047115134, 0.03298599609398348, 0.0),
        (-0.06464846032504241, 0.1064601405364523, 0.2697334631566794, 0.9548458901351906),
        (-0.034743432821168777, 0.467328715822675, -0.09497783884786012),
    ),
    (
        GeoType.CONE,
        (0.03929883731734333, 0.07553593566293515, 0.0),
        GeoType.CAPSULE,
        (0.11197675341090395, 0.13864135544884926, 0.0),
        (0.7791788206026583, -0.25468472132787073, -0.4977472848369689, 0.2833084867839658),
        (-0.0467585747846761, 0.05024803033131552, -0.2665155265740738),
    ),
    (
        GeoType.CYLINDER,
        (0.15456012773094185, 0.13246612184744727, 0.0),
        GeoType.CONE,
        (0.08733364535303789, 0.07619071293952659, 0.0),
        (0.35775010464355106, 0.5250499803170853, -0.5933219459147481, 0.49427365830326553),
        (0.23058515800559867, 0.009948621105464411, -0.09438178473954709),
    ),
)


def _random_boundary_pairs(rng, count):
    """Return random convex pairs with sizes from 3 mm to 3 m, placed near contact."""
    kinds = [GeoType.BOX, GeoType.CYLINDER, GeoType.CAPSULE, GeoType.CONE, GeoType.SPHERE, GeoType.ELLIPSOID]
    types = rng.choice([int(kind) for kind in kinds], 2 * count)
    size = np.exp(rng.uniform(np.log(0.003), np.log(3.0), 2 * count))
    scales = size[:, None] * rng.uniform(0.2, 1.0, size=(2 * count, 3))
    # Only boxes and ellipsoids use a third extent; a nonzero cylinder scale[2] selects a barrel profile.
    scales[(types != int(GeoType.BOX)) & (types != int(GeoType.ELLIPSOID)), 2] = 0.0
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = rng.uniform(0.0, np.pi, count)
    rotations = np.concatenate([axes * np.sin(0.5 * angles)[:, None], np.cos(0.5 * angles)[:, None]], axis=1)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    reach = 2.0 * (size[0::2] + size[1::2]) * rng.uniform(0.3, 1.2, count)
    return types, scales, rotations, directions * reach[:, None]


def test_separation_cutoff_matches_exact_query_at_the_boundary(test, device):
    """Match the exact query for cutoffs within a few float32 steps of each pair's exact distance."""
    types, scales, rotations, positions = _random_boundary_pairs(np.random.default_rng(29), 4096)
    fixed_types = np.array([[int(p[0]), int(p[2])] for p in _BOUNDARY_PAIRS]).reshape(-1)
    fixed_scales = np.array([[p[1], p[3]] for p in _BOUNDARY_PAIRS]).reshape(-1, 3)
    types = np.concatenate([fixed_types, types])
    scales = np.concatenate([fixed_scales, scales])
    rotations = np.concatenate([[p[4] for p in _BOUNDARY_PAIRS], rotations])
    positions = np.concatenate([[p[5] for p in _BOUNDARY_PAIRS], positions])
    count = len(rotations)
    arrays = [
        wp.array(types, dtype=int, device=device),
        wp.array(scales, dtype=wp.vec3, device=device),
        wp.array(rotations, dtype=wp.quat, device=device),
        wp.array(positions, dtype=wp.vec3, device=device),
    ]
    # Exact distances first (every cutoff zero), then cutoffs around them.
    exact = wp.zeros((count, 11), dtype=float, device=device)
    cut = wp.zeros((count, 1, 11), dtype=float, device=device)
    wp.launch(
        _query_pairs_at_cutoffs,
        dim=(count, 1),
        inputs=[*arrays, wp.zeros((count, 1), dtype=float, device=device)],
        outputs=[exact, cut],
        device=device,
    )
    distance = exact.numpy()[:, 1].astype(np.float32)
    steps = []
    for ulps in (-2, -1, 0, 1, 2):
        cutoff = distance.copy()
        for _ in range(abs(ulps)):
            cutoff = np.nextafter(cutoff, np.float32(np.sign(ulps) * np.inf))
        steps.append(cutoff)
    for relative in (-1e-2, -1e-4, -1e-6, 1e-6, 1e-4):
        steps.append(distance * np.float32(1.0 + relative))
    cutoffs = np.stack(steps, axis=1)
    cutoffs[distance <= 0.0] = 0.0
    width = cutoffs.shape[1]
    cut = wp.zeros((count, width, 11), dtype=float, device=device)
    wp.launch(
        _query_pairs_at_cutoffs,
        dim=(count, width),
        inputs=[*arrays, wp.array(cutoffs, dtype=float, device=device)],
        outputs=[exact, cut],
        device=device,
    )
    exact, cut = exact.numpy(), cut.numpy()
    separated = exact[:, 0] == 1.0
    test.assertGreater(int(np.count_nonzero(separated)), 3000)
    within = separated[:, None] & (exact[:, 1:2] <= cutoffs)
    beyond = separated[:, None] & ~within
    exact_rows = np.broadcast_to(exact[:, None, :], cut.shape)
    # Results the exact query keeps are reproduced bit for bit, including the pinned pairs.
    np.testing.assert_array_equal(cut[within], exact_rows[within])
    # Results it rejects stay rejected: separated and farther than the cutoff.
    np.testing.assert_array_equal(cut[beyond][:, 0], 1.0)
    test.assertTrue(np.all(cut[beyond][:, 1] > cutoffs[beyond]))
    # The cutoff still stops pairs that are 1% beyond it before convergence.
    stopped = np.any(cut[:, 5, :] != exact, axis=1)[separated]
    test.assertGreater(float(np.mean(stopped)), 0.5)


def test_separation_cutoff_keeps_contact_at_the_exact_boundary(test, device):
    """Keep the contact of a cylinder pair whose total gap equals its exact float32 distance."""
    type_a, scale_a, type_b, scale_b, rotation, position = _BOUNDARY_PAIRS[2]
    exact = wp.zeros((1, 11), dtype=float, device=device)
    wp.launch(
        _query_pairs_at_cutoffs,
        dim=(1, 1),
        inputs=[
            wp.array([int(type_a), int(type_b)], dtype=int, device=device),
            wp.array([scale_a, scale_b], dtype=wp.vec3, device=device),
            wp.array([rotation], dtype=wp.quat, device=device),
            wp.array([position], dtype=wp.vec3, device=device),
            wp.zeros((1, 1), dtype=float, device=device),
        ],
        outputs=[exact, wp.zeros((1, 1, 11), dtype=float, device=device)],
        device=device,
    )
    # The exact query's distance differs by device rounding; on CPU it is 0.1076592430472374.
    gap = float(exact.numpy()[0, 1])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = builder.ShapeConfig(gap=0.5 * gap, margin=0.0)
    body_a = builder.add_body(xform=wp.transform_identity())
    builder.add_shape_cylinder(body_a, radius=scale_a[0], half_height=scale_a[1], cfg=cfg)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(*position), wp.quat(*rotation)))
    builder.add_shape_cylinder(body_b, radius=scale_b[0], half_height=scale_b[1], cfg=cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)


devices = get_test_devices()
for _test in (
    test_positional_iteration_arguments_keep_their_meaning,
    test_separation_cutoff_matches_exact_query_at_the_boundary,
    test_separation_cutoff_keeps_contact_at_the_exact_boundary,
):
    add_function_test(TestGJKNearContact, _test.__name__, _test, devices=devices)


if __name__ == "__main__":
    unittest.main()
